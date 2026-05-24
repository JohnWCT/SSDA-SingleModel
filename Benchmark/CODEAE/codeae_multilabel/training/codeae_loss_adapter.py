"""Bridge to legacy CODE-AE auxiliary losses (WGAN deconfounding during fine-tune)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

import torch
from torch import Tensor

from codeae_multilabel.contracts import CodeAEMultilabelConfig

logger = logging.getLogger(__name__)

_CODEAE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODEAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEAE_ROOT))


class CodeAELossAdapter:
    """CODE-AE WGAN confounder critic + generator loss on paired source/target omics."""

    def __init__(
        self,
        config: Optional[CodeAEMultilabelConfig] = None,
        n_features: int = 0,
        pretrain_dir: Optional[str | Path] = None,
    ) -> None:
        self.config = config
        self.enabled = False
        self._step = 0
        self.s_dsnae: Any = None
        self.t_dsnae: Any = None
        self.critic: Any = None
        self.critic_optimizer: Optional[torch.optim.Optimizer] = None
        self._warned_missing = False

        if config is None or config.finetune_domain_loss != "adversarial":
            return
        if pretrain_dir is None and config.pretrain_checkpoint:
            pretrain_dir = Path(config.pretrain_checkpoint).parent
        if pretrain_dir is None or n_features <= 0:
            return
        self._try_load_deconfound_modules(config, n_features, Path(pretrain_dir))

    def _try_load_deconfound_modules(
        self, config: CodeAEMultilabelConfig, n_features: int, pretrain_dir: Path
    ) -> None:
        s_path = pretrain_dir / "a_s_dsnae.pt"
        t_path = pretrain_dir / "a_t_dsnae.pt"
        critic_path = pretrain_dir / "confounding_classifier.pt"
        if not (s_path.is_file() and t_path.is_file() and critic_path.is_file()):
            if not self._warned_missing:
                logger.warning(
                    "CODE-AE deconfounding artifacts missing under %s; "
                    "finetune domain adaptation disabled (re-run pretrain).",
                    pretrain_dir,
                )
                self._warned_missing = True
            return

        from dsn_ae import DSNAE  # noqa: WPS433
        from mlp import MLP  # noqa: WPS433
        from train_code_adv import compute_gradient_penalty  # noqa: WPS433

        self._compute_gradient_penalty = compute_gradient_penalty

        shared_encoder = MLP(
            input_dim=n_features,
            output_dim=config.latent_dim,
            hidden_dims=list(config.encoder_hidden_dims),
            dop=config.dop,
        )
        shared_decoder = MLP(
            input_dim=2 * config.latent_dim,
            output_dim=n_features,
            hidden_dims=list(config.encoder_hidden_dims)[::-1],
            dop=config.dop,
        )
        dsnae_kwargs = {
            "input_dim": n_features,
            "latent_dim": config.latent_dim,
            "hidden_dims": list(config.encoder_hidden_dims),
            "dop": config.dop,
            "norm_flag": config.norm_flag,
            "alpha": config.alpha,
        }
        self.s_dsnae = DSNAE(
            shared_encoder=shared_encoder,
            decoder=shared_decoder,
            **dsnae_kwargs,
        )
        self.t_dsnae = DSNAE(
            shared_encoder=shared_encoder,
            decoder=shared_decoder,
            **dsnae_kwargs,
        )
        self.s_dsnae.load_state_dict(
            torch.load(s_path, map_location="cpu", weights_only=False), strict=False
        )
        self.t_dsnae.load_state_dict(
            torch.load(t_path, map_location="cpu", weights_only=False), strict=False
        )
        for module in (self.s_dsnae, self.t_dsnae):
            module.private_encoder.eval()
            for param in module.private_encoder.parameters():
                param.requires_grad = False
            module.decoder.eval()
            for param in module.decoder.parameters():
                param.requires_grad = False

        self.critic = MLP(
            input_dim=config.latent_dim * 2,
            output_dim=1,
            hidden_dims=list(config.classifier_hidden_dims),
            dop=config.dop,
        )
        self.critic.load_state_dict(
            torch.load(critic_path, map_location="cpu", weights_only=False)
        )
        self.critic.to(config.device)
        self.critic_optimizer = torch.optim.RMSprop(self.critic.parameters(), lr=config.lr)
        self.s_dsnae.to(config.device)
        self.t_dsnae.to(config.device)
        self.enabled = True

    def attach_shared_encoder(self, model: Any) -> None:
        if not self.enabled or self.s_dsnae is None or self.t_dsnae is None:
            return
        self.s_dsnae.shared_encoder = model.codeae_core
        self.t_dsnae.shared_encoder = model.codeae_core

    def compute_pretrain_losses(
        self, source_batch: dict[str, Any], target_batch: dict[str, Any], model: Any
    ) -> dict[str, Tensor]:
        del source_batch, target_batch, model
        return {}

    def compute_finetune_losses(
        self, source_batch: dict[str, Any], target_batch: dict[str, Any], model: Any
    ) -> dict[str, Tensor]:
        if not self.enabled or self.config is None:
            return {}
        assert self.critic is not None and self.critic_optimizer is not None
        assert self.s_dsnae is not None and self.t_dsnae is not None

        self.attach_shared_encoder(model)
        device = self.config.device
        s_x = source_batch["x"].to(device)
        t_x = target_batch["x"].to(device)
        batch_n = min(s_x.shape[0], t_x.shape[0])
        if batch_n == 0:
            return {}
        if s_x.shape[0] != batch_n:
            s_x = s_x[:batch_n]
        if t_x.shape[0] != batch_n:
            t_x = t_x[:batch_n]

        self.s_dsnae.eval()
        self.t_dsnae.eval()
        self.critic.train()

        with torch.no_grad():
            s_code = self.s_dsnae.encode(s_x)
            t_code = self.t_dsnae.encode(t_x)

        self.critic_optimizer.zero_grad()
        critic_loss = torch.mean(self.critic(t_code)) - torch.mean(self.critic(s_code))
        gp_weight = self.config.finetune_wgan_gp
        if gp_weight > 0:
            critic_loss = critic_loss + gp_weight * self._compute_gradient_penalty(
                self.critic,
                real_samples=s_code,
                fake_samples=t_code,
                device=device,
            )
        critic_loss.backward()
        self.critic_optimizer.step()

        losses: dict[str, Tensor] = {}
        self._step += 1
        if self._step % self.config.finetune_gen_every != 0:
            return losses

        self.critic.eval()
        t_code_gen = self.t_dsnae.encode(t_x)
        gen_loss = -torch.mean(self.critic(t_code_gen))
        losses["adv_gen"] = self.config.alpha * gen_loss
        return losses
