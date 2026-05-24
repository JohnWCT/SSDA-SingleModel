"""Adapter around legacy CODE-AE modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from codeae_multilabel.contracts import CodeAEMultilabelConfig, OmicsTable, PreparedPretrainData

_PKG_ROOT = Path(__file__).resolve().parents[1]
_CODEAE_ROOT = _PKG_ROOT.parent
if str(_CODEAE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODEAE_ROOT))


def build_legacy_codeae_components(
    config: CodeAEMultilabelConfig, n_features: int
) -> dict[str, Any]:
    from mlp import MLP  # noqa: WPS433

    encoder = MLP(
        input_dim=n_features,
        output_dim=config.latent_dim,
        hidden_dims=list(config.encoder_hidden_dims),
        dop=config.dop,
    )
    setattr(encoder, "norm_flag", config.norm_flag)
    return {"shared_encoder": encoder, "latent_dim": config.latent_dim}


def encode_with_codeae(model: nn.Module, x: Tensor, deterministic: bool = True) -> Tensor:
    if deterministic:
        model.eval()
    with torch.set_grad_enabled(not deterministic):
        z: Tensor = model(x)
        if getattr(model, "norm_flag", False):
            z = torch.nn.functional.normalize(z, p=2, dim=1)
    return z


def get_latent_dim(model: nn.Module) -> int:
    if hasattr(model, "output_dim"):
        return int(model.output_dim)
    for module in reversed(list(model.modules())):
        if isinstance(module, nn.Linear):
            return int(module.out_features)
    raise ValueError("cannot determine latent dimension from encoder")


def build_pretrain_dataloaders(
    prepared: PreparedPretrainData,
    batch_size: int,
    seed: int,
) -> tuple[
    tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]],
    tuple[DataLoader[tuple[Tensor, ...]], DataLoader[tuple[Tensor, ...]]],
]:
    from sklearn.model_selection import train_test_split

    src_x = prepared.source_omics.x.loc[list(prepared.source_omics.sample_ids)].values.astype("float32")
    tgt_x = prepared.target_omics.x.loc[list(prepared.target_omics.sample_ids)].values.astype("float32")
    src_tr, src_te = train_test_split(src_x, test_size=0.1, random_state=seed)
    tgt_tr, tgt_te = train_test_split(tgt_x, test_size=0.1, random_state=seed)
    gen = torch.Generator().manual_seed(seed)

    def _dl(arr: Any, shuffle: bool) -> DataLoader[tuple[Tensor, ...]]:
        ds = TensorDataset(torch.from_numpy(arr))
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=gen if shuffle else None,
            drop_last=shuffle,
        )

    s_train, s_test = _dl(src_tr, True), _dl(src_te, False)
    t_train, t_test = _dl(tgt_tr, True), _dl(tgt_te, False)
    return (s_train, s_test), (t_train, t_test)


def run_pretrain(
    config: CodeAEMultilabelConfig,
    prepared: PreparedPretrainData,
    model_save_folder: str,
) -> nn.Module:
    import train_code_adv  # noqa: WPS433

    n_features = len(prepared.source_omics.feature_names)
    s_dl, t_dl = build_pretrain_dataloaders(prepared, config.batch_size, config.seed)
    kwargs: dict[str, Any] = {
        "input_dim": n_features,
        "latent_dim": config.latent_dim,
        "encoder_hidden_dims": list(config.encoder_hidden_dims),
        "classifier_hidden_dims": list(config.classifier_hidden_dims),
        "lr": config.lr,
        "batch_size": config.batch_size,
        "pretrain_num_epochs": config.pretrain_num_epochs,
        "train_num_epochs": config.train_num_epochs,
        "norm_flag": config.norm_flag,
        "alpha": config.alpha,
        "dop": config.dop,
        "device": config.device,
        "model_save_folder": model_save_folder,
        "es_flag": config.es_flag,
        "retrain_flag": config.retrain_flag,
    }
    shared_encoder: nn.Module
    shared_encoder, _hist = train_code_adv.train_code_adv(
        s_dataloaders=s_dl, t_dataloaders=t_dl, **kwargs
    )
    return shared_encoder
