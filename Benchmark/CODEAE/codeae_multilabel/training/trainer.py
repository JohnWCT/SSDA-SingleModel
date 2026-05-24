"""Fold-level fine-tuning trainer (CODE-AE supervised progressive unfreeze)."""

from __future__ import annotations

from itertools import cycle
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW

from codeae_multilabel.contracts import (
    CodeAEMultilabelConfig,
    PreparedFineTuneData,
    PredictionBundle,
    SourceFold,
    TrainingResult,
)
from codeae_multilabel.data.dataloader import (
    build_source_eval_loader,
    build_source_train_loader,
    build_target_eval_loader,
    build_target_unlabeled_loader,
)
from codeae_multilabel.evaluation.metrics import compute_metrics_from_predictions
from codeae_multilabel.evaluation.prediction import build_prediction_long_table
from codeae_multilabel.model.checkpoint import (
    load_finetune_checkpoint,
    save_finetune_checkpoint,
)
from codeae_multilabel.model.wrapper import MultiLabelCodeAEModel
from codeae_multilabel.training.codeae_loss_adapter import CodeAELossAdapter
from codeae_multilabel.training.losses import masked_bce_with_logits, masked_mae
from codeae_multilabel.training.selection import MetricSelector
from codeae_multilabel.training.train_state import EpochLog, epoch_logs_to_dataframe


def _set_requires_grad(module: nn.Module, requires_grad: bool) -> None:
    for param in module.parameters():
        param.requires_grad = requires_grad


def _collect_encoder_linear_layers(module: nn.Module) -> list[nn.Linear]:
    layers = [m for m in module.modules() if isinstance(m, nn.Linear)]
    return list(reversed(layers))


def _trainable_parameters(model: nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


def setup_finetune_parameter_freeze(
    model: MultiLabelCodeAEModel, *, freeze_encoder: bool
) -> None:
    _set_requires_grad(model.codeae_core, not freeze_encoder)
    _set_requires_grad(model.prediction_head, True)


class CodeAEMultilabelTrainer:
    def __init__(
        self,
        model: MultiLabelCodeAEModel,
        optimizer: torch.optim.Optimizer,
        config: CodeAEMultilabelConfig,
        codeae_loss_adapter: Optional[CodeAELossAdapter] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.loss_adapter = codeae_loss_adapter or CodeAELossAdapter()
        self.selector = MetricSelector(config.task_type, config.metric)
        self.device = config.device
        self.current_lr = config.lr
        if config.progressive_unfreeze:
            self.encoder_unfreeze_layers = _collect_encoder_linear_layers(model.codeae_core)
        else:
            self.encoder_unfreeze_layers = []

    def _supervised_loss(self, pred: torch.Tensor, batch: dict[str, Any]) -> torch.Tensor:
        y = batch["y"].to(self.device)
        mask = batch["mask"].to(self.device)
        if self.config.task_type == "classification":
            return masked_bce_with_logits(pred, y, mask)
        return masked_mae(pred, y, mask)

    def _rebuild_optimizer(self) -> None:
        self.optimizer = AdamW(_trainable_parameters(self.model), lr=self.current_lr)

    def _progressive_unfreeze_next_layer(self, best_path: str) -> bool:
        if not self.encoder_unfreeze_layers:
            return False
        load_finetune_checkpoint(self.model, best_path, map_location=self.device)
        layer = self.encoder_unfreeze_layers.pop(0)
        _set_requires_grad(layer, True)
        self.current_lr *= self.config.decay_coefficient
        self._rebuild_optimizer()
        return True

    def _validate(self, loader: Any, split_name: str, fold_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        self.model.eval()
        all_scores: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        all_mask: list[np.ndarray] = []
        all_sids: list[str] = []
        for batch in loader:
            x = batch["x"].to(self.device)
            with torch.no_grad():
                pred = self.model(x)
            all_scores.append(pred.detach().cpu().numpy())
            all_y.append(batch["y"].numpy())
            all_mask.append(batch["mask"].numpy())
            all_sids.extend(batch["sample_id"])
        scores = np.vstack(all_scores)
        y = np.vstack(all_y)
        mask = np.vstack(all_mask)
        pred_df = build_prediction_long_table(
            scores,
            y,
            mask,
            all_sids,
            self._drug_index,
            "source",
            split_name,
            self.config.task_type,
            self.config.prediction_threshold,
            self.config.regression_binary_threshold,
            fold_id,
            self.config.seed,
            self._cancer_type_table,
        )
        per, summ = compute_metrics_from_predictions(pred_df, self.config.task_type, "source")
        return per, summ

    def _use_finetune_aux(self, epoch: int) -> bool:
        if not self.config.uses_finetune_unlabeled_loss:
            return False
        return epoch >= self.config.finetune_domain_warmup_epochs

    def train_fold(
        self,
        fold: SourceFold,
        prepared: PreparedFineTuneData,
        fold_dir: str,
    ) -> tuple[TrainingResult, PredictionBundle]:
        self._drug_index = prepared.drug_index
        self._cancer_type_table = prepared.cancer_type_table
        train_loader = build_source_train_loader(
            prepared.source_omics,
            prepared.source_response,
            list(fold.train_sample_ids),
            self.config.batch_size,
            self.config.seed,
        )
        val_loader = build_source_eval_loader(
            prepared.source_omics,
            prepared.source_response,
            list(fold.val_sample_ids),
            self.config.batch_size,
        )
        test_loader = build_source_eval_loader(
            prepared.source_omics,
            prepared.source_response,
            list(fold.test_sample_ids),
            self.config.batch_size,
        )
        target_eval_loader = build_target_eval_loader(
            prepared.target_omics, prepared.target_response, self.config.batch_size
        )
        target_cycle: Iterator[dict[str, Any]] | None = None
        if self.config.uses_finetune_unlabeled_loss:
            target_loader = build_target_unlabeled_loader(
                prepared.target_omics, self.config.batch_size, self.config.seed
            )
            target_cycle = cycle(target_loader)
            self.loss_adapter.attach_shared_encoder(self.model)

        best_path = str(Path(fold_dir) / "best_model.pt")
        best_epoch = -1
        best_metric_name = ""
        best_metric_value: float | None = None
        logs: list[EpochLog] = []
        n_epochs = self.config.train_num_epochs

        patience_count = 0
        reset_count = 1
        stop_training = False

        if self.config.progressive_unfreeze:
            self.encoder_unfreeze_layers = _collect_encoder_linear_layers(self.model.codeae_core)
        else:
            self.encoder_unfreeze_layers = []

        for epoch in range(n_epochs):
            if epoch % 50 == 0:
                print(f"Fine tuning epoch {epoch}")
            self.model.train()
            epoch_sup_losses: list[float] = []
            epoch_aux_losses: list[float] = []
            use_aux = self._use_finetune_aux(epoch)

            for source_batch in train_loader:
                target_batch = next(target_cycle) if use_aux and target_cycle is not None else None
                self.optimizer.zero_grad()
                x = source_batch["x"].to(self.device)
                pred = self.model(x)
                sup_loss = self._supervised_loss(pred, source_batch)
                aux_loss = torch.tensor(0.0, device=self.device)
                if use_aux and target_batch is not None:
                    aux = self.loss_adapter.compute_finetune_losses(
                        source_batch, target_batch, self.model
                    )
                    if aux:
                        aux_loss = sum(aux.values()) * self.config.finetune_domain_lambda
                        epoch_aux_losses.append(float(aux_loss.detach().cpu().item()))
                total = sup_loss + aux_loss
                total.backward()
                self.optimizer.step()
                epoch_sup_losses.append(float(sup_loss.detach().cpu().item()))

            _, val_summ = self._validate(val_loader, "source_val", fold.fold_id)
            metric_name, metric_value, _direction = self.selector.select_metric(val_summ)
            mean_sup = float(np.mean(epoch_sup_losses)) if epoch_sup_losses else 0.0
            mean_aux = float(np.mean(epoch_aux_losses)) if epoch_aux_losses else 0.0
            logs.append(
                EpochLog(
                    fold_id=fold.fold_id,
                    epoch=epoch,
                    source_prediction_loss=mean_sup,
                    codeae_loss_total=mean_aux,
                    total_loss=mean_sup + mean_aux,
                    val_metric_name=metric_name,
                    val_metric_value=metric_value,
                    selected_metric_name=metric_name,
                    selected_metric_value=metric_value,
                )
            )

            if self.selector.is_better(metric_value, best_metric_value, metric_name):
                best_metric_value = metric_value
                best_metric_name = metric_name
                best_epoch = epoch
                patience_count = 0
                save_finetune_checkpoint(
                    self.model,
                    self.optimizer,
                    {
                        "epoch": epoch,
                        "metric_name": metric_name,
                        "metric_value": metric_value,
                        "current_lr": self.current_lr,
                        "progressive_unfreeze": self.config.progressive_unfreeze,
                    },
                    best_path,
                )
            else:
                patience_count += 1
                tolerance = self.config.early_stopping_tolerance * reset_count
                if patience_count > tolerance and best_epoch > 0:
                    if self.config.progressive_unfreeze:
                        if self._progressive_unfreeze_next_layer(best_path):
                            reset_count += 1
                            patience_count = 0
                        else:
                            stop_training = True
                    else:
                        stop_training = True

            if stop_training:
                break

        if best_epoch < 0:
            best_epoch = n_epochs - 1
            best_metric_name = self.config.metric or "macro_auroc"
            best_metric_value = float("nan")
            save_finetune_checkpoint(self.model, self.optimizer, {}, best_path)
        elif Path(best_path).is_file():
            load_finetune_checkpoint(self.model, best_path, map_location=self.device)

        bundle = self._evaluate_all_splits(
            fold, prepared, val_loader, test_loader, target_eval_loader
        )
        result = TrainingResult(
            fold_id=fold.fold_id,
            best_model_path=best_path,
            best_epoch=best_epoch,
            best_metric_name=best_metric_name,
            best_metric_value=float(best_metric_value or float("nan")),
            train_log=epoch_logs_to_dataframe(logs),
        )
        return result, bundle

    def _evaluate_all_splits(
        self,
        fold: SourceFold,
        prepared: PreparedFineTuneData,
        val_loader: Any,
        test_loader: Any,
        target_eval_loader: Any,
    ) -> PredictionBundle:
        self.model.eval()
        splits = [
            (val_loader, "source_val", "source"),
            (test_loader, "source_test", "source"),
            (target_eval_loader, "target_eval", "target"),
        ]
        source_parts: list[pd.DataFrame] = []
        target_parts: list[pd.DataFrame] = []
        for loader, split_name, domain in splits:
            scores_list, y_list, mask_list, sids = [], [], [], []
            for batch in loader:
                x = batch["x"].to(self.device)
                with torch.no_grad():
                    pred = self.model(x)
                scores_list.append(pred.detach().cpu().numpy())
                if "y" in batch:
                    y_list.append(batch["y"].numpy())
                    mask_list.append(batch["mask"].numpy())
                sids.extend(batch["sample_id"])
            if not scores_list:
                continue
            scores = np.vstack(scores_list)
            y = np.vstack(y_list)
            mask = np.vstack(mask_list)
            df = build_prediction_long_table(
                scores,
                y,
                mask,
                sids,
                prepared.drug_index,
                domain,  # type: ignore[arg-type]
                split_name,
                self.config.task_type,
                self.config.prediction_threshold,
                self.config.regression_binary_threshold,
                fold.fold_id,
                self.config.seed,
                prepared.cancer_type_table,
            )
            if domain == "source":
                source_parts.append(df)
            else:
                target_parts.append(df)
        source_pred = pd.concat(source_parts, ignore_index=True) if source_parts else pd.DataFrame()
        target_pred = pd.concat(target_parts, ignore_index=True) if target_parts else pd.DataFrame()
        src_per, src_summ = compute_metrics_from_predictions(
            source_pred, self.config.task_type, "source"
        )
        tgt_per, tgt_summ = compute_metrics_from_predictions(
            target_pred, self.config.task_type, "target"
        )
        return PredictionBundle(
            source_predictions=source_pred,
            target_predictions=target_pred,
            source_metrics_per_drug=src_per,
            target_metrics_per_drug=tgt_per,
            source_metrics_summary=src_summ,
            target_metrics_summary=tgt_summ,
        )


def build_trainer(
    model: MultiLabelCodeAEModel,
    config: CodeAEMultilabelConfig,
    n_features: int = 0,
    codeae_loss_adapter: Optional[CodeAELossAdapter] = None,
) -> CodeAEMultilabelTrainer:
    if config.freeze_encoder_initially or config.progressive_unfreeze:
        setup_finetune_parameter_freeze(model, freeze_encoder=True)
        trainable = list(model.prediction_head.parameters())
    else:
        _set_requires_grad(model.codeae_core, True)
        _set_requires_grad(model.prediction_head, True)
        trainable = list(model.parameters())
    optimizer = AdamW(trainable, lr=config.lr)
    if codeae_loss_adapter is None and config.finetune_domain_loss == "adversarial":
        pretrain_dir = (
            Path(config.pretrain_checkpoint).parent if config.pretrain_checkpoint else None
        )
        codeae_loss_adapter = CodeAELossAdapter(config, n_features, pretrain_dir)
    elif codeae_loss_adapter is None:
        codeae_loss_adapter = CodeAELossAdapter()
    return CodeAEMultilabelTrainer(model, optimizer, config, codeae_loss_adapter)
