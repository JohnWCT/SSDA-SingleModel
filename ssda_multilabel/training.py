"""Multi-label SSDA trainer: validation never updates parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import cycle
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ssda_multilabel.adaptation import masked_adentropy_loss
from ssda_multilabel.config import MultiLabelConfig
from ssda_multilabel.dataset import MultiLabelSampleDataset
from ssda_multilabel.losses import masked_bce_with_logits, masked_regression_loss
from ssda_multilabel.model import MultiLabelSSDAModel
from ssda_multilabel.schemas import PreparedData, SourceFold


@dataclass
class EpochLog:
    epoch: int
    source_loss: float
    target_labeled_loss: float
    target_adapt_loss: float
    total_loss: float
    val_loss: float


@dataclass
class TrainResult:
    fold_id: int
    epoch_logs: list[EpochLog] = field(default_factory=list)


class MultiLabelSSDTrainer:
    def __init__(
        self,
        model: MultiLabelSSDAModel,
        config: MultiLabelConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
        self.device = torch.device(config.device)

    def _to_loaders(
        self,
        prepared: PreparedData,
        fold: SourceFold,
    ) -> tuple[
        DataLoader[dict[str, Any]],
        DataLoader[dict[str, Any]],
        DataLoader[dict[str, Any]],
        DataLoader[dict[str, Any]],
    ]:
        so, sr = prepared.source_omics, prepared.source_response
        to, tr = prepared.target_omics, prepared.target_response
        tm = prepared.target_masks

        def _t(arr: np.ndarray[Any, Any]) -> torch.Tensor:
            return torch.from_numpy(arr)

        src_x = _t(so.x)
        src_y = _t(sr.y)
        src_m = _t(sr.mask)

        tr_x = _t(to.x)
        tr_y = _t(tr.y)
        tr_obs = _t(tm.observed_mask)
        tr_lab = _t(tm.labeled_mask)
        tr_unl = _t(tm.unlabeled_mask)

        tr_idx = list(fold.train_indices)
        va_idx = list(fold.val_indices)
        te_idx = list(fold.test_indices)
        train_ds = MultiLabelSampleDataset(
            src_x[tr_idx],
            src_y[tr_idx],
            src_m[tr_idx],
            [so.sample_ids[i] for i in tr_idx],
        )
        val_ds = MultiLabelSampleDataset(
            src_x[va_idx],
            src_y[va_idx],
            src_m[va_idx],
            [so.sample_ids[i] for i in va_idx],
        )
        tgt_ds = MultiLabelSampleDataset(tr_x, tr_y, tr_obs, list(to.sample_ids), tr_lab, tr_unl)
        test_ds = MultiLabelSampleDataset(
            src_x[te_idx],
            src_y[te_idx],
            src_m[te_idx],
            [so.sample_ids[i] for i in te_idx],
        )
        bs = self.config.batch_size
        return (
            DataLoader(train_ds, batch_size=bs, shuffle=True),
            DataLoader(val_ds, batch_size=bs, shuffle=False),
            DataLoader(tgt_ds, batch_size=bs, shuffle=True),
            DataLoader(test_ds, batch_size=bs, shuffle=False),
        )

    def _compute_losses(
        self,
        src_logits: torch.Tensor,
        src_y: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_logits: torch.Tensor,
        tgt_y: torch.Tensor,
        tgt_labeled: torch.Tensor,
        tgt_unlabeled: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if cfg.task_type == "classification":
            src_loss = masked_bce_with_logits(src_logits, src_y, src_mask)
        else:
            src_loss = masked_regression_loss(src_logits, src_y, src_mask, cfg.reg_loss)
        tgt_lab_loss = masked_bce_with_logits(tgt_logits, tgt_y, tgt_labeled)
        tgt_adapt = masked_adentropy_loss(tgt_logits, tgt_unlabeled, cfg.adapt_eta)
        return src_loss, tgt_lab_loss, tgt_adapt

    def train_fold(self, prepared: PreparedData, fold: SourceFold) -> TrainResult:
        self.model.to(self.device)
        src_loader, val_loader, tgt_loader, _ = self._to_loaders(prepared, fold)
        result = TrainResult(fold_id=fold.fold_id)
        tgt_cycle = cycle(tgt_loader)

        for epoch in range(self.config.epochs):
            self.model.train()
            src_sum = tgt_lab_sum = tgt_adapt_sum = total_sum = 0.0
            n_steps = 0
            for src_batch in src_loader:
                tgt_batch = next(tgt_cycle)
                self.optimizer.zero_grad()
                sx = src_batch["x"].to(self.device)
                sy = src_batch["y"].to(self.device)
                sm = src_batch["mask"].to(self.device)
                tx = tgt_batch["x"].to(self.device)
                ty = tgt_batch["y"].to(self.device)
                tl = tgt_batch["labeled_mask"].to(self.device)
                tu = tgt_batch["unlabeled_mask"].to(self.device)

                src_logits = self.model(sx)
                tgt_logits = self.model(tx)
                src_loss, tgt_lab_loss, tgt_adapt = self._compute_losses(
                    src_logits, sy, sm, tgt_logits, ty, tl, tu
                )
                total = src_loss + tgt_lab_loss + self.config.lambda_adapt * tgt_adapt
                total.backward()
                self.optimizer.step()
                src_sum += float(src_loss.item())
                tgt_lab_sum += float(tgt_lab_loss.item())
                tgt_adapt_sum += float(tgt_adapt.item())
                total_sum += float(total.item())
                n_steps += 1

            val_loss = self._validate(val_loader)
            result.epoch_logs.append(
                EpochLog(
                    epoch=epoch,
                    source_loss=src_sum / max(n_steps, 1),
                    target_labeled_loss=tgt_lab_sum / max(n_steps, 1),
                    target_adapt_loss=tgt_adapt_sum / max(n_steps, 1),
                    total_loss=total_sum / max(n_steps, 1),
                    val_loss=val_loss,
                )
            )
        return result

    def _validate(self, val_loader: DataLoader[dict[str, Any]]) -> float:
        self.model.eval()
        losses: list[float] = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(self.device)
                y = batch["y"].to(self.device)
                m = batch["mask"].to(self.device)
                logits = self.model(x)
                if self.config.task_type == "classification":
                    loss = masked_bce_with_logits(logits, y, m)
                else:
                    loss = masked_regression_loss(logits, y, m, self.config.reg_loss)
                losses.append(float(loss.item()))
        return float(sum(losses) / max(len(losses), 1))


def epoch_logs_to_dataframe(logs: list[EpochLog]) -> pd.DataFrame:
    return pd.DataFrame([log.__dict__ for log in logs])
