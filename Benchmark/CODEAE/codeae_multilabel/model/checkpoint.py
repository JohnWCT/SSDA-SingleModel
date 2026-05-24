"""Checkpoint save/load for multilabel CODE-AE."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from codeae_multilabel.model.wrapper import MultiLabelCodeAEModel


def load_pretrain_checkpoint(
    model: MultiLabelCodeAEModel,
    checkpoint_path: str,
    strict_encoder: bool = True,
    ignore_prediction_head: bool = True,
) -> dict[str, Any]:
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"pretrain checkpoint not found: {checkpoint_path}")
    state = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "shared_encoder" in state:
        enc_state = state["shared_encoder"]
    elif isinstance(state, dict):
        enc_state = {k.replace("codeae_core.", ""): v for k, v in state.items() if "codeae_core" in k}
        if not enc_state:
            enc_state = {k: v for k, v in state.items() if not k.startswith("prediction_head")}
    else:
        enc_state = state
    missing, unexpected = model.codeae_core.load_state_dict(enc_state, strict=strict_encoder)
    report: dict[str, Any] = {
        "missing_keys": list(missing),
        "unexpected_keys": list(unexpected),
        "ignore_prediction_head": ignore_prediction_head,
    }
    if ignore_prediction_head:
        report["prediction_head"] = "initialized fresh"
    return report


def save_finetune_checkpoint(
    model: MultiLabelCodeAEModel,
    optimizer: torch.optim.Optimizer,
    metadata: dict[str, Any],
    path: str,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_finetune_checkpoint(model: MultiLabelCodeAEModel, path: str) -> dict[str, Any]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    meta: dict[str, Any] = ckpt.get("metadata", {})
    return meta
