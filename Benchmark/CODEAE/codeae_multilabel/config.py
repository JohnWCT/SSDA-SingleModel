"""CLI configuration for CODE-AE multilabel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from codeae_multilabel.contracts import CodeAEMultilabelConfig

DEFAULT_TRAIN_PARAMS: dict[str, Any] = {
    "latent_dim": 128,
    "encoder_hidden_dims": [512, 256, 128],
    "classifier_hidden_dims": [128, 64],
    "lr": 1e-4,
    "batch_size": 64,
    "pretrain_num_epochs": 2,
    "train_num_epochs": 50,
    "norm_flag": True,
    "alpha": 1.0,
    "dop": 0.1,
}


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def load_train_params(codeae_root: Path | None = None) -> dict[str, Any]:
    params = dict(DEFAULT_TRAIN_PARAMS)
    if codeae_root is None:
        codeae_root = Path(__file__).resolve().parents[1]
    train_json = codeae_root / "model_save" / "train_params.json"
    if train_json.is_file():
        with train_json.open(encoding="utf-8") as f:
            loaded = json.load(f)
        unlabeled = loaded.get("unlabeled", loaded)
        for k in (
            "latent_dim",
            "encoder_hidden_dims",
            "classifier_hidden_dims",
            "lr",
            "batch_size",
            "pretrain_num_epochs",
            "train_num_epochs",
            "norm_flag",
            "alpha",
            "dop",
        ):
            if k in unlabeled:
                params[k] = unlabeled[k]
            elif k in loaded:
                params[k] = loaded[k]
    return params


def build_pretrain_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CODE-AE multilabel pretraining")
    p.add_argument("--source_omics_path", required=True)
    p.add_argument("--target_omics_path", required=True)
    p.add_argument("--source_sample_col", default="Sample_ID")
    p.add_argument("--target_sample_col", default="tissue_id")
    p.add_argument("--method", default="code_adv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--epochs", type=int, default=None, help="pretrain_num_epochs override")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm_flag", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--retrain_flag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default=None)
    p.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap on source/target omics samples (smoke/debug only)",
    )
    p.add_argument(
        "--max_drugs",
        type=int,
        default=None,
        help="Optional cap on drug list size after union (smoke/debug only)",
    )
    return p


def build_finetune_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CODE-AE multilabel fine-tuning")
    p.add_argument("--task_type", choices=["classification", "regression"], required=True)
    p.add_argument("--source_omics_path", required=True)
    p.add_argument("--target_omics_path", required=True)
    p.add_argument("--source_response_path", required=True)
    p.add_argument("--target_response_path", required=True)
    p.add_argument("--source_sample_col", default="Sample_ID")
    p.add_argument("--target_sample_col", default="tissue_id")
    p.add_argument("--target_response_sample_col", default="Patient_id")
    p.add_argument("--drug_col", default="drug_name")
    p.add_argument("--source_response_col", default="Label")
    p.add_argument("--target_response_col", default="Label")
    p.add_argument("--pretrain_checkpoint", required=True)
    p.add_argument("--method", default="code_adv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--metric", default=None)
    p.add_argument("--reg_loss", choices=["mae"], default="mae")
    p.add_argument("--prediction_threshold", type=float, default=0.5)
    p.add_argument("--regression_binary_threshold", type=float, default=1.0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--source_test_size", type=float, default=0.001)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--duplicate_strategy", choices=["mean", "median", "first", "error"], default="mean"
    )
    p.add_argument("--source_cancer_type_path", default=None)
    p.add_argument("--target_cancer_type_path", default=None)
    p.add_argument("--cancer_type_col", default="Cancer_type")
    p.add_argument("--device", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_drugs", type=int, default=None)
    return p


def _base_config_dict(
    args: argparse.Namespace, mode: Literal["pretrain", "finetune"]
) -> dict[str, Any]:
    codeae_root = Path(__file__).resolve().parents[1]
    tp = load_train_params(codeae_root)
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")
    epochs = args.epochs
    pretrain_epochs = int(epochs if epochs is not None else tp["pretrain_num_epochs"])
    train_epochs = int(epochs if epochs is not None else tp["train_num_epochs"])
    if mode == "pretrain":
        train_epochs = int(epochs if epochs is not None else tp["train_num_epochs"])
    else:
        pretrain_epochs = tp["pretrain_num_epochs"]
    return {
        "source_omics_path": args.source_omics_path,
        "target_omics_path": args.target_omics_path,
        "source_sample_col": getattr(args, "source_sample_col", "Sample_ID"),
        "target_sample_col": getattr(args, "target_sample_col", "tissue_id"),
        "target_response_sample_col": getattr(args, "target_response_sample_col", "Patient_id"),
        "drug_col": getattr(args, "drug_col", "drug_name"),
        "source_response_col": getattr(args, "source_response_col", "Label"),
        "target_response_col": getattr(args, "target_response_col", "Label"),
        "method": args.method,
        "output_dir": args.output_dir,
        "overwrite": args.overwrite,
        "batch_size": int(args.batch_size if args.batch_size is not None else tp["batch_size"]),
        "epochs": int(
            epochs
            if epochs is not None
            else (pretrain_epochs if mode == "pretrain" else train_epochs)
        ),
        "lr": float(args.lr if args.lr is not None else tp["lr"]),
        "seed": args.seed,
        "latent_dim": int(tp["latent_dim"]),
        "encoder_hidden_dims": tuple(int(x) for x in tp["encoder_hidden_dims"]),
        "classifier_hidden_dims": tuple(int(x) for x in tp["classifier_hidden_dims"]),
        "pretrain_num_epochs": pretrain_epochs,
        "train_num_epochs": train_epochs if mode == "finetune" else int(tp["train_num_epochs"]),
        "norm_flag": bool(
            getattr(args, "norm_flag", None)
            if getattr(args, "norm_flag", None) is not None
            else tp["norm_flag"]
        ),
        "alpha": float(tp["alpha"]),
        "dop": float(tp["dop"]),
        "device": device,
        "retrain_flag": bool(getattr(args, "retrain_flag", True)),
        "es_flag": False,
        "n_splits": int(getattr(args, "n_splits", 5)),
        "source_test_size": float(getattr(args, "source_test_size", 0.001)),
        "reg_loss": "mae",
        "prediction_threshold": float(getattr(args, "prediction_threshold", 0.5)),
        "regression_binary_threshold": float(getattr(args, "regression_binary_threshold", 1.0)),
        "duplicate_strategy": getattr(args, "duplicate_strategy", "mean"),
        "source_cancer_type_path": getattr(args, "source_cancer_type_path", None),
        "target_cancer_type_path": getattr(args, "target_cancer_type_path", None),
        "cancer_type_col": getattr(args, "cancer_type_col", None),
        "max_samples": getattr(args, "max_samples", None),
        "max_drugs": getattr(args, "max_drugs", None),
    }


def config_from_pretrain_args(args: argparse.Namespace) -> CodeAEMultilabelConfig:
    d = _base_config_dict(args, "pretrain")
    return CodeAEMultilabelConfig(
        task_type="classification",
        source_response_path=None,
        target_response_path=None,
        pretrain_checkpoint=None,
        metric=None,
        **d,
    )


def config_from_finetune_args(args: argparse.Namespace) -> CodeAEMultilabelConfig:
    if args.task_type not in ("classification", "regression"):
        raise ValueError(f"invalid task_type: {args.task_type}")
    if not args.pretrain_checkpoint:
        raise ValueError("fine-tune requires --pretrain_checkpoint")
    d = _base_config_dict(args, "finetune")
    metric = args.metric
    if metric is None:
        metric = "macro_auroc" if args.task_type == "classification" else "macro_mae"
    return CodeAEMultilabelConfig(
        task_type=args.task_type,
        source_response_path=args.source_response_path,
        target_response_path=args.target_response_path,
        pretrain_checkpoint=args.pretrain_checkpoint,
        metric=metric,
        **d,
    )


def config_to_dict(config: CodeAEMultilabelConfig) -> dict[str, Any]:
    d = asdict(config)
    d["encoder_hidden_dims"] = list(config.encoder_hidden_dims)
    d["classifier_hidden_dims"] = list(config.classifier_hidden_dims)
    return d
