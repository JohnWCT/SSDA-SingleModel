"""CLI configuration for CODE-AE multilabel."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from codeae_multilabel.contracts import CodeAEMultilabelConfig

# Merged unlabeled defaults when train_params.json is missing (upstream CODE-AE structure).
_ORIGINAL_JSON: dict[str, Any] = {
    "unlabeled": {
        "batch_size": 64,
        "lr": 0.0001,
        "pretrain_num_epochs": 500,
        "train_num_epochs": 1000,
        "alpha": 1.0,
        "classifier_hidden_dims": [64, 32],
    },
    "labeled": {
        "classifier_hidden_dims": [64, 32],
        "batch_size": 64,
        "lr": 0.0001,
        "train_num_epochs": 2000,
        "decay_coefficient": 0.1,
    },
    "encoder_hidden_dims": [512, 256],
    "latent_dim": 128,
    "dop": 0.1,
}

# SSDA4Drug fork drug_ft/pretrain hyper_main grid on unlabeled block (optional).
SSDA_BENCHMARK_PRETRAIN_GRID: dict[str, Any] = {
    "pretrain_num_epochs": 0,
    "train_num_epochs": 100,
    "dop": 0.0,
}

# Upstream drug_ft_hyper_main.py --metric -> multilabel early-stop metric names.
UPSTREAM_FINETUNE_METRIC_MAP: dict[str, str] = {
    "auroc": "macro_auroc",
    "auprc": "macro_aupr",
    "aps": "macro_aupr",
}

# fine_tuning.fine_tune_encoder model_save_check tolerance_count.
UPSTREAM_EARLY_STOPPING_TOLERANCE: int = 10

_TRAIN_PARAM_KEYS = (
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
    "decay_coefficient",
)


def wrap_training_params(
    training_params: dict[str, Any],
    section: Literal["unlabeled", "labeled"] = "unlabeled",
) -> dict[str, Any]:
    """Same merge as CODE-AE pretrain_hyper_main.wrap_training_params."""
    aux = {k: v for k, v in training_params.items() if k not in ("unlabeled", "labeled")}
    aux.update(training_params.get(section, {}))
    return aux


def _read_train_params_json(codeae_root: Path) -> dict[str, Any]:
    train_json = codeae_root / "model_save" / "train_params.json"
    if train_json.is_file():
        with train_json.open(encoding="utf-8") as f:
            return json.load(f)
    return json.loads(json.dumps(_ORIGINAL_JSON))


def load_train_params(
    codeae_root: Path | None = None,
    section: Literal["unlabeled", "labeled"] = "unlabeled",
) -> dict[str, Any]:
    """Load and merge train_params.json (root keys + section), matching upstream CODE-AE."""
    if codeae_root is None:
        codeae_root = Path(__file__).resolve().parents[1]
    raw = _read_train_params_json(codeae_root)
    params = wrap_training_params(raw, section=section)
    params.setdefault("norm_flag", True)
    if section == "labeled":
        params.setdefault("pretrain_num_epochs", wrap_training_params(raw, "unlabeled")["pretrain_num_epochs"])
    return params


def apply_ssda_benchmark_pretrain_grid(params: dict[str, Any], method: str) -> dict[str, Any]:
    """SSDA4Drug-local pretrain_hyper_main grid (0 AE + 100 WGAN + dop 0). Opt-in only."""
    out = dict(params)
    if method in ("code_adv", "adsn", "adae", "dsnw"):
        out.update(SSDA_BENCHMARK_PRETRAIN_GRID)
    return out


def build_pretrain_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CODE-AE multilabel pretraining")
    p.add_argument("--source_omics_path", required=True)
    p.add_argument("--target_omics_path", required=True)
    p.add_argument("--source_sample_col", default="Sample_ID")
    p.add_argument("--target_sample_col", default="tissue_id")
    p.add_argument("--method", default="code_adv")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override train_num_epochs (WGAN); same as --train_num_epochs",
    )
    p.add_argument("--pretrain_num_epochs", type=int, default=None)
    p.add_argument("--train_num_epochs", type=int, default=None)
    p.add_argument("--dop", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=2020, help="Upstream CODE-AE default random seed")
    p.add_argument("--norm_flag", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--retrain_flag", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--ssda_benchmark_grid",
        action="store_true",
        help="Apply SSDA4Drug fork grid (pretrain=0, train=100, dop=0) instead of upstream json",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_drugs", type=int, default=None)
    return p


def build_finetune_arg_parser() -> argparse.ArgumentParser:
    """CLI aligned with upstream CODE-AE drug_ft_hyper_main.py (labeled train_params block)."""
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
    p.add_argument(
        "--metric",
        default=None,
        help="Early-stop metric; upstream uses auroc/auprc (mapped to macro_auroc/macro_aupr)",
    )
    p.add_argument("--reg_loss", choices=["mae"], default="mae")
    p.add_argument("--prediction_threshold", type=float, default=0.5)
    p.add_argument("--regression_binary_threshold", type=float, default=1.0)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--source_test_size", type=float, default=0.001)
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override labeled train_num_epochs (upstream default 2000)",
    )
    p.add_argument("--train_num_epochs", type=int, default=None)
    p.add_argument("--dop", type=float, default=None)
    p.add_argument(
        "--decay_coefficient",
        type=float,
        default=None,
        help="Encoder unfreeze LR decay (upstream labeled default 0.1)",
    )
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--seed", type=int, default=2020)
    p.add_argument(
        "--duplicate_strategy", choices=["mean", "median", "first", "error"], default="mean"
    )
    p.add_argument("--source_cancer_type_path", default=None)
    p.add_argument("--target_cancer_type_path", default=None)
    p.add_argument("--cancer_type_col", default="Cancer_type")
    p.add_argument("--norm_flag", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument(
        "--retrain_flag",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Upstream drug_ft default False (pretrain done separately in multilabel)",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--max_drugs", type=int, default=None)
    return p


def _resolve_hyperparams(
    args: argparse.Namespace,
    mode: Literal["pretrain", "finetune"],
) -> dict[str, Any]:
    codeae_root = Path(__file__).resolve().parents[1]
    section: Literal["unlabeled", "labeled"] = "unlabeled" if mode == "pretrain" else "labeled"
    tp = load_train_params(codeae_root, section=section)
    if mode == "pretrain" and getattr(args, "ssda_benchmark_grid", False):
        tp = apply_ssda_benchmark_pretrain_grid(tp, args.method)

    pretrain_epochs = int(tp.get("pretrain_num_epochs", 500))
    train_epochs = int(tp["train_num_epochs"])
    dop = float(tp["dop"])

    if getattr(args, "pretrain_num_epochs", None) is not None:
        pretrain_epochs = int(args.pretrain_num_epochs)
    if getattr(args, "train_num_epochs", None) is not None:
        train_epochs = int(args.train_num_epochs)
    elif getattr(args, "epochs", None) is not None:
        train_epochs = int(args.epochs)

    if getattr(args, "dop", None) is not None:
        dop = float(args.dop)

    decay = float(tp.get("decay_coefficient", 0.1))
    if getattr(args, "decay_coefficient", None) is not None:
        decay = float(args.decay_coefficient)

    return {
        "pretrain_num_epochs": pretrain_epochs,
        "train_num_epochs": train_epochs,
        "dop": dop,
        "decay_coefficient": decay,
        "tp": tp,
    }


def _base_config_dict(
    args: argparse.Namespace, mode: Literal["pretrain", "finetune"]
) -> dict[str, Any]:
    resolved = _resolve_hyperparams(args, mode)
    tp = resolved["tp"]
    pretrain_epochs = resolved["pretrain_num_epochs"]
    train_epochs = resolved["train_num_epochs"]
    dop = resolved["dop"]
    decay_coefficient = resolved["decay_coefficient"]
    device = args.device or ("cuda" if __import__("torch").cuda.is_available() else "cpu")

    if mode == "finetune":
        unlabeled = load_train_params(Path(__file__).resolve().parents[1], section="unlabeled")
        pretrain_epochs = int(unlabeled.get("pretrain_num_epochs", 500))

    norm_default = bool(tp.get("norm_flag", True))
    norm_flag = (
        bool(args.norm_flag) if getattr(args, "norm_flag", None) is not None else norm_default
    )

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
        "epochs": int(train_epochs),
        "lr": float(args.lr if args.lr is not None else tp["lr"]),
        "seed": args.seed,
        "latent_dim": int(tp["latent_dim"]),
        "encoder_hidden_dims": tuple(int(x) for x in tp["encoder_hidden_dims"]),
        "classifier_hidden_dims": tuple(int(x) for x in tp["classifier_hidden_dims"]),
        "pretrain_num_epochs": pretrain_epochs,
        "train_num_epochs": train_epochs,
        "norm_flag": norm_flag,
        "alpha": float(tp.get("alpha", 1.0)),
        "dop": dop,
        "decay_coefficient": decay_coefficient,
        "early_stopping_tolerance": UPSTREAM_EARLY_STOPPING_TOLERANCE,
        "device": device,
        "retrain_flag": bool(args.retrain_flag),
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


def resolve_finetune_metric(task_type: str, metric: str | None) -> str:
    if metric is None:
        return "macro_auroc" if task_type == "classification" else "macro_mae"
    key = metric.strip().lower()
    if key in UPSTREAM_FINETUNE_METRIC_MAP:
        return UPSTREAM_FINETUNE_METRIC_MAP[key]
    return metric


def config_from_finetune_args(args: argparse.Namespace) -> CodeAEMultilabelConfig:
    if args.task_type not in ("classification", "regression"):
        raise ValueError(f"invalid task_type: {args.task_type}")
    if not args.pretrain_checkpoint:
        raise ValueError("fine-tune requires --pretrain_checkpoint")
    d = _base_config_dict(args, "finetune")
    metric = resolve_finetune_metric(args.task_type, args.metric)
    return CodeAEMultilabelConfig(
        task_type=args.task_type,
        source_response_path=args.source_response_path,
        target_response_path=args.target_response_path,
        pretrain_checkpoint=args.pretrain_checkpoint,
        metric=metric,
        **d,
    )


# Backward-compatible alias for tests/docs referring to upstream merge defaults.
DEFAULT_TRAIN_PARAMS: dict[str, Any] = load_train_params()


def config_to_dict(config: CodeAEMultilabelConfig) -> dict[str, Any]:
    d = asdict(config)
    d["encoder_hidden_dims"] = list(config.encoder_hidden_dims)
    d["classifier_hidden_dims"] = list(config.classifier_hidden_dims)
    return d
