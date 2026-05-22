"""CLI configuration for multi-label SSDA."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

from ssda_multilabel.sample_id import TARGET_RESPONSE_LABEL_COL

DEFAULT_OUTPUT_DIR = "outputs"
DEFAULT_MULTILABEL_SUBDIR = "ssda_multilabel"


@dataclass(frozen=True)
class MultiLabelConfig:
    task_type: Literal["classification", "regression"]
    source_omics_path: str
    target_omics_path: str
    source_response_path: str
    target_response_path: str
    source_response_col: str
    target_response_col: str
    source_cancer_type_path: str | None
    target_cancer_type_path: str | None
    cancer_type_col: str
    random_seed: int
    source_test_size: float
    n_splits: int
    n_shot: int
    reg_loss: Literal["mse", "mae", "huber"]
    lambda_adapt: float
    output_dir: str
    latent_output_dir: str
    encoder: str
    encoder_h_dims: tuple[int, ...]
    epochs: int
    lr: float
    batch_size: int
    dropout: float
    device: str
    duplicate_response_strategy: Literal["error", "first", "mean"]
    exclude_unknown_cancer_type_for_kmeans: bool
    adapt_eta: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["encoder_h_dims"] = list(self.encoder_h_dims)
        return d

    def save_json(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def resolve_multilabel_output_dir(
    output_dir: str,
    *,
    latent_output_dir: str | None = None,
) -> tuple[str, str]:
    root = Path(output_dir)
    # Default: write directly under --output_dir (no ssda_multilabel/seed_* nesting).
    resolved = str(latent_output_dir or root)
    return str(root), resolved


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-label SSDA experiment")
    p.add_argument("--task_type", choices=["classification", "regression"], required=True)
    p.add_argument("--source_omics_path", required=True)
    p.add_argument("--target_omics_path", required=True)
    p.add_argument("--source_response_path", required=True)
    p.add_argument("--target_response_path", required=True)

    p.add_argument(
        "--source_response_col",
        default=None,
        help="Source response value column (e.g. neg_log2_auc for PRISM, Label for GDSC)",
    )

    p.add_argument("--source_cancer_type_path", default=None)
    p.add_argument("--target_cancer_type_path", default=None)
    p.add_argument(
        "--cancer_type_col",
        default="Cancer_type",
        help="Used only when custom cancer-type CSV is passed; DAPL auto paths use built-in columns",
    )
    p.add_argument("--random_seed", type=int, default=42)
    p.add_argument("--source_test_size", type=float, default=0.1)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--n_shot", type=int, default=3)
    p.add_argument(
        "--reg_loss",
        choices=["mse", "mae", "huber"],
        default="mae",
        help="Source regression loss for training/validation (default: mae)",
    )
    p.add_argument("--lambda_adapt", type=float, default=0.1)
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--latent_output_dir", type=str, default=None)
    p.add_argument("--encoder", choices=["dae", "mlp"], default="mlp")
    p.add_argument("--encoder_h_dims", default="512,256")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    p.add_argument(
        "--duplicate_response_strategy",
        choices=["error", "first", "mean"],
        default="mean",
        help="Collapse duplicate (sample, drug) rows: mean (default), first, or error",
    )
    p.add_argument("--exclude_unknown_cancer_type_for_kmeans", action="store_true")
    p.add_argument("--adapt_eta", type=float, default=0.1)

    # Backward compatibility (deprecated)
    p.add_argument("--sample_id_col", default=None, help=argparse.SUPPRESS)
    p.add_argument("--response_col", default=None, help=argparse.SUPPRESS)
    return p


def config_from_args(args: argparse.Namespace) -> MultiLabelConfig:
    if args.task_type not in ("classification", "regression"):
        raise ValueError(f"invalid task_type: {args.task_type}")

    source_resp_col = args.source_response_col
    if args.response_col and not args.source_response_col:
        source_resp_col = args.response_col
    if not source_resp_col:
        raise ValueError("--source_response_col is required")

    out_root, latent_dir = resolve_multilabel_output_dir(
        args.output_dir,
        latent_output_dir=args.latent_output_dir,
    )
    return MultiLabelConfig(
        task_type=args.task_type,
        source_omics_path=args.source_omics_path,
        target_omics_path=args.target_omics_path,
        source_response_path=args.source_response_path,
        target_response_path=args.target_response_path,
        source_response_col=source_resp_col,
        target_response_col=TARGET_RESPONSE_LABEL_COL,
        source_cancer_type_path=args.source_cancer_type_path,
        target_cancer_type_path=args.target_cancer_type_path,
        cancer_type_col=args.cancer_type_col,
        random_seed=args.random_seed,
        source_test_size=args.source_test_size,
        n_splits=args.n_splits,
        n_shot=args.n_shot,
        reg_loss=args.reg_loss,
        lambda_adapt=args.lambda_adapt,
        output_dir=out_root,
        latent_output_dir=latent_dir,
        encoder=args.encoder,
        encoder_h_dims=_parse_int_tuple(args.encoder_h_dims),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        device=args.device,
        duplicate_response_strategy=args.duplicate_response_strategy,
        exclude_unknown_cancer_type_for_kmeans=args.exclude_unknown_cancer_type_for_kmeans,
        adapt_eta=args.adapt_eta,
    )
