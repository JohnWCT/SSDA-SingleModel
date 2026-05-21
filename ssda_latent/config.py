"""Experiment configuration from CLI arguments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

DEFAULT_OUTPUT_DIR = "outputs"


@dataclass(frozen=True)
class ExperimentConfig:
    drug: str
    gene: str
    n_shot: int
    random_seed: int
    source_test_size: float
    n_splits: int
    encoder: str
    method: str
    epochs: int
    lr: float
    batch_size: int
    dropout: float
    encoder_h_dims: tuple[int, ...]
    predictor_h_dims: tuple[int, ...]
    device: str
    data_path: str
    output_dir: str
    latent_output_dir: str
    umap_path: str
    result: str
    sc_all: str
    source_cancer_type_path: str | None
    target_cancer_type_path: str | None
    sample_id_col: str
    cancer_type_col: str
    missing_cancer_type_policy: Literal["unknown", "exclude"]
    save_legacy_outputs: bool = True
    bottleneck: int | None = None

    @property
    def gene_suffix(self) -> str:
        return self.gene

    @property
    def latent_dim(self) -> int | None:
        """Optional override; when None, ``utils.cell_dim`` pathway dim is used."""
        return self.bottleneck


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def _legacy_dir(path: Path) -> str:
    """Directory string with trailing slash for legacy path concatenation."""
    s = path.as_posix()
    return s if s.endswith("/") else f"{s}/"


def resolve_output_paths(
    output_dir: str,
    *,
    latent_output_dir: str | None = None,
    umap_path: str | None = None,
    result: str | None = None,
    sc_all: str | None = None,
) -> tuple[str, str, str, str]:
    """Resolve artifact paths under ``output_dir`` unless explicitly overridden."""
    root = Path(output_dir)
    latent = str(latent_output_dir or (root / "latent_ssda"))
    umap = _legacy_dir(Path(umap_path) if umap_path else root / "legacy" / "figure")
    result_p = _legacy_dir(Path(result) if result else root / "legacy" / "results" / "sc")
    sc_all_p = _legacy_dir(Path(sc_all) if sc_all else root / "legacy" / "sc" / "all_path")
    return latent, umap, result_p, sc_all_p


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSDA4Drug latent extension")
    parser.add_argument("--path", type=str, default="./Datasets/processedData")
    parser.add_argument("--drug", type=str, default="Gefitinib")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--gene", type=str, default="_tp4k")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--source_test_size", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Root directory for all experiment_shot_ssda artifacts "
            "(default: outputs/). Subdirs: latent_ssda/, legacy/..."
        ),
    )
    parser.add_argument(
        "--latent_output_dir",
        type=str,
        default=None,
        help="Override latent export root (default: <output_dir>/latent_ssda)",
    )
    parser.add_argument("--source_cancer_type_path", type=str, default=None)
    parser.add_argument("--target_cancer_type_path", type=str, default=None)
    parser.add_argument("--sample_id_col", type=str, default="Sample_ID")
    parser.add_argument("--cancer_type_col", type=str, default="Cancer_type")
    parser.add_argument(
        "--missing_cancer_type_policy",
        type=str,
        default="unknown",
        choices=["unknown", "exclude"],
    )
    parser.add_argument(
        "--umap_path",
        type=str,
        default=None,
        help="Override legacy figure dir (default: <output_dir>/legacy/figure/)",
    )
    parser.add_argument(
        "--result",
        type=str,
        default=None,
        help="Override legacy AUC log dir (default: <output_dir>/legacy/results/sc/)",
    )
    parser.add_argument(
        "--sc_all",
        type=str,
        default=None,
        help="Override legacy checkpoint dir (default: <output_dir>/legacy/sc/all_path/)",
    )
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--encoder", type=str, default="DAE")
    parser.add_argument("--method", type=str, default="adv")
    parser.add_argument("--encoder_h_dims", type=str, default="512,256")
    parser.add_argument("--predictor_h_dims", type=str, default="64,32")
    parser.add_argument(
        "--bottleneck",
        type=int,
        default=None,
        help="Optional latent dim override for MLP/DAE bottleneck; default uses utils.cell_dim pathway",
    )
    parser.add_argument("--sampling_method", type=str, default="weight")
    parser.add_argument("--shot_method", type=str, default="3-shot")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if not 0 < args.source_test_size < 1:
        raise ValueError("source_test_size must be in (0, 1)")
    if args.n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    latent_dir, umap_path, result_path, sc_all_path = resolve_output_paths(
        args.output_dir,
        latent_output_dir=args.latent_output_dir,
        umap_path=args.umap_path,
        result=args.result,
        sc_all=args.sc_all,
    )
    return ExperimentConfig(
        drug=args.drug,
        gene=args.gene,
        n_shot=args.n,
        random_seed=args.random_seed,
        source_test_size=args.source_test_size,
        n_splits=args.n_splits,
        encoder=args.encoder,
        method=args.method,
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        dropout=args.dropout,
        encoder_h_dims=_parse_int_tuple(args.encoder_h_dims),
        predictor_h_dims=_parse_int_tuple(args.predictor_h_dims),
        device=args.device,
        data_path=args.path,
        output_dir=args.output_dir,
        latent_output_dir=latent_dir,
        umap_path=umap_path,
        result=result_path,
        sc_all=sc_all_path,
        source_cancer_type_path=args.source_cancer_type_path,
        target_cancer_type_path=args.target_cancer_type_path,
        sample_id_col=args.sample_id_col,
        cancer_type_col=args.cancer_type_col,
        missing_cancer_type_policy=args.missing_cancer_type_policy,
        bottleneck=args.bottleneck,
    )
