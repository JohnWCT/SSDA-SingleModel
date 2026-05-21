"""Experiment configuration from CLI arguments."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Literal


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

    @property
    def gene_suffix(self) -> str:
        return self.gene

    @property
    def latent_dim(self) -> int:
        return 128


def _parse_int_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in value.split(",") if x.strip())


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="SSDA4Drug latent extension")
    parser.add_argument("--path", type=str, default="./Datasets/processedData")
    parser.add_argument("--drug", type=str, default="Gefitinib")
    parser.add_argument("--n", type=int, default=3)
    parser.add_argument("--gene", type=str, default="_tp4k")
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--source_test_size", type=float, default=0.1)
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--latent_output_dir", type=str, default="save/latent_ssda")
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
    parser.add_argument("--umap_path", type=str, default="save/figure/")
    parser.add_argument("--result", type=str, default="save/results/sc/")
    parser.add_argument("--sc_all", type=str, default="save/sc/all_path/")
    parser.add_argument("--device", type=str, default="gpu")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--encoder", type=str, default="DAE")
    parser.add_argument("--method", type=str, default="adv")
    parser.add_argument("--encoder_h_dims", type=str, default="512,256")
    parser.add_argument("--predictor_h_dims", type=str, default="64,32")
    parser.add_argument("--bottleneck", type=int, default=256)
    parser.add_argument("--sampling_method", type=str, default="weight")
    parser.add_argument("--shot_method", type=str, default="3-shot")
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    if not 0 < args.source_test_size < 1:
        raise ValueError("source_test_size must be in (0, 1)")
    if args.n_splits < 2:
        raise ValueError("n_splits must be >= 2")
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
        latent_output_dir=args.latent_output_dir,
        umap_path=args.umap_path,
        result=args.result,
        sc_all=args.sc_all,
        source_cancer_type_path=args.source_cancer_type_path,
        target_cancer_type_path=args.target_cancer_type_path,
        sample_id_col=args.sample_id_col,
        cancer_type_col=args.cancer_type_col,
        missing_cancer_type_policy=args.missing_cancer_type_policy,
    )
