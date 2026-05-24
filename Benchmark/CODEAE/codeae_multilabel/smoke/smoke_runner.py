"""Run CODE-AE multilabel smoke tests (classification + regression) in Docker-friendly paths."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

_DAPL = Path("/workspace/DAPL-master")
_REPO = Path("/workspace/SSDA4Drug")
_CODEAE = _REPO / "Benchmark" / "CODEAE"

# Aligned with experiment_multilabel_ssda.py SMOKE_*_ARGS
CLS = {
    "source_omics": _DAPL / "data/pretrain_ccle.csv",
    "target_omics": _DAPL / "data/TCGA/pretrain_tcga.csv",
    "source_response": _DAPL / "data/GDSC2_fitted_dose_response_MaxScreen_raw.csv",
    "target_response": _DAPL / "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv",
    "source_response_col": "Label",
    "out_pretrain": _REPO / "outputs_codeae_smoke_cls_pretrain",
    "out_ft": _REPO / "outputs_codeae_smoke_classification",
}
REG = {
    "source_omics": _DAPL / "data_Winnie/CCLE_impact_hotspot.csv",
    "target_omics": _DAPL / "data_Winnie/TCGA_impact_hotspot.csv",
    "source_response": _DAPL / "data_Winnie/PRISM_drug_sensitivity.csv",
    "target_response": _DAPL / "data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv",
    "source_response_col": "neg_log2_auc",
    "out_pretrain": _REPO / "outputs_codeae_smoke_reg_pretrain",
    "out_ft": _REPO / "outputs_codeae_smoke_regression",
}

SMOKE_BASE = [
    "--epochs",
    "1",
    "--batch_size",
    "32",
    "--max_samples",
    "512",
    "--overwrite",
    "--seed",
    "42",
]
SMOKE_FINETUNE_EXTRA = [
    "--n_splits",
    "2",
    "--source_test_size",
    "0.001",
]


def _run(cmd: list[str], cwd: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _check_artifacts(out_ft: Path) -> list[str]:
    required = [
        "drug_list.csv",
        "config.json",
        "source_test_metrics_summary_across_folds.csv",
        "source_test_metrics_summary_fold_mean_std.csv",
        "target_eval_metrics_summary_across_folds.csv",
        "target_eval_metrics_summary_fold_mean_std.csv",
        "eval_metrics_summary_fold_mean_std.csv",
        "source_test_metrics_per_drug_fold_mean_std.csv",
        "target_eval_metrics_per_drug_fold_mean_std.csv",
        "latent_metrics_summary.csv",
        "kmeans_cancer_type_summary.csv",
        "kmeans_cancer_type_fold_mean_std.csv",
        "fold_0/best_model.pt",
        "fold_0/source_prediction_results.csv",
        "fold_0/target_prediction_results.csv",
        "fold_0/source_metrics_summary.csv",
        "fold_0/target_metrics_summary.csv",
        "fold_0/source_latent_representation.csv",
        "fold_0/target_latent_representation.csv",
        "fold_0/latent_representation.pkl",
        "fold_0/latent_distribution_metrics.csv",
        "fold_0/kmeans_cancer_type_metrics.csv",
        "fold_0/tsne_domain_mixing.png",
    ]
    missing = [p for p in required if not (out_ft / p).is_file()]
    return missing


def _assert_overall_in_summary(path: Path) -> None:
    import pandas as pd

    df = pd.read_csv(path)
    if "overall" not in df.columns:
        raise RuntimeError(f"overall column missing in {path}")


def run_pretrain(paths: dict[str, Path], python: str) -> Path:
    out = paths["out_pretrain"]
    ckpt = out / "pretrain" / "checkpoint.pt"
    cmd = [
        python,
        str(_CODEAE / "pretrain_multilabel_hyper_main.py"),
        "--source_omics_path",
        str(paths["source_omics"]),
        "--target_omics_path",
        str(paths["target_omics"]),
        "--output_dir",
        str(out),
        *SMOKE_BASE,
    ]
    _run(cmd, _REPO)
    if not ckpt.is_file():
        raise FileNotFoundError(f"pretrain checkpoint not found: {ckpt}")
    return ckpt


def run_finetune(
    paths: dict[str, Path],
    task_type: str,
    ckpt: Path,
    python: str,
) -> None:
    out = paths["out_ft"]
    cmd = [
        python,
        str(_CODEAE / "drug_ft_multilabel_hyper_main.py"),
        "--task_type",
        task_type,
        "--source_omics_path",
        str(paths["source_omics"]),
        "--target_omics_path",
        str(paths["target_omics"]),
        "--source_response_path",
        str(paths["source_response"]),
        "--target_response_path",
        str(paths["target_response"]),
        "--source_response_col",
        paths["source_response_col"],
        "--pretrain_checkpoint",
        str(ckpt),
        "--output_dir",
        str(out),
        *SMOKE_BASE,
        *SMOKE_FINETUNE_EXTRA,
    ]
    if task_type == "regression":
        cmd.extend(["--metric", "macro_mae"])
    _run(cmd, _REPO)
    missing = _check_artifacts(out)
    if missing:
        raise FileNotFoundError(f"missing artifacts under {out}: {missing}")
    _assert_overall_in_summary(out / "fold_0" / "source_metrics_summary.csv")
    print(f"OK: {task_type} smoke artifacts verified under {out}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="CODE-AE multilabel smoke runner")
    parser.add_argument(
        "--mode",
        choices=["pretrain", "finetune_cls", "finetune_reg", "all", "cls", "reg"],
        default="all",
    )
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    if args.mode == "all":
        ckpt_cls = run_pretrain(CLS, args.python)
        run_finetune(CLS, "classification", ckpt_cls, args.python)
        ckpt_reg = run_pretrain(REG, args.python)
        run_finetune(REG, "regression", ckpt_reg, args.python)
    elif args.mode == "cls":
        ckpt = run_pretrain(CLS, args.python)
        run_finetune(CLS, "classification", ckpt, args.python)
    elif args.mode == "reg":
        ckpt = run_pretrain(REG, args.python)
        run_finetune(REG, "regression", ckpt, args.python)
    elif args.mode == "pretrain":
        run_pretrain(CLS, args.python)
        run_pretrain(REG, args.python)
    elif args.mode == "finetune_cls":
        ckpt = CLS["out_pretrain"] / "pretrain" / "checkpoint.pt"
        run_finetune(CLS, "classification", ckpt, args.python)
    elif args.mode == "finetune_reg":
        ckpt = REG["out_pretrain"] / "pretrain" / "checkpoint.pt"
        run_finetune(REG, "regression", ckpt, args.python)


if __name__ == "__main__":
    main()
