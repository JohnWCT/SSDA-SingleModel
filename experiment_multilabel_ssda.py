"""Entry point: multi-label / multi-drug SSDA pipeline."""

from __future__ import annotations

import warnings

import matplotlib
import pandas as pd

from ssda_multilabel.config import build_arg_parser, config_from_args
from ssda_multilabel.export import ArtifactWriter
from ssda_multilabel.latent import encode_latent_dict
from ssda_multilabel.latent_eval import (
    compute_distribution_metrics,
    compute_kmeans_cancer_type_metrics,
    plot_tsne_cancer_type,
    plot_tsne_domain,
)
from ssda_multilabel.metrics import compute_metrics_from_predictions
from ssda_multilabel.model import build_model
from ssda_multilabel.prediction import build_prediction_long_table, predict_matrix
from ssda_multilabel.prepare import prepare_multilabel_data
from ssda_multilabel.seed import set_global_seed
from ssda_multilabel.training import MultiLabelSSDTrainer, epoch_logs_to_dataframe


def main() -> None:
    parser = build_arg_parser()
    args, _ = parser.parse_known_args()
    matplotlib.use("Agg")
    warnings.filterwarnings("ignore")
    config = config_from_args(args)
    set_global_seed(config.random_seed)

    writer = ArtifactWriter(config.latent_output_dir, config.random_seed)
    writer.write_config(config)

    prepared = prepare_multilabel_data(config)
    cancer_summary = (
        prepared.cancer_type_df.groupby("domain")["cancer_type"]
        .apply(lambda s: (s == "Unknown").sum())
        .reset_index(name="n_unknown")
        if prepared.cancer_type_df is not None
        else pd.DataFrame()
    )
    writer.write_preparation_artifacts(
        prepared.drug_index,
        prepared.source_response,
        prepared.target_response,
        prepared.target_masks,
        prepared.alignment_report,
        prepared.nshot_summary,
        cancer_summary,
    )

    n_features = prepared.source_omics.x.shape[1]
    n_drugs = prepared.drug_index.n_drugs
    cancer_map = (
        dict(zip(prepared.cancer_type_df["sample_id"], prepared.cancer_type_df["cancer_type"]))
        if prepared.cancer_type_df is not None
        else {}
    )

    all_src_metrics: list[pd.DataFrame] = []
    all_tgt_metrics: list[pd.DataFrame] = []
    all_latent_metrics: list[pd.DataFrame] = []
    all_kmeans: list[pd.DataFrame] = []

    for fold in prepared.folds:
        model = build_model(
            n_features,
            n_drugs,
            config.encoder,
            config.encoder_h_dims,
            dropout=config.dropout,
        )
        trainer = MultiLabelSSDTrainer(model, config)
        train_result = trainer.train_fold(prepared, fold)
        writer.save_model(fold.fold_id, model)
        loss_log = epoch_logs_to_dataframe(train_result.epoch_logs)
        writer.write_fold_csv(fold.fold_id, "masked_loss_log.csv", loss_log)

        so, sr, to, tr = (
            prepared.source_omics,
            prepared.source_response,
            prepared.target_omics,
            prepared.target_response,
        )
        tm = prepared.target_masks

        src_scores = predict_matrix(model, so.x, config.batch_size, config.device)
        tgt_scores = predict_matrix(model, to.x, config.batch_size, config.device)

        src_roles = ["source_fold_train"] * len(so.sample_ids)
        for i in fold.train_indices:
            src_roles[i] = "source_fold_train"
        for i in fold.val_indices:
            src_roles[i] = "source_fold_val"
        for i in fold.test_indices:
            src_roles[i] = "source_test"

        tgt_roles = []
        for i in range(len(to.sample_ids)):
            if tm.labeled_mask[i].sum() > 0:
                tgt_roles.append("target_labeled")
            elif tm.unlabeled_mask[i].sum() > 0:
                tgt_roles.append("target_unlabeled")
            else:
                tgt_roles.append("target_observed_only")

        src_pred = build_prediction_long_table(
            src_scores,
            sr.y,
            sr.mask,
            list(so.sample_ids),
            prepared.drug_index,
            "source",
            src_roles,
            config.task_type,
            fold.fold_id,
            config.random_seed,
            cancer_map,
        )
        tgt_pred = build_prediction_long_table(
            tgt_scores,
            tr.y,
            tr.mask,
            list(to.sample_ids),
            prepared.drug_index,
            "target",
            tgt_roles,
            config.task_type,
            fold.fold_id,
            config.random_seed,
            cancer_map,
        )
        writer.write_fold_csv(fold.fold_id, "source_prediction_results.csv", src_pred)
        writer.write_fold_csv(fold.fold_id, "target_prediction_results.csv", tgt_pred)

        if config.task_type == "classification":
            src_per, src_sum = compute_metrics_from_predictions(
                src_pred, "classification", "source"
            )
            tgt_per, tgt_sum = compute_metrics_from_predictions(
                tgt_pred, "classification", "target"
            )
        else:
            src_per, src_sum = compute_metrics_from_predictions(src_pred, "regression", "source")
            tgt_per, tgt_sum = compute_metrics_from_predictions(
                tgt_pred, "classification", "target"
            )

        writer.write_fold_csv(fold.fold_id, "source_metrics_per_drug.csv", src_per)
        writer.write_fold_csv(fold.fold_id, "source_metrics_summary.csv", src_sum)
        writer.write_fold_csv(fold.fold_id, "target_metrics_per_drug.csv", tgt_per)
        writer.write_fold_csv(fold.fold_id, "target_metrics_summary.csv", tgt_sum)
        all_src_metrics.append(src_sum.assign(fold=fold.fold_id))
        all_tgt_metrics.append(tgt_sum.assign(fold=fold.fold_id))

        src_latent = encode_latent_dict(
            model, so.x, list(so.sample_ids), config.batch_size, config.device
        )
        tgt_latent = encode_latent_dict(
            model, to.x, list(to.sample_ids), config.batch_size, config.device
        )
        writer.write_fold_pkl(fold.fold_id, "source_latent_representation.pkl", src_latent)
        writer.write_fold_pkl(fold.fold_id, "target_latent_representation.pkl", tgt_latent)

        latent_m = compute_distribution_metrics(src_latent, tgt_latent)
        writer.write_fold_csv(
            fold.fold_id,
            "latent_distribution_metrics.csv",
            pd.DataFrame([latent_m]),
        )
        all_latent_metrics.append(pd.DataFrame([latent_m]).assign(fold=fold.fold_id))

        combined_ids = list(so.sample_ids) + list(to.sample_ids)
        cancer_labels = [cancer_map.get(sid, "Unknown") for sid in combined_ids]
        plot_tsne_domain(
            src_latent,
            tgt_latent,
            config.random_seed,
            writer.fold_dir(fold.fold_id) / "tsne_domain_mixing.png",
        )
        plot_tsne_cancer_type(
            src_latent,
            tgt_latent,
            cancer_labels,
            combined_ids,
            config.random_seed,
            writer.fold_dir(fold.fold_id) / "tsne_cancer_type.png",
        )

        combined_latent = {**src_latent, **tgt_latent}
        kmeans_m = compute_kmeans_cancer_type_metrics(
            combined_latent, cancer_map, config.random_seed
        )
        writer.write_fold_csv(
            fold.fold_id,
            "kmeans_cancer_type_metrics.csv",
            pd.DataFrame([kmeans_m]),
        )
        all_kmeans.append(pd.DataFrame([kmeans_m]).assign(fold=fold.fold_id))

    if all_src_metrics:
        writer.write_summary("metrics_summary.csv", pd.concat(all_src_metrics, ignore_index=True))
    if all_latent_metrics:
        latent_sum = pd.concat(all_latent_metrics, ignore_index=True)
        writer.write_summary("latent_metrics_summary.csv", latent_sum)
    if all_kmeans:
        writer.write_summary(
            "kmeans_cancer_type_summary.csv", pd.concat(all_kmeans, ignore_index=True)
        )


if __name__ == "__main__":
    main()
