# CODE-AE Multilabel Smoke Commands

Use real DAPL paths in Docker (`/workspace/DAPL-master`) with small epoch counts.

## One-shot runner (recommended)

```bash
cd /workspace/SSDA4Drug
PYTHONPATH=Benchmark/CODEAE:$PYTHONPATH \
  python Benchmark/CODEAE/codeae_multilabel/smoke/smoke_runner.py --mode all
```

Modes: `all`, `cls`, `reg`, `pretrain`, `finetune_cls`, `finetune_reg`.

Smoke defaults: `--epochs 1`, `--batch_size 32`, `--max_samples 128`, `--max_drugs 24`, `--n_splits 2`.

## Pretrain

```bash
cd Benchmark/CODEAE
python pretrain_multilabel_hyper_main.py \
  --source_omics_path /path/to/source_omics.csv \
  --target_omics_path /path/to/target_omics.csv \
  --epochs 2 \
  --batch_size 32 \
  --output_dir outputs_codeae_smoke_pretrain \
  --overwrite
```

## Fine-tune (classification)

```bash
python drug_ft_multilabel_hyper_main.py \
  --task_type classification \
  --source_omics_path /path/to/source_omics.csv \
  --target_omics_path /path/to/target_omics.csv \
  --source_response_path /path/to/source_response.csv \
  --target_response_path /path/to/target_response.csv \
  --pretrain_checkpoint outputs_codeae_smoke_pretrain/pretrain/checkpoint.pt \
  --epochs 1 \
  --n_splits 2 \
  --batch_size 32 \
  --output_dir outputs_codeae_smoke_classification \
  --overwrite
```

## Fine-tune (regression)

```bash
python drug_ft_multilabel_hyper_main.py \
  --task_type regression \
  --source_response_col neg_log2_auc \
  --metric macro_mae \
  --pretrain_checkpoint outputs_codeae_smoke_pretrain/pretrain/checkpoint.pt \
  --epochs 1 \
  --n_splits 2 \
  --output_dir outputs_codeae_smoke_regression \
  --overwrite \
  ... other required paths ...
```

## Acceptance

- `drug_list.csv`, `fold_0/best_model.pt`, prediction CSVs, metrics summaries, latent CSVs exist.
- t-SNE may be skipped with a warning on small sample counts.
