# Implementation Report: CODE-AE Multi-label / Multi-drug

## Summary

依 `docs/proposal_codeae.md` 與 `docs/design_codeae.md` 實作獨立套件 `Benchmark/CODEAE/codeae_multilabel/`，將 CODE-AE 從單藥模型擴展為 **單一共享 encoder + 多輸出藥物預測頭**，並保留兩階段流程（pretrain → fine-tune）。

## Implemented Modules

| 模組 | 職責 |
|------|------|
| `contracts.py` | `CodeAEMultilabelConfig`、`OmicsTable`、`DrugIndex`、`ResponseMatrix` 等 |
| `config.py` | CLI、預設訓練參數、`config.json` |
| `seed.py` | 全域隨機種子 |
| `io.py` | CSV/JSON/pickle、輸出目錄覆寫 |
| `validators.py` | Omics / response / matrix 驗證 |
| `data/*` | Omics 對齊、drug union、long→wide+mask、fold split、prepare |
| `model/*` | legacy CODE-AE 適配、`MultiOutputDrugHead`、checkpoint |
| `training/*` | masked BCE/MAE、trainer、early stopping、runners |
| `evaluation/*` | 預測 long table、macro/micro/weighted/**overall** 指標 |
| `export/*` | `ArtifactWriter`、latent、t-SNE |
| `smoke/*` | 小 epoch smoke 指令與 runner |

## Entry Commands

### Pretrain（不讀 response）

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && python Benchmark/CODEAE/pretrain_multilabel_hyper_main.py \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --method code_adv \
  --epochs 1 \
  --output_dir outputs_codeae_multilabel'
```

### Fine-tune（classification）

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && python Benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --task_type classification \
  --source_omics_path ... \
  --target_omics_path ... \
  --source_response_path ... \
  --target_response_path ... \
  --pretrain_checkpoint outputs_codeae_multilabel/pretrain/checkpoint.pt \
  --epochs 1 --n_splits 2 \
  --output_dir outputs_codeae_multilabel'
```

## Design Decisions

1. **Fine-tune 僅 source 標籤進 supervised loss**；target 僅評估（與原 CODE-AE 一致）。
2. **Early stopping**：分類 `macro_auroc`（fallback: aupr / balanced_accuracy / f1）；迴歸 `macro_mae`。
3. **Pretrain** 透過 `train_code_adv.train_code_adv` 訓練 DSNAE，儲存 `shared_encoder` 至 `pretrain/checkpoint.pt`。
4. **Latent / t-SNE** 在載入各 fold `best_model.pt` 後匯出（deterministic encode）。
5. **指標摘要** 含 macro、micro、weighted、**overall**（全樣本-藥物對合併計算）。

## Artifacts

```text
{output_dir}/
  config.json
  drug_list.csv
  pretrain/checkpoint.pt
  fold_0/
    best_model.pt
    source_prediction_results.csv
    target_prediction_results.csv
    source_metrics_per_drug.csv
    target_metrics_per_drug.csv
    source_metrics_summary.csv
    target_metrics_summary.csv
    source_latent.csv
    target_latent.csv
    tsne_domain_mixing.png
    training_log.csv
  fold_summary/metrics_summary_mean_std.csv
```

## Tests (Docker SSDA)

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && pip install -q pytest pytest-cov ruff mypy'
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && python -m pytest tests/test_codeae_multilabel_*.py -q'
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && ruff check Benchmark/CODEAE/codeae_multilabel tests/test_codeae_multilabel_*.py'
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && ruff format --check Benchmark/CODEAE/codeae_multilabel'
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && mypy Benchmark/CODEAE/codeae_multilabel'
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && pytest --cov=codeae_multilabel --cov-fail-under=80 tests/test_codeae_multilabel_*.py -q'
```

**結果（Docker `SSDA`，2026-05-24）：**

- `pytest`：49 passed
- `ruff check` / `ruff format --check`：通過
- `mypy Benchmark/CODEAE/codeae_multilabel`：通過
- `codeae_multilabel` coverage：**約 82%**（≥ 80%）

## Known Limitations

1. Fine-tune 階段未重跑完整 GAN critic 迴圈（與原 `fine_tuning.py` 一致，domain 適應主要在 pretrain）。
2. `codeae_loss_adapter` 預設不加入額外 target supervised loss。
3. 完整真實資料 smoke 需容器內 DAPL 路徑可用。

## Deviations

- 套件路徑為 `Benchmark/CODEAE/`（大寫 B），與文件 `benchmark/CODEAE/` 僅大小寫差異。
- Latent 同時輸出 CSV（並保留 pkl 相容介面於 `write_latent`）。
