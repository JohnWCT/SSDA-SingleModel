# Implementation Report: Multi-label SSDA

## Completed Modules

| 模組 | 職責 |
|------|------|
| `config.py` | CLI、`MultiLabelConfig`、`config.json` |
| `seed.py` | 全域隨機種子 |
| `schemas.py` | `DrugIndex`、`OmicsTable`、`ResponseMatrix`、`PreparedData` 等 |
| `omics_io.py` | Omics 讀取、共同 feature 對齊 |
| `drug_index.py` | source ∪ target drug 列表 |
| `response_matrix.py` | long → wide + mask |
| `masks.py` | target position-level n-shot |
| `split.py` | source test + K-fold |
| `dataset.py` | PyTorch Dataset |
| `model.py` | DAE/MLP encoder + multi-output head |
| `losses.py` | masked BCE / MSE / MAE / Huber |
| `adaptation.py` | masked entropy / adentropy |
| `training.py` | `MultiLabelSSDTrainer`（val 不更新參數） |
| `prediction.py` | 預測 long table |
| `metrics.py` | per-drug + summary 指標 |
| `latent.py` | sample-level pkl |
| `latent_eval.py` | 委派 `ssda_latent.latent_eval` |
| `export.py` | 輸出目錄與檔案 |
| `reports.py` | n-shot / 缺失報告 |
| `prepare.py` | 資料準備 orchestration |
| `cancer_type.py` | cancer type 對齊 |
| `experiment_multilabel_ssda.py` | 主入口 |

## Tests

- `tests/test_multilabel_modules.py` — 單元測試
- `tests/test_multilabel_toy.py` — 合成資料 end-to-end
- `tests/test_multilabel_extended.py` — metrics / export / losses 覆蓋

## Validation (Docker `SSDA`, `/workspace/SSDA4Drug`)

```bash
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && ruff check ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && ruff format --check ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && mypy ssda_multilabel experiment_multilabel_ssda.py"
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && pytest -q"
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && pytest --cov=ssda_multilabel --cov-fail-under=85 tests/test_multilabel_*.py -q"
```

**結果（實際執行）：**

- `ruff check .` — 通過
- `ruff format --check .` — 通過
- `mypy ssda_multilabel experiment_multilabel_ssda.py` — 通過
- `pytest` — 36 passed（含既有 `ssda_latent` 測試）
- `ssda_multilabel` coverage — **90.6%**（≥ 85%）

## Example Run

```bash
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && python experiment_multilabel_ssda.py \
  --task_type classification \
  --source_omics_path Datasets/processedData/Gefitinib/source_data/source_scaled_tp4k.csv \
  --target_omics_path Datasets/processedData/Gefitinib/target_data/target_scaled_tp4k.csv \
  --source_response_path path/to/source_response.csv \
  --target_response_path path/to/target_response.csv \
  --epochs 50 --random_seed 42"
```

輸出目錄：`save/ssda_multilabel/seed_{seed}/`（見 `docs/prompt.md` 第 10 節）。

## Known Limitations

1. 未整合 legacy `trainer.py` 的 FGM / SCAD 等進階選項。
2. Multi-label stratification 為簡化版 pseudo-label stratify。
3. `latent_eval.py` 為 thin wrapper，邏輯在 `ssda_latent`。
4. 完整真實資料端到端 run 需使用者提供 long-format response CSV（omics 前處理沿用既有 benchmark 流程）。
