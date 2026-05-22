# Implementation Decisions

> 完整 As-Built 規格與 `experiment_shot.py` → `experiment_multilabel_ssda.py` 遷移指南已併入：
> - `docs/proposal.md` §26、§22.0
> - `docs/design.md` §14、§10

## Environment

- **Project root (Docker):** `/workspace/SSDA4Drug-main`（與 host 掛載同步；舊名 `SSDA4Drug` 需 symlink）
- **DAPL root (Docker):** `/workspace/DAPL-master`
- **驗證：** 容器 `SSDA` 內執行；`python`（非系統 `python3`）含 pandas/torch
- **套件：** `ssda_multilabel/`（獨立於 `ssda_latent/` 與 `experiment_shot.py`）

## Architecture

| 決策 | 選擇 | 理由 |
|------|------|------|
| 模型型態 | Multi-output head `[batch, n_drugs]` | 符合 proposal，不需 drug latent |
| Drug 列表 | `sorted(source ∪ target)` on `drug_name` | 保留 source-only / target-only 藥物 |
| Target n-shot | sample-drug position-level，每 drug 每 class 至多 `n_shot` | multi-label 語意 |
| Source split | sample-level test + K-fold | 避免同 sample 洩漏到 train/val |
| Regression run | source masked regression；target masked BCE + 分類指標 | target 標籤恒為 0/1 |
| Latent export | 全體 omics；`encode(..., deterministic=True)` | 非 val-only；可重現 |
| 輸出目錄 | `latent_output_dir` 預設 = `output_dir` | 使用者要求扁平路徑 |
| Adaptation | multi-label sigmoid entropy + masked mean | unlabeled positions |
| Trainer | `MultiLabelSSDTrainer` | 舊 `trainer.py` 會在 val 更新參數 |
| Duplicate rows | `--duplicate_response_strategy mean` | PRISM 同藥多 broad_id |
| Cancer type | `cancer_type.py` 依 omics 路徑自動選表 | 不需手動 CLI（pretrain/Winnie） |

## Ambiguity Resolutions

1. **Stratification：** multi-label 完整分層過複雜 → sample-level pseudo-label 分層；失敗則 random split。
2. **Duplicate response rows：** 預設 `mean`（正式跑）；測試可用 `first` / `error`。
3. **mypy：** `ssda_multilabel.model` 對 legacy `model.py` 使用 `follow_imports = skip`。
4. **PyTorch tuple indexing：** fold indices 以 `list(...)` 索引 tensor。
5. **Target eval：** 所有 observed target positions（含 n-shot labeled），非僅 unlabeled。
6. **Source eval：** 僅 `source_test` per fold。

## Module Map

- `prepare.py` — 資料準備總控
- `omics_io.py` / `sample_id.py` — omics 與 ID join
- `response_matrix.py` / `masks.py` / `split.py` — 矩陣、n-shot、CV
- `cancer_type.py` — 自動 metadata
- `model.py` / `losses.py` / `adaptation.py` / `training.py` — 模型與訓練
- `prediction.py` / `metrics.py` / `latent.py` / `latent_eval.py` / `export.py` — 輸出
- `experiment_multilabel_ssda.py` — CLI 入口
- `scripts/patch_dapl_csv_columns.py` — DAPL CSV 欄位標準化

## Smoke

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug-main && PYTHONPATH=. python scripts/patch_dapl_csv_columns.py --dapl-root /workspace/DAPL-master && PYTHONPATH=. python experiment_multilabel_ssda.py --smoke_test classification'
```
