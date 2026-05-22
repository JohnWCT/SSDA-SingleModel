# Implementation Decisions

## Environment

- **Project root (Docker):** `/workspace/SSDA4Drug`（與 host 掛載目錄同步）
- **所有驗證均在容器 `SSDA` 內執行**，未修改 host Python 環境
- **新套件名稱：** `ssda_multilabel/`（獨立於 `ssda_latent/` 單藥流程，保持向後相容）

## Architecture

| 決策 | 選擇 | 理由 |
|------|------|------|
| 模型型態 | Multi-output head `[batch, n_drugs]` | 符合 proposal，不需 drug latent |
| Drug 列表 | `sorted(source ∪ target)` | 保留 source-only / target-only 藥物 |
| Target n-shot | sample-drug position-level，每 drug 每 class 至多 `n_shot` | multi-label 語意 |
| Source split | sample-level test + K-fold | 避免同 sample 洩漏到 train/val |
| Regression run | source 用 masked regression；target 仍用 masked BCE | target 標籤恒為 0/1 |
| Latent export | `encoder.ae.encode`（DAE）或 `encoder(x)`（MLP），不用隨機 denoising forward | 可重現 |
| Adaptation | multi-label sigmoid entropy + masked mean | 對應 unlabeled positions |
| Trainer | 自寫 `MultiLabelSSDTrainer`，validation 僅 forward | 舊 `trainer.py` 會在 val 更新參數 |
| t-SNE / FID / MMD | 重用 `ssda_latent.latent_eval` | 避免重複實作 |

## Ambiguity Resolutions

1. **Stratification：** multi-label 完整分層過複雜 → 使用 sample-level pseudo-label（觀測 drug 平均 response）分層；失敗則 random split（記錄於本文件）。
2. **Duplicate response rows：** 預設 `--duplicate_response_strategy error`；測試允許 `first`。
3. **mypy：** `ssda_multilabel.model` 對 legacy `model.py` 使用 `follow_imports = skip`。
4. **PyTorch tuple indexing：** fold indices 以 `list(...)` 索引 tensor，避免 `(i,j,k)` 被當成多維 index。

## Module Map

- `prepare.py` — 資料準備總控
- `omics_io.py` — omics 讀取與 feature 對齊
- `response_matrix.py` / `masks.py` / `split.py` — 矩陣、n-shot、CV
- `model.py` / `losses.py` / `adaptation.py` / `training.py` — 模型與訓練
- `prediction.py` / `metrics.py` / `latent.py` / `export.py` — 輸出
- `experiment_multilabel_ssda.py` — CLI 入口
