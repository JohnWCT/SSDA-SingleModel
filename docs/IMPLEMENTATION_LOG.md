# SSDA4Drug Latent 改良 — 實作紀錄

## 保護清單 baseline（SHA256）

| 檔案 | SHA256 |
|---|---|
| `experiment_shot.py` | `fe67aa5dd4969f3c452cc2b076c397f049a4f2cb7a17f38da5db2a7f5851689f` |
| `trainer.py` | `900132834ad66eb7b3df731ee369d291a86bf94a02233d2a9566d7498aa96c54` |
| `model.py` | `f751b60859fc5c2e8dbdcb7a3af8708e6bddcbbe3c1c1687f16ce9c13305bbcc` |
| `utils.py` | `4c8dde49b91fcf89f41e744fb4bc8fc3ae2f609417065a60abe9c8600845a88b` |

終局複驗（Docker `SSDA`，2026-05-21）：與上表一致。

## 新增檔案

- `ssda_latent/` — 改良版管線套件
- `experiment_shot_ssda.py` — 改良入口（copy-first 自 `experiment_shot.py` 概念重寫為薄 CLI）
- `pyproject.toml` — 依賴與 ruff/mypy/pytest 設定
- `tests/` — 單元測試
- `docs/IMPLEMENTATION_LOG.md` — 本檔

## 品質 gate（Docker `/workspace/SSDA4Drug`）

```bash
GATE_PATHS="ssda_latent tests experiment_shot_ssda.py"
ruff check $GATE_PATHS          # PASS
ruff format --check $GATE_PATHS # PASS
mypy $GATE_PATHS                # PASS（L0 模組 follow_imports=skip）
pytest tests/ -v                # 11 passed
```

## Smoke（Gefitinib，Q5=B）

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && python experiment_shot_ssda.py \
  --drug Gefitinib --n 3 --encoder DAE --epochs 2 \
  --random_seed 42 --source_test_size 0.1 --n_splits 2 \
  --latent_output_dir save/latent_ssda'
```

產物範例：`save/latent_ssda/Gefitinib/seed_42/fold_0/source_latent_representation.pkl`

## 完整 5-fold 跑完（2026-05-21，Docker SSDA）

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && python experiment_shot_ssda.py \
  --drug Gefitinib --n 3 --encoder DAE --epochs 50 \
  --random_seed 42 --source_test_size 0.1 --n_splits 5 \
  --latent_output_dir save/latent_ssda'
```

- 耗時約 53s（GPU）
- `fold_0`…`fold_4` 各 8 個 CSV + pkl/png/pth
- `metrics_summary.csv`：15 列（5 fold × source_val/source_test/target）
- `latent_metrics_summary.csv`：5 列
- 彙總與各 fold 內 CSV 數值交叉驗證通過

## 修復紀錄

1. **`split.assign_target_roles`**：移除錯誤覆寫全體為 `target_test` 的迴圈；改以 `in_target_test` 欄位標記評估集（= 全 target − train labeled）。
2. **`latent_eval._combined_matrix`**：修正 source-only sample id 的 `KeyError`。
3. **mypy**：`[[tool.mypy.overrides]]` 對 `utils`/`trainer`/`model` 設 `follow_imports = "skip"`。

## 文件

- 規格：`docs/proposal.md`
- 設計：`docs/design.md`
- Agent 流程：`docs/prompt.md`
