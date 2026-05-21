# SSDA4Drug Latent 改良 — 自動化開發提示詞（Agent Prompt）

> **版本**：v1.1-final  
> **權威文件**：[proposal.md](./proposal.md)、[design.md](./design.md)  
> **適用對象**：主 Agent（編排）與子 Agent（模組實作）  
> **使用者已確認**：Q0–Q10（見 §2）；**原始碼不可變**（見 §0 PRIORITY 0-B）

---

## §0 最高權重指令（不可違反）

```
┌──────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 0-A — 全自動開發                                                │
│  • 整個開發過程必須 100% 由 Agent 自主完成，禁止依賴人工參與。              │
│  • 禁止向使用者發送互動式問題、等待確認、或「請您手動執行 X」。              │
│  • 僅在無法自動修復時：寫入 docs/BLOCKED.md 並終止，說明原因與已嘗試步驟。  │
│  • 禁止 git commit（使用者明確要求）。                                      │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 0-B — 原始碼不可變（Copy-First）【使用者強制】                     │
│  • 禁止以任何方式修改 SSDA4Drug-main 既有原始檔案（見 §1.1 保護清單）。      │
│  • 若必須基於原始檔衍生：先 cp 複製到新檔名，僅能編輯複本。                  │
│  • 改良版訓練入口必須為新檔 experiment_shot_ssda.py，不得改 experiment_shot.py│
│  • L0（trainer/model/utils）僅 import 呼叫，不做 format / type ignore 等修補。 │
└──────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────┐
│  PRIORITY 1 — 核心防錯                                                    │
│  • ssda_latent/ 內「每一行業務邏輯」必須有對應單元測試（見 §6）。            │
│  • 新增程式範圍必須 100% 通過 mypy --strict 與 ruff（見 §5，不含保護清單）。 │
│  • 子 Agent 交付前必須自跑 §5 全部 gate；主 Agent 終局再跑全域 gate + smoke。│
└──────────────────────────────────────────────────────────────────────────┘
```

**衝突解決順序**：§0（含 0-B Copy-First）> §2 鎖定規格 > design.md > proposal.md > 本檔其他說明。  
**與舊版 Q2=B 的關係**：原「全專案含 L0 通過 mypy/ruff」讓位於 **0-B**；L0 不修改故不納入 strict gate（見 §5.4）。

---

## §1 任務目標

在 `SSDA4Drug-main/` 實作 **SSDA4Drug Latent 改良版**（**僅新增/複製檔案，不動原始碼**）：

- 單一 `--random_seed`；source 10% independent test + 5-fold stratified CV
- 保留原版 semi-supervised domain adaptation（**import** 既有 `trainer.py`，loss 不變）
- 每 fold 輸出：latent pkl、prediction CSV、t-SNE、FID/MMD/Wasserstein、KMeans metrics
- 新增套件 `ssda_latent/`（見 design.md §2）
- 新增入口 **`experiment_shot_ssda.py`**（改良版 CLI，原版 `experiment_shot.py` 保持不變）
- 更新 `README.md`；維護 `docs/IMPLEMENTATION_LOG.md`

### §1.1 原始碼保護清單（禁止直接修改）

以下檔案視為 **SSDA4Drug 原始碼**，Agent **不得** 對其做任何編輯（含 format、註解、type ignore、刪行、改名）：

| 類別 | 路徑 |
|---|---|
| 訓練入口（原版） | `experiment_shot.py` |
| 延伸實驗入口 | `experiment.py` |
| 訓練核心 | `trainer.py` |
| 模型 | `model.py` |
| 工具 | `utils.py` |
| 對照實驗 | `Benchmark/**` 下所有既有 `.py` |
| 前處理（既有） | `Datasets/preprocess_benchmark.py`、`Datasets/split_data.py`、`Datasets/n_shot.py` 等既有腳本 |

**允許新增、不視為修改原始碼**：

- `ssda_latent/**`
- `tests/**`
- `experiment_shot_ssda.py`（新建或自 `experiment_shot.py` **複製後**再改）
- `pyproject.toml`、`docs/*`（含本 prompt 與 IMPLEMENTATION_LOG）
- `save/**` 執行期輸出

### §1.2 Copy-First 工作流程（必須遵守）

當邏輯需參考或衍生自原始檔時：

```bash
# 正確：先複製，只改複本
cp experiment_shot.py experiment_shot_ssda.py
# 僅編輯 experiment_shot_ssda.py

# 錯誤：直接改原始檔
# vim experiment_shot.py   ← 禁止
```

- **禁止**「暫時改一下原版再改回來」；`git checkout` 還原仍視為違規。
- 從原版**複製邏輯**到 `ssda_latent/` 時：在複製出的模組內重寫（如 `split.py`），**不要** patch 原版。
- `training_adapter.py` 透過 `import trainer`、`import model as m`、`import utils` 呼叫 L0，**不得** fork 修改 `trainer.py` 副本（除非使用者日後明確要求且仍走 copy-first 命名，例如 `trainer_ssda.py`；**本專案預設不需要**）。

---

## §2 已鎖定規格（使用者 Q0 全部同意 + Q1–Q10）

| 類別 | 鎖定值 |
|---|---|
| 套件 | `ssda_latent/`（AD-01） |
| Source split | `source_test_size=0.1`，**stratified**；`StratifiedKFold(n_splits=5)` |
| Target split | 原版 80/20 → train/val 各 n-shot → test（含 val labeled 仍在 test，不修正 leakage） |
| Target few-shot | **固定一次**，所有 fold 共用 |
| Checkpoint | 僅 `model_final.pth`（**last epoch**） |
| DAE latent | `encoder.forward()` → `[0]`（含 denoising） |
| Confidence | softmax `P(class=0)`, `P(class=1)`, `confidence=P(class=1)` |
| Legacy 輸出 | **保留** `save/results/sc/`、`save/sc/all_path/` |
| Cancer type 缺失 | 標記 `Unknown`；KMeans **排除** Unknown |
| FID/MMD/Wasserstein | **本 repo 自實作**（不依賴 DAPL 路徑） |
| 輸出目錄已存在 | **overwrite** |
| 50-seed loop | **移除**，僅單一 seed |
| 品質 gate | mypy `--strict` + ruff（**僅新增程式範圍**，見 §5.4；**不修改 L0**） |
| 原始碼 | **禁止修改**保護清單（§1.1）；必要時 **copy-first**（§1.2） |
| 改良入口 | **`experiment_shot_ssda.py`**（不得改 `experiment_shot.py`） |
| Coverage | **不設**下限；每個公開函式 ≥1 測試（Q3=A） |
| 測試邊界 | 見 §6（Q4 同意） |
| Smoke | **必須**在 Docker `SSDA` 內跑真實 Gefitinib（Q5=B） |
| 子任務粒度 | **一模組一子 Agent**（Q6） |
| 重試 | 子任務失敗自動重試 **最多 3 次**（Q7） |
| Cancer type 檔 | 測試用 **synthetic fixture**（Q8 cancer=A） |
| 資料缺失 | **自動** `preprocess_benchmark.py`（Q8 data=A） |
| torch 型別 | `ignore_missing_imports` + 必要 `# type: ignore` 附理由（Q8 torch=A） |
| 規格衝突 | 以 proposal+design 為準，寫 log，不問人（Q8 conflict=A） |
| 阻塞終止 | 允許 `docs/BLOCKED.md`（Q9） |
| 文件 | 更新 README、維護 IMPLEMENTATION_LOG（Q10） |
| Git | **禁止 commit**（Q10） |

---

## §3 角色定義

### 3.1 主 Agent（Main Agent）

**職責**：

1. 閱讀 `docs/proposal.md`、`docs/design.md`、本 `prompt.md`
2. **環境準備**（§4）：建立 `pyproject.toml`、`ruff`/`mypy`/`pytest` 設定
3. 依 **P0→P4** 向子 Agent **派工**（§7）；每任務對應單一模組 + 測試檔
4. 審核子 Agent 交付：§5 gate 全綠、§6 測試對齊、§1.1 保護清單 **零 diff**
5. 子任務失敗 → **自動重試 ≤3 次**；仍失敗 → 記錄 `IMPLEMENTATION_LOG.md` 並嘗試主 Agent 自行修復或寫 `BLOCKED.md`
6. **終局驗證**：§5 全域 gate + §8 Docker smoke + §9 DoD 清單
7. 更新 `README.md`、`docs/IMPLEMENTATION_LOG.md`

**禁止**：

- 跳過子 Agent 測試要求直接合併大段未測試程式
- **修改 §1.1 保護清單內任一原始檔**（含 `experiment_shot.py`）
- `git commit` / 向使用者提問

**允許**：

- 新增 `experiment_shot_ssda.py`（建議：`cp experiment_shot.py` 後改為薄入口 + 新 CLI）
- 新增 `ssda_latent/`、`tests/`、`pyproject.toml`

---

### 3.2 子 Agent（Sub Agent）

**職責**：

1. 僅實作**被指派的一個模組**（`ssda_latent/<module>.py`）及對應 `tests/test_<module>.py`
2. 同步撰寫業務邏輯與測試（§6）；交付前執行 §5.1 本地 gate
3. 回報：變更檔案列表、通過的 command 輸出摘要、已知限制

**禁止**：

- 修改指派模組以外的檔案（除 `tests/`、`pyproject.toml` 若主 Agent 授權）
- 修改 §1.1 **保護清單**內任何原始檔
- 省略測試、跳過 mypy/ruff

**允許**：

- `import trainer` / `import model` / `import utils`（唯讀使用 L0）

---

### 3.3 協作協議

```text
主 Agent                          子 Agent #N
   │                                  │
   ├─ 派工單 (§7 模板) ──────────────►│
   │                                  ├─ 實作 module + test
   │                                  ├─ 跑 §5.1
   │◄──────── 交付報告 ───────────────┤
   ├─ 驗證 gate / §1.1 保護清單零 diff
   ├─ 失敗？重試 ≤3
   └─ 更新 IMPLEMENTATION_LOG
```

---

## §4 環境準備（主 Agent 首要任務）

### 4.1 工作目錄

```bash
cd /home/wasijk/Drug/SSDA4Drug-main
# 或 Docker 內：cd /workspace/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug
```

### 4.2 依賴安裝（全自動）

主 Agent 必須建立或更新 `pyproject.toml`（含 dev extras）：

```toml
[project]
name = "ssda4drug"
requires-python = ">=3.9"
dependencies = [
  "torch>=2.1",
  "pandas>=2.0",
  "numpy>=1.24",
  "scikit-learn>=1.3",
  "matplotlib>=3.7",
]

[project.optional-dependencies]
dev = ["pytest>=7.4", "mypy>=1.8", "ruff>=0.4", "pandas-stubs", "types-requests"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]

[tool.mypy]
python_version = "3.9"
strict = true
ignore_missing_imports = true

[tool.ruff]
line-length = 100
target-version = "py39"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM", "N"]
```

安裝：

```bash
pip install -e ".[dev]"
```

### 4.3 資料準備（全自動，Q8 data=A）

若缺少 processed 資料：

```bash
python Datasets/preprocess_benchmark.py --drug Gefitinib
```

確認四檔存在：

- `Datasets/processedData/Gefitinib/source_data/source_scaled_tp4k.csv`
- `Datasets/processedData/Gefitinib/source_data/source_meta_data.csv`
- `Datasets/processedData/Gefitinib/target_data/target_scaled_tp4k.csv`
- `Datasets/processedData/Gefitinib/target_data/target_meta_data.csv`

### 4.4 Cancer type 測試資料（Q8 cancer=A）

主 Agent 在 `tests/fixtures/` 建立 synthetic CSV，**不依賴**使用者提供實檔：

- `tests/fixtures/source_cancer_type.csv`（`Sample_ID`, `Cancer_type`）
- `tests/fixtures/target_cancer_type.csv`

sample ID 必須與 `Gefitinib` processed meta index 可對齊（讀取實際 meta 取子集生成）。

---

## §5 品質閘門（100% 通過定義）

### 5.1 每次子 Agent 交付前必跑

```bash
cd SSDA4Drug-main

# 檢查範圍（僅新增程式 — 見 §5.4）
GATE_PATHS="ssda_latent tests experiment_shot_ssda.py"

# 1. Lint
ruff check $GATE_PATHS
ruff format --check $GATE_PATHS

# 2. 型別
mypy ssda_latent tests experiment_shot_ssda.py

# 3. 測試
pytest tests/ -v --tb=short
```

**通過標準**：三項命令 **exit code 0**，無 error；warning 亦需消除或經 `pyproject.toml` 設定豁免（需註明理由）。

### 5.2 主 Agent 終局必跑

§5.1 全部，加上 **原始檔零變更檢查**（§5.5）。

### 5.3 與舊版 Q2=B 的取代說明

使用者曾選 Q2=B（全專案 gate），後新增 **PRIORITY 0-B（原始碼不可變）**。二者衝突時以 **0-B 為準**：

- **不再** 為通過 mypy/ruff 而修改 `trainer.py` / `model.py` / `utils.py` / `experiment_shot.py`
- 品質 gate **僅** 套用於 §5.4 範圍

### 5.4 mypy / ruff 檢查範圍（新增程式）

| 納入 | 不納入 |
|---|---|
| `ssda_latent/**` | `trainer.py`, `model.py`, `utils.py` |
| `tests/**` | `experiment_shot.py`, `experiment.py` |
| `experiment_shot_ssda.py` | `Benchmark/**` |

`pyproject.toml` 建議明確設定：

```toml
[tool.mypy]
exclude = ["Benchmark/", "trainer.py", "model.py", "utils.py", "experiment_shot.py", "experiment.py"]

[tool.ruff]
exclude = ["Benchmark", "trainer.py", "model.py", "utils.py", "experiment_shot.py", "experiment.py"]
```

### 5.5 原始檔零變更檢查（必須通過）

```bash
# 保護清單內檔案不得出現在 git diff / 檔案 hash 變更中
git diff -- experiment_shot.py experiment.py trainer.py model.py utils.py

# 若有輸出 → 違反 PRIORITY 0-B，任務失敗，須還原並改為 copy-first
```

主 Agent 亦可在無 git 時，對保護清單計算實作前後 SHA256 比對（實作開始前先記錄 baseline 至 `IMPLEMENTATION_LOG.md`）。

---

## §6 測試強制規則（核心防錯）

### 6.1 何謂「業務邏輯」（必須有測試）

- `ssda_latent/` 內所有 **`def` 公開函式**（不以下底線開頭）
- 含分支、迴圈、條件判斷的 **私有函式**（需直接測或透過公開 API 完整覆蓋分支）
- `experiment_shot_ssda.py` 中 `build_experiment_config` / `main` 編排邏輯

**規則**：新增或修改 N 行業務邏輯 → 同一 PR/任務內新增/更新測試，使該邏輯被執行到。

### 6.2 可免單元測試

| 項目 | 說明 |
|---|---|
| `model.py`、`trainer.py` 既有邏輯 | L0 **不可修改**；透過 import 覆蓋行為，不寫 L0 單測 |
| `experiment_shot_ssda.py` | 至少 1 個整合測試（mock Runner 或 subprocess smoke） |
| 純常數、`ARTIFACT_NAMES` 僅 dict 定義 | 由使用方測試覆蓋 |
| 空 `__init__.py` re-export | 無邏輯 |

### 6.3 測試檔對應表（必須齊全）

| 模組 | 測試檔 |
|---|---|
| `config.py` | `tests/test_config.py` |
| `seed.py` | `tests/test_seed.py` |
| `paths.py` | `tests/test_paths.py` |
| `data_loading.py` | `tests/test_data_loading.py` |
| `split.py` | `tests/test_split.py` |
| `cancer_type.py` | `tests/test_cancer_type.py` |
| `dataloader_factory.py` | `tests/test_dataloader_factory.py` |
| `training_adapter.py` | `tests/test_training_adapter.py` |
| `latent.py` | `tests/test_latent.py` |
| `prediction.py` | `tests/test_prediction.py` |
| `latent_eval.py` | `tests/test_latent_eval.py` |
| `artifacts.py` | `tests/test_artifacts.py` |
| `export_pipeline.py` | `tests/test_export_pipeline.py` |
| `summary.py` | `tests/test_summary.py` |
| `orchestrator.py` | `tests/test_orchestrator.py` |

### 6.4 測試設計要求

- **禁止** 依賴網路、人工互動
- **優先** 合成小 DataFrame / mock `nn.Module` / `tmp_path` fixture
- `split`：固定 seed 快照比對 split 互斥、stratified 比例
- `latent_eval`：固定矩陣 FID/MMD/Wasserstein 為有限值
- `prediction`：全 0/1 標籤邊界、AUC 行為
- Torch 模組：mock encoder 回傳固定 tensor；**禁止** 假設 CI 有 GPU 才能過單元測試

### 6.5 整合 / Smoke（Q5=B）

單元測試通過後，主 Agent **必須**在 Docker 執行：

```bash
docker exec SSDA bash -lc '
  cd /workspace/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug
  python Datasets/preprocess_benchmark.py --drug Gefitinib
  python experiment_shot_ssda.py \
    --drug Gefitinib --n 3 --encoder DAE --epochs 2 \
    --random_seed 42 --source_test_size 0.1 --n_splits 2 \
    --latent_output_dir save/latent_ssda
'
```

> **注意**：使用 **`experiment_shot_ssda.py`**，**不要** 執行原版 `experiment_shot.py`（後者應保持 50-seed 等原始行為不變）。

**Smoke 斷言（腳本或 pytest integration）**：

- 存在 `save/latent_ssda/Gefitinib/seed_42/fold_0/source_latent_representation.pkl`
- pkl 樣本數 = source 全體樣本數
- 每個 latent 向量長度 = **128**
- `source_prediction_results.csv` 行數 = source 樣本數
- `target_prediction_results.csv` 行數 = target 樣本數

若 Docker / GPU 不可用：寫 `BLOCKED.md`，**不得**改為跳過 smoke 仍宣稱完成（Q5=B）。

---

## §7 派工清單（主 Agent 執行順序）

### 派工單模板（複製給子 Agent）

```markdown
## Sub-Agent Task: <MODULE_ID>

**Scope**: 僅允許修改
- ssda_latent/<module>.py
- tests/test_<module>.py
- tests/fixtures/*（若需要）

**Forbidden**: 修改 §1.1 保護清單內原始檔、其他 ssda_latent 模組

**Spec**: docs/design.md §<section>, docs/proposal.md §<section>

**Deliverables**:
1. 實作完成
2. tests 覆蓋所有公開函式與關鍵分支
3. 貼上 §5.1 三項命令成功輸出（最後 20 行）

**Retry**: 這是第 {k}/3 次嘗試
```

### P0 — 基礎切分與骨架

| 任務 ID | 模組 | 依賴 | 驗收 |
|---|---|---|---|
| P0-1 | `config.py` + test | 無 | `ExperimentConfig` 從 argparse 建構 |
| P0-2 | `seed.py` + test | P0-1 | 同 seed 可重現 |
| P0-3 | `paths.py` + test | P0-1 | `RunLayout` 路徑正確 |
| P0-4 | `data_loading.py` + test | P0-1 | 讀 Gefitinib 四檔為 sample×gene |
| P0-5 | `split.py` + test | P0-4 | stratified test + 5-fold + target n-shot |
| P0-6 | `orchestrator.py` 骨架 + test | P0-1~5 | fold loop 空跑寫 split CSV |

### P1 — 訓練銜接

| 任務 ID | 模組 | 驗收 |
|---|---|---|
| P1-1 | `dataloader_factory.py` + test | 產出與原版結構相同的三組 DataLoader |
| P1-2 | `training_adapter.py` + test | mock 可呼叫 `train_semi_*`；寫 legacy + model_final.pth |
| P1-3 | `experiment_shot_ssda.py` | `cp experiment_shot.py` 後改薄入口；**原版不動** |

### P2 — Latent 與 Prediction 匯出

| 任務 ID | 模組 | 驗收 |
|---|---|---|
| P2-1 | `latent.py` + test | pkl dict 格式、dim=128 |
| P2-2 | `prediction.py` + test | softmax 欄位、metrics |
| P2-3 | `artifacts.py` + test | 寫檔不破壞、overwrite |
| P2-4 | `export_pipeline.py` + test | 串接 latent+pred+metrics 檔名 |

### P3 — Cancer type 與 Latent 評估

| 任務 ID | 模組 | 驗收 |
|---|---|---|
| P3-1 | `cancer_type.py` + test | fixture 對齊、Unknown 策略 |
| P3-2 | `latent_eval.py` + test | FID/MMD/WS、t-SNE 不 crash、KMeans 排除 Unknown |

### P4 — 彙整與終局

| 任務 ID | 工作 | 驗收 |
|---|---|---|
| P4-1 | `summary.py` + test | 5-fold summary CSV |
| P4-2 | 主 Agent | orchestrator 完整串接 export + summary |
| P4-3 | 主 Agent | README 更新、IMPLEMENTATION_LOG 完結 |
| P4-4 | 主 Agent | §8 Docker smoke 通過 |

---

## §8 輸出目錄契約（必須實作）

根路徑：`save/latent_ssda/{drug}/seed_{seed}/`

每 fold 至少包含（見 proposal §11）：

- `model_final.pth`
- `source_latent_representation.pkl`、`target_latent_representation.pkl`
- `source_latent_metadata.csv`、`target_latent_metadata.csv`
- `source_prediction_results.csv`、`target_prediction_results.csv`
- `tsne_domain_mixing.png`
- `latent_distribution_metrics.csv`
- `source_val_metrics.csv`、`source_test_metrics.csv`、`target_prediction_metrics.csv`
- 若提供 cancer type：`tsne_cancer_type.png`、`kmeans_cancer_type_metrics.csv`

Seed 級：

- `config.json`、`source_split.csv`、`target_fewshot_split.csv`
- `metrics_summary.csv`、`latent_metrics_summary.csv`（及 kmeans summary 若適用）

---

## §9 完成定義（Definition of Done）

主 Agent 僅在以下 **全部** 滿足時宣告完成：

- [ ] `ssda_latent/` 模組與 `tests/` 對應表 §6.3 **齊全**
- [ ] §5.1 gate 於 §5.4 範圍內 **全過**
- [ ] §5.5 保護清單（§1.1）**零 diff**
- [ ] `experiment_shot_ssda.py` 可執行：單 seed + 5-fold + `ExperimentRunner`
- [ ] `experiment_shot.py` **內容與實作前 baseline 一致**（未改動）
- [ ] §8 Docker smoke **通過**（Q5=B）
- [ ] `README.md` 已更新執行方式與新 CLI
- [ ] `docs/IMPLEMENTATION_LOG.md` 記錄各階段與 gate 結果
- [ ] **未執行** `git commit`
- [ ] 無未解決的 `BLOCKED.md`（或 BLOCKED 已解消）

---

## §10 文件維護

### 10.1 `docs/IMPLEMENTATION_LOG.md`（主 Agent 持續更新）

每完成子任務追加：

```markdown
## [YYYY-MM-DD HH:MM] P0-5 split.py
- Agent: sub-agent / retry 2
- Files: ssda_latent/split.py, tests/test_split.py
- Gates: ruff ✓ mypy ✓ pytest 42 passed
- Notes: ...
```

### 10.2 `docs/BLOCKED.md`（僅阻塞時建立）

必含：阻塞原因、重試 3 次摘要、建議後續（給人閱讀但**不等待**人修復）。

### 10.3 `README.md` 更新要點

- 新增 `ssda_latent` 流程說明
- **改良版**執行：`python experiment_shot_ssda.py ...`（新 CLI）
- **原版**仍為：`python experiment_shot.py ...`（行為不變，文件註明兩者並存）
- Docker smoke 命令（§6.5，使用 `experiment_shot_ssda.py`）

---

## §11 子 Agent 實作要點速查

### 11.1 `split.py`

- `train_test_split(..., stratify=y['response'])` for source test
- `StratifiedKFold(5, shuffle=True, random_state=seed)`
- Target：將 `experiment_shot.py` L90–118 **邏輯抄寫**至 `assign_target_roles()`（讀原版、寫 `ssda_latent/split.py`，**不改**原版檔案）
- 輸出 `SplitManifest`（design §4.3）

### 11.2 `training_adapter.py`

- 將 `experiment_shot.py` 內模型建構邏輯**抄寫**至本模組（或 `experiment_shot_ssda.py`），`import trainer` 呼叫 `train_semi_dae/mlp`
- 回傳 **last epoch** 模型；`torch.save` → `model_final.pth`
- 同時寫 legacy `save/results/sc/`、`save/sc/all_path/`（AD-05）

### 11.3 `latent.py`

```python
def get_encoder_latent(encoder, x):
    out = encoder(x)
    return out[0] if isinstance(out, tuple) else out
```

### 11.4 `prediction.py`

- 使用 `model.Test_Double_Model`
- `confidence = probs[:, 1]`

### 11.5 `latent_eval.py`

- 自實作 FID/MMD/Wasserstein（numpy）
- KMeans：`n_clusters = len(unique cancer types)`，**排除** `Unknown`
- t-SNE：`random_state=seed`；cancer 圖僅當 registry.is_available

---

## §12 主 Agent 啟動檢查清單

開始派工前，主 Agent 自行確認：

1. [ ] 已讀 `proposal.md`、`design.md`、本 `prompt.md`
2. [ ] 已記錄 §1.1 保護清單檔案 baseline（hash 或 `git diff` 為空）
3. [ ] `pyproject.toml` + dev 依賴已安裝；§5.4 exclude 已設定
4. [ ] `tests/fixtures/` cancer type 已建立
5. [ ] `preprocess_benchmark.py --drug Gefitinib` 已執行（**不修改**該腳本本體）
6. [ ] `IMPLEMENTATION_LOG.md` 已建立
7. [ ] 已存在 `experiment_shot_ssda.py`（copy-first）或列入 P1-3
8. [ ] Docker 容器 `SSDA` 可 `docker exec`（smoke 用）

---

## §13 給下一個 Agent 的單行啟動指令

```
你是主 Agent。請嚴格遵守 SSDA4Drug-main/docs/prompt.md（§0 含 0-B 原始碼不可變），
依 docs/proposal.md 與 docs/design.md 全自動完成 ssda_latent 實作。
禁止修改 §1.1 保護清單內原始檔；需衍生時先 cp 再改複本（experiment_shot_ssda.py）。
禁止人工參與、禁止 git commit。gate 僅跑 §5.4 新增程式範圍。
終局 Docker smoke 使用 experiment_shot_ssda.py。開始 §4 環境準備，然後 P0→P4。
```

---

*本提示詞由使用者 Q0–Q10 確認後定稿；v1.1 新增 PRIORITY 0-B（原始碼不可變、Copy-First）。任何與本檔衝突的口頭指示無效。*
