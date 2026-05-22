# Semi-supervised deep transfer learning accurately predicting single-cell drug responses

## License and attribution

This repository is distributed under the [MIT License](LICENSE). See [NOTICE](NOTICE) for
upstream attribution to the original [SSDA4Drug](https://github.com/hliulab/SSDA4Drug) project
(hliulab).

**Not tracked in Git** (regenerate locally): `Datasets/rawData/`, `Datasets/processedData/`,
and `save/`. See [資料前處理](#資料前處理建議使用-python-腳本) below.

## Introduction

**SSDA4Drug** is a deep learning framework that leverages semi-supervised domain adaptation techniques to translate drug response knowledge from source domain to the target domain. Also, adversarial training is introduced to reduce the model's sensitivity to slight input variations, thereby enhance its generalizability. SSDA4Drug not only reduces the distribution gap between domains but also learns discriminative features specific to the classification task.

## Model Architecture

![](framework.jpg)

## Requirements

The deep learning models were trained using 2*NVIDIA GeForce RTX 4090 on Linux system.

+ Python 3.9
+ PyTorch 2.1.0
+ Pandas 2.0.3
+ Numpy 1.24.3
+ Scikit-learn 1.3.1
+ Scanpy 1.9.5
+ Shap 0.45.1
+ Captum 0.7.0

## Installation Guide

```
1.Clone this repository
```

`git clone https://github.com/JohnWCT/SSDA-SingleModel.git`

`cd SSDA-SingleModel/`

```
2.Set up the Python environment (Docker recommended)
```

`docker build -f dockerfile.ssda -t ssda4drug:cuda121 .`

`docker run --gpus all -itd --name SSDA -v "$PWD":/workspace/SSDA4Drug ssda4drug:cuda121`

## Usage

### 資料前處理（建議使用 Python 腳本）

本專案已將 `Datasets/` 下的 Jupyter notebook 邏輯改寫為可重跑、可參數化的 Python 腳本。**執行環境建議在 Docker container 內完成**（例如已掛載專案目錄的 `SSDA` 容器），勿在本機額外改動 Python 環境。

#### 主模型最小依賴（跑 `experiment_shot.py` 只需這一步）

README 原文列出 `benchmark.ipynb` → `split_data.ipynb` → `n_shot.ipynb` → `experiment_shot.py`，但 **現行 `experiment_shot.py` 只讀取 benchmark 產出的四個 CSV**，其餘 split / n-shot 產物**不會**被主訓練腳本讀取。主模型會在程式內自行完成 source/target 的 train/val split、n-shot 抽樣、unlabeled 切分與 test set 建構。

**Git 倉庫不含** `Datasets/rawData/` 與 `Datasets/processedData/`（見 `.gitignore`）。克隆後請自行準備
raw 資料並執行下方前處理腳本重建 processed CSV。

必要輸入檔（由 benchmark 前處理產生）：

- `Datasets/processedData/<drug>/source_data/source_scaled_tp4k.csv`
- `Datasets/processedData/<drug>/source_data/source_meta_data.csv`
- `Datasets/processedData/<drug>/target_data/target_scaled_tp4k.csv`
- `Datasets/processedData/<drug>/target_data/target_meta_data.csv`

最小流程：

```
rawData/<drug>/  （或 rawData/<drug>.zip，腳本會自動解壓）
       ↓
preprocess_benchmark.py
       ↓
experiment_shot.py
```

在 Docker 內執行（專案根目錄，依掛載路徑擇一 `cd`）：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; python Datasets/preprocess_benchmark.py --drug Gefitinib'
```

`preprocess_benchmark.py` 參數說明：`python Datasets/preprocess_benchmark.py --help`（`--gene-set-tag`、`--tsv-suffix` 等見文末 Benchmark 章節「資料前處理」）。

#### 訓練主模型（原版）

前處理完成後，在 Docker 內執行：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; python experiment_shot.py --shot_method "3-shot" --drug "Gefitinib" --encoder_h_dims "512,256" --bottleneck 128 --predictor_h_dims "64,32" --epochs 50 --lr 0.001 --batch_size 32 --dropout 0.3'
```

> **注意**：`experiment_shot.py` 外層含 `for i in range(50)` 重複實驗迴圈，完整跑完耗時較長；測試時請勿用 `head` 等指令截斷 stdout，以免行程被 SIGPIPE 中斷。

#### 訓練改良版（Latent + 5-fold CV，`experiment_shot_ssda.py`）

**不修改** `experiment_shot.py` / `trainer.py` / `model.py` / `utils.py`；改良邏輯在 `ssda_latent/` 與新入口 `experiment_shot_ssda.py`。

| 項目 | 原版 `experiment_shot.py` | 改良 `experiment_shot_ssda.py` |
|---|---|---|
| 重複 seed | 50 次迴圈 | 單一 `--random_seed` |
| Source 切分 | 80/20 train/val | 10% stratified test + `--n_splits` fold CV |
| 輸出 | `save/results/sc/` 等 | 預設寫入 `outputs/`（可用 `--output_dir` 指定根目錄） |

預設目錄結構（`--output_dir outputs`）：

- `outputs/latent_ssda/<drug>/seed_<seed>/` — latent pkl、預測 CSV、t-SNE、FID/MMD 等
- `outputs/legacy/results/sc/` — 訓練 AUC 日誌（與原版相容路徑風格）
- `outputs/legacy/sc/all_path/` — legacy checkpoint（若 `--save_legacy_outputs`）

Docker 內快速驗證（2 epoch）：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug; python experiment_shot_ssda.py \
  --drug Gefitinib --n 3 --encoder DAE --epochs 2 \
  --random_seed 42 --source_test_size 0.1 --n_splits 5 \
  --output_dir outputs'
```

自訂輸出根目錄範例：`--output_dir /workspace/SSDA4Drug/runs/exp01`；若只要改 latent 路徑可加 `--latent_output_dir ...`。

開發依賴（僅容器內安裝即可）：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug; pip install pytest mypy ruff scipy pandas-stubs -q'
```

詳見 `docs/design.md`、`docs/IMPLEMENTATION_LOG.md`。

#### 多藥物 Multi-label SSDA（`experiment_multilabel_ssda.py`）

一次輸出所有藥物反應向量 `[batch_size, n_drugs]`；輸入為 **omics CSV + long-format response CSV**（非預先 latent）。

必要參數：`--task_type classification|regression`、`--source_omics_path`、`--target_omics_path`、`--source_response_path`、`--target_response_path`。

Docker 範例（請替換為實際 response 路徑）：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && python experiment_multilabel_ssda.py \
  --task_type classification \
  --source_omics_path Datasets/processedData/Gefitinib/source_data/source_scaled_tp4k.csv \
  --target_omics_path Datasets/processedData/Gefitinib/target_data/target_scaled_tp4k.csv \
  --source_response_path path/to/source_response_long.csv \
  --target_response_path path/to/target_response_long.csv \
  --random_seed 42 --n_splits 5 --n_shot 3 --epochs 50 \
  --latent_output_dir save/ssda_multilabel'
```

輸出：`save/ssda_multilabel/seed_<seed>/`（`drug_list.csv`、mask 矩陣、各 fold 預測／latent／metrics）。設計說明見 `docs/proposal.md`、`docs/design.md`；實作報告見 `docs/implementation_report.md`。

測試（容器內）：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug && pytest tests/test_multilabel_*.py -q'
```

#### 可廢棄的舊流程（重複、僅保留參考）

以下 **Jupyter notebook 與「先跑 split / n-shot 再跑主模型」的舊 README 順序，對 SSDA4Drug 主模型而言已屬重複**，建議改用上列 Python 腳本；notebook 可保留作對照，但**不必再手動執行**：

| 舊案（可廢棄／僅參考） | 取代方案 | 對主模型是否必要 |
|---|---|---|
| `Datasets/benchmark.ipynb` | `Datasets/preprocess_benchmark.py` | **必要**（邏輯相同，改腳本） |
| `Datasets/split_data.ipynb` | `Datasets/split_data.py` | 否（固定 CSV 匯出／人工檢查；**非**主模型或 `Benchmark/` 執行所需） |
| `Datasets/n-shot.ipynb` | `Datasets/n_shot.py` | 否（同上；須在 `split_data.py` 之後） |
| README 四步串接後再跑 `experiment_shot.py` | 僅 benchmark → `experiment_shot.py` | 主模型只需 benchmark |

若使用自訂資料集（非 benchmark），請將資料放入 `Datasets/` 對應目錄後執行：

```bash
python experiment_shot.py --shot_method "3-shot"
```

#### 延伸實驗（非 benchmark 藥物資料）

使用已處理好的細胞株資料（`Extended/`），`cell_name` 可選 `PC9`、`PCa`、`MCF7`、`OSCC`：

```bash
python experiment.py --cell <cell_name>
```

## Directory structure

+ `experiment_shot.py`: 原版訓練入口（50-seed 迴圈）。
+ `experiment_shot_ssda.py`: 改良版入口（單 seed、source CV、latent 匯出）；呼叫 `ssda_latent/`。
+ `experiment_multilabel_ssda.py`: **多藥物 multi-label SSDA** 入口；呼叫 `ssda_multilabel/`（見下方章節）。
+ `ssda_latent/`: 資料切分、訓練適配、latent/預測/評估匯出管線。
+ `ssda_multilabel/`: 多藥物 long table → wide matrix、mask loss、position-level n-shot、multi-output head。
+ `trainer.py`: The training loop, the hyper-parameters, and the evaluation.
+ `utils.py`: Contains auxiliary, general-purpose, or custom functions, which can be called and used in other parts of the project.
+ `model.py`: Model storage directory.
+ `Benchmark`: Competing / baseline methods for benchmark drugs（執行說明見文末 **Benchmark 對照實驗** 章節）:
  - `baseline`: Simple baseline classifier.
  - `CODEAE`: Two-stage CODE-AE (pretrain + drug fine-tune).
  - `SCAD`: SCAD domain adaptation method.
  - `scDeal`: Bulk pretrain + sc transfer (`bulkmodel.py` → `scmodel.py`); `scmodel_noPre.py` without bulk pretrain.
+ `Extended`:
  - `MCF7`: The processed dataset of MCF7 cell lines.
  - `OSCC`: The processed  dataset of OSCC cell lines.
  - `PC9`: The processed  dataset of PC9 cell lines.
  - `PCa`: Te processed  dataset of PCa cell lines.
+ `Datasets`:
  * `rawData`: The benchmark datasets used for performance comparison (`<drug>/` or `<drug>.zip`).
  * `processedData`: Processed outputs from preprocessing scripts (created on first run).
  * **Preprocessing scripts**:
    - `preprocess_benchmark.py`: Raw TSV → scaled/meta CSVs（**主模型與 `Benchmark/` 對照方法皆需要**）。
    - `split_data.py` / `n_shot.py`: 固定 split／n-shot CSV（**非**執行訓練腳本所需；說明見 **Benchmark 對照實驗** 章節）。
  * **Legacy notebooks (deprecated for routine use; kept for reference only)**:
    - `benchmark.ipynb`: Superseded by `preprocess_benchmark.py`.
    - `split_data.ipynb`: Superseded by `split_data.py` (not required by main model).
    - `n-shot.ipynb`: Superseded by `n_shot.py` (not required by main model).

---

## Benchmark 對照實驗（Competing Methods）

本章說明 `Benchmark/` 目錄下各**對照方法**的用途、資料依賴與執行方式。這些程式與根目錄的 **`experiment_shot.py`（SSDA4Drug 主模型）分開**，用於論文或消融實驗中的方法比較。

> **路徑注意**：目錄名為 `Benchmark/`（大寫 B）。以下指令假設在 **Docker container** 內、專案已掛載至 `/workspace/SSDA4Drug`（或 `.../SSDA4Drug-main`），並先 `cd` 到對應子目錄再執行。

### 資料前處理

#### 執行對照方法：只需 benchmark 前處理

`Benchmark/` 下 **baseline、CODE-AE、SCAD、scDeal** 等腳本，與主模型 `experiment_shot.py` 相同，僅讀取 **`preprocess_benchmark.py` 產出的四個檔**：

- `Datasets/processedData/<drug>/source_data/source_scaled_tp4k.csv`
- `Datasets/processedData/<drug>/source_data/source_meta_data.csv`
- `Datasets/processedData/<drug>/target_data/target_scaled_tp4k.csv`
- `Datasets/processedData/<drug>/target_data/target_meta_data.csv`

讀檔後，各對照程式在**內部**以 `train_test_split`（及部分方法的重複實驗迴圈）自行切分 train/val/test，**不會**讀取 `split_data.py` / `n_shot.py` 寫入的 `target_data/tp4k/` 或 `3-shot/tp4k/` 路徑。

對照實驗前處理（以 `Gefitinib` 為例）：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; python Datasets/preprocess_benchmark.py --drug Gefitinib'
```

流程：

```
rawData/<drug>/  （或 <drug>.zip，腳本會自動解壓）
       ↓
preprocess_benchmark.py
       ↓
Benchmark/<method>/*.py
```

**命名說明**（`preprocess_benchmark.py`）：

- `--gene-set-tag tp4k`：**基因集／特徵選擇方案**標籤（輸出檔名與子目錄用），**不是**單一基因名稱。
- `--tsv-suffix _tp4k`：raw TSV 檔名片段，與對照腳本的 `--gene _tp4k` 一致。

#### 可選：`split_data.py` / `n_shot.py`（非執行對照方法所需）

這兩支腳本對應舊 notebook，產出**固定** train/val 與 n-shot CSV，用途包括：

- 與歷史 notebook 輸出目錄對照、除錯；
- 人工檢查資料切分；
- 本 repo **以外** 若另有流程依賴固定 CSV 時使用。

**跑 `Benchmark/` 內任一對照方法時不必執行。** 若仍要產出固定檔案：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; python Datasets/split_data.py --drug Gefitinib && python Datasets/n_shot.py --drug Gefitinib --n 3'
```

（須先跑 `split_data.py`，再跑 `n_shot.py`。）

### 與主模型的資料差異

| 項目 | SSDA4Drug 主模型 (`experiment_shot.py`) | `Benchmark/` 對照方法 |
|---|---|---|
| 必要前處理 | `preprocess_benchmark.py` | **相同** |
| 是否讀取 `split_data` / `n_shot` 產物 | 否 | **否** |
| split / n-shot 切分 | 程式內動態完成 | 程式內 `train_test_split` 等動態完成 |
| 工作目錄 | 專案根目錄 | `Benchmark/<method>/` |
| 讀取資料路徑 | `./Datasets/processedData/...` | `../../Datasets/processedData/...` |

各對照腳本共通參數（多數方法支援）：

- `--drug`：藥物資料夾名稱，例如 `Gefitinib`。
- `--gene`：特徵集後綴，預設 `_tp4k`（對應 `source_scaled_tp4k.csv` 等檔名；**非**單一基因名稱）。

---

### `Benchmark/baseline/` — 簡單 baseline

| 項目 | 說明 |
|---|---|
| 入口 | `baseline.py` |
| 用途 | 在 benchmark 資料上訓練較簡單的 baseline 分類模型，作為效能下界參考。 |
| 主要讀檔 | `../../Datasets/processedData/<drug>/source_data/source_scaled<gene>.csv`、`source_meta_data.csv`；target 端 scaled + meta。 |
| 訓練邏輯 | 與主流程類似：讀取 processed 資料後做 train/val split、加權採樣等（見腳本內 `train_test_split`）。 |

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; cd Benchmark/baseline && python baseline.py --drug Gefitinib'
```

---

### `Benchmark/CODEAE/` — CODE-AE（兩階段）

| 項目 | 說明 |
|---|---|
| 入口 | **①** `pretrain_hyper_main.py` → **②** `drug_ft_hyper_main.py` |
| 用途 | 先在 source（bulk）上預訓練編碼器／對抗式表示學習，再針對目標藥物做 fine-tuning。 |
| 主要讀檔 | `../../Datasets/processedData/<drug>/` 下 source scaled + meta；fine-tune 階段另讀 target scaled + meta。 |
| 相關模組 | `train_code_adv.py`、`train_code_base.py`、`encoder_decoder.py`、`dsn_ae.py`、`fine_tuning.py`、`data_process.py` 等。 |

**必須依序執行**（② 依賴 ① 產出的預訓練權重）：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; cd Benchmark/CODEAE && python pretrain_hyper_main.py --drug Gefitinib && python drug_ft_hyper_main.py --drug Gefitinib'
```

---

### `Benchmark/SCAD/` — SCAD

| 項目 | 說明 |
|---|---|
| 入口 | `main.py` |
| 用途 | SCAD 對照方法：含生成器／映射模組的 domain adaptation 流程（見 `modules.py`）。 |
| 主要讀檔 | 與 baseline 相同路徑的 source/target scaled + meta。 |
| 訓練邏輯 | `train_test_split` 後以 PyTorch 訓練；含 `predict_label` 等評估函式。 |

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; cd Benchmark/SCAD && python main.py --drug Gefitinib'
```

---

### `Benchmark/scDeal/` — scDeal（含 bulk 預訓練）

| 項目 | 說明 |
|---|---|
| 入口 | **①** `bulkmodel.py` → **②** `scmodel.py` |
| 用途 | 先在 bulk（source）上預訓練 DAE／預測器，再將表示遷移至單細胞（target）並搭配 MMD（`DaNN/`）等。 |
| 主要讀檔 | `../../Datasets/processedData/<drug>/` 的 source/target scaled + meta。 |
| 相關模組 | `models.py`、`trainers.py`、`DaNN/mmd.py`、`DaNN/loss.py`。 |
| 模型輸出 | 預設寫入各腳本內 `save/bulk_pre_my/`、`save/sc_pre_my/` 等（相對 `Benchmark/scDeal/`）。 |

**必須依序執行**：

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; cd Benchmark/scDeal && python bulkmodel.py --drug Gefitinib && python scmodel.py --drug Gefitinib'
```

> `scmodel.py` 預設可能未在命令列帶 `--drug`，請依 `--help` 確認；上列命令已明確指定 `Gefitinib`。

---

### `Benchmark/scDeal/` — scDeal（無 bulk 預訓練）

| 項目 | 說明 |
|---|---|
| 入口 | `scmodel_noPre.py` |
| 用途 | 跳過 bulk 預訓練階段，直接在 target 端訓練 scDeal 變體，用於消融「是否需要 bulk pretrain」。 |
| 主要讀檔 | 同上 `processedData` 路徑。 |

```bash
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug/SSDA4Drug-main 2>/dev/null || cd /workspace/SSDA4Drug; cd Benchmark/scDeal && python scmodel_noPre.py --drug Gefitinib'
```

---

### `Benchmark/` 目錄結構速查

```
Benchmark/
├── baseline/
│   └── baseline.py          # 簡單 baseline
├── CODEAE/
│   ├── pretrain_hyper_main.py   # ① bulk 預訓練
│   ├── drug_ft_hyper_main.py    # ② 藥物 fine-tune
│   └── …                        # 編碼器、對抗訓練、評估工具
├── SCAD/
│   ├── main.py              # SCAD 主程式
│   └── modules.py           # 網路模組
└── scDeal/
    ├── bulkmodel.py         # ① bulk 預訓練
    ├── scmodel.py           # ② sc 遷移（有 pretrain）
    ├── scmodel_noPre.py     # 無 bulk pretrain 變體
    ├── models.py / trainers.py
    └── DaNN/                # MMD 等 domain 對齊
```

### 可廢棄的舊案（前處理與 Benchmark）

| 舊案（可廢棄／僅參考） | 取代方案 | 跑對照方法是否必要 |
|---|---|---|
| `benchmark.ipynb` | `preprocess_benchmark.py` | **必要** |
| `split_data.ipynb` | `split_data.py` | **否** |
| `n-shot.ipynb` | `n_shot.py` | **否** |
| 舊 README「先 split → n-shot → 再跑 baseline」 | 僅 `preprocess_benchmark.py` → `Benchmark/...` | **否**（對照腳本不讀固定 CSV） |
