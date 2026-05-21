# SSDA4Drug 改良版需求文件（Latent / 5-Fold / Prediction / Cancer Type 評估）

> **文件狀態**：初版（程式碼審閱完成；第 17 節所列項目尚待使用者確認後，方可進入實作）
>
> **範圍**：保留原版 SSDA4Drug semi-supervised domain adaptation 架構與訓練邏輯，新增單一 seed、source 5-fold、latent 輸出、prediction 結果表、cancer type 評估與 latent 分布 metrics。**不修改模型架構、不改 DataLoader 相依性、不接 DAPL `C_prototypical.py`。**

---

## 1. 專案背景與修改目的

### 1.1 背景

SSDA4Drug 使用 source domain（bulk / cell line）labeled data 與少量 target domain（single-cell）labeled data 進行 semi-supervised domain adaptation，預測 target 樣本的 drug response（二分類）。原版 `experiment_shot.py` 以 50 個 random seed 重複實驗，輸出 AUC/AUPR 與模型 checkpoint，但**未輸出**：

- domain-adapted encoder bottleneck latent（pkl）
- 全樣本 source / target prediction 結果表
- t-SNE domain mixing / cancer type 視覺化
- FID / MMD / Wasserstein latent 分布 metrics
- KMeans cancer type clustering metrics
- source independent test set 與 5-fold cross-validation

### 1.2 修改目的

在**不破壞原版 SSDA4Drug 核心訓練邏輯**的前提下，建立可重現、可追蹤的實驗流程：

1. 單一 `--random_seed` 控制所有隨機性
2. source 先切 independent test，再 5-fold CV
3. 每 fold 完成 semi-supervised adaptation 後輸出 final encoder latent
4. 輸出 source / target 全樣本 prediction 與 latent 評估
5. 透過額外 cancer type metadata 支援 cancer type t-SNE 與 KMeans metrics

### 1.3 參考設計

Latent pkl 格式與 cancer type 輸入方式參考 `DAPL-master/A_pretrain.py`（`{sample_id: latent_vector}` dict），但**不接** `C_prototypical.py`。

---

## 2. 原版 SSDA4Drug 資料使用方式說明

> 以下為閱讀 `experiment_shot.py`、`trainer.py`、`model.py`、`utils.py` 後之**程式碼事實**，非推測。

### 2.1 資料檔案路徑與格式

| 項目 | 路徑 | 格式 |
|---|---|---|
| Source expression | `Datasets/processedData/{drug}/source_data/source_scaled{gene}.csv` | CSV；**gene × sample**（gene 為 row index，sample 為 column） |
| Source metadata | `Datasets/processedData/{drug}/source_data/source_meta_data.csv` | CSV；index = sample ID；欄位：`response`, `logIC50` |
| Target expression | `Datasets/processedData/{drug}/target_data/target_scaled{gene}.csv` | 同上 |
| Target metadata | `Datasets/processedData/{drug}/target_data/target_meta_data.csv` | CSV；index = sample ID；欄位：`response` |

- 預設 `--gene _tp4k` → 讀取 `source_scaled_tp4k.csv` 等
- 前處理由 `Datasets/preprocess_benchmark.py` 自 raw TSV 產生上述四檔
- **Sample ID 位置**：expression matrix 的 **column header**（讀入後 `.T` 轉置，sample 成為 index / row）
- **Response label 欄位名稱**：`response`（二分類 0/1）

**範例（Gefitinib）**：

- Source sample ID：數字（如 `683665`）
- Target sample ID：字串（如 `c4653`）

### 2.2 Source data 使用方式

```python
# experiment_shot.py L58-86
x_source = pd.read_csv(..., index_col=0).T          # sample × gene
y_source = pd.read_csv("source_meta_data.csv", index_col=0)

x_train_source, x_val_source, y_train_source, y_val_source = train_test_split(
    x_source, y_source, test_size=0.2, random_state=i)  # 80/20，未 stratify

# source train 使用 WeightedRandomSampler（依 response 類別加權）
source_train = create_dataset(x_train_source, y_train_source, sampler=sampler)
source_valid = create_dataset(x_val_source, y_val_source)
```

| 問題 | 原版答案 |
|---|---|
| 是否 binary classification | 是（`response` 0/1） |
| train/val 比例 | 80% / 20% |
| 是否 stratified | **否**（`train_test_split` 未指定 `stratify`） |
| validation 用途 | 訓練 epoch 中計算 source AUC/AUPR；**不保存 best model** |

### 2.3 Target data 使用方式

```python
# experiment_shot.py L90-143
x_target = pd.read_csv(...).T
y_target = pd.read_csv("target_meta_data.csv", index_col=0)

# Step 1: 全 target 先 80/20 split（random_state=i）
x_train_target, x_val_target, y_train_target, y_val_target = train_test_split(
    x_target, y_target, test_size=0.2, random_state=i)

# Step 2: 從 train 部分每 class 抽 args.n 個 → target labeled (train)
random.seed(i)
sample_0_train = random.sample(y_train_target[y_train_target["response"]==0].index.tolist(), args.n)
sample_1_train = random.sample(y_train_target[y_train_target["response"]==1].index.tolist(), args.n)
# 剩餘 train → target unlabeled (train)

# Step 3: 從 val 部分同樣每 class 抽 args.n 個 → target labeled (val)
# 剩餘 val → target unlabeled (val)

# Step 4: test set = 全 target 去掉 train labeled（val labeled 仍留在 test 中）
y_test_target = y_target.drop(y_train_labeled.index, axis=0)
```

| 問題 | 原版答案 |
|---|---|
| few-shot 抽樣 | 每 class 各抽 `args.n` 個（**train 與 val 各抽一次**） |
| target unlabeled 定義 | 各 split 內去掉 labeled 後的剩餘樣本 |
| target label 在 adaptation loss 中 | **未使用**（`xt_unl, _ = iter_target_unl.__next__()`，label 被丟棄） |
| target test 建立 | 全 target − train labeled；**val labeled 不在 test 排除名單中**（已知 potential leakage，本計畫暫不修正） |

### 2.4 Supervised loss 與 adaptation loss 資料來源

**Supervised classification（DAE / MLP 共通）**：

```python
# trainer.py train_semi_dae / train_semi_mlp
data = torch.cat((xs, xt), 0)           # source batch + target labeled batch
target = torch.cat((ys, yt), 0)
feature = encoder(data)[0]              # DAE: (feature, ae_output)
output = adentropy_p(predictor(feature))
loss_c = CrossEntropyLoss(output, target)
```

**DAE 額外 reconstruction loss**：

```python
loss_ae_s = MSE(xs, ae_output_source_part)
loss_ae_t = MSE(xt, ae_output_target_part)
loss = loss_c + loss_ae_s + loss_ae_t
# method=="adv" 時另加 FGM adversarial 步驟
```

**Target unlabeled adaptation（MME / adentropy）**：

```python
output = encoder(xt_unl)[0]             # DAE
output = predictor(output)
loss_t = adentropy(adentropy_p, output, lamda=0.1)
```

**Validation metrics 計算範圍**：僅 source batch 部分（`softmax_output.narrow(0, 0, xs.size(0))`），不含 target labeled。

### 2.5 Prediction confidence（原版）

```python
# trainer.py test_shot / test_experiment
y_out = net(x)                          # logits
y_out = Softmax(dim=1)(y_out)
y_score = y_out[:, 1]                   # class 1 probability 作為 score
pred = y_out.max(1, keepdim=True)[1]
```

- 模型輸出：**logits** → **softmax probability**
- 原版 test 輸出：`sample_id \t response \t prob_class_1`
- **Confidence 建議**：`probability_class_1`（positive class probability），同時輸出 `probability_class_0`、`pred_label`

### 2.6 訓練結束後回傳的模型

`trainer.train_semi_dae` / `train_semi_mlp` 雖追蹤 `best_auc` / `best_loss`，但**回傳最後一個 epoch 的模型**，非 best validation checkpoint。

---

## 3. 新版資料切分設計

### 3.1 單一 random seed

**移除** `for i in range(50)`，改為 CLI：

```bash
--random_seed 42
```

控制範圍：

| 項目 | random seed 用途 |
|---|---|
| source independent test split | `random_state=random_seed` |
| source 5-fold | `StratifiedKFold(..., random_state=random_seed)`（若確認使用 stratify） |
| target 80/20 split | `random_state=random_seed` |
| target few-shot 抽樣 | `random.seed(random_seed)` |
| PyTorch / NumPy / Python | `manual_seed`, `np.random.seed`, `random.seed` |
| DataLoader shuffle | 由 seed 間接控制 |
| t-SNE | `random_state=random_seed` |
| KMeans | `random_state=random_seed` |

### 3.2 Source independent test split

```text
source_full (全部 source labeled samples)
  ├── source_test          ← 先切出（預設 10%，CLI: --source_test_size 0.1）
  └── source_train_val     ← 剩餘樣本做 5-fold
        ├── fold_k_train
        └── fold_k_val
```

- **建議**：stratified by `response`（待使用者確認，見第 17 節 Q1）
- `source_test` **不參與**任何 fold 的 training / validation

### 3.3 Source 5-fold cross-validation

```bash
--n_splits 5
```

- 在 `source_train_val` 上執行 5-fold
- **建議**：`StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)`（待確認 Q2）
- 每 fold 獨立訓練一個模型

### 3.4 Target few-shot split（維持原版邏輯）

**原則**：保留原版 target 80/20 → train/val 各自 n-shot → unlabeled → test 定義。

```text
target_full
  ├── target_train (80%)
  │     ├── target_labeled_train    (每 class n 個)
  │     └── target_unlabeled_train  (剩餘)
  ├── target_val (20%)
  │     ├── target_labeled_val      (每 class n 個)
  │     └── target_unlabeled_val    (剩餘)
  └── target_test (= target_full − target_labeled_train)
```

- **建議**：target few-shot split **固定一次**，所有 source fold **共用**（待確認 Q3）
- `--n` 維持原版語意（每 class labeled 數量）

### 3.5 與原版切分的差異摘要

| 項目 | 原版 | 新版 |
|---|---|---|
| 重複次數 | 50 seeds | 1 seed |
| Source split | 80/20 train/val | 10% test + 5-fold on remainder |
| Source stratify | 否 | **建議改為 yes**（待確認） |
| Target split | 同原版 | **同原版**（含已知 test leakage，暫不修正） |
| Target few-shot | 每 seed 重抽 | **建議固定一次**（待確認） |

---

## 4. 新版 Training Workflow

### 4.1 每 fold 資料流

| 資料 | 用途 |
|---|---|
| `source_fold_train` | supervised classification loss |
| `target_labeled_train` + `target_labeled_val`（原版 train/val phase） | supervised classification loss |
| `target_unlabeled_train` + `target_unlabeled_val` | adentropy / domain adaptation |
| `source_fold_val` | fold validation metrics |
| `source_test` | independent source test metrics（每 fold 評估） |
| `all_source`（source_full） | final latent + source prediction table |
| `all_target`（target_full） | final target latent + target prediction table |

### 4.2 保留的原版 semi-supervised domain adaptation

**不修改**：

1. `train_semi_dae` / `train_semi_mlp` 核心 loss 串接
2. source + target labeled 合併 supervised loss
3. target unlabeled adentropy loss（`lamda=0.1`）
4. DAE reconstruction loss + FGM adversarial（`method=="adv"`）
5. encoder / predictor / Predictor_adentropy 架構
6. `utils.create_dataset` DataLoader 建立方式
7. WeightedRandomSampler（source train）

**可修改（experiment 層）**：

1. 傳入 fold-specific source train/val index
2. 訓練完成後呼叫 latent / prediction / eval exporter
3. 新增輸出路徑 `save/latent_ssda/`

### 4.3 不應修改的部分

- `model.py` 模型架構
- `utils.create_dataset` 介面
- target few-shot 每 class 抽 `n` 的邏輯
- 原版 `save/results/sc/`、`save/sc/all_path/` 輸出（**建議保留**，見 Q17）

---

## 5. Latent 輸出設計

### 5.1 Latent 來源：encoder bottleneck

| Encoder | 取法 | 維度 |
|---|---|---|
| **DAE** | `encoder(x)[0]` 或 `encoder(x)` tuple 第一項 | `sc_tasks['pathway']` = **128** |
| **MLP** | `encoder(x)` | `sc_tasks['pathway']` = **128** |

```python
def get_encoder_latent(encoder, x):
    out = encoder(x)
    if isinstance(out, tuple):
        return out[0]
    return out
```

**程式碼注意事項**：

1. `--bottleneck` CLI 參數在現行 `experiment_shot.py` 中**未被使用**；實際 latent 維度由 `utils.cell_dim()` 的 `pathway=128` 決定
2. DAE 的 `forward()` 在 eval mode 下仍會套用 denoising mask（`np.random.binomial`）；latent 輸出時需確認是否沿用此行為（與原版 `Test_Double_Model` 一致）或改呼叫 `encoder.ae.encode()`（待確認 Q18）
3. Latent 必須來自**完成全部 epoch semi-supervised training 後**的 final encoder（原版回傳 last epoch，非 best checkpoint）

### 5.2 輸出時間點

每 fold training 結束後：

1. `encoder.eval()`；`predictor.eval()`；`adentropy_p.eval()`
2. 對 **all source samples** encode → `source_latent_representation.pkl`
3. 對 **all target samples** encode → `target_latent_representation.pkl`

### 5.3 pkl 格式

```python
{
    "sample_id_1": [0.123, -0.041, ..., 0.552],   # length 128
    "sample_id_2": [0.088, 0.017, ..., -0.230]
}
```

- key：`str(sample_id)`（與 metadata index 一致）
- value：`list[float]`，維度 128

### 5.4 Metadata CSV

**source_latent_metadata.csv**：

| 欄位 | 說明 |
|---|---|
| sample_id | sample ID |
| domain | `source` |
| split | `source_fold_train` / `source_fold_val` / `source_test` |
| response_label | GT response |
| cancer_type | 若提供 metadata；否則 `Unknown` 或缺失（待 Q8） |
| fold | fold index (0–4) |
| seed | random seed |
| drug | drug name |

**target_latent_metadata.csv**：

| 欄位 | 說明 |
|---|---|
| sample_id | sample ID |
| domain | `target` |
| target_role | `target_labeled_train` / `target_unlabeled_train` / `target_labeled_val` / `target_unlabeled_val` / `target_test` |
| response_label | GT response |
| cancer_type | 同上 |
| fold | fold index |
| seed | random seed |
| drug | drug name |

---

## 6. Prediction 結果輸出設計

### 6.1 推論管線

沿用 `Test_Double_Model`：

```python
feature = encoder(x)[0]       # DAE
output = adentropy_p(predictor(feature))   # logits
probs = softmax(output, dim=1)
pred_label = argmax(probs, dim=1)
confidence = probs[:, 1]      # 建議：positive class probability
```

### 6.2 Source prediction results

檔案：`source_prediction_results.csv`

包含 **all source samples**（train + val + test），欄位：

| 欄位 | 說明 |
|---|---|
| sample_id | sample ID |
| domain | `source` |
| split | `source_fold_train` / `source_fold_val` / `source_test` |
| response_label | GT |
| pred_label | 0/1 |
| confidence | `probability_class_1`（建議） |
| probability_class_0 | optional |
| probability_class_1 | optional |
| fold | fold index |
| seed | random seed |
| drug | drug name |
| cancer_type | optional |

### 6.3 Target prediction results

檔案：`target_prediction_results.csv`

包含 **all target samples**，欄位：

| 欄位 | 說明 |
|---|---|
| sample_id | sample ID |
| domain | `target` |
| target_role | 見 5.4 |
| response_label | GT |
| pred_label | 0/1 |
| confidence | `probability_class_1` |
| probability_class_0 | optional |
| probability_class_1 | optional |
| fold | fold index |
| seed | random seed |
| drug | drug name |
| cancer_type | optional |

### 6.4 Performance metrics CSV

| 檔案 | 評估資料 | 指標 |
|---|---|---|
| `source_val_metrics.csv` | source fold val | AUC, AUPR, Accuracy, F1, Balanced Accuracy |
| `source_test_metrics.csv` | source independent test | 同上 |
| `target_prediction_metrics.csv` | all target（或分 role，待 Q19） | 同上 |

---

## 7. Cancer Type Metadata 輸入設計

### 7.1 背景

SSDA4Drug 預設四個 CSV **不含** cancer type。需額外提供 metadata 檔案，設計參考 `DAPL-master/A_pretrain.py`。

### 7.2 建議 CLI

```bash
--source_cancer_type_path /path/to/source_cancer_type.csv
--target_cancer_type_path /path/to/target_cancer_type.csv
--sample_id_col Sample_ID
--cancer_type_col Cancer_type
```

### 7.3 檔案格式（參考 DAPL）

CSV，至少兩欄：

| 欄位 | 說明 |
|---|---|
| sample_id_col（預設 `Sample_ID`） | 與 expression matrix sample ID 對應 |
| cancer_type_col（預設 `Cancer_type`） | cancer type 字串 |

DAPL 的 `read_cancer_type()` 行為（可借鑑）：

- 欄位名稱大小寫不敏感 fallback
- TCGA sample ID 可 normalize 為 patient key（`TCGA-XX-XXXX` 前三段）
- 重複 sample ID：`keep="first"`
- 不在 expression matrix 的 metadata 樣本：排除並記錄

### 7.4 Sample ID matching / normalization

SSDA4Drug 現有資料：

| Domain | Sample ID 範例 | 位置 |
|---|---|---|
| Source (Gefitinib) | `683665`（數字） | expression column header / meta index |
| Target (Gefitinib) | `c4653`（字串） | 同上 |

**Normalization 規則（待使用者提供，見 Q7）**：

1. 統一轉 `str` 比對
2. 是否 strip 空白、lower case
3. Target 是否需去除 prefix（如 `c`）
4. TCGA 是否需 patient-level dedup（若資料為 TCGA）

### 7.5 Missing cancer type handling

若 expression 有 sample 但 cancer type metadata 無對應：

| 選項 | 行為 |
|---|---|
| A | 標記為 `Unknown` |
| B | 排除該 sample（不進 t-SNE / KMeans；仍可在 prediction table 標 `Unknown`） |

**待使用者確認 Q8**。

### 7.6 對齊檢查輸出

`cancer_type_mapping_summary.csv`：

| 欄位 | 說明 |
|---|---|
| domain | source / target |
| total_expression_samples | expression matrix 樣本數 |
| matched_samples | 成功對齊數 |
| missing_in_metadata | 有 expression 無 cancer type |
| extra_in_metadata | metadata 有但 expression 無 |
| unknown_samples | 標記 Unknown 數（若採 A） |
| excluded_samples | 排除數（若採 B） |

---

## 8. t-SNE 視覺化設計

每 fold 輸出兩張 PNG（預設 `--run_tsne` 未來可選；初版**預設開啟**）。

### 8.1 tsne_domain_mixing.png

- 資料：source + target combined latent（all samples）
- 著色：`domain`（source / target）
- 參數：`random_state=random_seed`；其餘 sklearn 預設

### 8.2 tsne_cancer_type.png

- 資料：source + target combined latent
- 著色：`cancer_type`
- 缺失樣本：依 Q8 策略（Unknown 或排除）
- **需 cancer type metadata**；若未提供則跳過並記錄於 log

---

## 9. Latent Distribution Metrics

檔案：`latent_distribution_metrics.csv`

| 指標 | 比較對象 | 解讀 |
|---|---|---|
| FID | source full latent vs target full latent | 越低越接近 |
| MMD | 同上 | 越低 domain gap 越小 |
| Wasserstein | 同上 | 越低 distribution shift 越小 |

**建議比較對象**：`source_full` vs `target_full`（待確認 Q12）

欄位：

```
fold, seed, drug, source_n, target_n,
fid_source_target, mmd_source_target, wasserstein_source_target
```

實作可參考 `DAPL-master/A_pretrain.py` → `tools.latent_metrics`（若可 import 則複用；否則在 `latent_eval.py` 內實作同等函式）。

---

## 10. KMeans Cancer Type Clustering Metrics

檔案：`kmeans_cancer_type_metrics.csv`

### 10.1 分析對象

**建議**：source + target combined latent（待確認 Q10）

### 10.2 Ground truth

- Label：`cancer_type`（字串類別）
- `n_clusters = len(unique cancer types)`（排除 `Unknown` 與否：待 Q11）

### 10.3 Metrics

| 指標 | 說明 |
|---|---|
| ARI | Adjusted Rand Index |
| NMI | Normalized Mutual Information |
| Silhouette | cluster separation |
| Calinski-Harabasz | compactness / separation |
| Davies-Bouldin | 越低越好 |

### 10.4 輸出欄位（combined 為主）

```
fold, seed, drug, k, k_eff, n_cancer_types, samples_used,
ari, nmi, silhouette, calinski_harabasz, davies_bouldin
```

---

## 11. 檔案與資料夾輸出結構

```text
save/
  latent_ssda/
    {drug}/
      seed_{random_seed}/
        config.json
        source_split.csv
        target_fewshot_split.csv
        cancer_type_mapping_summary.csv
        fold_0/
          model_final.pth
          source_latent_representation.pkl
          target_latent_representation.pkl
          source_latent_metadata.csv
          target_latent_metadata.csv
          source_prediction_results.csv
          target_prediction_results.csv
          tsne_domain_mixing.png
          tsne_cancer_type.png
          latent_distribution_metrics.csv
          kmeans_cancer_type_metrics.csv
          source_val_metrics.csv
          source_test_metrics.csv
          target_prediction_metrics.csv
        fold_1/
          ...
        fold_4/
          ...
        metrics_summary.csv
        latent_metrics_summary.csv
        kmeans_cancer_type_summary.csv
```

**split CSV 內容**：

- `source_split.csv`：`sample_id, response_label, split`（source_test / fold_k_train / fold_k_val）
- `target_fewshot_split.csv`：`sample_id, response_label, target_role`

**config.json**：完整 CLI args、git hash（optional）、latent dim、sample counts、timestamp

---

## 12. 需要修改的程式檔案

### 12.1 `experiment_shot.py`（主要）

1. 移除 / 參數化 50-seed loop → 單一 `--random_seed`
2. 新增 source test split + 5-fold loop
3. 固定 target few-shot split（所有 fold 共用）
4. 每 fold 呼叫 training + export pipeline
5. 新增 cancer type / latent output CLI
6. 保留原版 result path 輸出（若 Q17 確認）

### 12.2 `trainer.py`（最小修改）

1. 原則不改 loss 邏輯
2. 可選：新增 `predict_dataframe()` helper 或保持 exporter 在獨立模組
3. 可選：支援回傳 best model（若 Q14 確認需要 best checkpoint）
4. **注意**：原版回傳 last epoch model；latent 應與 test 使用同一 checkpoint

### 12.3 `model.py`

- **不修改**架構
- Latent helper 放 `latent_utils.py`

### 12.4 `utils.py`

- **尽量不修改**（保留 DataLoader 相容性）
- Sample ID 追蹤在 exporter 層用 pandas index 完成

---

## 13. 建議新增的工具檔案

| 檔案 | 職責 |
|---|---|
| `latent_utils.py` | `get_encoder_latent`, `encode_latent_dict`, `save_latent_pkl`, metadata builder |
| `prediction_utils.py` | `predict_dataframe`, `save_prediction_results`, `compute_prediction_metrics` |
| `latent_eval.py` | t-SNE, FID, MMD, Wasserstein, KMeans cancer type metrics |
| `split_utils.py` | source test split, 5-fold, target few-shot, split CSV export |
| `cancer_type_utils.py` | load/normalize/align cancer type metadata |

---

## 14. CLI 參數設計

### 14.1 新增參數

```bash
--random_seed 42
--source_test_size 0.1
--n_splits 5
--latent_output_dir save/latent_ssda
--source_cancer_type_path   # optional
--target_cancer_type_path   # optional
--sample_id_col Sample_ID
--cancer_type_col Cancer_type
```

### 14.2 保留原版參數

```bash
--drug --n --encoder --encoder_h_dims --bottleneck
--epochs --lr --batch_size --dropout
--method --gene --device
--result --sc_all --umap_path
--predictor_h_dims --dropout --fix_source
```

### 14.3 設計原則

- Latent / t-SNE / metrics **預設全部輸出**（不加 `--save_latent`）
- Cancer type 相關圖表：未提供 metadata 時 skip cancer type 圖，domain mixing 仍輸出

---

## 15. 實作順序

### P0：基礎流程

1. `split_utils.py`：source test + 5-fold + target few-shot（原版邏輯）
2. 改 `experiment_shot.py`：單 seed + fold loop
3. 驗證 split 不重疊、可重現

### P1：Latent + Prediction

1. `latent_utils.py` + `prediction_utils.py`
2. 每 fold 輸出 pkl + CSV tables + metrics

### P2：Cancer type

1. `cancer_type_utils.py` + CLI
2. mapping summary + metadata 欄位

### P3：Latent analysis

1. `latent_eval.py`：t-SNE + FID/MMD/Wasserstein + KMeans
2. fold / summary CSV

### P4：Summary + 相容性

1. 5-fold summary aggregation
2. 保留原版 log / result
3. `config.json` + 文件更新

---

## 16. 測試與驗證方式

### 16.1 Split 驗證

- [ ] `source_test` ∉ 任何 fold train/val
- [ ] 5-fold 覆蓋完整 `source_train_val`，無重疊
- [ ] target labeled 每 class = `n`（train 與 val 各一次）
- [ ] target labeled / unlabeled 無重疊
- [ ] 同 seed 重跑 split 完全一致

### 16.2 Latent 驗證

- [ ] source pkl 樣本數 = len(source_full)
- [ ] target pkl 樣本數 = len(target_full)
- [ ] 向量維度 = 128
- [ ] DAE / MLP 皆正確

### 16.3 Prediction 驗證

- [ ] 全樣本皆有 pred + confidence
- [ ] metrics 與原版 softmax class-1 一致
- [ ] GT 缺失時 skip metrics、保留 prediction

### 16.4 Cancer type 驗證

- [ ] 對齊摘要數字正確
- [ ] t-SNE / KMeans 在提供 metadata 時可執行

### 16.5 Smoke test 命令（實作後）

```bash
python experiment_shot.py \
  --drug Gefitinib --n 3 --encoder DAE --epochs 2 \
  --random_seed 42 --source_test_size 0.1 --n_splits 5 \
  --latent_output_dir save/latent_ssda
```

---

## 17. 尚未確定、必須向使用者確認的問題清單

以下問題**在實作前必須取得明確答案**。表中「目前傾向」僅供討論，**不視為已決定**。

### Source split

| ID | 問題 | 目前傾向 |
|---|---|---|
| **Q1** | `source_test` 是否 stratify by `response`？ | 是 |
| **Q2** | source 5-fold 是否 `StratifiedKFold` by `response`？ | 是 |

### Target split

| ID | 問題 | 目前傾向 |
|---|---|---|
| **Q3** | target few-shot 是否所有 fold **共用同一組** labeled/unlabeled？ | 是（固定一次） |
| **Q4** | 是否**完全保留**原版 target 80/20 → train/val 各自 n-shot → test 定義？ | 是（含 val labeled 仍在 test 中） |
| **Q5** | target prediction metrics 計算範圍：all target / 僅 test / 分 role 輸出？ | all target |

### Prediction

| ID | 問題 | 目前傾向 |
|---|---|---|
| **Q6** | confidence 定義：已確認原版為 softmax，`confidence = P(class=1)`，是否同時輸出 `P(class=0)`？ | 是，兩者都輸出 |
| **Q7** | source / target prediction 是否包含 **all samples** 並標註 split/role？ | 是 |

### Cancer type metadata

| ID | 問題 | 需使用者提供 |
|---|---|---|
| **Q8** | 缺少 cancer type 的 sample：標記 `Unknown`（A）或排除（B）？ | **必選** |
| **Q9** | `source_cancer_type_path` 實際檔案路徑？ | **必填**（若要做 cancer type 評估） |
| **Q10** | `target_cancer_type_path` 實際檔案路徑？ | **必填** |
| **Q11** | `sample_id_col` / `cancer_type_col` 實際欄位名稱？ | **必填** |
| **Q12** | sample ID 是否需要 normalization（如 TCGA patient key、target `c` prefix）？ | **必填** |
| **Q13** | t-SNE cancer type：source + target 合併著色？ | 是 |
| **Q14** | KMeans：combined latent / source-only / target-only？ | combined 為主 |
| **Q15** | KMeans 是否排除 `Unknown` cancer type？ | 待確認 |

### Latent / metrics / checkpoint

| ID | 問題 | 目前傾向 |
|---|---|---|
| **Q16** | FID/MMD/Wasserstein：source_full vs target_full？ | 是 |
| **Q17** | 每 fold 儲存 `model_final.pth`（last epoch）？ | 是（與原版一致） |
| **Q18** | 是否另存 **best validation** checkpoint？ | 未決定 |
| **Q19** | DAE latent 輸出：沿用 `encoder.forward()`（含 denoising）或 `ae.encode()`（無 denoising）？ | 與 `Test_Double_Model` 一致（forward） |
| **Q20** | 是否保留原版 `save/results/sc/` 等 log？ | 是，額外新增 `latent_ssda` |
| **Q21** | 是否需要 CSV 版 latent（sample × dim）？ | pkl 為主，CSV optional |

### 實驗範圍

| ID | 問題 | 目前傾向 |
|---|---|---|
| **Q22** | 本改良是否僅針對 `experiment_shot.py`，不修改 `experiment.py`（Extended）？ | 是 |
| **Q23** | 若未提供 cancer type 檔案：是否仍跑完整流程（skip cancer 圖/metrics）？ | 是 |

---

## 附錄 A：原版 vs 新版 Training 對照圖

```text
                    ┌─────────────────────────────────────┐
                    │         source_full (all)           │
                    └─────────────────┬───────────────────┘
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  source_test (10%)        source_train_val (90%)
                  [held-out]                      │
                                                  │ 5-fold
                                    ┌─────────────┼─────────────┐
                                    ▼             ▼             ▼
                              fold_k_train   fold_k_val    (each fold)

                    ┌─────────────────────────────────────┐
                    │         target_full (all)           │
                    └─────────────────┬───────────────────┘
                                      │ 80/20 (fixed, all folds)
                         ┌────────────┴────────────┐
                         ▼                         ▼
                   target_train               target_val
                   n-shot → labeled/unlabeled  n-shot → labeled/unlabeled

Training (per fold):
  supervised: source_fold_train + target_labeled_*
  adaptation: target_unlabeled_*
  metrics:    source_fold_val, source_test, target (all)
  export:     all_source latent/pred, all_target latent/pred
```

---

## 附錄 B：關鍵程式碼位置索引

| 主題 | 檔案 | 行號（約） |
|---|---|---|
| 50-seed loop | `experiment_shot.py` | L33 |
| Source 80/20 split | `experiment_shot.py` | L64-65 |
| Target n-shot | `experiment_shot.py` | L103-118 |
| Target test 定義 | `experiment_shot.py` | L139-141 |
| Supervised + adentropy | `trainer.py` | `train_semi_dae` L231-271 |
| Softmax confidence | `trainer.py` | `test_shot` L75-79 |
| DAE latent | `model.py` | `DAE.forward` L221-230 |
| MLP latent | `model.py` | `MLP.forward` L106-109 |
| Latent dim 128 | `utils.py` | `cell_dim` → `pathway: 128` |
| DAPL pkl 格式 | `DAPL-master/A_pretrain.py` | `encode_latent_dict` L405-417 |

---

*文件版本：2026-05-21 · 基於 SSDA4Drug-main 程式碼審閱*
