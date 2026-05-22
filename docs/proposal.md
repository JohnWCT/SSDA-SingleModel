# Proposal: Convert SSDA-SingleModel to Multi-label / Multi-drug SSDA Framework

> **實作已完成階段：** 多藥入口為 `experiment_multilabel_ssda.py`。若要從 `experiment_shot.py` 改寫或對照實作，請優先閱讀 **§26** 與 `docs/design.md` **§14**（As-Built + 遷移指南）。架構細節見 `design.md`；本文件 §2、§19–§22、§26 已同步 2025-05 實作。

## 1. Project Goal

本 proposal 目標是修改目前的：

```text
https://github.com/JohnWCT/SSDA-SingleModel
```

將目前的 single-drug SSDA-SingleModel 改造成 **multi-label / multi-drug SSDA framework**。

新版模型不再針對單一 drug 訓練，而是：

```text
omics data table
  -> SSDA encoder
  -> sample-level latent representation
  -> multi-output prediction head
  -> all-drug response vector
```

也就是每個 sample 一次輸出所有 drugs 的 response prediction。

本修改應保留 SSDA-SingleModel 的核心精神：

1. 使用 source labeled data 進 supervised training。
2. 使用少量 target labeled data 進 semi-supervised target adaptation。
3. 使用 target unlabeled data 進 entropy / adentropy domain adaptation。
4. 訓練完成後輸出 domain-adapted sample latent。
5. 不使用 DAPL-style drug latent。
6. 不使用 sample-drug pair model。
7. 不將 omics input 預先轉成 latent；omics 輸入仍然是 data table。

---

## 2. Key Design Decisions

目前已確認的設計如下。

| 項目 | 決議 |
|---|---|
| omics input | 使用原 SSDA-SingleModel 的 omics data table，不使用預先產生的 latent |
| sample latent | 由 SSDA encoder 在訓練後輸出 |
| drug latent | 不使用，刪除所有 drug latent 相關內容 |
| model type | multi-output model |
| prediction head | 每個 sample 一次輸出所有 drugs 的 response vector |
| output dimension | `n_drugs = len(drug_list.csv)` |
| response input | source / target response 使用 long table |
| model training label format | long table 轉 wide response matrix + mask tensor |
| missing label | 使用 mask loss |
| task type | 每次 run 使用 `--task_type classification` 或 `--task_type regression` 指定 |
| target label | target input 已經是 0/1 classification label |
| regression threshold | 固定使用 `neg_log2_auc >= 1.0` 轉 binary responder |
| source / target drug list | 使用 source ∪ target 的 drug union，不刪除任何 drug |
| drug order | 使用 `drug_list.csv` 固定 multi-output head 的欄位順序 |
| source / target split | 以 sample 為單位切分，避免同一 sample 同時出現在 train / validation |
| target n-shot | 在 sample-drug position level，每個 drug、每個 class 各抽 n 個 labeled positions |
| target drug 不存在 | 若 target 沒有該 drug label，該 drug 的 target n-shot 直接跳過並記錄 |
| target unlabeled | target 中有觀測 label 但未被抽為 n-shot labeled 的 positions |
| latent pkl | 維持 sample-level latent pkl，不是 sample-drug-level latent |
| **DAPL 欄位（程式寫死，不需 CLI）** | 見 §26.2；omics/response join 規則見 `ssda_multilabel/sample_id.py` |
| **輸出根目錄** | 預設直接寫入 `--output_dir`（例如 `outputs_smoke_classification/`），**不再**巢狀 `ssda_multilabel/seed_*` |
| **latent pkl 涵蓋範圍** | **完整** source/target omics 表（非 val-only）；split 僅標在 prediction CSV |
| **target 評估範圍** | 所有 `mask=1` 的 target sample-drug positions（含 n-shot labeled） |
| **source 評估範圍** | 僅 `split == source_test`（每 fold 固定 test set） |
| **regression run 的 target 指標** | 仍用 **分類** 指標（target 標籤恒為 0/1） |
| **duplicate (sample, drug)** | 預設 `--duplicate_response_strategy mean`（PRISM 同藥名多 broad_id） |
| **reg_loss 預設** | `mae`（非 proposal 初稿的 mse） |
| **cancer type** | 依 omics 路徑自動選 DAPL 表；不需手動 `--source_cancer_type_path`（pretrain / Winnie 各一套） |

---

## 3. Scope

### 3.1 In Scope

本次修改包含：

1. 新增 `experiment_multilabel_ssda.py` 作為 multi-label / multi-drug 任務入口（保留原版 `experiment_shot.py` 單藥流程）。
2. 新增資料整理模組，將 source / target long response table 轉成 wide matrix + mask tensor。
3. 新增 `drug_list.csv` 生成與讀取邏輯。
4. 新增 multi-output prediction head。
5. 新增 masked loss：
   - classification: masked BCEWithLogitsLoss
   - regression: masked MSE / Huber / MAE，預設 MSE
6. 保留 SSDA target n-shot semi-supervised adaptation。
7. 將 target n-shot 改為 sample-drug position-level n-shot。
8. 支援 classification run。
9. 支援 regression run。
10. 輸出 source / target prediction result table。
11. 輸出 source / target sample-level latent pkl。
12. 輸出 t-SNE domain mixing 與 cancer type 圖。
13. 輸出 FID / MMD / Wasserstein。
14. 輸出 KMeans cancer type clustering metrics。
15. 輸出 per-drug 與 summary metrics。

### 3.2 Out of Scope

本次不包含：

1. 不使用 drug latent pkl。
2. 不使用 DAPL `C_prototypical.py` 的 sample-drug pair model。
3. 不把 omics data 預先轉為 latent 後再訓練。
4. 不使用 sample-drug pair shared head。
5. 不刪除 source-only 或 target-only drugs。
6. 不修改 encoder 架構本身，除非 multi-output head 需要新增。
7. 不輸出 AE reconstruction。
8. 不破壞原版 SSDA-SingleModel 的既有輸出，應新增獨立 multi-label pipeline。

---

## 4. Input Design

## 4.1 Omics Input

新版 omics input 維持原 SSDA-SingleModel 的 data table 使用方式。

不使用 DAPL `A_pretrain.py` 中已經預先產生的 latent 作為模型輸入。

建議格式：

```text
source_omics.csv
target_omics.csv
```

範例（DAPL As-Built）：

| Source: `Sample_ID` | gene_1 | ... |
|---|---:|---:|
| ACH-000001 | 0.12 | ... |

| Target: `tissue_id` | gene_1 | ... |
|---|---:|---:|
| TCGA-AA-3695-01A | 0.33 | ... |

需求：

1. Source 使用 `Sample_ID`；Target 使用 `tissue_id`（pretrain 舊檔可用 `patch_dapl_csv_columns.py` 將第一欄改名）。
2. 其餘欄位為 numeric omics features。
3. source / target feature 必須對齊。
4. 若 source / target feature 不完全相同，應取共同 features，並記錄被移除的 features。
5. sample ID normalization 規則需清楚記錄。

---

## 4.2 Source Response Long Table

source response 使用 long table 作為標準輸入。

```text
source_response.csv
```

建議欄位（DAPL As-Built）：

| 欄位 | 說明 |
|---|---|
| `Sample_ID` | source sample ID（固定） |
| `drug_name` | drug 名稱（union 鍵） |
| `Label` 或 `--source_response_col` | GDSC 用 `Label`；PRISM 用 `neg_log2_auc` |

classification：`--source_response_col Label`（0/1）。

regression：`--source_response_col neg_log2_auc`（連續值）。

---

## 4.3 Target Response Long Table

target response 也使用 long table 作為標準輸入。

```text
target_response.csv
```

建議欄位（DAPL As-Built）：

| 欄位 | 說明 |
|---|---|
| `Patient_id` | TCGA 三段 patient key（固定） |
| `drug_name` | drug 名稱 |
| `Label` | binary 0/1（程式固定 `target_response_col=Label`） |

Target omics 用 `tissue_id`（四段）經 `tcga_patient_key` join 到 `Patient_id`。

target response 在本專案中一律視為 classification label。

也就是 target input 已經是 0/1，不需要再從 continuous value 轉換。

---

## 4.4 Drug List

新版必須使用 `drug_list.csv` 固定 multi-output head 的 drug order。

```text
drug_list.csv
```

### 4.4.1 Drug List 生成方式

`drug_list.csv` 應由 source / target response long table 的 drug 聯集產生：

```text
drug_list = sorted(unique(source_response.drug_id ∪ target_response.drug_id))
```

不刪除任何藥物。

也就是：

| drug type | 是否保留 |
|---|---|
| source 與 target 都有 | 保留 |
| source-only drug | 保留 |
| target-only drug | 保留 |

### 4.4.2 Drug List Format

```csv
drug_id,drug_index
DrugA,0
DrugB,1
DrugC,2
```

要求：

1. `drug_index` 必須從 0 開始連續編號。
2. source / target wide response matrix 必須使用完全相同的 drug order。
3. model output dimension 必須等於 `len(drug_list.csv)`。
4. prediction output table 必須包含 `drug_id` 與 `drug_index`。

---

## 4.5 Cancer Type Metadata

目前 SSDA-SingleModel 原始輸入不一定包含 cancer type。

若要繪製 cancer type t-SNE 與計算 KMeans cancer type metrics，需要額外支援 cancer type metadata。

建議輸入：

```text
source_cancer_type.csv
target_cancer_type.csv
```

建議欄位：

| 欄位 | 說明 |
|---|---|
| sample_id | sample ID |
| cancer_type | cancer type |

處理規則：

1. cancer type metadata 與 omics sample ID 對齊。
2. 若 metadata 中有 sample 不存在於 omics table，記錄並排除。
3. 若 omics sample 缺少 cancer type，預設標記為 `Unknown`。
4. KMeans cancer type metrics 是否排除 `Unknown` 應提供參數控制。

---

## 5. Data Preparation Module

需要新增一個專門的資料整理模組。

已實作於：

```text
ssda_multilabel/prepare.py   # prepare_multilabel_data(config)
```

### 5.1 Responsibilities

此模組負責：

1. 讀取 source omics table。
2. 讀取 target omics table。
3. 讀取 source response long table。
4. 讀取 target response long table。
5. 建立 `drug_list.csv`。
6. 將 source response long table 轉成 wide matrix。
7. 將 target response long table 轉成 wide matrix。
8. 建立 source mask matrix。
9. 建立 target observed mask matrix。
10. 執行 target n-shot position-level split。
11. 建立 target labeled mask。
12. 建立 target unlabeled mask。
13. 建立 source train / validation / test sample-level split。
14. 輸出資料對齊報告。
15. 輸出缺失值統計報告。

---

## 5.2 Long Table to Wide Matrix

source long table 轉換為：

```text
Y_source:      [n_source_samples, n_drugs]
mask_source:   [n_source_samples, n_drugs]
```

target long table 轉換為：

```text
Y_target:             [n_target_samples, n_drugs]
mask_target_observed: [n_target_samples, n_drugs]
```

其中：

```text
mask = 1 表示該 sample-drug position 有 response label
mask = 0 表示 missing
```

若某 sample 對某 drug 沒有 response，則：

```text
Y[sample, drug] = arbitrary value, e.g. 0
mask[sample, drug] = 0
```

loss 與 metrics 必須只使用 mask 為 1 的 positions。

---

## 6. Target N-shot Design in Multi-label Setting

原始 SSDA 是 single-drug，因此 target labeled / target unlabeled 是 sample-level。

新版是 multi-label，因此 target n-shot 必須改為：

```text
sample-drug position-level n-shot
```

而不是 sample-level n-shot。

---

## 6.1 Target N-shot Definition

對每一個 drug column，分別從 target observed positions 中抽樣：

```text
class 0: up to n positions
class 1: up to n positions
```

也就是：

```text
for each drug d:
    labeled_0 = sample up to n target samples where Y_target[:, d] == 0
    labeled_1 = sample up to n target samples where Y_target[:, d] == 1
```

抽出的 positions 形成：

```text
mask_target_labeled: [n_target_samples, n_drugs]
```

未被抽為 labeled，但 target 中有觀測 label 的 positions 形成：

```text
mask_target_unlabeled = mask_target_observed - mask_target_labeled
```

---

## 6.2 Target N-shot Edge Cases

| 情況 | 處理方式 |
|---|---|
| target 沒有某 drug | 跳過該 drug 的 n-shot 抽樣，記錄 warning |
| 某 drug 沒有 class 0 | class 0 不抽，記錄 warning |
| 某 drug 沒有 class 1 | class 1 不抽，記錄 warning |
| 某 drug 某 class 少於 n 個 | 有幾個抽幾個，記錄 warning |
| 同一 sample 多個 drug 被抽中 | 允許，因為 n-shot 是 position-level |
| target-only drug | 若 target 有 label，照樣可抽 n-shot；source mask 對該 drug 全 0 |
| source-only drug | target mask 對該 drug 全 0，因此 target n-shot 跳過 |

---

## 7. Source Split Design

source split 必須以 sample 為單位，而不是 sample-drug pair 為單位。

目標是避免同一個 sample 的不同 drug labels 同時出現在 train 與 validation / test 中。

### 7.1 Split Steps

```text
source samples
  ├── source_test
  └── source_train_val
        ├── fold_0_train / fold_0_val
        ├── fold_1_train / fold_1_val
        ├── fold_2_train / fold_2_val
        ├── fold_3_train / fold_3_val
        └── fold_4_train / fold_4_val
```

### 7.2 Required Behavior

1. 先從 source samples 切出 source independent test set。
2. 剩餘 source samples 做 K-fold cross-validation。
3. 每個 fold 的 train / val 都是 sample-level split。
4. 同一個 sample 不可同時出現在同一 fold 的 train 與 val。
5. source test 不可出現在任何 fold train / val。
6. fold split 應使用固定 random seed。
7. 若 classification labels 可用於 stratification，應盡量 stratify。
8. multi-label stratification 若難以實作，可先使用 sample-level random split，並記錄 label distribution。

---

## 8. Model Design

## 8.1 Overall Architecture

新版模型架構：

```text
omics features
  -> SSDA encoder
  -> sample latent
  -> multi-output prediction head
  -> response vector [n_drugs]
```

### 8.1.1 Output

模型輸出：

```text
prediction: [batch_size, n_drugs]
```

每個 column 對應 `drug_list.csv` 中的 drug order。

---

## 8.2 Encoder

保留 SSDA-SingleModel 原本 encoder 設計。

若 encoder 是 DAE：

```text
latent = encoder.ae.encode(x)
```

若 encoder 是 MLP：

```text
latent = encoder(x)
```

注意：如果 DAE forward 內部包含 random denoising mask，則 latent export 與 prediction 應使用 deterministic encoder path，避免輸出 latent 不穩定。

---

## 8.3 Multi-output Prediction Head

新增或修改 prediction head，使其輸出：

```text
[n_drugs]
```

而不是 single-drug binary logits。

分類任務：

```text
logits: [batch_size, n_drugs]
```

regression 任務：

```text
scores: [batch_size, n_drugs]
```

---

## 9. Task Type Design

每次 run 指定一種 task type：

```bash
--task_type classification
```

或：

```bash
--task_type regression
```

---

## 9.1 Classification Run

classification run 中：

| Domain | Label Type | Loss |
|---|---|---|
| source | 0/1 | masked BCEWithLogitsLoss |
| target labeled | 0/1 | masked BCEWithLogitsLoss |
| target unlabeled | no supervised loss | masked entropy / adentropy |

### 9.1.1 Classification Total Loss

```text
loss_total =
    source_classification_loss
  + target_labeled_classification_loss
  + lambda_adapt * target_unlabeled_adaptation_loss
  + optional_reconstruction_loss
```

---

## 9.2 Regression Run

regression run 中：

| Domain | Label Type | Loss / Usage |
|---|---|---|
| source | continuous | masked regression loss |
| target labeled | 0/1 classification | masked BCEWithLogitsLoss on the same output score |
| target unlabeled | no supervised loss | masked entropy / adentropy |
| target evaluation | 0/1 classification | classification metrics |

### 9.2.1 Regression Output Interpretation

模型仍然只輸出一個 multi-output score：

```text
score: [batch_size, n_drugs]
```

在 source regression positions：

```text
score -> continuous prediction
```

在 target classification positions：

```text
score -> binary logit
```

這樣可以保留原始 SSDA 中 target labeled samples 參與 supervised loss 的精神，而不需要額外建立 drug latent 或 sample-drug pair model。

### 9.2.2 Regression Total Loss

```text
loss_total =
    source_regression_loss
  + target_labeled_classification_loss
  + lambda_adapt * target_unlabeled_adaptation_loss
  + optional_reconstruction_loss
```

### 9.2.3 Regression Threshold

regression 轉 binary 使用固定門檻：

```text
neg_log2_auc >= 1.0 -> responder = 1
neg_log2_auc < 1.0  -> responder = 0
```

這用於：

1. source regression optional binary metrics。
2. target classification evaluation if needed。
3. 與 DAPL `C_prototypical.py` 的 responder definition 對齊。

---

## 10. Masked Loss Design

## 10.1 Classification Masked BCE Loss

```python
raw_loss = BCEWithLogitsLoss(reduction="none")(logits, y)
masked_loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1)
```

## 10.2 Regression Masked Loss

預設使用 MSE：

```python
raw_loss = MSELoss(reduction="none")(pred, y)
masked_loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1)
```

可選支援：

```text
--reg_loss mse
--reg_loss mae
--reg_loss huber
```

---

## 10.3 Target Labeled Mask

target labeled supervised loss 使用：

```text
mask_target_labeled
```

shape:

```text
[n_target_samples, n_drugs]
```

---

## 10.4 Target Unlabeled Adaptation Mask

target unlabeled adaptation loss 使用：

```text
mask_target_unlabeled
```

定義：

```text
mask_target_unlabeled = mask_target_observed - mask_target_labeled
```

只對 target response table 中存在 label、但沒有被抽進 n-shot labeled 的 positions 計算 adaptation loss。

---

## 11. Multi-label SSDA Training Workflow

每個 fold 中使用：

| Data | Tensor | Usage |
|---|---|---|
| source fold train | `X_source_train`, `Y_source_train`, `mask_source_train` | source supervised loss |
| source fold val | `X_source_val`, `Y_source_val`, `mask_source_val` | validation only |
| source test | `X_source_test`, `Y_source_test`, `mask_source_test` | final source test |
| target full | `X_target`, `Y_target`, `mask_target_observed` | target prediction / evaluation |
| target labeled positions | `mask_target_labeled` | target supervised loss |
| target unlabeled positions | `mask_target_unlabeled` | target adaptation loss |

### 11.1 Important Training Rule

Validation data must not update model parameters.

Legacy SSDA trainer should not be reused directly if it updates model during validation.

A safe trainer must ensure:

```python
if phase == "train":
    loss.backward()
    optimizer.step()
else:
    with torch.no_grad():
        only_forward_and_metrics()
```

---

## 12. Prediction Output Design

Even though the model output is wide matrix format, prediction results should be exported in long table format.

### 12.1 Source Prediction Table

Output file:

```text
source_prediction_results.csv
```

Rows:

```text
one row per observed source sample-drug position
```

Recommended columns:

| Column | Description |
|---|---|
| sample_id | source sample ID |
| drug_id | drug ID |
| drug_index | output column index |
| domain | source |
| split | source_fold_train / source_fold_val / source_test |
| ground_truth | observed source response |
| mask | should be 1 for exported observed rows |
| pred_score | raw model score |
| pred_label | binary prediction if applicable |
| confidence | probability or confidence |
| probability | sigmoid probability for classification |
| task_type | classification / regression |
| fold | fold index |
| seed | random seed |
| cancer_type | optional |

---

### 12.2 Target Prediction Table

Output file:

```text
target_prediction_results.csv
```

Rows:

```text
one row per observed target sample-drug position
```

Recommended columns:

| Column | Description |
|---|---|
| sample_id | target sample ID |
| drug_id | drug ID |
| drug_index | output column index |
| domain | target |
| target_role | target_labeled / target_unlabeled |
| ground_truth | observed target response |
| mask | should be 1 for exported observed rows |
| pred_score | raw model score |
| pred_label | binary prediction |
| confidence | sigmoid probability |
| probability | sigmoid probability |
| task_type | classification / regression |
| fold | fold index |
| seed | random seed |
| cancer_type | optional |

---

## 13. Latent Output Design

Latent remains sample-level.

It should not become sample-drug-level.

Every fold should output:

```text
source_latent_representation.pkl
target_latent_representation.pkl
```

Format:

```python
{
    "sample_id_1": [latent_0, latent_1, ..., latent_127],
    "sample_id_2": [latent_0, latent_1, ..., latent_127]
}
```

### 13.1 Latent Export Requirements

1. Latent must be exported after full SSDA training.
2. Latent must be domain-adapted.
3. Latent must come from the final encoder bottleneck.
4. Latent must be deterministic during export.
5. Source latent contains all source samples.
6. Target latent contains all target samples.
7. Latent pkl is sample-level only.

---

## 14. t-SNE and Latent Evaluation

Each fold should output:

```text
tsne_domain_mixing.png
tsne_cancer_type.png
latent_distribution_metrics.csv
kmeans_cancer_type_metrics.csv
```

### 14.1 t-SNE Domain Mixing

Use combined source + target sample latent.

Color by:

```text
source / target
```

### 14.2 t-SNE Cancer Type

Use combined source + target sample latent.

Color by:

```text
cancer_type
```

If cancer type is missing, default:

```text
Unknown
```

---

## 15. Latent Distribution Metrics

Compute source-target latent distribution metrics:

| Metric | Comparison |
|---|---|
| FID | source full latent vs target full latent |
| MMD | source full latent vs target full latent |
| Wasserstein distance | source full latent vs target full latent |

Output:

```text
latent_distribution_metrics.csv
```

---

## 16. KMeans Cancer Type Metrics

KMeans clustering should be performed on sample-level latent.

Ground truth label:

```text
cancer_type
```

Main analysis:

```text
source + target combined latent
```

Metrics:

| Metric |
|---|
| ARI |
| NMI |
| Silhouette score |
| Calinski-Harabasz score |
| Davies-Bouldin score |

KMeans cluster number:

```text
n_clusters = number of unique cancer types
```

Handling of `Unknown` cancer type should be configurable.

---

## 17. Prediction Metrics

## 17.1 Classification Metrics

For classification run:

Per-drug metrics:

| Metric |
|---|
| AUC |
| AUPR |
| Accuracy |
| F1 |
| Balanced accuracy |

Summary metrics:

| Metric Type | Description |
|---|---|
| macro | average over drugs |
| micro | flatten all observed sample-drug positions |
| weighted | weighted by number of observed labels per drug |

---

## 17.2 Regression Metrics

For regression run:

Per-drug metrics:

| Metric |
|---|
| RMSE |
| MAE |
| R2 |
| Pearson |
| Spearman |

Optional binary metrics after threshold:

```text
neg_log2_auc >= 1.0
```

---

## 17.3 Target Metrics

Target is always evaluated as classification.

For both classification and regression runs, target metrics should use binary target labels.

Output:

```text
target_metrics_per_drug.csv
target_metrics_summary.csv
```

---

## 18. Suggested File Structure

Recommended new files:

```text
ssda_multilabel/
  __init__.py
  config.py
  prepare_multilabel_data.py
  multilabel_dataset.py
  multilabel_model.py
  multilabel_loss.py
  multilabel_training.py
  multilabel_prediction.py
  multilabel_metrics.py
  multilabel_export.py
  cancer_type_utils.py
  latent_eval.py
```

Alternative: place these under existing `ssda_latent/` if the project prefers fewer packages.

---

## 19. Output Structure（As-Built，2025-05）

輸出根目錄 = `--output_dir`（或 `--latent_output_dir` 若明確指定）。**不再**建立 `ssda_multilabel/seed_{seed}/` 子目錄；`random_seed` 僅寫入 `config.json` 與 prediction 列。

```text
{output_dir}/
  config.json
  drug_list.csv
  data_alignment_report.csv

  fold_0/
    model_final.pth
    masked_loss_log.csv
    source_latent_representation.pkl    # 完整 source omics 全樣本
    target_latent_representation.pkl    # 完整 target omics 全樣本
    source_prediction_results.csv
    target_prediction_results.csv
    source_metrics_per_drug.csv         # 僅 source_test
    source_metrics_summary.csv
    target_metrics_per_drug.csv
    target_metrics_summary.csv
    latent_distribution_metrics.csv
    kmeans_cancer_type_metrics.csv
    tsne_domain_mixing.png
    tsne_cancer_type.png

  fold_1/
    ...

  source_test_metrics_summary_across_folds.csv
  source_test_metrics_summary_fold_mean_std.csv
  source_test_metrics_per_drug_fold_mean_std.csv
  target_eval_metrics_summary_across_folds.csv
  target_eval_metrics_summary_fold_mean_std.csv
  target_eval_metrics_per_drug_fold_mean_std.csv
  latent_metrics_summary.csv
  kmeans_cancer_type_summary.csv
  kmeans_cancer_type_fold_mean_std.csv
```

### 19.1 刻意不輸出的檔案

以下僅在記憶體／`PreparedData` 內使用，**不寫入磁碟**（減少 I/O 與磁碟占用）：

- `source_response_matrix.csv` / `target_response_matrix.csv`
- 各種 `*_mask.csv`
- `target_nshot_summary.csv`
- `cancer_type_mapping_summary.csv` / `cancer_type_paths_used.csv`

---

## 20. CLI Arguments（As-Built）

入口：`experiment_multilabel_ssda.py`（`ssda_multilabel.config.build_arg_parser`）。

### 20.1 必填

```bash
--task_type classification   # 或 regression
--source_omics_path
--target_omics_path
--source_response_path
--target_response_path
--source_response_col          # classification: Label (GDSC)；regression: neg_log2_auc (PRISM)
```

**不提供的 CLI（程式寫死）：**

| 用途 | 固定欄位 |
|------|----------|
| Source omics ID | `Sample_ID`（或 pretrain 第一欄 `Unnamed: 0`） |
| Target omics ID | `tissue_id`（四段 TCGA，或 pretrain `Unnamed: 0`） |
| Source response sample | `Sample_ID` |
| Target response sample | `Patient_id` |
| Target response label | `Label` |
| Drug | `drug_name` |

`--sample_id_col` / `--response_col` 僅 SUPPRESS 相容舊腳本，新流程勿用。

### 20.2 常用可選

```bash
--random_seed 42
--source_test_size 0.1
--n_splits 5
--n_shot 3
--reg_loss mae              # 預設 mae
--lambda_adapt 0.1
--duplicate_response_strategy mean
--output_dir outputs
--latent_output_dir         # 預設 = output_dir（扁平目錄）
--encoder mlp               # 或 dae
--encoder_h_dims 512,256
--epochs 50
--lr 1e-3
--batch_size 32
--device cuda
```

Cancer type：**預設自動**；僅在自訂表時傳 `--source_cancer_type_path` / `--target_cancer_type_path`（`--cancer_type_col` 預設 `Cancer_type`）。

### 20.3 Smoke（Docker 驗證用）

```bash
cd /workspace/SSDA4Drug-main
PYTHONPATH=. python experiment_multilabel_ssda.py --smoke_test classification
PYTHONPATH=. python experiment_multilabel_ssda.py --smoke_test regression
```

內建路徑見 `experiment_multilabel_ssda.py` 的 `SMOKE_CLASSIFICATION_ARGS` / `SMOKE_REGRESSION_ARGS`（DAPL 掛載 `/workspace/DAPL-master`）。

### 20.4 單藥版勿再使用的參數

`experiment_shot.py` 的 `--drug`、`--method`、`--gene`、`./Datasets/processedData/{drug}/` **不**套用到 multi-label 入口；藥物維度由 `drug_list.csv`（source ∪ target `drug_name`）決定。

---

## 21. Implementation Steps

### P0: Data Preparation

1. Implement long table reader.
2. Generate `drug_list.csv` from source ∪ target drug IDs.
3. Convert source long response table to wide response matrix.
4. Convert target long response table to wide response matrix.
5. Generate source mask.
6. Generate target observed mask.
7. Generate target labeled mask using position-level n-shot.
8. Generate target unlabeled mask.
9. Save all matrix / mask / alignment reports.

---

### P1: Multi-label Model

1. Modify or add multi-output prediction head.
2. Ensure output dimension equals number of drugs.
3. Ensure DAE / MLP encoder latent extraction works.
4. Ensure deterministic latent export.

---

### P2: Masked Loss

1. Implement masked BCEWithLogitsLoss.
2. Implement masked regression loss.
3. Implement target labeled classification loss.
4. Implement target unlabeled entropy/adentropy loss using target unlabeled mask.
5. Make sure validation does not update model.

---

### P3: Training Pipeline

1. Modify `experiment_shot_ssda.py` or create a new entry point.
2. Support `--task_type classification/regression`.
3. Support sample-level source test split.
4. Support sample-level source K-fold CV.
5. Use position-level target n-shot masks.
6. Train one model per fold.
7. Save model checkpoint per fold.

---

### P4: Prediction and Metrics

1. Export source prediction long table.
2. Export target prediction long table.
3. Compute per-drug metrics.
4. Compute summary metrics.
5. Compute regression metrics if `--task_type regression`.
6. Compute target classification metrics.

---

### P5: Latent and Visualization

1. Export source sample-level latent pkl.
2. Export target sample-level latent pkl.
3. Plot t-SNE domain mixing.
4. Plot t-SNE cancer type.
5. Compute FID / MMD / Wasserstein.
6. Compute KMeans cancer type metrics.

---

## 22. Testing and Validation

### 22.0 實作驗證節點（2025-05 已跑通）

| 節點 | 指令 / 檢查 | 通過標準 |
|------|-------------|----------|
| **P0 資料欄位** | `PYTHONPATH=. python scripts/patch_dapl_csv_columns.py --dapl-root /workspace/DAPL-master` | `pretrain_tcga.csv` 第一欄為 `tissue_id`；GDSC/PRISM 有 `Sample_ID` + `drug_name` |
| **P1 Smoke classification** | `python experiment_multilabel_ssda.py --smoke_test classification`（Docker `SSDA`） | exit 0；輸出在 `outputs_smoke_classification/`（無 `seed_42` 子目錄） |
| **P2 Smoke regression** | `--smoke_test regression` | 同上 → `outputs_smoke_regression/` |
| **P3 對齊報告** | `data_alignment_report.csv` | `n_source_samples` / `n_target_samples` 與 omics 列數一致；TCGA join 說明含 `tissue_id -> Patient_id` |
| **P4 Source 指標** | `source_metrics_*.csv` | 僅來自 `split==source_test`（非全 source train+val+test） |
| **P5 Target 指標** | `target_metrics_*.csv` | 含所有 observed target positions；regression run 仍為 AUC/F1 等分類指標 |
| **P6 Latent** | `*_latent_representation.pkl` key 數 | = 完整 omics 樣本數（1128 / 8969 等），非 val 子集 |
| **P7 視覺化** | `kmeans_cancer_type_summary.csv` | ARI/NMI 非全 0；`tsne_cancer_type.png` 非全 Unknown |
| **P8 靜態檢查** | `ruff` / `mypy` / `pytest tests/test_multilabel_*.py` | 容器內通過（見 `docs/implementation_report.md`） |

**常見失敗與修復（心得）：**

1. PRISM 同 `(Sample_ID, drug_name)` 多列 → `--duplicate_response_strategy mean`。
2. regression 時 target `probability=NaN` 導致 AUPR 崩潰 → `metrics._classification_scores` 回退 `pred_score`。
3. `metrics.py` 內 `del domain` 誤刪區域變數 → 改為不 shadow 參數名。
4. KMeans/t-SNE 全 Unknown → 確認未手動關閉 cancer type 自動解析；pretrain 用 `ccle_sample_info_df` / `xena_sample_info_df` 的 `cancer_type` 欄。
5. Docker 路徑 → 專案目錄 `/workspace/SSDA4Drug-main`（舊名 `SSDA4Drug` 需 symlink 或 `-w`）。

### 22.1 Data Alignment Tests

Verify:

1. `drug_list.csv` contains source ∪ target drug IDs.
2. Drug order is identical for source and target matrices.
3. Source-only drugs are retained.
4. Target-only drugs are retained.
5. Matrix columns match `drug_list.csv`.
6. Missing response positions have mask = 0.
7. Observed response positions have mask = 1.

---

### 22.2 Target N-shot Tests

Verify:

1. n-shot is performed per drug.
2. n-shot is performed per class.
3. n-shot returns position-level mask.
4. Target drugs with no labels are skipped.
5. Drugs with fewer than n samples per class do not crash.
6. Target labeled mask and target unlabeled mask do not overlap.
7. `target_labeled_mask + target_unlabeled_mask == target_observed_mask`.

---

### 22.3 Source Split Tests

Verify:

1. source test samples do not appear in train / validation.
2. train and validation are disjoint at sample level.
3. each fold uses only source train_val samples.
4. all source samples are assigned valid split labels.

---

### 22.4 Training Tests

Verify:

1. model output shape is `[batch_size, n_drugs]`.
2. masked classification loss ignores missing labels.
3. masked regression loss ignores missing labels.
4. target labeled loss uses `mask_target_labeled`.
5. target adaptation loss uses `mask_target_unlabeled`.
6. validation does not update model parameters.
7. regression run can use source regression loss and target classification loss.

---

### 22.5 Export Tests

Verify:

1. source latent pkl contains all source samples.
2. target latent pkl contains all target samples.
3. latent dimension is correct.
4. source prediction table contains observed source sample-drug positions.
5. target prediction table contains observed target sample-drug positions.
6. prediction rows include drug ID and drug index.
7. per-drug metrics are produced.
8. summary metrics are produced.

---

## 23. Completion Criteria

This project is complete when:

1. The pipeline supports multi-label / multi-drug training.
2. The pipeline uses raw omics data table as model input.
3. No drug latent input is required.
4. `drug_list.csv` is generated from source ∪ target drug IDs.
5. Source and target response matrices use identical drug order.
6. Missing labels are handled by mask loss.
7. Target n-shot is position-level, per drug, per class.
8. Classification run works.
9. Regression run works.
10. Target remains classification in both task modes.
11. Source split is sample-level.
12. Multi-output prediction head outputs `[batch_size, n_drugs]`.
13. Source and target prediction result tables are exported.
14. Source and target sample-level latent pkl files are exported.
15. t-SNE, FID / MMD / Wasserstein, and KMeans cancer type metrics are exported.
16. Validation does not update model parameters.
17. All outputs are reproducible with a fixed random seed.

---

## 24. Notes for Implementation

1. Do not reuse legacy trainer directly if it updates model parameters during validation.
2. Do not use `Test_Double_Model` if it assumes DAE tuple output only.
3. Do not export stochastic DAE denoising latent.
4. Do not remove drugs that appear only in source or only in target.
5. Do not treat target n-shot as sample-level split.
6. Do not compute loss on missing response positions.
7. Do not use drug latent.
8. Do not convert the model into sample-drug pair model.
9. Do not assume target regression labels exist.
10. Do not assume cancer type exists in the original SSDA input.

---

## 25. Remaining Configurable Decisions

These can be implemented as config / CLI options rather than fixed assumptions:

1. `--reg_loss`: `mse`, `mae`, or `huber`（**已實作，預設 mae**）。
2. `--exclude_unknown_cancer_type_for_kmeans`: true / false（**已實作**）。
3. `--source_test_size`: default 0.1（**已實作**）。
4. `--n_splits`: default 5（**已實作**）。
5. `--n_shot`: target n-shot per drug per class（**已實作**）。
6. `--lambda_adapt`: target adaptation loss weight（**已實作**）。
7. Whether to export wide prediction matrix in addition to long prediction table（**目前僅 long table**）。

---

## 26. As-Built 實作紀錄與 `experiment_shot.py` → `experiment_multilabel_ssda.py` 遷移指南

> **用途：** 僅閱讀本節 + `docs/design.md` §14，應能從頭改寫或對照 `experiment_multilabel_ssda.py`，而不必回溯聊天紀錄。

### 26.1 舊版 `experiment_shot.py` vs 新版入口

| 面向 | `experiment_shot.py`（單藥） | `experiment_multilabel_ssda.py`（多藥） |
|------|---------------------------|--------------------------------------|
| 資料根目錄 | `./Datasets/processedData/{drug}/` | CLI 四個 CSV 路徑（通常 DAPL-master） |
| 藥物維度 | `--drug` 迴圈 50 次 | `drug_list.csv` = sorted(source ∪ target `drug_name`) |
| Omics | `source_scaled{gene}.csv` 轉置後 split | `read_omics_table` + `align_omics_features`（共同 feature） |
| Response | `source_meta_data.csv` 單欄 `response` | long table → wide `[n_samples, n_drugs]` + mask |
| Target | 同 drug 的 target 資料（單藥管線） | 獨立 target omics + response；position-level n-shot |
| Split | `train_test_split` 0.2 做 val | sample-level **固定 test** + K-fold train/val |
| 取樣 | `WeightedRandomSampler` 類別平衡 | `MultiLabelSampleDataset` 逐 sample；mask 決定有效 label |
| 模型 | legacy `model.py` + `trainer.py` | `ssda_multilabel.model.MultiLabelSSDAModel` |
| Head 輸出 | 單輸出 logit | `[batch, n_drugs]` multi-output |
| Val | 舊 trainer 可能更新參數 | `MultiLabelSSDTrainer`：**val 僅 forward** |
| Latent | 單藥、單 domain 習慣 | 每 fold 輸出 **完整** source/target sample-level pkl |
| 輸出目錄 | 依 `para` 字串散落 | `--output_dir` 扁平 + `fold_{k}/` |

**遷移原則：** 不要把 `experiment_shot.py` 內聯改寫；改為薄入口 + `ssda_multilabel/*` 模組，保留 `experiment_shot.py` 給舊實驗。

### 26.2 DAPL 固定欄位與 Join（`sample_id.py`）

```text
Source omics / source response sample key:  Sample_ID  (ACH-*, CCLE)
Target omics sample key:                    tissue_id    (TCGA 四段，例 TCGA-AA-3695-01A)
Target response sample key:                   Patient_id   (TCGA 三段，例 TCGA-AA-3695)
Target response label:                        Label        (0/1，程式寫死 target_response_col)
Drug:                                         drug_name
```

**Target join：** `sample_match_key(tissue_id)` → `tcga_patient_key`（三段）再對齊 `Patient_id`。

**資料前處理腳本：** `scripts/patch_dapl_csv_columns.py`

- PRISM：`depmap_id` → `Sample_ID`，補 `drug_name`
- pretrain CCLE：第一欄 → `Sample_ID`
- pretrain TCGA：第一欄 → `tissue_id`
- GDSC：`ModelID` → `Sample_ID`（若缺）
- Winnie TCGA impact：正規化為四段 `tissue_id`

### 26.3 正式資料路徑（DAPL，與 smoke 一致）

**Classification**

```text
source_omics:   DAPL-master/data/pretrain_ccle.csv
target_omics:   DAPL-master/data/TCGA/pretrain_tcga.csv
source_response: DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv  (--source_response_col Label)
target_response: DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv
```

> 舊檔 `GDSC2_fitted_dose_response_27Oct23 from GDSC MaxScreen threshold ...csv` 請改用上列 `*_MaxScreen_raw.csv`（`patch_dapl_csv_columns.py` 亦會處理兩者）。

**Classification 正式指令（Docker `SSDA`）**

```bash
# 一次性：標準化 DAPL 欄位（Sample_ID / tissue_id / drug_name）
docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug-main && PYTHONPATH=. python scripts/patch_dapl_csv_columns.py --dapl-root /workspace/DAPL-master'

docker exec SSDA bash -lc 'cd /workspace/SSDA4Drug-main && PYTHONPATH=. python experiment_multilabel_ssda.py \
  --task_type classification \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --source_response_path /workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv \
  --target_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv \
  --source_response_col Label \
  --random_seed 42 --n_splits 5 --n_shot 3 --epochs 50 \
  --output_dir outputs_classification'
```

輸出目錄：`outputs_classification/`（扁平，直接含 `fold_0/`…，無 `ssda_multilabel/seed_42/`）。

**Regression（Winnie impact + PRISM）**

```text
source_omics:   data_Winnie/CCLE_impact_hotspot.csv
target_omics:   data_Winnie/TCGA_impact_hotspot.csv
source_response: data_Winnie/PRISM_drug_sensitivity.csv  (--source_response_col neg_log2_auc)
target_response: 同上 TCGA PMID 表
```

### 26.4 `experiment_multilabel_ssda.py` 編排（對照改寫）

```python
# 1. parse CLI / smoke_test 展開 SMOKE_*_ARGS
config = config_from_args(args)
set_global_seed(config.random_seed)

# 2. 輸出
writer = ArtifactWriter(config.latent_output_dir, config.random_seed)
writer.write_config(config)

# 3. 資料（唯一大塊邏輯在 prepare.py）
prepared = prepare_multilabel_data(config)
writer.write_preparation_artifacts(drug_index, alignment_report)

# 4. 每 fold
for fold in prepared.folds:
    model = build_model(n_features, n_drugs, ...)
    trainer.train_fold(prepared, fold)          # 訓練 + val，不更新 val
    predict_matrix → long tables + metrics
    encode_latent_dict(完整 so.x / to.x) → pkl
    t-SNE / FID / MMD / KMeans

# 5. 跨 fold 彙總 CSV（reports.py）
```

**Source 指標過濾：** `_filter_source_test(pred)` 只保留 `split == "source_test"`。

**Target 指標：** 不過濾 split；所有 `mask=1` 的 observed positions 都進 metrics。

### 26.5 實際套件配置（`ssda_multilabel/`）

| 模組 | 職責 |
|------|------|
| `config.py` | CLI、`MultiLabelConfig`、扁平 `resolve_multilabel_output_dir` |
| `prepare.py` | `prepare_multilabel_data` 總控 |
| `omics_io.py` | omics 讀取、feature 交集 |
| `response_matrix.py` | long → wide + duplicate 策略 |
| `masks.py` | target n-shot labeled / unlabeled |
| `split.py` | source test + K-fold |
| `sample_id.py` | TCGA key、join |
| `drug_index.py` / `drug_normalize.py` | drug_list |
| `cancer_type.py` | 自動路徑 + mapping |
| `model.py` / `losses.py` / `adaptation.py` / `training.py` | 訓練 |
| `prediction.py` / `metrics.py` | 預測與評估 |
| `latent.py` / `latent_eval.py` | pkl、t-SNE、FID/MMD、KMeans |
| `export.py` | `ArtifactWriter` |
| `reports.py` | 跨 fold mean/std |

### 26.6 訓練損失（每 epoch）

```text
L = L_source_sup(mask_source, train∪val indices 依 fold)
  + L_target_sup(mask_target_labeled)
  + lambda_adapt * L_adentropy(mask_target_unlabeled)
```

- **classification：** source/target supervised 用 masked BCE；target unlabeled 用 sigmoid entropy adaptation。
- **regression：** source 用 masked MAE/MSE/Huber；target labeled 仍 BCE；target unlabeled 仍 adaptation。

### 26.7 實作心得（避免重踩）

1. **不要**在 validation step 呼叫會 `backward` 的舊 `trainer.py`。
2. **不要**用 DAE 隨機 denoising forward 匯出 latent；用 `model.encode(..., deterministic=True)`。
3. **不要**假設 target 有 regression 連續標籤；target 指標永遠走 classification。
4. **不要**把 target n-shot 做成 sample-level split（會錯殺其他 drug 的 label）。
5. **不要**在磁碟重複輸出巨大 wide matrix（已在記憶體即可）。
6. Cancer type 與 omics ID 型態不同時（pretrain `Unnamed: 0` vs `Sample_ID`），靠 `cancer_type.py` profile 對應，勿硬編一種欄位名到所有表。
7. Docker 內用 `python`（有 pandas/torch），`python3` 可能是系統精簡版。

### 26.8 完成定義（本階段）

當 §22.0 全部節點通過，且 §23 Completion Criteria 1–17 滿足時，視為 `experiment_multilabel_ssda.py` 本階段完成；後續若改 `experiment_shot.py` 行為，應先更新本節與 `design.md` §14 再動程式。
