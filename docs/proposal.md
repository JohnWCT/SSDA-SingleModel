# Proposal: Convert SSDA-SingleModel to Multi-label / Multi-drug SSDA Framework

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

---

## 3. Scope

### 3.1 In Scope

本次修改包含：

1. 將 `experiment_shot_ssda.py` 改為 multi-label / multi-drug 任務入口。
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

範例：

| Sample_ID | gene_1 | gene_2 | gene_3 | ... |
|---|---:|---:|---:|---:|
| S1 | 0.12 | 0.45 | 0.78 | ... |
| S2 | 0.33 | 0.21 | 0.44 | ... |

需求：

1. 第一欄或指定欄位為 sample ID。
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

建議欄位：

| 欄位 | 說明 |
|---|---|
| sample_id | source sample ID |
| drug_id | drug ID |
| response | response value |
| cancer_type | optional |
| original_drug_name | optional |

classification task 時，`response` 應為 0/1。

regression task 時，`response` 應為 continuous value，例如 `neg_log2_auc`。

---

## 4.3 Target Response Long Table

target response 也使用 long table 作為標準輸入。

```text
target_response.csv
```

建議欄位：

| 欄位 | 說明 |
|---|---|
| sample_id | target sample ID |
| drug_id | drug ID |
| response | binary 0/1 target response |
| cancer_type | optional |
| original_drug_name | optional |

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

建議新增：

```text
ssda_multilabel/
  prepare_multilabel_data.py
```

或放在現有 package 內：

```text
ssda_latent/
  multilabel_data.py
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

## 19. Suggested Output Structure

```text
save/
  ssda_multilabel/
    seed_{random_seed}/
      config.json
      drug_list.csv
      data_alignment_report.csv
      source_response_matrix.csv
      source_response_mask.csv
      target_response_matrix.csv
      target_observed_mask.csv
      target_labeled_mask.csv
      target_unlabeled_mask.csv
      target_nshot_summary.csv
      cancer_type_mapping_summary.csv

      fold_0/
        model_final.pth
        source_latent_representation.pkl
        target_latent_representation.pkl
        source_prediction_results.csv
        target_prediction_results.csv
        source_metrics_per_drug.csv
        source_metrics_summary.csv
        target_metrics_per_drug.csv
        target_metrics_summary.csv
        masked_loss_log.csv
        latent_distribution_metrics.csv
        kmeans_cancer_type_metrics.csv
        tsne_domain_mixing.png
        tsne_cancer_type.png

      fold_1/
        ...

      fold_2/
        ...

      fold_3/
        ...

      fold_4/
        ...

      metrics_summary.csv
      latent_metrics_summary.csv
      kmeans_cancer_type_summary.csv
```

---

## 20. CLI Arguments

Suggested CLI arguments:

```bash
--task_type classification
--task_type regression

--source_omics_path
--target_omics_path
--source_response_path
--target_response_path

--sample_id_col Sample_ID
--drug_id_col drug_id
--response_col response

--source_cancer_type_path
--target_cancer_type_path
--cancer_type_col cancer_type

--random_seed 42
--source_test_size 0.1
--n_splits 5
--n_shot 3

--reg_loss mse
--lambda_adapt 0.1
--latent_output_dir save/ssda_multilabel
```

Retain relevant original SSDA arguments:

```bash
--encoder
--encoder_h_dims
--epochs
--lr
--batch_size
--result
--sc_all
```

Remove or avoid drug-specific single-task arguments if no longer applicable:

```bash
--drug
```

If `--drug` must be retained for backward compatibility, it should not control model output dimension in multi-label mode.

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

1. `--reg_loss`: `mse`, `mae`, or `huber`.
2. `--exclude_unknown_cancer_type_for_kmeans`: true / false.
3. `--source_test_size`: default 0.1.
4. `--n_splits`: default 5.
5. `--n_shot`: target n-shot per drug per class.
6. `--lambda_adapt`: target adaptation loss weight.
7. Whether to export wide prediction matrix in addition to long prediction table.
