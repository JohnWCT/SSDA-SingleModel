# Proposal: Convert CODE-AE from Single-drug Single-model to Multi-label / Multi-drug Single-model Framework

> **目標文件層級：** 本文件是高層 proposal，用於後續討論 `design_codeae.md`。因此本文件只定義需求邊界、架構方向、資料流、訓練流程、輸出與驗證原則；不在此階段展開完整 class/function-level implementation design。

---

## 1. Project Goal

本 proposal 目標是將目前 CODE-AE workflow：

```bash
cd benchmark/CODEAE/
python pretrain_hyper_main.py --drug "Gefitinib"
python drug_ft_hyper_main.py --drug "Gefitinib"
```

從 **single-drug single-model** 改造成 **multi-label / multi-drug single-model** framework。

新版 CODE-AE 不再針對單一 drug 個別訓練一個模型，而是：

```text
omics data table
  -> CODE-AE encoder / deconfounding module
  -> sample-level latent representation
  -> multi-output drug prediction head
  -> all-drug response vector [n_drugs]
```

也就是每個 sample 一次輸出所有 drugs 的 response prediction。

本修改應保留 CODE-AE 的核心精神：

1. 保留 CODE-AE 的 two-stage workflow：pretrain encoder → drug response fine-tuning。
2. 保留 CODE-AE 的 encoder / context-aware deconfounding / adversarial training 主架構。
3. 盡量只替換 single-drug prediction head 為 multi-output prediction head。
4. 不改成 sample-drug pair model。
5. 不使用 drug latent。
6. 不讓 domain adaptation loss 變成 drug-specific；domain adaptation 仍是 sample-level shared adaptation。
7. 訓練完成後輸出 CODE-AE encoder 產生的 sample-level latent representation。

---

## 2. Key Design Decisions

| 項目 | 決議 |
|---|---|
| 文件風格 | 參考既有 `docs/proposal.md` 的高層 proposal 寫法，但針對 CODE-AE 重新設計章節 |
| 新入口 | 新增 multilabel CODE-AE 入口，不修改成單藥 backward-compatible 入口 |
| workflow | 保留 two-stage：pretrain → fine-tune |
| pretrain 階段 | 不使用 drug response；只使用 source / target omics 做 CODE-AE encoder / deconfounding pretraining |
| fine-tune 階段 | 載入 pretrain checkpoint，替換或初始化 multi-output drug predictor head |
| model type | shared encoder + multi-output prediction head |
| output dimension | `n_drugs = len(drug_list.csv)` |
| prediction output | `[batch_size, n_drugs]` |
| task type | 每次 run 使用 `--task_type classification` 或 `--task_type regression` |
| source / target 資料 | 對齊 multilabel SSDA：CCLE/GDSC/PRISM → TCGA |
| response input | source / target response 使用 long table |
| drug list | 由 source / target response table 自動抽取 drug union |
| source-only / target-only drugs | 保留，不刪除 |
| missing label | 使用 mask loss |
| classification loss | masked `BCEWithLogitsLoss` |
| regression source loss | 優先沿用 CODE-AE 原本 regression loss；若原本無明確設定，預設採 MAE |
| regression target label | target 在 regression run 仍轉成 classification 問題 |
| regression threshold | 固定使用 `neg_log2_auc >= 1.0` 轉 binary responder |
| early stopping | 使用 validation macro average prediction metric |
| class weight | 不新增 class weighting；若 CODE-AE 原本有 positive / negative class weight，才沿用 |
| n-shot | CODE-AE multilabel 不考慮 n-shot |
| domain adaptation | 共用 sample-level adaptation，不依 drug 區分 |
| latent output | 訓練完成後才輸出 final CODE-AE latent |
| t-SNE | 訓練完成、latent export 後才產生 |
| backward compatibility | 不服務原 single-drug CLI；新版只服務 multilabel task |
| output structure | 預設 `{output_dir}/fold_0/...` |
| K-fold | 保留 fold-level training / fold summary，方便與 multilabel SSDA 比較 |

---

## 3. Scope

### 3.1 In Scope

本次 proposal 涵蓋：

1. 新增 CODE-AE multilabel pretrain 入口。
2. 新增 CODE-AE multilabel fine-tune 入口。
3. 保留 CODE-AE 原 two-stage training design。
4. 將 single-drug predictor 改成 multi-output drug predictor。
5. 將 source / target response long table 轉成 wide response matrix + mask tensor。
6. 自動建立 `drug_list.csv`。
7. 支援 classification run。
8. 支援 regression run。
9. classification 使用 masked BCE loss。
10. regression source 使用 masked regression loss；target 仍以 classification label 參與 target supervised / evaluation。
11. 使用 macro average validation metric 做 early stopping。
12. 輸出 per-drug metrics 與 macro / micro / weighted summary metrics。
13. 輸出 source / target prediction result tables。
14. 輸出 CODE-AE 訓練完成後的 source / target sample-level latent representation。
15. latent export 後輸出 t-SNE domain mixing / cancer type visualization。
16. 提供 classification / regression smoke test 規劃。

### 3.2 Out of Scope

本次 proposal 不包含：

1. 不維持原本 `--drug Gefitinib` 單藥 CLI backward compatibility。
2. 不直接修改 `experiment_multilabel_ssda.py`；只複用其資料處理與輸出設計思想。
3. 不導入 sample-drug pair model。
4. 不導入 drug latent。
5. 不做 drug-specific domain adaptation。
6. 不把 CODE-AE encoder / adversarial deconfounding 主架構重寫成另一個模型。
7. 不處理 n-shot 設計。
8. 不在 proposal 階段定義完整 function/class API；此部分留待 `design_codeae.md`。

---

## 4. Proposed Entry Points

為保留 CODE-AE 原本 two-stage workflow，建議新增兩個 multilabel 入口：

```text
benchmark/CODEAE/pretrain_multilabel_hyper_main.py
benchmark/CODEAE/drug_ft_multilabel_hyper_main.py
```

### 4.1 Pretrain Entry

```bash
python benchmark/CODEAE/pretrain_multilabel_hyper_main.py \
  --source_omics_path path/to/source_omics.csv \
  --target_omics_path path/to/target_omics.csv \
  --method code_adv \
  --output_dir outputs_codeae_multilabel
```

Pretrain 階段負責：

1. 讀取 source / target omics。
2. 對齊 source / target features。
3. 執行 CODE-AE 原本 encoder / deconfounding / adversarial pretraining。
4. 輸出 pretrain checkpoint。
5. 不讀取 drug response。
6. 不產生 prediction metrics。

### 4.2 Fine-tune Entry

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --task_type classification \
  --source_omics_path path/to/source_omics.csv \
  --target_omics_path path/to/target_omics.csv \
  --source_response_path path/to/source_response.csv \
  --target_response_path path/to/target_response.csv \
  --source_sample_col Sample_ID \
  --target_sample_col tissue_id \
  --target_response_sample_col Patient_id \
  --drug_col drug_name \
  --source_response_col Label \
  --target_response_col Label \
  --pretrain_checkpoint outputs_codeae_multilabel/pretrain/checkpoint.pt \
  --output_dir outputs_codeae_multilabel
```

Fine-tune 階段負責：

1. 載入 pretrain checkpoint。
2. 建立 `drug_list.csv`。
3. 將 source / target long response table 轉成 wide matrix + mask。
4. 初始化或替換 multi-output prediction head。
5. 執行 multilabel CODE-AE fine-tuning。
6. 使用 macro average validation metric 做 early stopping。
7. 訓練完成後輸出 prediction tables、metrics、latent、t-SNE。

---

## 5. Input Design

## 5.1 Omics Input

新版 CODE-AE multilabel pipeline 使用 source / target omics data table：

```text
source_omics.csv
target_omics.csv
```

建議格式：

| sample_id | gene_1 | gene_2 | ... |
|---|---:|---:|---:|
| sample_A | 0.12 | -0.43 | ... |
| sample_B | 0.33 | 0.10 | ... |

需求：

1. source / target omics 必須是 sample-level feature table。
2. source / target feature 必須對齊。
3. 若 feature 不完全相同，取共同 features，並輸出被移除 features 的紀錄。
4. sample ID normalization 必須清楚記錄。
5. target omics 若使用 TCGA `tissue_id` 四段位，需能轉換或 join 到 target response 的 patient key。

## 5.2 Source Response Long Table

source response 使用 long table：

```text
source_response.csv
```

建議欄位：

| 欄位 | 說明 |
|---|---|
| `--source_sample_col` | source sample ID，例如 `Sample_ID` |
| `--drug_col` | drug 名稱，例如 `drug_name` |
| `--source_response_col` | response label 或 continuous value |

classification run：

```text
source_response_col = binary 0/1 label
```

regression run：

```text
source_response_col = continuous value，例如 neg_log2_auc
```

## 5.3 Target Response Long Table

 target response 也使用 long table：

```text
target_response.csv
```

建議欄位：

| 欄位 | 說明 |
|---|---|
| `--target_response_sample_col` | target response sample ID，例如 `Patient_id` |
| `--drug_col` | drug 名稱，例如 `drug_name` |
| `--target_response_col` | target classification label，通常為 `Label` |

target 在本 proposal 中一律視為 classification label。

也就是：

1. classification run：target 為 0/1 label。
2. regression run：target 仍為 0/1 label。
3. regression run 的 target supervised loss 與 target evaluation 仍使用 classification logic。

## 5.4 Drug List

新版必須輸出並使用：

```text
drug_list.csv
```

產生方式：

```text
drug_list = sorted(unique(source_response[drug_col] ∪ target_response[drug_col]))
```

`drug_list.csv` 格式：

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
5. source-only drug 與 target-only drug 都保留。

## 5.5 Cancer Type Metadata

Cancer type metadata 為可選輸入，用於 t-SNE cancer type visualization 與 clustering metrics。

建議 CLI：

```text
--source_cancer_type_path optional/path.csv
--target_cancer_type_path optional/path.csv
```

若未提供：

1. cancer type 欄位可標記為 `Unknown`。
2. 不應阻止 CODE-AE multilabel training。
3. t-SNE domain mixing 仍可輸出。
4. cancer type t-SNE / clustering metrics 可跳過或輸出 warning。

---

## 6. Data Preparation Design

需要新增 CODE-AE 專用 multilabel data preparation 模組，概念上對齊 `experiment_multilabel_ssda.py` 的資料處理方式，但不直接耦合 SSDA training code。

建議模組位置：

```text
benchmark/CODEAE/codeae_multilabel/prepare.py
```

### 6.1 Responsibilities

此模組負責：

1. 讀取 source omics。
2. 讀取 target omics。
3. 對齊 source / target omics features。
4. 讀取 source response long table。
5. 讀取 target response long table。
6. 建立 `drug_list.csv`。
7. 將 source response long table 轉成 wide response matrix。
8. 將 target response long table 轉成 wide response matrix。
9. 建立 source mask matrix。
10. 建立 target observed mask matrix。
11. 建立 sample-level source train / validation / test split。
12. 建立 fold-level train / validation split。
13. 輸出資料對齊報告。
14. 輸出 missing label 統計報告。

### 6.2 Long Table to Wide Matrix

source long table 轉換為：

```text
Y_source:      [n_source_samples, n_drugs]
mask_source:   [n_source_samples, n_drugs]
```

target long table 轉換為：

```text
Y_target:      [n_target_samples, n_drugs]
mask_target:   [n_target_samples, n_drugs]
```

其中：

```text
mask = 1 表示該 sample-drug position 有 response label
mask = 0 表示 missing
```

若某 sample 對某 drug 沒有 response：

```text
Y[sample, drug] = arbitrary value, e.g. 0
mask[sample, drug] = 0
```

所有 supervised loss 與 metrics 都必須只使用 `mask == 1` 的 positions。

### 6.3 Duplicate Response Handling

若同一個 `(sample_id, drug_id)` 有多筆 response，需提供 aggregation strategy。

建議預設：

```text
--duplicate_response_strategy mean
```

可在後續 `design_codeae.md` 中決定是否支援：

```text
mean / median / first / error
```

---

## 7. Model Design

## 7.1 Overall Architecture

新版 CODE-AE multilabel architecture：

```text
omics features
  -> CODE-AE encoder / deconfounding module
  -> sample latent
  -> multi-output prediction head
  -> response vector [n_drugs]
```

### 7.2 Encoder

保留 CODE-AE 原本 encoder 設計。

要求：

1. 不為 multilabel task 重寫 encoder 主架構。
2. 不移除 CODE-AE 原有 deconfounding / adversarial components。
3. 若 latent export 時 encoder 有 dropout / denoising / stochastic behavior，export 必須使用 deterministic mode。

### 7.3 Multi-output Prediction Head

將原 single-drug prediction head 改成：

```text
prediction_head: latent_dim -> n_drugs
```

分類任務：

```text
logits: [batch_size, n_drugs]
```

regression 任務：

```text
scores: [batch_size, n_drugs]
```

每個 column 對應 `drug_list.csv` 中的 drug order。

### 7.4 Domain Adaptation / Deconfounding

domain adaptation / deconfounding loss 維持 CODE-AE 原本 sample-level 設計。

不新增：

1. drug-specific adversarial discriminator。
2. drug-specific encoder。
3. sample-drug pair latent。
4. drug embedding / drug latent。

---

## 8. Task Type Design

每次 run 指定一種 task：

```bash
--task_type classification
```

或：

```bash
--task_type regression
```

## 8.1 Classification Run

classification run 中：

| Domain | Label Type | Supervised Loss |
|---|---|---|
| source | 0/1 | masked BCEWithLogitsLoss |
| target | 0/1 | masked BCEWithLogitsLoss, if target labels are used during fine-tuning |

prediction：

```text
probability = sigmoid(logit)
pred_label = probability >= 0.5
```

### 8.1.1 Classification Total Loss

高層概念：

```text
loss_total = source_prediction_loss
           + optional target_prediction_loss
           + original CODE-AE deconfounding/adversarial losses
           + optional reconstruction loss if used by original CODE-AE
```

具體 loss weights 應沿用 CODE-AE 原本 training strategy，除非 multilabel head 需要額外 normalization。

## 8.2 Regression Run

regression run 中：

| Domain | Label Type | Usage |
|---|---|---|
| source | continuous | masked regression loss |
| target | 0/1 classification | target supervised / evaluation uses classification logic |

模型仍輸出：

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

### 8.2.1 Regression Threshold

source continuous response 轉 binary responder 使用固定門檻：

```text
neg_log2_auc >= 1.0 -> responder = 1
neg_log2_auc < 1.0  -> responder = 0
```

用途：

1. source optional binary metrics。
2. target classification evaluation 對齊。
3. 與 multilabel SSDA 的 regression target classification 設計對齊。

### 8.2.2 Regression Total Loss

高層概念：

```text
loss_total = source_regression_loss
           + optional target_classification_loss
           + original CODE-AE deconfounding/adversarial losses
           + optional reconstruction loss if used by original CODE-AE
```

source regression loss：

1. 優先沿用 CODE-AE 原本 regression fine-tuning loss。
2. 若原本沒有明確 regression loss，預設使用 masked MAE。
3. 可在後續 design 階段討論是否支援 `mae / mse / huber`。

---

## 9. Masked Loss Design

## 9.1 Classification Masked BCE Loss

```python
raw_loss = BCEWithLogitsLoss(reduction="none")(logits, y)
masked_loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1)
```

要求：

1. `logits.shape == [batch_size, n_drugs]`
2. `y.shape == [batch_size, n_drugs]`
3. `mask.shape == [batch_size, n_drugs]`
4. missing labels 不參與 loss。

## 9.2 Regression Masked Loss

概念：

```python
raw_loss = regression_loss_fn(pred, y, reduction="none")
masked_loss = (raw_loss * mask).sum() / mask.sum().clamp_min(1)
```

要求：

1. 只在 source observed continuous labels 上計算 regression loss。
2. target 若以 classification label 參與 supervised training，使用 masked BCE，而不是 regression loss。
3. mask denominator 必須使用 observed position 數量，避免 label density 不同造成 loss scale 偏差。

## 9.3 Class Weight

本 proposal 不主動新增 positive / negative class weighting。

規則：

1. 若 CODE-AE 原始 fine-tune 已有 class weight，則 multilabel 版本沿用相同思想。
2. 若 CODE-AE 原始 fine-tune 沒有 class weight，則 multilabel 版本也不新增。
3. per-drug class imbalance handling 留待後續 design 或 ablation 討論。

---

## 10. Training Workflow

## 10.1 Stage 1: CODE-AE Multilabel Pretraining

Pretrain input：

```text
source_omics.csv
target_omics.csv
```

Pretrain 不使用：

```text
source_response.csv
target_response.csv
drug_list.csv
```

Pretrain output：

```text
pretrain_checkpoint.pt
pretrain_config.json
feature_alignment_report.csv
```

Pretrain 階段保留 CODE-AE 原本：

1. encoder training。
2. deconfounding / adversarial learning。
3. reconstruction objective if applicable。
4. method-specific behavior such as `code_adv`, `code_base`, `code_mmd`, etc. if supported by original CODE-AE implementation。

## 10.2 Stage 2: Multilabel Drug Fine-tuning

Fine-tune input：

```text
source_omics.csv
target_omics.csv
source_response.csv
target_response.csv
pretrain_checkpoint.pt
```

Fine-tune steps：

1. Load aligned omics features。
2. Build `drug_list.csv` from source ∪ target response drugs。
3. Build source / target wide response matrices。
4. Build source / target mask matrices。
5. Load CODE-AE pretrain checkpoint。
6. Replace single-drug head with multi-output head。
7. Train with masked supervised loss + original CODE-AE losses。
8. Validate by macro average metric。
9. Save best checkpoint。
10. After final training, export latent and t-SNE。

## 10.3 Fold-level Training

建議保留 fold-level training / summary：

```text
source samples
  ├── source_test
  └── source_train_val
        ├── fold_0_train / fold_0_val
        ├── fold_1_train / fold_1_val
        └── ...
```

要求：

1. source split 以 sample 為單位。
2. 同一個 sample 不可同時出現在同一 fold 的 train 與 validation。
3. source test 不可出現在任何 fold train / validation。
4. fold split 使用固定 random seed。
5. validation 不更新模型參數。

---

## 11. Early Stopping and Model Selection

多藥物 fine-tuning 使用 macro average validation metric 作為 early stopping 依據。

### 11.1 Classification

建議 primary early stopping metric：

```text
validation_macro_auroc
```

若某些 validation fold 中可計算 AUROC 的 drug 太少，可 fallback：

```text
validation_macro_aupr
validation_macro_balanced_accuracy
validation_macro_f1
```

fallback 規則應在 `design_codeae.md` 中明確定義。

### 11.2 Regression

source regression validation 可使用：

```text
validation_macro_rmse
```

或若沿用 CODE-AE 原本 metric：

```text
validation_macro_original_metric
```

但 target evaluation 仍以 classification metrics 為主。

### 11.3 Selection Principle

本 proposal 採用：

```text
early stopping 以 validation macro prediction metric 為主，
不以 adversarial loss 或 reconstruction loss 作為主要 model selection metric。
```

原因：

1. 本任務核心是 multilabel drug response prediction。
2. adversarial / reconstruction loss 仍參與訓練，但不直接代表 drug response predictive performance。
3. macro average 可避免模型只偏向 label 數量最多的 drug。

---

## 12. Metrics Design

## 12.1 Per-drug Metrics

每個 drug 單獨計算 metrics。

classification metrics：

```text
AUROC
AUPR
Accuracy
F1
Precision
Recall
Balanced Accuracy
```

regression source metrics：

```text
MAE
RMSE
R2
Pearson
Spearman
```

target metrics in regression run：

```text
AUROC
AUPR
Accuracy
F1
Precision
Recall
Balanced Accuracy
```

## 12.2 Summary Metrics

輸出：

```text
macro average
micro average
weighted average
```

其中：

1. macro average：每個 drug 權重相同。
2. micro average：所有 observed sample-drug positions 合併後計算。
3. weighted average：依每個 drug 的 observed label 數量加權。

## 12.3 Missing / Invalid Metric Handling

某 drug 若 validation 或 test 中只有單一 class，AUROC / AUPR 可能不可計算。

處理原則：

1. per-drug metric 記為 `NaN`。
2. summary metric 計算時排除該 metric 的 invalid drugs。
3. 輸出 warning / metric availability report。
4. 不因單一 drug metric invalid 中斷整個 run。

---

## 13. Prediction Output Design

雖然模型 output 是 wide matrix，prediction output 應輸出成 long table，方便分析與和 multilabel SSDA 對齊。

## 13.1 Source Prediction Table

輸出檔：

```text
source_prediction_results.csv
```

每列代表一個 observed source sample-drug position。

建議欄位：

| Column | Description |
|---|---|
| `sample_id` | source sample ID |
| `drug_id` | drug name / ID |
| `drug_index` | output column index |
| `domain` | source |
| `split` | source_fold_train / source_fold_val / source_test |
| `ground_truth` | observed response |
| `mask` | should be 1 |
| `pred_score` | raw model score |
| `probability` | sigmoid probability if classification-style output |
| `pred_label` | binary prediction if applicable |
| `task_type` | classification / regression |
| `fold` | fold index |
| `seed` | random seed |
| `cancer_type` | optional |

## 13.2 Target Prediction Table

輸出檔：

```text
target_prediction_results.csv
```

每列代表一個 observed target sample-drug position。

建議欄位：

| Column | Description |
|---|---|
| `sample_id` | target sample ID |
| `drug_id` | drug name / ID |
| `drug_index` | output column index |
| `domain` | target |
| `ground_truth` | observed target response |
| `mask` | should be 1 |
| `pred_score` | raw model score / logit |
| `probability` | sigmoid probability |
| `pred_label` | probability >= 0.5 |
| `task_type` | classification / regression |
| `fold` | fold index |
| `seed` | random seed |
| `cancer_type` | optional |

---

## 14. Latent Output Design

Latent representation 必須是 sample-level，不是 sample-drug-level。

每個 fold 訓練完成後輸出：

```text
source_latent_representation.pkl
target_latent_representation.pkl
```

建議格式：

```python
{
  "sample_id_1": [latent_0, latent_1, ..., latent_k],
  "sample_id_2": [latent_0, latent_1, ..., latent_k],
}
```

### 14.1 Latent Export Requirements

1. latent 必須在 CODE-AE fine-tuning 完成後輸出。
2. latent 必須來自 final / best checkpoint 的 encoder。
3. latent 必須是 domain-adapted representation。
4. latent export 必須使用 deterministic mode。
5. source latent 包含完整 source omics samples。
6. target latent 包含完整 target omics samples。
7. latent 不展開成 sample-drug positions。

---

## 15. t-SNE and Latent Evaluation

latent export 完成後，每個 fold 可輸出：

```text
tsne_domain_mixing.png
tsne_cancer_type.png
latent_distribution_metrics.csv
kmeans_cancer_type_metrics.csv
```

### 15.1 t-SNE Domain Mixing

使用 combined source + target latent。

Color by：

```text
source / target
```

### 15.2 t-SNE Cancer Type

使用 combined source + target latent。

Color by：

```text
cancer_type
```

若 cancer type metadata 缺失：

1. 標記為 `Unknown`；或
2. 跳過 cancer type t-SNE 並輸出 warning。

### 15.3 Important Timing Rule

latent representation 與 t-SNE 必須在 CODE-AE 完成 fine-tuning 後輸出。

不可在 pretrain 後就把 latent 當成 final multilabel CODE-AE latent。

---

## 16. Output Directory Design

建議輸出結構：

```text
{output_dir}/
  config.json
  drug_list.csv
  feature_alignment_report.csv
  data_alignment_report.csv
  pretrain/
    checkpoint.pt
    pretrain_log.csv
  fold_0/
    best_model.pt
    train_log.csv
    source_prediction_results.csv
    target_prediction_results.csv
    source_metrics_per_drug.csv
    target_metrics_per_drug.csv
    source_metrics_summary.csv
    target_metrics_summary.csv
    source_latent_representation.pkl
    target_latent_representation.pkl
    tsne_domain_mixing.png
    tsne_cancer_type.png
    latent_distribution_metrics.csv
    kmeans_cancer_type_metrics.csv
  fold_1/
    ...
  fold_summary.csv
```

要求：

1. 預設直接寫入 `--output_dir`。
2. 不再巢狀 single-drug drug name 目錄。
3. checkpoint 命名可以保留 CODE-AE method / fold / seed 資訊。
4. 所有輸出都應記錄 `fold`、`seed`、`task_type`、`n_drugs`。

---

## 17. CLI Design

## 17.1 Required Arguments

```text
--task_type classification|regression
--source_omics_path
--target_omics_path
--source_response_path
--target_response_path
--drug_col
--source_response_col
--target_response_col
--output_dir
```

## 17.2 Sample / Join Arguments

```text
--source_sample_col
--target_sample_col
--target_response_sample_col
```

建議預設：

```text
--source_sample_col Sample_ID
--target_sample_col tissue_id
--target_response_sample_col Patient_id
```

## 17.3 CODE-AE Training Arguments

應保留 CODE-AE 原本重要訓練參數，例如：

```text
--method
--train / --no-train
--norm / --no-norm
--metric
--epochs
--batch_size
--lr
--dropout
--seed
```

其中 `--metric` 在 multilabel fine-tune 中應解釋為：

```text
macro average validation metric for early stopping
```

## 17.4 Multilabel-specific Arguments

```text
--duplicate_response_strategy mean
--reg_loss mae
--source_test_size 0.1
--n_splits 5
--prediction_threshold 0.5
--regression_binary_threshold 1.0
--source_cancer_type_path optional
--target_cancer_type_path optional
```

---

## 18. Example Commands

## 18.1 Pretrain

```bash
python benchmark/CODEAE/pretrain_multilabel_hyper_main.py \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --method code_adv \
  --train \
  --norm \
  --output_dir outputs_codeae_multilabel
```

## 18.2 Classification Fine-tune

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --task_type classification \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --target_omics_path /workspace/DAPL-master/data/TCGA/pretrain_tcga.csv \
  --source_response_path /workspace/DAPL-master/data/GDSC2_fitted_dose_response_MaxScreen_raw.csv \
  --target_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv \
  --source_sample_col Sample_ID \
  --target_sample_col tissue_id \
  --target_response_sample_col Patient_id \
  --drug_col drug_name \
  --source_response_col Label \
  --target_response_col Label \
  --pretrain_checkpoint outputs_codeae_multilabel/pretrain/checkpoint.pt \
  --method code_adv \
  --metric macro_auroc \
  --n_splits 5 \
  --output_dir outputs_codeae_multilabel
```

## 18.3 Regression Fine-tune

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --task_type regression \
  --source_omics_path /workspace/DAPL-master/data/pretrain_ccle.csv \
  --target_omics_path /workspace/DAPL-master/data_Winnie/TCGA_impact_hotspot.csv \
  --source_response_path /workspace/DAPL-master/data_Winnie/PRISM_drug_sensitivity.csv \
  --target_response_path /workspace/DAPL-master/data/TCGA/PMID27354694_DR_OMICS_ad_intersect_pretrain.csv \
  --source_sample_col Sample_ID \
  --target_sample_col tissue_id \
  --target_response_sample_col Patient_id \
  --drug_col drug_name \
  --source_response_col neg_log2_auc \
  --target_response_col Label \
  --pretrain_checkpoint outputs_codeae_multilabel/pretrain/checkpoint.pt \
  --method code_adv \
  --metric macro_rmse \
  --reg_loss mae \
  --regression_binary_threshold 1.0 \
  --n_splits 5 \
  --output_dir outputs_codeae_multilabel_regression
```

---

## 19. Smoke Test Plan

Smoke tests 只驗證 pipeline 是否可跑通，不代表最終 performance。

## 19.1 Classification Smoke Test

目的：確認 classification multilabel CODE-AE 可以：

1. 讀取 omics + response long tables。
2. 建立 `drug_list.csv`。
3. 建立 wide response matrix + mask。
4. 初始化 multi-output head。
5. 完成 1–2 epoch training。
6. 輸出 prediction / metrics / latent / t-SNE。

建議命令：

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --smoke_test classification \
  --epochs 1 \
  --n_splits 2 \
  --output_dir outputs_codeae_smoke_classification
```

## 19.2 Regression Smoke Test

目的：確認 regression source + target classification 設計可以跑通。

檢查：

1. source continuous response 使用 masked regression loss。
2. target binary label 使用 classification metrics。
3. `neg_log2_auc >= 1.0` binary conversion 可產生 source optional binary metrics。
4. missing labels 不進 loss。

建議命令：

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --smoke_test regression \
  --epochs 1 \
  --n_splits 2 \
  --output_dir outputs_codeae_smoke_regression
```

---

## 20. Implementation Phases

## Phase 1: CODE-AE Code Inspection and Mapping

目標：確認原 CODE-AE 中下列 components 的位置與責任：

1. pretrain entry。
2. fine-tune entry。
3. encoder construction。
4. predictor head construction。
5. adversarial / deconfounding losses。
6. early stopping。
7. checkpoint save / load。
8. prediction / evaluation utilities。

輸出：

```text
design_codeae.md § CODE-AE current architecture map
```

## Phase 2: Multilabel Data Module

目標：建立 CODE-AE 專用 multilabel data preparation module。

包含：

1. omics feature alignment。
2. response long-to-wide conversion。
3. mask matrix generation。
4. drug list generation。
5. fold split。
6. data reports。

## Phase 3: Multi-output Prediction Head

目標：替換 single-drug predictor。

包含：

1. `n_drugs` driven output dimension。
2. classification logits output。
3. regression score output。
4. checkpoint compatibility rules。

## Phase 4: Masked Loss and Metrics

目標：實作 multilabel supervised objective 與 metrics。

包含：

1. masked BCE。
2. masked regression loss。
3. per-drug metrics。
4. macro / micro / weighted summary。
5. invalid metric handling。

## Phase 5: Two-stage Pipeline Integration

目標：整合 pretrain checkpoint 與 multilabel fine-tune。

包含：

1. pretrain output checkpoint。
2. fine-tune checkpoint loading。
3. fold-level training。
4. macro metric early stopping。
5. best model saving。

## Phase 6: Output and Latent Export

目標：對齊 multilabel SSDA 的輸出可分析性。

包含：

1. source / target prediction CSV。
2. source / target metrics。
3. final latent pkl。
4. t-SNE plots。
5. fold summary。

## Phase 7: Smoke Tests and Regression Checks

目標：確認 classification / regression 基本可執行。

包含：

1. classification smoke test。
2. regression smoke test。
3. missing label test。
4. source-only / target-only drug test。
5. no cancer type metadata test。

---

## 21. Risks and Open Design Items for `design_codeae.md`

以下項目不在 proposal 階段完全決定，需在後續 `design_codeae.md` 中檢查 CODE-AE 原始碼後定義：

1. CODE-AE 原 fine-tune 是否已有 class weight。
2. CODE-AE 原 regression loss 實際使用 MAE / MSE / other。
3. pretrain checkpoint 中 predictor head 是否存在，以及 multilabel fine-tune 載入時如何忽略 single-drug head。
4. adversarial / deconfounding loss 與 prediction loss 的 weight 是否需因 mask density 調整。
5. validation macro metric fallback 順序。
6. regression run 中 source continuous score 與 target binary logit 共用同一 output 的 calibration 問題。
7. fold-level split 是否要 stratify multilabel distribution。
8. t-SNE 與 latent export 在 CODE-AE stochastic encoder 下如何保證 deterministic。
9. large `n_drugs` 時 multi-output head memory / speed 問題。
10. target-only drugs 沒有 source supervised signal 時，是否只輸出 prediction / metrics，不期待穩定訓練。

---

## 22. Acceptance Criteria

本 proposal 對應的 CODE-AE multilabel implementation 應滿足：

1. 可用新入口完成 pretrain 與 fine-tune 兩階段流程。
2. fine-tune 階段可一次輸出 `[batch_size, n_drugs]`。
3. `drug_list.csv` 自動由 source ∪ target response drugs 產生。
4. missing label 不進 supervised loss。
5. classification run 使用 masked BCEWithLogitsLoss。
6. regression run 支援 source continuous response；target 仍以 classification label evaluation。
7. early stopping 使用 validation macro average prediction metric。
8. CODE-AE encoder / deconfounding 主架構不因 multilabel 改造而被重寫。
9. 輸出 per-drug metrics 與 macro / micro / weighted summary。
10. 訓練完成後輸出 final sample-level latent。
11. latent export 後輸出 t-SNE。
12. 不使用 sample-drug pair model。
13. 不使用 drug latent。
14. classification / regression smoke tests 均可跑通。

---

## 23. Relationship to Existing Multilabel SSDA Work

本 proposal 會參考既有 `experiment_multilabel_ssda.py` 的以下思想：

1. long response table input。
2. source / target drug union。
3. `drug_list.csv` 固定 output order。
4. wide response matrix + mask tensor。
5. masked loss。
6. per-drug + summary metrics。
7. sample-level latent output。
8. t-SNE after training。

但 CODE-AE multilabel pipeline 不應直接變成 SSDA multilabel pipeline。

關鍵差異：

| 項目 | Multilabel SSDA | Multilabel CODE-AE |
|---|---|---|
| training style | SSDA-specific training | CODE-AE two-stage training |
| adaptation | SSDA semi-supervised / entropy-style adaptation | CODE-AE context-aware deconfounding / adversarial adaptation |
| n-shot | 有 n-shot 設計 | 不考慮 n-shot |
| encoder | SSDA encoder | CODE-AE encoder |
| proposal 後續 | `design.md` | `design_codeae.md` |

---

## 24. Summary

本 proposal 建議將 CODE-AE 從 single-drug single-model 改造成 multi-label / multi-drug single-model，但只改變 drug response prediction interface，不改變 CODE-AE 的核心 training philosophy。

最重要的設計原則是：

```text
保留 CODE-AE two-stage deconfounding architecture，
將 single-drug predictor 替換為 multi-output predictor，
用 mask loss 處理 sparse multi-drug labels，
用 macro average metric 做多藥物 model selection，
並在訓練完成後輸出 sample-level latent 與 t-SNE。
```

下一步應基於本 proposal 撰寫：

```text
docs/design_codeae.md
```

其中再詳細定義 CODE-AE 原始碼 mapping、module API、trainer 修改點、checkpoint loading、masked metrics 實作與 smoke test script。
