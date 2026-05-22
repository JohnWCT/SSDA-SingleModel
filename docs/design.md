# Design: Multi-label / Multi-drug SSDA-SingleModel System Architecture

## 1. Design Goal

本文件根據 `docs/proposal.md`，為 SSDA-SingleModel 的 multi-label / multi-drug 改造版本定義詳細系統架構。

新版系統目標是：

```text
omics data table
  -> SSDA encoder
  -> sample-level latent representation
  -> multi-output prediction head
  -> all-drug response vector
```

核心原則：

1. 使用 raw omics data table 作為模型輸入。
2. 不使用 drug latent。
3. 不使用 sample-drug pair model。
4. 將 source / target response long table 轉為 wide matrix + mask tensor。
5. multi-output head 一次輸出所有 drug response。
6. 使用 mask loss 處理 missing labels。
7. target n-shot 在 sample-drug position level 執行。
8. 保留 SSDA 的 source supervised + target labeled supervised + target unlabeled adaptation 訓練精神。
9. 模組間高度獨立、低耦合，方便獨立開發與單元測試。
10. 每個 fold 輸出 sample-level latent、prediction table、metrics、t-SNE 與 latent distribution evaluation。

---

## 2. High-level Architecture

### 2.1 System Flow

```text
CLI / Config
  |
  v
Data Preparation
  |
  |-- read source / target omics
  |-- read source / target response long tables
  |-- generate drug_list.csv from source ∪ target
  |-- build wide response matrices
  |-- build mask matrices
  |-- build target n-shot masks
  |-- build source sample-level splits
  |
  v
Dataset / DataLoader Construction
  |
  v
Model Construction
  |
  |-- SSDA encoder
  |-- multi-output prediction head
  |
  v
Training Engine
  |
  |-- source supervised masked loss
  |-- target labeled masked supervised loss
  |-- target unlabeled masked adaptation loss
  |-- validation without parameter update
  |
  v
Prediction / Evaluation
  |
  |-- source prediction long table
  |-- target prediction long table
  |-- per-drug metrics
  |-- summary metrics
  |
  v
Latent Export / Visualization
  |
  |-- source latent pkl
  |-- target latent pkl
  |-- t-SNE domain mixing
  |-- t-SNE cancer type
  |-- FID / MMD / Wasserstein
  |-- KMeans cancer type metrics
  |
  v
Artifact Writer
```

---

## 3. Package Layout

建議新增獨立 package：

```text
ssda_multilabel/
  __init__.py
  config.py
  seed.py
  io.py
  schemas.py
  validators.py
  data_preparation.py
  drug_index.py
  response_matrix.py
  target_nshot.py
  source_split.py
  cancer_type.py
  datasets.py
  dataloaders.py
  model.py
  encoder_adapter.py
  heads.py
  losses.py
  adaptation.py
  trainer.py
  train_state.py
  prediction.py
  metrics.py
  latent.py
  latent_eval.py
  visualization.py
  artifacts.py
  reports.py

experiment_multilabel_ssda.py
docs/
  proposal.md
  design.md
```

### 3.1 Why a New Package?

使用 `ssda_multilabel/` 而不是直接大量改動 `ssda_latent/` 的原因：

1. 保留目前 SSDA-SingleModel 的 single-drug pipeline。
2. 避免 multi-label logic 汙染既有 single-drug modules。
3. 讓新模組可以被獨立測試。
4. 保持 backward compatibility。
5. 後續如果要移除 single-drug pipeline，也能逐步遷移。

---

## 4. Module Responsibilities

## 4.1 `config.py`

### Responsibility

集中管理 CLI 參數與 runtime config。

### Input

CLI arguments.

### Output

`MultiLabelConfig` dataclass.

### Key Fields

```python
@dataclass
class MultiLabelConfig:
    task_type: Literal["classification", "regression"]

    source_omics_path: str
    target_omics_path: str
    source_response_path: str
    target_response_path: str

    sample_id_col: str
    drug_id_col: str
    response_col: str

    source_cancer_type_path: Optional[str]
    target_cancer_type_path: Optional[str]
    cancer_type_col: str

    random_seed: int
    source_test_size: float
    n_splits: int
    n_shot: int

    encoder: str
    encoder_h_dims: str
    batch_size: int
    epochs: int
    lr: float

    reg_loss: Literal["mse", "mae", "huber"]
    lambda_adapt: float

    output_dir: str
```

### Design Notes

`config.py` 不應讀取資料，也不應建立模型。它只負責解析與保存設定。

### Unit Tests

1. CLI default values are valid.
2. invalid `task_type` raises error.
3. config can be serialized to `config.json`.

---

## 4.2 `seed.py`

### Responsibility

集中設定所有 random seed。

### Functions

```python
def set_global_seed(seed: int) -> None:
    ...
```

### Should Control

1. Python `random`
2. NumPy
3. PyTorch CPU
4. PyTorch CUDA
5. DataLoader generator
6. sklearn split random state
7. t-SNE random state
8. KMeans random state

### Unit Tests

1. same seed produces same NumPy random sequence.
2. same seed produces same torch random tensor.

---

## 4.3 `schemas.py`

### Responsibility

定義系統內部資料結構，避免各模組傳遞鬆散 dict。

### Core Dataclasses

```python
@dataclass
class OmicsTable:
    x: pd.DataFrame
    sample_ids: list[str]
    feature_names: list[str]
    domain: Literal["source", "target"]

@dataclass
class DrugIndex:
    drug_ids: list[str]
    drug_to_index: dict[str, int]
    index_to_drug: dict[int, str]

@dataclass
class ResponseMatrix:
    y: np.ndarray
    mask: np.ndarray
    sample_ids: list[str]
    drug_index: DrugIndex
    domain: Literal["source", "target"]

@dataclass
class TargetMasks:
    observed_mask: np.ndarray
    labeled_mask: np.ndarray
    unlabeled_mask: np.ndarray

@dataclass
class SourceFold:
    fold_id: int
    train_sample_ids: list[str]
    val_sample_ids: list[str]
    test_sample_ids: list[str]

@dataclass
class PreparedData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    source_response: ResponseMatrix
    target_response: ResponseMatrix
    target_masks: TargetMasks
    drug_index: DrugIndex
    folds: list[SourceFold]
    cancer_type_table: Optional[pd.DataFrame]
```

### Design Notes

所有 downstream modules 都應使用 dataclass，不直接重新解析 CSV。

### Unit Tests

1. dataclass objects preserve shape consistency.
2. `ResponseMatrix.y.shape == ResponseMatrix.mask.shape`.
3. drug index length equals matrix column count.

---

## 4.4 `io.py`

### Responsibility

低階檔案讀取與寫入，不做業務邏輯。

### Functions

```python
def read_csv(path: str) -> pd.DataFrame:
    ...

def write_csv(df: pd.DataFrame, path: str) -> None:
    ...

def write_pickle(obj: Any, path: str) -> None:
    ...

def read_pickle(path: str) -> Any:
    ...

def ensure_dir(path: str) -> None:
    ...
```

### Design Notes

此模組不應知道 source / target / drug / model 的概念。

### Unit Tests

1. creates missing output directory.
2. writes and reads pickle correctly.
3. fails clearly on missing file.

---

## 4.5 `validators.py`

### Responsibility

集中資料格式檢查與錯誤訊息。

### Functions

```python
def validate_omics_table(df: pd.DataFrame, sample_id_col: str) -> None:
    ...

def validate_response_long_table(
    df: pd.DataFrame,
    sample_id_col: str,
    drug_id_col: str,
    response_col: str,
    task_type: str,
    domain: str,
) -> None:
    ...

def validate_drug_index(drug_index: DrugIndex) -> None:
    ...

def validate_matrix_and_mask(y: np.ndarray, mask: np.ndarray) -> None:
    ...
```

### Checks

1. Required columns exist.
2. No duplicate sample ID in omics table.
3. Omics features are numeric.
4. response values are numeric.
5. classification labels are 0/1.
6. target labels are 0/1.
7. drug IDs are not empty.
8. matrix and mask shapes match.
9. mask contains only 0/1.

### Unit Tests

1. missing required column raises error.
2. non-binary target response raises error.
3. duplicate sample IDs raise error.
4. non-numeric omics feature raises error.

---

## 4.6 `drug_index.py`

### Responsibility

建立與保存全局 drug order。

### Input

source response long table, target response long table.

### Output

`DrugIndex` and `drug_list.csv`.

### Functions

```python
def build_drug_index_from_union(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_id_col: str,
) -> DrugIndex:
    ...

def save_drug_list(drug_index: DrugIndex, path: str) -> None:
    ...

def load_drug_list(path: str) -> DrugIndex:
    ...
```

### Required Behavior

```text
drug_list = sorted(unique(source_drugs ∪ target_drugs))
```

不刪除任何 drug。

### Edge Cases

| Case | Behavior |
|---|---|
| source-only drug | keep |
| target-only drug | keep |
| duplicated drug rows | allowed in long table |
| empty drug ID | error |
| no overlapping drugs | allowed but warning |

### Unit Tests

1. source-only drug is retained.
2. target-only drug is retained.
3. drug order is deterministic.
4. saved and loaded drug index are identical.

---

## 4.7 `data_preparation.py`

### Responsibility

高階資料整理 orchestration。

### Input

`MultiLabelConfig`

### Output

`PreparedData`

### Main Function

```python
def prepare_multilabel_data(config: MultiLabelConfig) -> PreparedData:
    ...
```

### Internal Steps

1. Read source / target omics tables.
2. Validate omics tables.
3. Align source / target omics features.
4. Read source / target response long tables.
5. Validate response long tables.
6. Build `DrugIndex` from source ∪ target drugs.
7. Build source response matrix + mask.
8. Build target response matrix + observed mask.
9. Build target n-shot labeled / unlabeled masks.
10. Build source test split and K-fold split.
11. Load and align cancer type metadata.
12. Write alignment reports.

### Design Notes

`data_preparation.py` 可呼叫其他 data modules，但 downstream trainer 不應重新處理 long table。

### Unit Tests

1. full preparation creates all expected dataclass objects.
2. source / target matrix columns follow same drug order.
3. source and target omics features are aligned.

---

## 4.8 `response_matrix.py`

### Responsibility

將 long response table 轉換為 wide matrix + mask。

### Functions

```python
def long_to_response_matrix(
    response_df: pd.DataFrame,
    sample_ids: list[str],
    drug_index: DrugIndex,
    sample_id_col: str,
    drug_id_col: str,
    response_col: str,
    domain: str,
) -> ResponseMatrix:
    ...
```

### Behavior

For each observed `(sample_id, drug_id, response)`:

```text
Y[sample_index, drug_index] = response
mask[sample_index, drug_index] = 1
```

For missing positions:

```text
Y = 0
mask = 0
```

### Duplicate Handling

If duplicate `(sample_id, drug_id)` rows exist, do not guess. The implementation should support a configurable strategy:

```text
--duplicate_response_strategy error | mean | median | first
```

Default recommendation:

```text
error
```

because duplicate labels may indicate upstream data issues.

### Unit Tests

1. converts long table to correct wide shape.
2. missing labels have mask 0.
3. source-only and target-only drugs create correct zero masks.
4. duplicate rows raise error by default.

---

## 4.9 `target_nshot.py`

### Responsibility

建立 target position-level n-shot masks。

### Input

`Y_target`, `mask_target_observed`, `n_shot`, `random_seed`.

### Output

`TargetMasks`

### Functions

```python
def build_target_nshot_masks(
    y_target: np.ndarray,
    observed_mask: np.ndarray,
    n_shot: int,
    seed: int,
) -> TargetMasks:
    ...
```

### Algorithm

For each drug column:

```text
if observed_mask[:, d].sum() == 0:
    skip this drug

class_0_positions = positions where observed_mask[:, d] == 1 and y_target[:, d] == 0
class_1_positions = positions where observed_mask[:, d] == 1 and y_target[:, d] == 1

sample up to n_shot from class_0_positions
sample up to n_shot from class_1_positions

set selected positions in labeled_mask = 1
```

Then:

```text
unlabeled_mask = observed_mask - labeled_mask
```

### Edge Cases

| Case | Behavior |
|---|---|
| target has no label for a drug | skip and log warning |
| class 0 count < n_shot | sample all available and log warning |
| class 1 count < n_shot | sample all available and log warning |
| class absent | skip that class and log warning |
| same sample selected for different drugs | allowed |
| labeled and unlabeled overlap | must never happen |

### Unit Tests

1. each drug gets up to n class 0 and n class 1 positions.
2. drug with no target label is skipped.
3. insufficient class samples do not crash.
4. labeled + unlabeled equals observed.
5. labeled and unlabeled masks are disjoint.
6. same seed produces same mask.

---

## 4.10 `source_split.py`

### Responsibility

建立 source sample-level test split 與 K-fold split。

### Functions

```python
def split_source_samples(
    sample_ids: list[str],
    y_source: np.ndarray,
    mask_source: np.ndarray,
    test_size: float,
    n_splits: int,
    seed: int,
) -> list[SourceFold]:
    ...
```

### Split Policy

1. Split is sample-level.
2. `source_test` is created first.
3. Remaining samples are split into K folds.
4. No sample appears in both train and validation in the same fold.
5. No test sample appears in any train / validation.

### Stratification

Multi-label stratification can be non-trivial.

Recommended policy:

1. If a stable multi-label stratification implementation is available, use it.
2. Otherwise, use sample-level random split and report per-drug label distribution.
3. Do not perform sample-drug pair split.

### Reports

Output:

```text
source_split.csv
source_split_label_distribution.csv
```

### Unit Tests

1. train / val / test are disjoint.
2. all source samples are assigned.
3. fixed seed gives reproducible split.
4. every fold has non-empty train and val.

---

## 4.11 `cancer_type.py`

### Responsibility

Load and align cancer type metadata.

### Functions

```python
def load_cancer_type_table(path: Optional[str]) -> Optional[pd.DataFrame]:
    ...

def align_cancer_type(
    source_sample_ids: list[str],
    target_sample_ids: list[str],
    source_cancer_df: Optional[pd.DataFrame],
    target_cancer_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    ...
```

### Output Format

| sample_id | domain | cancer_type |
|---|---|---|

### Missing Policy

Default:

```text
Unknown
```

Configurable:

```text
--unknown_cancer_type_policy unknown | exclude
```

### Unit Tests

1. missing metadata becomes Unknown.
2. extra metadata rows are reported.
3. source and target domains are preserved.

---

## 4.12 `datasets.py`

### Responsibility

Define PyTorch dataset objects.

### Dataset Types

```python
class MultiLabelSampleDataset(Dataset):
    ...
```

### Return Value

Each item should contain:

```python
{
    "x": torch.FloatTensor[n_features],
    "y": torch.FloatTensor[n_drugs],
    "mask": torch.FloatTensor[n_drugs],
    "sample_id": str,
}
```

For target data, dataset may also return:

```python
{
    "observed_mask": ...,
    "labeled_mask": ...,
    "unlabeled_mask": ...,
}
```

### Design Notes

The dataset should not decide loss type. It only provides tensors.

### Unit Tests

1. `__len__` equals number of samples.
2. `__getitem__` returns correct shapes.
3. masks are float tensors.
4. sample IDs are preserved.

---

## 4.13 `dataloaders.py`

### Responsibility

Build DataLoaders from datasets.

### Functions

```python
def build_source_loader(...):
    ...

def build_target_loader(...):
    ...

def build_eval_loader(...):
    ...
```

### Design Notes

1. Training loaders may shuffle.
2. Evaluation loaders must not shuffle.
3. DataLoader random generator should use config seed.
4. Avoid using legacy single-drug weighted sampler unless adapted for multi-label.

### Unit Tests

1. train loader can iterate.
2. eval loader preserves sample order.
3. batch tensors have expected shapes.

---

## 4.14 `encoder_adapter.py`

### Responsibility

Provide a stable interface to original SSDA encoders.

### Functions

```python
def get_encoder_latent(encoder: nn.Module, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    ...
```

### DAE Handling

If DAE forward uses random denoising, deterministic latent extraction should call:

```python
encoder.ae.encode(x)
```

or another deterministic encoder path.

### MLP Handling

If MLP directly returns latent:

```python
latent = encoder(x)
```

### Unit Tests

1. DAE latent shape is `[batch, latent_dim]`.
2. MLP latent shape is `[batch, latent_dim]`.
3. deterministic DAE latent is identical across repeated calls.

---

## 4.15 `heads.py`

### Responsibility

Define multi-output prediction head.

### Class

```python
class MultiOutputHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], n_drugs: int):
        ...

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ...
```

### Output

```text
[batch_size, n_drugs]
```

### Design Notes

No task-specific activation inside the head.

1. Classification uses raw logits.
2. Regression uses raw continuous scores.
3. Target BCE also uses raw logits.

### Unit Tests

1. output shape equals `[batch_size, n_drugs]`.
2. no activation is applied by default.

---

## 4.16 `model.py`

### Responsibility

Compose encoder + multi-output head.

### Class

```python
class MultiLabelSSDAModel(nn.Module):
    def __init__(self, encoder: nn.Module, head: MultiOutputHead):
        ...

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ...
```

### Design Notes

This wrapper should replace legacy `Test_Double_Model` in the multi-label pipeline.

### Unit Tests

1. forward output shape is `[batch, n_drugs]`.
2. encode output shape is `[batch, latent_dim]`.

---

## 4.17 `losses.py`

### Responsibility

Implement task-specific masked supervised losses.

### Functions

```python
def masked_bce_with_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...

def masked_mse(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...

def masked_mae(...):
    ...

def masked_huber(...):
    ...
```

### Safety Rule

If `mask.sum() == 0`, return zero loss with gradient support, or skip the loss in trainer.

Recommended helper:

```python
def safe_masked_mean(raw_loss, mask):
    denom = mask.sum().clamp_min(1.0)
    return (raw_loss * mask).sum() / denom
```

### Unit Tests

1. masked positions do not contribute.
2. all-zero mask does not crash.
3. BCE loss matches PyTorch BCE on valid positions.
4. MSE loss matches manual MSE on valid positions.

---

## 4.18 `adaptation.py`

### Responsibility

Implement target unlabeled entropy / adentropy losses for multi-output predictions.

### Functions

```python
def masked_entropy_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...

def masked_adentropy_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    eta: float,
) -> torch.Tensor:
    ...
```

### Multi-label Interpretation

For binary multi-label outputs, use sigmoid probability per drug:

```python
p = sigmoid(logits)
entropy = -p * log(p) - (1 - p) * log(1 - p)
```

Then apply:

```text
entropy * target_unlabeled_mask
```

### Unit Tests

1. entropy shape matches logits.
2. mask excludes missing positions.
3. all-zero mask does not crash.
4. loss is finite.

---

## 4.19 `trainer.py`

### Responsibility

Run training and validation loops.

### Main Class

```python
class MultiLabelSSDTrainer:
    def __init__(
        self,
        model: MultiLabelSSDAModel,
        optimizer: torch.optim.Optimizer,
        config: MultiLabelConfig,
    ):
        ...

    def train_fold(
        self,
        fold: SourceFold,
        prepared_data: PreparedData,
    ) -> TrainResult:
        ...
```

### Training Losses

#### Classification Run

```text
loss_total =
    source_masked_bce
  + target_labeled_masked_bce
  + lambda_adapt * target_unlabeled_adaptation
  + optional_reconstruction_loss
```

#### Regression Run

```text
loss_total =
    source_masked_regression_loss
  + target_labeled_masked_bce
  + lambda_adapt * target_unlabeled_adaptation
  + optional_reconstruction_loss
```

### Critical Rule

Validation must never call:

```python
loss.backward()
optimizer.step()
```

### Training Batch Strategy

There are two possible approaches:

#### Approach A: paired source and target batches

Each training step draws one source batch and one target batch.

Pros:

1. Close to original SSDA.
2. Source and target losses updated together.

Cons:

1. Source and target loader lengths may differ.

#### Approach B: separate source and target steps

Each epoch has source supervised pass and target adaptation pass.

Pros:

1. Simpler masks.
2. Easier debug.

Cons:

1. Less close to original SSDA.

### Design Decision

Use **Approach A** by default to preserve original SSDA spirit.

If loaders have different lengths, cycle the shorter loader.

### Unit Tests

1. one training step updates parameters.
2. validation step does not update parameters.
3. classification run computes all required losses.
4. regression run computes source regression + target BCE.
5. target unlabeled loss uses `target_unlabeled_mask`.

---

## 4.20 `train_state.py`

### Responsibility

Define training result containers.

### Dataclasses

```python
@dataclass
class EpochLog:
    epoch: int
    source_loss: float
    target_labeled_loss: float
    target_adapt_loss: float
    total_loss: float
    val_loss: float

@dataclass
class TrainResult:
    fold_id: int
    model_path: str
    epoch_logs: list[EpochLog]
```

### Unit Tests

1. can convert logs to DataFrame.
2. can write logs to CSV.

---

## 4.21 `prediction.py`

### Responsibility

Generate prediction long tables.

### Functions

```python
def predict_matrix(
    model: MultiLabelSSDAModel,
    x: np.ndarray,
    batch_size: int,
    device: str,
) -> np.ndarray:
    ...

def build_prediction_long_table(
    scores: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    sample_ids: list[str],
    drug_index: DrugIndex,
    domain: str,
    split_or_role: np.ndarray,
    task_type: str,
) -> pd.DataFrame:
    ...
```

### Output Rule

Export only observed sample-drug positions by default:

```text
mask == 1
```

Optional future extension:

```text
export full matrix prediction
```

### Classification Output

For classification:

```text
pred_score = logits
probability = sigmoid(logit)
pred_label = probability >= 0.5
confidence = probability
```

### Regression Output

For regression:

```text
pred_score = continuous score
pred_label = pred_score >= 1.0
confidence = optional sigmoid(pred_score), or empty
```

Target evaluation in regression run should still use binary label and binary prediction threshold.

### Unit Tests

1. output table has one row per observed position.
2. source-only drug predictions export if observed in source.
3. target-only drug predictions export if observed in target.
4. probability is present for classification.
5. prediction table includes drug_id and drug_index.

---

## 4.22 `metrics.py`

### Responsibility

Compute prediction metrics.

### Functions

```python
def compute_classification_metrics_per_drug(df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_classification_metrics_summary(df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_regression_metrics_per_drug(df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_regression_metrics_summary(df: pd.DataFrame) -> pd.DataFrame:
    ...
```

### Classification Metrics

1. AUC
2. AUPR
3. Accuracy
4. F1
5. Balanced accuracy

### Regression Metrics

1. RMSE
2. MAE
3. R2
4. Pearson
5. Spearman

### Edge Cases

| Case | Behavior |
|---|---|
| only one class for a drug | AUC/AUPR = NaN with warning |
| fewer than 2 samples | metric = NaN |
| no observed rows | empty metrics table |
| regression constant vector | Pearson/Spearman = NaN |

### Unit Tests

1. per-drug metrics group correctly.
2. micro/macro summaries work.
3. single-class AUC does not crash.
4. regression metrics match sklearn/manual calculations.

---

## 4.23 `latent.py`

### Responsibility

Export sample-level latent dictionaries.

### Functions

```python
def encode_latent_dict(
    model: MultiLabelSSDAModel,
    x: np.ndarray,
    sample_ids: list[str],
    batch_size: int,
    device: str,
) -> dict[str, list[float]]:
    ...
```

### Requirements

1. Export after fold training.
2. Use deterministic encoder path.
3. Source latent contains all source samples.
4. Target latent contains all target samples.
5. Latent is sample-level only.

### Unit Tests

1. dict length equals number of samples.
2. each vector length equals latent dim.
3. repeated deterministic export gives same values.

---

## 4.24 `latent_eval.py`

### Responsibility

Compute latent distribution and clustering metrics.

### Functions

```python
def compute_fid(source_z: np.ndarray, target_z: np.ndarray) -> float:
    ...

def compute_mmd(source_z: np.ndarray, target_z: np.ndarray) -> float:
    ...

def compute_wasserstein(source_z: np.ndarray, target_z: np.ndarray) -> float:
    ...

def compute_kmeans_cancer_type_metrics(
    z: np.ndarray,
    cancer_type: list[str],
    seed: int,
) -> dict[str, float]:
    ...
```

### Unit Tests

1. metrics are finite for normal input.
2. insufficient cancer types returns NaN with warning.
3. Unknown cancer type policy is respected.

---

## 4.25 `visualization.py`

### Responsibility

Generate t-SNE figures.

### Functions

```python
def plot_tsne_domain_mixing(...):
    ...

def plot_tsne_cancer_type(...):
    ...
```

### t-SNE Perplexity

Use dynamic perplexity to avoid small sample failures:

```python
perplexity = min(30, max(2, (n_samples - 1) // 3))
```

### Unit Tests

1. plot file is created.
2. small sample size does not crash.
3. missing cancer type handled.

---

## 4.26 `artifacts.py`

### Responsibility

Centralized artifact writing and path management.

### Class

```python
class ArtifactWriter:
    def __init__(self, root_dir: str, seed: int):
        ...

    def fold_dir(self, fold_id: int) -> Path:
        ...

    def write_config(self, config: MultiLabelConfig) -> None:
        ...

    def write_fold_artifacts(...):
        ...
```

### Design Notes

No model or data logic should exist here.

### Unit Tests

1. creates expected directory tree.
2. writes all required files.
3. paths are deterministic.

---

## 4.27 `reports.py`

### Responsibility

Generate human-readable summary reports.

### Reports

1. `data_alignment_report.csv`
2. `drug_list.csv`
3. `source_response_matrix.csv`
4. `source_response_mask.csv`
5. `target_response_matrix.csv`
6. `target_observed_mask.csv`
7. `target_labeled_mask.csv`
8. `target_unlabeled_mask.csv`
9. `target_nshot_summary.csv`
10. `source_split_label_distribution.csv`
11. `metrics_summary.csv`
12. `latent_metrics_summary.csv`

### Unit Tests

1. reports contain expected columns.
2. warning cases are captured.

---

## 5. Main Entry Point

## 5.1 `experiment_multilabel_ssda.py`

### Responsibility

High-level orchestration only.

### Pseudocode

```python
def main():
    config = parse_args()
    set_global_seed(config.random_seed)

    writer = ArtifactWriter(config.output_dir, config.random_seed)
    writer.write_config(config)

    prepared = prepare_multilabel_data(config)
    writer.write_preparation_artifacts(prepared)

    for fold in prepared.folds:
        model = build_multilabel_ssda_model(config, prepared.drug_index)
        trainer = MultiLabelSSDTrainer(model, config)

        train_result = trainer.train_fold(fold, prepared)
        writer.write_train_logs(fold.fold_id, train_result)

        source_pred = predict_source(...)
        target_pred = predict_target(...)

        source_metrics = compute_source_metrics(source_pred, config.task_type)
        target_metrics = compute_target_metrics(target_pred, config.task_type)

        source_latent = encode_source_latent(...)
        target_latent = encode_target_latent(...)

        latent_metrics = compute_latent_metrics(...)
        tsne_paths = plot_tsne(...)

        writer.write_fold_outputs(...)

    writer.write_summary_outputs(...)
```

### Rule

The entry point should not contain detailed data transformation, model internals, or metrics implementation.

---

## 6. Data Contracts Between Modules

## 6.1 Prepared Data Contract

All training modules receive `PreparedData`.

They must not read original CSV again.

## 6.2 Drug Index Contract

All modules must use `DrugIndex`.

No module may independently sort drug IDs.

## 6.3 Matrix Contract

For any response matrix:

```text
Y.shape == mask.shape == [n_samples, n_drugs]
```

## 6.4 Model Output Contract

For any model forward pass:

```text
output.shape == [batch_size, n_drugs]
```

## 6.5 Prediction Table Contract

All prediction tables must include:

| Column |
|---|
| sample_id |
| drug_id |
| drug_index |
| domain |
| split_or_role |
| ground_truth |
| pred_score |
| pred_label |
| task_type |
| fold |
| seed |

---

## 7. Low-coupling Design Rules

1. `io.py` must not know biological domain concepts.
2. `drug_index.py` must not know model logic.
3. `response_matrix.py` must not know training logic.
4. `target_nshot.py` must only operate on arrays and masks.
5. `source_split.py` must not build DataLoaders.
6. `datasets.py` must not compute losses.
7. `losses.py` must not know source / target semantics.
8. `adaptation.py` must not know file paths.
9. `trainer.py` must not write artifacts directly.
10. `prediction.py` must not compute training loss.
11. `metrics.py` must not run model inference.
12. `latent.py` must not plot figures.
13. `visualization.py` must not compute prediction metrics.
14. `artifacts.py` must not transform data.

---

## 8. Error Handling and Logging

### 8.1 Hard Errors

The system should stop when:

1. Required input file is missing.
2. Required column is missing.
3. Omics features are non-numeric.
4. Target response contains values other than 0/1.
5. Drug list is empty.
6. Matrix and mask shapes mismatch.
7. Model output dimension does not equal number of drugs.

### 8.2 Warnings

The system should continue but log warning when:

1. source-only drug exists.
2. target-only drug exists.
3. target drug has no labels.
4. target drug class count is below `n_shot`.
5. a drug has only one class.
6. AUC cannot be computed.
7. cancer type is missing.
8. duplicate response strategy is not `error`.

### 8.3 Logs

Write:

```text
run.log
target_nshot_summary.csv
data_alignment_report.csv
metrics_warning_report.csv
```

---

## 9. Testing Strategy

## 9.1 Unit Tests

Each module should have independent tests.

Suggested structure:

```text
tests/
  test_drug_index.py
  test_response_matrix.py
  test_target_nshot.py
  test_source_split.py
  test_losses.py
  test_adaptation.py
  test_model.py
  test_prediction.py
  test_metrics.py
  test_latent.py
```

## 9.2 Integration Tests

Create a small synthetic dataset:

```text
source samples: 8
target samples: 6
drugs: 4
features: 10
missing labels: yes
source-only drug: yes
target-only drug: yes
```

Integration test should verify:

1. data preparation completes.
2. one fold trains for one epoch.
3. prediction tables are written.
4. latent pkl files are written.
5. metrics files are written.

---

## 10. Output Directory Contract

The system should create:

```text
save/
  ssda_multilabel/
    seed_{random_seed}/
      config.json
      run.log
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
        train_log.csv
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

      metrics_summary.csv
      latent_metrics_summary.csv
      kmeans_cancer_type_summary.csv
```

---

## 11. Technical Tradeoffs and Decisions

### 11.1 Pair Model vs Multi-output Model

Decision:

```text
Use multi-output model.
```

Reason:

1. User explicitly requested multi-output model.
2. No drug latent input is needed.
3. Source and target response matrices can be perfectly aligned through `drug_list.csv`.

---

### 11.2 Drug Intersection vs Drug Union

Decision:

```text
Use source ∪ target union.
```

Reason:

1. User explicitly requested not deleting any drug.
2. Source-only and target-only drugs are retained.
3. Missing domains are handled by mask.

---

### 11.3 Sample-level vs Position-level Target N-shot

Decision:

```text
Use sample-drug position-level n-shot.
```

Reason:

1. Multi-label setting requires per-drug labels.
2. Same sample can be labeled for one drug and unlabeled for another.
3. This preserves original SSDA n-shot idea while adapting it to multi-label data.

---

### 11.4 Regression Run Target Supervision

Decision:

```text
Use source regression loss + target labeled BCE loss + target unlabeled adaptation loss.
```

Reason:

1. Original SSDA target labeled samples participate in supervised classification loss.
2. Target labels are always classification.
3. A single multi-output score can be interpreted as continuous prediction for source regression and binary logit for target classification.

This is a deliberate design compromise. It should be documented in code and logs.

---

### 11.5 Multi-label Stratification

Decision:

```text
Use sample-level split. Multi-label stratification optional.
```

Reason:

1. Avoid leakage across sample-drug pairs.
2. Multi-label stratification can add dependency complexity.
3. Label distribution reports can detect severe imbalance.

Future extension can add iterative stratification.

---

## 12. Open Configurable Options

These options should be configurable rather than hard-coded:

1. `--reg_loss`: `mse`, `mae`, `huber`.
2. `--lambda_adapt`.
3. `--source_test_size`.
4. `--n_splits`.
5. `--n_shot`.
6. `--unknown_cancer_type_policy`.
7. `--duplicate_response_strategy`.
8. `--export_wide_predictions`.
9. `--use_multilabel_stratification`.

---

## 13. Final Acceptance Criteria

The architecture is implemented correctly when:

1. `drug_list.csv` contains source ∪ target drugs.
2. model output dimension equals number of drugs.
3. source and target matrices use identical drug order.
4. source-only and target-only drugs are retained.
5. missing labels do not contribute to supervised loss.
6. target n-shot is position-level.
7. target labeled mask and target unlabeled mask are disjoint.
8. validation does not update model parameters.
9. classification run works end-to-end.
10. regression run works end-to-end.
11. source and target prediction tables are exported in long format.
12. source and target sample-level latent pkl files are exported.
13. latent evaluation and t-SNE outputs are generated.
14. all modules are independently testable.
15. no drug latent or sample-drug pair model is introduced.
