# Design: Multi-label / Multi-drug CODE-AE System Architecture

> **Document status:** Architecture design draft based on `proposal_codeae.md`.
>
> **Scope:** This document defines the modular architecture for converting CODE-AE from a single-drug single-model workflow into a multi-label / multi-drug single-model workflow. It focuses on module boundaries, data contracts, responsibilities, dependencies, testability, and implementation decisions. It does not attempt to rewrite the original CODE-AE algorithm.
>
> **Primary design rule:** preserve CODE-AE's two-stage encoder pretraining and drug response fine-tuning philosophy, while replacing the single-drug predictor interface with a multi-output predictor and masked multi-drug data handling.

---

## 1. Design Goal

The existing CODE-AE workflow is single-drug oriented:

```bash
cd benchmark/CODEAE/
python pretrain_hyper_main.py --drug "Gefitinib"
python drug_ft_hyper_main.py --drug "Gefitinib"
```

The redesigned workflow should support:

```text
omics data table
  -> CODE-AE encoder / deconfounding module
  -> sample-level latent representation
  -> multi-output drug prediction head
  -> all-drug response vector [n_drugs]
```

The main goals are:

1. Use one shared CODE-AE model for multiple drugs.
2. Preserve CODE-AE pretrain -> fine-tune two-stage training.
3. Preserve CODE-AE encoder, context-aware deconfounding, adversarial/domain components, and training philosophy.
4. Replace only the drug prediction interface from single-output to multi-output.
5. Support sparse source / target drug response labels by using wide response matrices plus masks.
6. Treat target response labels as evaluation labels only, because the original CODE-AE setting does not use target labels for supervised training.
7. Export final sample-level latent representations after CODE-AE fine-tuning, not after pretraining.
8. Keep modules independent and low-coupled so they can be implemented and tested separately.

---

## 2. Architecture Decisions

| ID | Decision | Final Choice |
|---|---|---|
| AD-01 | Package location | Add independent package `benchmark/CODEAE/codeae_multilabel/` |
| AD-02 | Entry points | Add two thin entries: `pretrain_multilabel_hyper_main.py` and `drug_ft_multilabel_hyper_main.py` |
| AD-03 | Orchestration | Use `PretrainRunner` and `FineTuneRunner` |
| AD-04 | Data module | Isolate omics alignment, response matrix, masks, drug list, and splits into data modules |
| AD-05 | Drug list timing | Build drug list only in fine-tune stage |
| AD-06 | Checkpoint requirement | Fine-tune requires a pretrain checkpoint; do not silently train from scratch |
| AD-07 | Checkpoint compatibility | Load encoder / CODE-AE modules and ignore incompatible single-drug heads |
| AD-08 | Prediction head | Implement independent multi-output head module |
| AD-09 | Losses | Implement masked losses in `losses.py`; trainer only calls loss APIs |
| AD-10 | Metrics | Implement metrics as pure pandas / numpy utilities, independent of PyTorch models |
| AD-11 | Model selection | Implement early stopping / selection in `selection.py` |
| AD-12 | Classification early stopping | Primary metric: `macro_auroc`; fallback: `macro_aupr`, `macro_balanced_accuracy`, `macro_f1` |
| AD-13 | Regression early stopping | Primary metric: `macro_mae`; lower is better |
| AD-14 | Target labels | Target labels are evaluation-only; no target supervised training option |
| AD-15 | Source-only / target-only drugs | Keep all drugs in source ∪ target; skip loss for drugs with no source training labels |
| AD-16 | Fold split | Sample-level random split with fixed seed; no multilabel stratification in first design |
| AD-17 | Contracts | Use dataclasses in `contracts.py` instead of loose dictionaries |
| AD-18 | Artifact writing | Use `ArtifactWriter`; other modules should not write files directly |
| AD-19 | Latent / t-SNE | Export after final training via dedicated export / visualization modules |
| AD-20 | Visualization failure | Missing cancer type or insufficient samples should warn and skip, not crash |
| AD-21 | Output overwrite | Default behavior is overwrite existing `output_dir` |
| AD-22 | Smoke test | No synthetic dataset builder; run real small-epoch smoke runs |
| AD-23 | Forbidden dependencies | Data must not import trainer; metrics must not import model; legacy CODE-AE should not import new package |
| AD-24 | Document path | Generate `docs/design_codeae.md`, not overwrite `docs/design.md` |

---

## 3. High-level System Flow

```text
CLI / Config
  |
  v
Stage 1: CODE-AE Multilabel Pretraining
  |
  |-- read source / target omics
  |-- validate omics tables
  |-- align source / target features
  |-- call original CODE-AE pretraining logic
  |-- save pretrain checkpoint
  |
  v
Stage 2: CODE-AE Multilabel Fine-tuning
  |
  |-- read source / target omics
  |-- read source / target response long tables
  |-- generate drug_list.csv from source ∪ target
  |-- build wide source / target response matrices
  |-- build source / target observed masks
  |-- build source sample-level train / val / test folds
  |-- load pretrain checkpoint
  |-- attach multi-output prediction head
  |-- train on source supervised masked loss + original CODE-AE losses
  |-- evaluate source validation and target labels without target supervised updates
  |-- early stop by macro validation metric
  |-- save best model
  |
  v
Prediction / Evaluation
  |
  |-- source prediction long table
  |-- target prediction long table
  |-- per-drug metrics
  |-- macro / micro / weighted summary metrics
  |
  v
Final Latent Export / Visualization
  |
  |-- source sample-level latent
  |-- target sample-level latent
  |-- t-SNE domain mixing
  |-- t-SNE cancer type, if metadata is available
  |-- latent distribution metrics, if enabled
  |
  v
ArtifactWriter
```

---

## 4. Proposed Package Layout

```text
benchmark/CODEAE/
  pretrain_multilabel_hyper_main.py
  drug_ft_multilabel_hyper_main.py

  codeae_multilabel/
    __init__.py
    config.py
    contracts.py
    seed.py
    io.py
    validators.py

    data/
      __init__.py
      omics.py
      sample_id.py
      drug_index.py
      response_matrix.py
      cancer_type.py
      split.py
      prepare_pretrain.py
      prepare_finetune.py

    model/
      __init__.py
      legacy_adapter.py
      heads.py
      wrapper.py
      checkpoint.py

    training/
      __init__.py
      losses.py
      trainer.py
      selection.py
      train_state.py
      runners.py

    evaluation/
      __init__.py
      prediction.py
      metrics.py
      reports.py

    export/
      __init__.py
      artifacts.py
      latent.py
      visualization.py
      latent_eval.py

    smoke/
      __init__.py
      smoke_commands.md
      smoke_runner.py
```

### 4.1 Why a New Package?

Use `benchmark/CODEAE/codeae_multilabel/` instead of directly modifying all existing CODE-AE files because:

1. The single-drug CODE-AE implementation remains available as a reference.
2. Multi-label data logic does not leak into legacy single-drug scripts.
3. New modules can be unit-tested without invoking full CODE-AE training.
4. Checkpoint, data, metrics, and export contracts are explicit.
5. Future `design_codeae.md` revisions can map individual legacy CODE-AE components into the new adapter layer without changing high-level modules.

### 4.2 Thin Entry Principle

The new entry scripts should only:

1. Parse CLI arguments.
2. Build config objects.
3. Set global seed.
4. Instantiate the correct runner.
5. Call `runner.run()`.

They should not directly implement data preparation, training loops, metrics, or artifact writing.

---

## 5. Module Dependency Rules

### 5.1 Allowed Dependency Direction

```text
config / contracts / seed / io / validators
        ^
        |
data modules
        ^
        |
model modules
        ^
        |
training modules
        ^
        |
evaluation modules
        ^
        |
export modules
        ^
        |
runners / entry scripts
```

In practice, the runner imports most modules, but lower-level modules should not import orchestration code.

### 5.2 Forbidden Dependencies

| Module | Must Not Import | Reason |
|---|---|---|
| `data/*` | `training/*`, `model/*` | Data preparation must be independently testable |
| `metrics.py` | PyTorch model classes | Metrics should consume prediction tables only |
| `losses.py` | File writers / metrics | Loss functions should be pure tensor functions |
| `model/*` | CSV readers / response table logic | Model should not know file formats |
| `ArtifactWriter` | Trainer internals | Writer should only consume explicit output objects |
| Legacy CODE-AE modules | `codeae_multilabel/*` | Avoid coupling legacy code to new pipeline |
| Entry scripts | business logic | Keep entries thin and testable |

---

## 6. Core Data Contracts

All modules should pass typed dataclass objects instead of loose dictionaries where practical.

### 6.1 `contracts.py`

#### `CodeAEMultilabelConfig`

```python
@dataclass
class CodeAEMultilabelConfig:
    task_type: Literal["classification", "regression"]

    source_omics_path: str
    target_omics_path: str
    source_response_path: Optional[str]
    target_response_path: Optional[str]

    source_sample_col: str
    target_sample_col: str
    target_response_sample_col: str
    drug_col: str
    source_response_col: str
    target_response_col: str

    method: str
    pretrain_checkpoint: Optional[str]
    output_dir: str
    overwrite: bool

    batch_size: int
    epochs: int
    lr: float
    seed: int
    n_splits: int
    source_test_size: float

    metric: Optional[str]
    reg_loss: Literal["mae"]
    prediction_threshold: float
    regression_binary_threshold: float

    source_cancer_type_path: Optional[str]
    target_cancer_type_path: Optional[str]
    cancer_type_col: Optional[str]
```

Design notes:

1. `source_response_path` and `target_response_path` are optional for pretraining but required for fine-tuning.
2. `pretrain_checkpoint` is optional for pretraining but required for fine-tuning.
3. `overwrite` defaults to `True`.
4. `reg_loss` is fixed to `mae` in this design unless later CODE-AE code inspection proves that another original loss must be preserved.
5. No `n_shot` field is needed.
6. No `use_target_supervision` field is needed because target labels are evaluation-only.

#### `OmicsTable`

```python
@dataclass
class OmicsTable:
    x: pd.DataFrame
    sample_ids: list[str]
    feature_names: list[str]
    domain: Literal["source", "target"]
```

Rules:

1. `x.index` must align with `sample_ids`.
2. All features must be numeric.
3. Source and target feature columns must be aligned before downstream use.

#### `DrugIndex`

```python
@dataclass
class DrugIndex:
    drug_ids: list[str]
    drug_to_index: dict[str, int]
    index_to_drug: dict[int, str]
```

Rules:

1. Drug order is deterministic.
2. Drug order is shared by source and target matrices.
3. Model output column `j` corresponds to `drug_ids[j]`.

#### `ResponseMatrix`

```python
@dataclass
class ResponseMatrix:
    y: np.ndarray
    mask: np.ndarray
    sample_ids: list[str]
    drug_index: DrugIndex
    domain: Literal["source", "target"]
    label_semantics: Literal["binary", "continuous"]
```

Rules:

1. `y.shape == mask.shape == [n_samples, n_drugs]`.
2. `mask == 1` indicates observed response.
3. Missing values in `y` are ignored by loss and metrics when `mask == 0`.
4. Source regression uses `label_semantics="continuous"`.
5. Target always uses `label_semantics="binary"` for evaluation.

#### `PreparedPretrainData`

```python
@dataclass
class PreparedPretrainData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    feature_alignment: pd.DataFrame
```

#### `PreparedFineTuneData`

```python
@dataclass
class PreparedFineTuneData:
    source_omics: OmicsTable
    target_omics: OmicsTable
    source_response: ResponseMatrix
    target_response: ResponseMatrix
    drug_index: DrugIndex
    folds: list[SourceFold]
    cancer_type_table: Optional[pd.DataFrame]
```

#### `SourceFold`

```python
@dataclass
class SourceFold:
    fold_id: int
    train_sample_ids: list[str]
    val_sample_ids: list[str]
    test_sample_ids: list[str]
```

#### `TrainingResult`

```python
@dataclass
class TrainingResult:
    fold_id: int
    best_model_path: str
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    train_log: pd.DataFrame
```

#### `PredictionBundle`

```python
@dataclass
class PredictionBundle:
    source_predictions: pd.DataFrame
    target_predictions: pd.DataFrame
    source_metrics_per_drug: pd.DataFrame
    target_metrics_per_drug: pd.DataFrame
    source_metrics_summary: pd.DataFrame
    target_metrics_summary: pd.DataFrame
```

---

## 7. Configuration Module: `config.py`

### Responsibility

Centralize CLI parsing and runtime configuration.

### Inputs

1. CLI arguments from pretrain entry.
2. CLI arguments from fine-tune entry.

### Outputs

1. `CodeAEMultilabelConfig`.
2. `config.json` through `ArtifactWriter`.

### Key Behaviors

1. Validate `task_type`.
2. Validate fine-tune-only required arguments.
3. Default `overwrite=True`.
4. Default `reg_loss="mae"`.
5. Default classification metric to `macro_auroc`.
6. Default regression metric to `macro_mae`.
7. Do not read CSV files.
8. Do not instantiate models.
9. Do not create output directories directly; delegate to `ArtifactWriter` or runner setup.

### Unit Tests

1. Pretrain config can be created without response paths.
2. Fine-tune config fails clearly if response paths are missing.
3. Fine-tune config fails clearly if `pretrain_checkpoint` is missing.
4. `overwrite` defaults to `True`.
5. Invalid task type raises a descriptive error.
6. Config serializes to JSON.

---

## 8. Seed Module: `seed.py`

### Responsibility

Set reproducible random seeds.

### API

```python
def set_global_seed(seed: int) -> None:
    ...
```

### Should Control

1. Python `random`.
2. NumPy.
3. PyTorch CPU.
4. PyTorch CUDA.
5. DataLoader generator.
6. sklearn split random state.
7. t-SNE random state.
8. KMeans random state.

### Unit Tests

1. Same seed produces same NumPy random sequence.
2. Same seed produces same torch tensor.
3. Different seed produces different split assignment where applicable.

---

## 9. I/O Module: `io.py`

### Responsibility

Low-level file reading and writing only.

### API

```python
def read_csv(path: str) -> pd.DataFrame:
    ...

def write_csv(df: pd.DataFrame, path: str) -> None:
    ...

def write_json(obj: dict, path: str) -> None:
    ...

def write_pickle(obj: Any, path: str) -> None:
    ...

def read_pickle(path: str) -> Any:
    ...

def ensure_clean_dir(path: str, overwrite: bool = True) -> None:
    ...
```

### Output Directory Policy

Default behavior:

```text
overwrite = True
```

If `output_dir` already exists:

1. Delete or clear existing generated contents when `overwrite=True`.
2. Raise an error only when `overwrite=False`.
3. Write an explicit `run_manifest.json` after initialization.

### Unit Tests

1. Missing file raises clear error.
2. CSV round-trip preserves columns.
3. Pickle round-trip works.
4. Existing output directory is overwritten by default.
5. Existing output directory raises when `overwrite=False`.

---

## 10. Validators Module: `validators.py`

### Responsibility

Centralize input validation and shape validation.

### API

```python
def validate_omics_table(df: pd.DataFrame, sample_id_col: str) -> None:
    ...

def validate_response_long_table(
    df: pd.DataFrame,
    sample_id_col: str,
    drug_col: str,
    response_col: str,
    task_type: str,
    domain: str,
) -> None:
    ...

def validate_drug_index(drug_index: DrugIndex) -> None:
    ...

def validate_response_matrix(matrix: ResponseMatrix) -> None:
    ...

def validate_folds(folds: list[SourceFold]) -> None:
    ...
```

### Validation Rules

#### Omics

1. Required sample column exists.
2. No duplicate sample IDs after normalization.
3. Feature columns are numeric.
4. At least one common source-target feature remains after alignment.

#### Response Long Table

1. Required sample, drug, and response columns exist.
2. Drug IDs are non-empty strings.
3. Response column is numeric.
4. Classification responses are binary 0/1.
5. Target responses are binary 0/1 for both classification and regression runs.
6. Source regression responses are continuous numeric.

#### Matrix and Mask

1. `y.shape == mask.shape`.
2. `mask` contains only 0/1.
3. Matrix column count equals `len(drug_index.drug_ids)`.
4. Matrix row count equals `len(sample_ids)`.

### Unit Tests

1. Missing required column raises.
2. Duplicate sample ID raises.
3. Non-numeric feature raises.
4. Non-binary target label raises.
5. Invalid mask values raise.

---

## 11. Data Module: Omics Alignment

### File

```text
codeae_multilabel/data/omics.py
```

### Responsibility

Read and align source / target omics tables.

### API

```python
def read_omics_table(
    path: str,
    sample_id_col: str,
    domain: Literal["source", "target"],
) -> OmicsTable:
    ...

def align_omics_features(
    source: OmicsTable,
    target: OmicsTable,
) -> tuple[OmicsTable, OmicsTable, pd.DataFrame]:
    ...
```

### Behavior

1. Read CSV file.
2. Move sample ID column into explicit `sample_ids` list.
3. Keep only numeric features.
4. Align source and target to the intersection of features.
5. Preserve feature order deterministically.
6. Return a feature alignment report.

### Feature Alignment Report

Suggested columns:

| Column | Description |
|---|---|
| `feature_name` | Feature / gene symbol |
| `in_source` | Whether present in source |
| `in_target` | Whether present in target |
| `used` | Whether used in aligned matrix |

### Unit Tests

1. Source and target output feature columns match.
2. Non-overlapping features raise a clear error.
3. Feature alignment report includes removed features.
4. Sample order is preserved.

---

## 12. Data Module: Sample ID Normalization

### File

```text
codeae_multilabel/data/sample_id.py
```

### Responsibility

Centralize sample ID normalization and TCGA-style joins.

### API

```python
def normalize_source_sample_id(sample_id: str) -> str:
    ...

def normalize_target_omics_sample_id(sample_id: str) -> str:
    ...

def normalize_target_response_sample_id(sample_id: str) -> str:
    ...

def build_target_omics_response_join_key(sample_id: str) -> str:
    ...
```

### Design Notes

1. The rest of the system should not manually slice TCGA IDs.
2. Target omics may use `tissue_id`, while target response may use `Patient_id`.
3. The join key logic should be explicit and tested.
4. If no normalization is requested, functions should behave as identity transforms.

### Unit Tests

1. Source ID identity behavior works.
2. TCGA tissue ID to patient ID mapping is deterministic.
3. Duplicate IDs after normalization are detected by validators.
4. Unjoinable target samples are reported, not silently dropped without accounting.

---

## 13. Data Module: Drug Index

### File

```text
codeae_multilabel/data/drug_index.py
```

### Responsibility

Create and persist global drug order.

### API

```python
def build_drug_index_from_union(
    source_response: pd.DataFrame,
    target_response: pd.DataFrame,
    drug_col: str,
) -> DrugIndex:
    ...

def save_drug_list(drug_index: DrugIndex, path: str) -> pd.DataFrame:
    ...

def load_drug_list(path: str) -> DrugIndex:
    ...
```

### Required Behavior

```text
drug_list = sorted(unique(source_drugs ∪ target_drugs))
```

### Edge Cases

| Case | Behavior |
|---|---|
| Source-only drug | Keep in `drug_list.csv` |
| Target-only drug | Keep in `drug_list.csv` |
| No overlap between source and target drugs | Allow but warn |
| Empty drug ID | Raise error |
| Duplicate drug names | Collapse to one drug ID after normalization |

### Output: `drug_list.csv`

```csv
drug_id,drug_index
DrugA,0
DrugB,1
DrugC,2
```

### Unit Tests

1. Source-only drug is retained.
2. Target-only drug is retained.
3. Drug order is deterministic.
4. Saved and loaded drug index are identical.
5. Model output dimension can be derived from drug index length.

---

## 14. Data Module: Response Matrix

### File

```text
codeae_multilabel/data/response_matrix.py
```

### Responsibility

Convert long response tables into wide matrices and masks.

### API

```python
def long_to_response_matrix(
    response_df: pd.DataFrame,
    sample_ids: list[str],
    drug_index: DrugIndex,
    sample_id_col: str,
    drug_col: str,
    response_col: str,
    domain: Literal["source", "target"],
    label_semantics: Literal["binary", "continuous"],
    duplicate_strategy: Literal["mean", "median", "first", "error"] = "mean",
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
Y[sample_index, drug_index] = 0
mask[sample_index, drug_index] = 0
```

### Duplicate Handling

Default strategy:

```text
mean
```

Supported strategies:

| Strategy | Behavior |
|---|---|
| `mean` | Average duplicate responses |
| `median` | Median duplicate responses |
| `first` | Keep first occurrence after stable ordering |
| `error` | Raise on duplicates |

### Source / Target Semantics

| Task | Domain | Label Semantics | Training Use |
|---|---|---|---|
| classification | source | binary | source supervised loss |
| classification | target | binary | evaluation only |
| regression | source | continuous | source supervised MAE loss |
| regression | target | binary | evaluation only |

### Unit Tests

1. Long table converts to expected wide shape.
2. Missing labels have `mask=0`.
3. Observed labels have `mask=1`.
4. Duplicate strategy `mean` works.
5. Duplicate strategy `error` raises.
6. Source-only and target-only drugs produce correct zero mask columns where missing.
7. Target response matrix is binary for both task types.

---

## 15. Data Module: Source Split

### File

```text
codeae_multilabel/data/split.py
```

### Responsibility

Create source sample-level train / validation / test splits.

### API

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

1. Split at sample level, not sample-drug position level.
2. Create `source_test` first.
3. Split the remaining source samples into K train / validation folds.
4. Use fixed seed random split.
5. Do not use multilabel stratification in this design.
6. Report per-drug label availability for each fold.

### Reports

1. `source_split.csv`
2. `source_split_label_distribution.csv`

### Unit Tests

1. Train / validation / test are disjoint.
2. Every source sample appears in exactly one role for each fold.
3. Source test samples never appear in train or validation.
4. Fixed seed gives reproducible splits.
5. Each fold has non-empty train and validation sets.

---

## 16. Data Module: Cancer Type Metadata

### File

```text
codeae_multilabel/data/cancer_type.py
```

### Responsibility

Load optional cancer type metadata for visualization and latent evaluation.

### API

```python
def load_cancer_type_table(path: Optional[str], sample_col: str, cancer_type_col: str) -> Optional[pd.DataFrame]:
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
| sample_A | source | LUAD |
| sample_B | target | BRCA |

### Missing Policy

Default:

```text
Unknown
```

If metadata is missing:

1. Training continues.
2. Domain t-SNE can still be generated.
3. Cancer-type t-SNE and clustering metrics can be skipped with warnings.

### Unit Tests

1. Missing metadata becomes `Unknown`.
2. Extra metadata rows are reported.
3. Source and target domains are preserved.
4. Visualization modules can consume the aligned table.

---

## 17. Data Preparation Orchestrators

### 17.1 Pretrain Data Preparation

File:

```text
codeae_multilabel/data/prepare_pretrain.py
```

API:

```python
def prepare_pretrain_data(config: CodeAEMultilabelConfig) -> PreparedPretrainData:
    ...
```

Steps:

1. Read source omics.
2. Read target omics.
3. Validate omics tables.
4. Align features.
5. Return `PreparedPretrainData`.
6. Do not read response tables.
7. Do not create drug list.
8. Do not create response masks.

### 17.2 Fine-tune Data Preparation

File:

```text
codeae_multilabel/data/prepare_finetune.py
```

API:

```python
def prepare_finetune_data(config: CodeAEMultilabelConfig) -> PreparedFineTuneData:
    ...
```

Steps:

1. Read source omics.
2. Read target omics.
3. Validate and align omics features.
4. Read source response long table.
5. Read target response long table.
6. Validate response long tables.
7. Build `DrugIndex` from source ∪ target drugs.
8. Build source response matrix and source mask.
9. Build target response matrix and target mask.
10. Build source sample-level folds.
11. Load optional cancer type metadata.
12. Return `PreparedFineTuneData`.

### Unit Tests

1. Pretrain preparation does not require response files.
2. Fine-tune preparation requires response files.
3. Source and target matrices share drug order.
4. Source and target omics share feature order.
5. Target labels are not converted into training masks.
6. Drug union behavior is preserved.

---

## 18. Dataset and DataLoader Modules

### Files

```text
codeae_multilabel/data/dataset.py
codeae_multilabel/data/dataloader.py
```

### Dataset Responsibility

Wrap prepared arrays into PyTorch datasets.

### Dataset API

```python
class MultiDrugSampleDataset(torch.utils.data.Dataset):
    def __init__(self, x, y=None, mask=None, sample_ids=None, domain=None):
        ...

    def __getitem__(self, idx):
        return {
            "x": torch.FloatTensor[n_features],
            "y": torch.FloatTensor[n_drugs],
            "mask": torch.FloatTensor[n_drugs],
            "sample_id": str,
            "domain": str,
        }
```

### Dataset Rules

1. Dataset does not decide task type.
2. Dataset does not compute loss.
3. Dataset does not perform response aggregation.
4. Evaluation datasets must preserve sample order.
5. Target dataset may include `y` and `mask` for evaluation, but trainer must not use target labels for supervised updates.

### DataLoader API

```python
def build_source_train_loader(...):
    ...

def build_source_eval_loader(...):
    ...

def build_target_unlabeled_loader(...):
    ...

def build_target_eval_loader(...):
    ...
```

### DataLoader Rules

1. Training loaders can shuffle.
2. Evaluation loaders must not shuffle.
3. DataLoader generator uses config seed.
4. Do not use single-drug weighted sampler unless it is redesigned for multilabel.

### Unit Tests

1. Dataset length equals number of samples.
2. `__getitem__` returns expected tensor shapes.
3. Source training batch has `x`, `y`, and `mask`.
4. Target unlabeled training batch can omit labels or ignore labels.
5. Evaluation loader preserves sample order.

---

## 19. Model Module: Legacy CODE-AE Adapter

### File

```text
codeae_multilabel/model/legacy_adapter.py
```

### Responsibility

Provide a stable interface around original CODE-AE model construction and encoder behavior.

### API

```python
def build_legacy_codeae_components(config: CodeAEMultilabelConfig, n_features: int):
    ...

def encode_with_codeae(model: nn.Module, x: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
    ...

def get_latent_dim(model: nn.Module) -> int:
    ...
```

### Design Notes

1. Legacy CODE-AE details should be isolated here.
2. Other modules should not depend on legacy CODE-AE class names.
3. If original CODE-AE has stochastic denoising or dropout, `encode_with_codeae(..., deterministic=True)` must use eval mode and deterministic encoder path.
4. If original CODE-AE returns multiple outputs, adapter extracts the latent representation consistently.

### Unit Tests

1. Adapter returns latent shape `[batch, latent_dim]`.
2. Deterministic encoding gives stable results in eval mode.
3. Adapter can identify latent dimension.
4. Unsupported legacy model output raises a clear error.

---

## 20. Model Module: Multi-output Head

### File

```text
codeae_multilabel/model/heads.py
```

### Responsibility

Define the multi-drug prediction head.

### API

```python
class MultiOutputDrugHead(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], n_drugs: int, dropout: float = 0.0):
        ...

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ...
```

### Output

```text
[batch_size, n_drugs]
```

### Activation Rule

No activation inside the head.

| Task | Head Output | Post-processing |
|---|---|---|
| classification | logits | sigmoid outside model |
| regression | continuous scores | raw score for source regression; threshold only for binary evaluation |

### Unit Tests

1. Output shape is `[batch_size, n_drugs]`.
2. No sigmoid or softmax is applied internally.
3. Head supports `n_drugs=1` and `n_drugs>1`.
4. Head has no dependency on drug names.

---

## 21. Model Module: Multilabel CODE-AE Wrapper

### File

```text
codeae_multilabel/model/wrapper.py
```

### Responsibility

Compose CODE-AE encoder / deconfounding components with multi-output drug head.

### API

```python
class MultiLabelCodeAEModel(nn.Module):
    def __init__(self, codeae_core: nn.Module, prediction_head: MultiOutputDrugHead):
        ...

    def encode(self, x: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        ...

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, deterministic=False)
        return self.prediction_head(z)
```

### Design Rules

1. Do not duplicate CODE-AE encoder logic.
2. Do not implement loss inside model.
3. Do not store response matrices inside model.
4. Do not store drug names except optional metadata.
5. Model forward returns prediction logits / scores only.
6. If original CODE-AE training requires additional outputs, expose them through explicit methods rather than hidden side effects.

### Unit Tests

1. Forward output shape equals `[batch, n_drugs]`.
2. `encode()` output shape equals `[batch, latent_dim]`.
3. Model can be moved to CUDA.
4. Model can save and load state dict with matching `n_drugs`.

---

## 22. Model Module: Checkpoint Handling

### File

```text
codeae_multilabel/model/checkpoint.py
```

### Responsibility

Save and load pretrain and fine-tune checkpoints.

### API

```python
def load_pretrain_checkpoint(
    model: MultiLabelCodeAEModel,
    checkpoint_path: str,
    strict_encoder: bool = True,
    ignore_prediction_head: bool = True,
) -> dict:
    ...

def save_finetune_checkpoint(
    model: MultiLabelCodeAEModel,
    optimizer: torch.optim.Optimizer,
    metadata: dict,
    path: str,
) -> None:
    ...
```

### Fine-tune Loading Rule

1. Pretrain checkpoint is required.
2. Encoder / CODE-AE modules are loaded.
3. Single-drug predictor parameters from legacy checkpoints are ignored.
4. Multi-output head is initialized for `n_drugs`.
5. Any ignored / missing keys are logged to `checkpoint_load_report.json`.

### Unit Tests

1. Missing checkpoint raises.
2. Incompatible single-drug head is ignored without crashing.
3. Encoder weights are loaded.
4. Multi-output head shape matches `n_drugs`.
5. Load report includes missing and unexpected keys.

---

## 23. Training Module: Losses

### File

```text
codeae_multilabel/training/losses.py
```

### Responsibility

Implement task-specific masked supervised losses.

### API

```python
def safe_masked_mean(raw_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    ...

def masked_bce_with_logits(
    logits: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...

def masked_mae(
    pred: torch.Tensor,
    y: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    ...
```

### Classification Loss

```python
raw_loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
loss = safe_masked_mean(raw_loss, mask)
```

### Regression Loss

```python
raw_loss = torch.abs(pred - y)
loss = safe_masked_mean(raw_loss, mask)
```

### Target Label Rule

Target labels must not enter supervised training loss.

Therefore:

1. `masked_bce_with_logits` is used for source classification training.
2. `masked_mae` is used for source regression training.
3. Target response masks are used only by prediction / metrics modules.
4. Original CODE-AE target-domain adaptation losses can still use target omics, but not target response labels.

### All-zero Mask Rule

If `mask.sum() == 0`:

1. Return zero loss with gradient support, or
2. Let trainer skip that supervised component.

Recommended implementation:

```python
def safe_masked_mean(raw_loss, mask):
    denom = mask.sum().clamp_min(1.0)
    return (raw_loss * mask).sum() / denom
```

### Unit Tests

1. Masked positions do not contribute.
2. All-zero mask does not crash.
3. BCE result matches PyTorch BCE on valid positions.
4. MAE result matches manual MAE on valid positions.
5. Target labels are not referenced by trainer loss path.

---

## 24. Training Module: Trainer

### File

```text
codeae_multilabel/training/trainer.py
```

### Responsibility

Run fold-level fine-tuning and validation.

### API

```python
class CodeAEMultilabelTrainer:
    def __init__(
        self,
        model: MultiLabelCodeAEModel,
        optimizer: torch.optim.Optimizer,
        config: CodeAEMultilabelConfig,
        codeae_loss_adapter: Optional[Any] = None,
    ):
        ...

    def train_fold(
        self,
        fold: SourceFold,
        prepared: PreparedFineTuneData,
    ) -> TrainingResult:
        ...
```

### Training Inputs

1. Source train samples with source response matrix and mask.
2. Target omics samples for original CODE-AE domain adaptation / deconfounding.
3. No target supervised labels for parameter updates.

### Classification Training Loss

```text
loss_total = source_masked_bce
           + original_CODEAE_deconfounding_or_adversarial_losses
           + original_CODEAE_reconstruction_loss_if_applicable
```

### Regression Training Loss

```text
loss_total = source_masked_mae
           + original_CODEAE_deconfounding_or_adversarial_losses
           + original_CODEAE_reconstruction_loss_if_applicable
```

### Target Label Exclusion Rule

The trainer must not compute:

```text
target_masked_bce
```

or any equivalent target supervised response loss.

Target labels may be loaded into evaluation datasets, but they should not be passed into the loss computation path.

### Batch Strategy

Use paired source and target batches by default to preserve CODE-AE source-target training semantics.

```text
for source_batch, target_batch in paired_cycle(source_loader, target_loader):
    source_pred = model(source_batch.x)
    source_prediction_loss = masked_loss(source_pred, source_batch.y, source_batch.mask)

    codeae_losses = compute_original_codeae_losses(source_batch.x, target_batch.x)

    loss_total = source_prediction_loss + weighted_codeae_losses
    backward / optimizer.step
```

If source and target loaders have different lengths, cycle the shorter loader.

### Validation Rule

Validation must never call:

```python
loss.backward()
optimizer.step()
```

Validation should:

1. Run source validation prediction.
2. Build validation prediction long table.
3. Compute validation per-drug and summary metrics.
4. Pass summary metrics to `selection.py`.
5. Optionally compute target evaluation metrics without updating parameters.

### Unit Tests

1. One training step updates model parameters.
2. Validation step does not update model parameters.
3. Classification run computes source BCE.
4. Regression run computes source MAE.
5. Target labels are ignored during training.
6. Original CODE-AE loss adapter is called with source and target omics.
7. All-zero supervised mask does not crash.

---

## 25. Training Module: CODE-AE Loss Adapter

### File

```text
codeae_multilabel/training/codeae_loss_adapter.py
```

### Responsibility

Bridge original CODE-AE losses into the new trainer without tightly coupling trainer logic to legacy implementation details.

### API

```python
class CodeAELossAdapter:
    def compute_pretrain_losses(self, source_batch, target_batch, model) -> dict[str, torch.Tensor]:
        ...

    def compute_finetune_losses(self, source_batch, target_batch, model) -> dict[str, torch.Tensor]:
        ...
```

### Design Notes

1. The adapter is the only training module that should know original CODE-AE loss names.
2. If original CODE-AE uses method-specific losses such as adversarial, MMD, reconstruction, or context-aware deconfounding, expose them as named tensors.
3. The trainer only sums losses according to config / legacy defaults.
4. Do not alter the original weighting unless required to preserve CODE-AE behavior.

### Unit Tests

1. Adapter returns a dictionary of scalar tensors.
2. Missing optional loss returns empty dictionary or zero safely.
3. Trainer can run with a mock adapter.
4. Loss names are logged in train log.

---

## 26. Training Module: Early Stopping and Selection

### File

```text
codeae_multilabel/training/selection.py
```

### Responsibility

Choose best epoch using macro validation metric.

### API

```python
class MetricSelector:
    def __init__(self, task_type: str, requested_metric: Optional[str] = None):
        ...

    def select_metric(self, metrics_summary: pd.DataFrame) -> tuple[str, float, str]:
        ...

    def is_better(self, new_value: float, best_value: Optional[float], metric_name: str) -> bool:
        ...
```

### Classification Selection

Primary metric:

```text
macro_auroc
```

Fallback order:

```text
macro_aupr
macro_balanced_accuracy
macro_f1
```

Direction:

```text
higher is better
```

### Regression Selection

Primary metric:

```text
macro_mae
```

Direction:

```text
lower is better
```

Fallback order for regression if `macro_mae` is unavailable:

```text
macro_rmse
negative_macro_mae_if_available
```

The implementation should strongly prefer `macro_mae`; fallback should be rare and reported.

### Unit Tests

1. Classification selects `macro_auroc` when available.
2. Classification falls back when AUROC is NaN.
3. Regression selects `macro_mae`.
4. Regression treats lower value as better.
5. Selection report records chosen metric and fallback reason.

---

## 27. Training Module: Train State

### File

```text
codeae_multilabel/training/train_state.py
```

### Responsibility

Define training logs and state containers.

### Dataclasses

```python
@dataclass
class EpochLog:
    fold_id: int
    epoch: int
    source_prediction_loss: float
    codeae_loss_total: float
    total_loss: float
    val_metric_name: str
    val_metric_value: float
    selected_metric_name: str
    selected_metric_value: float

@dataclass
class FoldTrainState:
    fold_id: int
    best_epoch: int
    best_metric_name: str
    best_metric_value: float
    best_model_path: str
    epoch_logs: list[EpochLog]
```

### Unit Tests

1. Logs convert to DataFrame.
2. Missing optional losses serialize as NaN.
3. Best epoch is preserved.

---

## 28. Runner Module

### File

```text
codeae_multilabel/training/runners.py
```

### Responsibility

Orchestrate full pretrain and fine-tune workflows.

### PretrainRunner

```python
class PretrainRunner:
    def __init__(self, config: CodeAEMultilabelConfig):
        ...

    def run(self) -> None:
        ...
```

Responsibilities:

1. Initialize output directory.
2. Save config.
3. Prepare pretrain data.
4. Build original CODE-AE pretraining model / components.
5. Call original CODE-AE pretraining loop or adapter.
6. Save pretrain checkpoint.
7. Save feature alignment report.

### FineTuneRunner

```python
class FineTuneRunner:
    def __init__(self, config: CodeAEMultilabelConfig):
        ...

    def run(self) -> None:
        ...
```

Responsibilities:

1. Initialize output directory with overwrite behavior.
2. Save config.
3. Prepare fine-tune data.
4. Save `drug_list.csv`.
5. Build model and multi-output head.
6. Load pretrain checkpoint.
7. Loop over folds.
8. Train each fold.
9. Run prediction / metrics.
10. Export final latent and t-SNE.
11. Write fold summary.

### Unit Tests

1. Pretrain runner can execute with mocked CODE-AE adapter.
2. Fine-tune runner calls data preparation before model construction.
3. Fine-tune runner requires checkpoint.
4. Fine-tune runner writes drug list.
5. Fold loop writes one fold directory per fold.

---

## 29. Evaluation Module: Prediction

### File

```text
codeae_multilabel/evaluation/prediction.py
```

### Responsibility

Generate prediction matrices and long-format prediction tables.

### API

```python
def predict_matrix(
    model: MultiLabelCodeAEModel,
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
    domain: Literal["source", "target"],
    split: str,
    task_type: str,
    prediction_threshold: float,
    regression_binary_threshold: float,
    cancer_type_table: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    ...
```

### Export Policy

Export only observed positions by default:

```text
mask == 1
```

### Classification Output

| Column | Value |
|---|---|
| `pred_score` | raw logit |
| `probability` | sigmoid(logit) |
| `pred_label` | `probability >= 0.5` |

### Regression Output

For source regression:

| Column | Value |
|---|---|
| `pred_score` | continuous predicted score |
| `pred_label` | `pred_score >= regression_binary_threshold`, for optional binary analysis |
| `ground_truth_binary` | `ground_truth >= regression_binary_threshold` |

For target evaluation in regression run:

| Column | Value |
|---|---|
| `pred_score` | model output score |
| `probability` | sigmoid(pred_score), used as binary confidence |
| `pred_label` | `probability >= 0.5` |
| `ground_truth` | target binary label |

### Unit Tests

1. Prediction matrix shape equals `[n_samples, n_drugs]`.
2. Long table has one row per observed position.
3. Drug ID and drug index match `drug_list.csv`.
4. Classification probability is present.
5. Regression source table has continuous score and optional binary labels.
6. Target table uses binary evaluation semantics.

---

## 30. Evaluation Module: Metrics

### File

```text
codeae_multilabel/evaluation/metrics.py
```

### Responsibility

Compute per-drug and summary metrics from prediction long tables.

### API

```python
def compute_classification_metrics_per_drug(pred_df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_classification_metrics_summary(per_drug: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_regression_metrics_per_drug(pred_df: pd.DataFrame) -> pd.DataFrame:
    ...

def compute_regression_metrics_summary(per_drug: pd.DataFrame, pred_df: pd.DataFrame) -> pd.DataFrame:
    ...
```

### Classification Metrics

1. AUROC.
2. AUPR.
3. Accuracy.
4. F1.
5. Precision.
6. Recall.
7. Balanced accuracy.
8. Number of observed labels.
9. Number of positive labels.
10. Number of negative labels.

### Regression Metrics

1. MAE.
2. RMSE.
3. R2.
4. Pearson.
5. Spearman.
6. Optional binary classification metrics based on thresholded source response.
7. Number of observed labels.

### Summary Metrics

1. Macro average.
2. Micro average.
3. Weighted average.

### Invalid Metric Rule

If a metric cannot be computed for a drug:

1. Store `NaN`.
2. Do not crash.
3. Exclude `NaN` from macro average for that metric.
4. Record invalid reason in `metric_availability_report.csv`.

### Unit Tests

1. Per-drug metrics compute for valid binary labels.
2. Single-class AUROC becomes NaN.
3. Macro average ignores NaN.
4. Weighted average uses observed label counts.
5. Regression MAE is computed correctly.
6. Metrics module works without importing PyTorch.

---

## 31. Evaluation Module: Reports

### File

```text
codeae_multilabel/evaluation/reports.py
```

### Responsibility

Generate human-readable CSV reports about data and metrics.

### Reports

1. `data_alignment_report.csv`
2. `feature_alignment_report.csv`
3. `drug_availability_report.csv`
4. `source_split_label_distribution.csv`
5. `metric_availability_report.csv`
6. `checkpoint_load_report.json`
7. `run_manifest.json`

### Unit Tests

1. Report functions accept dataclasses and return DataFrames.
2. Report functions do not write files directly.
3. Missing optional metadata is reported clearly.

---

## 32. Export Module: ArtifactWriter

### File

```text
codeae_multilabel/export/artifacts.py
```

### Responsibility

Centralize writing all output files.

### API

```python
class ArtifactWriter:
    def __init__(self, output_dir: str, overwrite: bool = True):
        ...

    def write_config(self, config: CodeAEMultilabelConfig) -> None:
        ...

    def write_drug_list(self, drug_index: DrugIndex) -> None:
        ...

    def write_fold_predictions(self, fold_id: int, bundle: PredictionBundle) -> None:
        ...

    def write_fold_training_result(self, result: TrainingResult) -> None:
        ...

    def write_latent(self, fold_id: int, domain: str, latent_obj: dict) -> None:
        ...

    def write_plot(self, fold_id: int, name: str, fig_or_path: Any) -> None:
        ...
```

### Design Rules

1. Modules should return DataFrames / objects, not write files themselves.
2. Writer owns path naming.
3. Writer creates fold directories.
4. Writer follows overwrite default behavior.
5. Writer records a manifest of generated artifacts.

### Unit Tests

1. Writer creates expected directory tree.
2. Writer overwrites by default.
3. Writer writes fold prediction files to the correct path.
4. Manifest lists generated files.

---

## 33. Export Module: Latent Export

### File

```text
codeae_multilabel/export/latent.py
```

### Responsibility

Export final sample-level latent representations after fine-tuning.

### API

```python
def extract_latent_table(
    model: MultiLabelCodeAEModel,
    omics: OmicsTable,
    batch_size: int,
    device: str,
) -> pd.DataFrame:
    ...
```

### Output Format

Preferred DataFrame format:

| sample_id | domain | latent_0 | latent_1 | ... |
|---|---|---:|---:|---:|
| sample_A | source | 0.12 | -0.44 | ... |
| sample_B | target | 0.01 | 0.29 | ... |

Also allowed for compatibility:

```python
{
  "sample_id_1": [latent_0, latent_1, ...],
  "sample_id_2": [latent_0, latent_1, ...],
}
```

### Timing Rule

Latent export must happen only after:

1. Fine-tuning finishes.
2. Best checkpoint is loaded.
3. Model is set to eval mode.
4. Deterministic encoder path is active.

### Unit Tests

1. Latent table row count equals omics sample count.
2. Latent columns match latent dimension.
3. Source and target latent are sample-level only.
4. Latent export uses eval mode.

---

## 34. Export Module: Visualization

### File

```text
codeae_multilabel/export/visualization.py
```

### Responsibility

Generate t-SNE plots and optional latent visualizations.

### API

```python
def run_tsne(latent_df: pd.DataFrame, seed: int) -> pd.DataFrame:
    ...

def plot_tsne_by_domain(tsne_df: pd.DataFrame, output_path: str) -> None:
    ...

def plot_tsne_by_cancer_type(tsne_df: pd.DataFrame, output_path: str) -> None:
    ...
```

### Failure Policy

Visualization should not crash the whole training run.

Skip with warning if:

1. Number of samples is too small.
2. t-SNE perplexity cannot be set safely.
3. Cancer type metadata is missing.
4. All cancer types are `Unknown`.
5. Latent contains NaN or infinite values.

### Unit Tests

1. t-SNE output has two coordinates.
2. Small sample input returns warning and skips.
3. Missing cancer type skips cancer plot only.
4. Domain plot works with source and target labels.

---

## 35. Export Module: Latent Evaluation

### File

```text
codeae_multilabel/export/latent_eval.py
```

### Responsibility

Compute optional latent distribution and clustering metrics.

### Candidate Metrics

1. MMD between source and target latent.
2. Wasserstein distance summary.
3. KMeans cancer type clustering metrics, if cancer type labels are available.
4. Silhouette score, if labels and sample counts are valid.

### Failure Policy

If metrics cannot be computed:

1. Return `NaN`.
2. Record reason.
3. Do not interrupt training.

### Unit Tests

1. Metrics return finite values on valid input.
2. Missing cancer type produces NaN and reason.
3. Single-class clustering labels do not crash.

---

## 36. Output Directory Structure

Default output structure:

```text
{output_dir}/
  config.json
  run_manifest.json
  drug_list.csv
  feature_alignment_report.csv
  data_alignment_report.csv
  drug_availability_report.csv

  pretrain/
    checkpoint.pt
    pretrain_config.json
    pretrain_log.csv
    feature_alignment_report.csv

  fold_0/
    best_model.pt
    train_log.csv
    selection_report.csv
    checkpoint_load_report.json

    source_prediction_results.csv
    target_prediction_results.csv

    source_metrics_per_drug.csv
    target_metrics_per_drug.csv
    source_metrics_summary.csv
    target_metrics_summary.csv
    metric_availability_report.csv

    source_latent_representation.pkl
    target_latent_representation.pkl
    source_latent_representation.csv
    target_latent_representation.csv

    tsne_domain_mixing.png
    tsne_cancer_type.png
    latent_distribution_metrics.csv
    kmeans_cancer_type_metrics.csv

  fold_1/
    ...

  fold_summary.csv
```

### Output Rules

1. `output_dir` is overwritten by default.
2. Fold directories are named `fold_0`, `fold_1`, etc.
3. No single-drug nested directories are used.
4. All outputs should include `fold`, `seed`, `task_type`, and `n_drugs` where relevant.
5. Prediction files are long tables, not wide matrices.
6. Latent files are sample-level, not sample-drug-level.

---

## 37. CLI Design

### 37.1 Pretrain Entry

```bash
python benchmark/CODEAE/pretrain_multilabel_hyper_main.py \
  --source_omics_path path/to/source_omics.csv \
  --target_omics_path path/to/target_omics.csv \
  --method code_adv \
  --epochs 100 \
  --batch_size 128 \
  --lr 1e-4 \
  --seed 42 \
  --output_dir outputs_codeae_multilabel \
  --overwrite
```

### 37.2 Fine-tune Entry: Classification

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
  --method code_adv \
  --metric macro_auroc \
  --n_splits 5 \
  --epochs 50 \
  --output_dir outputs_codeae_multilabel \
  --overwrite
```

### 37.3 Fine-tune Entry: Regression

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --task_type regression \
  --source_omics_path path/to/source_omics.csv \
  --target_omics_path path/to/target_omics.csv \
  --source_response_path path/to/source_response.csv \
  --target_response_path path/to/target_response.csv \
  --source_sample_col Sample_ID \
  --target_sample_col tissue_id \
  --target_response_sample_col Patient_id \
  --drug_col drug_name \
  --source_response_col neg_log2_auc \
  --target_response_col Label \
  --pretrain_checkpoint outputs_codeae_multilabel/pretrain/checkpoint.pt \
  --method code_adv \
  --metric macro_mae \
  --reg_loss mae \
  --regression_binary_threshold 1.0 \
  --n_splits 5 \
  --epochs 50 \
  --output_dir outputs_codeae_multilabel_regression \
  --overwrite
```

### 37.4 Required Fine-tune Arguments

```text
--task_type classification|regression
--source_omics_path
--target_omics_path
--source_response_path
--target_response_path
--drug_col
--source_response_col
--target_response_col
--pretrain_checkpoint
--output_dir
```

### 37.5 Removed / Not Supported Arguments

```text
--drug
--n_shot
--use_target_supervision
```

Rationale:

1. `--drug` conflicts with multi-drug model design.
2. CODE-AE multilabel does not use n-shot.
3. Target labels are evaluation-only.

---

## 38. Pretraining Workflow

### Runner Steps

```text
PretrainRunner.run()
  1. set_global_seed(config.seed)
  2. ArtifactWriter initializes output_dir
  3. write config.json
  4. prepare_pretrain_data(config)
  5. write feature_alignment_report.csv
  6. build original CODE-AE pretraining components
  7. run original CODE-AE pretraining loop via adapter
  8. save pretrain/checkpoint.pt
  9. save pretrain/pretrain_log.csv
```

### Pretrain Data Use

Pretraining uses:

1. Source omics.
2. Target omics.
3. Domain labels if original CODE-AE needs them.

Pretraining does not use:

1. Source response.
2. Target response.
3. Drug list.
4. Drug prediction head.

### Acceptance Checks

1. Pretrain can run without response files.
2. Pretrain checkpoint exists.
3. Checkpoint contains encoder / CODE-AE module weights.
4. Feature alignment report exists.

---

## 39. Fine-tuning Workflow

### Runner Steps

```text
FineTuneRunner.run()
  1. set_global_seed(config.seed)
  2. ArtifactWriter initializes output_dir with overwrite=True by default
  3. write config.json
  4. prepare_finetune_data(config)
  5. write drug_list.csv and data reports
  6. build CODE-AE core through legacy adapter
  7. create MultiOutputDrugHead(latent_dim, n_drugs)
  8. compose MultiLabelCodeAEModel
  9. load pretrain checkpoint, ignoring incompatible predictor head
 10. for each SourceFold:
       a. build loaders
       b. train fold
       c. save best model
       d. run source and target prediction
       e. compute metrics
       f. load best checkpoint
       g. export latent
       h. generate t-SNE and latent metrics
       i. write fold artifacts
 11. write fold_summary.csv
```

### Fine-tuning Data Use

Training uses:

1. Source omics.
2. Source response labels.
3. Source response masks.
4. Target omics for original CODE-AE adaptation / deconfounding.

Evaluation uses:

1. Source labels.
2. Target labels.
3. Source and target masks.

Training does not use:

1. Target response labels.
2. Target response masks as supervised loss masks.
3. Sample-drug pair expansion.
4. Drug embeddings.

---

## 40. Prediction Output Schema

### 40.1 Shared Columns

| Column | Required | Description |
|---|---|---|
| `sample_id` | Yes | Source or target sample ID |
| `drug_id` | Yes | Drug name / ID |
| `drug_index` | Yes | Output column index |
| `domain` | Yes | `source` or `target` |
| `split` | Yes | `source_train`, `source_val`, `source_test`, or `target_eval` |
| `fold` | Yes | Fold ID |
| `seed` | Yes | Random seed |
| `task_type` | Yes | `classification` or `regression` |
| `ground_truth` | Yes | Observed label / response |
| `mask` | Yes | Should be 1 for exported rows |
| `pred_score` | Yes | Raw model output |
| `pred_label` | Task-dependent | Binary prediction if applicable |
| `probability` | Task-dependent | Sigmoid output for classification-style evaluation |
| `cancer_type` | Optional | Cancer type if available |

### 40.2 Classification Rows

```text
probability = sigmoid(pred_score)
pred_label = probability >= prediction_threshold
```

### 40.3 Regression Source Rows

```text
pred_score = continuous model output
ground_truth = continuous source response
ground_truth_binary = ground_truth >= regression_binary_threshold
pred_label = pred_score >= regression_binary_threshold
```

### 40.4 Regression Target Rows

```text
pred_score = raw model score
probability = sigmoid(pred_score)
pred_label = probability >= prediction_threshold
ground_truth = target binary label
```

---

## 41. Metrics Output Schema

### 41.1 Per-drug Metrics

Suggested columns:

| Column | Description |
|---|---|
| `drug_id` | Drug name |
| `drug_index` | Output index |
| `domain` | source / target |
| `split` | validation / test / target_eval |
| `n_observed` | Number of observed labels |
| `n_positive` | Number of positive labels, if binary |
| `n_negative` | Number of negative labels, if binary |
| `auroc` | Classification AUROC |
| `aupr` | Classification AUPR |
| `accuracy` | Classification accuracy |
| `f1` | Classification F1 |
| `precision` | Classification precision |
| `recall` | Classification recall |
| `balanced_accuracy` | Classification balanced accuracy |
| `mae` | Regression MAE |
| `rmse` | Regression RMSE |
| `r2` | Regression R2 |
| `pearson` | Regression Pearson |
| `spearman` | Regression Spearman |
| `valid_metric_flags` | Optional serialized flags |

### 41.2 Summary Metrics

Suggested columns:

| Column | Description |
|---|---|
| `domain` | source / target |
| `split` | validation / test / target_eval |
| `metric_name` | e.g. `macro_mae` |
| `metric_value` | computed value |
| `aggregation` | macro / micro / weighted |
| `n_valid_drugs` | number of drugs included |
| `n_observed_positions` | total observed positions |

---

## 42. Smoke Test Design

### Principle

No synthetic mini dataset builder is required. Instead, use real input files with small epoch counts.

### 42.1 Classification Smoke Run

Purpose:

1. Confirm data loading.
2. Confirm drug union.
3. Confirm wide matrix + mask generation.
4. Confirm multi-output head shape.
5. Confirm 1-2 epoch training.
6. Confirm prediction, metrics, latent, and t-SNE artifact writing.

Example:

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --task_type classification \
  --epochs 1 \
  --n_splits 2 \
  --batch_size 32 \
  --output_dir outputs_codeae_smoke_classification \
  --overwrite \
  ... required real data paths ...
```

### 42.2 Regression Smoke Run

Purpose:

1. Confirm source continuous MAE loss.
2. Confirm target binary evaluation only.
3. Confirm `macro_mae` selection.
4. Confirm thresholded source binary analysis.
5. Confirm missing labels do not enter loss.

Example:

```bash
python benchmark/CODEAE/drug_ft_multilabel_hyper_main.py \
  --task_type regression \
  --epochs 1 \
  --n_splits 2 \
  --batch_size 32 \
  --metric macro_mae \
  --reg_loss mae \
  --output_dir outputs_codeae_smoke_regression \
  --overwrite \
  ... required real data paths ...
```

### 42.3 Smoke Acceptance Criteria

1. Command exits successfully.
2. `drug_list.csv` exists.
3. `fold_0/best_model.pt` exists.
4. Prediction CSVs exist.
5. Metrics CSVs exist.
6. Latent files exist after fine-tuning.
7. t-SNE generation either succeeds or records a warning without failing.

---

## 43. Unit Test Plan by Module

| Module | Test Focus |
|---|---|
| `config.py` | CLI defaults, required args, serialization |
| `io.py` | overwrite behavior, file round-trip |
| `validators.py` | missing columns, invalid labels, invalid masks |
| `omics.py` | feature alignment and sample order |
| `sample_id.py` | TCGA join key and duplicate detection |
| `drug_index.py` | source ∪ target union and deterministic order |
| `response_matrix.py` | long-to-wide conversion, masks, duplicate strategy |
| `split.py` | disjoint sample-level folds |
| `dataset.py` | tensor shapes and sample IDs |
| `legacy_adapter.py` | latent extraction interface |
| `heads.py` | output shape `[batch, n_drugs]` |
| `wrapper.py` | encoder + head composition |
| `checkpoint.py` | ignore incompatible single-drug head |
| `losses.py` | masked BCE and masked MAE |
| `trainer.py` | target labels excluded from training |
| `selection.py` | macro AUROC and macro MAE selection |
| `prediction.py` | long table schema |
| `metrics.py` | per-drug and summary metrics |
| `artifacts.py` | directory tree and overwrite |
| `latent.py` | sample-level latent export |
| `visualization.py` | skip-on-warning behavior |

---

## 44. Implementation Phases

### Phase 1: CODE-AE Source Mapping

Goal: identify legacy CODE-AE components that must be reused.

Tasks:

1. Locate pretrain model construction.
2. Locate fine-tune model construction.
3. Locate encoder / latent extraction path.
4. Locate deconfounding / adversarial / reconstruction losses.
5. Locate checkpoint format.
6. Locate current single-drug prediction head.
7. Locate current early stopping implementation.

Output:

```text
CODE-AE current architecture map
legacy component mapping table
```

### Phase 2: Data Contracts and Preparation

Tasks:

1. Implement dataclasses.
2. Implement omics alignment.
3. Implement drug union.
4. Implement response long-to-wide.
5. Implement source sample folds.
6. Implement data reports.

### Phase 3: Model Adapter and Multi-output Head

Tasks:

1. Implement legacy adapter.
2. Implement multi-output head.
3. Implement wrapper model.
4. Implement checkpoint load rules.
5. Confirm incompatible single-drug head is ignored.

### Phase 4: Masked Loss, Trainer, and Selection

Tasks:

1. Implement masked BCE.
2. Implement masked MAE.
3. Implement target-label-excluded trainer.
4. Implement macro metric selection.
5. Implement logging.

### Phase 5: Evaluation and Export

Tasks:

1. Implement prediction long tables.
2. Implement metrics.
3. Implement artifact writer.
4. Implement latent export.
5. Implement t-SNE and latent metrics.

### Phase 6: Small-epoch Smoke Runs

Tasks:

1. Classification smoke run with real data paths.
2. Regression smoke run with real data paths.
3. Confirm output artifacts.
4. Confirm no target label supervised training.
5. Confirm overwrite behavior.

---

## 45. Risks and Mitigations

| Risk | Impact | Mitigation |
|---|---|---|
| Original CODE-AE checkpoint includes single-drug head | Fine-tune load failure | Ignore incompatible prediction head keys and log report |
| Original CODE-AE latent path is not obvious | Latent export may be wrong | Isolate extraction in `legacy_adapter.py` and test shape / determinism |
| Target-only drugs have no source supervised signal | Poor or unstable prediction for target-only drugs | Keep them in output, report low training availability, do not drop silently |
| Sparse labels cause invalid AUROC / AUPR | Early stopping instability | Use metric fallback and availability report |
| Regression output reused for target binary evaluation | Calibration ambiguity | Report clearly; target labels remain evaluation-only |
| Large drug count increases output head size | Memory pressure | User VRAM is sufficient; still log `n_drugs` and model parameter count |
| t-SNE fails on small sample count | Run interruption | Skip with warning |
| Existing output overwritten | Loss of old run | This is intended default; preserve `run_manifest.json` |

---

## 46. Acceptance Criteria

A correct implementation should satisfy:

1. New pretrain entry runs without response tables.
2. New fine-tune entry requires pretrain checkpoint.
3. Fine-tune creates `drug_list.csv` from source ∪ target drugs.
4. Model output shape is `[batch_size, n_drugs]`.
5. Source classification uses masked BCE.
6. Source regression uses masked MAE.
7. Target labels are not used for supervised training.
8. Target labels are used for evaluation only.
9. Missing source labels do not contribute to loss.
10. Early stopping uses macro AUROC for classification.
11. Early stopping uses macro MAE for regression.
12. Source and target predictions are exported as long tables.
13. Per-drug and summary metrics are exported.
14. Final latent representations are exported after fine-tuning.
15. t-SNE runs after latent export or skips with warning.
16. Output directory is overwritten by default.
17. Small-epoch classification and regression smoke runs complete.
18. The implementation does not introduce sample-drug pair modeling.
19. The implementation does not introduce drug latent.
20. The implementation does not rewrite CODE-AE encoder / deconfounding architecture.

---

## 47. Relationship to Existing Multilabel SSDA Design

This CODE-AE design borrows the following engineering lessons from the existing multilabel SSDA design:

1. Use long response tables as input.
2. Convert response data into wide matrices plus masks.
3. Build a deterministic drug union from source ∪ target.
4. Save `drug_list.csv` to preserve output column semantics.
5. Use masked loss instead of filling missing labels as real zeros.
6. Export prediction results as long tables for analysis.
7. Compute per-drug metrics and macro / micro / weighted summaries.
8. Keep latent representation sample-level.
9. Export latent and t-SNE only after training is complete.
10. Keep the new multilabel pipeline separate from the legacy single-drug pipeline.

However, this design intentionally differs from multilabel SSDA in these ways:

| Aspect | Multilabel SSDA | Multilabel CODE-AE |
|---|---|---|
| Training structure | SSDA-specific single-stage / semi-supervised training | CODE-AE two-stage pretrain -> fine-tune |
| Target labels | May support target labeled supervised components | Evaluation-only in this design |
| n-shot | Supported in SSDA design | Not supported |
| Adaptation | SSDA adaptation logic | Original CODE-AE deconfounding / adversarial logic |
| Encoder | SSDA encoder | CODE-AE encoder |
| Main modification | SSDA single-drug to multilabel | CODE-AE single-drug head to multilabel head |

---

## 48. Summary

The proposed CODE-AE multilabel architecture is built around a strict separation of responsibilities:

```text
data preparation -> model adapter -> masked source-supervised training -> evaluation -> final latent export
```

The most important implementation constraints are:

1. Keep CODE-AE's original encoder and deconfounding training philosophy.
2. Add a new multi-output prediction head instead of building a sample-drug pair model.
3. Use source labels for supervised training and target labels only for evaluation.
4. Use masked BCE for classification and masked MAE for regression.
5. Use macro AUROC for classification early stopping and macro MAE for regression early stopping.
6. Make each module independently testable and avoid cross-layer dependencies.
7. Export all final outputs after loading the best fine-tuned checkpoint.

This document should be used as the architecture reference for implementing:

```text
docs/design_codeae.md
benchmark/CODEAE/codeae_multilabel/
benchmark/CODEAE/pretrain_multilabel_hyper_main.py
benchmark/CODEAE/drug_ft_multilabel_hyper_main.py
```
