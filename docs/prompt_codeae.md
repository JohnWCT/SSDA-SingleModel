# Automated Development Prompt: Multi-label / Multi-drug CODE-AE

> **Purpose:** This prompt instructs an autonomous coding agent system to implement the CODE-AE multi-label / multi-drug single-model framework according to `proposal_codeae.md` and `design_codeae.md`.
>
> **Primary source documents:**
>
> 1. `docs/proposal_codeae.md`
> 2. `docs/design_codeae.md`
>
> If there is any conflict between this prompt and the source documents, apply the following priority order:
>
> 1. This prompt's non-negotiable automation and testing rules.
> 2. `docs/design_codeae.md`.
> 3. `docs/proposal_codeae.md`.
> 4. Existing repository conventions.

---

## 0. Highest-priority Non-negotiable Rules

The entire development process must be **fully automatic**.

The agent system must not ask the user to manually inspect files, manually decide implementation details, manually patch code, manually run commands, manually prepare test data, or manually fix errors.

If an ambiguity is encountered, the Main Agent must:

1. Resolve it conservatively using `proposal_codeae.md` and `design_codeae.md`.
2. Prefer designs that preserve original CODE-AE behavior.
3. Prefer modularity and testability over shortcut implementation.
4. Record the decision in `docs/implementation_report_codeae.md`.
5. Continue implementation automatically.

The system must not stop with an incomplete implementation and return work to the user. It must iterate until all required checks pass.

### 0.1 Testing Rule with Highest Weight

Every line of newly added or modified **business logic** must be accompanied by corresponding unit tests.

Operational interpretation:

- Every public function must have tests.
- Every important private helper must have tests if it contains non-trivial logic.
- Every branch must be tested.
- Every edge case must be tested.
- Every error path must be tested.
- Every data contract must be tested.
- Every metric formula must be tested numerically.
- Every masked-loss behavior must be tested numerically.
- Every artifact schema must be tested.
- Every CLI argument contract must be tested.

Do not interpret this as writing one test assertion per physical line of code. Interpret it as requiring behavioral coverage of all business logic represented by those lines.

### 0.2 Anti-cheating Rule

The agents must not fake compliance.

Forbidden actions:

1. Do not skip core tests.
2. Do not weaken assertions to make tests pass.
3. Do not mock away the main business logic under test.
4. Do not test only tensor shapes when numeric behavior matters.
5. Do not mark failing tests as xfail unless the feature is explicitly out of scope in `design_codeae.md`.
6. Do not remove existing tests to make the suite pass.
7. Do not silence warnings that indicate real bugs.
8. Do not bypass lint, formatting, type checking, or coverage gates.
9. Do not declare success until all required checks pass 100%.

---

## 1. Roles

The agent system must operate with two explicit roles.

## 1.1 Main Agent

The Main Agent is responsible for the complete development lifecycle.

Responsibilities:

1. Read and internalize `docs/proposal_codeae.md` and `docs/design_codeae.md` before coding.
2. Inspect the existing repository structure and current CODE-AE implementation.
3. Prepare the environment.
4. Build a task dependency graph from the design document.
5. Assign implementation tasks to Sub Agents by module.
6. Enforce module boundaries and dependency rules.
7. Integrate patches from Sub Agents.
8. Run all required checks.
9. Diagnose failing checks.
10. Dispatch fixes automatically.
11. Maintain implementation notes and decision logs.
12. Ensure final artifacts match the design document.
13. Produce `docs/implementation_report_codeae.md`.

The Main Agent must not directly pack all business logic into entry scripts. It must enforce the modular architecture in `design_codeae.md`.

## 1.2 Sub Agents

Sub Agents implement concrete modules.

Each Sub Agent must:

1. Read the relevant parts of `docs/proposal_codeae.md` and `docs/design_codeae.md`.
2. Implement only its assigned module.
3. Respect allowed and forbidden dependencies.
4. Add or update tests at the same time as code.
5. Verify local tests before reporting completion.
6. Produce a module-level checklist for the Main Agent.

Sub Agents must not change unrelated modules unless the Main Agent explicitly assigns that integration task.

---

## 2. Target Implementation Summary

Implement a new CODE-AE multi-label / multi-drug single-model pipeline.

The original workflow is single-drug oriented:

```bash
cd benchmark/CODEAE/
python pretrain_hyper_main.py --drug "Gefitinib"
python drug_ft_hyper_main.py --drug "Gefitinib"
```

The new workflow must preserve CODE-AE's two-stage philosophy:

1. Stage 1: encoder / deconfounding pretraining.
2. Stage 2: drug response fine-tuning with a multi-output prediction head.

The new model must be:

```text
omics data table
  -> CODE-AE encoder / deconfounding module
  -> sample-level latent representation
  -> multi-output drug prediction head
  -> all-drug response vector [n_drugs]
```

Required design constraints:

1. Use one shared CODE-AE model for multiple drugs.
2. Do not implement a sample-drug pair model.
3. Do not implement drug latent embeddings.
4. Do not make domain adaptation drug-specific.
5. Preserve CODE-AE encoder and adversarial/deconfounding logic as much as possible.
6. Replace only the drug prediction interface with a multi-output head.
7. Use wide response matrices plus masks for sparse drug labels.
8. Treat target labels as evaluation-only.
9. Export final sample-level latent representations only after fine-tuning is complete.
10. Use modular, low-coupling architecture.

---

## 3. Required File and Package Layout

Implement the new code in an independent package:

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

The Main Agent may adjust filenames only if the existing repository structure makes a name impossible, but the final implementation must preserve the same conceptual modules.

---

## 4. Module Boundaries and Dependency Rules

Allowed direction:

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

Forbidden dependencies:

| Module | Must Not Import | Reason |
|---|---|---|
| `data/*` | `training/*`, `model/*` | Data preparation must be independently testable |
| `evaluation/metrics.py` | PyTorch model classes | Metrics must consume arrays/tables only |
| `training/losses.py` | file writers or metrics modules | Loss functions must remain pure tensor functions |
| `model/*` | CSV readers or response table logic | Model code must not know file formats |
| `export/artifacts.py` | trainer internals | Writer consumes explicit output objects only |
| legacy CODE-AE modules | `codeae_multilabel/*` | Legacy code must not depend on new pipeline |
| entry scripts | business logic | Entry scripts must remain thin |

The Main Agent must verify these dependency rules during code review and testing.

---

## 5. Entry Points

## 5.1 Pretrain Entry

Create:

```text
benchmark/CODEAE/pretrain_multilabel_hyper_main.py
```

Expected CLI style:

```bash
python benchmark/CODEAE/pretrain_multilabel_hyper_main.py \
  --source_omics_path path/to/source_omics.csv \
  --target_omics_path path/to/target_omics.csv \
  --output_dir outputs_codeae_multilabel
```

Responsibilities:

1. Parse arguments.
2. Build typed config object.
3. Set seed.
4. Instantiate `PretrainRunner`.
5. Call `runner.run()`.

The pretrain entry must not read drug response files.

## 5.2 Fine-tune Entry

Create:

```text
benchmark/CODEAE/drug_ft_multilabel_hyper_main.py
```

Expected CLI style:

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

Responsibilities:

1. Parse arguments.
2. Build typed config object.
3. Set seed.
4. Instantiate `FineTuneRunner`.
5. Call `runner.run()`.

The fine-tune entry must not directly implement response matrix construction, model training, metrics, or artifact writing.

---

## 6. Core Data and Training Semantics

## 6.1 Drug List

The fine-tune stage must automatically build the drug list from:

```text
source_drugs ∪ target_drugs
```

Rules:

1. Preserve source-only drugs.
2. Preserve target-only drugs.
3. Write `drug_list.csv`.
4. Use `drug_list.csv` order as the canonical output dimension order.
5. Never allow prediction columns to drift from drug order.

## 6.2 Response Matrix and Mask

Convert source and target response long tables into wide matrices:

```text
Y_source: [n_source_samples, n_drugs]
M_source: [n_source_samples, n_drugs]
Y_target: [n_target_samples, n_drugs]
M_target: [n_target_samples, n_drugs]
```

`M_*` is 1 where the label is observed and 0 where missing.

Missing labels must not be filled with 0 and treated as real labels.

## 6.3 Classification Loss

Use masked `BCEWithLogitsLoss`.

Only observed source labels contribute to supervised training loss.

Target labels must not contribute to supervised training loss.

## 6.4 Regression Loss

Use masked MAE for regression training unless the existing CODE-AE regression implementation clearly requires an equivalent legacy loss.

Early stopping metric for regression is:

```text
macro_mae
```

Direction:

```text
lower is better
```

## 6.5 Target Labels

Target labels are evaluation-only.

The implementation must not include a `--use_target_supervision` option unless explicitly required by the design document in a future revision.

The target response table is used for:

1. Target prediction result annotation.
2. Target per-drug metrics.
3. Target summary metrics.
4. Target regression-to-classification evaluation when applicable.

It is not used for supervised optimization.

## 6.6 Regression Target Classification Evaluation

For regression runs, target continuous labels must be converted to binary classification labels for target classification evaluation.

Use:

```text
neg_log2_auc >= 1.0
```

unless the design document explicitly specifies a different threshold in a future revision.

## 6.7 Domain Adaptation

Domain adaptation / adversarial / deconfounding losses are shared sample-level losses.

Do not make them drug-specific.

---

## 7. Metrics and Model Selection

## 7.1 Metrics Required

The implementation must output per-drug metrics and summary metrics.

For classification, include where valid:

1. AUROC.
2. AUPR.
3. F1.
4. Balanced accuracy.
5. Accuracy.
6. Precision.
7. Recall.

For regression, include where valid:

1. MAE.
2. RMSE.
3. R².
4. Pearson correlation.
5. Spearman correlation.

Invalid metrics must be recorded as `NaN`, not crash training.

## 7.2 Summary Metrics

Summary metrics must include all of the following:

1. `macro`: unweighted mean across valid drugs.
2. `micro`: metric computed by pooling valid sample-drug observations when mathematically appropriate.
3. `weighted`: drug-level average weighted by each drug's valid observed label count.
4. `overall`: ignore drug/category boundaries and treat all valid observed sample-label pairs as one single collection.
5. `mean` and `std` across folds for all supported summary types.

The `overall` score is mandatory. It represents the user's requirement:

```text
不管類別、全部樣本視為單一集合的 overall score
```

## 7.3 Model Selection

Classification early stopping:

1. Primary: `macro_auroc`.
2. Fallback order: `macro_aupr`, `macro_balanced_accuracy`, `macro_f1`.
3. Direction: higher is better.

Regression early stopping:

1. Primary: `macro_mae`.
2. Direction: lower is better.

The early stopping utility must be independent and testable.

---

## 8. Artifacts and Output Contract

Use an `ArtifactWriter` module. Other modules must not write files directly unless they are low-level IO utilities called by `ArtifactWriter`.

The pipeline must write artifacts under `--output_dir`.

Default behavior:

```text
overwrite existing output_dir
```

Required artifacts include:

```text
output_dir/
  config.json
  drug_list.csv
  pretrain/
    checkpoint.pt
    training_log.csv
  fold_0/
    best_model.pt
    source_prediction_results.csv
    target_prediction_results.csv
    metrics_per_drug.csv
    metrics_summary.csv
    source_latent.csv
    target_latent.csv
    tsne_domain.csv
    tsne_cancer_type.csv        # optional when metadata is available
    training_log.csv
  fold_summary/
    metrics_summary_mean_std.csv
  implementation_report_codeae.md
```

If the final exact filenames differ due to repository conventions, the information content must remain equivalent and the deviation must be documented in `implementation_report_codeae.md`.

## 8.1 Latent and t-SNE Timing

Latent representation and t-SNE export must occur after fine-tuning finishes and the best checkpoint is loaded.

Do not use pretrain latent as the final reported CODE-AE latent.

If sample count is too small for t-SNE or metadata is missing:

1. Warn.
2. Skip only the affected visualization artifact.
3. Continue pipeline.

---

## 9. Testing Requirements

## 9.1 Unit Tests

Each module must have corresponding unit tests.

Minimum required test targets:

1. Config parsing and validation.
2. Dataclass contracts.
3. Omics feature alignment.
4. Sample ID normalization.
5. Drug index generation.
6. Long-to-wide response matrix construction.
7. Mask construction.
8. Source-only and target-only drug handling.
9. Duplicate response handling, if implemented.
10. Classification masked BCE numeric behavior.
11. Regression masked MAE numeric behavior.
12. Empty-mask behavior.
13. Multi-output head output shape and dtype.
14. Checkpoint loading while ignoring incompatible single-drug heads.
15. Prediction table construction and drug order correctness.
16. Classification metrics numeric behavior.
17. Regression metrics numeric behavior.
18. Invalid metric -> `NaN` behavior.
19. Macro / micro / weighted / overall summary metrics.
20. Fold mean/std summary.
21. Early stopping direction and fallback behavior.
22. Artifact writer schemas.
23. CLI argument validation.
24. Target labels are not included in supervised training loss.
25. Latent export runs only after best checkpoint is available.
26. t-SNE skip behavior for insufficient samples.

## 9.2 Integration Tests

Implement integration tests for:

1. Pretrain entry can parse config and initialize runner.
2. Fine-tune entry can parse config and initialize runner.
3. Fine-tune requires pretrain checkpoint.
4. Multi-output prediction head is attached correctly.
5. ArtifactWriter writes required outputs.
6. Metrics summary includes `overall`.

## 9.3 Smoke Tests

Do not build a synthetic mini dataset builder.

Smoke tests should use real available data paths in the repository when present, with small-epoch settings.

Required smoke coverage:

1. Classification 1 epoch.
2. Regression 1 epoch.
3. Pretrain checkpoint -> fine-tune load.
4. Latent export after fine-tune.
5. t-SNE export or graceful skip.

The implementation may add CLI parameters for smoke/debug execution, such as:

```text
--max_samples
--max_drugs
--epochs 1
--folds 1
--debug
```

These parameters must not alter the default full-run behavior.

---

## 10. Required Check Tools

All code must pass 100% of the following checks before completion can be declared.

```bash
python -m pytest
ruff check .
ruff format --check .
mypy benchmark/CODEAE/codeae_multilabel
pytest --cov=benchmark/CODEAE/codeae_multilabel --cov-report=term-missing --cov-fail-under=80
```

Coverage threshold:

```text
80%
```

If the repository does not already contain configuration for `ruff`, `mypy`, or coverage, the Main Agent may add minimal configuration in `pyproject.toml` or equivalent config files.

Configuration must be strict enough to catch real issues but must not require large unrelated rewrites of legacy CODE-AE files.

Recommended scope:

1. Apply strict checks to `benchmark/CODEAE/codeae_multilabel`.
2. Keep legacy CODE-AE checks limited unless changes are made there.
3. Do not hide new package errors by excluding new code from tools.

---

## 11. Task Decomposition for Sub Agents

The Main Agent should create a task DAG and assign Sub Agents in roughly this order.

## 11.1 Sub Agent A: Contracts and Config

Implement:

1. `contracts.py`.
2. `config.py`.
3. `seed.py`.
4. `validators.py`.

Must define typed dataclasses for:

1. Experiment config.
2. Pretrain config.
3. Fine-tune config.
4. Drug index.
5. Prepared pretrain data.
6. Prepared fine-tune data.
7. Training result.
8. Prediction bundle.
9. Metric bundle.
10. Artifact manifest.

Testing requirements:

1. Valid config creation.
2. Missing required fields.
3. Invalid task type.
4. Invalid metric direction.
5. Serialization to/from JSON where applicable.

## 11.2 Sub Agent B: Data Preparation

Implement:

1. `data/omics.py`.
2. `data/sample_id.py`.
3. `data/drug_index.py`.
4. `data/response_matrix.py`.
5. `data/cancer_type.py`.
6. `data/split.py`.
7. `data/prepare_pretrain.py`.
8. `data/prepare_finetune.py`.

Testing requirements:

1. Feature intersection.
2. Stable feature order.
3. Sample alignment.
4. Drug union.
5. Stable drug order.
6. Long-to-wide transformation.
7. Mask correctness.
8. Source-only and target-only drug behavior.
9. Missing label behavior.
10. Random split reproducibility.

## 11.3 Sub Agent C: Model Adapter and Heads

Implement:

1. `model/legacy_adapter.py`.
2. `model/heads.py`.
3. `model/wrapper.py`.
4. `model/checkpoint.py`.

Requirements:

1. Reuse original CODE-AE encoder/deconfounding modules where possible.
2. Attach multi-output prediction head.
3. Output logits/predictions with shape `[batch_size, n_drugs]`.
4. Load pretrain checkpoint.
5. Ignore incompatible single-drug prediction heads.
6. Do not make legacy CODE-AE import `codeae_multilabel`.

Testing requirements:

1. Head output shape.
2. Head parameter initialization.
3. Checkpoint partial-load behavior.
4. Missing checkpoint error.
5. Wrapper forward contract.

## 11.4 Sub Agent D: Losses and Selection

Implement:

1. `training/losses.py`.
2. `training/selection.py`.
3. `training/train_state.py`.

Requirements:

1. Masked BCE with logits.
2. Masked MAE.
3. Empty-mask safe behavior.
4. Early stopping for classification.
5. Early stopping for regression.
6. Metric fallback order.

Testing requirements:

1. Exact numeric masked BCE examples.
2. Exact numeric masked MAE examples.
3. Empty mask returns safe loss or documented behavior.
4. Classification higher-is-better selection.
5. Regression lower-is-better selection.
6. Fallback metric selection.

## 11.5 Sub Agent E: Training Runners

Implement:

1. `training/trainer.py`.
2. `training/runners.py`.
3. Integration with original CODE-AE pretraining/fine-tuning logic.

Requirements:

1. `PretrainRunner` handles omics-only pretraining.
2. `FineTuneRunner` loads pretrain checkpoint.
3. Fine-tune uses source supervised masked loss.
4. Target labels are evaluation-only.
5. Original CODE-AE domain/adversarial/deconfounding logic is preserved as much as possible.
6. Best checkpoint is selected by macro validation metric.

Testing requirements:

1. Runner initialization.
2. Fine-tune requires checkpoint.
3. Target labels do not affect training loss.
4. Best checkpoint save/load path.
5. Small-epoch training path with debug limits.

## 11.6 Sub Agent F: Evaluation

Implement:

1. `evaluation/prediction.py`.
2. `evaluation/metrics.py`.
3. `evaluation/reports.py`.

Requirements:

1. Predictions must respect `drug_list.csv` order.
2. Output long prediction tables.
3. Compute per-drug metrics.
4. Compute macro/micro/weighted/overall summary.
5. Compute fold mean/std.
6. Store invalid metrics as `NaN`.

Testing requirements:

1. Prediction long table schema.
2. Drug order correctness.
3. Classification metric numerical tests.
4. Regression metric numerical tests.
5. Invalid single-class AUROC behavior.
6. Overall score behavior.
7. Mean/std across folds.

## 11.7 Sub Agent G: Export and Visualization

Implement:

1. `export/artifacts.py`.
2. `export/latent.py`.
3. `export/visualization.py`.
4. `export/latent_eval.py`.

Requirements:

1. Centralize writing of all artifacts.
2. Export source and target latent after fine-tuning.
3. Run t-SNE only after final best checkpoint is loaded.
4. Gracefully skip t-SNE when samples are insufficient.
5. Preserve output schemas.

Testing requirements:

1. Artifact manifest.
2. CSV schema.
3. Output overwrite behavior.
4. Latent export shape and sample IDs.
5. t-SNE skip behavior.

## 11.8 Sub Agent H: CLI and Smoke

Implement:

1. `pretrain_multilabel_hyper_main.py`.
2. `drug_ft_multilabel_hyper_main.py`.
3. `smoke/smoke_runner.py`.
4. `smoke/smoke_commands.md`.

Requirements:

1. Thin entries only.
2. CLI validates required paths and columns.
3. Debug parameters support small smoke runs.
4. Smoke commands cover classification and regression.

Testing requirements:

1. CLI help works.
2. Missing required args fails clearly.
3. Valid args create config.
4. Smoke commands are documented.

---

## 12. Main Agent Workflow

The Main Agent must execute the following workflow.

### Step 1: Repository Inspection

1. Inspect CODE-AE existing files.
2. Identify reusable encoder/pretraining/fine-tuning modules.
3. Identify dependencies and current style.
4. Record findings in `docs/implementation_report_codeae.md`.

### Step 2: Environment Preparation

1. Check Python version.
2. Check installed packages.
3. Install or configure development dependencies if the environment permits.
4. Add minimal `pyproject.toml` tool configuration if missing.

### Step 3: Task DAG

Create a task graph reflecting module dependencies:

```text
contracts/config
  -> data
  -> model adapter/heads
  -> losses/selection
  -> training runners
  -> evaluation
  -> export
  -> CLI/smoke
  -> integration checks
```

### Step 4: Sub Agent Dispatch

Assign each module to a Sub Agent.

Require each Sub Agent to return:

1. Files changed.
2. Tests added.
3. Module checklist.
4. Known assumptions.
5. Local test commands run.

### Step 5: Integration

Integrate patches in dependency order.

After each integration block, run targeted tests.

### Step 6: Full Verification

Run all required checks:

```bash
python -m pytest
ruff check .
ruff format --check .
mypy benchmark/CODEAE/codeae_multilabel
pytest --cov=benchmark/CODEAE/codeae_multilabel --cov-report=term-missing --cov-fail-under=80
```

### Step 7: Automatic Repair Loop

If any check fails:

1. Parse the failure.
2. Identify responsible module.
3. Dispatch repair to the appropriate Sub Agent.
4. Add or fix tests if the failure reveals missing coverage.
5. Re-run the failing check.
6. Re-run the full check suite when targeted checks pass.

Repeat until all checks pass.

### Step 8: Implementation Report

Write `docs/implementation_report_codeae.md` with:

1. Summary of implemented modules.
2. Final file tree.
3. CLI examples.
4. Artifact schemas.
5. Metrics summary behavior, including `overall`.
6. Test commands and results.
7. Coverage result.
8. Deviations from `proposal_codeae.md` or `design_codeae.md`.
9. Conservative decisions made without user input.
10. Known limitations and future TODOs.

### Step 9: Final Completion Criteria

The Main Agent may declare completion only if:

1. All required modules exist.
2. Entry scripts exist.
3. Unit tests exist for all business logic.
4. Smoke commands exist.
5. `implementation_report_codeae.md` exists.
6. All required checks pass.
7. Coverage is at least 80% for `benchmark/CODEAE/codeae_multilabel`.
8. No core tests are skipped or weakened.
9. Target labels are not used for supervised training.
10. Metrics summary includes `overall`.
11. Latent/t-SNE export happens after final fine-tuning checkpoint.

---

## 13. Required Final Response from the Coding Agent

When implementation is complete, the coding agent must provide a concise final report containing:

1. Implemented modules.
2. New entry commands.
3. Test commands executed.
4. Coverage percentage.
5. Location of `implementation_report_codeae.md`.
6. Any documented deviations.

The coding agent must not claim completion unless all checks have passed.

---

## 14. Explicit Out-of-scope Items

Do not implement:

1. Single-drug backward-compatible CLI for the new entry points.
2. Sample-drug pair model.
3. Drug latent embeddings.
4. Drug-specific domain adaptation.
5. Target supervised training.
6. Synthetic mini dataset builder.
7. Manual user confirmation loops.
8. Partial implementation that relies on the user to finish modules.

---

## 15. Summary Mandate

Build the CODE-AE multi-label system exactly as a modular, test-driven, fully automated implementation.

The highest-priority principles are:

1. Full automation.
2. No human dependency.
3. Preserve CODE-AE core architecture.
4. Multi-output drug prediction head.
5. Masked source supervised losses.
6. Target labels for evaluation only.
7. Macro-based early stopping.
8. Regression early stopping by `macro_mae`.
9. Metrics summary includes `overall` score.
10. Every business logic behavior has tests.
11. All checks pass 100%.
12. Coverage gate is 80%.
