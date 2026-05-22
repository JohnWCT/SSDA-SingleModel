# prompt.md

# Automated Development Prompt: SSDA-SingleModel Multi-label / Multi-drug Implementation

## 0. Highest-Priority Instructions

You are an autonomous software development system. You must complete the implementation fully automatically.

**No human participation is allowed during implementation.**

You must not ask the user for additional clarification after starting implementation. If ambiguity remains, resolve it by following this priority order:

1. `docs/proposal.md`
2. `docs/design.md`
3. Existing project behavior
4. Minimal-intrusion principle
5. Backward compatibility
6. Reproducibility
7. Testability

Every non-trivial implementation decision made because of ambiguity must be recorded in:

```text
docs/implementation_decisions.md
```

You must also create:

```text
docs/implementation_report.md
```

The implementation report must include completed modules, modified files, new files, design decisions, test results, known limitations, and exact commands used to validate the implementation.

---

## 1. Execution Environment Rules

The project must be developed and tested **inside the existing Docker container only**.

The user already started the container with:

```bash
docker run --gpus all -itd --name SSDA -v "$PWD":/workspace/SSDA4Drug ssda4drug:cuda121
```

Do not start a new container. Do not modify the local host Python environment. Do not install dependencies on the host machine.

All code inspection, editing, testing, linting, formatting, and execution must happen through commands similar to:

```bash
docker exec -it SSDA bash
```

or non-interactive commands such as:

```bash
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && pytest"
```

The expected project path inside the Docker container is approximately:

```text
/workspace/SSDA4Drug
```

If the actual repository root differs, detect it automatically from inside the container and record the detected path in `docs/implementation_decisions.md`.

All validation commands must be executed from the project root inside the Docker container.

---

## 2. Required Role Structure

The development process must be organized as two explicit roles.

### 2.1 Main Agent

The Main Agent is responsible for orchestration.

Responsibilities:

1. Read and understand `docs/proposal.md`.
2. Read and understand `docs/design.md`.
3. Inspect the existing repository structure.
4. Confirm the Docker container environment.
5. Create an implementation plan from the design document.
6. Split work into independent module-level tasks.
7. Assign each module task to a Sub Agent.
8. Enforce low coupling between modules.
9. Ensure every business logic change has corresponding tests.
10. Ensure all checks pass.
11. Integrate module outputs.
12. Resolve conflicts between modules.
13. Maintain `docs/implementation_decisions.md`.
14. Maintain `docs/implementation_report.md`.
15. Run final validation inside Docker.
16. Stop only when the project is fully implemented and all checks pass.

The Main Agent must not write large feature modules directly unless a Sub Agent task is too small to justify delegation.

### 2.2 Sub Agent

A Sub Agent is responsible for concrete implementation of a specific module.

Each Sub Agent must:

1. Read the relevant section of `docs/proposal.md`.
2. Read the relevant section of `docs/design.md`.
3. Implement only its assigned module.
4. Avoid unrelated changes.
5. Add unit tests for every public function, class, branch, data transformation rule, mask rule, loss rule, and edge case.
6. Add or update documentation comments where helpful.
7. Run module-specific tests inside Docker.
8. Report changed files and test results to the Main Agent.

A Sub Agent must not silently change interfaces owned by another module. If an interface change is required, the Sub Agent must document the reason and ask the Main Agent to coordinate the change internally. This does not involve asking the human user.

---

## 3. Core Safety and Quality Rules

### 3.1 Unit Test Requirement

For every line of business logic code, the corresponding behavior must be covered by unit tests.

This means:

1. Every public function must have tests.
2. Every public class must have tests.
3. Every important private helper must have tests if it contains business logic.
4. Every branch must be tested.
5. Every mask rule must be tested.
6. Every target n-shot edge case must be tested.
7. Every long-to-wide conversion rule must be tested.
8. Every drug list union/order rule must be tested.
9. Every loss function must be tested.
10. Every metric function must be tested.
11. Every export format must be tested.
12. Every source/target split rule must be tested.
13. Every deterministic latent export rule must be tested.
14. Every validation-no-update rule must be tested.

Test code must be committed alongside implementation code.

### 3.2 Required Check Tools

All code must pass 100% of the following checks:

```bash
ruff check .
ruff format --check .
mypy .
pytest
pytest --cov
```

If the repository does not already contain configuration files, the Main Agent may add minimal project configuration files such as `pyproject.toml`, `pytest.ini`, or `mypy.ini`.

If legacy code is not type-clean, configure `mypy` to focus on the new multi-label modules first, while documenting this decision in `docs/implementation_decisions.md`.

Core new modules must reach at least 85% test coverage. If legacy code prevents total repository coverage from reaching 85%, record the reason in `docs/implementation_report.md` and still ensure the new modules satisfy the threshold.

### 3.3 Docker-only Validation Commands

All checks must be executed inside Docker.

Preferred pattern:

```bash
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && ruff check ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && ruff format --check ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && mypy ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && pytest"
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && pytest --cov"
```

If the project root is different, replace `/workspace/SSDA4Drug` with the detected path.

Do not run these commands on the host machine.

---

## 4. Required Development Style

Every module implementation must include clear explanation of the logic and design choices.

For each implemented module, document:

1. What the module does.
2. What inputs it expects.
3. What outputs it produces.
4. How it avoids coupling with other modules.
5. What assumptions it makes.
6. What edge cases it handles.
7. Which tests verify it.

The final `docs/implementation_report.md` must explain each step in plain language.

---

## 5. Project Goal

Implement a multi-label / multi-drug version of SSDA-SingleModel.

The new system must transform the current single-drug SSDA pipeline into:

```text
omics data table
  -> SSDA encoder
  -> sample-level latent representation
  -> multi-output drug response head
  -> all-drug response vector
```

The model must output one response vector per sample:

```text
prediction shape = [batch_size, n_drugs]
```

The drug dimension is defined by `drug_list.csv`.

The system must support multi-label classification, multi-label regression, masked loss, source sample-level train/validation/test split, target sample-drug position-level n-shot adaptation, target classification labels, source/target prediction exports, source/target sample-level latent pkl export, and latent visual/metric evaluation.

---

## 6. Non-negotiable Design Decisions

1. Omics input must remain raw data tables, as in SSDA-SingleModel.
2. Do not require precomputed sample latent input.
3. Do not use drug latent input.
4. Delete or avoid all drug latent assumptions.
5. Do not implement a sample-drug pair model.
6. Use a multi-output head.
7. Each sample outputs all drug predictions at once.
8. Source and target response inputs are long tables.
9. Long tables must be converted to wide response matrices plus mask tensors.
10. Missing labels must be handled by mask loss.
11. Each run uses one task type: `--task_type classification` or `--task_type regression`.
12. Target labels are always binary classification labels.
13. Regression threshold is fixed: `neg_log2_auc >= 1.0 -> responder = 1`.
14. Drug list is the union of source and target drugs.
15. No drug is removed merely because it appears in only one domain.
16. `drug_list.csv` fixes the output column order.
17. Source split is sample-level.
18. Target n-shot is sample-drug position-level.
19. Each drug and each class may contribute up to `n_shot` target labeled positions.
20. If a target drug/class has fewer than `n_shot`, sample all available positions and record a warning.
21. If a target drug has no labels, skip its target n-shot sampling and record a warning.
22. Latent export remains sample-level, not sample-drug-level.
23. Validation must not update model parameters.
24. Preserve legacy behavior where practical.

---

## 7. Required Modules

The system must be divided into independent modules with low coupling.

Recommended package:

```text
ssda_multilabel/
  __init__.py
  config.py
  io.py
  drug_index.py
  response_matrix.py
  masks.py
  split.py
  dataset.py
  model.py
  losses.py
  adaptation.py
  training.py
  prediction.py
  metrics.py
  latent.py
  latent_eval.py
  export.py
  reports.py
  seed.py
```

If integrating with the existing `ssda_latent/` package is cleaner, the Main Agent may do so, but the module boundaries must remain clear.

---

## 8. Module Specifications

### 8.1 `config.py`

Responsibilities:

1. Define CLI arguments.
2. Parse configuration.
3. Validate required inputs.
4. Store task type, paths, seed, output directory, n-shot and fold settings, loss weights, and regression loss type.
5. Serialize configuration to `config.json`.

Required CLI arguments include:

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

Tests must verify required arguments, invalid task type, default values, path handling, and serialization.

### 8.2 `io.py`

Responsibilities:

1. Read omics tables.
2. Read response long tables.
3. Read cancer type metadata.
4. Validate required columns.
5. Normalize sample IDs if needed.
6. Align source and target omics features.
7. Report removed or missing features.

Tests must verify CSV loading, missing-column errors, feature intersection, numeric feature coercion, and sample ID retention.

### 8.3 `drug_index.py`

Responsibilities:

1. Build drug union from source and target response tables.
2. Sort drug IDs deterministically.
3. Create `drug_list.csv`.
4. Load existing `drug_list.csv` if needed.
5. Map `drug_id -> drug_index` and `drug_index -> drug_id`.

Rule:

```text
drug_list = sorted(unique(source.drug_id ∪ target.drug_id))
```

Tests must verify source-only drugs retained, target-only drugs retained, shared drugs retained once, deterministic sorting, stable mapping, and continuous indices starting at 0.

### 8.4 `response_matrix.py`

Responsibilities:

1. Convert source long response table to wide matrix.
2. Convert target long response table to wide matrix.
3. Generate response masks.
4. Use the shared drug index.
5. Preserve sample order.
6. Handle duplicated sample-drug rows deterministically.

Required outputs:

```text
Y_source: [n_source_samples, n_drugs]
mask_source: [n_source_samples, n_drugs]
Y_target: [n_target_samples, n_drugs]
mask_target_observed: [n_target_samples, n_drugs]
```

Tests must verify long-to-wide conversion, missing labels, observed labels, source-only and target-only drug columns, duplicate handling, and drug order.

### 8.5 `masks.py`

Responsibilities:

1. Generate target labeled mask.
2. Generate target unlabeled mask.
3. Implement position-level n-shot.
4. Record warnings for insufficient labels.
5. Ensure masks are non-overlapping.

Rules:

```text
for each drug:
    sample up to n_shot positions from class 0
    sample up to n_shot positions from class 1
```

Definitions:

```text
mask_target_labeled: observed positions selected for supervised target loss
mask_target_unlabeled = mask_target_observed - mask_target_labeled
```

Tests must verify per-drug sampling, per-class sampling, position-level behavior, repeated samples across different drugs, no-label drugs, low-count classes, non-overlap, mask identity, and reproducibility.

### 8.6 `split.py`

Responsibilities:

1. Split source samples into source_test and source_train_val.
2. Create sample-level K-fold splits.
3. Ensure no sample leakage.
4. Save split tables.

Rules:

1. Source split is sample-level.
2. Source test is excluded from all fold train/validation sets.
3. Use deterministic random seed.
4. Use stratification when feasible; if not feasible, fall back safely and record decision.

Tests must verify no leakage, fold coverage, reproducibility, correct split labels, and small dataset behavior.

### 8.7 `dataset.py`

Responsibilities:

1. Provide PyTorch datasets for source and target matrices.
2. Return omics features, response vectors, mask vectors, and sample indices or IDs if needed.
3. Keep DataLoader independent of export logic.

Tests must verify tensor shapes, sample order, masks, dtype, classification, and regression.

### 8.8 `model.py`

Responsibilities:

1. Wrap existing SSDA encoder.
2. Add multi-output prediction head.
3. Output `[batch_size, n_drugs]`.
4. Provide deterministic latent extraction.
5. Avoid stochastic DAE denoising during export.

Tests must verify output shape, latent shape, deterministic latent export, DAE-like encoder, MLP-like encoder, and no stochastic latent during evaluation/export.

### 8.9 `losses.py`

Responsibilities:

1. Implement masked BCEWithLogitsLoss.
2. Implement masked MSE.
3. Implement masked MAE.
4. Implement masked Huber loss.
5. Safely handle empty masks.

Required functions:

```python
masked_bce_with_logits(logits, targets, mask)
masked_mse(pred, targets, mask)
masked_mae(pred, targets, mask)
masked_huber(pred, targets, mask)
```

Tests must verify missing labels ignored, correct averaging, empty mask behavior, gradients, classification, and regression.

### 8.10 `adaptation.py`

Responsibilities:

1. Implement masked entropy loss.
2. Implement masked adentropy/MME-compatible loss if required.
3. Apply only to `mask_target_unlabeled`.
4. Support multi-output logits/scores.

Tests must verify masked positions ignored, only target unlabeled positions used, empty masks handled, shape correctness, and documented gradient behavior.

### 8.11 `training.py`

Responsibilities:

1. Train multi-label SSDA model.
2. Support classification run.
3. Support regression run.
4. Use source supervised loss.
5. Use target labeled supervised loss.
6. Use target unlabeled adaptation loss.
7. Ensure validation never updates model.
8. Save training logs.
9. Save model checkpoints.

Classification total loss:

```text
loss_total = source_classification_loss + target_labeled_classification_loss + lambda_adapt * target_unlabeled_adaptation_loss + optional_reconstruction_loss
```

Regression total loss:

```text
loss_total = source_regression_loss + target_labeled_classification_loss + lambda_adapt * target_unlabeled_adaptation_loss + optional_reconstruction_loss
```

Important regression rule:

The model still outputs one score matrix. Source regression interprets scores as continuous predictions. Target classification interprets scores as binary logits.

Tests must verify correct loss calls, inclusion of target labeled loss, inclusion of adaptation loss, no validation backward/optimizer step, training parameter updates, and validation parameter stability.

### 8.12 `prediction.py`

Responsibilities:

1. Predict source outputs.
2. Predict target outputs.
3. Convert wide prediction matrix to long result table.
4. Export only observed sample-drug positions by default.
5. Include drug ID, drug index, split/role, probability/confidence, and task type.

Tests must verify long table shape, observed-only export, drug IDs, split labels, target roles, sigmoid probability for classification, continuous regression score, and target classification prediction during regression run.

### 8.13 `metrics.py`

Responsibilities:

1. Classification per-drug metrics.
2. Classification micro/macro/weighted summary metrics.
3. Regression per-drug metrics.
4. Regression summary metrics.
5. Target classification metrics.
6. Handle drugs with insufficient positive/negative samples.

Classification metrics: AUC, AUPR, accuracy, F1, balanced accuracy.

Regression metrics: RMSE, MAE, R2, Pearson, Spearman.

Tests must verify per-drug metrics, macro/micro/weighted metrics, insufficient class handling, regression correctness, and empty mask safety.

### 8.14 `latent.py`

Responsibilities:

1. Export sample-level source latent.
2. Export sample-level target latent.
3. Save pkl dictionaries.
4. Ensure deterministic encoder output.
5. Preserve sample IDs.

Output format:

```python
{"sample_id": [latent_0, latent_1, ..., latent_n]}
```

Tests must verify all source/target samples exported, sample IDs preserved, latent dimension, and deterministic output.

### 8.15 `latent_eval.py`

Responsibilities:

1. t-SNE domain mixing plot.
2. t-SNE cancer type plot.
3. FID.
4. MMD.
5. Wasserstein.
6. KMeans cancer type metrics.

Tests must verify toy-data metric execution, small sample handling, safe t-SNE perplexity, Unknown cancer type handling, and KMeans insufficient-class behavior.

### 8.16 `export.py`

Responsibilities:

1. Create output directories.
2. Save matrices, masks, drug list, predictions, metrics, latent pkl, plots, config, fold outputs, and summary outputs.

Tests must verify files, CSV schemas, PKL schema, fold paths, and summary paths.

### 8.17 `reports.py`

Responsibilities:

1. Generate data alignment report.
2. Generate target n-shot summary.
3. Generate missing data report.
4. Generate implementation decisions report.
5. Generate implementation report.

Tests must verify expected report sections, warning counts, and missing source/target/drug cases.

### 8.18 `seed.py`

Responsibilities:

1. Seed Python.
2. Seed NumPy.
3. Seed PyTorch.
4. Seed CUDA when available.
5. Configure deterministic behavior where practical.

Tests must verify NumPy repeatability, PyTorch CPU repeatability, and no crash without CUDA.

---

## 9. Main Entry Point

Implement a clear new entry point:

```text
experiment_multilabel_ssda.py
```

Prefer this over rewriting existing single-drug entry points.

The entry point must:

1. Parse config.
2. Set seed.
3. Load data.
4. Build drug index.
5. Build response matrices.
6. Build masks.
7. Build source splits.
8. Build model.
9. Train fold models.
10. Export predictions.
11. Export latent.
12. Export metrics.
13. Export reports.

---

## 10. Required Output Structure

The implementation must produce:

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
      fold_2/
      fold_3/
      fold_4/

      metrics_summary.csv
      latent_metrics_summary.csv
      kmeans_cancer_type_summary.csv
```

---

## 11. Synthetic Toy Dataset Requirement

The implementation must include a synthetic toy dataset test.

The toy dataset must include source omics, target omics, source response long table, target response long table, source-only drug, target-only drug, shared drug, missing labels, insufficient n-shot class samples, and cancer type metadata.

The toy pipeline test must verify end-to-end data preparation, drug union creation, matrix/mask creation, target n-shot masks, one training step, prediction export, latent export, and metrics export.

This test can be lightweight and CPU-only.

---

## 12. Development Workflow

The Main Agent must follow this workflow:

1. Inspect repository.
2. Locate project root inside Docker.
3. Read `docs/proposal.md`.
4. Read `docs/design.md`.
5. Create `docs/implementation_decisions.md`.
6. Create task list.
7. Delegate modules to Sub Agents.
8. Implement each module with tests.
9. Run module-level tests after each module.
10. Integrate modules.
11. Run full checks.
12. Fix failures.
13. Repeat until all checks pass.
14. Generate `docs/implementation_report.md`.

---

## 13. Required Validation Commands

Run inside Docker only:

```bash
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && ruff check ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && ruff format --check ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && mypy ."
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && pytest"
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && pytest --cov"
```

If the project root is different, replace `/workspace/SSDA4Drug` with the detected path.

All final commands and outputs must be recorded in `docs/implementation_report.md`.

---

## 14. Failure Handling

If a check fails:

1. Read the error.
2. Fix the root cause.
3. Add a regression test if the failure represents missing coverage.
4. Re-run the failed check.
5. Re-run the full check suite.
6. Record the fix in the implementation report.

Do not bypass checks. Do not delete tests to make checks pass. Do not weaken tests unless the test itself is demonstrably incorrect, and document the reason.

---

## 15. Backward Compatibility Rules

1. Do not break existing single-drug SSDA functionality.
2. Do not remove existing scripts unless clearly obsolete and documented.
3. Prefer adding `experiment_multilabel_ssda.py` over heavily rewriting existing entry points.
4. If legacy code must be modified, keep the change minimal and covered by tests.
5. New code should depend on legacy model components through small adapters, not direct deep coupling.

---

## 16. Documentation Requirements

The implementation must update or create:

```text
docs/implementation_decisions.md
docs/implementation_report.md
```

If appropriate, also update `README.md`.

README update should include how to run the multi-label pipeline inside Docker, required input files, expected output files, example command, and testing commands.

All commands must use Docker:

```bash
docker exec SSDA bash -lc "cd /workspace/SSDA4Drug && python experiment_multilabel_ssda.py ..."
```

---

## 17. Final Completion Criteria

The task is complete only when:

1. Multi-label pipeline is implemented.
2. Drug list union is implemented.
3. Long-to-wide response conversion is implemented.
4. Mask loss is implemented.
5. Source sample-level split is implemented.
6. Target position-level n-shot is implemented.
7. Classification run is supported.
8. Regression run is supported.
9. Target classification loss is included.
10. Target adaptation loss is included.
11. Prediction export works.
12. Latent export works.
13. Metrics export works.
14. Synthetic toy dataset test passes.
15. `ruff check .` passes.
16. `ruff format --check .` passes.
17. `mypy .` passes or is configured reasonably for new modules.
18. `pytest` passes.
19. `pytest --cov` passes.
20. New core modules have at least 85% coverage.
21. All validation is run inside Docker.
22. `docs/implementation_decisions.md` exists.
23. `docs/implementation_report.md` exists.
24. No host environment modification was performed.
25. No human clarification was required during implementation.

---

## 18. Final Reminder

You must implement the code, tests, reports, and validation fully automatically.

Do not ask the user for help.

Do not pause for confirmation.

Use Docker only.

Explain every module’s logic and design in the implementation report.

Every business logic behavior must be tested.

All checks must pass before completion.
