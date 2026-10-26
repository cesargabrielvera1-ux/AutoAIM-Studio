# Changelog

All notable changes to AutoAIM Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.0.0] - 2026-10-26 — Zero Data Leakage via sklearn Pipeline

### Gaussian Process — Full Kernel Family Support

- **Kernel family selection**: GP now supports tuning across 5 kernel families:
  - `RBF` — infinitely differentiable (default)
  - `Matern` — controlled smoothness via `nu` parameter (0.5, 1.5, 2.5)
  - `RationalQuadratic` — mixture of RBFs with different length-scales
  - `WhiteKernel` — explicit observation noise term
  - `RBF+WhiteKernel` — signal + noise decomposition
- **Kernel hyperparameters tuneable**:
  - `kernel_length_scale` — characteristic length scale (float_log, 1e-3 → 100)
  - `kernel_nu` — Matern smoothness (categorical: 0.5, 1.5, 2.5)
  - `kernel_alpha_rq` — Rational Quadratic tail heaviness (float_log, 1e-3 → 100)
  - `kernel_noise_level` — WhiteKernel noise magnitude (float_log, 1e-5 → 1.0)
- **Bayesian optimization**: Full kernel tuning via Optuna TPE (recommended).
- **Grid Search**: Kernel params are automatically filtered out with a warning; scalar GP params (`alpha`, `n_restarts_optimizer`, `normalize_y`) are still grid-searchable.
- **Model provenance**: `gp_metadata` now captures `kernel_family`, `kernel_length_scale`, `kernel_nu`, `kernel_alpha`, `kernel_noise_level` for full scientific reproducibility.
- **Results Tab**: Extended GP details popup shows all kernel parameters.
- **Training Tab**: Added `kernel_type`, `kernel_nu` dropdowns with educational tooltips. Added generic `categorical` parameter handler.
- **Session Export**: GP kernel metadata and prediction uncertainty are now rendered in session reports.

### Session Report Cleanup

- `_extract_model_params()` now filters out `preprocessor__*` noise from sklearn Pipelines.
- Pipeline param keys are stripped of `model__` prefix for cleaner output.
- Internal sklearn objects (`steps`, `memory`, `verbose`) are excluded.
- Report now shows GP-specific sections: "Gaussian Process Metadata" and "Prediction Uncertainty".

### Out-of-Distribution (OOD) Detection + Disclaimer Banner (Peer-Review Compliant Transparency)

- **Disclaimer banner**: Yellow warning banner displayed in the Predict tab whenever a model is loaded. Text explicitly states that predictions are *model estimates for screening and prioritisation, NOT experimentally validated properties*.
- **OOD detector**: New `app/core/ood_detector.py` module implementing k-NN distance in standardised feature space.
  - `OodDetector.fit(X_train)`: Standardises training features, fits 1-NN, computes threshold as 95th percentile of training-set nearest-neighbour distances.
  - `OodDetector.predict(X_new)`: Returns `(distance, flag)` for each sample; flag is `True` when distance > threshold.
  - Automatic save/load via `joblib` in model bundle (`ood_detector.joblib`).
  - Metadata (method, threshold, percentile, n_train_samples) stored in `manifest.json`.
- **Integration**:
  - `trainer.py`: Fits OOD detector on `X_train_np` after training; attaches to `TrainingResult`.
  - `model_saver.py`: Saves `ood_detector.joblib` + metadata in manifest.
  - `inference_engine.py`: Loads OOD detector if present; `predict_with_ood()` returns base predictions + `ood_distance`, `ood_flag`, `ood_warning` columns.
  - `inference_tab.py`: Shows disclaimer banner on model load; displays OOD columns in results table; tints OOD rows red; includes OOD count in statistics label.
  - `MODEL_CARD.md`: Updated Limitations section with disclaimer text + OOD documentation.
- **Legacy compatibility**: Pre-v2.0 bundles load without errors; OOD columns are filled with neutral values (`NaN`, `False`, `""`).
- **Tests**: `tests/core/test_ood_detector.py` — 11 tests (construction, fitting, in-domain/out-of-domain prediction, feature mismatch, persistence, metadata, warning formatting).

### Disk-Based Model Cache (Memory-Safe Training for Low-RAM Machines)

- **New module**: `app/core/model_cache.py` — singleton disk cache for fitted models.
  - `ModelCache.save(model, name, is_neural_network, nn_architecture)`: Serialises sklearn models via `joblib` (compress=3) and PyTorch models via `torch.save()` (state_dict + architecture). Returns the cache path.
  - `ModelCache.load(path, is_neural_network)`: Loads the model back from disk on demand.
  - `ModelCache.clear_all()`: Removes all cached files (user-accessible for manual cleanup).
  - `ModelCache.get_stats()`: Reports cache size, entry count, and directory path.
  - Auto-cleanup on startup: files older than 7 days are removed.
- **TrainingResult**: New fields `model_cache_path` (str) and `is_cached` (bool). The heavy `model` object is set to `None` after caching so Python can reclaim RAM.
- **trainer.py**: After each training (sklearn and NN), the fitted model is cached to `~/.autoaim/model_cache/` and `result.model = None`. All lightweight metadata (metrics, predictions, configs) stays in RAM.
- **model_saver.py**: `save_training_result()` now detects cached models (`result.is_cached`) and loads them from disk before serialising to the final bundle. If the cache file is missing, a descriptive error is raised.
- **results_tab.py**: When a user clicks "Save" or "Export", if the model was offloaded to cache, it is transparently reloaded from disk before the save operation proceeds.
- **RAM impact**: On a typical session with 20 regressors, peak RAM drops from ~4-6 GB (all models in memory) to ~800 MB-1.2 GB (only metadata in memory), making AutoAIM usable on 8 GB RAM laptops.

### Optimization Tab

- Added **Gaussian Process** to the algorithm dropdown.
- GP parameters are now fully consistent with other regressors:
  - `alpha`, `n_restarts_optimizer`, `normalize_y`
  - `kernel_type`, `kernel_length_scale`, `kernel_nu`, `kernel_alpha_rq`, `kernel_noise_level`
- Random State is correctly applied to GP (`random_state` parameter).
- Models that do NOT accept random state (SVR, k-NN) continue to disable the RS spinner.

### Fixed (CRITICAL: Data Leakage in Cross-Validation)

#### Zero Data Leakage via sklearn Pipeline
- **CRITICAL FIX**: Eliminated data leakage in cross-validation. Previously,
  `StandardScaler` was fitted on the **entire training set** before CV, meaning
  every validation fold's statistics leaked into the scaler. This produced
  optimistically biased CV scores.
- **Migrated entire stack to sklearn Pipeline**: `Pipeline([('preprocessor',
  preprocessor), ('model', model)])`. The preprocessor is now refitted **inside
  each CV fold**, so the validation fold never influences the training statistics.
- **Core changes**:
  - `data_manager.prepare_data()` now returns **RAW DataFrames** (not preprocessed
    arrays). The preprocessor is created but **not fitted**.
  - `data_manager.get_preprocessor()` uses **integer column indices** instead of
    column names, making it compatible with both DataFrames and numpy arrays.
  - `trainer.train()` constructs a Pipeline when a preprocessor is passed, and
    `cross_validate(pipeline, X_raw, y, ...)` handles per-fold refitting.
  - `trainer.train_neural_network()` clones and refits the preprocessor inside
    each NN CV fold (manual Pipeline behavior for PyTorch).
  - `optimizer.optimize_sklearn_model()` (Bayesian) wraps each trial estimator
    in a Pipeline.
  - `optimizer.optimize_sklearn_model()` (Grid Search) wraps the estimator in
    a Pipeline and prefixes param_grid keys with `model__`.
  - `model_saver._save_sklearn_model()` only wraps the model in a Pipeline if
    the preprocessor is actually fitted (prevents saving unfitted preprocessors).
  - `inference_engine` accepts raw DataFrames; the saved Pipeline applies
    preprocessing internally.
  - `explainer` drills into Pipeline steps for SHAP explainability.
- **All GUI tabs updated**: Training, Optimization (Bayesian + Grid Search),
  Ensemble, and Neural Network tabs all pass raw DataFrames + preprocessor.
- **Backward compatible**: Code that doesn't pass a preprocessor still works
  with numpy arrays (legacy path).
- **New test suite**: `tests/core/test_pipeline_no_leakage.py` (15 tests)
  verifies zero leakage, Pipeline construction, save/load roundtrip, and
  GP `return_std` inside Pipeline.

### Added

#### Stratified and Grouped Cross-Validation

### Added

#### Stratified and Grouped Cross-Validation

Available consistently across **sklearn regressors, Neural Networks, and
Ensembles** through one backend (`cv_factory`) and one UI widget
(`CVConfigWidget`). Each NN fold re-trains a fresh network; ensemble CV
evaluates the combined prediction per fold, and ensemble weight optimization
shares the strategy/groups.

- **Stratified K-Fold (quantiles)** for regression: the target is binned into
  quantiles (configurable, default 5) and every fold preserves the target
  distribution — ideal for skewed targets (Tc, band gaps, formation energies).
  Automatic bin reduction with an info note when the target has many repeated
  values. Regression-only (classification already stratifies by class under K-Fold).
- **Group K-Fold**: samples from the same group never cross the train/test
  boundary, preventing chemical/batch data leakage. Four grouping sources, no
  new third-party dependencies (matminer NOT required):
  - *Chemical system* — same element set (all Y-Ba-Cu-O together); the standard
    anti-leakage split for materials data.
  - *Exact formula* — reduced stoichiometry (Y₂Ba₄Cu₆O₁₄ groups with YBa₂Cu₃O₇).
  - *Clustering (advanced)* — deterministic agglomerative clustering over element
    fractions (no random seed → reproducible).
  - *Dataset column* — any column: synthesis batch, laboratory, source, year.
- **Leave-One-Group-Out (LOGO)**: each group left out once; folds = number of
  groups. Uses out-of-fold predictions with a global R² + per-group MAE (R² is
  undefined for a single left-out sample/group).
- **Reserved bin for a degenerate value (Stratified)**: isolate a point mass
  meaning "property absent" (Tc = 0 K, Eg = 0 eV, E_hull = 0, activity = 0) into
  its own stratum so every fold carries the same proportion; the rest of the
  target is binned into the remaining `bins − 1` quantiles. Auto-detected with a
  cyan suggestion (trigger: ≥ ~5% of samples AND ≥ 2 per fold), or set manually
  with data-grounded validation — blocked if too few samples to stratify
  (< folds), warned with real counts if the value is the majority (> 50%) or a
  tiny fraction (< 5%). Configurable tolerance (default exact match).

#### Core architecture (no duplicated code)
- New `app/core/grouping.py`: four group-label generators + composition-column
  detection, all reusing `CompositionParser`.
- New `app/core/cv_factory.py`: single source of truth for splitter construction,
  scoring selection, and the out-of-fold decision. One decision rule — any
  regression strategy whose test folds can hold a single sample uses the OOF path
  (global R² + per-unit MAE). Also fixes a real defect: classification K-Fold now
  uses `StratifiedKFold` by class.
- New `app/core/cv_display.py`: single formatter (HTML + text) for the CV
  breakdown, previously duplicated across Training, Optimization (×2) and Results.
- New `assemble_cv_metrics` in `cv_factory.py`: shared OOF/per-fold metric
  assembly so Neural Networks and Ensembles produce the exact same cv_metrics
  structure as sklearn (each NN fold re-trains a fresh network; ensemble CV
  evaluates the combined prediction per fold).
- New `app/gui/cv_config_widget.py`: single shared CV configuration widget used in
  the Training tab, Optimization tab, Neural Network tab, Ensemble tab, and Manual
  Parameter dialog — eliminating the previously triplicated LOO UI logic.
- `MaterialsAwareValidator` rewritten as a thin wrapper over the factory, fixing
  three latent defects: stratified now actually stratifies (it previously computed
  bins and discarded them); `loco` no longer silently clamps folds to n_groups;
  group labels reuse CompositionParser instead of a local regex.

#### Validation, warnings, and fallbacks (user-informed)
- Blocks (hard stop with a clear message): more folds than groups; LOGO with < 3
  or > 100 groups; a group column with missing values; LOO with N > 100.
- Warnings (ask for confirmation): LOO/LOGO cost (30 < N ≤ 100 samples/groups),
  highly imbalanced groups (> 10×), LOGO in optimization (n_groups × n_trials).
- Info (inline orange note): singleton groups, quantile-bin reduction, OOF
  fallback for single-sample group folds.

#### Reporting & provenance
- Per-fold reports add a **Test Size** column for uneven (grouped) folds.
- Details popups and the session export show the full strategy label (e.g.
  "Group K-Fold (by chemical system)") and surface CV diagnostics as warnings.
- `model_saver` writes the CV configuration into `manifest.json`; the Predict tab
  shows a read-only "Validated with: …" line (prediction never re-runs CV).

#### Gaussian Process — Native Uncertainty Quantification (v2.0)
- **New model**: `GaussianProcessRegressor` (regression) and
  `GaussianProcessClassifier` (classification) added to `model_registry.py`.
  Default: RBF kernel, alpha=1e-10, n_restarts_optimizer=10, normalize_y=True.
- **Native uncertainty**: `predict(return_std=True)` gives the posterior standard
  deviation (±1σ). The InferenceEngine adds a `prediction_std` column automatically
  when the loaded model is a GP. The Predict tab shows this column with an amber
  highlight and an informational tooltip.
- **Console warning** when GP is selected with N > 2,000 samples (O(n³) scaling).
- **Optimization**: GridSearch + Bayesian support for `alpha`,
  `n_restarts_optimizer`, and `normalize_y`.
- **Manifest / Model Card**: `gp_metadata` stores the learned kernel string,
  alpha, and log marginal likelihood value. The Model Card includes an
  "Uncertainty Quantification" section.
- **SHAP**: Falls back to `KernelExplainer` (correct, slower than TreeExplainer).
- **Tests**: 9 tests in `test_gpr.py` covering creation, reproducibility,
  return_std, trainer capture, inference engine, manifest serialization,
  GridSearch optimization, and learning curves.

#### Random State honoring (reproducibility)
- **CatBoost silent bug fixed**: the optimizers (Bayesian + Grid Search) passed
  `random_state` to CatBoost, which only reads `random_seed` — the user's seed was
  silently ignored. A shared `_apply_random_state` helper now maps the seed to the
  correct parameter per library (CatBoost → `random_seed`; SVR/k-NN → no seed).
- **CV fold shuffling now follows the user's seed**: the tab's Random State/Seed is
  synced into `cv_config.random_state` across Training, Optimization, NN, and
  Ensemble tabs, so the seed controls the data split, model init, sampler, AND the
  CV folds (previously the folds always used the default 42).
- **Optimization Random Seed is now model-aware**: it is disabled for the
  deterministic models (SVR, k-NN) that accept no seed, so users are never offered
  a control with no effect.
- Verified end-to-end: same seed → identical predictions (sklearn regressors,
  optimizers, and Neural Networks — NNs seed torch/numpy/random before weight init).

#### Executable robustness
- Fixed a `NameError: name 'Optional' is not defined` crash in the PyInstaller
  build (a type annotation used an un-imported typing name). Added a permanent
  AST-based regression test (`test_annotation_imports.py`, 43 tests) that fails if
  any annotation references an un-imported/undefined name — this class of bug is
  invisible to `py_compile` (syntax-only) and is now caught before any build.
- Ensemble tab control panel is now vertically scrollable (same pattern as the NN
  tab), so all options stay reachable on laptop/desktop screens.

### Tests
- 114 new CV tests: `test_grouping.py` (25), `test_cv_factory.py` (31),
  `test_cv_display.py` (16), `test_cv_strategies_e2e.py` (12, incl. no-leakage
  assertion and grouped GridSearch/Bayesian), `test_special_value_bin.py` (22,
  incl. reserved-bin binner, point-mass detection, validation, serialization, e2e),
  `test_cv_nn_ensemble.py` (10, NN + Ensemble training/weight-opt with all
  strategies, incl. group no-leakage and the cv_folds=1 skip guard).
- 59 robustness tests: `test_annotation_imports.py` (43, undefined-annotation-name
  guard), `test_random_state.py` (16, RS honoring: naming, reproducibility,
  fold-shuffle sync, NN weight init).
- 9 GP tests: `test_gpr.py` (creation, reproducibility, return_std, trainer
  capture, inference engine, manifest, GridSearch, learning curves).
- **Total: 1024 tests — 0 failures, 74 skipped** (optional dependencies).

---

## [2.0.0] - 2026-08-24 — Post-Release Final (CV Report Fixes, LOO-CV R²/MAE, GUI Consistency)

### Added

#### Link Crystal Structures
- **Bridge tabular data with CIF structures** via an ID column (`app/core/crystal_feature_extractor.py`)
- Extracts 149 crystal descriptors per CIF (unified with direct CIF loading): lattice (~10), composition (~13), structural (~3), Magpie (106), advanced (~15), derived (~2)
- Auto-detects ID column candidates from CSV (columns with >=50% unique values)
- Pre-link availability check: reports matched/missing/duplicate IDs before processing
- Polymorph detection: warns when multiple rows share the same ID
- **Non-crystalline structure detection**: detects isolated molecules in artificial simulation boxes and warns the user that descriptors will be physically meaningless
- Automatic replacement of composition features (comp_*) to avoid duplication with CIF-derived composition descriptors
- Magpie featurizer disabled when crystal structures are linked (with tooltip explanation)
- ID column automatically excluded from training features (marked as non-feature)
- Full export support: crystal link metadata in all export formats
- 16 tests in `tests/core/test_crystal_feature_extractor.py`

#### Link Molecular Structures
- **Bridge tabular data with .xyz molecular structures** via an ID column (`app/core/molecule_feature_linker.py`)
- 100% Python pure (numpy only) — zero external dependencies, no RDKit required
- **Three featurization levels**: Composition (~54 features including 34 periodic properties), Geometry 3D (~20 features), Topology (~45 features) + 19 oxidation state features — 149 molecular descriptors total
- **Data Tab GUI**: Two separate sections — "Crystal Structures" (CIF) and "Molecular Structures" (.xyz) — each with Load Files, Load Directory, and Link to CSV buttons. Consistent with Predict Tab behavior.
- Composition: atom counts per element (H, C, N, O, S, P, F, Cl, Br, I, Si, B), molecular mass, heavy atom count, elemental diversity, H/C ratio, and more
- Geometry 3D: center of mass, radius of gyration, bounding box (volume, surface, diagonal), principal moments of inertia (with oblate/prolate shape index), asphericity, spherical defect, pairwise distance stats
- Topology: estimated bond graph, coordination numbers, bond angle statistics, ring detection (3-8 membered), functional group detection (OH, NH, C=O, C=C, C≡C, C≡N, C-F, C-Cl, C-Br, PO, CS, SO), bond/angle histograms (10 bins each), bond type ratios, saturation/unsaturation indices
- Auto-detects ID column candidates from CSV
- Pre-link availability check: reports matched/missing/duplicate IDs before processing
- Missing structures filled with NaN (trainable after imputation)
- **Identical API to CrystalFeatureExtractor** — interchangeable usage pattern
- 24 tests in `tests/integration/test_molecule_linker.py`
- 44 tests in `tests/core/test_molecule_featurizer.py`

#### Export Featurized Dataset (Data Tab & Predict Tab)
- **Export Dataset** button in Data Tab (green) — exports the current featurized dataset to CSV regardless of origin (pure CSV, crystal structures, molecular structures, linked features, or any combination)
- **Export Featurized Dataset** button in Predict Tab — exports the input data with all generated features before prediction is applied
- Feature breakdown in export dialog: Crystal, Molecular, Functional groups, Composition (Magpie)
- Intelligent index handling: meaningful named indices are preserved, default numeric indices are omitted
- Smart default filenames derived from original file name
- Shared testable core: `app/core/dataset_exporter.py` with `DatasetExporter` class
- 35 tests in `tests/core/test_dataset_exporter.py`

### Fixed

#### Critical Import Bugs (Data Tab)
- **`name 'os' is not defined`** when loading molecular structures — `import os` was missing from `app/gui/data_tab.py` even though `os.path.join`, `os.path.splitext`, and `os.path.basename` were used in `_load_molecular_structures`
- **`name 'QInputDialog' is not defined`** when linking molecular structures — `QInputDialog` was not imported in `from PyQt6.QtWidgets` block, causing failure when the ID column selection dialog was triggered
- **Inline `QInputDialog` imports** removed (2 occurrences) — now imported once at module top level for consistency and clarity

#### DataManager Property Assignment Bug
- **`property 'data' of 'DataManager' object has no setter`** when linking molecular structures — `_on_link_molecular_structures` attempted `self.data_manager.data = linked_df` but `data` is a read-only property. Fixed to use `self.data_manager._data = linked_df` (matching the crystal link and structure load patterns)

#### Molecular Unlink Failed Silently
- **`_on_unlink_molecular_structures` attempted to reload from `_file_path`** which was never stored in DataManager. Now:
  - `DataManager.load_data()` stores the original path in `self._file_path`
  - `unset_crystal_link`-style fallback: removes `mol_*` and `has_*` columns directly from the DataFrame if path is unavailable

#### .xyz Parser BOM Crash
- **`ValueError: invalid literal for int() with base 10: '\ufeff3'`** when loading .xyz files saved with Windows editors (UTF-8 Byte Order Mark)
- Fixed `_parse_xyz` to use `encoding='utf-8-sig'` which automatically strips BOM, with `UnicodeDecodeError` fallback to plain UTF-8
- First line now defensively strips any remaining BOM characters before parsing atom count
- Clear error message when first line is not a valid integer: `"Invalid .xyz file: first line must be number of atoms, got '...'"`

#### .xyz Parser: Raw Coordinate Format (GaussView/Spartan)
- **Files without standard XYZ header** (no atom count line, no comment line) are now auto-detected and parsed correctly
- Auto-detection logic: if first line is not a positive integer → switch to raw coordinate parser
- Raw parser scans every line for valid element + 3 float coordinates; empty lines are skipped
- Also handles **Gaussian 5-column format** (`Index Element X Y Z`) via `_detect_element_token`
- `_clean_element()` validates symbols: 1-2 letters, first uppercase, second lowercase
- Error message for truly invalid files now includes line count: `"No valid atoms parsed from {path}. Expected format: 'Element X Y Z'. File has N lines."`
- 7 new tests in `test_molecule_featurizer_bom.py` covering raw format, Gaussian format, mixed whitespace, empty lines, realistic GaussView export

#### DataManager `get_column_info()` Missing Method
- **`'DataManager' object has no attribute 'get_column_info'`** when linking/unlinking molecular structures — `_on_link_molecular_structures` and `_on_unlink_crystal_structures` called `self.data_manager.get_column_info()` but the method did not exist
- Added `get_column_info()` to `DataManager` — re-runs `_analyze_dataset()` and returns refreshed `DatasetInfo`; safe to call after any data modification

#### Exhaustive Code Audit — Hidden Bugs (v2.0)
A systematic review of all GUI and core files uncovered multiple latent bugs:

- **`self.parent = parent` PyQt overwrite** in 5 GUI tabs (`ensemble_tab.py`, `explainability_tab.py`, `nn_builder_tab.py`, `optimization_tab.py`, `training_tab.py`) — same root cause as `inference_tab.py` fix from earlier session. `self.parent` overwrites the PyQt `parent()` method, causing subtle and intermittent failures. All changed to `self._main_window = parent` with mechanical `self.parent` → `self._main_window` replacements (41 references across 5 files)
- **`import os` missing in `molecule_featurizer.py`** — `os.unlink(tmp)` used in `list_molecule_feature_names()` but `os` was never imported; would crash only when calling that specific method (rare path)
- **Direct `_analyze_dataset()` calls from GUI** — `data_tab.py` called the private `self.data_manager._analyze_dataset()` in 4 places instead of the new public `get_column_info()` method. Converted all 4 calls for consistency and future maintainability
- **`'DataTab' object has no attribute 'feature_checkboxes'`** in `_on_link_molecular_structures` — 3 lines of dead copy-paste code attempted to access a non-existent `feature_checkboxes` dict. The ID column is already handled by the target combo update above. Removed

#### Error Reporting in Molecular Featurization
- **`No .xyz files could be featurized`** was a completely opaque error — no indication of which files failed or why
- Now collects per-file errors with `type(e).__name__: e` format and displays up to 10 in the error dialog, plus common causes (BOM, missing atom count, invalid coordinates)
- Partial failures (some files OK, some bad) are logged as warnings but do not block processing

#### Inference with Linked Structures
- **No more "crash asegurado"** when predicting with models trained on linked features
- `ModelSaver` auto-detects linked features (`crystal_*`, `mol_*`, `comp_*`) and stores `linked_feature_prefixes` + `requires_structure_linking` in manifest
- `InferenceEngine.is_structure_linked_model()` — detect if loaded model requires linked structures
- `InferenceEngine.get_linked_structure_requirements()` — get structure type (crystal/molecule), missing feature count, and linked prefixes
- `InferenceEngine.validate_input()` — when ONLY linked features are missing, returns `needs_linking_action=True` with actionable message instead of cryptic error
- `InferenceEngine.predict()` — gives informative error with example code when linked features are missing
- `InferenceEngine.predict_with_linked_structures(df, id_column, structure_folder)` — **automatically links structure files before prediction**, auto-detects structure type from manifest
- 22 tests in `tests/core/test_inference_linked_structures.py`

#### Data Health Check
- Comprehensive 10-category data quality analysis before training (`app/core/data_health_check.py`)
- Detects: missing values (critical >50%, high 20-50%, low 5-20%), duplicate rows, constant columns, outliers (IQR), scaling mismatches, high correlations (>90%), class imbalance
- Overall health score (0-100) with severity levels (info, low, medium, high, critical)
- Actionable recommendations for each detected issue
- HTML report dialog with severity-colored display
- Access via "Data Health Check" button in Data tab

#### Session Exporter
- Export complete session results in 4 formats: .txt (human-readable), .md (Markdown), .csv (spreadsheet), .json (machine-readable)
- Includes: all trained models, optimized parameters with full optimization config (trials, CV folds, random state, selected params, pruning), NN training history, ensemble results, per-fold CV scores
- Learning curves data (tabular: train_sizes, train_scores, test_scores)
- Confidence interval parameters
- Final ranking table sorted by primary metric (R2/Accuracy)
- Reproducibility checklist
- Auto-filters orphaned optimization entries (v1.5.0 fix preserved)
- Crystal link metadata when structures are linked

#### Confidence Intervals for Predictions
- Regression: CV residual-based intervals using 68% and 95% quantiles
- Computed via `_estimate_cv_residuals()` in trainer.py
- `predict_with_intervals()` returns (prediction, lower_bound, upper_bound)
- Integrated into Predict tab with "Include Confidence Intervals" checkbox
- Stored in model bundle manifest
- Works for ALL model types: sklearn, NN, ensemble

#### Model Cards
- Auto-generated MODEL_CARD.md in each saved model bundle
- Includes: model details, intended use, dataset characteristics, metrics, confidence intervals, feature names, limitations, reproducibility instructions, citation guidance

#### Dual Metrics: Internal + External Validation
- Models report metrics on BOTH internal test split AND external validation set
- 3 new columns in Results tab: "Val. RMSE/Acc", "Val. MAE", "Val. R2/F1"
- NO data leakage: external validation NEVER used for training, only final evaluation
- `get_validation_arrays()` applies fitted preprocessor before returning data (critical fix for correct scaling)
- All training flows (single, all, optimized, NN, ensemble) properly pass external validation

#### User-Friendly Error Messages
- `ErrorMessageTranslator` converts 15+ technical error patterns into plain English
- Covers: FileNotFound, PermissionError, empty data, encoding issues, NaN values, memory errors, missing columns, small datasets, convergence failures
- `show_friendly_error()` displays formatted QMessageBox with HTML formatting
- All messages in English

#### Comprehensive Tooltips
- 53+ educational tooltips across all 8 tabs
- Explains every parameter, button, and configuration option

### Changed
- **Version bump to 2.0.0** across all files (app, config, docs, tests)
- All user-facing messages translated from Spanish to English
- Lazy torch imports: `app/core/__init__.py` and `app/__init__.py` no longer import nn_builder at package level (prevents PyTorch DLL errors during test collection)
- Results table column count fixed: 12 columns (was 13, causing phantom "13" column)

### Fixed
- **External validation preprocessing** (CRITICAL): `get_validation_arrays()` now applies the fitted preprocessor to validation data before returning. Without this, models trained on scaled data produced garbage predictions on raw validation data, resulting in negative R2 values
- **Optimization config export**: `filter_orphaned_entries()` now uses normalized name comparison (lowercase + underscore) and never removes optimizations with valid data (best_params, best_score, or all_trials)
- **Random state respected**: `create_estimator()` now accepts and applies user's random_state. No longer hardcoded to 42. Special handling for CatBoost (random_state -> random_seed)
- **CV folds respected**: All training threads (TrainingThread, TrainBestThread, NN, Ensemble) properly receive and pass cv_folds to trainer
- **Feature Importance export**: Removed # bar chart symbols that caused visual clutter
- **PyTorch 2.6+ weights_only compatibility**: `torch.load()` uses explicit `weights_only=False`
- **Test suite**: 419 tests, 19 skipped (optional dependencies), 0 failures, 0 warnings

#### Histogram Gradient Boosting (New Model)
- Added `HistGradientBoostingRegressor/Classifier` as "Histogram Gradient Boosting" in model registry
- Parallel tree building via OpenMP (10-100x faster than classic Gradient Boosting on large datasets)
- Full integration: training, optimization (6 optimizable parameters), CV, ensemble, session export
- Does NOT replace classic Gradient Boosting — both available for backward compatibility
- 12 dedicated tests in `test_trainer.py::TestHistGradientBoosting`

#### SHAP Explainer Architecture v2.0
- Centralized `_compute_shap_with_best_explainer()` helper — single source of truth for explainer selection
- **TreeExplainer** for all tree models (RF, XGB, LGBM, HGB, GB, CatBoost) — exact, deterministic
- **LinearExplainer** for linear models (Ridge, LR, SVM linear) — exact
- **DeepExplainer** for PyTorch Neural Networks — specialized, faster than KernelExplainer
- **KernelExplainer** only as absolute last resort for generic models (k-NN, etc.)
- Removed silent TreeExplainer→KernelExplainer fallback that caused slow, non-reproducible results
- 4 tests verifying correct explainer selection per model family

#### Parallelization v2.0
- All tree models use `n_jobs=recommended_n_jobs` (all cores minus one for OS):
  - RandomForest, XGBoost, LightGBM, CatBoost (`thread_count`)
- `GradientBoostingRegressor` remains sequential (sklearn design limitation)
- `HistGradientBoostingRegressor` uses OpenMP internally (automatic parallelism)
- sklearn CV uses `n_jobs=1` to avoid nested parallelism / oversubscription
- `torch.set_num_threads(recommended_n_jobs)` for NN training on CPU
- Explainer `n_jobs` auto-detects model type: `n_jobs=1` for multi-core models, full cores for single-threaded models
- **Windows executable safety**: `n_jobs=1` in explainability when `sys.frozen=True` (prevents joblib multiprocessing from spawning duplicate application windows)

#### Explainability Tab Guards
- Anti-double-click guard on "Run Explainability Analysis" button (prevents duplicate threads)
- `_analysis_completed` flag prevents multiple QMessageBox dialogs from single run
- Thread reference cleaned up after completion (enables re-runs)
- "Explain Prediction" button also protected against rapid clicks

#### Neural Network Export & Reproducibility
- Session export now includes complete NN training configuration: random_seed, cv_folds, problem_type, epochs, batch_size, learning_rate, optimizer, scheduler, weight_decay, early_stopping settings, gradient clip value
- Session export now includes NN optimization configuration: opt_n_trials, opt_cv_folds, opt_epochs_per_trial, cv_epochs_per_trial, opt_patience, cv_patience, selected_params
- TXT/MD renderers reorganized to show reproducibility-critical fields first (random_seed, cv_folds, problem_type) before other config
- Architecture layer details now shown with proper "Layer Details" section in both TXT and MD
- Fixed NN entries potentially missing from export when `nn_training_results` had None entries

#### NN Export — Eliminated Duplicate Entries (v2.0)
- **Root cause**: NNs appeared TWICE in export — once from `trainer.results` (as generic "Training" entry) and once from `nn_tab.nn_training_results` (as "Neural Network" entry with full config)
- **Fix**: NN results are now collected ONLY from `trainer.results`. The collection loop separates NNs (`is_neural_network=True`) from sklearn models. NNs go to `add_nn_training_result()` with full architecture/history/config. Sklearn models go to `add_training_results()`.
- **Eliminated**: The separate `nn_tab.nn_training_results` collection block that caused duplication
- **Side effect fixed**: `Random State: None` — now shows correct seed from `result.random_state`
- **Side effect fixed**: `add_nn_training_result()` now accepts `metrics` and `training_time_sec` parameters so NNs appear correctly in the ranking (not with R2=0.0000)
- **Ranking**: NNs now sorted by `-best_val_loss` proxy when standard metrics unavailable

#### NN Export — Training Config Saved in Result (v2.0)
- `TrainingResult` now stores complete `training_config` dict at training time
- `TrainingResult.random_state` now guaranteed non-None (defaults to 42)
- NN export reads config FROM the `TrainingResult` itself — not from GUI tab widgets
- This means the config is always accurate even if user changed GUI values after training
- Sklearn models also store their full params in `training_config`
- Architecture description extracted from stored `nn_builder.architecture` (layer counts, activations, dropout)
- Renderer shows reproducibility-critical fields first: random_seed, cv_folds, problem_type, epochs, batch_size, LR, optimizer, scheduler, weight_decay, ES settings

#### torch.manual_seed() for Neural Networks (Critical Fix)
- `build_model()` now calls `torch.manual_seed()`, `np.random.seed()`, `random.seed()`, and `torch.cuda.manual_seed_all()` before creating the model
- Seed priority: (1) explicit parameter, (2) TrainingConfig.random_seed, (3) default 42
- Fixed order in training flow: `create_training_config()` must be called BEFORE `build_model()` so the seed is available during weight initialization
- Fixed same issue in CV fold training (`trainer.py`) and GUI training/optimization flows (`nn_builder_tab.py`)
- 2 tests verifying reproducible weight initialization: same seed → identical weights, different seed → different weights

#### Integration Fixes — Export & Optimization Reliability (Critical)
These fixes were validated against a comprehensive integration test suite that simulates the exact multi-step GUI workflows users perform. See `tests/integration/test_full_pipeline.py` (32 tests) and `tests/core/test_session_exporter_integration.py` (14 tests).

- **CRITICAL — Optimizations were DELETED when repeating same algorithm**: `optimization_tab.py` and `nn_builder_tab.py` replaced previous optimizations of the same algorithm instead of appending. When a user optimized XGBoost (5 folds) then XGBoost again (3 folds), only the second optimization remained. The first trained model (`XGBoost_Optimized`) lost all its optimization details in export. **Fix**: Changed from replace-filter (`[r for r in list if r[0] != model_type]`) to simple append. **Covered by**: `test_multiple_optimizations_preserved`, `test_both_optimizations_preserved_in_export`

- **CRITICAL — Export read optimization config from GUI widgets**: When collecting optimization results for export, `results_tab.py` read `cv_folds`, `n_trials`, `random_state` from the current GUI spinners, not from the actual optimization that was run. If user optimized with 5 folds then changed to 3 folds and optimized again, the first model's export showed 3 folds. **Fix**: Added `optimization_config` field to `OptimizationResult` dataclass (stores actual config at optimization time). `results_tab.py` now reads `result.optimization_config` instead of GUI values. **Covered by**: `test_optimization_config_not_from_gui`, `test_each_model_links_to_correct_optimization_config`

- **CRITICAL — NN export silently destroyed ALL results**: `add_nn_training_result()` was called with `history_stats=hist_stats`, but that method does not accept that parameter (computes stats internally). The `TypeError` was caught by a single `try/except` wrapping the entire results loop, causing **zero** results in export. **Fix**: Removed invalid parameter. Changed from single try/except around entire loop to per-result error handling so one bad model doesn't destroy the export. **Covered by**: `test_no_zero_results_message`, `test_export_with_all_model_types`

- **CRITICAL — `'algorithm' is not defined` broke NN optimization**: `optimize_nn_architecture()` added `optimization_config` with `'algorithm': algorithm`, but that function has no `algorithm` parameter. NN optimization completely failed. **Fix**: Changed to `'algorithm': "Neural Network"`. **Covered by**: optimizer tests

- **Linking broken for `_Optimized_1`, `_Optimized_2` models**: `XGBoost_Optimized_1` normalizes to `xgboost_optimized_1`. The code used `endswith('_optimized')` which failed for names with counter suffixes. **Fix**: Changed to `'_optimized' in norm` with `split('_optimized')[0]` for base name extraction. **Covered by**: `test_each_model_links_to_correct_optimization_config`

- **NN export missing CV fold breakdown**: Neural Network sections in export did not show per-fold CV scores. **Fix**: `add_nn_training_result()` now accepts `cv_metrics` parameter. TXT/MD renderers show CV fold table for NNs (same format as sklearn models). **Covered by**: `test_nn_has_cv_fold_breakdown`, `test_nn_cv_renders_in_txt_output`

- **NN optimized missing Optimization Time**: Optimized+retrained NNs did not show optimization time in export. **Fix**: `add_nn_training_result()` accepts `optimization_time_sec`. Post-processing in `results_tab.py` matches NN entries with `nn_optimizer_results` to populate the field. **Covered by**: `test_optimized_nn_has_optimization_time`, `test_optimized_nn_opt_time_renders`

- **NN history reference sharing**: Multiple NNs in export showed identical histories because `nn_history` was passed by reference. **Fix**: `copy.deepcopy(history)` in `train_neural_network()`. **Covered by**: `test_nn_history_not_shared`

#### NN Optimization — 'max_layers' NameError (Critical Fix)
- **Root cause**: `optimize_nn_architecture()` referenced `max_layers`, `min_units`, `max_units` at line 667 when building `optimization_config`, but these variables were never defined in the function scope (not parameters, not local variables). The optimization would run all trials successfully, then crash at the very end when creating the `OptimizationResult`.
- **Fix**: Extract values from `param_ranges` dict: `param_ranges.get('n_layers', (1,5,'int'))[1]` for max_layers, `param_ranges.get('n_units', (16,512,'int_log'))[0/1]` for min/max_units.
- **Covered by**: `test_nn_optimization_completes_with_config`, `test_nn_optimization_with_custom_param_ranges`

#### NN Optimized — [OPTIMIZATION DETAILS] Missing (Critical Fix)
- **Root cause**: Two separate problems: (1) NN optimizations from `nn_builder_tab.nn_optimizer_results` were NEVER added to the exporter as Optimization entries — only used for `optimization_time_sec` in post-processing. (2) `_link_optimization_entries()` only iterated over `section == 'Training'` entries, but optimized NNs go to `section == 'Neural Network'`. So even if the optimization entries existed, they couldn't be linked.
- **Fix in 3 files**:
  - `results_tab.py`: Added collection block that iterates `nn_builder_tab.nn_optimizer_results` and calls `add_optimization_result(algorithm='Neural Network', ...)` for each NN optimization. Uses algorithm `'Neural Network'` which normalizes to `'neural_network'` in `_opt_entries`.
  - `session_exporter.py`: Extended `_link_optimization_entries()` to also process `'Neural Network'` entries. When a NN model name like `NN_Optimized` normalizes to `nn_optimized`, the base is `nn`, which maps to lookup key `'neural_network'` (the key used when adding NN optimizations). Also extended `_get_linked_algorithms()` for orphan filtering.
  - `session_exporter.py`: Added `_render_optimization_inline_txt/md()` call in the Neural Network rendering block (both TXT and MD renderers) to display `[OPTIMIZATION DETAILS]` inline with the trained NN.
- **Covered by**: `test_nn_optimized_has_optimization_details`, `test_two_nn_optimizations_both_have_details`

#### Manual Training — CV Folds Not Configurable (Reproducibility Fix)
- **Root cause**: `ManualParamDialog` had no `cv_folds` field, and `_train_with_manual_params()` created `TrainingThread` without passing `cv_folds` — so all manual models used the hardcoded default of 5 folds. Two users with identical manual parameters but different desired CV folds would get indistinguishable results, compromising reproducibility.
- **Fix in `training_tab.py`**: (1) Added `cv_folds` QSpinBox to `ManualParamDialog` (range 2-10, default 5) in the Common Parameters section. (2) Stored `cv_folds` in `custom_params` via `_on_accept()`. (3) `_train_with_manual_params()` now reads `custom_params.get('cv_folds', self.cv_spin.value())` and passes it to `TrainingThread`.
- **Covered by**: `test_manual_training_with_custom_cv_folds`

#### Explainability — HistGradientBoosting & LightGBM Show Zero/N/A (Critical Fix)
- **Root cause**: `HistGradientBoostingRegressor` does NOT have `feature_importances_` (unlike RandomForest, XGBoost, etc.). In `get_feature_importance_summary()`, the code only checked `hasattr(model, 'feature_importances_')` — when this was false (for HistGB), no `model_importance` column was created. Additionally, `_get_explainer_n_jobs()` only checked the model's `n_jobs` parameter to detect parallelism, but HistGradientBoosting uses OpenMP internally WITHOUT exposing an `n_jobs` parameter — so the function returned `n_jobs=-1` (all cores), causing oversubscription and zero/NaN results.
- **Fix in `explainer.py`**:
  1. **Surrogate `model_importance`**: For models without `feature_importances_`, compute a surrogate via `permutation_importance` and store it as `model_importance`. The GUI always has a value to display.
  2. **Smart parallelization in `_get_explainer_n_jobs`**: Now detects multi-threaded models by class name (RandomForest, XGB, LGBM, HistGradientBoosting, CatBoost, etc.) in addition to the `n_jobs` parameter. Single-threaded models (SVR, Ridge) get `n_jobs=-1` (fast). Multi-threaded models get `n_jobs=1` (no oversubscription). PyInstaller Windows always gets `n_jobs=1`.
  3. **Better error handling**: Wrapped surrogate computation in try/except with informative logging.
- **Covered by**: `test_hist_gradient_boosting_has_model_importance`, `test_lightgbm_feature_importance_nonzero`, `test_all_registry_models_feature_importance`

#### Explainability — UnboundLocalError: 'analysis_model' (Critical Fix)
- **Root cause**: In `explainability_tab.py` line 637, `hasattr(analysis_model, 'feature_importances_')` was called BEFORE `analysis_model` was assigned. The assignment happened in lines 647/651, but the `hasattr` check was in line 637.
- **Fix**: Moved the `feature_importances_` check block to AFTER `analysis_model` is assigned.
- **Covered by**: 9 new explainability integration tests covering all model types (tree, linear, kernel, optimized, all-registry)

#### ManualParamDialog — random_state/n_jobs shown for deterministic models (UX Fix)
- **Root cause**: `ManualParamDialog` showed `random_state` and `n_jobs` for ALL models, including deterministic ones like SVR and Ridge that don't accept these parameters. Users would set values that had no effect, causing confusion.
- **Fix**: Dialog now inspects the selected model's constructor signature and only shows `random_state` if the model accepts it, and `n_jobs` if the model accepts it. `cv_folds` is always shown (applies to all models).

---

## [2.0.0] - 2026-08-24 — Post-Release Final (CV Report Fixes, LOO-CV R²/MAE, GUI Consistency)

### Fixed

#### Session Export CV Report (BUG-001 / BUG-002)
- **Train/Test columns interchanged in Mean row**: `session_exporter.py` printed `cv_test_mean` under the Train Score column and `cv_test_std` under the Test Score column. Fixed in both TXT and Markdown renderers for training, optimization, and NN sections.
- **Std mislabeled as Mean Test Score**: `cv_test_std` was printed in the Test Score column with the label "Mean" and a trailing "(std)" note. Now Mean and Std are on separate rows, each with correct Train and Test columns.
- **New fields exported**: `cv_train_mean`, `cv_train_std`, `cv_test_metric` ("r2" or "mae"), `cv_r2_global` added to all training and optimization entries.

#### LOO-CV R²/MAE Fix (CRITICAL)
- **Root cause**: `cross_val_score` with `scoring='r2'` returns `nan` for every LOO fold because R² requires variance (≥2 points). Each LOO fold has exactly 1 test sample.
- **Fix in `trainer.py`**: `_cross_validate()` now uses `cross_val_predict` for LOO-CV + regression, then computes `r2_global` (R² over all N OOF predictions) and `test_score` as per-sample MAE.
- **Fix in `optimizer.py`**: `HyperparameterOptimizer` and `GridSearchOptimizer` use `scoring='neg_mean_absolute_error'` for LOO-CV + regression, ensuring every trial has a valid objective value.
- **Fix in `session_exporter.py`**: LOO-CV sections now show `R² (global)` at the top and a per-sample MAE table below. K-Fold sections show Train + Test Score side by side with separate Mean/Std rows.
- **UI fixes**: `training_tab.py`, `optimization_tab.py`, `results_tab.py` Details popups updated to show Train + Test Score for K-Fold, and R² (global) + MAE for LOO-CV.

#### Optimization Tab UX
- **Grid Resolution info label**: Added `grid_info_label` that shows an orange info message when Bayesian is selected: "Grid Resolution is only used with Grid Search. Bayesian optimization uses adaptive sampling, not fixed grids." Cleared when Grid Search is selected.
- **ManualParamDialog LOO info**: Added `_manual_loo_label` that disables the CV Folds spinner and shows an info message when LOO-CV is selected.

#### NN Builder Tab Layout
- Right-side console panels reduced from ~50% to ~25% of screen width (`splitter.setSizes([750, 250])`).
- Minimum heights reduced to 60/40/60/40 (was 100/60/120/60).
- Action buttons reorganized into a non-overlapping 2-row layout (info labels in row 0 spanning 3 columns; buttons in row 1, columns 0/1/2).

#### Test Suite Updates
- `tests/core/test_loo_cv.py` updated to verify `test_score_metric == 'mae'`, `r2_global` is present and not `nan`.
- All 842 tests pass, 0 failures.

---

#### Test Suite: 842 tests (590 unit + 120 integration + 143 new + 14 GridSearch + 10 LOO-CV), 69 skipped, 0 failures
- **43 full-pipeline integration tests** (was 32): Added NN optimization completion, ensemble weight optimization, classification pipeline, reproducibility, all export formats (TXT/MD/CSV/JSON), model save/load with manifest, registry parameter testing
- **14 session exporter integration tests** (unchanged)
- **421 unit tests** across all 16 test modules
- **32 full-pipeline integration tests** (`tests/integration/test_full_pipeline.py`): Cover complete workflows from data loading → preprocessing → training (OOTB, manual, optimized) → neural networks → ensemble → explainability → inference → export. Test every model in the registry, all imputation/scaling strategies, CV fold handling, model persistence, unique naming, and edge cases (NaN data, single CV score, empty optimizations).
- **14 session exporter integration tests** (`tests/core/test_session_exporter_integration.py`): Cover the exact multi-optimization multi-model scenarios that previously failed: two optimizations of same algorithm, NN + sklearn in same session, optimized NN with opt time, all models in ranking, no "0 result(s)".
- **421 unit tests** across all 16 test modules covering individual classes in isolation.

#### Additional Bug Fixes (Post-v2.0.0 release)
- **Fixed**: `self.parent = parent` in `DataTab.__init__` overwriting PyQt's `parent()` method (same bug as ResultsTab)
- **Fixed**: Missing `_update_feature_columns()` method in `DataManager` — caused `AttributeError` when setting ID column
- **Fixed**: `getattr(opt_tab, 'timeout_spin', lambda: None)().value()` causing `TypeError: 'QSpinBox' object is not callable` — silently aborted ALL optimization result collection in session export
- **Fixed**: Optimization entries appearing as phantom headers `[N] OPTIMIZATION: XGBoost` in export — now pre-filtered before rendering loop
- **Fixed**: `candidate_names.append("Cancel")` dead code in crystal linking dialog
- **Fixed**: `scan_for_noncrystalline()` iterating entire DataFrame instead of just unique matched IDs
- **Fixed**: `non_crystalline_ids`/`non_crystalline_reasons` declared but never used in `link_structures()`
- **Fixed**: Session Summary counting linked Optimization entries (now only counts visible entries)
- **Fixed**: Final Ranking including duplicate Optimization entries
- **Fixed**: Reproducibility Checklist listing duplicate optimization entries
- **Fixed**: `n_features` in `_analyze_dataset()` counting target + ID columns (now excludes both)
- **Fixed**: `ModelConfig.n_jobs` default changed from `-1` (dangerous) to `0` (use model default)
- **Fixed**: `IsolationForest(n_jobs=-1)` in domain_applicability → `recommended_n_jobs`
- **Fixed**: `main_executable.py` version string: `1.4.0` → `2.0.0`

#### New Feature: Explainability Results Export (Post-v2.0.0)
- **"Export Results as Text"** button in Explainability Tab — exports all analysis results to a `.txt` file
- Exports: Feature Importance table, SHAP Values, Partial Dependence, Domain Applicability, Prediction Explanation, and full Analysis Log
- Auto-generated filename includes model name and timestamp: `explainability_{model_name}_{timestamp}.txt`
- Formatted report with clear section separators and aligned tables
- Requires no external dependencies — uses standard Python file I/O
- Non-destructive: existing functionality remains completely unchanged

#### Crystal Feature Unification (Post-v2.0.0) — Critical Architecture Fix
- **Root cause**: Two separate classes (`CrystalStructureFeaturizer` with 149 features and `CrystalFeatureExtractor` with 42 features) computed crystal descriptors independently using different formulas, naming conventions, and property sources. Models trained with "direct CIF loading" were incompatible with predictions using "CIF-to-CSV linking".
- **Fix**: `CrystalFeatureExtractor` now uses `CrystalStructureFeaturizer` internally as its sole featurization core. Both workflows produce the same 149 features.
- **Naming**: Direct loading uses original names (`lattice_a`, `density`, `magpie_comp_*`, `rdf_peak_1`, etc.). Linking adds `crystal_` prefix (`crystal_lattice_a`, `crystal_density`, `crystal_magpie_comp_*`, `crystal_rdf_peak_1`, etc.).
- **Impact**: All crystal workflows (direct loading, linking, prediction) now use identical computation logic, eliminating the feature mismatch artifact.
- **Tests updated**: `test_crystal_feature_extractor.py`, `test_dataset_exporter.py`, `test_feature_consistency.py`, `test_inference_linked_structures.py`, `test_coverage_gaps.py` updated for unified naming.
- **Documentation**: `FEATURES_CRYSTAL.md` rewritten to document the unified 149-feature set across both workflows.

#### Molecule Feature Verification (Post-v2.0.0)
- Verified that `MoleculeFeatureLinker` already uses `MoleculeFeaturizer` internally — molecules correctly produce the same 149 features in both direct loading and linking workflows. No changes required.

#### Grid Search Integration (Post-v2.0.0)
- **New `GridSearchOptimizer` class** in `app/core/optimizer.py` — drop-in replacement for `HyperparameterOptimizer` with identical `optimize_sklearn_model()` signature.
- **Supported**: All sklearn regressors/classifiers (Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost, SVR, k-NN, Ridge).
- **Resolution levels**: Coarse (3 points), Medium (5), Fine (7) per parameter.
- **Safety caps**: Maximum 5,000 combinations; PyInstaller uses `n_jobs=1` to avoid `joblib` re-import issues.
- **UI**: Strategy dropdown in Optimization Tab with real-time combination estimate. Grid resolution selector. Pruning disabled automatically for Grid Search.
- **Not offered for**: Neural Networks (non-rectangular architecture space, PyTorch incompatible with `GridSearchCV`) and Ensemble weight optimization (continuous simplex with sum-to-one constraint).
- **Tests**: 14 tests in `tests/core/test_gridsearch_optimizer.py` covering initialization, grid generation, all parameter types, combination estimation, overflow guard, progress callback, and result format.
- **`OptimizationResult`**: New `search_strategy` field (`"bayesian"` or `"grid"`) for session export and reproducibility tracking.

#### LOO-CV Integration (Post-v2.0.0)
- **Leave-One-Out Cross-Validation** now available for sklearn regressors in Training Tab and Optimization Tab (both Bayesian and Grid Search).
- **Hard cap**: LOO-CV blocked for datasets with N > 100 samples.
- **Soft warning**: For N > 30, AutoAIM warns that K-Fold is generally recommended but allows the user to proceed.
- **Backend**: `ModelTrainer._cross_validate()` and `._estimate_cv_residuals()` accept `cv_strategy='loo'`. `HyperparameterOptimizer` and `GridSearchOptimizer` pass `cv_strategy` through to `cross_val_score` / `GridSearchCV`.
- **GridSearch LOO**: Dynamic split detection in `cv_results_` (LOO produces N splits, not K).
- **Not offered for**: Neural Networks and Ensembles (NNs overfit catastrophically with N < 50; ensembles need N ≥ 100 to be robust).
- **Tests**: 10 tests in `tests/core/test_loo_cv.py` covering trainer CV, residual estimation, Bayesian optimization with LOO, GridSearch with LOO, dynamic split detection, and dataclass fields.
- **`TrainingResult`** and **`OptimizationResult`**: New `cv_strategy` field (`"kfold"` or `"loo"`) for session export.
- **`session_exporter.py`**: Exports `cv_strategy` for both training and optimization entries.

---

## [2.0.0] - 2026-10-26 — JOSS-Level Audit & Test Completion

### Code Audit — 18 Bugs Found and Fixed
An exhaustive line-by-line audit of all 35 source files (~15,000 LOC) was performed following Journal of Open Source Software (JOSS) review standards.

| # | File | Line | Bug Description | Fix |
|---|------|------|-----------------|-----|
| 1-10 | Various | — | Import errors (`os`, `QInputDialog`), BOM parsing, XYZ raw format, `self.parent` PyQt overwrite, `get_column_info` missing, dead `feature_checkboxes` code, `_export_joblib` signature, `_file_path` persistence | Corrected in prior sessions |
| 11 | `model_registry.py` | 450 | `Ridge` mapped to `LogisticRegression` instead of `RidgeClassifier` | Fixed classifier mapping |
| 12 | `results_tab.py` | 847 | `_export_joblib_legacy(model_name)` called with invalid argument | Removed extraneous argument |
| 13 | `nn_builder.py` | 790 | `val_loss if val_loss else 0` — `0.0` is falsy, causing valid loss to be discarded | Changed to `val_loss is not None else 0.0` |
| 14-15 | `optimizer.py` | 274, 324, 614, 642 | `study.best_value` raises `ValueError` when all trials fail | Added `completed_trials` guard before access |
| 16 | `ensemble_trainer.py` | 290-292 | `hasattr(result, 'model')` without `is not None` check | Added null verification |
| 17-18 | `crystal_feature_extractor.py` | Various | Silent fallbacks without logging | Added `logger.warning()` / `logger.debug()` |

### Test Suite Completion — 143 New Tests
Three previously uncovered core modules now have full test coverage:

- **`tests/core/test_periodic_data.py`** — 73 tests
  - 12 periodic dictionaries (118 elements): ATOMIC_NUMBERS, ATOMIC_MASSES, ATOMIC_RADII, COVALENT_RADII, ELECTRONEGATIVITIES, IONIZATION_ENERGIES, ELECTRON_AFFINITIES, MELTING_POINTS, COMMON_OXIDATION_STATES, GROUP, PERIOD, THERMAL_CONDUCTIVITIES
  - 4 helper functions: `weighted_mean()`, `weighted_range()`, `weighted_std()`, `count_oxidation_states()`, `assign_oxidation_states()`
  - Greenwood & Earnshaw validation: Fe(+3), Cu(+2), Mn(+2), O(-2), F(-1), alkali metals, alkaline earth, transition metals, lanthanides, actinides

- **`tests/core/test_molecule_feature_linker.py`** — 34 tests
  - `MoleculeLinkReport`: match rate, text serialization, list isolation
  - `MoleculeFeatureLinker`: initialization, ID candidate detection, XYZ availability check, full linking pipeline, duplicate ID handling, missing structure NaN filling, clear/idempotency

- **`tests/core/test_crystal_structure.py`** — 36 tests (18 run + 18 skip if no pymatgen)
  - `CrystalStructureLoader`: init, formats, load_file, load_directory, load_files, DataFrame export, failed file tracking
  - `CrystalStructureFeaturizer`: init, feature names, featurize(), featurize_structures(), featurize_dataframe()
  - `CrystalStructureDatasetBuilder`: init, build_from_directory, build_from_files, build_from_structures, target values
  - Pymatgen graceful degradation: loader/featurizer work without pymatgen installed

### Test Suite Final Status
- **842 tests** — 0 failures, 69 skipped (optional dependencies: PyTorch, Optuna, SHAP, pymatgen)
- **35 test modules** covering core, integration, and utilities
- **pytest** with `-W error::DeprecationWarning` passes cleanly

---

## [1.4.0] - 2026-06-27

### Added
- **Select Parameters to Optimize** — choose which hyperparameters Optuna optimizes:
  - Dialog with checkboxes for every parameter of every model
  - "Select All" / "Select Recommended" / "Clear All" quick buttons
  - Recommended mode selects only high-impact parameters for faster CPU optimization
- **LightGBM min_child_samples** — added to manual configuration and Bayesian optimization
- **Random State actually works** — changing the random seed produces genuinely different results
- **High-precision parameter input** — 10-12 decimal places for fine sensitivity testing
- **NN Tab redesign** — compact toolbar, limited-height panels, epoch-by-epoch Training Results table
- **Comprehensive test suite** — 290 automated tests covering all core and utility modules

### Fixed
- Ensemble deduplication, safe parent access, parameter sync between optimizer and registry, manual model deletion

---

## [1.3.0] - 2026-06-24

### Added
- **Train with Custom Parameters** — manually configure any regressor's hyperparameters
- **Manual Ensemble Weights** — configure custom weights for each model
- **NN Training Configuration Expansion** — LR scheduler, min delta, gradient clip
- **Educational Tooltips** — 53 tooltips across all configuration tabs
- **Cross-Validation Fold Breakdown** — per-fold CV scores in Model Details dialog
- **Tab Reorder** — logical workflow: Data -> Training -> Optimization -> Ensemble -> NN -> Results -> Explainability -> Predict

### Fixed
- SVR Gamma crash, GB min_samples_leaf, CatBoost random seed conflict, GPU bundle save

---

## [1.2.0] - 2026-06-14

### Added
- **Crystal Structure Support** — load and featurize CIF, POSCAR, CONTCAR, VASP, XYZ files
- 149 numeric descriptors per structure (lattice, composition, Magpie, structural, advanced)
- **Crystal Structure Prediction** in Inference tab
- **PyInstaller Build Fix** — reliable bundling of pymatgen data files

### Fixed
- SVR & KNN optimization crash, pymatgen data missing, feature explosion (771+ features), string index crash

---

## [1.1.0] - 2026-05-18

### Added
- Neural Network hyperparameter optimization with custom ranges
- "Train with Best Parameters" for regressors
- Configurable K-fold cross-validation for all model types
- Ensemble weight optimization via Bayesian optimization
- Dedicated Ensemble and NN Results tabs

---

## [1.0.0] - 2026-04-08

### Added
- Initial stable release
- Complete desktop AutoML application for materials science
- PyQt6-based GUI
- 106 custom compositional descriptors
- Multiple ML algorithms (RF, XGBoost, LightGBM, CatBoost, SVM, k-NN)
- Neural network training with PyTorch
- Bayesian optimization with Optuna
- SHAP-based explainability
- Model Bundle system for deployment

---

## Version History

| Version | Date       | Description                                                          |
|---------|------------|----------------------------------------------------------------------|
| **2.0.0** | **2026-10-26** | **Zero Data Leakage via sklearn Pipeline (CRITICAL FIX)**: preprocessor refitted inside each CV fold; entire stack migrated (Training, Optimization, Ensemble, NN); saved models are fitted Pipelines; GP Kernel Tuning (5 families, 4 hyperparameters); Session Report Cleanup; Optimization Tab GP integration; 1024+ tests (0 failures, 75 skipped)** |
| 1.4.0   | 2026-06-27 | Select Parameters to Optimize, LightGBM coverage, random state fix   |
| 1.3.0   | 2026-06-24 | Custom Parameters, manual ensemble weights, NN config expansion      |
| 1.2.0   | 2026-06-14 | Crystal structure support, PyInstaller fix                           |
| 1.1.0   | 2026-05-18 | NN optimization, CV folds, ensemble weight optimization              |
| 1.0.0   | 2026-04-08 | Initial stable release                                               |

---

## How to Upgrade

### From Source

```bash
# Backup your existing installation
cp -r autoaim-studio autoaim-studio-backup

# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Run tests to verify
pytest tests/
```

### Pre-built Executable

Download the latest release from the [Releases](https://github.com/cesargabrielvera1-ux/AutoAIM-Studio/releases) page and replace the existing executable.

---

## Contributing to Changelog

When submitting pull requests, please:
1. Add your changes to the `[Unreleased]` section
2. Use the appropriate subsection (Added, Changed, Fixed, etc.)
3. Include the issue/PR number when applicable
4. Keep entries concise but descriptive
