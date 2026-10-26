# AutoAIM-Studio: Auto Artificial Intelligence for Materials Studio v2.0.0

**AutoAIM Studio** is a standalone desktop application for automated machine learning in materials science and beyond. It provides a complete no-code AutoML pipeline with custom compositional feature engineering, crystal structure linking, neural network design, hyperparameter optimization, manual parameter configuration, parameter selection for optimization, session export, model cards, and standalone model deployment.

> **Auto Aim to predictions in minutes, not days.**

**Latest Release**: v2.0.0 (October 26, 2026)  
**Status**: Production-ready | 1024+ tests | 0 failures | 75 skipped (optional deps)

---

## What's New in v2.0.0

### Gaussian Process — Full Kernel Family Tuning (v2.0)
- **5 kernel families**: RBF, Matern, RationalQuadratic, WhiteKernel, RBF+WhiteKernel
- **Kernel hyperparameters tuneable**: `kernel_length_scale`, `kernel_nu` (Matern smoothness), `kernel_alpha_rq` (RQ tail), `kernel_noise_level` (WhiteKernel)
- **Bayesian optimization**: Full kernel tuning via Optuna TPE (recommended)
- **Grid Search**: Scalar GP params only; kernel params automatically excluded with warning
- **Model provenance**: `gp_metadata` captures kernel family, fitted length scale, nu, alpha, noise level
- **Results Tab**: Extended GP details popup with all kernel parameters
- **Training Tab**: `kernel_type` and `kernel_nu` dropdowns with educational tooltips

### Session Report Cleanup (v2.0)
- Pipeline param noise (`preprocessor__*`, `model__` prefix) filtered from exports
- Internal sklearn objects (`steps`, `memory`, `verbose`) excluded
- GP-specific sections: "Gaussian Process Metadata" and "Prediction Uncertainty"

### Optimization Tab — GP Integration (v2.0)
- Gaussian Process added to algorithm dropdown (was missing in v2.0)
- Full parameter consistency with other regressors
- Random State correctly applied to GP; disabled for SVR/k-NN

### Link Crystal Structures (Major Feature)
- **Bridge tabular data with CIF structures** via an ID column
- Extract **149 crystal descriptors** per CIF (lattice, composition, Magpie, structural, advanced)
- Same 149 features as direct CIF loading — unified computation core, `crystal_` prefix for linking
- Auto-detects ID column candidates from your CSV
- Warns about missing CIFs and duplicate IDs (polymorph detection)
- Automatically replaces composition features (comp_*) to avoid duplication
- Full export support with crystal link metadata

### Data Health Check
- Automated 10-category quality analysis (missing values, outliers, correlations, scaling, etc.)
- Score from 0-100 with severity levels (info, low, medium, high, critical)
- Actionable recommendations for each issue found
- HTML report dialog with full detail

### Session Exporter
- Export complete session results in 4 formats: .txt, .md, .csv, .json
- Includes: model parameters, CV per-fold scores, optimization trials, NN epoch history
- Learning curves data (tabular, for external plotting)
- Confidence intervals (68%, 95%) for regression predictions
- Model Cards (auto-generated MODEL_CARD.md in each bundle)
- Reproducibility checklist

### Dual Metrics: Internal + External Validation
- Internal test split metrics + external validation set metrics
- No data leakage — external set only used for final evaluation
- 3 new columns in Results: Val. RMSE/Acc, Val. MAE, Val. R2/F1

### Confidence Intervals for Predictions
- Regression: CV residual quantiles (68%, 95%)
- Available for all model types (sklearn, NN, ensemble)
- Integrated into Predict tab with checkbox

### Model Cards
- Auto-generated MODEL_CARD.md in each saved model bundle
- Includes: model details, intended use, dataset characteristics, metrics, confidence intervals, feature names, limitations, reproducibility instructions, citation guidance

### Learning Curves (Tabular)
- Returns train_sizes, train_scores, test_scores as tables
- Exportable to CSV for external plotting

### User-Friendly Error Messages
- Technical error pattern recognition and translation
- Plain English messages with actionable suggestions
- All messages in English

### New Model: Histogram Gradient Boosting
- Parallel tree building via OpenMP (10-100x faster than classic GB on large datasets)
- 6 optimizable parameters via Bayesian optimization
- Full integration: training, optimization, ensemble, export, explainability

### Zero Data Leakage via sklearn Pipeline (v2.0 CRITICAL FIX)
- **Eliminated data leakage in cross-validation** — previously `StandardScaler` was fitted on the full training set before CV, causing every validation fold to leak into the scaler statistics
- **Migrated entire stack to sklearn Pipeline** — `Pipeline([('preprocessor', preprocessor), ('model', model)])`
- **Preprocessor is refitted inside each CV fold** — `cross_validate(pipeline, X_raw, y, cv=...)` ensures the scaler sees only the training fold
- **All tabs updated** — Training, Optimization (Bayesian + Grid Search), Ensemble, and Neural Network all pass raw DataFrames through Pipeline
- **NN CV loop clones and refits preprocessor per fold** — manual Pipeline behavior for PyTorch models
- **Saved models are complete fitted Pipelines** — inference receives raw DataFrames; preprocessing happens internally with training-only statistics
- **Backward compatible** — code without preprocessor still works with numpy arrays

### New Model: Gaussian Process (v2.0)
- **Native uncertainty quantification** — every prediction includes `prediction_std` (posterior standard deviation, ±1σ)
- **Probabilistic regressor** with automatic length-scale optimization
- **Full kernel family tuning** — RBF, Matern, RationalQuadratic, WhiteKernel, RBF+WhiteKernel, all with tuneable hyperparameters
- Best for small-to-medium datasets (N < 2,000) where prediction confidence matters
- Console warning when selected with N > 2,000 (O(n³) scaling)
- Full integration: training, optimization (alpha, kernel, n_restarts, normalize_y), ensemble, export, Predict tab
- SHAP via KernelExplainer (fallback, slower than TreeExplainer)

### Out-of-Distribution (OOD) Detection + Disclaimer Banner (v2.0)
- **Peer-review compliant transparency** — Every prediction is explicitly labelled as a *model estimate for screening and prioritisation, NOT an experimentally validated property*
- **Yellow disclaimer banner** appears automatically in the Predict tab when any model is loaded, satisfying journal requirements for UI-visible disclaimers
- **Automatic OOD flagging** — Each newly trained model bundles an `OodDetector` that computes the Euclidean distance to the nearest training neighbour in **standardised feature space**
- **Threshold** = 95th percentile of training-set 1-NN distances; fully reproducible and documented in the model manifest
- **Inference** — Samples beyond the threshold receive `ood_flag=True`, their row is tinted red in the results table, and an `ood_warning` column explains the extrapolation
- **Legacy compatibility** — Pre-v2.0 bundles load without errors; OOD columns are filled with neutral values
- **Model Card update** — Limitations section now documents the OOD method, threshold percentile, and training-set size

### Disk-Based Model Cache (v2.0) — Low-RAM Protection
- **Automatic disk caching** — Every trained model is serialised to `~/.autoaim/model_cache/` immediately after training, then removed from RAM. Lightweight metadata (metrics, predictions, configs) stays in memory.
- **On-demand reload** — When you click "Save" or "Export" in the Results tab, the model is transparently loaded back from disk. You never notice the difference.
- **RAM impact** — Training 20 regressors drops from ~4-6 GB peak RAM to ~800 MB-1.2 GB, making AutoAIM usable on 8 GB laptops.
- **Auto-cleanup** — Cache files older than 7 days are removed on startup. Manual cleanup available via `ModelCache.clear_all()`.
- **Cache stats** — `ModelCache.get_stats()` reports total size and entry count.

### Grid Search for Regressors
- Alternative to Bayesian optimization for sklearn regressors
- Exhaustive evaluation with configurable resolution (Coarse/Medium/Fine)
- Safety cap at 5,000 combinations with real-time estimate
- Deterministic and fully reproducible
- Neural Networks and Ensemble weight optimization remain Bayesian-only by design

### Leave-One-Out Cross-Validation (LOO-CV)
- Available for sklearn regressors in Training and Optimization (both Bayesian and Grid Search)
- Recommended for small experimental datasets (N ≤ 30)
- Hard cap at N > 100; warning at N > 30
- Now also available for Neural Networks and Ensembles (each fold re-trains a fresh model — expensive for NNs, so best for small datasets)

### Stratified and Grouped Cross-Validation (v2.0 final)
- **Stratified K-Fold (quantiles)** for regression — every fold preserves the target distribution; ideal for skewed targets (e.g. Tc, band gaps). Automatic bin reduction with an info note when the target has many repeated values. Regression-only; classification already stratifies by class under K-Fold.
- **Reserved bin for a degenerate value** (within Stratified) — isolate a point mass meaning "property absent" (Tc = 0 K, Eg = 0 eV, E_hull = 0) into its own stratum so every fold carries the same proportion of it. Auto-detected with a suggestion, or set manually with a data-grounded warning (blocked if too few samples to stratify).
- **Group K-Fold** — samples from the same group never cross the train/test boundary, preventing chemical/batch data leakage. Four grouping sources, no new dependencies (matminer NOT required):
  - *Chemical system* (same element set — the standard anti-leakage split for materials data)
  - *Exact formula* (reduced stoichiometry)
  - *Clustering* (deterministic agglomerative clustering over element fractions)
  - *Any dataset column* (synthesis batch, lab, source, year)
- **Leave-One-Group-Out (LOGO)** — each group left out once; folds = number of groups. Uses out-of-fold predictions with global R² + per-group MAE (R² is undefined for single-group folds).
- Single shared `cv_factory` builds splitters/scoring for trainer, both optimizers, Neural Networks, and Ensembles; single shared `CVConfigWidget` provides the UI in the Training tab, Optimization tab, Neural Network tab, Ensemble tab, and Manual Parameter dialog.
- Validation blocks impossible configs (more folds than groups, LOGO with <3 or >100 groups, group column with missing values) and warns about expensive/imbalanced ones.

### Unified Crystal Featurization (149 descriptors)
- Direct CIF loading and CIF-to-CSV linking now produce the same 149 features
- Eliminates the previous artifact where linking only produced 42 features with different names
- `CrystalFeatureExtractor` rewritten to use `CrystalStructureFeaturizer` as its sole core

### Explainability Results Export
- Export all Explainability Tab results as formatted text (Feature Importance, SHAP, Partial Dependence, Domain Analysis, Prediction Breakdown, Log)
- Auto-generated filename with model name and timestamp

### SHAP Architecture v2.0
- **TreeExplainer** for all tree models (exact, deterministic)
- **LinearExplainer** for linear models (exact)
- **DeepExplainer** for Neural Networks (specialized, fast)
- **KernelExplainer** only as last resort for generic models
- Removed silent Tree→Kernel fallback (was slow and non-reproducible)

### Parallelization
- All tree models use all available CPU cores (minus one for OS)
- sklearn CV uses single core to avoid nested parallelism
- Automatic thread limiting on Windows executables (prevents duplicate windows)

### Bug Fixes
- **Export reliability** — per-result error handling prevents one bad model from destroying the entire export
- **Optimization accumulation** — multiple optimizations of the same algorithm are preserved (no longer overwrite previous)
- **Optimization config accuracy** — export reads actual config from stored result instead of current GUI values
- **NN export completeness** — CV fold breakdown, optimization time, and architecture details for all NNs
- **NN optimization** — `'algorithm' is not defined` and `'max_layers' is not defined` errors fixed
- **Model linking** — `_Optimized_1`, `_Optimized_2` correctly link to their respective optimizations
- **Manual training CV folds** — cv_folds now configurable in ManualParamDialog (was hardcoded to 5)
- **Explainability zero values** — HistGradientBoosting & LightGBM now produce valid feature importance via surrogate permutation fallback with smart parallelization
- **Explainability crash** — UnboundLocalError on 'analysis_model' fixed (variable used before assignment)
- **Explainability NN dedup** — Neural Networks no longer appear duplicated in the model combo
- **UX: Deterministic models** — random_state/n_jobs hidden for SVR/Ridge that don't accept them
- External validation preprocessing (critical fix for R2 accuracy)
- Filter orphaned entries now preserves valid optimization data
- Results tab phantom Optimization headers eliminated
- Feature count now excludes target and ID columns
- Column index fix in Results table (column "13" phantom removed)
- Random state respected across all models (no longer hardcoded to 42)
- **CatBoost random-seed fix** — optimizers passed `random_state` (ignored by CatBoost); now mapped to `random_seed`. CV fold shuffling now follows the user's seed (synced into cv_config across all tabs). Optimization RS is model-aware (disabled for deterministic SVR/k-NN)
- CV folds user-configurable and respected in all training paths
- Feature Importance export without # bar symbols
- Educational tooltips across all 8 tabs (53+ tooltips)
- **Session Export CV columns fixed (BUG-001/BUG-002)** — Mean row no longer interchanges Train/Test columns; Std is now reported as a separate row instead of being mislabeled as Mean Test Score
- **LOO-CV R²/MAE fix (CRITICAL)** — R² is mathematically undefined for single-sample test sets. LOO-CV now computes a **global R²** over all out-of-fold predictions, and reports **per-sample MAE** as the per-fold metric. Session exporter shows R² (global) and per-sample MAE for LOO-CV.
- **GUI CV Details consistency** — All "Details" popups (Training, Optimization, Results tabs) now display **Train Score + Test Score** side by side for K-Fold, matching the session export format. LOO-CV shows R² (global) + per-sample MAE.
- **Optimization Grid Resolution UX** — When Bayesian optimization is selected, the Grid Resolution selector is disabled and an info message explains that Bayesian uses adaptive sampling, not fixed grids.
- **NN Builder layout** — Right-side console panels reduced from ~50% to ~25% of screen width, with smaller minimum heights, giving adequate space for all configuration buttons on the left panel.

### JOSS-Level Code Audit (August 2026)
An exhaustive line-by-line audit of all 35 source files (~15,000 LOC) was performed, following Journal of Open Source Software (JOSS) review standards. **18 latent bugs were identified and corrected:**

| # | File | Bug | Fix |
|---|------|-----|-----|
| 1-10 | Various | Import errors, BOM parsing, XYZ raw format, `self.parent` PyQt overwrite, `get_column_info` missing, dead `feature_checkboxes` code, `_export_joblib` signature, `_file_path` persistence | Corrected in prior sessions |
| 11 | `model_registry.py:450` | `Ridge` mapped to `LogisticRegression` instead of `RidgeClassifier` | Fixed classifier mapping |
| 12 | `results_tab.py:847` | `_export_joblib_legacy(model_name)` called with invalid argument | Removed extraneous argument |
| 13 | `nn_builder.py:790` | `val_loss if val_loss else 0` — `0.0` is falsy, causing valid loss to be discarded | Changed to `val_loss is not None` |
| 14-15 | `optimizer.py:274,324,614,642` | `study.best_value` raises `ValueError` when all trials fail | Added `completed_trials` guard before access |
| 16 | `ensemble_trainer.py:290-292` | `hasattr(result, 'model')` without `is not None` check | Added null verification |
| 17-18 | `crystal_feature_extractor.py` | Silent fallbacks without logging | Added `logger.warning()` / `logger.debug()` |

### Test Suite Completion (August 2026)
- **143 new unit tests** added across 3 previously uncovered core modules:
  - `test_periodic_data.py` — 73 tests (12 periodic dictionaries, 4 helper functions, oxidation states)
  - `test_molecule_feature_linker.py` — 34 tests (ID detection, XYZ linking, pipeline, edge cases)
  - `test_crystal_structure.py` — 36 tests (loader, featurizer, dataset builder, pymatgen graceful degradation)
- **Total: 1024+ tests** — 0 failures, 75 skipped (optional dependencies)
- **114 new tests** for stratified/grouped CV + reserved-bin + NN/Ensemble extension: `test_grouping.py` (25), `test_cv_factory.py` (31), `test_cv_display.py` (16), `test_cv_strategies_e2e.py` (12), `test_special_value_bin.py` (22), `test_cv_nn_ensemble.py` (10)

---

## Previous Highlights

- **v1.4.0** — Published release with stable AutoML pipeline
- **v1.3.0** — Train with Custom Parameters (7 regressors, all params), 53 tooltips, CV fold breakdown, NN scheduler/grad clip, tab reorder, GPU bundle fix
- **v1.2.0** — Crystal Structure Support (CIF, POSCAR, XYZ), SVR & KNN optimization fix, PyInstaller packaging fix
- **v1.1.0** — Initial release with AutoML training, neural networks, Bayesian optimization, ensembles, and model deployment

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Link Crystal Structures** | Connect CIF files to tabular datasets via ID column (v2.0) |
| **Link Molecular Structures** | Connect XYZ files to tabular datasets via ID column (v2.0) |
| **Crystal Structure Loading** | Import CIF, POSCAR, CONTCAR, XYZ, and VASP files for automatic featurization |
| **Crystallographic Featurization** | 149 descriptors per structure: lattice, composition, Magpie (106), bond analysis, RDF |
| **Compositional Featurization** | Custom 106-descriptor Magpie generator from chemical formulas (no MatMiner dependency) |
| **Data Health Check** | 10-category quality analysis with scoring and recommendations (v2.0) |
| **AutoML Training** | Train multiple algorithms simultaneously with cross-validation |
| **Custom Parameter Training** | Manually configure any regressor's parameters and train without optimization |
| **Visual Neural Networks** | Design and train custom PyTorch architectures with a visual interface |
| **Bayesian Optimization** | Hyperparameter tuning with Optuna for regressors, neural networks, and ensembles |
| **Grid Search** | Exhaustive hyperparameter search for sklearn regressors (deterministic, reproducible) |
| **Select Parameters to Optimize** | Choose which hyperparameters to optimize, with "Recommended" mode (v1.5) |
| **Manual Ensemble Weights** | Set custom weights for each model in a weighted-average ensemble |
| **Ensemble Weight Optimization** | Bayesian optimization of ensemble member weights via Optuna |
| **Model Bundles** | Self-contained model packages with manifest.json and MODEL_CARD.md (v2.0) |
| **Standalone Prediction** | Deploy trained models for inference without the full application |
| **Confidence Intervals** | Prediction intervals for regression (68%, 95%) via CV residuals (v2.0) |
| **Session Export** | Export all results in .txt/.md/.csv/.json with full reproducibility info (v2.0) |
| **Explainability** | SHAP values, permutation importance, partial dependence plots, domain applicability, prediction explanations — all exportable as text |
| **Ensemble Models** | Combine multiple models for improved predictions |
| **Cross-Validation Strategies** | K-Fold (default), Stratified K-Fold (quantiles, regression), Group K-Fold, Leave-One-Out (LOO), and Leave-One-Group-Out (LOGO). Grouped strategies prevent chemical/batch leakage (4 grouping sources, no matminer). LOO recommended for small datasets (N ≤ 30) |
| **Educational Tooltips** | 60+ tooltips explaining every parameter across all configuration tabs |
| **Dual Metrics** | Internal test split + external validation metrics without data leakage (v2.0) |

---

## Crystal Structure Linking (New in v2.0.0)

AutoAIM Studio v2.0 introduces the ability to **link crystal structure files (CIF) to tabular datasets** via an ID column. This bridges the gap between tabular machine learning and structural information.

### How It Works

1. **Prepare your CSV** with an ID column that matches your CIF filenames:
   ```
   material_id | formula | band_gap | formation_energy
   mp-1234     | TiO2    | 3.0      | -5.2
   mp-5678     | Fe2O3   | 2.1      | -6.1
   ```

2. **Name your CIFs** to match the ID values:
   ```
   structures/
     mp-1234.cif
     mp-5678.cif
   ```

3. **Click "Link Crystal Structures"** in the Data tab
   - Select your ID column
   - Select the CIF folder
   - AutoAIM matches, extracts features, and merges

### Crystal Features Extracted (149 descriptors)

When linking CIF files to tabular data, AutoAIM extracts the same 149 features as direct CIF loading, with a `crystal_` prefix:

| Category | Count | Examples |
|----------|-------|----------|
| **Lattice** | ~10 | `crystal_lattice_a`, `crystal_lattice_b`, `crystal_lattice_c`, `crystal_volume`, `crystal_density` |
| **Composition** | ~13 | `crystal_avg_atomic_mass`, `crystal_avg_electronegativity`, `crystal_atomic_fraction_std` |
| **Structural** | ~3 | `crystal_packing_fraction`, `crystal_vol_per_atom`, `crystal_spacegroup_num` |
| **Magpie** | **106** | `crystal_magpie_comp_*` (weighted mean, std, min, max, median, p25, p75 for 13 elemental properties) |
| **Advanced** | ~15 | `crystal_rdf_peak_1`, `crystal_avg_coordination`, `crystal_lattice_anisotropy` |
| **Derived** | ~2 | `crystal_total_electrons`, `crystal_bond_lengths_mean` |

See `FEATURES_CRYSTAL.md` for the complete list.

### Polymorph Detection

If multiple rows share the same ID (polymorphs), AutoAIM warns you so you can decide whether to proceed or add unique IDs.

### ⚠️ Limitation: Designed for Crystalline Materials

**AutoAIM's crystal featurizer is designed for periodic crystalline materials** (solids with defined unit cells). It is NOT suitable for:

| Structure Type | Status | Reason |
|---------------|--------|--------|
| **Crystalline molecular solids** | ✅ Supported | Has real unit cell (e.g., organic crystals, MOFs, COFs) |
| **Inorganic crystals** | ✅ Supported | Has real unit cell (e.g., oxides, metals, perovskites) |
| **Isolated molecules** | ❌ Not supported | No unit cell — descriptors reflect simulation box, not molecule |
| **Amorphous materials** | ❌ Not supported | No periodicity — unit cell concept doesn't apply |

**What happens with isolated molecules:** If you load a CIF/XYZ of a single molecule, pymatgen places it in an artificial simulation box. The extracted "density" reflects the box size, not the material. The model may learn spurious patterns.

**For isolated organic molecules**, consider using molecular descriptors (e.g., RDKit) instead. This is on our roadmap for future versions.

---

## Crystal Structure Support (v1.2.0+)

AutoAIM Studio also supports direct loading of crystallographic files as standalone datasets:

### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CIF | `.cif` | Crystallographic Information File |
| POSCAR | `.poscar` | VASP position file |
| CONTCAR | `.contcar` | VASP continuation file |
| XYZ | `.xyz` | Cartesian coordinates |
| VASP | `.vasp` | Generic VASP format |

### Automatic Feature Extraction

When you load structure files directly, the application automatically extracts **149 numeric features** per structure:

| Category | Count | Examples |
|----------|-------|----------|
| **Lattice** | ~10 | `lattice_a`, `lattice_b`, `lattice_c`, `volume`, `density`, `abc_ratio` |
| **Composition** | ~13 | `avg_atomic_mass`, `avg_electronegativity`, `atomic_fraction_std` |
| **Structural** | ~3 | `packing_fraction`, `vol_per_atom`, `spacegroup_number` |
| **Magpie** | **106** | `magpie_comp_*` — weighted mean, std, min, max, median, p25, p75 for 13 elemental properties |
| **Advanced** | ~15 | `rdf_peak_1`, `avg_coordination`, `lattice_anisotropy`, `complexity_index` |
| **Derived** | ~2 | `total_electrons`, `bond_lengths_mean` |

---

## Installation

### Windows (Pre-built Executable)

1. Download `AutoAIM-Studio-v2.0.0-Windows.zip` from the [Releases](https://github.com/your-org/AutoAIM-Studio/releases) page
2. Extract to your preferred location
3. Run `MaterialsAutoML.exe`

No Python installation required. All dependencies are bundled.

### From Source

**Requirements:**
- Python 3.11+
- Windows 10/11, macOS, or Linux

```bash
# Clone the repository
git clone https://github.com/your-org/AutoAIM-Studio.git
cd AutoAIM-Studio

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install pymatgen for crystal structure support (optional but recommended)
pip install pymatgen-core

# Run
python main.py
```

### Building from Source (Windows)

For creating a standalone executable:

```batch
# Setup build environment
setup_build_env.bat

# Compile (onedir mode - recommended)
build_windows_onedir.bat
```

The executable will be created in `dist\MaterialsAutoML\MaterialsAutoML.exe`.

See [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) for detailed build configuration.

---

## Quick Start

### Tab 1: Data
- **Load CSV:** Import your dataset (must contain a target column)
- **Link Crystal Structures:** Connect CIF files to your dataset via ID column (v2.0)
- **Load Crystal Structures:** Import CIF/POSCAR/XYZ files directly (v1.2.0)
- **Apply Featurizer:** Extract Magpie features from chemical formulas (for CSV datasets)
- **Set Target:** Select your prediction target column
- **Data Health Check:** Analyze dataset quality before training (v2.0)

### Tab 2: Train
- Select algorithms (Random Forest, Histogram Gradient Boosting, XGBoost, LightGBM, CatBoost, Neural Network, etc.)
- Configure cross-validation strategy: **K-Fold** (default), **Stratified (quantiles)**, **Group K-Fold**, **LOO**, or **LOGO**
- Grouped CV (Group K-Fold / LOGO) prevents chemical/batch leakage; LOO/LOGO for small datasets
- Click "Train All" and monitor real-time progress

### Tab 3: Optimization
- Select algorithm and choose search strategy: **Bayesian (Optuna)** or **Grid Search**
- Choose cross-validation strategy: **K-Fold** (default), **Stratified (quantiles)**, **Group K-Fold**, **LOO**, or **LOGO**
- Choose which parameters to optimize ("Recommended" or custom selection)
- Set custom parameter ranges (optional)
- Click "Optimize" and review results
- "Train with Best Parameters" for instant deployment

**Bayesian** (default): Smart exploration, fewer trials, handles large search spaces. Recommended for most cases.

**Grid Search**: Exhaustive evaluation of all parameter combinations. Deterministic and fully reproducible. Recommended for ≤3 parameters, small discrete ranges, or baseline comparisons against Bayesian results.

**K-Fold CV** (default): Split data into K parts. Recommended for N > 30.

**Stratified (quantiles)**: Each fold preserves the target distribution. For skewed regression targets. Optionally **reserve a bin for an exact value** (e.g. Tc = 0 K) so every fold carries the same proportion of a degenerate "property absent" marker.

**Group K-Fold / LOGO**: Group never crosses train/test — prevents leakage from chemical families, batches, or any grouping column.

**LOO-CV**: Leave each sample out once. Recommended for small datasets (N ≤ 30). Blocked for N > 100 with a warning.

### Tab 4: Neural Network
- Design architecture visually (layers, units, activations)
- Configure training parameters
- Train with or without cross-validation
- Save/load architectures

### Tab 5: Ensemble
- Select multiple trained models
- Choose strategy (weighted average or stacking)
- Optimize weights with Bayesian optimization

### Tab 6: Results
- Review training, optimization, NN, and ensemble results
- View learning curves and metrics
- External validation metrics when validation data is loaded (v2.0)

### Tab 7: Explainability
- SHAP values, permutation importance, partial dependence plots, domain applicability
- Model-specific explanations with individual prediction breakdowns
- **Export all results as text** for documentation and reporting

### Tab 8: Deploy
- Export trained models as standalone Model Bundles
- Generate prediction scripts for deployment
- Model Cards auto-generated with each save (v2.0)

---

## Test Suite: 1024+ Tests (0 failures, 75 skipped)

AutoAIM Studio has **1024+ automated tests** (1100+ collected, 75 skipped for optional dependencies) covering every module and workflow:

### Unit Tests
Individual class testing across all 30 core and utility modules:
- `test_trainer.py` (47) — Model training, parallelization, HistGradientBoosting
- `test_session_exporter.py` (48) — TXT/MD/CSV/JSON rendering, optimization linking, ranking
- `test_nn_builder.py` (37) — Architecture design, training config, reproducibility
- `test_optimizer.py` (17) — Bayesian optimization, parameter selection
- `test_data_manager.py` (26) — Data loading, preprocessing, feature columns
- `test_data_health_check.py` (24) — Quality analysis, scoring, recommendations
- `test_ensemble_trainer.py` (18) — Weighted average, stacking, weight optimization
- `test_explainer.py` (17) — SHAP, permutation importance, partial dependence
- `test_feature_engineering.py` (23) — Composition parsing, Magpie features
- `test_inference_engine.py` (24) — Model loading, prediction, intervals
- `test_model_registry.py` (19) — Model registration, parameter grids
- `test_model_saver.py` (14) — Bundle saving, manifest generation
- `test_validator.py` (21) — Cross-validation, uncertainty estimation
- `test_crystal_feature_extractor.py` (16) — CIF linking, feature extraction
- `test_domain_applicability.py` (15) — Out-of-domain detection
- `test_error_messages.py` (23) — Friendly error translation
- `test_molecule_featurizer.py` (44) — 3-level featurization, composition/geometry/topology
- `test_inference_linked_structures.py` (22) — Linked-structure detection, predict_with_linked_structures
- `test_dataset_exporter.py` (35) — Feature breakdown, filename suggestion, CSV export, round-trip, edge cases
- `test_molecule_featurizer_bom.py` (24) — BOM, Windows line endings, malformed .xyz, raw format (GaussView/Spartan), parser robustness
- `test_periodic_data.py` (73) — 12 periodic dictionaries, 4 helper functions, oxidation states (NEW v2.0)
- `test_molecule_feature_linker.py` (34) — ID detection, XYZ linking, pipeline, edge cases (NEW v2.0)
- `test_crystal_structure.py` (36) — Loader, featurizer, dataset builder, pymatgen graceful degradation (NEW v2.0)
- `test_grouping.py` (25) — 4 grouping sources, composition-column detection, edge cases (NEW v2.0)
- `test_cv_factory.py` (31) — all 5 CV strategies, scoring, OOF decisions, validation, serialization (NEW v2.0)
- `test_cv_display.py` (16) — shared HTML/text CV breakdown formatter (NEW v2.0)
- `test_cv_strategies_e2e.py` (12) — trainer/optimizer e2e with stratified/group/LOGO, group no-leakage (NEW v2.0)
- `test_special_value_bin.py` (22) — reserved-bin binner, point-mass detection, validation, serialization, e2e (NEW v2.0)
- `test_cv_nn_ensemble.py` (10) — NN + Ensemble training/weight-opt with all CV strategies, group no-leakage (NEW v2.0)
- `test_random_state.py` (16) — RS honoring: per-library naming, reproducibility, fold-shuffle sync, NN weight init (NEW v2.0)
- `test_annotation_imports.py` (43) — undefined-annotation-name guard for frozen builds (NEW v2.0)
- `test_gpr.py` (9) — Gaussian Process: creation, reproducibility, return_std, trainer capture, inference engine, manifest, GridSearch, learning curves, kernel tuning (v2.0)
- `test_gp_kernel_tuning.py` — `_build_gp_kernel`, kernel object construction, `create_estimator` with all 5 kernel families, Grid Search kernel exclusion, Bayesian optimization with kernel params, model provenance metadata (v2.0)

### Integration Tests
End-to-end workflow testing that simulates the complete user journey:

**`tests/core/test_session_exporter_integration.py` (14 tests)**
- Multiple optimizations of same algorithm (both preserved, correct linking)
- Optimization config from stored result (not GUI widgets)
- Neural Network CV fold breakdown in export
- Optimized NN with optimization time
- Mixed sessions with all model types
- No "0 result(s)" in export

**`tests/integration/test_full_pipeline.py` (44 tests)**
**`tests/core/test_explainer.py` (16 tests, 4 skipped — SHAP-dependent)**
- Data loading → preprocessing (all imputation/scaling strategies)
- Training: all registry models, manual params, unique naming
- Manual training: cv_folds configurable (was hardcoded to 5)
- Neural Networks: training, unique names, history isolation, reproducibility
- NN Optimization: completes without NameError, config from param_ranges
- Optimization: sklearn, config preservation, multiple runs
- Ensemble: manual, weighted, unique names
- Ensemble Weight Optimization: Bayesian optimization of weights
- Explainability: SHAP on sklearn and optimized models, all model types, no zero values
- Save/Load: model bundles, manifest.json, inference engine
- Export: all model types, multiple optimizations, TXT/MD/CSV/JSON formats
- Crystal Structure: feature extractor, composition features
- Data Health: quality check, validation, domain applicability
- Classification: full classification pipeline

**`tests/integration/test_coverage_gaps.py` (49 tests, 4 skipped — PyTorch/Optuna)**
- CrystalFeatureExtractor: ID candidates, CIF availability, linking, missing CIFs, reports
- FeatureEngineering: polynomial, statistical, composition, Magpie + oxidation
- MoleculeFeaturizer + training pipeline integration
- Ensemble: equal weights, unequal weights, stacking, single-model graceful failure
- Export with NN optimization results
- DataManager: single row, all-NaN, missing targets, mixed types, all strategy combos (12)
- Model save/load roundtrip with preprocessor pipeline
- Multi-algorithm optimization workflow
- DataHealthCheck: outliers, correlation, constant features, duplicates
- DomainApplicability, SessionExporter orphan filtering, Validator extended

**`tests/integration/test_molecule_linker.py` (24 tests)**
- ID candidate detection, XYZ availability check
- Full linking: all matched, partial, none, missing folder, duplicate IDs
- Feature value differentiation across molecules
- Linked state management (is_linked, clear, get_id_column)
- End-to-end: .xyz → DataManager → train model
- Partial linking still trainable after imputation
- Numeric prefix element parsing (e.g. "1C" → C)

**`tests/integration/test_predict_tab_linked.py` (9 tests, 3 skipped — pymatgen)**
- **Molecule linked**: CSV + .xyz files → link by ID → predict (149 features: mol_* + has_* + 34 periodic properties including oxidation states)
- **Crystal linked**: CSV + .cif files → link by ID → predict (16 features: crystal_* + periodic props)
- **Plain model**: not linked, predict works normally without structure files
- **Error handling**: missing ID column, empty structure folder, partial match (NaN fill)
- Reproducibility: same seed → same results, different seeds → different results
- Edge Cases: NaN data, single CV score, empty optimizations, empty export, NN dedup

Run all tests:
```bash
pytest tests/
```

Run only integration tests:
```bash
pytest tests/integration/ tests/core/test_session_exporter_integration.py
```

---

## Grid Search vs Bayesian Optimization — Technical Justification

AutoAIM v2.0 offers **both** search strategies for sklearn regressors, with Bayesian as the default. Here is why each exists and when to use it:

### Bayesian Optimization (Optuna) — Default
- **How it works**: Builds a surrogate model of the objective function and samples the most promising regions first.
- **Strengths**: Handles large search spaces efficiently; needs 50–100 trials vs. thousands for GridSearch; supports pruning (early stopping of bad trials).
- **Best for**: Most real-world problems with 3+ hyperparameters, continuous ranges, or expensive models.
- **Limitation**: Stochastic — same seed gives similar but not identical results across runs.

### Grid Search — Alternative
- **How it works**: Evaluates every combination of the parameter grid exhaustively.
- **Strengths**: Deterministic and fully reproducible; baseline for papers; easy to reason about.
- **Best for**: ≤3 parameters, small discrete ranges, sanity checks against Bayesian results.
- **Limitation**: Combinatorial explosion — 4 params × 5 values = 625 combinations × 5 CV folds = 3,125 model fits. AutoAIM caps GridSearch at 5,000 combinations and warns the user upfront.

### Why Grid Search is NOT offered for Neural Networks or Ensemble Weight Optimization

| Component | Reason |
|-----------|--------|
| **Neural Networks** | Architecture search spaces are non-rectangular (variable layer counts and sizes). PyTorch models cannot be evaluated with sklearn's `GridSearchCV`. |
| **Ensemble Weights** | Weight spaces are continuous with a sum-to-one constraint. Enumerating a grid over a simplex is computationally wasteful (most grid points violate the constraint). |

These are principled design decisions aligned with the AutoML literature (e.g., Auto-sklearn, TPOT, and H2O AutoML also use Bayesian or evolutionary search for NNs, not GridSearch).

---

## Cross-Validation Strategy — K-Fold, Stratified, Grouped, LOO, LOGO

AutoAIM v2.0 offers **five** cross-validation strategies for sklearn regressors, all sharing a single `cv_factory` backend and a single `CVConfigWidget` UI.

### K-Fold CV — Default
- **How it works**: Splits data into K folds; trains on K-1, validates on 1, rotating K times.
- **Strengths**: Computationally efficient; statistically stable for N > 30.
- **Best for**: Datasets with N > 30. The standard for most ML workflows. (Classification folds are stratified by class automatically.)

### Stratified K-Fold (quantiles) — Skewed targets
- **How it works**: The regression target is binned into quantiles; StratifiedKFold then ensures every fold preserves the target distribution.
- **Strengths**: Removes "unlucky fold" distortion for skewed targets; lower-variance CV estimates.
- **Best for**: Skewed regression targets (Tc, band gaps, formation energies with long tails). Regression-only.
- **Robustness**: Bin count auto-reduces (with an info note) when the target has many repeated values.
- **Reserved bin (optional)**: Isolate a degenerate "property absent" value (Tc = 0 K, Eg = 0 eV, E_hull = 0) into its own stratum so every fold carries the same proportion. Auto-detected with a suggestion, or set manually; blocked if too few samples to stratify, warned (with real counts) if it is the majority or a tiny fraction.

### Group K-Fold — Anti-leakage
- **How it works**: You define groups; a group never crosses the train/test boundary.
- **Why it matters**: Random splits put chemically similar compounds (same family: cuprates, pnictides) or the same batch in both train and test, inflating performance. Grouped CV gives an honest generalization estimate.
- **Grouping sources** (no matminer needed): chemical system (element set), exact/reduced formula, agglomerative clustering over element fractions, or any dataset column (batch, lab, year).
- **Fold sizes are uneven** by design; the report adds a Test Size column.

### Leave-One-Out CV (LOO-CV) — Small datasets
- **How it works**: For N samples, trains N models, each leaving one sample out for validation.
- **Strengths**: Unbiased estimate for small datasets; deterministic; maximizes training data per fold.
- **Best for**: Small datasets (N ≤ 30) common in experimental materials science.
- **Limitation**: Trains N models. Blocked for N > 100; warns for N > 30.

### Leave-One-Group-Out (LOGO) — Strict group holdout
- **How it works**: Each group is left out once; folds = number of groups.
- **Best for**: Grouped data where you want each group strictly held out (e.g. leave-one-chemical-system-out on SuperCon-type datasets).
- **Metrics**: Out-of-fold predictions → global R² + per-group MAE (R² is undefined for a single left-out sample).
- **Limitation**: Needs ≥ 3 groups; blocked for > 100 groups; warns for > 30.

### Advanced CV strategies now extend to Neural Networks and Ensembles

Since the final v2.0.0 polish, all five CV strategies (K-Fold, Stratified with reserved bin, Group K-Fold, LOO, LOGO) are available **consistently across sklearn regressors, Neural Networks, and Ensembles**, through the single shared `cv_factory` backend and `CVConfigWidget` UI. Each NN fold re-trains a fresh network; ensemble CV evaluates the combined prediction per fold (and weight optimization shares the strategy/groups).

Two scoping notes remain by design:
- **Search method**: sklearn regressors offer Bayesian + Grid Search; NN architecture search and Ensemble weight optimization use **Bayesian only** (their spaces are non-rectangular / continuous with a sum-to-one constraint).
- **Cost realism**: LOO/LOGO train one model per sample/group — fine for fast sklearn models, expensive for NNs. The shared validation blocks/warns appropriately, and NNs are still best for N ≥ 50, ensembles for N ≥ 100.

---

## System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Windows 10, macOS 11, Ubuntu 20.04 | Windows 11, macOS 14, Ubuntu 22.04 |
| CPU | 4 cores | 8+ cores (for parallel training) |
| RAM | 8 GB | 16+ GB (for large datasets and NNs) |
| Storage | 2 GB | 5 GB (for PyTorch and full environment) |
| GPU | Not required | Optional (CUDA for PyTorch acceleration) |

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| **v2.0.0** | **Sep 2026** | **Zero Data Leakage via sklearn Pipeline (CRITICAL FIX)**: preprocessor refitted inside each CV fold; entire stack migrated (Training, Optimization, Ensemble, NN); saved models are fitted Pipelines; GP Kernel Tuning (5 families, 4 hyperparameters); Session Report Cleanup; Optimization Tab GP integration; 60+ tooltips; 1024+ tests (0 failures, 75 skipped)** |
| v1.4.0 | Jun 2026 | Published stable release — AutoML pipeline, NN builder, ensembles, deployment |
| v1.3.0 | May 2026 | Train with Custom Parameters, 53 tooltips, CV fold breakdown, NN scheduler/grad clip |
| v1.2.0 | May 2026 | Crystal Structure Support (CIF, POSCAR, XYZ), SVR/KNN fix, PyInstaller fix |
| v1.1.0 | Apr 2026 | Neural Network Optimization, CV for all models, Ensemble Weight Optimization |
| v1.0.0 | Mar 2026 | Initial release — AutoML pipeline, NN builder, ensembles, deployment |

---

## Documentation

| Document | Description |
|----------|-------------|
| [README.md](README.md) | This file — overview and quick start |
| [CHANGELOG.md](CHANGELOG.md) | Complete version history |
| [BUILD_INSTRUCTIONS.md](BUILD_INSTRUCTIONS.md) | How to build from source |
| [USAGE.md](USAGE.md) | Detailed user guide |
| [USER_MANUAL.md](USER_MANUAL.md) | Complete manual with screenshots |
| [FEATURES_CRYSTAL.md](FEATURES_CRYSTAL.md) | Complete list of 149 crystal descriptors |
| [FEATURES_MOLECULAR.md](FEATURES_MOLECULAR.md) | Complete list of 149 molecular descriptors |
| [FEATURES_COMPOSITION.md](FEATURES_COMPOSITION.md) | Complete list of 106 Magpie composition descriptors |

---

## Citation

cff-version: 1.2.0
message: "If you use this software, please cite it as below."
type: software
title: "AutoAIM Studio: Auto Artificial Intelligence for Materials"
authors:
  - family-names: "Vera De la Garza"
    given-names: "Cesar Gabriel"
    orcid: "https://orcid.org/0000-0003-2367-3534"
    affiliation: "National Autonomous University of Mexico. Materials Research Institute."
  - family-names: "Fomine"
    given-names: "Serguei"
    orcid: "https://orcid.org/0000-0002-7068-7579"
    affiliation: "National Autonomous University of Mexico. Materials Research Institute."
repository-code: "https://github.com/cesargabrielvera1-ux/AutoAIM-Studio"
url: "https://github.com/cesargabrielvera1-ux/AutoAIM-Studio"
version: 2.0.0
date-released: 2026-10-26
keywords:
  - machine learning
  - materials science
  - materials informatics
  - automated machine learning
  - neural networks
  - featurization
  - AutoML
  - no-code
  - desktop application
  - PyQt6
license: MIT
preferred-citation:
  type: article
  title: "AutoAIM Studio: A desktop application for automated machine learning in materials science"
  authors:
    - family-names: "Vera De la Garza"
      given-names: "Cesar Gabriel"
      orcid: "https://orcid.org/0000-0003-2367-3534"
    - family-names: "Fomine"
      given-names: "Serguei"
      orcid: "https://orcid.org/0000-0002-7068-7579"
  year: 2026
  journal: "SoftwareX"
  volume: 35
  pages: 102945
  doi: "10.1016/j.softx.2026.102945"
  url: "https://doi.org/10.1016/j.softx.2026.102945"

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- **pymatgen** — Crystal structure parsing and analysis ([pymatgen.org](https://pymatgen.org))
- **PyTorch** — Neural network backend ([pytorch.org](https://pytorch.org))
- **Optuna** — Bayesian hyperparameter optimization ([optuna.org](https://optuna.org))
- **scikit-learn** — Classical machine learning algorithms ([scikit-learn.org](https://scikit-learn.org))
- **SHAP** — Model explainability ([shap.readthedocs.io](https://shap.readthedocs.io))

---

<p align="center">
  <b>AutoAIM Studio v2.0.0</b> — Materials ML made simple.
</p>
