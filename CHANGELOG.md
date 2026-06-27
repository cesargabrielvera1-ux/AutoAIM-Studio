# Changelog

All notable changes to AutoAIM Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.4.0] - 2026-06-26

### Added
- **Select Parameters to Optimize** (v1.4.0) — choose which hyperparameters Optuna optimizes:
  - Dialog with checkboxes for every parameter of every model
  - "Select All" / "Select Recommended" / "Clear All" quick buttons
  - Recommended mode selects only high-impact parameters for faster CPU optimization
  - Educational tooltips on each parameter explaining its impact level
  - Available for both regressor optimization (Optimization Tab) and NN architecture optimization (NN Tab)
- **LightGBM min_child_samples** — added to manual configuration and Bayesian optimization (9 params total for LightGBM)
- **Random State actually works** — changing the random seed in Custom Parameters now produces different results:
  - Train/test split uses the seed from the dialog (was hardcoded to 42)
  - KFold CV uses the model's random_state (extracted via `get_params()`)
  - NN CV uses `TrainingConfig.random_seed`
- **High-precision parameter input** — Custom Parameters dialog now shows 10-12 decimal places (was 6-8) for fine sensitivity testing
- **NN Tab redesign** — compact layout: Actions + Presets merged into single toolbar, all 4 right panels have limited height, more horizontal space for configuration
- **Training Results populated** — the epoch-by-epoch table in NN Tab now shows Train Loss, Val Loss, Train Metric, Val Metric, and Time for each epoch

### Fixed
- **Ensemble results not appearing in Results Tab** — `get_ensemble_trainer()` (inexistent method) changed to direct attribute access
- **Ensemble results duplicated** — removed code that saved ensembles to both `ensemble_trainer.results` and `trainer.results`
- **Manual models not deletable** — delete now searches in `training_tab.training_results` as well as `trainer.results`
- **Optimizer/Registry parameter desync** — 7 parameters were in `model_registry` but not in `optimizer.py` (GB min_samples_leaf, LGBM reg_alpha/reg_lambda, SVR epsilon, Ridge alpha, SVR kernel missing 'linear')
- **Safe parent access** — `_safe_get_parent_attr()` helper replaces ~50 direct `self.parent.*` accesses across all GUI tabs, preventing AttributeError crashes
- **Thread error reporting** — all training/optimization threads now include full traceback in error messages
- **Logging unified** — replaced `print()` calls with `logger.info/debug/warning/error` across the GUI

---

## [1.3.0] - 2026-06-24

### Added
- **Train with Custom Parameters** -- manually configure any regressor's hyperparameters and train directly, bypassing Bayesian optimization:
  - Dialog with auto-generated parameter widgets for all 7 models, now matching the optimization parameter space:
    - **Random Forest**: n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features (float or sqrt/log2)
    - **Gradient Boosting**: n_estimators, max_depth, learning_rate, subsample, min_samples_split, min_samples_leaf
    - **XGBoost**: n_estimators, max_depth, learning_rate, subsample, colsample_bytree, reg_alpha, reg_lambda, min_child_weight, gamma
    - **LightGBM**: n_estimators, max_depth, learning_rate, num_leaves, subsample, colsample_bytree, reg_alpha, reg_lambda
    - **CatBoost**: iterations, depth, learning_rate, l2_leaf_reg (uses `random_seed` instead of `random_state`)
    - **SVR**: C, epsilon, gamma (scale/auto/numeric), kernel (linear/poly/rbf/sigmoid)
    - **Ridge**: alpha
  - Educational tooltips on every parameter explaining what it does and typical values
  - Dedicated Random State and n_jobs controls for reproducibility testing
  - Model-specific parameter filtering -- only parameters accepted by the selected model are shown
- **Full Optimization Parameter Coverage** -- all manually-configurable parameters are now also optimized by Bayesian optimization:
  - XGBoost: +min_child_weight, +gamma (9 params total)
  - Random Forest: +max_features (5 params total)
  - Gradient Boosting: +min_samples_split, +min_samples_leaf (6 params total)
  - CatBoost: +l2_leaf_reg (4 params total)
  - SVR: +kernel (categorical: rbf/linear/poly/sigmoid) (4 params total)
  - Neural Network: +gradient_clip_val (0.0-5.0) in architecture optimization
- **Manual Ensemble Weights** -- configure custom weights for each model in a weighted-average ensemble:
  - "Manual Weights" panel with per-model weight spinboxes
  - "Train with Manual Weights" button trains the ensemble using user-defined weights
  - Falls back to equal weights if no manual weights are set
- **NN Training Configuration Expansion** -- 3 new training parameters exposed in the NN Training Config panel:
  - **LR Scheduler** -- choose from `reduce_on_plateau`, `cosine_annealing`, `step_lr`, `exponential`, or `none`
  - **Min Delta** -- early stopping minimum improvement threshold (1e-6 to 0.1, default 1e-4)
  - **Gradient Clip** -- maximum gradient norm for clipping (0 to 10, default 1.0; 0 = disabled)
- **Educational Tooltips** -- 53 tooltips added across all 4 configuration tabs (Training, Optimization, NN, Ensemble); hover over any spinbox, combo, or checkbox to see what the parameter does
- **Cross-Validation Fold Breakdown** -- the Model Details dialog now shows a table of per-fold CV scores for all model types (regressors, ensembles, NNs) in both Training and Results tabs
- **Optimization Results Cleanup** -- trial results table reduced to clean 4-column layout (Rank, Score, Parameters, Trial#); CV fold scores moved to the "Best Trial Details" text panel in the same tabular format as Training Tab
- **NN Post-Optimization Status Messages** -- progress bar switches to animated indeterminate mode after the last Optuna trial while CV compilation is in progress; prevents the "frozen UI" appearance
- **Tab Reorder** -- tabs reorganized to logical workflow: Data → Training → Optimization → Ensemble → Neural Network → Results → Explainability → Predict

### Fixed
- **SVR Gamma Crash** -- `ValueError: could not convert string to float: 'scale'` when selecting SVR in Custom Parameters dialog. Gamma is now a combo box accepting `'scale'`, `'auto'`, or a numeric value.
- **Gradient Boosting min_samples_leaf** -- `Got 3.0 instead of int` error because QDoubleSpinBox always returns float. Now preserves original int type when the default value is an integer.
- **CatBoost Random Seed Conflict** -- `only one of random_seed, random_state should be initialized`. Custom Parameters now detects CatBoost and uses `random_seed` instead of `random_state`.
- **Duplicate Random State** -- `random_state` appeared twice in Custom Parameters dialog (once from model defaults, once in Common Parameters). Now excluded from model-specific list.
- **NN Tab Overflow** -- action buttons (Auto-Optimize, Apply Best, Train NN) moved to compact 3-column grid; Training Progress log reduced from 150px to 100px
- **GPU Model Bundle Save** -- TorchScript conversion now moves model to CPU before tracing; fallback saves full nn.Module if TorchScript fails. Inference engine auto-detects format (TorchScript vs full model) on load.
- **Optimized Model Overwrite** -- sequential optimized models (e.g., XGBoost_Optimized_2) and sequential NNs (Neural Network_2) now preserve all results instead of overwriting; `_get_unique_model_name()` used consistently for all model types
- **TrainBestThread Overwrite Bug** -- removed code that deliberately deleted previous results to avoid `_1` suffix; now uses proper sequential naming

### Changed
- `create_estimator()` in `model_registry.py` now filters out parameters not accepted by the estimator class using `inspect.signature()`, preventing TypeError for unsupported parameters (e.g., `random_state` on SVR)
- `train_neural_network()` in `trainer.py` now uses `_get_unique_model_name()` for consistent sequential naming (same as `train()` for regressors)
- `results_tab.py` deduplicates results by name before rendering the comparison table
- `ensemble_trainer.py` `train_ensemble()` accepts optional `weights` parameter for manual weight configuration

---

## [1.2.0] - 2026-06-14

### Added
- **Crystal Structure Support** -- load and featurize CIF, POSCAR, CONTCAR, VASP, and XYZ files
  - `CrystalStructureLoader` -- parses structure files via pymatgen (no MatMiner dependency)
  - `CrystalStructureFeaturizer` -- extracts ~154 numeric descriptors per structure:
    - Lattice features (10): a, b, c, angles, volume, symmetry flags
    - Composition features (13): avg atomic number, electronegativity, radii, ionization energy
    - Structural features (3): volume per atom, packing fraction, space group number
    - Magpie descriptors (104): mean/std/min/max/median/p25/p75/range for 13 elemental properties
    - Advanced structural (15): bond lengths, coordination numbers, RDF peaks, lattice anisotropy
    - Derived features (2): density, total electrons
  - `CrystalStructureDatasetBuilder` -- builds complete ML-ready datasets from directories or file lists
  - Target CSV merge with `structure_name` as join key
- **Crystal Structure Prediction** in Inference tab -- load CIF/POSCAR/XYZ files and predict without external CSV metadata; structures are automatically featurized using the same pipeline as training
- **Target CSV Column Selection** dialog -- when loading a target CSV, the user selects which numeric column is the property to predict; all other columns (composition, file paths, etc.) are automatically discarded
- **Smart Featurizer Messages** -- clicking "Apply Featurizer" on a crystal-structure dataset now shows an informative dialog explaining that ~154 features are already present, instead of the confusing "No composition columns detected"
- **ID-Bug Detection** -- Inference tab detects models trained with the old one-hot encoding bug (400+ features including `structure_name_*` columns) and warns the user to retrain
- **Export with Structure Names** -- prediction exports preserve `structure_name` as the first column for crystal-structure datasets
- **PyInstaller Build Fix** -- reliable bundling of pymatgen data files via `--hidden-import pymatgen` + post-build `xcopy` + `setup_pymatgen()` runtime hook that auto-configures `PMG_HOME`
- **Documentation** -- `BUILD_v1.2.0_FIXES.md`, `RELEASE_NOTES_v1.2.0.md`, `RELEASE_CHECKLIST_v1.2.0.md`, `.zenodo.json`

### Changed
- `DataManager.set_target_column()` now selects **only numeric columns** as features; string columns (`structure_name`, composition strings, file paths) are automatically excluded and logged
- `build_windows_onedir.bat` uses `--hidden-import pymatgen` + post-build `xcopy` instead of `--collect-all pymatgen` (avoids namespace-package crash)
- `crystal_structure.py._build_dataset()` keeps `structure_name` as a regular column (not index) to enable CSV merge; caller is responsible for index management
- Results tab `refresh()` no longer displays intermediate `OptimizationResult` or `nn_optimizer_results` entries; only final `TrainingResult` entries are shown (avoids duplicate rows after "Train with Best Parameters")
- `inference_tab.py._update_data_preview()` and `explainability_tab.py._update_importance_table()` use `enumerate()` for row indexing instead of the DataFrame index (fixes `setItem` crash when index contains strings)
- `optimizer.py` conditionally applies `random_state` based on algorithm type (excluded for SVR/SVC and KNeighbors)

### Fixed
- **SVR & KNN Optimization Crash** -- `TypeError: unexpected keyword argument 'random_state'` during hyperparameter optimization
- **pymatgen Data Missing in PyInstaller** -- `FileNotFoundError: periodic_table.json.gz` when loading crystal structures in the standalone executable
- **PyInstaller Build Crash** -- `TypeError: expected str, bytes or os.PathLike object, not NoneType` caused by `--collect-all pymatgen` on namespace package (`__file__` = `None`)
- **Feature Explosion (771+ features)** -- string columns from metadata CSV (composition, file paths) were included as features and one-hot encoded, exploding feature count from ~154 to 600+; now only numeric columns are used as features
- **Missing Features on Prediction** -- `formation_energy_per_atom` and `num_sites` from metadata CSV leaked into training features but were unavailable during prediction; target CSV dialog now isolates only the target column
- **String Index Crash** -- `setItem(self, row: int, ...): argument 1 has unexpected type 'str'` when displaying crystal structure data with `structure_name` as DataFrame index
- **Duplicate Results** -- "xgboost (Optimized)" and "xgboost_Optimized" both appeared in Results tab after optimize + train; intermediate optimization results are now hidden from Results tab

---

## [1.1.0] - 2026-05-18

### Added
- Neural Network hyperparameter optimization with custom range editing for layers, units, dropout, activation, learning rate, batch size, and optimizer selection
- "Apply Best Parameters to Architecture" button for neural networks -- one-click rebuild of the network with optimized hyperparameters
- "Train with Best Parameters" button for regressors -- one-click training using the best model found during optimization
- Configurable K-fold cross-validation (1-10 folds) for neural network training and optimization
- Configurable K-fold cross-validation (1-10 folds) for regressor optimization
- Configurable K-fold cross-validation (1-10 folds) for ensemble training and weight optimization
- Ensemble weight optimization using Bayesian optimization (Optuna) to find optimal member contributions
- New ensemble training engine (`core/ensemble_trainer.py`) supporting weighted average and stacking strategies
- Dedicated Ensemble Results tab displaying CV scores, member weights, and per-member contributions
- Dedicated Neural Network Results tab showing NN-specific metrics, training epochs, loss history, and cross-validation results
- CV Score display for all model types in the redesigned Results tab (Training, Optimization, Neural Network, and Ensemble views)
- NNParameterRangeDialog for interactive editing of neural network optimization search ranges.

### Changed
- Neural network training now performs full K-fold cross-validation when CV folds > 1, with fresh model instances per fold and early stopping on each fold
- Regressor optimization passes user-configured CV folds to all Optuna trials; per-trial CV scores are tracked and stored
- Ensemble training respects user-selected model subsets and skips nested ensemble models to prevent prediction errors
- Data manager now trusts pandas dtype detection for numeric columns; binary 0/1 columns are preserved as numeric
- XGBoost `tree_method` changed to `'auto'` for compatibility with both CPU-only and GPU-enabled installations
- All training and optimization tabs now properly use external validation data when loaded
- SVM and KNeighbors optimization no longer passes `random_state`, preventing crashes with unsupported estimators
- `batch_size` sampled as categorical during NN optimization is now cast to `int` before use
- Inference engine sanitizes "Unnamed" artifact columns from manifest feature names and prediction input DataFrames
- `index_col=False` used on all CSV loads to prevent trailing-comma artifacts

### Fixed
- CV Score not appearing for Neural Networks in the Results tab
- CV Score not appearing for Optimized Models in the Results tab
- CV Score not appearing for Ensemble models in the Results tab
- Escaped `\n` characters appearing literally in console output
- `AttributeError: 'DataManager' object has no attribute '_problem_type'`
- `AttributeError: 'EnsembleTab' object has no attribute '_on_training_error'`
- `AttributeError: 'NNBuilderTab' object has no attribute 'nn_optimizer_results'`
- `AttributeError: 'dict' object has no attribute 'model_names'` (ensemble `optimize_weights` return type)
- Ensemble using all models instead of the user-selected subset
- Ensemble attempting to use other ensembles as base models
- Incorrect CV score calculation during optimization (was averaging across all trials; now stores best-trial fold scores)
- `name 'X_test' is not defined` error during neural network optimization
- NN Optimized model not found for learning curves and model export
- NN Optimized model not appearing in the Explainability tab
- Ensemble Explainability crash due to missing predict method on ensemble models
- Binary 0/1 columns incorrectly classified as categorical
- Numeric columns with repeated values incorrectly classified as categorical
- External validation data being ignored by training and optimization tabs
- SVM optimization crash caused by passing `random_state` to SVR/SVC
- Neural network optimization crash caused by `batch_size` categorical comparison between `int` and `str`
- XGBoost GPU incompatibility with CPU-only installations
- Prediction failure due to "Unnamed: xxx" CSV artifact columns

---

## [1.0.0] - 2026-04-08

### Added
- Initial stable release of AutoAIM Studio
- Complete desktop AutoML application for materials science
- PyQt6-based graphical user interface with modern design
- Compositional featurization engine with 106 custom descriptors
  - Elemental properties (40 descriptors)
  - Statistical moments (24 descriptors)
  - Composition ratios (12 descriptors)
  - Electronic structure (18 descriptors)
  - Thermodynamic estimates (12 descriptors)
- Support for multiple machine learning algorithms:
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM, CatBoost)
  - Support Vector Machines (SVM)
  - k-Nearest Neighbors (k-NN)
- Neural network training with PyTorch
  - Configurable architectures
  - Automatic hyperparameter tuning
  - TorchScript export for production
- Bayesian hyperparameter optimization with Optuna
- SHAP-based model explainability
  - Global feature importance
  - Individual prediction explanations
  - Dependence plots
- Model Bundle system for deployment
  - Self-contained model packages
  - Standalone inference capability
  - JSON manifest with complete metadata
- Data loading and preprocessing
  - CSV and Excel file support
  - Automatic column type detection
  - Missing value handling
  - Outlier detection
- Cross-validation and model evaluation
  - k-fold cross-validation
  - Hold-out test set evaluation
  - Multiple metrics (R2, RMSE, MAE, MAPE)
- Visualization capabilities
  - Training progress plots
  - Feature importance charts
  - SHAP summary plots
  - Actual vs. predicted scatter plots
- Model comparison and selection
- Export functionality
  - Model bundles (.zip)
  - Trained models (.joblib, .pt)
  - Results and visualizations (.png, .csv)
- Comprehensive test suite with pytest
- Documentation and examples
- MIT License

### Technical Details
- Python 3.9+ compatibility
- Cross-platform support (Windows, Linux, macOS)
- PyInstaller configuration for executable generation
- Modern packaging with pyproject.toml
- Citation file (CITATION.cff) for academic use
- Contributing guidelines

---

## [Unreleased]

### Planned for Future Releases

#### [1.3.0] -- Target: Q4 2026
- **New Features**
  - Extended crystallographic featurization (Voronoi analysis, site statistics, graph topology, Coulomb/Sine matrix)
  - Configurable featurization levels (Basic / Standard / Advanced / Extended) for crystal structures
  - Configurable featurization levels (Basic / Full / Extended) for formula-based datasets
  - Multi-target prediction support
  - Classification mode for categorical properties
  - Plugin system for custom algorithms

- **Improvements**
  - Faster featurization with caching
  - Improved GUI responsiveness
  - Better error messages and validation
  - GPU acceleration for neural networks

#### [2.0.0] -- Target: 2027
- **New Features**
  - Web-based interface option
  - REST API for integration
  - Collaborative features
  - Advanced uncertainty quantification
  - Active learning support

- **Breaking Changes**
  - Potential API changes for model bundles
  - Updated minimum Python version

---

## Release Notes Format

Each release section includes:

- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements

---

## Version History

| Version | Date       | Description                                                          |
|---------|------------|----------------------------------------------------------------------|
| 1.3.0   | 2026-06-19 | Crystal structures, configurable NN opt, random seed, UI redesign    |
| 1.2.0   | 2026-06-14 | Crystal structure support, PyInstaller fix, duplicate results fix    |
| 1.1.0   | 2026-05-18 | Neural network optimization, CV folds, ensemble weight optimization  |
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

---


