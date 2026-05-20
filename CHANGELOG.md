# Changelog

All notable changes to AutoAIM Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [1.1.0] - 2026-05-18

### Added
- Neural Network hyperparameter optimization with custom range editing for layers, units, dropout, activation, learning rate, batch size, and optimizer selection
- "Apply Best Parameters to Architecture" button for neural networks — one-click rebuild of the network with optimized hyperparameters
- "Train with Best Parameters" button for regressors — one-click training using the best model found during optimization
- Configurable K-fold cross-validation (1–10 folds) for neural network training and optimization
- Configurable K-fold cross-validation (1–10 folds) for regressor optimization
- Configurable K-fold cross-validation (1–10 folds) for ensemble training and weight optimization
- Ensemble weight optimization using Bayesian optimization (Optuna) to find optimal member contributions
- New ensemble training engine (`core/ensemble_trainer.py`) supporting weighted average and stacking strategies
- Dedicated Ensemble Results tab displaying CV scores, member weights, and per-member contributions
- Dedicated Neural Network Results tab showing NN-specific metrics, training epochs, loss history, and cross-validation results
- CV Score display for all model types in the redesigned Results tab (Training, Optimization, Neural Network, and Ensemble views)
- NNParameterRangeDialog for interactive editing of neural network optimization search ranges

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
  - Multiple metrics (R², RMSE, MAE, MAPE)
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

#### [1.2.0] — Target: Q3 2026
- **New Features**
  - Crystal structure featurization (CIF file support)
  - Additional elemental properties database
  - Custom descriptor builder
  - Batch prediction interface
  - Model versioning and management

- **Improvements**
  - Enhanced neural network architectures (CNN, Transformer)
  - Faster featurization with caching
  - Improved GUI responsiveness
  - Better error messages and validation

#### [1.3.0] — Target: Q4 2026
- **New Features**
  - Multi-target prediction support
  - Classification mode for categorical properties
  - Time-series analysis for degradation modeling
  - Integration with materials databases (Materials Project, AFLOW)

- **Improvements**
  - Distributed training support
  - GPU acceleration for neural networks
  - Advanced visualization options
  - Plugin system for custom algorithms

#### [2.0.0] — Target: 2027
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

| Version | Date       | Description                                                         |
|---------|------------|---------------------------------------------------------------------|
| 1.1.0   | 2026-05-18 | Neural network optimization, CV folds, ensemble weight optimization |
| 1.0.0   | 2026-04-08 | Initial stable release                                              |

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

For detailed migration guides between versions, see the [Migration Guide](docs/MIGRATION.md).


For detailed migration guides between versions, see the [Migration Guide](docs/MIGRATION.md).
