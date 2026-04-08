# Changelog

All notable changes to AutoAIM Studio will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

#### [1.1.0] - Target: Q3 2026
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

#### [1.2.0] - Target: Q4 2026
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

#### [2.0.0] - Target: 2027
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

| Version | Date | Description |
|---------|------|-------------|
| 1.0.0 | 2026-04-08 | Initial stable release |

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

Download the latest release from the [Releases](https://github.com/yourusername/autoaim-studio/releases) page and replace the existing executable.

---

## Contributing to Changelog

When submitting pull requests, please:
1. Add your changes to the `[Unreleased]` section
2. Use the appropriate subsection (Added, Changed, Fixed, etc.)
3. Include the issue/PR number when applicable
4. Keep entries concise but descriptive

---

For detailed migration guides between versions, see the [Migration Guide](docs/MIGRATION.md).
