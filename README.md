# AutoAIM-Studio: Auto Artificial Intelligence for Materials Studio V 1.0.0.

**AutoAIM Studio** is a standalone desktop application for automated machine learning in materials science and beyond!. It provides a complete no-code AutoML pipeline with custom compositional feature engineering, neural network design, hyperparameter optimization, and standalone model deployment.

> ** Auto Aim to predictions in minutes, not days.**

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Compositional Featurization** | Custom 106-descriptor generator from chemical formulas (no MatMiner dependency) |
| **AutoML Training** | Train multiple algorithms simultaneously with cross-validation |
| **Visual Neural Networks** | Design and train custom PyTorch architectures with a visual interface |
| **Bayesian Optimization** | Hyperparameter tuning with Optuna for best model performance |
| **Model Bundles** | Self-contained model packages with manifest.json for reproducibility |
| **Standalone Prediction** | Deploy trained models for inference without the full application |
| **Explainability** | SHAP values, permutation importance, and partial dependence plots |
| **Ensemble Models** | Combine multiple models for improved predictions |

---

## Installation (Right now AutoAIM is only available for Windows)

### Option A: Use Pre-built Executable (Recommended)

1. Download the latest release from (https://github.com/cesargabrielvera1-ux/AutoAIM-Studio.git))
2. Extract the ZIP file
3. Run `AutoAIM Studio.exe`

No Python installation required!

### Option B: Install from Source

```bash
# Clone the repository
git clone https://github.com/cesargabrielvera1-ux/AutoAIM-Studio.git
cd autoaim-studio

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
python -m app.main
```

**Requirements:**
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU support in progress.

---

## Quick Start

### 1. Load Your Data

```
Data Tab → Load Data → Select your CSV/Excel file
```

### 2. Select Target & Preprocess

```
Select target column → Check preprocessing options → Prepare Data
```

### 3. Train Models

```
Training Tab → Select algorithms → Train Selected Models
```

### 4. Make Predictions

```
Predict Tab → Load Model Bundle → Load Prediction Data → Make Predictions
```

---

## Screenshots

<p align="center">
  <em>Screenshots will be added here upon release</em>
</p>

| Data Loading | Model Training | Neural Network Design |
|:------------:|:--------------:|:---------------------:|
| *Placeholder: CSV import with automatic column detection* | *Placeholder: Multiple models training simultaneously* | *Placeholder: Visual layer configuration* |

| Hyperparameter Optimization | Model Explainability | Standalone Prediction |
|:---------------------------:|:--------------------:|:---------------------:|
| *Placeholder: Bayesian optimization progress* | *Placeholder: SHAP value visualization* | *Placeholder: Inference on new data* |

---

## Documentation

- [User Manual](USER_MANUAL.md) - Complete guide to all features
- [API Documentation](docs/API.md) - For developers extending the software
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project

---

## Architecture

```
AutoAIM Studio
├── Frontend: PyQt6 GUI
├── Backend:
│   ├── ML: scikit-learn, XGBoost, LightGBM, CatBoost
│   ├── NN: PyTorch with TorchScript export
│   ├── Optimization: Optuna
│   └── Explainability: SHAP
└── Model Format: Custom Bundle (manifest.json + model files)
```

---

## Citation

If you use AutoAIM Studio in your research, please cite:

```bibtex
@software{autoaim_studio_2026,
  authors = {Cesar Gabriel Vera de la Garza; Serguei Fomine},
  title = {AutoAIM-Studio: Auto Artificial Intelligence for Materials Studio},
  year = {2026},
  url = {https://github.com/cesargabrielvera1-ux/AutoAIM-Studio.git},
  doi = {10.5281/zenodo.19478357}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This work was supported by [C. G. V. de la G. received support from CONAHCyT under postdoctoral fellowship I1200/311/2023 “Estancias Posdoctorales por Mexico Convocatoria 2023(1) Estancia Posdoctoral Inicial/Academica”. We also acknowledge support from DGAPA-UNAM (PAPIIT IN200125).].
- Thanks to the open-source community for scikit-learn, PyTorch, Optuna, and SHAP.
- Special thanks to beta testers and early adopters.

---

<p align="center">
  <strong>⭐ Star this repository if you find it useful! ⭐</strong>
</p>

