# AutoAIM-Studio: Auto Artificial Intelligence for Materials Studio V 1.1.0.

**AutoAIM Studio** is a standalone desktop application for automated machine learning in materials science and beyond!. It provides a complete no-code AutoML pipeline with custom compositional feature engineering, neural network design, hyperparameter optimization, and standalone model deployment.

> ** Auto Aim to predictions in minutes, not days.**

---

## What's New in v1.1.0

- **Neural Network Hyperparameter Optimization** — Custom search ranges for layers, units, dropout, activation, learning rate, and batch size, with one-click apply to architecture
- **Cross-Validation for All Models** — Configurable K-fold CV (1–10 folds) for neural networks, regressors, and ensemble models
- **Ensemble Weight Optimization** — Bayesian optimization of ensemble member weights via Optuna for improved predictive performance
- **One-Click "Train with Best Parameters"** — Train optimized regressors instantly after hyperparameter tuning completes
- **Full Backward Compatibility** — All v1.0.0 model bundles work without modification; no migration needed

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
| **Neural Network Optimization** | Hyperparameter tuning with custom ranges and one-click architecture rebuild |
| **Ensemble Weight Optimization** | Bayesian optimization of ensemble member weights via Optuna |
| **Cross-Validation for All Models** | Configurable K-fold CV for regressors, neural networks, and ensembles |

---

## Installation

### Windows (Standalone Executable — Recommended)

Requires: Windows 10 / 11 (Recommended)

1. Download the latest release from [GitHub Releases](https://github.com/cesargabrielvera1-ux/AutoAIM-Studio/releases)
2. Extract the ZIP file
3. Run `AutoAIM Studio.exe`

No Python installation required!

### Linux (Install from Source)

Requires: Ubuntu 20.04/22.04 (or compatible), Python 3.9+, 8 GB RAM minimum

```bash
# 1. Clone the repository
git clone https://github.com/cesargabrielvera1-ux/AutoAIM-Studio.git
cd AutoAIM-Studio

# 2. Run the automated installer
chmod +x scripts/install_linux.sh
./scripts/install_linux.sh

# 3. Launch the application
python -m app.main
```

**Requirements:**
- Python 3.9 or higher
- 8GB RAM minimum (16GB recommended)
- CUDA-compatible GPU support only available when running from source code (if you have configured drivers, created a virtual environment and downloaded all requirements. If you want to try follow the instructions for Windows 10/11 if not just run the .exe:
          1.- Open Power Shell as Admin.
          2.- Create a Virtual Environment (you must have python 3.11.8). Use the following command: python -m venv venv-gpu
          3.- Activate the Virtual Environment. Use the following command: venv-gpu\Scripts\activate
          4.- From this repo download requirements-gpu.txt and the source code and put it in your working directory.
          5.- FIRST INSTALL THE FOLLOWING: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
          6.- Then install the remaining requirements using the following command: pip install requirements-gpu.txt
          7.- Run using python -m app.main
  NOTE: Following these instructions does not guarantees being able to run AutoAIM from source. Advanced users required. If experiencing problems contact us or just run the .exe
          

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
 
</p>

## Neural Network Training and Optimization Tab
<img width="2559" height="1496" alt="Captura de pantalla 2026-05-20 113623" src="https://github.com/user-attachments/assets/ebbbfe77-7334-4ffc-ac60-3d91838ad91b" />
Build a custom Neural Network and train it, or select optimization ranges and optimize! Apply best hyperparameters and retrain.

## Regressors Optimization Tab
<img width="2556" height="1495" alt="Captura de pantalla 2026-05-20 113637" src="https://github.com/user-attachments/assets/00b5af58-d198-479a-9916-a17662871673" />
Select optimization ranges for each regressor and optimize. Apply best hyperparameters found and train!

## Ensemble Models Optimization Tab
<img width="2559" height="1492" alt="Captura de pantalla 2026-05-20 113741" src="https://github.com/user-attachments/assets/bb6c0ec1-9c9f-4f37-8837-aaaeed9147b3" />
Optimize weights of custom ensemble models.


## Model Training 
<img width="2559" height="1529" alt="Figure-1" src="https://github.com/user-attachments/assets/5ba246af-7c5f-4461-91cd-99902c748fd1" />
Multiple models training simultaneously

## Results Tab
<img width="2559" height="1526" alt="Figure-2" src="https://github.com/user-attachments/assets/d478eefb-3d7e-4597-989f-3abe60b0ff06" />
Mutiple results showing in real time

---

## Documentation

- [User Manual](USER_MANUAL_EN.pdf) - Complete guide to all features also available in spanish
- [Contributing Guide](CONTRIBUTING.md) - How to contribute to the project

---

## Architecture

```
AutoAIM Studio
├── Frontend: PyQt6 GUI
├── Backend:
│   ├── ML: scikit-learn, XGBoost, LightGBM, CatBoost
│   ├── NN: PyTorch with TorchScript export
│   ├── Ensemble: EnsembleTrainer with weight optimization (NEW in v1.1.0)
│   ├── Optimization: Optuna (regressors, neural networks, ensemble weights)
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
  url = {https://github.com/cesargabrielvera1-ux/AutoAIM-Studio},
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

