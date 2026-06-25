# AutoAIM-Studio: Auto Artificial Intelligence for Materials Studio v1.3.0

**AutoAIM Studio** is a standalone desktop application for automated machine learning in materials science and beyond. It provides a complete no-code AutoML pipeline with custom compositional feature engineering, crystal structure analysis, neural network design, hyperparameter optimization, and standalone model deployment.

> **Auto Aim to predictions in minutes, not days.**

---

## What's New in v1.3.0

- **Train with Custom Parameters** — Manually configure all hyperparameters for 7 regressors (RF, GB, XGBoost, LightGBM, CatBoost, SVR, Ridge) including XGB min_child_weight/gamma, SVR kernel selection, RF max_features. All configurable params are also included in Bayesian optimization
- **Manual Ensemble Weights** — Set custom weights for each model in a weighted-average ensemble instead of relying on Bayesian optimization
- **53 Educational Tooltips** — Hover over any parameter to see what it does, when to change it, and typical values. Covers all Training, Optimization, NN, and Ensemble parameters
- **Cross-Validation Fold Breakdown** — Model Details dialog now shows per-fold CV scores as a table for all model types (regressors, ensembles, NNs)
- **NN Scheduler + Min Delta + Grad Clip** — Full control over learning rate scheduling, early stopping sensitivity, and gradient clipping in the NN Training Config panel
- **Tab Reorder** — Logical workflow: Data → Training → Optimization → Ensemble → Neural Network → Results → Explainability → Predict
- **GPU Model Bundle Fix** — Models trained on GPU can now be saved and loaded correctly; CPU-only executables remain unaffected

## Previous Highlights

- **v1.2.0** — Crystal Structure Support (CIF, POSCAR, XYZ), SVR & KNN optimization fix, PyInstaller packaging fix
- **v1.1.0** — Initial release with AutoML training, neural networks, Bayesian optimization, ensembles, and model deployment

---

## Key Features

| Feature | Description |
|---------|-------------|
| **Crystal Structure Loading** | Import CIF, POSCAR, CONTCAR, XYZ, and VASP files for automatic featurization |
| **Crystallographic Featurization** | ~154 descriptors per structure: lattice, composition, Magpie (106), bond analysis, RDF |
| **Compositional Featurization** | Custom 106-descriptor Magpie generator from chemical formulas (no MatMiner dependency) |
| **AutoML Training** | Train multiple algorithms simultaneously with cross-validation |
| **Custom Parameter Training** | Manually configure any regressor's parameters and train without optimization |
| **Visual Neural Networks** | Design and train custom PyTorch architectures with a visual interface |
| **Bayesian Optimization** | Hyperparameter tuning with Optuna for regressors and neural networks |
| **Manual Ensemble Weights** | Set custom weights for each model in a weighted-average ensemble |
| **Model Bundles** | Self-contained model packages with manifest.json for reproducibility |
| **Standalone Prediction** | Deploy trained models for inference without the full application |
| **Explainability** | SHAP values, permutation importance, and partial dependence plots |
| **Ensemble Models** | Combine multiple models for improved predictions |
| **Neural Network Optimization** | Hyperparameter tuning with custom ranges and one-click architecture rebuild |
| **Ensemble Weight Optimization** | Bayesian optimization of ensemble member weights via Optuna |
| **Cross-Validation for All Models** | Configurable K-fold CV for regressors, neural networks, and ensembles |
| **Educational Tooltips** | 53 tooltips explaining every parameter across all configuration tabs |

---

## Crystal Structure Support (New in v1.2.0)

AutoAIM Studio now supports direct loading of crystallographic files:

### Supported Formats

| Format | Extension | Description |
|--------|-----------|-------------|
| CIF | `.cif` | Crystallographic Information File |
| POSCAR | `.poscar` | VASP position file |
| CONTCAR | `.contcar` | VASP continuation file |
| XYZ | `.xyz` | Cartesian coordinates |
| VASP | `.vasp` | Generic VASP format |

### Automatic Feature Extraction

When you load structure files, the application automatically extracts **~154 numeric features** per structure:

| Category | Count | Examples |
|----------|-------|----------|
| **Lattice** | ~10 | a, b, c, alpha, beta, gamma, volume, abc_ratio, symmetry flags |
| **Composition** | ~13 | avg_atomic_number, avg_electronegativity, avg_ionization_energy, radii, variance |
| **Structural** | ~3 | volume_per_atom, packing_fraction, spacegroup_number |
| **Magpie** | **106** | Weighted mean, std, min, max, median, p25, p75 for 13 elemental properties |
| **Advanced** | ~15 | Bond lengths, coordination numbers, RDF peaks, lattice anisotropy, complexity |
| **Derived** | ~2 | Density, total_electrons |

These features are pure numeric vectors that feed directly into the AutoAIM training pipeline — no additional preprocessing required.

### How to Use

1. **Load Structures:** Click "Load Directory" to import all structure files from a folder, or "Load Files" for individual files
2. **Attach Targets:** Provide target values (e.g., band gap, formation energy) via CSV or manual entry
3. **Train:** The featurized dataset feeds directly into any AutoAIM model (Random Forest, XGBoost, Neural Networks, etc.)

> **Note:** The manual featurizer ("Apply Featurizer" button) is designed for CSV files with a `formula` or `composition` column. Crystal structure datasets are already fully featurized upon loading.

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
git clone https://github.com/cesargabrielvera1-ux/AutoAIM-Studio
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
- CUDA-compatible GPU support only available when running from source code (if you have configured drivers, created a virtual environment and downloaded all requirements). If you want to try follow the instructions for Windows 10/11 if not just run the .exe:
-  1. Open Power Shell as Admin.
   2. Create a Virtual Environment (you must have python 3.11.8). Use the following command: python -m venv venv-gpu
   3. Activate the Virtual Environment. Use the following command: venv-gpu\Scripts\activate
   4. From this repo download requirements-gpu.txt and the source code and put it in your working directory.
   5. FIRST INSTALL THE FOLLOWING: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
   6. Then install the remaining requirements using the following command: pip install requirements-gpu.txt
   7. Run using python -m app.main
   8. NOTE: Following these instructions does not guarantees being able to run AutoAIM from source. Advanced users required. If experiencing problems contact us or just run the .exe
          

---

## Quick Start

### Tab 1: Data
- **Load CSV:** Import your dataset (must contain a target column)
- **Load Crystal Structures:** Import CIF/POSCAR/XYZ files (new in v1.2.0)
- **Apply Featurizer:** Extract Magpie features from chemical formulas (for CSV datasets)
- **Set Target:** Select your prediction target column
- **Split:** Configure train/test split and optional external validation

### Tab 2: Train
- Select algorithms (Random Forest, XGBoost, LightGBM, Neural Network, etc.)
- Configure cross-validation folds
- Click "Train All" and monitor real-time progress

### Tab 3: Optimization
- Select algorithm and configure Optuna trials
- Set custom parameter ranges (optional)
- Click "Optimize" and review results
- "Train with Best Parameters" for instant deployment

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

### Tab 7: Explainability
- SHAP values, permutation importance, partial dependence plots
- Model-specific explanations

### Tab 8: Deploy
- Export trained models as standalone Model Bundles
- Generate prediction scripts for deployment

---

## Screenshots

<p align="center">
 
</p>

## Loading Crystal Structures from Data Tab
<img width="1279" height="746" alt="Captura de pantalla 2026-06-14 090919" src="https://github.com/user-attachments/assets/b5f574df-4d9c-4278-8721-84f488690087" />
Load crystal structure supported formats by individually selecting them or loading the entire directory. Once the structures are Loaded the program will show a messasge requesting the user to choose the .csv with target and a column named 'structure_name' matching the structure files. If different than 'structure_name' column is available in the .csv the program will show an error. Rename the column as requested and try again. Structure names in the .csv MUST match the provided structures names.

## Data Tab Showing Crystal Structure Data After Successfull Data Load
<img width="1279" height="748" alt="Captura de pantalla 2026-06-14 091005" src="https://github.com/user-attachments/assets/3376b5ae-923f-472e-94e7-3896accbab94" />
The same interface you love remains untouched with same funcionality.

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

|Document|Description|
|-|-|
|[README.md](README.md)|This file — overview and quick start|
|[CHANGELOG](CHANGELOG.md)|Detailed changelog for v1.3.0|
|[USAGE.md](USAGE.md)|Detailed user guide|
|[USER\_MANUAL.md](USER_MANUAL.md)|Complete manual|

---

## Version History

| Version | Date | Highlights |
|---------|------|------------|
| v1.3.0 | Jun 2026 | Custom parameters, Manual Ensemble Weights, Educational Tooltips, CV Fold Breakdown, More NN Training Config, Tab Reorder |
| v1.2.0 | Jun 2026 | Crystal Structure Support, SVR/KNN fix, PyInstaller fix |
| v1.1.0 | May 2026 | Neural Network Optimization, CV for all models, Ensemble Weight Optimization |
| v1.0.0 | Mar 2026 | Initial release — AutoML pipeline, NN builder, ensembles, deployment 

See (RELEASE_NOTES.md) for detailed changelog.

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
  doi = {10.5281/zenodo.20837713}
}
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

- This work was supported by [C. G. V. de la G. received support from CONAHCyT under postdoctoral fellowship I1200/311/2023 “Estancias Posdoctorales por Mexico Convocatoria 2023(1) Estancia Posdoctoral Inicial/Academica”. We also acknowledge support from DGAPA-UNAM (PAPIIT IN200125).].
- Special thanks to beta testers and early adopters.
* **pymatgen** — Crystal structure parsing and analysis ([pymatgen.org](https://pymatgen.org))
* **PyTorch** — Neural network backend ([pytorch.org](https://pytorch.org))
* **Optuna** — Bayesian hyperparameter optimization ([optuna.org](https://optuna.org))
* **scikit-learn** — Classical machine learning algorithms ([scikit-learn.org](https://scikit-learn.org))
* **SHAP** — Model explainability ([shap.readthedocs.io](https://shap.readthedocs.io))


---

<p align="center">
  <strong>⭐ Star this repository if you find it useful! ⭐</strong>
</p>

