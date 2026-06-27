# AutoAIM-Studio: Auto Artificial Intelligence for Materials Studio v1.4.0

**AutoAIM Studio** is a standalone desktop application for automated machine learning in materials science and beyond. It provides a complete no-code AutoML pipeline with custom compositional feature engineering, crystal structure analysis, neural network design, hyperparameter optimization, manual parameter configuration, parameter selection for optimization, and standalone model deployment.

> **Auto Aim to predictions in minutes, not days.**

---

## What's New in v1.4.0

- **Select Parameters to Optimize** — Choose which hyperparameters Optuna optimizes. "Recommended" mode selects only high-impact parameters for faster CPU optimization. Available for regressors and neural networks
- **Random State Sensitivity Testing** — Changing the random seed in Custom Parameters now produces genuinely different results (train/test split and CV folds both respond to the seed)
- **High-Precision Parameter Input** — 10-12 decimal places for fine-grained sensitivity testing
- **LightGBM min_child_samples** — Complete parameter coverage: 9 tunable parameters, matching XGBoost
- **NN Tab Redesign** — Compact toolbar, limited-height panels, epoch-by-epoch Training Results table
- **Bug Fixes** — Ensemble deduplication, safe parent access, parameter sync between optimizer and registry, manual model deletion

## Previous Highlights

- **v1.3.0** — Train with Custom Parameters (7 regressors, all params), 53 tooltips, CV fold breakdown, NN scheduler/grad clip, tab reorder, GPU bundle fix
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

## Regressors Training Tab
<img width="1279" height="763" alt="Captura de pantalla 2026-06-25 115739" src="https://github.com/user-attachments/assets/1620af9c-d24b-4371-8fcb-e4b13b60b1da" />
Train one or all regressors with Default parameters or click "Train with Custom Parameters" button to manually configure them.

## Train with Custom Parameters Pop up Tab
<img width="1279" height="763" alt="Captura de pantalla 2026-06-26 200125" src="https://github.com/user-attachments/assets/c80650d9-8ed9-47f0-8f8c-fc9bfacac2f4" />
Manually select training parameters for every regressor you want to train. Configure CV Folds and Random State.

## Regressors Optimization Tab
<img width="1279" height="764" alt="Captura de pantalla 2026-06-26 200223" src="https://github.com/user-attachments/assets/2869ddf2-3b9c-45ae-a476-b90ee80a9743" />
Optimize any regressor. Configure number of trails, Random State, CV Folds and Parameter Optimization Ranges. You can also select which parameters you want to optimize.

## Parameter selection and optimization ranges
<img width="1279" height="763" alt="Captura de pantalla 2026-06-26 200517" src="https://github.com/user-attachments/assets/d5ff07cb-5605-46df-8130-e20b20ee7a32" />
Select all or just a few parameters to optimize if you want to cut optimization time.

<img width="1279" height="763" alt="Captura de pantalla 2026-06-26 200538" src="https://github.com/user-attachments/assets/5067e0e1-fa00-4da6-a11d-26a46de751e3" />
Edit optimization ranges. All parameters for all regressors are available.

## Rgressors Optimization Tab Optimization Details
<img width="1279" height="764" alt="Captura de pantalla 2026-06-26 200738" src="https://github.com/user-attachments/assets/b8a5d47c-08e3-40ea-bc8e-ea6d1aba43a0" />
Optimization details are shown on the right. TOptimization progress, Optimization Results and Best Trail Details.

## Ensemble Tab
<img width="1279" height="764" alt="Ensamble" src="https://github.com/user-attachments/assets/75f7a0f3-7b10-41b3-9e14-2c08993de563" />
Train custom Ensemble models. You can choose either Optimizer weights or try with custom weights.

## Neural Network Training and Optimization Tab
<img width="1279" height="766" alt="Captura de pantalla 2026-06-26 200819" src="https://github.com/user-attachments/assets/bcc13eca-d2f8-42ed-a552-10f482e89eda" />
Completely redesigned Neural Network Tab! Build a custom Neural Network and train it, or select optimization ranges and optimize! As for regressors, you can now choose which parameters you want to optimize. Apply best hyperparameters and retrain. You can see results in real time.

## Neural Network Custom Optimization Ranges and parameter selection for optimization
<img width="1279" height="763" alt="Captura de pantalla 2026-06-26 201021" src="https://github.com/user-attachments/assets/d74e74aa-5644-44ea-a302-a54f0744a364" />
Select parameters for optimization.

<img width="1279" height="762" alt="NN-2" src="https://github.com/user-attachments/assets/f5f6aff0-d631-44a9-a2c3-5d84c4b8524d" />
Set optimization ranges and Optimize!

## Neural Network Optimization and Training Progress
<img width="1279" height="763" alt="Captura de pantalla 2026-06-26 201331" src="https://github.com/user-attachments/assets/b562af5f-5a96-4426-887a-060ec385e324" />
Watch results in real time as NN Train or Optimize.

## Results Tab
<img width="1279" height="763" alt="Results-1" src="https://github.com/user-attachments/assets/58c2e759-cb3d-4392-a1ad-1a2a912680ef" />
See all results. Delete them or request details. You can export them as Model Bundles for further predictions, generate Learning Curves data and more!

## Results Tab Details
<img width="1279" height="763" alt="Results-2" src="https://github.com/user-attachments/assets/8526e616-9b5b-457b-ba6a-3fba8bfea4ba" />
When clicking Details button on any trained or optimized model AutoAIM shows Metrics, importance and CV results per fold.

## Explainability
<img width="1279" height="763" alt="SHAP" src="https://github.com/user-attachments/assets/aa87ea15-f2b4-49ed-ad4b-30d08e9c8bbb" />
Compute explainability analysis with just a few clicks.

## Predict.
<img width="1279" height="763" alt="Predict" src="https://github.com/user-attachments/assets/d98c5569-94b3-404c-ac98-93704ad322c4" />
Load a Model Bundle, load your prediction data and predict in seconds!

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
  doi = {10.5281/zenodo.20947495}
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

