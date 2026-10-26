# AutoAIM Studio v2.1.0 - Usage Guide

**Last Updated**: September 11, 2026  
**Tested With**: 1024+ automated tests (0 failures, 75 skipped)  
**Compatibility**: Python 3.10+, Windows 10/11, Linux

## Table of Contents
1. [Getting Started](#getting-started)
2. [Data Loading](#data-loading)
3. [Linking Crystal Structures](#linking-crystal-structures)
4. [Linking Molecular Structures](#linking-molecular-structures)
5. [Data Health Check](#data-health-check)
6. [Training Models](#training-models)
7. [Building Neural Networks](#building-neural-networks)
8. [Hyperparameter Optimization](#hyperparameter-optimization)
9. [Model Explainability](#model-explainability)
10. [Confidence Intervals](#confidence-intervals)
11. [Exporting Results](#exporting-results)

## Getting Started

### Launching the Application

```bash
# From the project directory
python run.py

# Or using the module
python -m app.main
```

### Main Interface

The application consists of eight main tabs:
1. **Data** - Load and preprocess data, link crystal structures, health check
2. **Training** - Train traditional ML models
3. **Optimization** - Hyperparameter tuning with Optuna
4. **Neural Network** - Build and train custom neural networks
5. **Ensemble** - Combine multiple models with weight optimization
6. **Results** - Compare models, view metrics, export session
7. **Explainability** - SHAP values and model interpretation
8. **Predict** - Make predictions with confidence intervals (v2.0)

## Data Loading

### Supported Formats
- CSV files (.csv)
- Excel files (.xlsx, .xls)
- JSON files (.json)
- Parquet files (.parquet)

### Steps
1. Click "Load Data" in the Data tab
2. Select your data file
3. The application will automatically:
   - Detect column types (numeric, categorical, composition)
   - Show data preview
   - Display column statistics

### Selecting Target Variable
1. From the "Target Column" dropdown, select your prediction target
2. The application will automatically detect if it's a regression or classification problem

### Preprocessing Options
- **Imputation Strategy**: mean, median, most_frequent, knn
- **Scaling Strategy**: standard, minmax, robust, none
- **Encoding Strategy**: onehot, label, none

### Feature Engineering
If your dataset contains chemical formulas (e.g., "Fe2O3", "SiO2"):
1. Check "Generate composition features (Magpie)"
2. Click "Apply Feature Engineering"
3. The application will generate Magpie-style features automatically

## Linking Crystal Structures (v2.0)

Connect CIF structure files to your tabular dataset for combined tabular + structural machine learning.

### Requirements
- A CSV file with an **ID column** (unique values matching CIF filenames)
- CIF files named `{ID_value}.cif` in a folder
- pymatgen installed (`pip install pymatgen`)

### Steps
1. Load your CSV in the Data tab
2. Click **"Link Crystal Structures"**
3. Select the ID column from the candidate list
4. Choose the folder containing CIF files
5. Review the match report (matched/missing/duplicate IDs)
6. Click **Yes** to proceed

### Crystal Features Extracted
- `crystal_density` — Density in g/cm^3
- `crystal_volume` — Unit cell volume
- `crystal_spacegroup_num` — Space group number
- `crystal_a`, `crystal_b`, `crystal_c` — Lattice parameters
- `crystal_alpha`, `crystal_beta`, `crystal_gamma` — Lattice angles
- `crystal_n_sites` — Number of atomic sites
- `crystal_vol_per_atom` — Volume per atom

### Important Notes
- The ID column is automatically excluded from training
- If composition features (`comp_*`) already exist, you will be asked to replace them
- The Magpie featurizer is disabled when crystal structures are linked
- Missing CIFs result in NaN values (filled by imputation)
- Duplicate IDs (polymorphs) trigger a warning

## Data Health Check (v2.0)

Analyze dataset quality before training.

### How to Run
1. Load data in the Data tab
2. Click **"Data Health Check"**
3. Review the report dialog

### Checks Performed
- **Missing values**: Critical (>50%), High (20-50%), Low (5-20%)
- **Duplicate rows**: Detects exact duplicates
- **Constant columns**: Columns with single value
- **Outliers**: IQR-based detection
- **Scaling mismatch**: Features on very different scales
- **Correlations**: Features correlated >90%
- **Sample size**: Adequacy of data volume
- **Class imbalance**: For classification problems
- **Target quality**: Missing/constant target values

### Interpreting Results
- **Score 90-100**: Healthy data — proceed with training
- **Score 70-89**: Acceptable with warnings — review recommendations
- **Score 50-69**: Problems detected — fix before training
- **Score 0-49**: Critical issues — data needs significant work

## Training Models

### Available Algorithms
- Random Forest
- Gradient Boosting
- Histogram Gradient Boosting
- XGBoost (if installed)
- LightGBM (if installed)
- CatBoost (if installed)
- Support Vector Machine
- k-Nearest Neighbors
- Ridge Regression
- **Gaussian Process** (v2.1 — full kernel family tuning + native uncertainty quantification)

### Steps
1. Go to the "Training" tab
2. Select problem type (regression/classification)
3. Select models to train (or click "Select All")
4. Configure training options:
   - CV Folds: Number of cross-validation folds (2-10)
   - Test Size: Fraction for test set (0.1-0.5)
   - Random State: For reproducibility
5. Click "Train Selected Models"

### Interpreting Results
- **RMSE/Acc**: Root Mean Squared Error (regression) or Accuracy (classification)
- **MAE**: Mean Absolute Error
- **R²/F1**: Coefficient of determination or F1 score
- **CV Score**: Cross-validation score with standard deviation

## Building Neural Networks

### Architecture Configuration
1. Go to the "Neural Network" tab
2. Set number of hidden layers (1-5)
3. For each layer, configure:
   - **Units**: Number of neurons (16-2048)
   - **Activation**: relu, leaky_relu, tanh, sigmoid, gelu, elu
   - **Dropout**: Regularization rate (0.0-0.5)
   - **Batch Normalization**: Enable/disable

### Training Configuration
- **Epochs**: Number of training iterations (10-10000)
- **Batch Size**: Samples per batch (8-512)
- **Learning Rate**: Step size for optimization (1e-5 to 1e-1)
- **Optimizer**: adam, adamw, sgd, rmsprop
- **Weight Decay**: L2 regularization
- **Early Stopping Patience**: Stop if no improvement

### Training
1. Click "Update Architecture Preview" to see network structure
2. Click "Train Neural Network"
3. Monitor training progress in real-time

## Hyperparameter Optimization

### Search Strategy Selection

AutoAIM offers two search strategies in the Optimization Tab:

**Bayesian (Optuna)** — Default, recommended for most cases:
- Smart exploration that tests the most promising regions first
- Needs fewer trials (50–100)
- Supports pruning (early stopping of bad trials)
- Best for 3+ parameters or wide ranges

**Grid Search** — Deterministic, reproducible:
- Evaluates every combination exhaustively
- Choose resolution: Coarse (3 pts), Medium (5), Fine (7) per parameter
- Best for ≤3 parameters with small discrete ranges
- Useful as a baseline comparison
- AutoAIM estimates combinations upfront and blocks runs >5,000

### Cross-Validation Strategy Selection

AutoAIM offers **five** CV strategies (sklearn regressors), in both the Training and Optimization tabs and the Manual Parameter dialog:

**K-Fold** (default): Splits data into K parts. Recommended for N > 30. (Classification folds are stratified by class automatically.)

**Stratified K-Fold (quantiles)**: The target is binned into quantiles so every fold preserves the target distribution. For skewed regression targets (e.g. Tc). A **Quantile Bins** spinner (2–20, default 5) appears only for this strategy; the count is a request that may be auto-reduced (with an info note) if the data can't support it. Optionally enable **"Reserve a bin for an exact value"** to isolate a degenerate "property absent" marker (e.g. Tc = 0 K, Eg = 0 eV) into its own stratum — AutoAIM auto-detects a dominant repeated value and suggests it, and manual use is validated with real counts (blocked if too few samples to stratify). Regression only.

**Group K-Fold**: A user-defined group (chemical system, exact formula, clustering, or any dataset column) never crosses the train/test split — prevents chemical/batch leakage. Uneven folds show a Test Size column.

**Leave-One-Out (LOO)**: Each sample is left out once. Recommended for small datasets (N ≤ 30). Blocked for N > 100; warning for N > 30.

**Leave-One-Group-Out (LOGO)**: Each group is left out once (folds = number of groups). Needs ≥ 3 groups; blocked for > 100 groups. Uses out-of-fold global R² + per-group MAE.

See **Choosing a CV Strategy** in the User Manual for the full decision guide.

### Bayesian Optimization with Optuna
1. Go to the "Optimization" tab
2. Select **"Bayesian (Optuna)"** as search strategy
3. Select a CV strategy (K-Fold, Stratified, Group K-Fold, LOO, or LOGO)
4. Select model to optimize
5. Configure settings:
   - Number of Trials: Optimization iterations (10–200)
   - Timeout: Maximum time (0 = no limit)
   - Enable Pruning: Stop unpromising trials early
6. Click "Start Optimization"

### Grid Search
1. Go to the "Optimization" tab
2. Select **"Grid Search"** as search strategy
3. Select a CV strategy (K-Fold, Stratified, Group K-Fold, LOO, or LOGO)
4. Select model to optimize
5. Choose **Grid Resolution**: Coarse / Medium / Fine
6. Review the estimated combination count shown below
7. Configure Random Seed
8. Click "Start Optimization"

**Note on scope**: The five CV strategies (K-Fold, Stratified, Group K-Fold, LOO, LOGO) apply consistently to **sklearn regressors, Neural Network training, and Ensemble training + weight optimization**. Separately, the *search method* differs: sklearn regressors offer Bayesian and Grid Search, while Neural Network architecture search and Ensemble weight optimization use Bayesian optimization only (their search spaces are non-rectangular / continuous).

### Gaussian Process Optimization (v2.1)

Gaussian Process supports **full kernel family tuning** via Bayesian optimization:

**Tuneable Parameters:**
- `kernel_type` — RBF, Matern, RationalQuadratic, WhiteKernel, RBF+WhiteKernel
- `kernel_length_scale` — Characteristic length scale (1e-3 to 100.0)
- `kernel_nu` — Matern smoothness (0.5, 1.5, 2.5)
- `kernel_alpha_rq` — Rational Quadratic tail heaviness
- `kernel_noise_level` — WhiteKernel noise magnitude
- `alpha` — Tikhonov regularization (noise)
- `n_restarts_optimizer` — L-BFGS-B restarts
- `normalize_y` — Target normalization

**Recommendation**: Use **Bayesian optimization** for kernel tuning. Grid Search supports scalar GP params (`alpha`, `n_restarts_optimizer`, `normalize_y`) but automatically excludes kernel parameters with a warning.

### Ensemble Creation
1. Check "Create Ensemble from Top Models"
2. Set number of models in ensemble
3. Choose ensemble type: voting, averaging, or stacking

### Interpreting Results
- **Best Score**: Highest cross-validation score achieved
- **Best Parameters**: Optimal hyperparameter values
- **Trial History**: All optimization attempts (Bayesian) or all grid points (Grid Search)
- **Search Strategy**: Recorded in session export for reproducibility

## Model Explainability

### Feature Importance
1. Go to the "Explainability" tab
2. Select model to analyze
3. Choose analysis options:
   - SHAP Values
   - Permutation Importance
   - Partial Dependence Plots
   - Domain Applicability
4. Click "Run Explainability Analysis"

### SHAP Values
- Global importance: Mean absolute SHAP value per feature
- Local explanations: Feature contributions for individual predictions

### Partial Dependence Plots
Shows how predictions change as a feature varies

### Domain Applicability
- Identifies when predictions are outside training domain
- Provides reliability scores
- Uses leverage, Mahalanobis distance, and k-NN metrics

### Explaining Individual Predictions
1. Go to "Prediction Explanation" tab
2. Enter instance index
3. Click "Explain Prediction"
4. View feature contributions

### Exporting Explainability Results
Export all explainability analysis results to a formatted text file.

**Steps:**
1. After running explainability analysis, click **"Export Results as Text"**
2. Choose save location (default: `explainability_{model_name}_{timestamp}.txt`)
3. The file includes:
   - Feature Importance table (all columns: Feature, Model Importance, Permutation, SHAP, Average Rank)
   - SHAP Values summary
   - Partial Dependence analysis (or NN unavailable message)
   - Domain Applicability metrics (training samples, PCA, thresholds)
   - Prediction Explanation (if generated)
   - Complete Analysis Log with diagnostics

**Tip:** Export after each analysis to keep a permanent record of model interpretation results.

## Confidence Intervals (v2.0)

For regression predictions, AutoAIM can compute confidence intervals based on cross-validation residuals.

### How to Use
1. Go to **"Predict"** tab
2. Load a trained model and input data
3. Check **"Include Confidence Intervals"** checkbox
4. Run prediction
5. Results include: `prediction`, `lower_bound`, `upper_bound`

### Interpretation
- **68% interval** (1 std): ~68% of future observations will fall within this range
- **95% interval** (2 std): ~95% of future observations will fall within this range
- Wider intervals = more uncertainty in the model's prediction for that input

### Requirements
- Only available for **regression** models
- Model must have been trained with cross-validation (to estimate residual distribution)
- Works for all model types: sklearn, neural networks, ensembles

## Exporting Results (v2.0 → v2.1)

### Session Exporter
Export your complete session with all results, configurations, and metadata.

**Available Formats:**
| Format | Extension | Best For |
|--------|-----------|----------|
| **Text** | .txt | Human-readable, complete detail |
| **Markdown** | .md | Documentation, GitHub, reports |
| **CSV** | .csv | Spreadsheet analysis, plotting |
| **JSON** | .json | Machine processing, reproducibility |

**Steps:**
1. Go to **"Results"** tab
2. Click **"Export Session"**
3. Choose format and save location

**What's Included:**
- All trained models with full parameters (v2.1: cleaned — no Pipeline noise, no `model__` prefix)
- Cross-validation per-fold scores, with the full strategy label (e.g. "Group K-Fold (by chemical system)"), a Test Size column for grouped folds, group counts, and any CV diagnostics (bin reductions, OOF fallbacks)
- Optimization trials and configuration (trials, CV strategy, CV folds, random state, selected params)
- Neural network training history (epoch-by-epoch)
- Ensemble results and weight configurations
- Learning curves data (tabular, for external plotting)
- Confidence interval parameters
- Crystal structure linking metadata (if used)
- **v2.1: Gaussian Process Metadata** — kernel family, fitted hyperparameters, log marginal likelihood
- **v2.1: Prediction Uncertainty** — mean, std, min, max of GP `prediction_std`
- Reproducibility checklist
- Final ranking table

**v2.1 Cleanup**: The export now filters out internal Pipeline noise (`preprocessor__*`, `steps`, `memory`, `verbose`) and strips the `model__` prefix from parameter keys. This makes the report significantly cleaner and more readable.

### Model Export
1. Go to **"Results"** tab
2. Select model from dropdown
3. Click **"Save Model"** — generates a Model Bundle with:
   - Trained model file
   - `manifest.json` with all metadata
   - `MODEL_CARD.md` (auto-generated documentation)
   - Preprocessor configuration

### Model Cards
Each saved model includes an auto-generated **MODEL_CARD.md**:
- Model details and architecture
- Intended use and dataset characteristics
- Performance metrics and confidence intervals
- Feature names used during training
- Known limitations
- Reproducibility instructions
- Citation guidance

### Prediction Script
1. Click **"Generate Python Script"**
2. Review script preview
3. Save to file
4. Use script for batch predictions

## Tips and Best Practices

### Data Preparation
- Ensure no missing target values
- Remove duplicate rows
- Check for outliers
- Normalize/standardize features if needed

### Model Selection
- Start with simpler models (Random Forest, Ridge)
- Try gradient boosting for better performance
- Use neural networks for large datasets

### Validation Strategy
- Use **Group K-Fold** or **LOGO** (by chemical system or batch) for materials data prone to leakage
- Use **Stratified K-Fold (quantiles)** for skewed regression targets
- Use **LOO** for very small datasets (N ≤ 30)
- Standard **K-Fold** for larger, balanced datasets

### Hyperparameter Optimization
- Start with 50 trials
- Use pruning to save time
- Check parameter importance plots

### Neural Networks
- Start with 2 hidden layers
- Use dropout (0.2-0.5) to prevent overfitting
- Enable early stopping
- Monitor validation loss

## Troubleshooting

### Application Won't Start
- Check Python version (3.8+)
- Install all dependencies: `pip install -r requirements.txt`
- Check for conflicting packages

### Out of Memory
- Reduce batch size for neural networks
- Use fewer CV folds
- Process data in chunks

### Slow Training
- Enable GPU if available
- Reduce number of estimators
- Use early stopping

### Poor Model Performance
- Check data quality
- Try feature engineering
- Use hyperparameter optimization
- Consider ensemble methods

## Keyboard Shortcuts

- `Ctrl+O`: Load data
- `Ctrl+S`: Save model
- `Ctrl+Q`: Exit application

## Getting Help

- Check log files in `~/.materials_automl/logs/`
- Review error messages in the application
- Consult the README.md for installation issues
