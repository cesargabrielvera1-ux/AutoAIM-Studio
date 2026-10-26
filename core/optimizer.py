"""Hyperparameter optimization using Optuna."""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

from ..utils.logger import LoggerMixin
from ..utils.hardware_detector import get_hardware_detector
from .cv_factory import CVConfig, build_cv_plan
from .model_registry import _build_gp_kernel


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""
    best_params: Dict[str, Any]
    best_score: float
    best_model: Any
    n_trials: int
    optimization_time: float
    all_trials: Optional[pd.DataFrame] = None
    study_name: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)  # Additional metrics
    problem_type: str = "regression"
    cv_metrics: Optional[Dict[str, list]] = None  # Cross-validation scores per trial
    # v2.0: Complete optimization configuration used for THIS run.
    # Stored at optimization time so the export shows the ACTUAL config
    # instead of reading from the current GUI values (which may have changed
    # if the user ran multiple optimizations with different settings).
    optimization_config: Dict[str, Any] = field(default_factory=dict)
    # v2.0: Search strategy used ("bayesian" or "grid")
    search_strategy: str = "bayesian"
    # v2.0: Cross-validation strategy used ("kfold" or "loo")
    cv_strategy: str = "kfold"
    # v2.0: Full CV configuration (strategy, folds, bins, group source, ...)
    # serialized for the session exporter and Details popups.
    cv_config_dict: Optional[Dict[str, Any]] = None


def _apply_random_state(params: Dict[str, Any], algorithm: str,
                        random_state: int) -> Dict[str, Any]:
    """Apply a random seed to a model param dict, honoring each library's name.

    v2.0 FIX: CatBoost uses ``random_seed`` (not ``random_state``); without this
    mapping the optimizer silently ignored the user's seed for CatBoost. SVR and
    k-NN are deterministic and accept no seed, so the key is removed entirely.
    """
    params = params.copy()
    if algorithm in ('Support Vector Machine', 'k-Nearest Neighbors'):
        params.pop('random_state', None)
        params.pop('random_seed', None)
        return params
    if algorithm == 'CatBoost':
        params.pop('random_state', None)
        params['random_seed'] = random_state
        return params
    params.pop('random_seed', None)
    params['random_state'] = random_state
    return params


class HyperparameterOptimizer(LoggerMixin):
    """Optimize hyperparameters using Bayesian optimization (Optuna)."""

    def __init__(self):
        """Initialize optimizer."""
        self._cv_scores: List[float] = []
        self._cv_scores_nn: List[float] = []
    
    def get_param_ranges(self, algorithm: str) -> Dict[str, Tuple[float, float, str]]:
        """Get default parameter ranges for an algorithm.
        
        Args:
            algorithm: Algorithm name
            
        Returns:
            Dictionary mapping parameter names to (min, max, type) tuples
        """
        ranges = {
            "Random Forest": {
                'n_estimators': (50, 500, 'int'),
                'max_depth': (3, 30, 'int'),
                'min_samples_split': (2, 20, 'int'),
                'min_samples_leaf': (1, 10, 'int'),
                'max_features': (['sqrt', 'log2', None], None, 'categorical'),
            },
            "Gradient Boosting": {
                'n_estimators': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'max_depth': (3, 10, 'int'),
                'min_samples_split': (2, 20, 'int'),
                'min_samples_leaf': (1, 20, 'int'),
                'subsample': (0.6, 1.0, 'float'),
            },
            "Histogram Gradient Boosting": {
                'max_iter': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'max_depth': (3, 10, 'int'),
                'min_samples_leaf': (1, 100, 'int'),
                'max_bins': (50, 255, 'int'),
                'l2_regularization': (1e-8, 10.0, 'float_log'),
            },
            "XGBoost": {
                'n_estimators': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'max_depth': (3, 10, 'int'),
                'min_child_weight': (1, 20, 'int'),
                'subsample': (0.6, 1.0, 'float'),
                'colsample_bytree': (0.6, 1.0, 'float'),
                'reg_alpha': (1e-8, 10.0, 'float'),
                'reg_lambda': (1e-8, 10.0, 'float'),
                'gamma': (0.0, 5.0, 'float'),
            },
            "LightGBM": {
                'n_estimators': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'num_leaves': (20, 150, 'int'),
                'max_depth': (3, 15, 'int'),
                'min_child_samples': (1, 100, 'int'),
                'subsample': (0.6, 1.0, 'float'),
                'colsample_bytree': (0.6, 1.0, 'float'),
                'reg_alpha': (1e-8, 10.0, 'float'),
                'reg_lambda': (1e-8, 10.0, 'float'),
            },
            "CatBoost": {
                'iterations': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'depth': (3, 10, 'int'),
                'l2_leaf_reg': (1, 10, 'float_log'),
            },
            "Support Vector Machine": {
                'C': (0.001, 1000.0, 'float_log'),
                'epsilon': (0.001, 1.0, 'float'),
                'gamma': (1e-4, 10.0, 'float_log'),
                'kernel': (['rbf', 'linear', 'poly', 'sigmoid'], None, 'categorical'),
            },
            "k-Nearest Neighbors": {
                'n_neighbors': (3, 20, 'int'),
                'weights': (['uniform', 'distance'], None, 'categorical'),
                'p': ([1, 2], None, 'categorical'),
            },
            "Gaussian Process": {
                'alpha': (1e-10, 1.0, 'float_log'),
                'n_restarts_optimizer': (1, 20, 'int'),
                'normalize_y': ([True, False], None, 'categorical'),
                # v2.1: Full kernel family support
                'kernel_type': (['RBF', 'Matern', 'RationalQuadratic', 'WhiteKernel', 'RBF+WhiteKernel'], None, 'categorical'),
                'kernel_length_scale': (1e-3, 100.0, 'float_log'),
                'kernel_nu': ([0.5, 1.5, 2.5], None, 'categorical'),
                'kernel_alpha_rq': (1e-3, 100.0, 'float_log'),
                'kernel_noise_level': (1e-5, 1.0, 'float_log'),
            },
            "Ridge Regression": {
                'alpha': (0.001, 100.0, 'float_log'),
            },
        }
        
        return ranges.get(algorithm, {})
    
    def _suggest_param(self, trial, param_name, min_val, max_val, param_type):
        """Suggest a parameter value using Optuna."""
        if param_type == 'int':
            return trial.suggest_int(param_name, int(min_val), int(max_val))
        elif param_type == 'int_log':
            return trial.suggest_int(param_name, int(min_val), int(max_val), log=True)
        elif param_type == 'float':
            return trial.suggest_float(param_name, float(min_val), float(max_val))
        elif param_type == 'float_log':
            return trial.suggest_float(param_name, float(min_val), float(max_val), log=True)
        elif param_type == 'categorical':
            choices = min_val if isinstance(min_val, (list, tuple)) else [min_val]
            return trial.suggest_categorical(param_name, list(choices))
        else:
            return trial.suggest_float(param_name, 0.0, 1.0)
    
    def optimize_sklearn_model(
        self,
        model_type: str,
        algorithm: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        problem_type: str = 'regression',
        param_ranges: Optional[Dict[str, Tuple]] = None,
        n_trials: int = 50,
        cv_folds: int = 5,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float, Dict], None]] = None,
        random_state: int = 42,
        pruning: bool = True,
        selected_params: Optional[Dict[str, bool]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        cv_strategy: str = 'kfold',
        cv_config: Optional[CVConfig] = None,
        cv_groups: Optional[np.ndarray] = None,
        # v2.0: Preprocessor for sklearn Pipeline — eliminates data leakage
        preprocessor: Optional[Any] = None,
    ) -> OptimizationResult:
        """Optimize scikit-learn model hyperparameters.

        v2.0 CRITICAL FIX: When ``preprocessor`` is provided, each trial builds a
        ``Pipeline([preprocessor, model])`` so that cross-validation refits the
        scaler inside each fold. This removes the data leakage that existed when
        the preprocessor was fitted on the full training set before CV.

        Args:
            model_type: Type of model
            algorithm: Algorithm name
            X_train: Training features (raw DataFrame or numpy array)
            y_train: Training targets
            X_test: Test features (raw DataFrame or numpy array)
            y_test: Test targets
            problem_type: 'regression' or 'classification'
            param_ranges: Custom parameter ranges (optional)
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds
            timeout: Optimization timeout in seconds
            progress_callback: Callback(trial_num, score, params)
            random_state: Random seed
            pruning: Whether to use pruning (stop unpromising trials)
            default_params: Fixed params (e.g., n_jobs) merged with optimized ones.
                Preserves settings not in param_ranges (v2.0).
            cv_strategy: 'kfold' or 'loo' (v2.0)
            cv_config: full CV configuration (v2.0). Takes precedence over
                cv_strategy; carries group labels for grouped strategies.
            cv_groups: group labels aligned with X_train (v2.0).
            preprocessor: sklearn ColumnTransformer (or compatible). When provided,
                the optimizer wraps every trial model in a Pipeline so that the
                preprocessor is refitted per CV fold.

        Returns:
            OptimizationResult
        """
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter optimization. "
                            "Install it with: pip install optuna")
        
        # Get parameter ranges
        if param_ranges is None:
            param_ranges = self.get_param_ranges(algorithm)
        
        if not param_ranges:
            raise ValueError(f"No parameter ranges defined for {algorithm}")
        
        # Import model class
        model_class = self._get_model_class(algorithm, problem_type)

        # v2.0: Base params (fixed settings like n_jobs) merged with optimized params
        base_params = (default_params or {}).copy()

        # v2.0: sklearn CV uses n_jobs=1 because individual models already
        # have internal parallelism (XGB n_jobs, RF n_jobs, etc.).
        # Using n_jobs > 1 would cause nested parallelism / oversubscription.
        cv_n_jobs = 1

        # v2.0: Resolve the CV plan via the shared factory (single decision
        # point for splitter + scoring + OOF). A full cv_config takes
        # precedence; otherwise build one from the legacy cv_strategy string.
        if cv_config is None:
            cv_config = CVConfig(
                strategy=('loo' if cv_strategy == 'loo' else 'kfold'),
                n_folds=cv_folds,
                random_state=random_state,
            )
        if cv_groups is not None:
            cv_config.groups = np.asarray(cv_groups)
        cv_strategy = cv_config.strategy
        cv_plan = build_cv_plan(cv_config, X_train, y_train, problem_type=problem_type)
        cv_splitter = cv_plan.splitter
        cv_groups_resolved = cv_plan.split_labels
        opt_scoring = cv_plan.scoring
        for diag in cv_plan.diagnostics:
            self.logger.warning(f"CV diagnostic: {diag}")

        # Helper: create model with base + optimized params merged.
        # v2.0: delegate random-state naming to _apply_random_state (CatBoost
        # uses random_seed; SVR/kNN accept no seed).
        # v2.1: For GP, build the kernel object from string parameters before
        # passing to the estimator constructor.
        def _make_model(trial_params: Dict):
            merged = base_params.copy()
            merged.update(trial_params)
            if algorithm == "Gaussian Process":
                merged = _build_gp_kernel(merged)
            merged = _apply_random_state(merged, algorithm, random_state)
            return model_class(**merged)

        # v2.0: Build Pipeline wrapper so the preprocessor is refitted per fold.
        # When preprocessor is None, the wrapper just returns the bare model.
        def _wrap_pipeline(model):
            if preprocessor is not None:
                from sklearn.pipeline import Pipeline
                return Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model),
                ])
            return model

        # Create study
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=10) if pruning else None
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state),
            pruner=pruner
        )

        # Track CV scores from the best trial
        best_cv_scores = []  # Will store the CV fold scores from the best trial
        best_trial_score = float('-inf')

        def objective(trial):
            nonlocal best_cv_scores, best_trial_score

            # Sample parameters (only selected ones; unselected use model defaults)
            params = {}
            for param_name, (min_val, max_val, param_type) in param_ranges.items():
                # v1.5.0: Only suggest parameters the user selected for optimization.
                # Unselected parameters use the model's built-in defaults.
                is_selected = selected_params.get(param_name, True) if selected_params else True
                if is_selected:
                    params[param_name] = self._suggest_param(trial, param_name, min_val, max_val, param_type)

            # v2.0: Create model + Pipeline wrapper (zero data leakage)
            model = _make_model(params)
            trial_estimator = _wrap_pipeline(model)

            # v2.0: Scoring comes from the shared CV plan. The factory already
            # decided neg_mean_absolute_error for OOF strategies (LOO / LOGO /
            # single-sample group folds) where R² is undefined for one sample.
            scoring = opt_scoring

            try:
                cv_scores = cross_val_score(trial_estimator, X_train, y_train,
                                           cv=cv_splitter, scoring=scoring,
                                           groups=cv_groups_resolved,
                                           n_jobs=cv_n_jobs)
                score = cv_scores.mean()
                # Store CV scores if this is the best trial so far
                if score > best_trial_score:
                    best_trial_score = score
                    best_cv_scores = cv_scores.tolist()
            except Exception as e:
                self.logger.error(f"CV error: {e}")
                score = float('-inf')

            if progress_callback:
                progress_callback(trial.number, score, params)

            return score

        start_time = time.time()

        if timeout:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        optimization_time = time.time() - start_time

        # Check if any trial completed successfully
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            raise RuntimeError(
                f"Optimization failed: none of the {n_trials} trials completed successfully. "
                f"Check the logs for CV errors (NaN in data, incompatible model, etc.)."
            )

        # v2.0: Train best model with merged base + optimized params
        best_params = study.best_params
        best_model = _wrap_pipeline(_make_model(best_params))
        best_model.fit(X_train, y_train)
        
        # Compute metrics
        metrics = {}
        if X_test is not None and y_test is not None:
            predictions = best_model.predict(X_test)
            if problem_type == 'regression':
                metrics['r2'] = r2_score(y_test, predictions)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            else:
                metrics['accuracy'] = accuracy_score(y_test, predictions)
        
        # Create trials DataFrame
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                }
                trials_data.append(trial_data)
        
        trials_df = pd.DataFrame(trials_data) if trials_data else None
        
        # Compile cv_metrics from the best trial's fold scores
        cv_metrics = None
        if best_cv_scores:
            cv_metrics = {'test_score': best_cv_scores}
            # v2.0: annotate OOF strategies (MAE, not R²) + grouping diagnostics
            if cv_plan.use_oof and problem_type == 'regression':
                cv_metrics['test_score_metric'] = 'mae'
            if cv_plan.group_info:
                cv_metrics['group_info'] = cv_plan.group_info
            if cv_plan.diagnostics:
                cv_metrics['cv_diagnostics'] = cv_plan.diagnostics

        # v2.0: Store the ACTUAL configuration used for this optimization.
        # This ensures the export shows correct settings even when the user
        # runs multiple optimizations with different parameters.
        opt_config = {
            'algorithm': algorithm,
            'n_trials': n_trials,
            'cv_folds': cv_folds,
            'random_state': random_state,
            'timeout_sec': timeout,
            'pruning_enabled': pruning,
            'problem_type': problem_type,
            'selected_params': selected_params or {},
            'cv_strategy': cv_strategy,
        }
        if param_ranges:
            opt_config['param_ranges'] = {k: str(v) for k, v in param_ranges.items()}

        # v2.0: serialize the full CV configuration for export / Details popups
        # (single source of truth: CVConfig.to_dict, includes the reserved bin).
        cv_config_dict = cv_config.to_dict()

        return OptimizationResult(
            best_params=best_params,
            best_score=study.best_value,
            best_model=best_model,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            all_trials=trials_df,
            study_name=f"{algorithm}_optimization",
            metrics=metrics,
            problem_type=problem_type,
            cv_metrics=cv_metrics,
            optimization_config=opt_config,
            search_strategy="bayesian",
            cv_strategy=cv_strategy,
            cv_config_dict=cv_config_dict,
        )
    
    def optimize_nn_architecture(
        self,
        nn_builder,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray],
        y_val: Optional[np.ndarray],
        input_dim: int,
        output_dim: int,
        problem_type: str = 'regression',
        n_trials: int = 50,
        timeout: Optional[int] = None,
        cv_folds: int = 1,
        param_ranges: Optional[Dict[str, Tuple]] = None,
        study_name: str = "nn_optimization",
        progress_callback: Optional[Callable[[int, float, Dict], None]] = None,
        random_state: int = 42,
        opt_epochs: int = 200,
        cv_epochs: int = 100,
        opt_patience: int = 30,
        cv_patience: int = 20
    ) -> OptimizationResult:
        """Optimize neural network architecture and hyperparameters.
        
        Args:
            nn_builder: NeuralNetworkBuilder instance
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            input_dim: Input dimension
            output_dim: Output dimension
            problem_type: 'regression' or 'classification'
            n_trials: Number of trials
            timeout: Timeout in seconds
            cv_folds: Number of CV folds (if > 1, uses K-fold CV)
            param_ranges: Custom parameter ranges (optional)
            study_name: Study name
            progress_callback: Callback(trial_num, score, params)
            random_state: Random seed
            opt_epochs: Epochs per trial when using simple validation (default: 200)
            cv_epochs: Epochs per fold when using K-fold CV (default: 100)
            opt_patience: Early stopping patience for simple validation (default: 30)
            cv_patience: Early stopping patience per CV fold (default: 20)
            
        Returns:
            OptimizationResult
        """
        self.logger.info(
            f"NN optimization: opt_epochs={opt_epochs}, cv_epochs={cv_epochs}, "
            f"opt_patience={opt_patience}, cv_patience={cv_patience}"
        )
        try:
            import optuna
        except ImportError:
            raise ImportError("Optuna is required for hyperparameter optimization.")

        # v2.0 FIX: Defensively convert pandas objects to numpy arrays.
        # torch.FloatTensor cannot infer shape from DataFrames/Series.
        if hasattr(X_train, 'values'):
            X_train = np.asarray(X_train)
        if hasattr(y_train, 'values'):
            y_train = np.asarray(y_train)
        if X_val is not None and hasattr(X_val, 'values'):
            X_val = np.asarray(X_val)
        if y_val is not None and hasattr(y_val, 'values'):
            y_val = np.asarray(y_val)

        # Default parameter ranges
        if param_ranges is None:
            param_ranges = {
                'n_layers': (1, 5, 'int'),
                'n_units': (16, 512, 'int_log'),
                'dropout': (0.0, 0.5, 'float'),
                'activation': (['relu', 'leaky_relu', 'tanh', 'gelu'], None, 'categorical'),
                'batch_norm': ([True, False], None, 'categorical'),
                'learning_rate': (1e-5, 1e-2, 'float_log'),
                'batch_size': ([16, 32, 64, 128], None, 'categorical'),
                'weight_decay': (1e-6, 1e-3, 'float_log'),
                'optimizer': (['adam', 'adamw'], None, 'categorical'),
                'gradient_clip_val': (0.0, 5.0, 'float'),
            }
        
        # Helper function to sample with ranges
        # v1.5.0 FIX: Return sensible defaults when a param was deselected by user
        _NN_PARAM_DEFAULTS = {
            'n_layers': 2, 'n_units': 64, 'dropout': 0.0,
            'activation': 'relu', 'batch_norm': True,
            'learning_rate': 1e-3, 'batch_size': 32,
            'weight_decay': 0.0, 'optimizer': 'adam',
            'gradient_clip_val': 0.0, 'use_layer_norm': False,
        }

        def _suggest_with_ranges(trial, param_name, ranges):
            # v1.2.1: For per-layer params like 'n_units_0', look up the base name
            # 'n_units' in ranges. Optuna still uses the full unique name.
            range_key = param_name
            if param_name not in ranges:
                # Try stripping _{number} suffix to find base parameter range
                import re
                m = re.match(r'^(.+)_\d+$', param_name)
                if m and m.group(1) in ranges:
                    range_key = m.group(1)
                elif m and m.group(1) in _NN_PARAM_DEFAULTS:
                    # v1.5.0: Param was deselected by user → return default
                    return _NN_PARAM_DEFAULTS[m.group(1)]
                else:
                    # v1.5.0: Unknown param → return default if known
                    return _NN_PARAM_DEFAULTS.get(param_name, 0.0)
            min_val, max_val, ptype = ranges[range_key]
            if ptype == 'int':
                return trial.suggest_int(param_name, int(min_val), int(max_val))
            elif ptype == 'int_log':
                return trial.suggest_int(param_name, int(min_val), int(max_val), log=True)
            elif ptype == 'float':
                return trial.suggest_float(param_name, float(min_val), float(max_val))
            elif ptype == 'float_log':
                return trial.suggest_float(param_name, float(min_val), float(max_val), log=True)
            elif ptype == 'categorical':
                choices = min_val if isinstance(min_val, (list, tuple)) else [min_val]
                return trial.suggest_categorical(param_name, list(choices))
            return trial.suggest_float(param_name, 0.0, 1.0)
        
        self._cv_scores_nn = []
        
        # Shared state to track consecutive failures
        _last_exception = [None]  # Use list for mutable reference in closure
        
        def objective(trial):
            # Validate data before starting — these are FATAL, propagate immediately
            if X_train is None or y_train is None:
                raise ValueError("Training data (X_train, y_train) is None. "
                                 "Please load data and set target column first.")
            if cv_folds <= 1 and (X_val is None or y_val is None):
                raise ValueError(
                    f"Validation data is None but cv_folds={cv_folds}. "
                    f"Either set cv_folds > 1 in the NN tab, "
                    f"or load external validation data in the Data tab."
                )
            # v2.0 FIX: Use .to_numpy() before np.isnan() to avoid
            # "The truth value of a Series is ambiguous" when X_train is a
            # pandas DataFrame. np.isnan(df) returns a DataFrame of bools, and
            # .any() without axis returns a Series — which cannot be evaluated
            # in a boolean context.
            if np.isnan(X_train.to_numpy()).any() or np.isnan(y_train).any():
                nan_cols = []
                raise ValueError(
                    f"Training data contains NaN values. "
                    f"Please clean your dataset before optimizing."
                )
            if cv_folds <= 1 and (np.isnan(X_val.to_numpy()).any() or np.isnan(y_val).any()):
                raise ValueError("Validation data contains NaN values.")
            
            # Sample architecture parameters
            n_layers = _suggest_with_ranges(trial, 'n_layers', param_ranges)
            
            hidden_layers = []
            for i in range(n_layers):
                # v1.2.1 FIX: Use per-layer parameter names so each layer can
                # have different values. Without the _{i} suffix, Optuna treats
                # all layers as the same parameter and they all get identical values.
                n_units = _suggest_with_ranges(trial, f'n_units_{i}', param_ranges)
                dropout = _suggest_with_ranges(trial, f'dropout_{i}', param_ranges)
                activation = _suggest_with_ranges(trial, f'activation_{i}', param_ranges)
                use_bn = _suggest_with_ranges(trial, f'batch_norm_{i}', param_ranges)
                # Categorical boolean may come as string from Optuna
                if isinstance(use_bn, str):
                    use_bn = use_bn.lower() == 'true'
                
                hidden_layers.append({
                    'n_units': n_units,
                    'activation': activation,
                    'dropout_rate': dropout,
                    'use_batch_norm': bool(use_bn),
                    'use_layer_norm': False
                })
            
            # Training parameters
            learning_rate = _suggest_with_ranges(trial, 'learning_rate', param_ranges)
            batch_size = _suggest_with_ranges(trial, 'batch_size', param_ranges)
            # Categorical values may come as strings - convert to proper types
            if isinstance(batch_size, str):
                batch_size = int(batch_size)
            weight_decay = _suggest_with_ranges(trial, 'weight_decay', param_ranges)
            optimizer_name = _suggest_with_ranges(trial, 'optimizer', param_ranges)
            gradient_clip_val = _suggest_with_ranges(trial, 'gradient_clip_val', param_ranges)
            
            # Build and train model
            from ..core.nn_builder import NeuralNetworkBuilder
            builder = NeuralNetworkBuilder()
            
            try:
                builder.create_architecture(
                    input_dim=input_dim,
                    output_dim=output_dim,
                    hidden_layers_config=hidden_layers,
                    problem_type=problem_type
                )
                builder.build_model()
                builder.create_training_config(
                    epochs=opt_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    optimizer=optimizer_name,
                    weight_decay=weight_decay,
                    early_stopping_patience=opt_patience,
                    gradient_clip_val=gradient_clip_val
                )
                
                if cv_folds > 1 and len(X_train) >= cv_folds * 2:
                    # K-fold CV for NN optimization
                    from sklearn.model_selection import KFold
                    from sklearn.metrics import r2_score, accuracy_score
                    
                    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
                    fold_scores = []
                    
                    for train_idx, val_idx in kf.split(X_train):
                        X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                        y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                        
                        fold_builder = NeuralNetworkBuilder()
                        fold_builder.create_architecture(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            hidden_layers_config=hidden_layers,
                            problem_type=problem_type
                        )
                        fold_builder.build_model()
                        fold_builder.create_training_config(
                            epochs=cv_epochs,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            optimizer=optimizer_name,
                            weight_decay=weight_decay,
                            early_stopping_patience=cv_patience,
                            gradient_clip_val=gradient_clip_val
                        )
                        
                        fold_builder.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val, verbose=False)
                        
                        fold_pred = fold_builder.predict(X_fold_val)
                        if fold_pred.ndim > 1 and fold_pred.shape[1] == 1:
                            fold_pred = fold_pred.flatten()
                        
                        if problem_type == 'regression':
                            fold_scores.append(r2_score(y_fold_val, fold_pred))
                        else:
                            fold_scores.append(accuracy_score(y_fold_val, fold_pred))
                    
                    score = np.mean(fold_scores)
                    self._cv_scores_nn.append(score)
                else:
                    # Simple validation
                    builder.fit(X_train, y_train, X_val, y_val, verbose=False)
                    
                    pred = builder.predict(X_val)
                    if pred.ndim > 1 and pred.shape[1] == 1:
                        pred = pred.flatten()
                    
                    if problem_type == 'regression':
                        score = r2_score(y_val, pred)
                    else:
                        score = accuracy_score(y_val, pred)
                    
                    self._cv_scores_nn.append(score)
                
                if progress_callback:
                    progress_callback(trial.number, score, trial.params)
                
                return score
            
            except Exception as e:
                import traceback
                _last_exception[0] = (e, traceback.format_exc())
                self.logger.error(f"Trial {trial.number} failed: {e}")
                self.logger.error(traceback.format_exc())
                return float('-inf')
        
        start_time = time.time()
        
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        
        if timeout:
            study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        else:
            study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        
        optimization_time = time.time() - start_time
        
        # Check if any trial completed successfully
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            last_err, last_tb = _last_exception[0] if _last_exception[0] else (None, None)
            if last_err:
                raise RuntimeError(
                    f"NN optimization failed: all {n_trials} trials returned -inf.\n\n"
                    f"Last error: {last_err}\n\n"
                    f"Common causes:\n"
                    f"1. NaN values in training/validation data\n"
                    f"2. X_val/y_val is None (set CV folds > 1 or load validation data)\n"
                    f"3. PyTorch model build error (check input/output dimensions)\n"
                    f"4. All predictions are NaN (learning rate too high, bad architecture)\n\n"
                    f"Check the application logs for the full traceback."
                )
            else:
                raise RuntimeError(
                    f"NN optimization failed: all {n_trials} trials returned -inf. "
                    f"No exception was captured — check the logs."
                )
        
        # Compile results
        best_params = study.best_params
        
        trials_data = []
        for trial in study.trials:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trials_data.append({
                    'number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                })
        
        trials_df = pd.DataFrame(trials_data) if trials_data else None
        
        cv_metrics = None
        if self._cv_scores_nn:
            cv_metrics = {'test_score': self._cv_scores_nn}
        
        # Note: Test metrics are not computed here because X_test/y_test
        # are not available in this scope. Metrics are computed during
        # post-optimization if needed.
        metrics = {}

        # v2.0: Store the ACTUAL configuration used for this NN optimization.
        # NOTE: 'algorithm' is hardcoded as "Neural Network" because
        # optimize_nn_architecture() does not have an 'algorithm' parameter.
        # Extract range bounds from param_ranges for the export config.
        # param_ranges is guaranteed to be set (defaults applied above if None).
        n_layers_range = param_ranges.get('n_layers', (1, 5, 'int'))
        n_units_range = param_ranges.get('n_units', (16, 512, 'int_log'))
        opt_config = {
            'algorithm': "Neural Network",
            'n_trials': n_trials,
            'cv_folds': cv_folds,
            'random_state': random_state,
            'timeout_sec': timeout,
            'pruning_enabled': False,
            'problem_type': problem_type,
            'nn_max_layers': int(n_layers_range[1]) if len(n_layers_range) > 1 else 5,
            'nn_min_units': int(n_units_range[0]) if len(n_units_range) > 0 else 16,
            'nn_max_units': int(n_units_range[1]) if len(n_units_range) > 1 else 512,
        }

        return OptimizationResult(
            best_params=best_params,
            best_score=study.best_value,
            best_model=None,
            n_trials=len(study.trials),
            optimization_time=optimization_time,
            all_trials=trials_df,
            study_name=study_name,
            metrics=metrics,
            problem_type=problem_type,
            cv_metrics=cv_metrics,
            optimization_config=opt_config,
            search_strategy="bayesian",
        )
    
    def _get_model_class(self, algorithm: str, problem_type: str):
        """Get model class for algorithm."""
        if problem_type == 'regression':
            if algorithm == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor
            elif algorithm == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor
            elif algorithm == "Histogram Gradient Boosting":
                from sklearn.ensemble import HistGradientBoostingRegressor
                return HistGradientBoostingRegressor
            elif algorithm == "XGBoost":
                from xgboost import XGBRegressor
                return XGBRegressor
            elif algorithm == "LightGBM":
                from lightgbm import LGBMRegressor
                return LGBMRegressor
            elif algorithm == "CatBoost":
                from catboost import CatBoostRegressor
                return CatBoostRegressor
            elif algorithm == "Support Vector Machine":
                from sklearn.svm import SVR
                return SVR
            elif algorithm == "k-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor
            elif algorithm == "Gaussian Process":
                from sklearn.gaussian_process import GaussianProcessRegressor
                return GaussianProcessRegressor
            elif algorithm == "Ridge Regression":
                from sklearn.linear_model import Ridge
                return Ridge
        else:
            if algorithm == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier
            elif algorithm == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier
            elif algorithm == "Histogram Gradient Boosting":
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier
            elif algorithm == "XGBoost":
                from xgboost import XGBClassifier
                return XGBClassifier
            elif algorithm == "LightGBM":
                from lightgbm import LGBMClassifier
                return LGBMClassifier
            elif algorithm == "CatBoost":
                from catboost import CatBoostClassifier
                return CatBoostClassifier
            elif algorithm == "Support Vector Machine":
                from sklearn.svm import SVC
                return SVC
            elif algorithm == "k-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier
            elif algorithm == "Gaussian Process":
                from sklearn.gaussian_process import GaussianProcessClassifier
                return GaussianProcessClassifier
            elif algorithm == "Ridge Regression":
                from sklearn.linear_model import RidgeClassifier
                return RidgeClassifier
        
        raise ValueError(f"Unknown algorithm: {algorithm}")


class GridSearchOptimizer(LoggerMixin):
    """Optimize hyperparameters using exhaustive Grid Search (sklearn GridSearchCV).

    v2.0: Provides a deterministic alternative to Bayesian optimization.
    Recommended for:
      - <=3 hyperparameters with small discrete ranges
      - Reproducibility-critical experiments
      - Baseline comparisons against Bayesian results

    GridSearch is NOT offered for:
      - Neural Network architecture search (non-rectangular search space)
      - Ensemble weight optimization (continuous space with constraints)
    """

    RESOLUTION_POINTS = {'coarse': 3, 'medium': 5, 'fine': 7}

    def __init__(self, resolution: str = 'medium'):
        """Initialize GridSearch optimizer.

        Args:
            resolution: Grid density — 'coarse' (3 points), 'medium' (5), 'fine' (7).
        """
        self.resolution = resolution
        self.n_points = self.RESOLUTION_POINTS.get(resolution, 5)

    def _ranges_to_grid(self, param_ranges: Dict[str, Tuple],
                        selected_params: Optional[Dict[str, bool]] = None) -> Dict[str, List]:
        """Convert Optuna-style ranges to sklearn param_grid.

        Args:
            param_ranges: Dict of param_name -> (min, max, type)
            selected_params: Dict of param_name -> bool (only optimize selected)

        Returns:
            param_grid suitable for sklearn GridSearchCV.
        """
        param_grid = {}
        for param_name, (min_val, max_val, param_type) in param_ranges.items():
            is_selected = selected_params.get(param_name, True) if selected_params else True
            if not is_selected:
                continue

            if param_type == 'categorical':
                choices = min_val if isinstance(min_val, (list, tuple)) else [min_val]
                param_grid[param_name] = list(choices)
            elif param_type == 'int':
                lo, hi = int(min_val), int(max_val)
                if lo >= hi:
                    param_grid[param_name] = [lo]
                else:
                    points = min(self.n_points, hi - lo + 1)
                    vals = np.linspace(lo, hi, points, dtype=int)
                    param_grid[param_name] = sorted(set(int(v) for v in vals))
            elif param_type == 'int_log':
                lo, hi = int(min_val), int(max_val)
                if lo >= hi:
                    param_grid[param_name] = [lo]
                else:
                    points = min(self.n_points, max(2, int(np.log2(hi)) - int(np.log2(max(lo, 1))) + 2))
                    log_vals = np.logspace(np.log10(max(lo, 1)), np.log10(hi), points)
                    vals = np.unique(np.round(log_vals).astype(int))
                    vals = vals[(vals >= lo) & (vals <= hi)]
                    param_grid[param_name] = sorted(set(int(v) for v in vals)) if len(vals) > 0 else [lo]
            elif param_type == 'float':
                lo, hi = float(min_val), float(max_val)
                if abs(hi - lo) < 1e-12:
                    param_grid[param_name] = [lo]
                else:
                    vals = np.linspace(lo, hi, self.n_points)
                    param_grid[param_name] = [round(float(v), 6) for v in vals]
            elif param_type == 'float_log':
                lo, hi = float(min_val), float(max_val)
                if lo <= 0 or hi <= 0:
                    # Fallback to linear for non-positive ranges
                    vals = np.linspace(lo, hi, self.n_points)
                    param_grid[param_name] = [round(float(v), 6) for v in vals]
                else:
                    vals = np.logspace(np.log10(lo), np.log10(hi), self.n_points)
                    param_grid[param_name] = [round(float(v), 8) for v in vals]
            else:
                param_grid[param_name] = [min_val]
        return param_grid

    def _estimate_combinations(self, param_grid: Dict[str, List]) -> int:
        """Estimate total grid combinations."""
        total = 1
        for vals in param_grid.values():
            total *= len(vals)
        return total

    def optimize_sklearn_model(
        self,
        model_type: str,
        algorithm: str,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[np.ndarray] = None,
        problem_type: str = 'regression',
        param_ranges: Optional[Dict[str, Tuple]] = None,
        cv_folds: int = 5,
        timeout: Optional[int] = None,
        progress_callback: Optional[Callable[[int, float, Dict], None]] = None,
        random_state: int = 42,
        selected_params: Optional[Dict[str, bool]] = None,
        default_params: Optional[Dict[str, Any]] = None,
        cv_strategy: str = 'kfold',
        cv_config: Optional[CVConfig] = None,
        cv_groups: Optional[np.ndarray] = None,
        # v2.0: Preprocessor for sklearn Pipeline — eliminates data leakage
        preprocessor: Optional[Any] = None,
    ) -> OptimizationResult:
        """Optimize scikit-learn model hyperparameters via Grid Search.

        Same signature as HyperparameterOptimizer.optimize_sklearn_model
        for drop-in compatibility. v2.0: accepts cv_config / cv_groups for
        stratified and grouped cross-validation.

        v2.0 CRITICAL FIX: When ``preprocessor`` is provided, the search wraps
        the estimator in a ``Pipeline([preprocessor, model])`` so that CV refits
        the scaler inside each fold. The param_grid keys are automatically
        prefixed with ``model__`` to target the estimator step inside the Pipeline.
        """
        from sklearn.model_selection import GridSearchCV

        if param_ranges is None:
            # Reuse HyperparameterOptimizer's ranges as defaults
            param_ranges = HyperparameterOptimizer().get_param_ranges(algorithm)
        if not param_ranges:
            raise ValueError(f"No parameter ranges defined for {algorithm}")

        param_grid = self._ranges_to_grid(param_ranges, selected_params)
        if not param_grid:
            raise ValueError("No parameters selected for optimization.")

        # v2.1: GP kernel objects can't be grid-searched by sklearn (they are
        # complex objects, not simple scalars). Filter kernel_* params out of
        # the grid and build a fixed kernel for the base estimator.
        if algorithm == "Gaussian Process":
            kernel_grid_keys = [k for k in param_grid if k.startswith('kernel_')]
            if kernel_grid_keys:
                self.logger.warning(
                    f"Grid Search does not support kernel tuning for GP. "
                    f"Kernel parameters {kernel_grid_keys} will use defaults. "
                    f"Use Bayesian optimization to tune kernels."
                )
                for k in kernel_grid_keys:
                    param_grid.pop(k)

        n_combinations = self._estimate_combinations(param_grid)
        if n_combinations > 5000:
            raise ValueError(
                f"Grid Search would evaluate {n_combinations} combinations — too many. "
                f"Reduce parameters, shrink ranges, or use coarser resolution. "
                f"Maximum allowed: 5000."
            )

        model_class = self._get_model_class(algorithm, problem_type)
        base_params = (default_params or {}).copy()

        # Remove params that are in the grid from base_params to avoid conflicts
        for p in param_grid:
            base_params.pop(p, None)

        # Build base estimator with fixed params.
        # v2.0: delegate random-state naming to _apply_random_state (CatBoost
        # uses random_seed; SVR/kNN accept no seed at all).
        # v2.1: For GP, convert string kernel params to actual kernel object.
        base_params = _apply_random_state(base_params, algorithm, random_state)
        if algorithm == "Gaussian Process":
            base_params = _build_gp_kernel(base_params)
        estimator = model_class(**base_params)

        # v2.0: Wrap in Pipeline when preprocessor is provided. The param_grid
        # keys must target the model step, so we prefix them with "model__".
        if preprocessor is not None:
            from sklearn.pipeline import Pipeline
            estimator = Pipeline([
                ('preprocessor', preprocessor),
                ('model', estimator),
            ])
            param_grid = {f'model__{k}': v for k, v in param_grid.items()}

        # n_jobs: avoid nested parallelism / PyInstaller issues
        import sys
        n_jobs_gs = 1 if getattr(sys, 'frozen', False) else -1

        # v2.0: Resolve the CV plan via the shared factory (single decision
        # point for splitter + scoring). The factory decides
        # neg_mean_absolute_error for OOF strategies (LOO / LOGO /
        # single-sample group folds) where R² is undefined for one sample.
        if cv_config is None:
            cv_config = CVConfig(
                strategy=('loo' if cv_strategy == 'loo' else 'kfold'),
                n_folds=cv_folds,
                random_state=random_state,
            )
        if cv_groups is not None:
            cv_config.groups = np.asarray(cv_groups)
        cv_strategy = cv_config.strategy
        cv_plan = build_cv_plan(cv_config, X_train, y_train, problem_type=problem_type)
        cv_splitter = cv_plan.splitter
        cv_groups_resolved = cv_plan.split_labels
        scoring = cv_plan.scoring
        for diag in cv_plan.diagnostics:
            self.logger.warning(f"CV diagnostic: {diag}")

        start_time = time.time()

        gs = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv_splitter,
            n_jobs=n_jobs_gs,
            refit=True,
            verbose=0,
        )

        # v2.0: pass group labels so grouped splitters (Group K-Fold / LOGO)
        # keep group membership intact inside GridSearchCV.
        if cv_groups_resolved is not None:
            gs.fit(X_train, y_train, groups=cv_groups_resolved)
        else:
            gs.fit(X_train, y_train)

        optimization_time = time.time() - start_time

        best_params = gs.best_params_.copy()
        best_model = gs.best_estimator_
        best_score = gs.best_score_

        # Build trials DataFrame (all grid points evaluated)
        cv_results = gs.cv_results_
        trials_data = []
        for i in range(len(cv_results['params'])):
            trials_data.append({
                'number': i,
                'value': float(cv_results['mean_test_score'][i]),
                'params': cv_results['params'][i],
            })
        trials_df = pd.DataFrame(trials_data) if trials_data else None

        # Extract CV scores from best index
        cv_metrics = None
        if 'split0_test_score' in cv_results:
            # v2.0: Determine actual number of splits dynamically
            # (LOO has N splits, KFold has cv_folds splits)
            n_splits = len([
                k for k in cv_results.keys()
                if k.startswith('split') and k.endswith('_test_score')
            ])
            fold_scores = [
                float(cv_results[f'split{j}_test_score'][gs.best_index_])
                for j in range(n_splits)
                if f'split{j}_test_score' in cv_results
            ]
            if fold_scores:
                cv_metrics = {'test_score': fold_scores}
                # v2.0: annotate OOF strategies (MAE, not R²) + grouping info
                if cv_plan.use_oof and problem_type == 'regression':
                    cv_metrics['test_score_metric'] = 'mae'
                if cv_plan.group_info:
                    cv_metrics['group_info'] = cv_plan.group_info
                if cv_plan.diagnostics:
                    cv_metrics['cv_diagnostics'] = cv_plan.diagnostics

        # Test metrics
        metrics = {}
        if X_test is not None and y_test is not None:
            predictions = best_model.predict(X_test)
            if problem_type == 'regression':
                metrics['r2'] = r2_score(y_test, predictions)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, predictions))
            else:
                metrics['accuracy'] = accuracy_score(y_test, predictions)

        # Emit progress for the best result
        if progress_callback:
            progress_callback(0, best_score, best_params)

        opt_config = {
            'algorithm': algorithm,
            'search_strategy': 'grid',
            'grid_resolution': self.resolution,
            'n_combinations': n_combinations,
            'cv_folds': cv_folds,
            'cv_strategy': cv_strategy,
            'random_state': random_state,
            'timeout_sec': timeout,
            'problem_type': problem_type,
            'selected_params': selected_params or {},
        }
        if param_ranges:
            opt_config['param_ranges'] = {k: str(v) for k, v in param_ranges.items()}

        # v2.0: serialize the full CV configuration for export / Details popups
        # (single source of truth: CVConfig.to_dict, includes the reserved bin).
        cv_config_dict = cv_config.to_dict()

        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_model=best_model,
            n_trials=n_combinations,
            optimization_time=optimization_time,
            all_trials=trials_df,
            study_name=f"{algorithm}_grid_search",
            metrics=metrics,
            problem_type=problem_type,
            cv_metrics=cv_metrics,
            optimization_config=opt_config,
            search_strategy="grid",
            cv_strategy=cv_strategy,
            cv_config_dict=cv_config_dict,
        )

    def _get_model_class(self, algorithm: str, problem_type: str):
        """Get model class (identical to HyperparameterOptimizer._get_model_class)."""
        if problem_type == 'regression':
            if algorithm == "Random Forest":
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor
            elif algorithm == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingRegressor
                return GradientBoostingRegressor
            elif algorithm == "Histogram Gradient Boosting":
                from sklearn.ensemble import HistGradientBoostingRegressor
                return HistGradientBoostingRegressor
            elif algorithm == "XGBoost":
                from xgboost import XGBRegressor
                return XGBRegressor
            elif algorithm == "LightGBM":
                from lightgbm import LGBMRegressor
                return LGBMRegressor
            elif algorithm == "CatBoost":
                from catboost import CatBoostRegressor
                return CatBoostRegressor
            elif algorithm == "Support Vector Machine":
                from sklearn.svm import SVR
                return SVR
            elif algorithm == "k-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsRegressor
                return KNeighborsRegressor
            elif algorithm == "Gaussian Process":
                from sklearn.gaussian_process import GaussianProcessRegressor
                return GaussianProcessRegressor
            elif algorithm == "Ridge Regression":
                from sklearn.linear_model import Ridge
                return Ridge
        else:
            if algorithm == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier
            elif algorithm == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier
            elif algorithm == "Histogram Gradient Boosting":
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier
            elif algorithm == "XGBoost":
                from xgboost import XGBClassifier
                return XGBClassifier
            elif algorithm == "LightGBM":
                from lightgbm import LGBMClassifier
                return LGBMClassifier
            elif algorithm == "CatBoost":
                from catboost import CatBoostClassifier
                return CatBoostClassifier
            elif algorithm == "Support Vector Machine":
                from sklearn.svm import SVC
                return SVC
            elif algorithm == "k-Nearest Neighbors":
                from sklearn.neighbors import KNeighborsClassifier
                return KNeighborsClassifier
            elif algorithm == "Gaussian Process":
                from sklearn.gaussian_process import GaussianProcessClassifier
                return GaussianProcessClassifier
            elif algorithm == "Ridge Regression":
                from sklearn.linear_model import RidgeClassifier
                return RidgeClassifier
        raise ValueError(f"Unknown algorithm: {algorithm}")
