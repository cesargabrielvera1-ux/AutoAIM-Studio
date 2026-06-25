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
                'subsample': (0.6, 1.0, 'float'),
            },
            "XGBoost": {
                'n_estimators': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'max_depth': (3, 10, 'int'),
                'min_child_weight': (1, 10, 'int'),
                'subsample': (0.6, 1.0, 'float'),
                'colsample_bytree': (0.6, 1.0, 'float'),
                'gamma': (0.0, 0.5, 'float'),
            },
            "LightGBM": {
                'n_estimators': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'num_leaves': (20, 150, 'int'),
                'max_depth': (-1, 15, 'int'),
                'min_child_samples': (5, 50, 'int'),
                'subsample': (0.6, 1.0, 'float'),
                'colsample_bytree': (0.6, 1.0, 'float'),
            },
            "CatBoost": {
                'iterations': (50, 500, 'int'),
                'learning_rate': (0.01, 0.3, 'float_log'),
                'depth': (3, 10, 'int'),
                'l2_leaf_reg': (1, 10, 'float_log'),
            },
            "Support Vector Machine": {
                'C': (0.1, 100.0, 'float_log'),
                'gamma': (1e-4, 1.0, 'float_log'),
                'kernel': (['rbf', 'poly', 'sigmoid'], None, 'categorical'),
            },
            "k-Nearest Neighbors": {
                'n_neighbors': (3, 20, 'int'),
                'weights': (['uniform', 'distance'], None, 'categorical'),
                'p': ([1, 2], None, 'categorical'),
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
        pruning: bool = True
    ) -> OptimizationResult:
        """Optimize scikit-learn model hyperparameters.
        
        Args:
            model_type: Type of model
            algorithm: Algorithm name
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            problem_type: 'regression' or 'classification'
            param_ranges: Custom parameter ranges (optional)
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds
            timeout: Optimization timeout in seconds
            progress_callback: Callback(trial_num, score, params)
            random_state: Random seed
            pruning: Whether to use pruning (stop unpromising trials)
            
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
            
            # Sample parameters
            params = {}
            for param_name, (min_val, max_val, param_type) in param_ranges.items():
                params[param_name] = self._suggest_param(trial, param_name, min_val, max_val, param_type)
            
            # Create and evaluate model
            # NOTE: SVR/SVC and KNeighbors do NOT accept random_state
            if algorithm in ('Support Vector Machine', 'k-Nearest Neighbors'):
                model = model_class(**params)
            else:
                model = model_class(**params, random_state=random_state)
            
            # Cross-validation with configurable folds
            scoring = 'r2' if problem_type == 'regression' else 'accuracy'
            
            try:
                cv_scores = cross_val_score(model, X_train, y_train,
                                           cv=cv_folds, scoring=scoring, n_jobs=1)
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
        
        # Train best model
        best_params = study.best_params
        # NOTE: SVR/SVC and KNeighbors do NOT accept random_state
        if algorithm in ('Support Vector Machine', 'k-Nearest Neighbors'):
            best_model = model_class(**best_params)
        else:
            best_model = model_class(**best_params, random_state=random_state)
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
            cv_metrics=cv_metrics
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
            if np.isnan(X_train).any() or np.isnan(y_train).any():
                nan_cols = []
                raise ValueError(
                    f"Training data contains NaN values. "
                    f"Please clean your dataset before optimizing."
                )
            if cv_folds <= 1 and (np.isnan(X_val).any() or np.isnan(y_val).any()):
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
        
        # Check if ALL trials failed
        if study.best_value == float('-inf') or study.best_value is None:
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
            cv_metrics=cv_metrics
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
        else:
            if algorithm == "Random Forest":
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier
            elif algorithm == "Gradient Boosting":
                from sklearn.ensemble import GradientBoostingClassifier
                return GradientBoostingClassifier
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
        
        raise ValueError(f"Unknown algorithm: {algorithm}")
