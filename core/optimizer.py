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
        random_state: int = 42
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
            
        Returns:
            OptimizationResult
        """
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
            }
        
        # Helper function to sample with ranges
        def _suggest_with_ranges(trial, param_name, ranges):
            min_val, max_val, ptype = ranges[param_name]
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
        
        def objective(trial):
            # Sample architecture parameters
            n_layers = _suggest_with_ranges(trial, 'n_layers', param_ranges)
            
            hidden_layers = []
            for i in range(n_layers):
                n_units = _suggest_with_ranges(trial, 'n_units', param_ranges)
                dropout = _suggest_with_ranges(trial, 'dropout', param_ranges)
                activation = _suggest_with_ranges(trial, 'activation', param_ranges)
                use_bn = _suggest_with_ranges(trial, 'batch_norm', param_ranges)
                
                hidden_layers.append({
                    'n_units': n_units,
                    'activation': activation,
                    'dropout_rate': dropout,
                    'use_batch_norm': use_bn,
                    'use_layer_norm': False
                })
            
            # Training parameters
            learning_rate = _suggest_with_ranges(trial, 'learning_rate', param_ranges)
            batch_size = _suggest_with_ranges(trial, 'batch_size', param_ranges)
            weight_decay = _suggest_with_ranges(trial, 'weight_decay', param_ranges)
            optimizer_name = _suggest_with_ranges(trial, 'optimizer', param_ranges)
            
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
                    epochs=200,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    optimizer=optimizer_name,
                    weight_decay=weight_decay,
                    early_stopping_patience=30
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
                            epochs=100,
                            batch_size=batch_size,
                            learning_rate=learning_rate,
                            optimizer=optimizer_name,
                            weight_decay=weight_decay,
                            early_stopping_patience=20
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
                self.logger.error(f"Trial {trial.number} failed: {e}")
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
