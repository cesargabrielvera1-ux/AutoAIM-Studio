"""Ensemble training module."""

import time
import warnings
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

from ..utils.logger import LoggerMixin


@dataclass
class EnsembleResult:
    """Result of ensemble training."""
    ensemble_type: str
    model_names: List[str]
    weights: Dict[str, float]
    metrics: Dict[str, float]
    cv_scores: List[float] = field(default_factory=list)
    training_time: float = 0.0
    problem_type: str = 'regression'
    predictions: Optional[np.ndarray] = None
    true_values: Optional[np.ndarray] = None
    optimized_weights: Optional[Dict[str, float]] = None
    optimization_trials: int = 0
    base_models: Optional[Dict[str, Any]] = None  # v1.3.0: Store base models for prediction
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions using stored base models and weights.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Ensemble predictions (n_samples,)
        """
        if self.base_models is None:
            raise RuntimeError(
                "This ensemble was loaded from a saved bundle and its base models "
                "are not available. Re-train the ensemble in the current session "
                "to enable predictions, or use the individual base models directly."
            )
        
        # Get predictions from each available base model
        model_predictions = {}
        for name in self.model_names:
            if name in self.base_models:
                model = self.base_models[name]
                try:
                    model_predictions[name] = model.predict(X)
                except Exception as e:
                    raise RuntimeError(f"Failed to predict with base model '{name}': {e}")
        
        if not model_predictions:
            raise RuntimeError("No base models available for prediction")
        
        # Weighted average
        if self.ensemble_type == 'weighted_average':
            total_weight = 0.0
            weighted_sum = np.zeros(len(X))
            for name, preds in model_predictions.items():
                w = self.weights.get(name, 1.0 / len(model_predictions))
                weight = w / sum(self.weights.values())
                weighted_sum += weight * preds
                total_weight += weight
            return weighted_sum
        
        # Stacking (simple average of available models if meta-learner not stored)
        preds_array = np.array(list(model_predictions.values()))
        return np.mean(preds_array, axis=0)


class EnsembleTrainer(LoggerMixin):
    """Train ensemble models."""
    
    def __init__(self, trainer):
        """Initialize ensemble trainer.
        
        Args:
            trainer: ModelTrainer instance with trained models
        """
        self.trainer = trainer
        self.results: Dict[str, EnsembleResult] = {}
    
    def train_ensemble(
        self,
        ensemble_type: str,
        model_names: List[str],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        cv_folds: int = 5,
        random_state: int = 42,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        weights: Optional[Dict[str, float]] = None
    ) -> EnsembleResult:
        """Train an ensemble of models.
        
        Args:
            ensemble_type: Type of ensemble ('weighted_average', 'stacking')
            model_names: List of trained model names to include
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            cv_folds: Number of CV folds for ensemble evaluation
            progress_callback: Callback for progress updates
            
        Returns:
            EnsembleResult with ensemble metrics
        """
        start_time = time.time()
        
        if progress_callback:
            progress_callback("Collecting model predictions...", 0.1)
        
        # Get predictions from each model
        predictions = {}
        train_predictions = {}
        
        for name in model_names:
            if name not in self.trainer.results:
                self.logger.warning(f"Model '{name}' not found, skipping")
                continue
            
            result = self.trainer.results[name]
            
            # Skip ensemble models (cannot use as base)
            if isinstance(result.model, EnsembleResult):
                self.logger.warning(f"Skipping ensemble model '{name}' - cannot use as base model")
                continue
            
            if not hasattr(result, 'model') or result.model is None:
                self.logger.warning(f"Model '{name}' has no valid model object, skipping")
                continue
            
            try:
                predictions[name] = result.model.predict(X_test) if X_test is not None else None
                train_predictions[name] = result.model.predict(X_train)
            except Exception as e:
                self.logger.warning(f"Could not get predictions from '{name}': {e}, skipping")
        
        if not predictions:
            raise ValueError("No valid models found for ensemble. Ensure you have trained at least 2 non-ensemble models.")
        
        if progress_callback:
            progress_callback(f"Computing ensemble with {len(predictions)} models...", 0.3)
        
        # Calculate weights (custom > equal default)
        if weights is not None:
            # Validate and normalize custom weights
            valid_weights = {name: weights.get(name, 1.0) for name in predictions}
            total = sum(valid_weights.values())
            if total > 0:
                valid_weights = {name: w / total for name, w in valid_weights.items()}
            else:
                valid_weights = {name: 1.0 / len(predictions) for name in predictions}
            weights = valid_weights
        else:
            weights = {name: 1.0 / len(predictions) for name in predictions}
        
        # Compute ensemble predictions
        if ensemble_type == 'weighted_average':
            ensemble_pred = self._weighted_average(predictions, weights)
            train_ensemble_pred = self._weighted_average(train_predictions, weights)
        elif ensemble_type == 'stacking':
            ensemble_pred = self._stacking(train_predictions, train_predictions, y_train, predictions)
            train_ensemble_pred = self._stacking(train_predictions, train_predictions, y_train, train_predictions)
        else:
            raise ValueError(f"Unknown ensemble type: {ensemble_type}")
        
        if progress_callback:
            progress_callback("Computing metrics...", 0.7)
        
        # Compute metrics
        problem_type = self.trainer.results[model_names[0]].problem_type
        metrics = {}
        
        if y_test is not None and ensemble_pred is not None:
            # Flatten predictions if needed
            if ensemble_pred.ndim > 1 and ensemble_pred.shape[1] == 1:
                ensemble_pred = ensemble_pred.flatten()
            
            if problem_type == 'regression':
                metrics['r2'] = r2_score(y_test, ensemble_pred)
                metrics['rmse'] = np.sqrt(mean_squared_error(y_test, ensemble_pred))
            else:
                metrics['accuracy'] = accuracy_score(y_test, ensemble_pred)
        
        # Cross-validation for ensemble
        cv_scores = []
        if cv_folds > 1 and len(X_train) >= cv_folds * 2:
            if progress_callback:
                progress_callback(f"Running {cv_folds}-fold CV...", 0.8)
            
            try:
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                
                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    # Get fold predictions from each model
                    fold_predictions = {}
                    fold_train_preds = {}
                    
                    for name in model_names:
                        if name not in self.trainer.results:
                            continue
                        result = self.trainer.results[name]
                        
                        if result.is_neural_network:
                            fold_train_preds[name] = result.model.predict(X_fold_train)
                            fold_predictions[name] = result.model.predict(X_fold_val)
                        else:
                            fold_train_preds[name] = result.model.predict(X_fold_train)
                            fold_predictions[name] = result.model.predict(X_fold_val)
                    
                    # Compute ensemble prediction for this fold
                    if ensemble_type == 'weighted_average':
                        fold_weights = {name: 1.0 / len(fold_predictions) for name in fold_predictions}
                        fold_ensemble_pred = self._weighted_average(fold_predictions, fold_weights)
                    else:
                        fold_ensemble_pred = self._stacking(
                            fold_train_preds, fold_train_preds, y_fold_train, fold_predictions
                        )
                    
                    if fold_ensemble_pred is not None:
                        if fold_ensemble_pred.ndim > 1 and fold_ensemble_pred.shape[1] == 1:
                            fold_ensemble_pred = fold_ensemble_pred.flatten()
                        
                        if problem_type == 'regression':
                            cv_scores.append(r2_score(y_fold_val, fold_ensemble_pred))
                        else:
                            cv_scores.append(accuracy_score(y_fold_val, fold_ensemble_pred))
            except Exception as e:
                self.logger.error(f"Ensemble CV failed: {e}")
        
        if progress_callback:
            progress_callback("Done!", 1.0)
        
        training_time = time.time() - start_time
        
        # v1.3.0: Store base models for later prediction
        base_models = {}
        for name in predictions.keys():
            if name in self.trainer.results:
                base_models[name] = self.trainer.results[name].model
        
        result = EnsembleResult(
            ensemble_type=ensemble_type,
            model_names=list(predictions.keys()),
            weights=weights,
            metrics=metrics,
            cv_scores=cv_scores,
            training_time=training_time,
            problem_type=problem_type,
            predictions=ensemble_pred,
            true_values=y_test,
            base_models=base_models
        )
        
        ensemble_name = f"Ensemble_{ensemble_type}_{len(self.results) + 1}"
        self.results[ensemble_name] = result
        
        self.logger.info(
            f"Ensemble '{ensemble_name}' trained in {training_time:.2f}s. "
            f"CV Score: {np.mean(cv_scores):.4f}" if cv_scores else ""
        )
        
        return result
    
    def optimize_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_names: Optional[List[str]] = None,
        n_trials: int = 50,
        cv_folds: int = 5,
        random_state: int = 42,
        progress_callback: Optional[Callable[[int, float, Dict], None]] = None
    ) -> EnsembleResult:
        """Optimize ensemble weights using Bayesian optimization.

        Args:
            X: Features (used for both optimization and final evaluation)
            y: Targets
            model_names: List of model names to include. If None, uses all non-ensemble models.
            n_trials: Number of optimization trials
            cv_folds: Number of CV folds
            progress_callback: Callback(trial_num, score, weights)

        Returns:
            EnsembleResult with optimized weights and computed metrics
        """
        import time
        start_time = time.time()

        try:
            import optuna
        except ImportError:
            self.logger.warning("Optuna not available, using equal weights")
            names = model_names if model_names else list(self.trainer.results.keys())
            equal_weights = {name: 1.0 / len(names) for name in names}
            return EnsembleResult(
                ensemble_type='weighted_average',
                model_names=names,
                weights=equal_weights,
                metrics={},
                training_time=0.0,
                problem_type='regression'
            )

        # Use provided model_names, or default to all non-ensemble models
        if model_names is None:
            model_names = [
                name for name, result in self.trainer.results.items()
                if not isinstance(result, EnsembleResult) and hasattr(result, 'model') and result.model is not None
            ]
        
        # Filter to only valid models that exist in trainer.results
        valid_names = []
        model_preds = {}

        for name in model_names:
            if name not in self.trainer.results:
                self.logger.warning(f"Model '{name}' not found in trainer results, skipping")
                continue
            result = self.trainer.results[name]
            if not hasattr(result, 'model') or result.model is None:
                self.logger.warning(f"Model '{name}' has no valid model object, skipping")
                continue
            # Skip if the model itself is an EnsembleResult (no .predict method)
            if isinstance(result.model, EnsembleResult):
                self.logger.warning(f"Skipping ensemble model '{name}' - cannot use as base model")
                continue
            try:
                preds = result.model.predict(X)
                valid_names.append(name)
                model_preds[name] = preds
            except Exception as e:
                self.logger.warning(f"Could not get predictions from '{name}': {e}, skipping")

        model_names = valid_names

        if len(model_names) < 2:
            self.logger.warning("Need at least 2 valid models for weight optimization")
            return EnsembleResult(
                ensemble_type='weighted_average',
                model_names=model_names,
                weights={name: 1.0 for name in model_names},
                metrics={},
                training_time=0.0,
                problem_type='regression'
            )

        problem_type = self.trainer.results[model_names[0]].problem_type

        # Store CV scores from the best trial
        best_cv_scores = []

        def objective(trial):
            nonlocal best_cv_scores
            # Sample weights for each model (must sum to 1)
            raw_weights = []
            for name in model_names:
                w = trial.suggest_float(f'weight_{name}', 0.0, 1.0)
                raw_weights.append(w)

            # Normalize weights
            total = sum(raw_weights)
            if total > 0:
                weights = [w / total for w in raw_weights]
            else:
                weights = [1.0 / len(model_names)] * len(model_names)

            weight_dict = dict(zip(model_names, weights))

            # Evaluate with cross-validation
            scores = []
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

            for train_idx, val_idx in kf.split(X):
                y_val = y[val_idx]

                # Combine predictions with weights for validation fold
                ensemble_pred = None
                for name, weight in weight_dict.items():
                    pred = model_preds[name][val_idx]
                    if pred.ndim > 1 and pred.shape[1] == 1:
                        pred = pred.flatten()

                    if ensemble_pred is None:
                        ensemble_pred = weight * pred
                    else:
                        ensemble_pred += weight * pred

                if problem_type == 'regression':
                    scores.append(r2_score(y_val, ensemble_pred))
                else:
                    scores.append(accuracy_score(y_val, ensemble_pred))

            mean_score = np.mean(scores)

            # Store CV scores for the best trial
            if trial.number == 0 or mean_score > (study.best_value if study.trials else float('-inf')):
                best_cv_scores = scores

            if progress_callback:
                progress_callback(trial.number, mean_score, weight_dict)

            return mean_score

        # Create and run study
        study = optuna.create_study(
            direction='maximize',
            sampler=optuna.samplers.TPESampler(seed=random_state)
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        # Get best weights
        best_params = study.best_params
        raw_weights = [best_params.get(f'weight_{name}', 0.5) for name in model_names]
        total = sum(raw_weights)
        if total > 0:
            best_weights = {name: w / total for name, w in zip(model_names, raw_weights)}
        else:
            best_weights = {name: 1.0 / len(model_names) for name in model_names}

        self.logger.info(f"Weight optimization complete. Best score: {study.best_value:.4f}")
        self.logger.info(f"Optimized weights: {best_weights}")

        # Compute final metrics with optimized weights on full data
        ensemble_pred = None
        for name, weight in best_weights.items():
            pred = model_preds[name]
            if pred.ndim > 1 and pred.shape[1] == 1:
                pred = pred.flatten()
            if ensemble_pred is None:
                ensemble_pred = weight * pred
            else:
                ensemble_pred += weight * pred

        metrics = {}
        if problem_type == 'regression':
            metrics['r2'] = r2_score(y, ensemble_pred)
            metrics['rmse'] = float(np.sqrt(mean_squared_error(y, ensemble_pred)))
            metrics['cv_r2_mean'] = float(np.mean(best_cv_scores)) if best_cv_scores else 0.0
        else:
            metrics['accuracy'] = accuracy_score(y, ensemble_pred)
            metrics['cv_acc_mean'] = float(np.mean(best_cv_scores)) if best_cv_scores else 0.0

        training_time = time.time() - start_time
        
        # v1.3.0: Store base models for later prediction
        base_models = {}
        for name in model_names:
            if name in self.trainer.results:
                base_models[name] = self.trainer.results[name].model

        result = EnsembleResult(
            ensemble_type='weighted_average',
            model_names=model_names,
            weights=best_weights,
            metrics=metrics,
            cv_scores=best_cv_scores,
            training_time=training_time,
            problem_type=problem_type,
            predictions=ensemble_pred,
            true_values=y,
            optimized_weights=best_weights,
            optimization_trials=n_trials,
            base_models=base_models
        )

        ensemble_name = f"Ensemble_WeightedOptimized_{len(self.results) + 1}"
        self.results[ensemble_name] = result

        self.logger.info(
            f"Ensemble '{ensemble_name}' optimized in {training_time:.2f}s. "
            f"Best CV Score: {study.best_value:.4f}"
        )

        return result
    
    def _weighted_average(
        self,
        predictions: Dict[str, np.ndarray],
        weights: Dict[str, float]
    ) -> Optional[np.ndarray]:
        """Compute weighted average of predictions.
        
        Args:
            predictions: Dictionary of model predictions
            weights: Dictionary of model weights
            
        Returns:
            Weighted average predictions
        """
        if not predictions:
            return None
        
        result = None
        total_weight = 0.0
        
        for name, pred in predictions.items():
            if pred is None:
                continue
            
            weight = weights.get(name, 1.0)
            
            if pred.ndim > 1 and pred.shape[1] == 1:
                pred = pred.flatten()
            
            if result is None:
                result = weight * pred
            else:
                result += weight * pred
            
            total_weight += weight
        
        if result is not None and total_weight > 0:
            result /= total_weight
        
        return result
    
    def _stacking(
        self,
        train_predictions: Dict[str, np.ndarray],
        meta_train_features: Dict[str, np.ndarray],
        y_train: np.ndarray,
        test_predictions: Dict[str, np.ndarray]
    ) -> Optional[np.ndarray]:
        """Stacking ensemble using a meta-learner.
        
        Args:
            train_predictions: Predictions on training set from base models
            meta_train_features: Features for meta-learner training
            y_train: Training targets
            test_predictions: Predictions on test set from base models
            
        Returns:
            Stacked predictions
        """
        try:
            from sklearn.linear_model import Ridge
            
            # Create meta-features from base model predictions
            meta_features = []
            for name in sorted(train_predictions.keys()):
                pred = train_predictions[name]
                if pred.ndim > 1 and pred.shape[1] == 1:
                    pred = pred.flatten()
                meta_features.append(pred)
            
            X_meta = np.column_stack(meta_features)
            
            # Train meta-learner
            meta_learner = Ridge(alpha=1.0)
            meta_learner.fit(X_meta, y_train)
            
            # Create meta-features for test set
            test_meta_features = []
            for name in sorted(test_predictions.keys()):
                pred = test_predictions[name]
                if pred.ndim > 1 and pred.shape[1] == 1:
                    pred = pred.flatten()
                test_meta_features.append(pred)
            
            X_meta_test = np.column_stack(test_meta_features)
            
            # Predict with meta-learner
            return meta_learner.predict(X_meta_test)
        except Exception as e:
            self.logger.error(f"Stacking failed: {e}")
            # Fall back to weighted average
            weights = {name: 1.0 / len(test_predictions) for name in test_predictions}
            return self._weighted_average(test_predictions, weights)
    
    def predict(self, ensemble_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with an ensemble.
        
        Args:
            ensemble_name: Name of the ensemble
            X: Input features
            
        Returns:
            Predictions
        """
        if ensemble_name not in self.results:
            raise ValueError(f"Ensemble '{ensemble_name}' not found")
        
        result = self.results[ensemble_name]
        predictions = {}
        
        for name in result.model_names:
            if name in self.trainer.results:
                model_result = self.trainer.results[name]
                if model_result.is_neural_network:
                    predictions[name] = model_result.model.predict(X)
                else:
                    predictions[name] = model_result.model.predict(X)
        
        if result.ensemble_type == 'weighted_average':
            weights = result.optimized_weights if result.optimized_weights else result.weights
            return self._weighted_average(predictions, weights)
        elif result.ensemble_type == 'stacking':
            train_preds = {}
            for name in result.model_names:
                if name in self.trainer.results:
                    model_result = self.trainer.results[name]
                    if model_result.is_neural_network:
                        train_preds[name] = model_result.model.predict(self.trainer.results[name].predictions)
                    else:
                        train_preds[name] = model_result.model.predict(self.trainer.results[name].predictions)
            return self._stacking(train_preds, train_preds, 
                                self.trainer.results[result.model_names[0]].true_values, 
                                predictions)
        
        return self._weighted_average(predictions, result.weights)
