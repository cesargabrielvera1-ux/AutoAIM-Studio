"""Model training with materials-aware validation."""

import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score, cross_validate, KFold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)

from ..utils.logger import LoggerMixin
from ..utils.hardware_detector import get_hardware_detector


@dataclass
class TrainingResult:
    """Result of model training."""
    model: BaseEstimator
    model_name: str
    problem_type: str
    metrics: Dict[str, float]
    cv_metrics: Dict[str, List[float]]
    training_time: float
    feature_importance: Optional[Dict[str, float]] = None
    predictions: Optional[np.ndarray] = None
    true_values: Optional[np.ndarray] = None
    is_neural_network: bool = False
    nn_history: Optional[Dict] = None


class ModelTrainer(LoggerMixin):
    """Train and evaluate ML models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self._hardware = get_hardware_detector()
        self._results: Dict[str, TrainingResult] = {}
        self._result_counter: Dict[str, int] = {}
    
    @property
    def results(self) -> Dict[str, TrainingResult]:
        """Get all training results."""
        return self._results
    
    def _get_unique_model_name(self, model_name: str) -> str:
        """Generate unique model name by adding counter suffix if needed."""
        if model_name not in self._results:
            return model_name
        
        counter = self._result_counter.get(model_name, 0) + 1
        self._result_counter[model_name] = counter
        
        return f"{model_name}_{counter}"
    
    def train(
        self,
        model: BaseEstimator,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        problem_type: str = 'regression',
        cv_folds: int = 5,
        feature_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> TrainingResult:
        """Train a model with cross-validation."""
        start_time = time.time()
        
        if progress_callback:
            progress_callback(f"Training {model_name}...", 0.1)
        
        unique_model_name = self._get_unique_model_name(model_name)
        
        self.logger.info(f"Training {unique_model_name}...")
        model.fit(X_train, y_train)
        
        if progress_callback:
            progress_callback(f"Running cross-validation...", 0.4)
        
        # v1.3.0 FIX: Extract random_state from model for reproducible CV
        model_rs = 42
        try:
            model_rs = model.get_params().get('random_state', 42)
        except Exception:
            pass
        
        self.logger.info(f"Starting training for {unique_model_name} with {X_train.shape[0]} samples, {X_train.shape[1]} features, model_rs={model_rs}")
        cv_metrics = self._cross_validate(model, X_train, y_train, problem_type, cv_folds, random_state=model_rs)
        self.logger.info(f"CV metrics for {unique_model_name}: {cv_metrics}")
        
        if progress_callback:
            progress_callback(f"Computing metrics...", 0.7)
        
        metrics = {}
        predictions = None
        true_values = None
        
        if X_test is not None and y_test is not None:
            predictions = model.predict(X_test)
            true_values = y_test
            metrics = self._compute_metrics(true_values, predictions, problem_type)
        
        feature_importance = self._get_feature_importance(model, feature_names)
        
        if progress_callback:
            progress_callback(f"Training complete!", 1.0)
        
        training_time = time.time() - start_time
        
        result = TrainingResult(
            model=model,
            model_name=unique_model_name,
            problem_type=problem_type,
            metrics=metrics,
            cv_metrics=cv_metrics,
            training_time=training_time,
            feature_importance=feature_importance,
            predictions=predictions,
            true_values=true_values,
            is_neural_network=False
        )
        
        self._results[unique_model_name] = result
        
        self.logger.info(
            f"Training complete for {unique_model_name} in {training_time:.2f}s. "
            f"CV Score: {np.mean(cv_metrics.get('test_score', [0])):.4f}"
        )
        
        return result
    
    def train_neural_network(
        self,
        nn_builder,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        problem_type: str = 'regression',
        cv_folds: int = 5,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ) -> TrainingResult:
        """Train a neural network with optional cross-validation."""
        start_time = time.time()
        
        if progress_callback:
            progress_callback(f"Training Neural Network...", 0.1)
        
        # Train final model with train/val split
        def nn_progress(epoch, train_loss, val_loss):
            if progress_callback:
                progress = 0.1 + 0.8 * (epoch / nn_builder._training_config.epochs)
                progress_callback(f"Epoch {epoch} - Loss: {train_loss:.4f}", progress)
        
        history = nn_builder.fit(X_train, y_train, X_test, y_test,
                                  verbose=False, progress_callback=nn_progress)
        
        if progress_callback:
            progress_callback(f"Computing metrics...", 0.9)
        
        metrics = {}
        predictions = None
        true_values = None
        
        if X_test is not None and y_test is not None:
            predictions = nn_builder.predict(X_test)
            true_values = y_test
            
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            metrics = self._compute_metrics(true_values, predictions, problem_type)
        
        # Cross-validation for neural network
        cv_metrics = {'test_score': [], 'train_score': []}
        if cv_folds > 1 and len(X_train) >= cv_folds * 2:
            if progress_callback:
                progress_callback(f"Running {cv_folds}-fold CV...", 0.92)
            try:
                # v1.3.0 FIX: Use model's random_seed for reproducible CV folds
                nn_rs = getattr(nn_builder._training_config, 'random_seed', 42)
                kf = KFold(n_splits=cv_folds, shuffle=True, random_state=nn_rs)
                fold_scores = []
                train_scores = []
                
                for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    from ..core.nn_builder import NeuralNetworkBuilder
                    fold_builder = NeuralNetworkBuilder()
                    
                    arch = nn_builder.architecture
                    hidden_layers = []
                    for layer in arch.hidden_layers:
                        hidden_layers.append({
                            'n_units': layer.n_units,
                            'activation': layer.activation.value,
                            'dropout_rate': layer.dropout_rate,
                            'use_batch_norm': layer.use_batch_norm,
                            'use_layer_norm': layer.use_layer_norm
                        })
                    
                    fold_builder.create_architecture(
                        input_dim=arch.input_dim,
                        output_dim=arch.output_dim,
                        hidden_layers_config=hidden_layers,
                        problem_type=arch.problem_type
                    )
                    fold_builder.build_model()
                    
                    tc = nn_builder._training_config
                    fold_builder.create_training_config(
                        epochs=min(tc.epochs, 100),
                        batch_size=tc.batch_size,
                        learning_rate=tc.learning_rate,
                        optimizer=tc.optimizer.value,
                        scheduler=tc.scheduler.value,
                        weight_decay=tc.weight_decay,
                        early_stopping_patience=tc.early_stopping_patience,
                        early_stopping_min_delta=tc.early_stopping_min_delta
                    )
                    
                    fold_builder.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val, verbose=False)
                    
                    fold_pred = fold_builder.predict(X_fold_val)
                    if fold_pred.ndim > 1 and fold_pred.shape[1] == 1:
                        fold_pred = fold_pred.flatten()
                    
                    if problem_type == 'regression':
                        fold_scores.append(r2_score(y_fold_val, fold_pred))
                        train_pred = fold_builder.predict(X_fold_train)
                        if train_pred.ndim > 1 and train_pred.shape[1] == 1:
                            train_pred = train_pred.flatten()
                        train_scores.append(r2_score(y_fold_train, train_pred))
                    else:
                        fold_scores.append(accuracy_score(y_fold_val, fold_pred))
                        train_pred = fold_builder.predict(X_fold_train)
                        if train_pred.ndim > 1 and train_pred.shape[1] == 1:
                            train_pred = train_pred.flatten()
                        train_scores.append(accuracy_score(y_fold_train, train_pred))
                
                cv_metrics['test_score'] = fold_scores
                cv_metrics['train_score'] = train_scores
                
                self.logger.info(f"NN CV completed. Test scores: {fold_scores}")
            except Exception as e:
                self.logger.error(f"NN cross-validation failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                cv_metrics = {'test_score': [metrics.get('r2', metrics.get('accuracy', 0))], 'train_score': []}
        else:
            if metrics:
                cv_metrics['test_score'] = [metrics.get('r2', metrics.get('accuracy', 0))]
        
        training_time = time.time() - start_time
        
        # v1.3.0 FIX: Use _get_unique_model_name() so multiple NNs don't overwrite each other.
        # Same mechanism as train() for sklearn models.
        unique_name = self._get_unique_model_name(model_name)
        
        result = TrainingResult(
            model=nn_builder,
            model_name=unique_name,
            problem_type=problem_type,
            metrics=metrics,
            cv_metrics=cv_metrics,
            training_time=training_time,
            feature_importance=None,
            predictions=predictions,
            true_values=true_values,
            is_neural_network=True,
            nn_history=history
        )
        
        self._results[unique_name] = result
        
        self.logger.info(
            f"Neural Network training complete in {training_time:.2f}s. "
            f"CV Score: {np.mean(cv_metrics.get('test_score', [0])):.4f}"
        )
        
        return result
    
    def _cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
        cv_folds: int,
        random_state: int = 42
    ) -> Dict[str, List[float]]:
        """Perform cross-validation with explicit KFold for reproducible random_state."""
        scoring = 'r2' if problem_type == 'regression' else 'accuracy'
        
        try:
            self.logger.info(f"Starting CV with {cv_folds} folds, scoring={scoring}, "
                           f"random_state={random_state}, X.shape={X.shape}")
            
            # v1.3.0 FIX: Create explicit KFold with the model's random_state
            # instead of passing an integer to cross_validate (which creates
            # a non-shuffled KFold without random_state control).
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            cv_results = cross_validate(
                model, X, y,
                cv=kf,
                scoring=scoring,
                return_train_score=True,
                n_jobs=1
            )
            
            self.logger.info(f"CV completed successfully. Test scores: {cv_results['test_score']}")
            
            return {
                'test_score': cv_results['test_score'].tolist(),
                'train_score': cv_results['train_score'].tolist(),
                'fit_time': cv_results['fit_time'].tolist(),
            }
        except Exception as e:
            self.logger.error(f"Cross-validation failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return {
                'test_score': [],
                'train_score': [],
                'fit_time': [],
                'error': str(e)
            }
    
    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        problem_type: str
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {}
        
        if problem_type == 'regression':
            mse = mean_squared_error(y_true, y_pred)
            metrics['mse'] = mse
            metrics['rmse'] = np.sqrt(mse)
            metrics['mae'] = mean_absolute_error(y_true, y_pred)
            metrics['r2'] = r2_score(y_true, y_pred)
            
            y_std = np.std(y_true)
            if y_std > 0:
                metrics['rmse_normalized'] = metrics['rmse'] / y_std
            else:
                metrics['rmse_normalized'] = 0.0
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                if np.isfinite(mape):
                    metrics['mape'] = mape
        else:
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            
            try:
                metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
                metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            except Exception:
                pass
        
        return metrics
    
    def _get_feature_importance(
        self,
        model: BaseEstimator,
        feature_names: Optional[List[str]]
    ) -> Optional[Dict[str, float]]:
        """Extract feature importance from model."""
        importance = None
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            importance = np.abs(coef)
        
        if importance is not None and feature_names is not None:
            if len(importance) == len(feature_names):
                return dict(zip(feature_names, importance.tolist()))
        
        return None
    
    def get_best_model(self, metric: str = 'r2') -> Optional[str]:
        """Get the best performing model."""
        if not self._results:
            return None
        
        best_model = None
        best_score = float('-inf')
        
        for name, result in self._results.items():
            score = result.metrics.get(metric)
            if score is not None and score > best_score:
                best_score = score
                best_model = name
        
        return best_model
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all trained models."""
        if not self._results:
            return pd.DataFrame()
        
        comparisons = []
        
        for name, result in self._results.items():
            row = {
                'Model': name,
                'Training Time (s)': f"{result.training_time:.2f}",
            }
            
            for metric, value in result.metrics.items():
                row[metric.upper()] = f"{value:.4f}"
            
            if result.cv_metrics.get('test_score'):
                cv_mean = np.mean(result.cv_metrics['test_score'])
                cv_std = np.std(result.cv_metrics['test_score'])
                row['CV Score'] = f"{cv_mean:.4f} ± {cv_std:.4f}"
            
            comparisons.append(row)
        
        return pd.DataFrame(comparisons)
    
    def predict(self, model_name: str, X: np.ndarray) -> np.ndarray:
        """Make predictions with a trained model."""
        if model_name not in self._results:
            raise ValueError(f"Model '{model_name}' not found")
        
        result = self._results[model_name]
        
        if result.is_neural_network:
            return result.model.predict(X)
        else:
            return result.model.predict(X)
    
    def save_model(self, model_name: str, path: str) -> None:
        """Save a trained model."""
        if model_name not in self._results:
            raise ValueError(f"Model '{model_name}' not found")
        
        result = self._results[model_name]
        
        if result.is_neural_network:
            result.model.save_model(path)
        else:
            import joblib
            joblib.dump(result.model, path)
        
        self.logger.info(f"Model '{model_name}' saved to {path}")
    
    def save_model_bundle(
        self,
        model_name: str,
        output_path: str,
        data_manager: Any
    ) -> Path:
        """Save model in unified bundle format."""
        from .model_saver import save_training_result
        
        if model_name not in self._results:
            raise ValueError(f"Model '{model_name}' not found")
        
        result = self._results[model_name]
        
        return save_training_result(
            result=result,
            output_path=output_path,
            data_manager=data_manager,
            model_name=model_name
        )
