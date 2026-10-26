"""Model training with materials-aware validation."""

import gc
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, cross_validate, KFold, learning_curve
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score
)

from ..utils.logger import LoggerMixin
from ..utils.hardware_detector import get_hardware_detector
from .cv_factory import CVConfig, build_cv_plan, assemble_cv_metrics
from .ood_detector import OodDetector
from .model_cache import get_model_cache


def _extract_model_params(model) -> Dict[str, Any]:
    """Extract all relevant parameters from a model for reproducibility.

    v2.0: Duplicated from session_exporter to avoid circular imports.
    """
    if model is None:
        return {}
    try:
        params = model.get_params()
        # Filter out None values and internal sklearn objects
        return {k: v for k, v in sorted(params.items())
                if v is not None and not callable(v) and not hasattr(v, 'fit')}
    except Exception:
        return {}


@dataclass
class TrainingResult:
    """Result of model training.
    
    v1.5.0: Added external_metrics for dual reporting when external
    validation set is loaded.
    v1.5.0: Added learning_curve_data and confidence_interval_params
    for learning curves and prediction confidence intervals.
    """
    model: BaseEstimator
    model_name: str
    problem_type: str
    metrics: Dict[str, float]
    cv_metrics: Dict[str, List[float]]
    training_time: float
    feature_importance: Optional[Dict[str, float]] = None
    # v2.0: Gaussian Process native uncertainty (standard deviation of posterior)
    predictions: Optional[np.ndarray] = None
    prediction_std: Optional[np.ndarray] = None
    # v2.0: GP-specific metadata (kernel, alpha, log_marginal_likelihood)
    # Populated only for GaussianProcessRegressor/Classifier models.
    gp_metadata: Optional[Dict[str, Any]] = None
    true_values: Optional[np.ndarray] = None
    is_neural_network: bool = False
    nn_history: Optional[Dict] = None
    # v1.5.0: External validation metrics (separate from internal test split)
    external_metrics: Optional[Dict[str, float]] = None
    has_external_validation: bool = False
    # v1.5.0: Learning curve data (train_sizes, train_scores, test_scores)
    learning_curve_data: Optional[Dict[str, Any]] = None
    # v1.5.0: Confidence interval parameters (cv_residuals_std for regression)
    confidence_interval_params: Optional[Dict[str, Any]] = None
    # v2.0: Training configuration (optimizer, epochs, batch_size, etc.)
    # Stored at training time for complete reproducibility in exports.
    training_config: Optional[Dict[str, Any]] = None
    # v2.0: Random state used for training / weight initialization.
    # Guaranteed non-None — defaults to 42 if not explicitly set.
    random_state: int = 42
    # v2.0: NN-only fields — stored at training time to avoid dependency
    # on the live nn_builder object during export.
    nn_architecture_desc: Optional[str] = None
    nn_architecture_detail: Optional[Dict[str, Any]] = None
    nn_epochs_trained: int = 0
    # v2.0: Cross-validation strategy used during training
    cv_strategy: str = 'kfold'
    # v2.0: Full CV configuration (strategy, folds, bins, group source, ...)
    # serialized for the session exporter and Details popups.
    cv_config_dict: Optional[Dict[str, Any]] = None
    # v2.0: Out-of-distribution detector trained on X_train for applicability
    # domain warnings during inference.  None for legacy bundles.
    ood_detector: Optional[Any] = None
    # v2.0: Disk-cache fields — when a model is cached to disk to free RAM,
    # ``model`` is set to None and these fields point to the cache location.
    model_cache_path: Optional[str] = None
    is_cached: bool = False


class ModelTrainer(LoggerMixin):
    """Train and evaluate ML models."""
    
    def __init__(self):
        """Initialize model trainer."""
        self._hardware = get_hardware_detector()
        self._results: Dict[str, TrainingResult] = {}
        self._result_counter: Dict[str, int] = {}
        # v2.0: sklearn CV uses n_jobs=1 because individual models already
        # have their own internal parallelism (XGB n_jobs, RF n_jobs, etc.).
        # Using n_jobs > 1 here would cause nested parallelism / oversubscription.
        self._cv_n_jobs = 1
    
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
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[np.ndarray] = None,
        problem_type: str = 'regression',
        cv_folds: int = 5,
        feature_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        # v1.5.0: External validation set (separate from internal test split)
        X_external_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_external_val: Optional[np.ndarray] = None,
        # v2.0: Cross-validation strategy
        cv_strategy: str = 'kfold',
        # v2.0: Full CV configuration (takes precedence over cv_strategy when
        # provided). Carries strategy + folds + bins + group source + resolved
        # group labels. Enables stratified / group / LOGO cross-validation.
        cv_config: Optional[CVConfig] = None,
        cv_groups: Optional[np.ndarray] = None,
        # v2.0: Preprocessor for sklearn Pipeline — eliminates data leakage by
        # fitting the scaler INSIDE each CV fold. When provided, the returned
        # model is a Pipeline([preprocessor, model]) that accepts raw DataFrames.
        preprocessor: Optional[Any] = None,
    ) -> TrainingResult:
        """Train a model with cross-validation.

        v1.5.0: Added X_external_val/y_external_val for dual metrics reporting.
        When external validation data is provided, metrics on X_test/y_test are
        reported as "Internal Test" and metrics on X_external_val/y_external_val
        are reported as "External Validation".

        v2.0: Added cv_strategy ('kfold' or 'loo') for small datasets.
        v2.0: Added cv_config for stratified / group / LOGO cross-validation.
        """
        start_time = time.time()

        if progress_callback:
            progress_callback(f"Training {model_name}...", 0.1)

        unique_model_name = self._get_unique_model_name(model_name)

        # v2.0: Build sklearn Pipeline when a preprocessor is provided. This is
        # the ONLY way to eliminate data leakage: the scaler is refitted inside
        # each CV fold, so validation data never leaks into training statistics.
        if preprocessor is not None:
            estimator = Pipeline([
                ('preprocessor', preprocessor),
                ('model', model),
            ])
            self.logger.info(
                f"Training {unique_model_name} with sklearn Pipeline "
                f"(preprocessor + model) — zero data leakage in CV."
            )
        else:
            estimator = model
            self.logger.info(
                f"Training {unique_model_name} WITHOUT Pipeline — "
                f"WARNING: data leakage possible if data was preprocessed "
                f"before the train/test split."
            )

        estimator.fit(X_train, y_train)

        if progress_callback:
            progress_callback(f"Running cross-validation...", 0.4)

        # v1.3.0 FIX: Extract random_state from model for reproducible CV
        model_rs = 42
        try:
            model_rs = model.get_params().get('random_state', 42)
        except Exception:
            pass

        # v2.0: Resolve the effective CVConfig. A full cv_config takes
        # precedence; otherwise build one from the legacy cv_strategy string.
        if cv_config is None:
            cv_config = CVConfig(
                strategy=('loo' if cv_strategy == 'loo' else 'kfold'),
                n_folds=cv_folds,
                random_state=model_rs if model_rs is not None else 42,
            )
        if cv_groups is not None:
            cv_config.groups = np.asarray(cv_groups)
        # Keep the human-readable strategy string in sync for downstream code.
        cv_strategy = cv_config.strategy

        self.logger.info(f"Starting training for {unique_model_name} with {X_train.shape[0]} samples, {X_train.shape[1]} features, model_rs={model_rs}, cv_strategy={cv_strategy}")
        cv_metrics = self._cross_validate(estimator, X_train, y_train, problem_type, cv_folds, random_state=model_rs, cv_strategy=cv_strategy, cv_config=cv_config)
        self.logger.info(f"CV metrics for {unique_model_name}: {cv_metrics}")
        
        if progress_callback:
            progress_callback(f"Computing metrics...", 0.7)
        
        # Internal test split metrics (always computed when X_test/y_test provided)
        metrics = {}
        predictions = None
        true_values = None
        prediction_std = None

        if X_test is not None and y_test is not None:
            # v2.0: Gaussian Process has native uncertainty (return_std).
            # Pipeline.predict() forwards **kwargs to the final estimator (sklearn
            # 1.0+), so estimator.predict(X_test, return_std=True) works when the
            # underlying model is GaussianProcessRegressor.
            is_gp = (
                'GaussianProcessRegressor' in type(model).__name__
                if preprocessor is None
                else 'GaussianProcessRegressor' in type(estimator.named_steps['model']).__name__
            )
            if is_gp:
                predictions, prediction_std = estimator.predict(X_test, return_std=True)
            else:
                predictions = estimator.predict(X_test)
            true_values = y_test
            metrics = self._compute_metrics(true_values, predictions, problem_type)

        # v1.5.0: External validation metrics (separate from internal test)
        external_metrics = {}
        has_external_validation = False
        if X_external_val is not None and y_external_val is not None:
            ext_predictions = estimator.predict(X_external_val)
            external_metrics = self._compute_metrics(y_external_val, ext_predictions, problem_type)
            has_external_validation = True
            self.logger.info(
                f"External validation metrics for {unique_model_name}: "
                f"{ {k: f'{v:.4f}' for k, v in external_metrics.items()} }"
            )

        # v2.0: Feature importance — when Pipeline is used, the real model is
        # inside the last step; _get_feature_importance handles that.
        feature_importance = self._get_feature_importance(estimator, feature_names)

        # v1.5.0: Estimate confidence interval parameters from CV residuals
        ci_params = self._estimate_cv_residuals(
            estimator, X_train, y_train, problem_type, cv_folds=cv_folds, random_state=model_rs, cv_strategy=cv_strategy, cv_config=cv_config
        )

        if progress_callback:
            progress_callback(f"Training complete!", 1.0)

        training_time = time.time() - start_time

        # v2.0: Extract model params as training_config for reproducibility
        sklearn_config = _extract_model_params(model)
        sklearn_config['cv_folds'] = cv_folds
        sklearn_config['cv_strategy'] = cv_strategy
        sklearn_config['problem_type'] = problem_type

        # v2.0: Serialize the full CV configuration for export / Details popups
        # (single source of truth: CVConfig.to_dict, includes the reserved bin).
        cv_config_dict = cv_config.to_dict()

        # v2.0: Capture GP metadata when the underlying model is GaussianProcess.
        # v2.1: Extended to capture kernel family and fitted hyperparameters for
        # full model provenance (scientific reproducibility).
        gp_metadata = None
        _gp_check_model = model
        if hasattr(estimator, 'named_steps') and estimator.named_steps:
            _gp_check_model = estimator.named_steps['model']
        if 'GaussianProcess' in type(_gp_check_model).__name__:
            try:
                kernel_obj = getattr(_gp_check_model, 'kernel_', getattr(_gp_check_model, 'kernel', None))
                kernel_str = str(kernel_obj) if kernel_obj is not None else 'unknown'
                # v2.1: Extract kernel family and fitted hyperparameters
                kernel_family = 'unknown'
                kernel_ls = None
                kernel_nu = None
                kernel_alpha = None
                kernel_noise = None
                if kernel_obj is not None:
                    kname = type(kernel_obj).__name__
                    if 'RBF' in kname:
                        kernel_family = 'RBF'
                        if hasattr(kernel_obj, 'length_scale'):
                            kernel_ls = float(kernel_obj.length_scale)
                    elif 'Matern' in kname:
                        kernel_family = 'Matern'
                        if hasattr(kernel_obj, 'length_scale'):
                            kernel_ls = float(kernel_obj.length_scale)
                        if hasattr(kernel_obj, 'nu'):
                            kernel_nu = float(kernel_obj.nu)
                    elif 'RationalQuadratic' in kname:
                        kernel_family = 'RationalQuadratic'
                        if hasattr(kernel_obj, 'length_scale'):
                            kernel_ls = float(kernel_obj.length_scale)
                        if hasattr(kernel_obj, 'alpha'):
                            kernel_alpha = float(kernel_obj.alpha)
                    elif 'WhiteKernel' in kname:
                        kernel_family = 'WhiteKernel'
                        if hasattr(kernel_obj, 'noise_level'):
                            kernel_noise = float(kernel_obj.noise_level)
                    elif 'Sum' in kname or '+' in kernel_str:
                        kernel_family = 'Composite'

                gp_metadata = {
                    'kernel_str': kernel_str,
                    'kernel_family': kernel_family,
                    'kernel_length_scale': kernel_ls,
                    'kernel_nu': kernel_nu,
                    'kernel_alpha': kernel_alpha,
                    'kernel_noise_level': kernel_noise,
                    'alpha': float(getattr(_gp_check_model, 'alpha', 1e-10)),
                    'log_marginal_likelihood_value': float(_gp_check_model.log_marginal_likelihood_value_) if hasattr(_gp_check_model, 'log_marginal_likelihood_value_') else None,
                    'n_restarts_optimizer': int(getattr(_gp_check_model, 'n_restarts_optimizer', 0)),
                    'normalize_y': bool(getattr(_gp_check_model, 'normalize_y', False)),
                }
            except Exception as e:
                self.logger.warning(f"Could not extract GP metadata: {e}")

        # v2.0: Fit the out-of-distribution detector on the training features.
        # This provides an applicability-domain flag during inference.
        try:
            ood_detector = OodDetector(n_neighbors=1, threshold_percentile=95.0)
            ood_detector.fit(X_train_np)
        except Exception as e:
            self.logger.warning(f"Could not fit OOD detector: {e}")
            ood_detector = None

        # v2.0: Save the Pipeline (preprocessor + model) as the model artifact.
        # This guarantees that inference receives raw DataFrames and the scaler
        # is applied internally with the statistics learned from the training data.
        result = TrainingResult(
            model=estimator,
            model_name=unique_model_name,
            problem_type=problem_type,
            metrics=metrics,
            cv_metrics=cv_metrics,
            training_time=training_time,
            feature_importance=feature_importance,
            predictions=predictions,
            prediction_std=prediction_std,
            gp_metadata=gp_metadata,
            true_values=true_values,
            is_neural_network=False,
            external_metrics=external_metrics if has_external_validation else None,
            has_external_validation=has_external_validation,
            confidence_interval_params=ci_params,
            training_config=sklearn_config,
            random_state=model_rs if model_rs is not None else 42,
            cv_strategy=cv_strategy,
            cv_config_dict=cv_config_dict,
            ood_detector=ood_detector,
        )
        
        self._results[unique_model_name] = result

        # v2.0 MEMORY FIX: Cache the heavy model to disk and free RAM.
        # The model object (Pipeline, XGB trees, GP kernel, etc.) is the
        # dominant memory consumer.  Caching it lets us keep dozens of
        # lightweight TrainingResult objects in memory while the heavy
        # estimators live on disk, loadable on demand for export/inference.
        try:
            cache = get_model_cache()
            cache_path = cache.save(estimator, unique_model_name)
            result.model_cache_path = str(cache_path)
            result.is_cached = True
            result.model = None  # free RAM — the real object is on disk
            self.logger.info(
                f"Model cached to disk: {cache_path} "
                f"({cache_path.stat().st_size / (1024 * 1024):.1f} MB)"
            )
        except Exception as e:
            self.logger.warning(f"Could not cache model to disk: {e}")

        self.logger.info(
            f"Training complete for {unique_model_name} in {training_time:.2f}s. "
            f"CV Score: {np.mean(cv_metrics.get('test_score', [0])):.4f}" if cv_metrics.get('test_score') else "CV Score: N/A"
        )

        # v2.0 MEMORY FIX: Force garbage collection after a heavy sklearn model
        # (RF, XGB, GP, etc.) so memory is freed before the next model starts.
        gc.collect()

        return result
    
    def train_neural_network(
        self,
        nn_builder,
        model_name: str,
        X_train: Union[np.ndarray, pd.DataFrame],
        y_train: np.ndarray,
        X_test: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_test: Optional[np.ndarray] = None,
        problem_type: str = 'regression',
        cv_folds: int = 5,
        progress_callback: Optional[Callable[[str, float], None]] = None,
        # v1.5.0: External validation set for dual metrics
        X_external_val: Optional[Union[np.ndarray, pd.DataFrame]] = None,
        y_external_val: Optional[np.ndarray] = None,
        # v2.0: full CV configuration + group labels (stratified/group/LOO/LOGO)
        cv_config: Optional[CVConfig] = None,
        cv_groups: Optional[np.ndarray] = None,
        # v2.0: preprocessor for manual per-fold application inside NN CV
        preprocessor: Optional[Any] = None,
    ) -> TrainingResult:
        """Train a neural network with optional cross-validation.

        v1.5.0: Added X_external_val/y_external_val for dual metrics reporting.
        v2.0: CV now goes through the shared cv_factory plan, so NNs support the
        same five strategies as sklearn models (K-Fold, Stratified, Group K-Fold,
        LOO, LOGO), including the reserved-bin point-mass stratification.

        v2.0 CRITICAL FIX: When ``preprocessor`` is provided, the final NN is
        trained on preprocessed data (fit_transform on full train), BUT the CV
        loop clones the preprocessor and refits it inside each fold so that the
        validation fold never leaks into the scaler statistics.
        """
        start_time = time.time()

        # v2.0: If a preprocessor is provided, fit it on the full training set
        # and transform train/test/external data. The NN always needs numpy arrays.
        if preprocessor is not None:
            preprocessor = clone(preprocessor)
            X_train_np = preprocessor.fit_transform(X_train)
            X_test_np = preprocessor.transform(X_test) if X_test is not None else None
            X_external_val_np = preprocessor.transform(X_external_val) if X_external_val is not None else None
            # Store the fitted preprocessor so the CV loop can clone it fresh each fold
            _preprocessor_template = preprocessor
        else:
            X_train_np = X_train.values if hasattr(X_train, 'values') else X_train
            X_test_np = X_test.values if hasattr(X_test, 'values') else X_test
            X_external_val_np = X_external_val.values if hasattr(X_external_val, 'values') else X_external_val
            _preprocessor_template = None

        # v2.0 FIX: Ensure y values are numpy arrays, not pandas Series/DataFrames.
        # torch.FloatTensor cannot infer shape from pandas objects.
        y_train = y_train.values if hasattr(y_train, 'values') else y_train
        y_test = y_test.values if hasattr(y_test, 'values') else y_test
        if y_external_val is not None:
            y_external_val = y_external_val.values if hasattr(y_external_val, 'values') else y_external_val

        if progress_callback:
            progress_callback(f"Training Neural Network...", 0.1)

        # Train final model with train/val split
        def nn_progress(epoch, train_loss, val_loss):
            if progress_callback:
                progress = 0.1 + 0.8 * (epoch / nn_builder._training_config.epochs)
                progress_callback(f"Epoch {epoch} - Loss: {train_loss:.4f}", progress)

        history = nn_builder.fit(X_train_np, y_train, X_test_np, y_test,
                                  verbose=False, progress_callback=nn_progress)
        
        if progress_callback:
            progress_callback(f"Computing metrics...", 0.9)
        
        metrics = {}
        predictions = None
        true_values = None
        
        if X_test_np is not None and y_test is not None:
            predictions = nn_builder.predict(X_test_np)
            true_values = y_test
            
            if predictions.ndim > 1 and predictions.shape[1] == 1:
                predictions = predictions.flatten()
            
            metrics = self._compute_metrics(true_values, predictions, problem_type)
        
        # v2.0: Cross-validation via the shared cv_factory plan (same five
        # strategies as sklearn: K-Fold, Stratified, Group K-Fold, LOO, LOGO).
        cv_metrics = {'test_score': [], 'train_score': []}
        # Resolve the effective CVConfig (legacy cv_folds → K-Fold if not given).
        if cv_config is None:
            cv_config = CVConfig(strategy='kfold', n_folds=cv_folds,
                                 random_state=getattr(nn_builder._training_config, 'random_seed', 42))
        if cv_groups is not None:
            cv_config.groups = np.asarray(cv_groups)

        # v2.0: Decide whether CV should run BEFORE building the plan (building
        # a fold-based splitter with n_folds<2 raises). LOO always runs; LOGO
        # needs >=3 groups; fold strategies need n_folds>1.
        strategy = cv_config.strategy
        if strategy in ('kfold', 'stratified', 'group'):
            run_cv = cv_config.n_folds > 1 and len(X_train) >= cv_config.n_folds * 2
        elif strategy == 'loo':
            run_cv = len(X_train) >= 2
        else:  # logo
            run_cv = (cv_config.groups is not None
                      and len(np.unique(cv_config.groups)) >= 3
                      and len(X_train) >= 2)

        if run_cv:
            cv_plan = build_cv_plan(cv_config, X_train, y_train, problem_type=problem_type)
            n_splits = cv_plan.n_splits_actual
            if progress_callback:
                progress_callback(f"Running {n_splits}-fold CV...", 0.92)
            try:
                # v1.3.0 FIX: Use model's random_seed for reproducible CV folds
                nn_rs = getattr(nn_builder._training_config, 'random_seed', 42)
                fold_scores = []
                train_scores = []
                fold_sizes = []
                # v2.0: collect out-of-fold predictions for OOF strategies.
                oof_preds = np.full(len(X_train), np.nan)

                # v2.0 helper: slice data safely for both DataFrames and arrays.
                def _slice_data(X, idx):
                    if hasattr(X, 'iloc'):
                        return X.iloc[idx]
                    return X[idx]

                for fold_idx, (train_idx, val_idx) in enumerate(
                        cv_plan.splitter.split(X_train, y_train, groups=cv_plan.split_labels)):
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    fold_sizes.append(len(val_idx))

                    # v2.0 CRITICAL FIX: If a preprocessor was provided, clone it
                    # and fit it ONLY on the fold's training data. This ensures
                    # the validation fold never leaks into the scaler statistics.
                    if _preprocessor_template is not None:
                        fold_preprocessor = clone(_preprocessor_template)
                        X_fold_train_raw = _slice_data(X_train, train_idx)
                        X_fold_val_raw = _slice_data(X_train, val_idx)
                        X_fold_train = fold_preprocessor.fit_transform(X_fold_train_raw)
                        X_fold_val = fold_preprocessor.transform(X_fold_val_raw)
                    else:
                        X_fold_train = _slice_data(X_train, train_idx)
                        X_fold_val = _slice_data(X_train, val_idx)

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
                    # v2.0: create_training_config BEFORE build_model so the seed
                    # is available for torch.manual_seed() during weight init.
                    tc = nn_builder._training_config
                    fold_builder.create_training_config(
                        epochs=min(tc.epochs, 100),
                        batch_size=tc.batch_size,
                        learning_rate=tc.learning_rate,
                        optimizer=tc.optimizer.value,
                        scheduler=tc.scheduler.value,
                        weight_decay=tc.weight_decay,
                        early_stopping_patience=tc.early_stopping_patience,
                        early_stopping_min_delta=tc.early_stopping_min_delta,
                        random_seed=tc.random_seed,  # Pass the seed!
                    )
                    fold_builder.build_model()

                    fold_builder.fit(X_fold_train, y_fold_train, X_fold_val, y_fold_val, verbose=False)

                    fold_pred = fold_builder.predict(X_fold_val)
                    if fold_pred.ndim > 1 and fold_pred.shape[1] == 1:
                        fold_pred = fold_pred.flatten()

                    # v2.0: always collect OOF predictions (needed for LOO/LOGO).
                    oof_preds[val_idx] = fold_pred

                    # Per-fold scores only meaningful when the fold is large
                    # enough (R² needs >= 2 points). OOF path reports MAE instead.
                    if problem_type == 'regression':
                        if len(val_idx) >= 2:
                            fold_scores.append(r2_score(y_fold_val, fold_pred))
                        else:
                            fold_scores.append(np.nan)
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

                    # v2.0 MEMORY FIX: Explicitly delete the fold model so PyTorch
                    # weights are reclaimed before the next fold is created.
                    del fold_builder
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass

                # v2.0: assemble via the shared helper — OOF (global R² + per-unit
                # MAE) for LOO/LOGO, per-fold R²/accuracy otherwise.
                cv_metrics = assemble_cv_metrics(
                    y=y_train,
                    plan=cv_plan,
                    problem_type=problem_type,
                    oof_preds=oof_preds,
                    fold_test_scores=fold_scores,
                    fold_train_scores=train_scores,
                    fold_sizes=fold_sizes,
                )

                self.logger.info(f"NN CV completed ({cv_plan.strategy}). "
                                 f"Test scores: {cv_metrics.get('test_score')}")
            except Exception as e:
                self.logger.error(f"NN cross-validation failed: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                cv_metrics = {'test_score': [metrics.get('r2', metrics.get('accuracy', 0))], 'train_score': []}
        else:
            if metrics:
                cv_metrics['test_score'] = [metrics.get('r2', metrics.get('accuracy', 0))]
        
        # v1.5.0: External validation metrics (separate from internal test)
        external_metrics = {}
        has_external_validation = False
        if X_external_val_np is not None and y_external_val is not None:
            ext_pred = nn_builder.predict(X_external_val_np)
            if ext_pred.ndim > 1 and ext_pred.shape[1] == 1:
                ext_pred = ext_pred.flatten()
            external_metrics = self._compute_metrics(y_external_val, ext_pred, problem_type)
            has_external_validation = True
        
        training_time = time.time() - start_time

        # v1.3.0 FIX: Use _get_unique_model_name() so multiple NNs don't overwrite each other.
        # Same mechanism as train() for sklearn models.
        unique_name = self._get_unique_model_name(model_name)

        # v2.0: Extract complete training config from nn_builder for reproducibility
        tc = getattr(nn_builder, '_training_config', None)
        nn_config = {}
        nn_rs = 42
        if tc is not None:
            nn_rs = getattr(tc, 'random_seed', 42) or 42
            nn_config = {
                'random_seed': nn_rs,
                'cv_folds': cv_folds,
                'cv_strategy': cv_config.strategy,
                'problem_type': problem_type,
                'epochs': getattr(tc, 'epochs', 'N/A'),
                'batch_size': getattr(tc, 'batch_size', 'N/A'),
                'learning_rate': getattr(tc, 'learning_rate', 'N/A'),
                'optimizer': getattr(tc, 'optimizer', 'N/A').value if hasattr(getattr(tc, 'optimizer', None), 'value') else str(getattr(tc, 'optimizer', 'N/A')),
                'scheduler': getattr(tc, 'scheduler', 'N/A').value if hasattr(getattr(tc, 'scheduler', None), 'value') else str(getattr(tc, 'scheduler', 'N/A')),
                'weight_decay': getattr(tc, 'weight_decay', 'N/A'),
                'early_stopping_patience': getattr(tc, 'early_stopping_patience', 'N/A'),
                'early_stopping_min_delta': getattr(tc, 'early_stopping_min_delta', 'N/A'),
                'gradient_clip_val': getattr(tc, 'gradient_clip_val', 'N/A'),
            }

        # v2.0 CRITICAL: Deep-copy nn_history to prevent reference sharing.
        import copy
        history_copy = copy.deepcopy(history) if history else {}

        # v2.0: Capture architecture as immutable fields at training time.
        # This avoids dependency on the live nn_builder during export.
        arch_desc = 'Unknown'
        arch_detail = {}
        arch = getattr(nn_builder, 'architecture', None)
        if arch is not None:
            layers = arch.hidden_layers if hasattr(arch, 'hidden_layers') else []
            arch_desc = '[' + ', '.join(
                f"{l.n_units}-{l.activation.value}" for l in layers
            ) + ']'
            for i, layer in enumerate(layers):
                arch_detail[f'layer_{i+1}'] = {
                    'n_units': layer.n_units,
                    'activation': layer.activation.value,
                    'dropout_rate': layer.dropout_rate,
                    'use_batch_norm': layer.use_batch_norm,
                    'use_layer_norm': layer.use_layer_norm,
                }
        epochs_trained = len(history_copy.get('train_loss', [])) if history_copy else 0

        # v2.0: Fit the out-of-distribution detector on the training features.
        try:
            ood_detector = OodDetector(n_neighbors=1, threshold_percentile=95.0)
            ood_detector.fit(X_train_np)
        except Exception as e:
            self.logger.warning(f"Could not fit OOD detector for NN: {e}")
            ood_detector = None

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
            nn_history=history_copy,
            external_metrics=external_metrics if has_external_validation else None,
            has_external_validation=has_external_validation,
            training_config=nn_config,
            random_state=nn_rs,
            # v2.0: Immutable NN fields for robust export
            nn_architecture_desc=arch_desc,
            nn_architecture_detail=arch_detail,
            nn_epochs_trained=epochs_trained,
            cv_strategy=cv_config.strategy,
            cv_config_dict=cv_config.to_dict(),
            ood_detector=ood_detector,
        )

        self._results[unique_name] = result

        # v2.0 MEMORY FIX: Cache the heavy NN to disk and free RAM.
        try:
            cache = get_model_cache()
            cache_path = cache.save(
                nn_builder, unique_name,
                is_neural_network=True,
                nn_architecture=arch_detail if arch_detail else None,
            )
            result.model_cache_path = str(cache_path)
            result.is_cached = True
            result.model = None  # free RAM
            self.logger.info(
                f"NN model cached to disk: {cache_path} "
                f"({cache_path.stat().st_size / (1024 * 1024):.1f} MB)"
            )
        except Exception as e:
            self.logger.warning(f"Could not cache NN model to disk: {e}")

        self.logger.info(
            f"Neural Network training complete in {training_time:.2f}s. "
            f"CV Score: {np.mean(cv_metrics.get('test_score', [0])):.4f}" if cv_metrics.get('test_score') else "CV Score: N/A"
        )

        # v2.0 MEMORY FIX: PyTorch models and their computation graphs can
        # retain significant memory even after they go out of scope.  Force
        # garbage collection to reclaim CUDA / CPU memory before the next
        # model in the queue starts.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

        return result
    
    def _cross_validate(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str,
        cv_folds: int,
        random_state: int = 42,
        cv_strategy: str = 'kfold',
        cv_config: Optional[CVConfig] = None,
    ) -> Dict[str, List[float]]:
        """Perform cross-validation with a configurable strategy.

        v2.0: All strategies are built by ``cv_factory.build_cv_plan`` so the
        splitter / scoring / out-of-fold decision lives in ONE place:

          kfold      Standard K-Fold (classification → StratifiedKFold by class)
          stratified Quantile-stratified K-Fold (regression; folds preserve the
                     target distribution)
          group      Group K-Fold (a group never crosses the train/test split)
          loo        Leave-One-Out (out-of-fold predictions)
          logo       Leave-One-Group-Out (out-of-fold predictions)

        v2.0 FIX (retained): For any regression strategy whose test folds can
        hold a single sample (LOO, LOGO, or a group fold of one sample), R² is
        mathematically undefined (variance needs ≥2 points). Those use
        cross_val_predict to obtain out-of-fold predictions, then compute:
          - R² globally over all OOF predictions
          - MAE per unit (sample for LOO, group for LOGO) for per-fold display
        """
        try:
            # Resolve config (legacy cv_strategy string → CVConfig if needed).
            if cv_config is None:
                cv_config = CVConfig(
                    strategy=('loo' if cv_strategy == 'loo' else 'kfold'),
                    n_folds=cv_folds,
                    random_state=random_state,
                )

            self.logger.info(f"Starting CV with strategy={cv_config.strategy}, folds={cv_config.n_folds}, "
                           f"X.shape={X.shape}, problem_type={problem_type}")

            plan = build_cv_plan(cv_config, X, y, problem_type=problem_type)
            for diag in plan.diagnostics:
                self.logger.warning(f"CV diagnostic: {diag}")

            # ── Out-of-fold path (regression, single-sample-capable folds) ──
            if plan.use_oof and problem_type == 'regression':
                from sklearn.model_selection import cross_val_predict
                oof_preds = cross_val_predict(
                    model, X, y, cv=plan.splitter, groups=plan.split_labels,
                    n_jobs=self._cv_n_jobs
                )
                r2_global = float(r2_score(y, oof_preds))
                abs_err = np.abs(y - oof_preds)

                if cv_config.strategy == 'loo':
                    # One MAE value per sample.
                    test_scores = abs_err.tolist()
                    fold_sizes = [1] * len(y)
                else:
                    # Grouped OOF (LOGO / group-with-singleton): one MAE per group.
                    groups = np.asarray(plan.split_labels)
                    test_scores = []
                    fold_sizes = []
                    for g in pd.unique(groups):
                        mask = groups == g
                        test_scores.append(float(np.mean(abs_err[mask])))
                        fold_sizes.append(int(mask.sum()))

                self.logger.info(
                    f"{cv_config.strategy.upper()}-CV (OOF) completed. Global R2={r2_global:.4f}, "
                    f"units={len(test_scores)}"
                )
                result = {
                    'test_score': test_scores,          # per-unit MAE
                    'test_score_metric': 'mae',          # marker: MAE, not R²
                    'r2_global': r2_global,              # global R² from OOF preds
                    'train_score': [],                   # not meaningful per-unit
                    'fit_time': [],
                    'fold_test_sizes': fold_sizes,
                    'cv_diagnostics': plan.diagnostics,
                }
                if plan.group_info:
                    result['group_info'] = plan.group_info
                return result

            # ── Per-fold path (K-Fold / Stratified / Group / classification) ──
            cv_results = cross_validate(
                model, X, y,
                cv=plan.splitter,
                groups=plan.split_labels,
                scoring=plan.scoring,
                return_train_score=True,
                n_jobs=self._cv_n_jobs
            )

            self.logger.info(f"CV completed successfully. Test scores: {cv_results['test_score']}")

            # Fold test sizes (informative for uneven Group K-Fold splits).
            fold_sizes = self._fold_test_sizes(plan, len(y))

            out = {
                'test_score': cv_results['test_score'].tolist(),
                'train_score': cv_results['train_score'].tolist(),
                'fit_time': cv_results['fit_time'].tolist(),
                'fold_test_sizes': fold_sizes,
                'cv_diagnostics': plan.diagnostics,
            }
            if plan.group_info:
                out['group_info'] = plan.group_info
            return out
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

    @staticmethod
    def _fold_test_sizes(plan, n_samples: int) -> List[int]:
        """Compute the test-set size of each fold produced by a CV plan."""
        try:
            dummy = np.zeros((n_samples, 1))
            sizes = []
            for _, test_idx in plan.splitter.split(dummy, groups=plan.split_labels):
                sizes.append(int(len(test_idx)))
            return sizes
        except Exception:
            return []
    
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
        """Extract feature importance from model.

        v2.0: Handles sklearn Pipeline transparently — drills into the last
        step (the actual estimator) so feature importances survive the
        Pipeline wrapper.
        """
        estimator = model
        # If this is a Pipeline, the real model is the last step
        if hasattr(model, 'named_steps') and model.named_steps:
            estimator = list(model.named_steps.values())[-1]

        importance = None

        if hasattr(estimator, 'feature_importances_'):
            importance = estimator.feature_importances_
        elif hasattr(estimator, 'coef_'):
            coef = estimator.coef_
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
    
    def compute_learning_curve(
        self,
        model_name: str,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        cv_folds: int = 5,
        train_sizes: int = 10,
        random_state: int = 42,
    ) -> Optional[Dict[str, Any]]:
        """Compute learning curve data as a table for export.
        
        Returns a dict with train_sizes, train_scores_mean, train_scores_std,
        test_scores_mean, test_scores_std — all as lists for easy CSV export.
        The user can plot these with their preferred tool.
        
        v1.5.0: Applies to ALL sklearn-compatible models.
        """
        if model_name not in self._results:
            self.logger.error(f"Model '{model_name}' not found for learning curve")
            return None
        
        result = self._results[model_name]
        model = result.model
        problem_type = result.problem_type
        scoring = 'r2' if problem_type == 'regression' else 'accuracy'
        
        try:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
            
            train_sizes_abs, train_scores, test_scores = learning_curve(
                model, X, y,
                cv=kf,
                scoring=scoring,
                n_jobs=self._cv_n_jobs,
                train_sizes=np.linspace(0.1, 1.0, train_sizes),
                random_state=random_state,
            )
            
            data = {
                'train_sizes_abs': train_sizes_abs.tolist(),
                'train_sizes_pct': (train_sizes_abs / len(y) * 100).tolist(),
                'train_scores_mean': np.mean(train_scores, axis=1).tolist(),
                'train_scores_std': np.std(train_scores, axis=1).tolist(),
                'test_scores_mean': np.mean(test_scores, axis=1).tolist(),
                'test_scores_std': np.std(test_scores, axis=1).tolist(),
                'n_folds': cv_folds,
                'scoring': scoring,
                'model_name': model_name,
            }
            
            # Store in result
            result.learning_curve_data = data
            self.logger.info(f"Learning curve computed for {model_name}: {len(train_sizes_abs)} points")
            return data
            
        except Exception as e:
            self.logger.error(f"Learning curve failed for {model_name}: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return None
    
    def _estimate_cv_residuals(
        self,
        model: BaseEstimator,
        X: Union[np.ndarray, pd.DataFrame],
        y: np.ndarray,
        problem_type: str,
        cv_folds: int = 5,
        random_state: int = 42,
        cv_strategy: str = 'kfold',
        cv_config: Optional[CVConfig] = None,
    ) -> Dict[str, Any]:
        """Estimate confidence interval parameters from CV residuals.

        For regression: computes std of CV residuals for ±intervals.
        For classification: stores predict_proba std across folds.

        v1.5.0: Applies to ALL sklearn-compatible models.
        v2.0: Supports LOO-CV strategy.
        v2.0: Uses the same cv_factory plan as _cross_validate so that grouped
        strategies (group / LOGO) keep group membership intact when estimating
        out-of-fold residuals (no leakage across the train/test boundary).
        """
        try:
            from sklearn.model_selection import cross_val_predict
            # Resolve config (legacy cv_strategy string → CVConfig if needed).
            if cv_config is None:
                cv_config = CVConfig(
                    strategy=('loo' if cv_strategy == 'loo' else 'kfold'),
                    n_folds=cv_folds,
                    random_state=random_state,
                )
            plan = build_cv_plan(cv_config, X, y, problem_type=problem_type)

            if problem_type == 'regression':
                # Get out-of-fold predictions (group-aware when applicable)
                oof_preds = cross_val_predict(
                    model, X, y, cv=plan.splitter, groups=plan.split_labels,
                    n_jobs=self._cv_n_jobs
                )
                residuals = y - oof_preds
                resid_std = float(np.std(residuals))
                resid_mean = float(np.mean(residuals))
                
                # Compute empirical quantiles for better intervals
                sorted_residuals = np.sort(np.abs(residuals))
                q_68 = float(sorted_residuals[int(len(sorted_residuals) * 0.68)]) if len(sorted_residuals) > 1 else resid_std
                q_95 = float(sorted_residuals[int(len(sorted_residuals) * 0.95)]) if len(sorted_residuals) > 1 else 2.0 * resid_std
                
                return {
                    'type': 'regression',
                    'residuals_std': resid_std,
                    'residuals_mean': resid_mean,
                    'quantile_68': q_68,   # ~1 std
                    'quantile_95': q_95,   # ~2 std
                    'n_samples': len(y),
                }
            else:
                # Classification: store predict_proba capability
                has_proba = hasattr(model, 'predict_proba')
                return {
                    'type': 'classification',
                    'has_predict_proba': has_proba,
                    'n_samples': len(y),
                }
        except Exception as e:
            self.logger.error(f"CV residual estimation failed: {e}")
            return {'type': problem_type, 'error': str(e)}
    
    def predict_with_intervals(
        self,
        model_name: str,
        X: Union[np.ndarray, pd.DataFrame],
        confidence: float = 0.95,
    ) -> Dict[str, np.ndarray]:
        """Get predictions with confidence intervals.
        
        v1.5.0: Returns a dict with 'predictions', 'lower', 'upper' bounds.
        For regression: uses CV residual quantiles.
        For classification: returns class probabilities if available.
        Applies to ALL models (sklearn, NN, ensemble).
        """
        if model_name not in self._results:
            raise ValueError(f"Model '{model_name}' not found")
        
        result = self._results[model_name]
        model = result.model
        ci_params = result.confidence_interval_params or {}
        
        preds = model.predict(X)
        
        lower = preds.copy()
        upper = preds.copy()
        
        if result.problem_type == 'regression' and ci_params.get('type') == 'regression':
            if confidence >= 0.95:
                margin = ci_params.get('quantile_95', 2.0 * ci_params.get('residuals_std', 0))
            else:
                margin = ci_params.get('quantile_68', ci_params.get('residuals_std', 0))
            lower = preds - margin
            upper = preds + margin
        elif result.problem_type == 'classification' and hasattr(model, 'predict_proba'):
            try:
                proba = model.predict_proba(X)
                margins = np.std(proba, axis=1)
                lower = margins  # uncertainty measure
                upper = np.max(proba, axis=1)  # confidence
            except Exception:
                pass
        
        return {
            'predictions': preds,
            'lower': lower,
            'upper': upper,
            'confidence_level': confidence,
            'confidence_interval_params': ci_params,
        }
    
    def predict(self, model_name: str, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Make predictions with a trained model."""
        if model_name not in self._results:
            raise ValueError(f"Model '{model_name}' not found")

        result = self._results[model_name]
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
