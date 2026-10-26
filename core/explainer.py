"""Model explainability with SHAP, permutation importance, and PDP."""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, partial_dependence

from ..utils.logger import LoggerMixin
from ..utils.hardware_detector import get_hardware_detector


class ModelExplainer(LoggerMixin):
    """Explain model predictions using various techniques."""
    
    def __init__(self):
        """Initialize explainer."""
        self._model = None
        self._model_name = None
        self._explainer = None
        self._shap_values = None
        self._feature_names = None
        self._background_data = None
    
    @staticmethod
    def _get_explainer_n_jobs(model: Any) -> int:
        """v2.0: Determine safe n_jobs for permutation_importance.

        Strategy — parallelize permutation_importance when safe, never when not:

        1. PyInstaller frozen Windows → ALWAYS n_jobs=1 (joblib multiprocessing
           re-imports main module, spawning duplicate app windows).
        2. Model has internal multi-core parallelism (RF, XGB, LGBM, HistGB,
           CatBoost, etc.) → n_jobs=1 (avoid oversubscription).
        3. Model is single-threaded (SVR, Ridge, KNN with n_jobs=1, etc.)
           → n_jobs=-1 (use all cores, model won't fight for them).

        Detection for #2 checks BOTH the n_jobs/nthread parameter AND the
        model class name, because some models (HistGradientBoosting) use
        OpenMP internally without exposing an n_jobs parameter.
        """
        import sys
        # 1. PyInstaller frozen Windows: NEVER use multiprocessing
        if getattr(sys, 'frozen', False) and sys.platform == 'win32':
            return 1

        cls_name = model.__class__.__name__

        # 2. Detect models with internal multi-core parallelism
        # These models already saturate CPU cores — adding joblib processes
        # on top causes oversubscription (slower + unreliable results).
        _MULTI_THREADED_CLASSES = (
            'RandomForest', 'ExtraTrees', 'GradientBoosting',
            'HistGradientBoosting', 'AdaBoost', 'Bagging', 'Stacking',
            'XGBRegressor', 'XGBClassifier', 'XGB', 'LGBM',
            'CatBoostRegressor', 'CatBoostClassifier', 'CatBoost',
        )
        is_multithreaded = any(t in cls_name for t in _MULTI_THREADED_CLASSES)

        # Also check n_jobs/nthread param if model has get_params
        if not is_multithreaded and hasattr(model, 'get_params'):
            try:
                params = model.get_params()
                model_n_jobs = params.get('n_jobs', params.get('thread_count', 1))
                if isinstance(model_n_jobs, int) and (model_n_jobs > 1 or model_n_jobs == -1):
                    is_multithreaded = True
            except Exception:
                pass

        if is_multithreaded:
            return 1  # Don't oversubscribe

        # 3. Single-threaded model — safe to parallelize permutation
        try:
            return get_hardware_detector().info.recommended_n_jobs
        except Exception:
            import os
            return max(1, (os.cpu_count() or 4) - 1)

    def set_model(self, model: Any, model_name: str = "model") -> None:
        """Set the model to explain.

        Args:
            model: Trained model
            model_name: Name of the model
        """
        self._model = model
        self._model_name = model_name
        self.logger.info(f"Model set: {model_name}")
    
    def calculate_shap_values(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        progress_callback: Optional[Callable[[int], None]] = None
    ) -> np.ndarray:
        """Calculate SHAP values for the set model.
        
        Args:
            X: Feature matrix
            feature_names: Names of features
            progress_callback: Callback function(progress_percent)
            
        Returns:
            SHAP values array
        """
        if self._model is None:
            raise ValueError("No model set. Call set_model() first.")
        
        result = self.compute_shap_values(
            model=self._model,
            X=X,
            feature_names=feature_names
        )
        
        if progress_callback:
            progress_callback(100)
        
        return result.get('shap_values', np.array([]))
    
    def _compute_shap_with_best_explainer(
        self,
        model: Any,
        X_sample: np.ndarray,
        background_data: Optional[np.ndarray] = None
    ) -> tuple:
        """Select the best SHAP explainer for the model type and compute values.

        v2.0: Architecture — always use the specialized explainer for each model
        family. KernelExplainer is the absolute last resort.

        Returns (shap_values, explainer_instance) or raises Exception.
        """
        import shap

        # v2.0: Conditionally import torch — may not be available in all envs
        try:
            import torch
            _torch_available = True
        except ImportError:
            _torch_available = False
            torch = None  # type: ignore[assignment]

        cls_name = model.__class__.__name__
        module_name = model.__class__.__module__ if hasattr(model, '__class__') else ''

        # v2.0: If model is a sklearn Pipeline, drill into the last step (the
        # actual estimator) for explainer selection. SHAP explainers need the
        # real model, not the Pipeline wrapper. We keep the original Pipeline
        # for prediction but use the inner estimator for SHAP type detection.
        _shap_model = model
        if hasattr(model, 'named_steps') and model.named_steps:
            _shap_model = list(model.named_steps.values())[-1]
            _shap_cls_name = _shap_model.__class__.__name__
            _shap_module_name = _shap_model.__class__.__module__
        else:
            _shap_cls_name = cls_name
            _shap_module_name = module_name

        # --- 1. Neural Networks (PyTorch) → DeepExplainer ---
        if _torch_available and torch is not None and isinstance(_shap_model, torch.nn.Module):
            self.logger.info("Using DeepExplainer for PyTorch Neural Network")
            _shap_model.eval()  # Ensure eval mode
            # DeepExplainer expects torch Tensors
            # v2.0 FIX: np.asarray() ensures DataFrames/Series are converted to numpy
            # arrays before torch.FloatTensor, which cannot infer shape from pandas objects.
            X_tensor = torch.FloatTensor(np.asarray(X_sample))
            bg = background_data if background_data is not None else X_sample[:min(10, len(X_sample))]
            bg_tensor = torch.FloatTensor(np.asarray(bg))
            explainer = shap.DeepExplainer(_shap_model, bg_tensor)
            shap_values = explainer.shap_values(X_tensor)
            return shap_values, explainer

        # --- 2. Tree-based models → TreeExplainer (exact, fast) ---
        tree_model_names = (
            'HistGradientBoosting', 'GradientBoosting', 'RandomForest',
            'ExtraTrees', 'DecisionTree', 'XGB', 'LGBM', 'CatBoost',
            'AdaBoost', 'Bagging'
        )
        is_tree = (
            hasattr(_shap_model, 'feature_importances_')
            or any(t in _shap_cls_name for t in tree_model_names)
            or 'sklearn.tree' in _shap_module_name
            or 'sklearn.ensemble' in _shap_module_name
        )
        if is_tree:
            self.logger.info(f"Using TreeExplainer for {_shap_cls_name}")
            explainer = shap.TreeExplainer(_shap_model)
            shap_values = explainer.shap_values(X_sample)
            return shap_values, explainer

        # --- 3. Linear models → LinearExplainer ---
        is_linear = (
            hasattr(_shap_model, 'coef_')
            or any(t in _shap_cls_name for t in ('Ridge', 'Lasso', 'ElasticNet',
                                                    'LinearRegression', 'LogisticRegression',
                                                    'LinearSVC', 'LinearSVR', 'SGD'))
            or 'linear_model' in _shap_module_name
        )
        if is_linear:
            self.logger.info(f"Using LinearExplainer for {_shap_cls_name}")
            bg = background_data if background_data is not None else X_sample
            explainer = shap.LinearExplainer(_shap_model, bg)
            shap_values = explainer.shap_values(X_sample)
            return shap_values, explainer

        # --- 4. Fallback → KernelExplainer (slow, approximate) ---
        self.logger.warning(
            f"No specialized explainer for {_shap_cls_name} — falling back to KernelExplainer. "
            "This will be slow and results may vary between runs."
        )
        bg = background_data if background_data is not None else X_sample[:min(10, len(X_sample))]
        explainer = shap.KernelExplainer(_shap_model.predict, bg)
        shap_values = explainer.shap_values(X_sample, nsamples=100)
        return shap_values, explainer

    def compute_shap_values(
        self,
        model: Any,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        background_data: Optional[np.ndarray] = None,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """Compute SHAP values for model interpretation.
        
        Args:
            model: Trained model
            X: Feature matrix
            feature_names: Names of features
            background_data: Background data for SHAP
            sample_size: Number of samples to explain
            
        Returns:
            Dictionary with SHAP values and metadata
        """
        try:
            import shap
        except ImportError:
            self.logger.warning("SHAP not available. Install with: pip install shap")
            return {}

        # v2.0: Convert DataFrames/Series to numpy arrays for SHAP compatibility.
        if hasattr(X, 'values'):
            X = X.values
        if background_data is not None and hasattr(background_data, 'values'):
            background_data = background_data.values

        # SIMPLIFICADO: Usar nombres basados en X.shape[1]
        n_features = X.shape[1]
        self._feature_names = feature_names[:n_features] if feature_names and len(feature_names) >= n_features else [f"feature_{i}" for i in range(n_features)]

        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # v2.0: Use the best specialized explainer for each model family
        try:
            self._shap_values, self._explainer = self._compute_shap_with_best_explainer(
                model, X_sample, background_data
            )

            # Handle multi-output
            if isinstance(self._shap_values, list):
                # For classification, use values for predicted class
                self._shap_values = np.array(self._shap_values)
                if self._shap_values.ndim > 2:
                    self._shap_values = self._shap_values[0]
            
            # Global feature importance (mean absolute SHAP value)
            global_importance = np.abs(self._shap_values).mean(axis=0)
            
            result = {
                'shap_values': self._shap_values,
                'expected_value': getattr(self._explainer, 'expected_value', None),
                'feature_names': self._feature_names,
                'X_sample': X_sample,
                'global_importance': dict(zip(self._feature_names, global_importance.tolist())),
            }
            
            self.logger.info("SHAP values computed successfully")
            return result
            
        except Exception as e:
            self.logger.warning(f"Could not compute SHAP values: {e}")
            import traceback
            self.logger.debug(traceback.format_exc())
            return {}
    
    def compute_permutation_importance(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_repeats: int = 10,
        n_jobs: int = 1,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Compute permutation feature importance."""
        n_features = X.shape[1]
        self._feature_names = feature_names[:n_features] if feature_names and len(feature_names) >= n_features else [f"feature_{i}" for i in range(n_features)]
        
        try:
            # Determine scoring
            if len(np.unique(y)) <= 10:
                scoring = 'accuracy'
            else:
                scoring = 'r2'
            
            # v2.0: Use model-aware n_jobs to avoid oversubscription
            explainer_n_jobs = self._get_explainer_n_jobs(model)
            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=scoring,
                n_jobs=explainer_n_jobs
            )
            
            importance_dict = {
                name: {
                    'importance_mean': mean,
                    'importance_std': std,
                }
                for name, mean, std in zip(
                    self._feature_names,
                    result.importances_mean,
                    result.importances_std
                )
            }
            
            return {
                'importances': importance_dict,
                'importances_mean': result.importances_mean,
                'importances_std': result.importances_std,
                'importances_full': result.importances,
            }
            
        except Exception as e:
            self.logger.warning(f"Could not compute permutation importance: {e}")
            return {}
    
    def compute_partial_dependence(
        self,
        model: Any,
        X: np.ndarray,
        features: Union[List[int], List[str]],
        feature_names: Optional[List[str]] = None,
        grid_resolution: int = 20
    ) -> Dict[str, Any]:
        """Compute partial dependence plots.
        
        Args:
            model: Trained model
            X: Feature matrix
            features: Features to analyze (indices or names)
            feature_names: Names of all features
            grid_resolution: Number of grid points
            
        Returns:
            Dictionary with PDP data
        """
        self._feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        # Convert feature names to indices
        feature_indices = []
        for f in features:
            if isinstance(f, str):
                if f in self._feature_names:
                    feature_indices.append(self._feature_names.index(f))
                else:
                    raise ValueError(f"Feature '{f}' not found")
            else:
                feature_indices.append(f)
        
        try:
            pd_results = partial_dependence(
                model, X,
                features=feature_indices,
                grid_resolution=grid_resolution,
                kind='average'
            )
            
            pdp_data = {}
            for i, (feature_idx, grid_values, pdp_values) in enumerate(
                zip(feature_indices, pd_results['grid_values'], pd_results['average'])
            ):
                feature_name = self._feature_names[feature_idx]
                pdp_data[feature_name] = {
                    'grid_values': grid_values,
                    'pdp_values': pdp_values[0] if pdp_values.ndim > 1 else pdp_values,
                }
            
            return pdp_data
            
        except Exception as e:
            self.logger.warning(f"Could not compute partial dependence: {e}")
            return {}
    
    def get_feature_importance_summary(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        random_state: int = 42
    ) -> pd.DataFrame:
        """Get comprehensive feature importance summary."""
        n_features = X.shape[1]
        
        # Usar feature_names si coinciden, sino crear genericos
        if feature_names and len(feature_names) == n_features:
            self._feature_names = list(feature_names)
        else:
            self._feature_names = [f"feature_{i}" for i in range(n_features)]
        
        # Crear DataFrame base con los nombres de features
        df = pd.DataFrame({'feature': self._feature_names})
        
        # --- 1. Model-specific importance (built-in) ---
        if hasattr(model, 'feature_importances_'):
            imp = np.array(model.feature_importances_)
            if len(imp) == n_features:
                df['model_importance'] = imp
        elif hasattr(model, 'coef_'):
            coef = model.coef_
            if coef.ndim > 1:
                coef = np.abs(coef).mean(axis=0)
            coef = np.array(coef)
            if len(coef) == n_features:
                df['model_importance'] = np.abs(coef)
        # v2.0 FIX: For models WITHOUT built-in feature importance
        # (e.g., HistGradientBoosting, which has NO feature_importances_),
        # compute a surrogate using permutation importance so it acts as the
        # 'model_importance' column that the GUI expects.
        else:
            try:
                self.logger.info(
                    f"Model {model.__class__.__name__} has no built-in feature "
                    f"importance — computing surrogate via permutation."
                )
                # Use _get_explainer_n_jobs: single-threaded models get n_jobs=-1
                # (fast), multi-threaded models get n_jobs=1 (no oversubscription).
                surr_njobs = self._get_explainer_n_jobs(model)
                surr = permutation_importance(
                    model, X, y, n_repeats=3, random_state=random_state,
                    n_jobs=surr_njobs
                )
                if len(surr.importances_mean) == n_features:
                    df['model_importance'] = surr.importances_mean
            except Exception as e:
                self.logger.warning(f"Surrogate feature importance failed: {e}")
        
        # --- 2. Permutation importance (full, robust) ---
        try:
            # v2.0: Use _get_explainer_n_jobs for smart parallelization.
            # Single-threaded models (SVR, Ridge) get n_jobs=-1 → FAST.
            # Multi-threaded models (RF, XGB, LGBM, HistGB) get n_jobs=1
            # → no oversubscription. The n_repeats param provides variance.
            explainer_n_jobs = self._get_explainer_n_jobs(model)
            perm_result = permutation_importance(
                model, X, y, n_repeats=5, random_state=random_state,
                n_jobs=explainer_n_jobs
            )
            if len(perm_result.importances_mean) == n_features:
                df['permutation_importance'] = perm_result.importances_mean
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
        
        # --- 3. SHAP importance ---
        try:
            X_sample = X[:min(100, len(X))]
            shap_values, _ = self._compute_shap_with_best_explainer(
                model, X_sample, background_data=X_sample
            )
            # Handle multi-output (para clasificación)
            if isinstance(shap_values, list):
                shap_values = shap_values[0] if len(shap_values) > 0 else shap_values
            shap_imp = np.abs(shap_values).mean(axis=0)
            if len(shap_imp) == n_features:
                df['shap_importance'] = shap_imp
        except Exception as e:
            self.logger.warning(f"SHAP importance failed: {e}")
        
        # Normalize each column to 0-1
        for col in df.columns:
            if col != 'feature' and df[col].max() > 0:
                df[f'{col}_norm'] = df[col] / df[col].max()
        
        # Compute average rank
        rank_cols = [c for c in df.columns if c.endswith('_norm')]
        if rank_cols:
            df['average_rank'] = df[rank_cols].mean(axis=1)
            df = df.sort_values('average_rank', ascending=False)
        
        return df
    
    def explain_prediction(
        self,
        model: Any,
        X_instance: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Explain a single prediction."""
        # v2.0: Convert DataFrames/Series to numpy arrays.
        if hasattr(X_instance, 'values'):
            X_instance = X_instance.values
        # Ensure 2D
        if X_instance.ndim == 1:
            X_instance = X_instance.reshape(1, -1)
        
        # SIMPLIFICADO: Usar nombres basados en X_instance.shape[1]
        n_features = X_instance.shape[1]
        self._feature_names = feature_names[:n_features] if feature_names and len(feature_names) >= n_features else [f"feature_{i}" for i in range(n_features)]
        
        # Get prediction
        prediction = model.predict(X_instance)[0]
        
        # Get SHAP values if available
        shap_result = self.compute_shap_values(model, X_instance, self._feature_names)
        
        explanation = {
            'prediction': prediction,
            'feature_values': dict(zip(self._feature_names, X_instance[0].tolist())),
        }
        
        if shap_result and shap_result.get('shap_values') is not None:
            shap_values = shap_result['shap_values']
            if shap_values.ndim > 1:
                shap_values = shap_values[0]
            
            # Feature contributions
            contributions = dict(zip(self._feature_names, shap_values.tolist()))
            explanation['shap_contributions'] = contributions
            explanation['base_value'] = shap_result.get('expected_value')
            
            # Top contributing features
            sorted_contribs = sorted(
                contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )
            explanation['top_positive'] = [(f, v) for f, v in sorted_contribs if v > 0][:5]
            explanation['top_negative'] = [(f, v) for f, v in sorted_contribs if v < 0][:5]
        
        return explanation
    
    def plot_shap_summary(self, max_display: int = 20, figsize: Tuple[int, int] = (10, 8)):
        """Plot SHAP summary plot.
        
        Args:
            max_display: Maximum features to display
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        try:
            import shap
            
            if self._shap_values is None:
                raise ValueError("SHAP values not computed. Call compute_shap_values first.")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            shap.summary_plot(
                self._shap_values,
                self._background_data,
                feature_names=self._feature_names,
                max_display=max_display,
                show=False
            )
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.warning(f"Could not create SHAP summary plot: {e}")
            return None
    
    def plot_shap_waterfall(
        self,
        instance_idx: int = 0,
        figsize: Tuple[int, int] = (10, 6)
    ):
        """Plot SHAP waterfall plot for a single instance.
        
        Args:
            instance_idx: Index of instance to explain
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        try:
            import shap
            
            if self._shap_values is None:
                raise ValueError("SHAP values not computed")
            
            fig, ax = plt.subplots(figsize=figsize)
            
            shap.waterfall_plot(
                shap.Explanation(
                    values=self._shap_values[instance_idx],
                    base_values=self._explainer.expected_value,
                    feature_names=self._feature_names,
                    data=self._background_data[instance_idx] if self._background_data is not None else None
                ),
                show=False
            )
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.warning(f"Could not create waterfall plot: {e}")
            return None
    
    def plot_partial_dependence(
        self,
        pdp_data: Dict[str, Any],
        features: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """Plot partial dependence plots.
        
        Args:
            pdp_data: PDP data from compute_partial_dependence
            features: Features to plot
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        try:
            if features is None:
                features = list(pdp_data.keys())[:4]  # Plot first 4 by default
            
            n_features = len(features)
            n_cols = min(2, n_features)
            n_rows = (n_features + 1) // 2
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows))
            
            if n_features == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = [axes] if n_cols == 1 else axes
            else:
                axes = axes.flatten()
            
            for i, feature in enumerate(features):
                if feature in pdp_data:
                    data = pdp_data[feature]
                    axes[i].plot(data['grid_values'], data['pdp_values'])
                    axes[i].set_xlabel(feature)
                    axes[i].set_ylabel('Partial Dependence')
                    axes[i].set_title(f'PDP: {feature}')
                    axes[i].grid(True, alpha=0.3)
            
            # Hide unused subplots
            for i in range(n_features, len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.warning(f"Could not create PDP plot: {e}")
            return None
