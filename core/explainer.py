"""Model explainability with SHAP, permutation importance, and PDP."""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance, partial_dependence

from ..utils.logger import LoggerMixin


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
        
        # SIMPLIFICADO: Usar nombres basados en X.shape[1]
        n_features = X.shape[1]
        self._feature_names = feature_names[:n_features] if feature_names and len(feature_names) >= n_features else [f"feature_{i}" for i in range(n_features)]
        
        # Sample data if too large
        if len(X) > sample_size:
            indices = np.random.choice(len(X), sample_size, replace=False)
            X_sample = X[indices]
        else:
            X_sample = X
        
        # Create explainer based on model type
        try:
            # Tree-based models (CatBoost sometimes fails with TreeExplainer, so we try/except)
            if hasattr(model, 'feature_importances_') or 'XGB' in model.__class__.__name__ or 'LGBM' in model.__class__.__name__:
                try:
                    self._explainer = shap.TreeExplainer(model)
                    self._shap_values = self._explainer.shap_values(X_sample)
                except Exception as e:
                    self.logger.warning(f"TreeExplainer failed ({e}), falling back to KernelExplainer")
                    bg_data = background_data if background_data is not None else X_sample[:min(10, len(X_sample))]
                    self._explainer = shap.KernelExplainer(model.predict, bg_data)
                    self._shap_values = self._explainer.shap_values(X_sample, nsamples=100)
            # Linear models
            elif hasattr(model, 'coef_'):
                self._explainer = shap.LinearExplainer(model, background_data if background_data is not None else X_sample)
                self._shap_values = self._explainer.shap_values(X_sample)
            # For NN and other models, use KernelExplainer (universal)
            else:
                bg_data = background_data if background_data is not None else X_sample[:min(10, len(X_sample))]
                self._explainer = shap.KernelExplainer(model.predict, bg_data)
                self._shap_values = self._explainer.shap_values(X_sample, nsamples=100)
            
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
            
            result = permutation_importance(
                model, X, y,
                n_repeats=n_repeats,
                random_state=random_state,
                scoring=scoring,
                n_jobs=n_jobs
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
        
        # Model-specific importance (asegurar longitud correcta)
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
        
        # Permutation importance (funciona para todos los modelos con .predict())
        try:
            perm_result = permutation_importance(model, X, y, n_repeats=5, random_state=random_state, n_jobs=1)
            if len(perm_result.importances_mean) == n_features:
                df['permutation_importance'] = perm_result.importances_mean
        except Exception as e:
            self.logger.warning(f"Permutation importance failed: {e}")
        
        # SHAP importance
        try:
            import shap
            
            X_sample = X[:min(100, len(X))]
            
            # Detectar tipo de modelo
            is_tree_model = hasattr(model, 'feature_importances_') or 'XGB' in model.__class__.__name__ or 'LGBM' in model.__class__.__name__
            is_linear_model = hasattr(model, 'coef_')
            
            if is_tree_model:
                try:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(X_sample)
                except Exception as e:
                    self.logger.warning(f"TreeExplainer failed ({e}), falling back to KernelExplainer")
                    explainer = shap.KernelExplainer(model.predict, X_sample)
                    shap_values = explainer.shap_values(X_sample, nsamples=100)
            elif is_linear_model:
                explainer = shap.LinearExplainer(model, X_sample)
                shap_values = explainer.shap_values(X_sample)
            else:
                # Para NN y otros modelos, usar KernelExplainer (más lento pero universal)
                explainer = shap.KernelExplainer(model.predict, X_sample)
                shap_values = explainer.shap_values(X_sample, nsamples=100)
            
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
