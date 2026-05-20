"""Model registry with all supported ML algorithms."""

import warnings
from typing import Dict, List, Optional, Any, Type, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.svm import SVR, SVC
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import cross_val_score

from ..utils.logger import LoggerMixin
from ..utils.hardware_detector import get_hardware_detector


@dataclass
class ModelInfo:
    """Information about a model."""
    name: str
    display_name: str
    description: str
    supports_classification: bool
    supports_regression: bool
    supports_gpu: bool
    default_params: Dict[str, Any]
    param_ranges: Dict[str, tuple]  # (min, max, type)
    estimator_class: Type


class ModelRegistry(LoggerMixin):
    """Registry of all available models."""
    
    def __init__(self):
        """Initialize model registry."""
        self._models: Dict[str, ModelInfo] = {}
        self._hardware = get_hardware_detector()
        self._register_default_models()
    
    def _register_default_models(self) -> None:
        """Register all default models."""
        # Random Forest
        self.register(
            ModelInfo(
                name='random_forest',
                display_name='Random Forest',
                description='Ensemble of decision trees with bagging',
                supports_classification=True,
                supports_regression=True,
                supports_gpu=False,
                default_params={
                    'n_estimators': 100,
                    'max_depth': None,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'max_features': 'sqrt',
                    'random_state': 42,
                    'n_jobs': self._hardware.info.recommended_n_jobs
                },
                param_ranges={
                    'n_estimators': (10, 1000, 'int'),
                    'max_depth': (3, 50, 'int'),
                    'min_samples_split': (2, 20, 'int'),
                    'min_samples_leaf': (1, 20, 'int'),
                },
                estimator_class=RandomForestRegressor
            )
        )
        
        # Gradient Boosting
        self.register(
            ModelInfo(
                name='gradient_boosting',
                display_name='Gradient Boosting',
                description='Gradient boosted decision trees (sklearn)',
                supports_classification=True,
                supports_regression=True,
                supports_gpu=False,
                default_params={
                    'n_estimators': 100,
                    'max_depth': 3,
                    'learning_rate': 0.1,
                    'min_samples_split': 2,
                    'min_samples_leaf': 1,
                    'subsample': 1.0,
                    'random_state': 42
                },
                param_ranges={
                    'n_estimators': (10, 1000, 'int'),
                    'max_depth': (2, 20, 'int'),
                    'learning_rate': (0.001, 1.0, 'float'),
                    'subsample': (0.5, 1.0, 'float'),
                },
                estimator_class=GradientBoostingRegressor
            )
        )
        
        # XGBoost
        self._register_xgboost()
        
        # LightGBM
        self._register_lightgbm()
        
        # CatBoost
        self._register_catboost()
        
        # Support Vector Machine
        self.register(
            ModelInfo(
                name='svr',
                display_name='Support Vector Machine',
                description='Support Vector Regression/Classification with RBF kernel',
                supports_classification=True,
                supports_regression=True,
                supports_gpu=False,
                default_params={
                    'C': 1.0,
                    'epsilon': 0.1,
                    'kernel': 'rbf',
                    'gamma': 'scale',
                },
                param_ranges={
                    'C': (0.001, 1000.0, 'float'),
                    'epsilon': (0.001, 1.0, 'float'),
                    'gamma': (0.0001, 10.0, 'float'),
                },
                estimator_class=SVR
            )
        )
        
        # Ridge Regression
        self.register(
            ModelInfo(
                name='ridge',
                display_name='Ridge Regression',
                description='Linear regression with L2 regularization',
                supports_classification=False,
                supports_regression=True,
                supports_gpu=False,
                default_params={
                    'alpha': 1.0,
                    'random_state': 42
                },
                param_ranges={
                    'alpha': (0.001, 100.0, 'float'),
                },
                estimator_class=Ridge
            )
        )
    
    def _register_xgboost(self) -> None:
        """Register XGBoost models."""
        try:
            import xgboost as xgb
            
            # Check GPU support
            gpu_available = self._hardware.info.has_cuda
            
            self.register(
                ModelInfo(
                    name='xgboost',
                    display_name='XGBoost',
                    description='Extreme Gradient Boosting',
                    supports_classification=True,
                    supports_regression=True,
                    supports_gpu=gpu_available,
                    default_params={
                        'n_estimators': 100,
                        'max_depth': 6,
                        'learning_rate': 0.1,
                        'subsample': 0.8,
                        'colsample_bytree': 0.8,
                        'reg_alpha': 0.0,
                        'reg_lambda': 1.0,
                        'random_state': 42,
                        'n_jobs': self._hardware.info.recommended_n_jobs,
                        'tree_method': 'auto'
                    },
                    param_ranges={
                        'n_estimators': (10, 2000, 'int'),
                        'max_depth': (2, 15, 'int'),
                        'learning_rate': (0.001, 0.5, 'float'),
                        'subsample': (0.5, 1.0, 'float'),
                        'colsample_bytree': (0.5, 1.0, 'float'),
                        'reg_alpha': (1e-8, 10.0, 'float'),
                        'reg_lambda': (1e-8, 10.0, 'float'),
                    },
                    estimator_class=xgb.XGBRegressor
                )
            )
            self.logger.info("XGBoost registered with GPU support" if gpu_available else "XGBoost registered (CPU)")
        except ImportError:
            self.logger.warning("XGBoost not available")
    
    def _register_lightgbm(self) -> None:
        """Register LightGBM models."""
        try:
            import lightgbm as lgb
            
            gpu_available = self._hardware.info.has_cuda
            
            # FIX: Simplified default params to avoid compatibility issues
            # Removed 'device' parameter which can cause issues in some versions
            default_params = {
                'n_estimators': 100,
                'max_depth': -1,
                'learning_rate': 0.1,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'reg_alpha': 0.0,
                'reg_lambda': 0.0,
                'random_state': 42,
                'n_jobs': self._hardware.info.recommended_n_jobs,
                'verbose': -1,  # Suppress warnings
            }
            
            # Only add GPU params if GPU is available and properly configured
            if gpu_available:
                default_params['device'] = 'gpu'
                default_params['gpu_platform_id'] = 0
                default_params['gpu_device_id'] = 0
            
            self.register(
                ModelInfo(
                    name='lightgbm',
                    display_name='LightGBM',
                    description='Light Gradient Boosting Machine',
                    supports_classification=True,
                    supports_regression=True,
                    supports_gpu=gpu_available,
                    default_params=default_params,
                    param_ranges={
                        'n_estimators': (10, 2000, 'int'),
                        'max_depth': (3, 50, 'int'),  # FIX: Changed from -1 to 3 as minimum to avoid issues
                        'learning_rate': (0.001, 0.5, 'float'),
                        'num_leaves': (10, 200, 'int'),
                        'subsample': (0.5, 1.0, 'float'),
                        'colsample_bytree': (0.5, 1.0, 'float'),
                        'reg_alpha': (1e-8, 10.0, 'float'),
                        'reg_lambda': (1e-8, 10.0, 'float'),
                    },
                    estimator_class=lgb.LGBMRegressor
                )
            )
            self.logger.info("LightGBM registered with GPU support" if gpu_available else "LightGBM registered (CPU)")
        except ImportError:
            self.logger.warning("LightGBM not available")
    
    def _register_catboost(self) -> None:
        """Register CatBoost models."""
        try:
            import catboost as cb
            
            gpu_available = self._hardware.info.has_cuda
            
            self.register(
                ModelInfo(
                    name='catboost',
                    display_name='CatBoost',
                    description='Categorical Boosting',
                    supports_classification=True,
                    supports_regression=True,
                    supports_gpu=gpu_available,
                    default_params={
                        'iterations': 100,
                        'depth': 6,
                        'learning_rate': 0.1,
                        'l2_leaf_reg': 3.0,
                        'random_seed': 42,
                        'verbose': False,
                        'task_type': 'GPU' if gpu_available else 'CPU'
                    },
                    param_ranges={
                        'iterations': (10, 2000, 'int'),
                        'depth': (2, 12, 'int'),
                        'learning_rate': (0.001, 0.5, 'float'),
                        'l2_leaf_reg': (0.1, 10.0, 'float'),
                    },
                    estimator_class=cb.CatBoostRegressor
                )
            )
            self.logger.info("CatBoost registered with GPU support" if gpu_available else "CatBoost registered (CPU)")
        except ImportError:
            self.logger.warning("CatBoost not available")
    
    def register(self, model_info: ModelInfo) -> None:
        """Register a model.
        
        Args:
            model_info: Model information
        """
        self._models[model_info.name] = model_info
        self.logger.debug(f"Registered model: {model_info.name}")
    
    def get_model(self, name: str) -> ModelInfo:
        """Get model information.
        
        Args:
            name: Model name
            
        Returns:
            Model information
        """
        if name not in self._models:
            raise ValueError(f"Model '{name}' not found. Available: {list(self._models.keys())}")
        return self._models[name]
    
    def create_estimator(
        self,
        name: str,
        problem_type: str = 'regression',
        params: Optional[Dict[str, Any]] = None
    ) -> BaseEstimator:
        """Create model estimator.
        
        Args:
            name: Model name
            problem_type: 'regression' or 'classification'
            params: Override default parameters
            
        Returns:
            Configured estimator
        """
        model_info = self.get_model(name)
        
        # Get appropriate estimator class
        if problem_type == 'classification':
            if not model_info.supports_classification:
                raise ValueError(f"Model '{name}' does not support classification")
            
            # Map regressor to classifier
            estimator_class = self._get_classifier_class(model_info.estimator_class)
        else:
            if not model_info.supports_regression:
                raise ValueError(f"Model '{name}' does not support regression")
            estimator_class = model_info.estimator_class
        
        # Merge parameters
        final_params = model_info.default_params.copy()
        if params:
            final_params.update(params)
        
        # Create estimator
        estimator = estimator_class(**final_params)
        
        return estimator
    
    def _get_classifier_class(self, regressor_class: Type) -> Type:
        """Get corresponding classifier class for a regressor.
        
        Args:
            regressor_class: Regressor class
            
        Returns:
            Classifier class
        """
        class_mapping = {
            'RandomForestRegressor': RandomForestClassifier,
            'GradientBoostingRegressor': GradientBoostingClassifier,
            'SVR': SVC,
            'Ridge': LogisticRegression,
        }
        
        class_name = regressor_class.__name__
        
        if class_name in class_mapping:
            return class_mapping[class_name]
        
        # Try to import and map XGBoost, LightGBM, CatBoost
        try:
            import xgboost as xgb
            if regressor_class == xgb.XGBRegressor:
                return xgb.XGBClassifier
        except ImportError:
            pass
        
        try:
            import lightgbm as lgb
            if regressor_class == lgb.LGBMRegressor:
                return lgb.LGBMClassifier
        except ImportError:
            pass
        
        try:
            import catboost as cb
            if regressor_class == cb.CatBoostRegressor:
                return cb.CatBoostClassifier
        except ImportError:
            pass
        
        raise ValueError(f"No classifier found for {class_name}")
    
    def list_models(
        self,
        problem_type: Optional[str] = None,
        gpu_only: bool = False
    ) -> List[str]:
        """List available models.
        
        Args:
            problem_type: Filter by 'regression', 'classification', or None for all
            gpu_only: Only show GPU-supported models
            
        Returns:
            List of model names
        """
        result = []
        
        for name, info in self._models.items():
            if problem_type == 'regression' and not info.supports_regression:
                continue
            if problem_type == 'classification' and not info.supports_classification:
                continue
            if gpu_only and not info.supports_gpu:
                continue
            result.append(name)
        
        return result
    
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all models.
        
        Returns:
            Dictionary with model information
        """
        return {
            name: {
                'display_name': info.display_name,
                'description': info.description,
                'supports_classification': info.supports_classification,
                'supports_regression': info.supports_regression,
                'supports_gpu': info.supports_gpu,
                'default_params': info.default_params,
            }
            for name, info in self._models.items()
        }
    
    def get_param_grid(self, name: str, n_points: int = 5) -> Dict[str, List]:
        """Get parameter grid for grid search.
        
        Args:
            name: Model name
            n_points: Number of points per parameter
            
        Returns:
            Parameter grid dictionary
        """
        model_info = self.get_model(name)
        param_grid = {}
        
        for param, (min_val, max_val, param_type) in model_info.param_ranges.items():
            if param_type == 'int':
                values = np.linspace(min_val, max_val, n_points, dtype=int).tolist()
            elif param_type == 'float':
                values = np.linspace(min_val, max_val, n_points).tolist()
            else:
                values = [min_val, max_val]
            
            param_grid[param] = values
        
        return param_grid
