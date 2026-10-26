"""Tests for model_registry module."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge

from app.core.model_registry import ModelRegistry, ModelInfo


class TestModelRegistry:
    """Test suite for ModelRegistry."""

    def test_registry_initialization(self):
        """Test that registry initializes with default models."""
        registry = ModelRegistry()
        models = registry.list_models()
        assert len(models) >= 4  # At least RF, GB, SVR, Ridge
        assert 'random_forest' in models
        assert 'gradient_boosting' in models
        assert 'svr' in models
        assert 'ridge' in models

    def test_get_model(self):
        """Test retrieving model info."""
        registry = ModelRegistry()
        info = registry.get_model('random_forest')
        assert isinstance(info, ModelInfo)
        assert info.name == 'random_forest'
        assert info.display_name == 'Random Forest'
        assert info.supports_regression is True

    def test_get_nonexistent_model(self):
        """Test that getting a nonexistent model raises ValueError."""
        registry = ModelRegistry()
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            registry.get_model('nonexistent')

    def test_create_estimator_regression(self):
        """Test creating a regressor estimator."""
        registry = ModelRegistry()
        estimator = registry.create_estimator('random_forest', problem_type='regression')
        assert isinstance(estimator, RandomForestRegressor)

    def test_create_estimator_classification(self):
        """Test creating a classifier estimator."""
        registry = ModelRegistry()
        # Skip if model doesn't support classification
        info = registry.get_model('ridge')
        if not info.supports_classification:
            pytest.skip("Ridge doesn't support classification")
        estimator = registry.create_estimator('random_forest', problem_type='classification')
        from sklearn.ensemble import RandomForestClassifier
        assert isinstance(estimator, RandomForestClassifier)

    def test_create_estimator_svr_no_random_state(self):
        """Test that SVR doesn't get random_state passed (bug fix v1.3.0)."""
        registry = ModelRegistry()
        # SVR doesn't accept random_state - this should not raise
        estimator = registry.create_estimator('svr', problem_type='regression')
        assert isinstance(estimator, SVR)
        # Verify random_state is NOT in the estimator
        params = estimator.get_params()
        assert 'random_state' not in params or params['random_state'] is None

    def test_create_estimator_with_custom_params(self):
        """Test creating estimator with custom parameters."""
        registry = ModelRegistry()
        params = {'n_estimators': 50, 'max_depth': 10}
        estimator = registry.create_estimator('random_forest', params=params)
        assert estimator.n_estimators == 50
        assert estimator.max_depth == 10

    def test_create_estimator_filters_invalid_params(self):
        """Test that invalid params are filtered out (bug fix v1.3.0)."""
        registry = ModelRegistry()
        # Pass a param that SVR doesn't accept - should not raise
        params = {'C': 1.0, 'invalid_param': 999}
        estimator = registry.create_estimator('svr', params=params)
        assert hasattr(estimator, 'C')
        assert estimator.C == 1.0

    def test_create_estimator_problem_type_validation(self):
        """Test that problem type is validated."""
        registry = ModelRegistry()
        # Ridge doesn't support classification
        with pytest.raises(ValueError):
            registry.create_estimator('ridge', problem_type='classification')

    def test_list_models_filter_regression(self):
        """Test listing only regression models."""
        registry = ModelRegistry()
        models = registry.list_models(problem_type='regression')
        assert 'random_forest' in models
        assert 'ridge' in models

    def test_list_models_filter_classification(self):
        """Test listing only classification models."""
        registry = ModelRegistry()
        models = registry.list_models(problem_type='classification')
        assert 'random_forest' in models  # RF supports both
        # Ridge doesn't support classification
        assert 'ridge' not in models

    def test_list_models_gpu_only(self):
        """Test listing GPU-supported models."""
        registry = ModelRegistry()
        models = registry.list_models(gpu_only=True)
        # Most default models don't support GPU
        assert isinstance(models, list)

    def test_get_model_info(self):
        """Test getting info for all models."""
        registry = ModelRegistry()
        info = registry.get_model_info()
        assert isinstance(info, dict)
        assert 'random_forest' in info
        assert 'display_name' in info['random_forest']
        assert 'description' in info['random_forest']

    def test_get_param_grid(self):
        """Test generating parameter grid."""
        registry = ModelRegistry()
        grid = registry.get_param_grid('random_forest', n_points=3)
        assert isinstance(grid, dict)
        assert 'n_estimators' in grid
        assert len(grid['n_estimators']) == 3

    def test_register_custom_model(self):
        """Test registering a custom model."""
        registry = ModelRegistry()
        custom_model = ModelInfo(
            name='custom_model',
            display_name='Custom Model',
            description='A custom model',
            supports_classification=False,
            supports_regression=True,
            supports_gpu=False,
            default_params={'alpha': 1.0},
            param_ranges={'alpha': (0.1, 10.0, 'float')},
            estimator_class=Ridge
        )
        registry.register(custom_model)
        assert 'custom_model' in registry.list_models()

    def test_get_classifier_class_xgboost(self):
        """Test classifier mapping for XGBoost."""
        try:
            import xgboost as xgb
        except ImportError:
            pytest.skip("xgboost not installed")
        registry = ModelRegistry()
        clf_class = registry._get_classifier_class(xgb.XGBRegressor)
        assert clf_class == xgb.XGBClassifier

    def test_get_classifier_class_lightgbm(self):
        """Test classifier mapping for LightGBM."""
        registry = ModelRegistry()
        import lightgbm as lgb
        clf_class = registry._get_classifier_class(lgb.LGBMRegressor)
        assert clf_class == lgb.LGBMClassifier

    def test_get_classifier_class_catboost(self):
        """Test classifier mapping for CatBoost."""
        try:
            import catboost as cb
        except ImportError:
            pytest.skip("catboost not installed")
        registry = ModelRegistry()
        clf_class = registry._get_classifier_class(cb.CatBoostRegressor)
        assert clf_class == cb.CatBoostClassifier

    def test_gradient_boosting_int_params_preserved(self):
        """Test that int params like min_samples_leaf keep their type (bug fix)."""
        registry = ModelRegistry()
        estimator = registry.create_estimator('gradient_boosting', params={'min_samples_leaf': 1})
        # min_samples_leaf should remain int, not float
        assert estimator.min_samples_leaf == 1
        assert isinstance(estimator.min_samples_leaf, int)
