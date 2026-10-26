"""Tests for GridSearchOptimizer (AutoAIM v2.0)."""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch

from app.core.optimizer import GridSearchOptimizer, OptimizationResult


class TestGridSearchOptimizer:
    """Test GridSearchOptimizer functionality."""

    def test_initialization(self):
        """Test initialization with different resolutions."""
        gs = GridSearchOptimizer(resolution='coarse')
        assert gs.n_points == 3

        gs = GridSearchOptimizer(resolution='medium')
        assert gs.n_points == 5

        gs = GridSearchOptimizer(resolution='fine')
        assert gs.n_points == 7

        gs = GridSearchOptimizer(resolution='unknown')
        assert gs.n_points == 5  # default fallback

    def test_ranges_to_grid_int(self):
        """Test integer parameter grid generation."""
        gs = GridSearchOptimizer(resolution='medium')
        ranges = {'n_estimators': (10, 50, 'int')}
        grid = gs._ranges_to_grid(ranges)
        assert 'n_estimators' in grid
        assert len(grid['n_estimators']) == 5
        assert all(isinstance(v, int) for v in grid['n_estimators'])
        assert min(grid['n_estimators']) >= 10
        assert max(grid['n_estimators']) <= 50

    def test_ranges_to_grid_int_log(self):
        """Test integer log-scale grid generation."""
        gs = GridSearchOptimizer(resolution='coarse')
        ranges = {'max_depth': (2, 32, 'int_log')}
        grid = gs._ranges_to_grid(ranges)
        assert 'max_depth' in grid
        vals = grid['max_depth']
        assert all(isinstance(v, int) for v in vals)
        assert min(vals) >= 2
        assert max(vals) <= 32

    def test_ranges_to_grid_float(self):
        """Test float parameter grid generation."""
        gs = GridSearchOptimizer(resolution='medium')
        ranges = {'learning_rate': (0.01, 0.3, 'float')}
        grid = gs._ranges_to_grid(ranges)
        assert 'learning_rate' in grid
        assert len(grid['learning_rate']) == 5
        vals = grid['learning_rate']
        assert min(vals) >= 0.009  # allow rounding
        assert max(vals) <= 0.31

    def test_ranges_to_grid_float_log(self):
        """Test float log-scale grid generation."""
        gs = GridSearchOptimizer(resolution='coarse')
        ranges = {'learning_rate': (1e-4, 1e-1, 'float_log')}
        grid = gs._ranges_to_grid(ranges)
        assert 'learning_rate' in grid
        vals = grid['learning_rate']
        assert all(v > 0 for v in vals)
        assert min(vals) >= 1e-5
        assert max(vals) <= 1.0

    def test_ranges_to_grid_categorical(self):
        """Test categorical parameter grid generation."""
        gs = GridSearchOptimizer(resolution='medium')
        ranges = {'kernel': (['rbf', 'linear', 'poly'], None, 'categorical')}
        grid = gs._ranges_to_grid(ranges)
        assert 'kernel' in grid
        assert set(grid['kernel']) == {'rbf', 'linear', 'poly'}

    def test_ranges_to_grid_with_selected_params(self):
        """Test that unselected params are excluded from grid."""
        gs = GridSearchOptimizer(resolution='medium')
        ranges = {
            'n_estimators': (10, 50, 'int'),
            'max_depth': (2, 10, 'int'),
        }
        selected = {'n_estimators': True, 'max_depth': False}
        grid = gs._ranges_to_grid(ranges, selected)
        assert 'n_estimators' in grid
        assert 'max_depth' not in grid

    def test_estimate_combinations(self):
        """Test combination estimation."""
        gs = GridSearchOptimizer(resolution='medium')
        grid = {'a': [1, 2, 3], 'b': [4, 5]}
        assert gs._estimate_combinations(grid) == 6

    def test_estimate_combinations_empty(self):
        """Test empty grid returns 1."""
        gs = GridSearchOptimizer(resolution='medium')
        assert gs._estimate_combinations({}) == 1

    def test_too_many_combinations_raises(self):
        """Test that >5000 combinations raises ValueError."""
        gs = GridSearchOptimizer(resolution='fine')
        # Build a grid with >5000 combinations directly
        param_grid = {
            'n_estimators': list(range(7)),
            'max_depth': list(range(7)),
            'min_samples_split': list(range(7)),
            'min_samples_leaf': list(range(7)),
            'max_features': list(range(7)),
        }
        n = gs._estimate_combinations(param_grid)
        assert n == 16807  # 7^5
        # Now test that optimize_sklearn_model rejects it
        X = np.random.randn(10, 2)
        y = np.random.randn(10)
        param_ranges = {
            'n_estimators': (10, 200, 'int'),
            'max_depth': (2, 100, 'int'),
            'min_samples_split': (2, 50, 'int'),
            'min_samples_leaf': (1, 30, 'int'),
            'max_features': (['sqrt', 'log2', None, 0.3, 0.5, 0.7, 0.9], None, 'categorical'),
        }
        with pytest.raises(ValueError, match="too many"):
            gs.optimize_sklearn_model(
                model_type="Random Forest",
                algorithm="Random Forest",
                X_train=X, y_train=y,
                param_ranges=param_ranges,
                cv_folds=2,
            )

    @pytest.mark.skipif(True, reason="Requires sklearn installed")
    def test_optimize_sklearn_model_returns_result(self):
        """Test that optimize_sklearn_model returns OptimizationResult."""
        gs = GridSearchOptimizer(resolution='coarse')
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        param_ranges = {'n_estimators': (10, 20, 'int')}
        result = gs.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X, y_train=y,
            X_test=X, y_test=y,
            param_ranges=param_ranges,
            cv_folds=2,
            random_state=42,
        )
        assert isinstance(result, OptimizationResult)
        assert result.search_strategy == "grid"
        assert result.best_model is not None
        assert result.n_trials > 0

    @pytest.mark.skipif(True, reason="Requires sklearn installed")
    def test_optimize_sklearn_model_classification(self):
        """Test GridSearch on classification problem."""
        gs = GridSearchOptimizer(resolution='coarse')
        X = np.random.randn(20, 3)
        y = np.array([0, 1] * 10)
        param_ranges = {'n_neighbors': (2, 5, 'int')}
        result = gs.optimize_sklearn_model(
            model_type="k-Nearest Neighbors",
            algorithm="k-Nearest Neighbors",
            X_train=X, y_train=y,
            X_test=X, y_test=y,
            problem_type='classification',
            param_ranges=param_ranges,
            cv_folds=2,
        )
        assert isinstance(result, OptimizationResult)
        assert result.search_strategy == "grid"

    def test_optimize_with_progress_callback(self):
        """Test that progress_callback is called."""
        gs = GridSearchOptimizer(resolution='coarse')
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        param_ranges = {'n_estimators': (10, 20, 'int')}
        calls = []

        def cb(trial_num, score, params):
            calls.append((trial_num, score, params))

        result = gs.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X, y_train=y,
            param_ranges=param_ranges,
            cv_folds=2,
            progress_callback=cb,
            random_state=42,
        )
        assert len(calls) >= 1
        assert calls[0][0] == 0  # trial_num

    def test_get_model_class_same_as_bayesian(self):
        """Verify _get_model_class produces same classes as HyperparameterOptimizer."""
        from app.core.optimizer import HyperparameterOptimizer
        gs = GridSearchOptimizer()
        bayes = HyperparameterOptimizer()
        # Only test algorithms available in the environment
        available = ["Random Forest", "Gradient Boosting"]
        try:
            import xgboost
            available.append("XGBoost")
        except ImportError:
            pass
        try:
            import lightgbm
            available.append("LightGBM")
        except ImportError:
            pass
        for algo in available:
            cls_b = bayes._get_model_class(algo, 'regression')
            cls_g = gs._get_model_class(algo, 'regression')
            assert cls_b == cls_g, f"Mismatch for {algo}"

    def test_all_trials_dataframe(self):
        """Test that all_trials is a DataFrame with all grid points."""
        gs = GridSearchOptimizer(resolution='coarse')
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        param_ranges = {'n_estimators': (10, 20, 'int')}
        result = gs.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X, y_train=y,
            param_ranges=param_ranges,
            cv_folds=2,
            random_state=42,
        )
        assert result.all_trials is not None
        assert len(result.all_trials) == result.n_trials
        assert 'number' in result.all_trials.columns
        assert 'value' in result.all_trials.columns
        assert 'params' in result.all_trials.columns

    def test_optimization_config_contains_strategy(self):
        """Test that optimization_config records grid settings."""
        gs = GridSearchOptimizer(resolution='medium')
        X = np.random.randn(20, 3)
        y = np.random.randn(20)
        param_ranges = {'n_estimators': (10, 20, 'int')}
        result = gs.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X, y_train=y,
            param_ranges=param_ranges,
            cv_folds=2,
            random_state=42,
        )
        assert result.optimization_config['search_strategy'] == 'grid'
        assert result.optimization_config['grid_resolution'] == 'medium'
        assert 'n_combinations' in result.optimization_config
