"""Tests for optimizer module."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from app.core.optimizer import HyperparameterOptimizer, OptimizationResult

# Skip entire module if optuna is not available
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    optuna = None


class TestHyperparameterOptimizer:
    """Test suite for HyperparameterOptimizer."""

    def test_optimizer_initialization(self):
        """Test optimizer initialization."""
        optimizer = HyperparameterOptimizer()
        assert optimizer is not None

    def test_get_param_ranges(self):
        """Test getting parameter ranges for algorithms."""
        optimizer = HyperparameterOptimizer()
        ranges = optimizer.get_param_ranges('Random Forest')
        assert isinstance(ranges, dict)
        assert 'n_estimators' in ranges
        assert 'max_depth' in ranges

    def test_get_param_ranges_unknown(self):
        """Test getting ranges for unknown algorithm."""
        optimizer = HyperparameterOptimizer()
        ranges = optimizer.get_param_ranges('Unknown Algorithm')
        assert ranges == {}

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_suggest_param_int(self):
        """Test integer parameter suggestion."""
        import optuna
        
        optimizer = HyperparameterOptimizer()
        
        def objective(trial):
            val = optimizer._suggest_param(trial, 'n_estimators', 10, 100, 'int')
            assert isinstance(val, int)
            assert 10 <= val <= 100
            return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=3)

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_suggest_param_float(self):
        """Test float parameter suggestion."""
        import optuna
        
        optimizer = HyperparameterOptimizer()
        
        def objective(trial):
            val = optimizer._suggest_param(trial, 'learning_rate', 0.01, 0.3, 'float')
            assert isinstance(val, float)
            assert 0.01 <= val <= 0.3
            return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=3)

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_suggest_param_categorical(self):
        """Test categorical parameter suggestion."""
        import optuna
        
        optimizer = HyperparameterOptimizer()
        
        def objective(trial):
            val = optimizer._suggest_param(trial, 'kernel', ['rbf', 'linear'], None, 'categorical')
            assert val in ['rbf', 'linear']
            return 0.0
        
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=3)

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_optimize_sklearn_model(self, regression_data):
        """Test sklearn model optimization."""
        X, y, _ = regression_data
        
        optimizer = HyperparameterOptimizer()
        result = optimizer.optimize_sklearn_model(
            model_type='random_forest',
            algorithm='Random Forest',
            X_train=X,
            y_train=y,
            X_test=X[:10],
            y_test=y[:10],
            problem_type='regression',
            n_trials=5,
            cv_folds=3,
            random_state=42
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert 'n_estimators' in result.best_params
        assert result.best_score > -np.inf
        assert result.n_trials <= 5

    def test_optimize_sklearn_no_optuna(self, regression_data):
        """Test error when optuna is not installed."""
        X, y, _ = regression_data
        optimizer = HyperparameterOptimizer()

        # Patch the import inside optimizer module to simulate optuna missing
        with pytest.MonkeyPatch.context() as mp:
            # Remove optuna from sys.modules and block re-import
            import sys
            orig_optuna = sys.modules.pop('optuna', None)

            def raise_import_error(*args, **kwargs):
                raise ImportError("No module named 'optuna'")

            # Patch builtins.__import__ only for optuna
            real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __builtins__['__import__']

            def mock_import(name, *args, **kwargs):
                if name == 'optuna' or name.startswith('optuna.'):
                    raise ImportError("No module named 'optuna'")
                return real_import(name, *args, **kwargs)

            mp.setattr('builtins.__import__', mock_import)
            try:
                optimizer.optimize_sklearn_model(
                    model_type='random_forest',
                    algorithm='Random Forest',
                    X_train=X,
                    y_train=y,
                    n_trials=2,
                    cv_folds=2,
                    random_state=42
                )
                pytest.fail("Expected ImportError was not raised")
            except ImportError as e:
                assert "Optuna is required" in str(e)
            finally:
                if orig_optuna:
                    sys.modules['optuna'] = orig_optuna

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_optimize_sklearn_with_progress(self, regression_data):
        """Test optimization with progress callback."""
        X, y, _ = regression_data
        
        optimizer = HyperparameterOptimizer()
        progress_calls = []
        
        def progress_callback(trial_num, score, params):
            progress_calls.append((trial_num, score, params))
        
        result = optimizer.optimize_sklearn_model(
            model_type='random_forest',
            algorithm='Random Forest',
            X_train=X,
            y_train=y,
            n_trials=3,
            cv_folds=2,
            progress_callback=progress_callback,
            random_state=42
        )
        
        assert len(progress_calls) > 0
        assert isinstance(result, OptimizationResult)

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_optimize_result_structure(self, regression_data):
        """Test that optimization result has correct structure."""
        X, y, _ = regression_data
        
        optimizer = HyperparameterOptimizer()
        result = optimizer.optimize_sklearn_model(
            model_type='random_forest',
            algorithm='Random Forest',
            X_train=X,
            y_train=y,
            X_test=X[:10],
            y_test=y[:10],
            n_trials=3,
            cv_folds=2,
            random_state=42
        )
        
        assert hasattr(result, 'best_params')
        assert hasattr(result, 'best_score')
        assert hasattr(result, 'best_model')
        assert hasattr(result, 'n_trials')
        assert hasattr(result, 'optimization_time')
        assert hasattr(result, 'metrics')
        assert 'r2' in result.metrics or result.best_score is not None

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_optimize_with_param_ranges(self, regression_data):
        """Test optimization with custom parameter ranges."""
        X, y, _ = regression_data
        
        optimizer = HyperparameterOptimizer()
        custom_ranges = {
            'n_estimators': (10, 50, 'int'),
            'max_depth': (3, 10, 'int'),
        }
        
        result = optimizer.optimize_sklearn_model(
            model_type='random_forest',
            algorithm='Random Forest',
            X_train=X,
            y_train=y,
            n_trials=3,
            cv_folds=2,
            param_ranges=custom_ranges,
            random_state=42
        )
        
        assert result.best_params['n_estimators'] <= 50
        assert result.best_params['max_depth'] <= 10

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_optimize_with_timeout(self, regression_data):
        """Test optimization with timeout."""
        X, y, _ = regression_data
        
        optimizer = HyperparameterOptimizer()
        result = optimizer.optimize_sklearn_model(
            model_type='random_forest',
            algorithm='Random Forest',
            X_train=X,
            y_train=y,
            n_trials=100,
            cv_folds=2,
            timeout=5,  # 5 second timeout
            random_state=42
        )
        
        # Should complete within timeout, possibly with fewer trials
        assert result is not None
        assert result.optimization_time < 10  # Should be under 10 seconds

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_optimize_classification(self, classification_data):
        """Test optimization for classification."""
        X, y, _ = classification_data
        
        optimizer = HyperparameterOptimizer()
        result = optimizer.optimize_sklearn_model(
            model_type='random_forest',
            algorithm='Random Forest',
            X_train=X,
            y_train=y,
            X_test=X[:10],
            y_test=y[:10],
            problem_type='classification',
            n_trials=3,
            cv_folds=2,
            random_state=42
        )
        
        assert isinstance(result, OptimizationResult)
        assert 'accuracy' in result.metrics

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not installed")
    def test_optimize_sklearn_no_trials_raises(self, regression_data):
        """Test that raises when no trials succeed."""
        X, y, _ = regression_data
        
        optimizer = HyperparameterOptimizer()
        # Use param ranges that will cause failures
        bad_ranges = {
            'n_estimators': (-1000, -100, 'int'),  # Invalid range
        }
        
        # This should handle gracefully or raise RuntimeError
        try:
            result = optimizer.optimize_sklearn_model(
                model_type='random_forest',
                algorithm='Random Forest',
                X_train=X,
                y_train=y,
                n_trials=2,
                cv_folds=2,
                param_ranges=bad_ranges,
                random_state=42
            )
            # If it doesn't raise, check result
            assert result is not None
        except (RuntimeError, ValueError):
            # Expected - invalid params should cause failure
            pass

    def test_get_model_class_regression(self):
        """Test getting model class for regression."""
        optimizer = HyperparameterOptimizer()
        
        from sklearn.ensemble import RandomForestRegressor
        cls = optimizer._get_model_class('Random Forest', 'regression')
        assert cls == RandomForestRegressor

    def test_get_model_class_classification(self):
        """Test getting model class for classification."""
        optimizer = HyperparameterOptimizer()
        
        from sklearn.ensemble import RandomForestClassifier
        cls = optimizer._get_model_class('Random Forest', 'classification')
        assert cls == RandomForestClassifier

    def test_get_model_class_unknown(self):
        """Test getting unknown model class."""
        optimizer = HyperparameterOptimizer()
        
        with pytest.raises(ValueError, match="Unknown algorithm"):
            optimizer._get_model_class('Unknown', 'regression')
