"""Tests for Leave-One-Out Cross-Validation (LOO-CV) integration.

v2.0: LOO-CV is available for small datasets (N ≤ 100) with sklearn regressors.
NNs and Ensembles use K-Fold exclusively.
"""

import numpy as np
import pytest

from app.core.trainer import ModelTrainer, TrainingResult
from app.core.optimizer import GridSearchOptimizer, OptimizationResult

# Check optional dependencies
try:
    import optuna
    from app.core.optimizer import HyperparameterOptimizer
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


class TestLOOCVTrainer:
    """Test LOO-CV in ModelTrainer."""

    def test_cross_validate_loo(self):
        """LOO-CV should produce N scores for N samples."""
        trainer = ModelTrainer()
        X = np.random.randn(15, 5)
        y = np.random.randn(15)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)

        cv_metrics = trainer._cross_validate(
            model, X, y, 'regression', cv_folds=5, cv_strategy='loo'
        )

        assert 'test_score' in cv_metrics
        assert len(cv_metrics['test_score']) == 15  # N MAE scores for N samples
        assert cv_metrics.get('test_score_metric') == 'mae'
        assert 'r2_global' in cv_metrics
        assert cv_metrics['r2_global'] is not None and not np.isnan(cv_metrics['r2_global'])
        # train_score is empty for LOO-CV (not meaningful per-sample)
        assert 'train_score' in cv_metrics

    def test_cross_validate_kfold(self):
        """K-Fold should produce K scores."""
        trainer = ModelTrainer()
        X = np.random.randn(50, 5)
        y = np.random.randn(50)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)

        cv_metrics = trainer._cross_validate(
            model, X, y, 'regression', cv_folds=5, cv_strategy='kfold'
        )

        assert len(cv_metrics['test_score']) == 5

    def test_estimate_cv_residuals_loo(self):
        """LOO-CV residual estimation should work."""
        trainer = ModelTrainer()
        X = np.random.randn(12, 3)
        y = np.random.randn(12)

        from sklearn.linear_model import Ridge
        model = Ridge(alpha=1.0)
        model.fit(X, y)

        ci = trainer._estimate_cv_residuals(
            model, X, y, 'regression', cv_folds=5, cv_strategy='loo'
        )

        assert ci['type'] == 'regression'
        assert 'residuals_std' in ci
        assert ci['n_samples'] == 12

    def test_training_result_has_cv_strategy(self):
        """TrainingResult dataclass should include cv_strategy field."""
        result = TrainingResult(
            model=None,
            model_name="Test",
            problem_type="regression",
            metrics={},
            cv_metrics={},
            training_time=1.0,
            cv_strategy='loo',
        )
        assert result.cv_strategy == 'loo'


@pytest.mark.skipif(not HAS_OPTUNA, reason="Optuna not installed")
class TestLOOCVBayesianOptimizer:
    """Test LOO-CV in Bayesian optimization."""

    def test_optimize_with_loo(self):
        """Bayesian optimization should accept cv_strategy='loo'."""
        opt = HyperparameterOptimizer()
        X = np.random.randn(18, 4)
        y = np.random.randn(18)

        result = opt.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X,
            y_train=y,
            problem_type='regression',
            n_trials=3,
            cv_folds=5,
            random_state=42,
            pruning=False,
            cv_strategy='loo',
            default_params={'n_estimators': 5},
        )

        assert isinstance(result, OptimizationResult)
        assert result.cv_strategy == 'loo'
        assert result.search_strategy == 'bayesian'
        # LOO with N=18 should produce 18 CV scores
        assert len(result.cv_metrics['test_score']) == 18

    def test_optimize_with_kfold(self):
        """Bayesian optimization with kfold should produce K scores."""
        opt = HyperparameterOptimizer()
        X = np.random.randn(40, 4)
        y = np.random.randn(40)

        result = opt.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X,
            y_train=y,
            problem_type='regression',
            n_trials=3,
            cv_folds=5,
            random_state=42,
            pruning=False,
            cv_strategy='kfold',
            default_params={'n_estimators': 5},
        )

        assert result.cv_strategy == 'kfold'
        assert len(result.cv_metrics['test_score']) == 5


class TestLOOCVGridSearchOptimizer:
    """Test LOO-CV in Grid Search optimization."""

    def test_grid_search_with_loo(self):
        """GridSearch should accept cv_strategy='loo'."""
        gs = GridSearchOptimizer(resolution='coarse')
        X = np.random.randn(16, 3)
        y = np.random.randn(16)

        # Ultra-small param grid: 3 combinations max
        result = gs.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X,
            y_train=y,
            problem_type='regression',
            param_ranges={'max_depth': (2, 4, 'int')},
            cv_folds=5,
            random_state=42,
            cv_strategy='loo',
            default_params={'n_estimators': 5},
            selected_params={'max_depth': True},
        )

        assert isinstance(result, OptimizationResult)
        assert result.cv_strategy == 'loo'
        assert result.search_strategy == 'grid'
        # LOO with N=16 should produce 16 CV scores
        assert len(result.cv_metrics['test_score']) == 16

    def test_grid_search_with_kfold(self):
        """GridSearch with kfold should produce K scores."""
        gs = GridSearchOptimizer(resolution='coarse')
        X = np.random.randn(30, 3)
        y = np.random.randn(30)

        result = gs.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X,
            y_train=y,
            problem_type='regression',
            param_ranges={'max_depth': (2, 4, 'int')},
            cv_folds=5,
            random_state=42,
            cv_strategy='kfold',
            default_params={'n_estimators': 5},
            selected_params={'max_depth': True},
        )

        assert result.cv_strategy == 'kfold'
        assert len(result.cv_metrics['test_score']) == 5

    def test_grid_search_loo_dynamic_split_detection(self):
        """GridSearch LOO should dynamically detect number of splits in cv_results."""
        gs = GridSearchOptimizer(resolution='coarse')
        X = np.random.randn(20, 3)
        y = np.random.randn(20)

        result = gs.optimize_sklearn_model(
            model_type="Random Forest",
            algorithm="Random Forest",
            X_train=X,
            y_train=y,
            problem_type='regression',
            param_ranges={'max_depth': (2, 4, 'int')},
            cv_folds=5,  # ignored when cv_strategy='loo'
            random_state=42,
            cv_strategy='loo',
            default_params={'n_estimators': 5},
            selected_params={'max_depth': True},
        )

        # Even though cv_folds=5 was passed, LOO produces 20 splits
        assert len(result.cv_metrics['test_score']) == 20


class TestOptimizationResultFields:
    """Test OptimizationResult dataclass fields."""

    def test_has_cv_strategy_field(self):
        """OptimizationResult should include cv_strategy field."""
        result = OptimizationResult(
            best_params={},
            best_score=0.5,
            best_model=None,
            n_trials=10,
            optimization_time=1.0,
            search_strategy='bayesian',
            cv_strategy='loo',
        )
        assert result.cv_strategy == 'loo'
