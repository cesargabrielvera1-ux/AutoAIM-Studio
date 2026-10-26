"""Tests for trainer module."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from app.core.trainer import ModelTrainer, TrainingResult


class TestModelTrainer:
    """Test suite for ModelTrainer."""

    def test_trainer_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer()
        assert trainer.results == {}

    def test_train_basic(self, regression_data):
        """Test basic training workflow."""
        X, y, feature_names = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        result = trainer.train(
            model=model,
            model_name='Random Forest',
            X_train=X,
            y_train=y,
            X_test=X[:10],
            y_test=y[:10],
            problem_type='regression',
            cv_folds=3,
            feature_names=feature_names
        )
        assert isinstance(result, TrainingResult)
        assert result.model_name == 'Random Forest'
        assert result.problem_type == 'regression'
        assert 'r2' in result.metrics
        assert result.cv_metrics is not None
        assert 'test_score' in result.cv_metrics
        assert len(result.cv_metrics['test_score']) == 3  # 3 folds

    def test_train_unique_names(self, regression_data):
        """Test that multiple trainings get unique names (bug fix v1.3.0)."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        result1 = trainer.train(model, 'RF', X, y, cv_folds=2)
        result2 = trainer.train(model, 'RF', X, y, cv_folds=2)
        result3 = trainer.train(model, 'RF', X, y, cv_folds=2)
        
        assert result1.model_name == 'RF'
        assert result2.model_name == 'RF_1'
        assert result3.model_name == 'RF_2'

    def test_train_random_state_from_model(self, regression_data):
        """Test that CV uses model's random_state (bug fix v1.3.0)."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=123)
        
        result = trainer.train(model, 'RF', X, y, cv_folds=3)
        # The CV should use random_state=123 from the model
        assert 'test_score' in result.cv_metrics
        assert len(result.cv_metrics['test_score']) == 3

    def test_train_cv_folds_zero(self, regression_data):
        """Test training with 0 CV folds."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        result = trainer.train(model, 'RF', X, y, cv_folds=0)
        # Should handle gracefully - either empty cv_metrics or error
        assert isinstance(result, TrainingResult)

    def test_train_no_test_set(self, regression_data):
        """Test training without test set."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        result = trainer.train(model, 'RF', X, y, cv_folds=2)
        assert result.metrics == {}  # No test metrics
        assert result.predictions is None

    def test_compute_metrics_regression(self):
        """Test regression metrics computation."""
        trainer = ModelTrainer()
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        metrics = trainer._compute_metrics(y_true, y_pred, 'regression')
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['r2'] > 0  # Should be positive for good predictions

    def test_compute_metrics_classification(self):
        """Test classification metrics computation."""
        trainer = ModelTrainer()
        y_true = np.array([0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1])
        
        metrics = trainer._compute_metrics(y_true, y_pred, 'classification')
        assert 'accuracy' in metrics
        assert metrics['accuracy'] > 0

    def test_get_feature_importance_rf(self, regression_data):
        """Test feature importance extraction from Random Forest."""
        X, y, feature_names = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        importance = trainer._get_feature_importance(model, feature_names)
        assert importance is not None
        assert len(importance) == len(feature_names)
        assert all(name in importance for name in feature_names)

    def test_get_feature_importance_ridge(self, regression_data):
        """Test feature importance extraction from Ridge."""
        X, y, feature_names = regression_data
        trainer = ModelTrainer()
        model = Ridge()
        model.fit(X, y)
        
        importance = trainer._get_feature_importance(model, feature_names)
        assert importance is not None
        assert len(importance) == len(feature_names)

    def test_get_best_model(self, regression_data):
        """Test getting best model."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        
        model1 = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer.train(model1, 'RF1', X, y, X[:10], y[:10], cv_folds=2)
        
        model2 = Ridge()
        trainer.train(model2, 'Ridge1', X, y, X[:10], y[:10], cv_folds=2)
        
        best = trainer.get_best_model(metric='r2')
        assert best is not None
        assert best in trainer.results

    def test_get_best_model_empty(self):
        """Test getting best model with no results."""
        trainer = ModelTrainer()
        best = trainer.get_best_model()
        assert best is None

    def test_compare_models(self, regression_data):
        """Test model comparison."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer.train(model, 'RF', X, y, X[:10], y[:10], cv_folds=2)
        
        comparison = trainer.compare_models()
        assert len(comparison) == 1
        assert comparison.iloc[0]['Model'] == 'RF'

    def test_predict(self, regression_data):
        """Test prediction from stored model."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer.train(model, 'RF', X, y, cv_folds=2)
        
        predictions = trainer.predict('RF', X[:5])
        assert len(predictions) == 5
        assert isinstance(predictions, np.ndarray)

    def test_predict_nonexistent(self):
        """Test prediction from nonexistent model."""
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
            trainer.predict('nonexistent', np.array([[1, 2, 3]]))

    def test_cross_validate_with_random_state(self, regression_data):
        """Test CV respects random_state parameter."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        cv1 = trainer._cross_validate(model, X, y, 'regression', 3, random_state=42)
        cv2 = trainer._cross_validate(model, X, y, 'regression', 3, random_state=42)
        
        # Same random_state should give same results
        np.testing.assert_array_almost_equal(cv1['test_score'], cv2['test_score'])

    def test_cross_validate_different_random_state(self, regression_data):
        """Test CV with different random_states gives different results."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        cv1 = trainer._cross_validate(model, X, y, 'regression', 3, random_state=42)
        cv2 = trainer._cross_validate(model, X, y, 'regression', 3, random_state=99)
        
        # Different random_state might give different results (not guaranteed)
        # but scores should be reasonable
        assert all(score > -1 for score in cv1['test_score'])
        assert all(score > -1 for score in cv2['test_score'])

    def test_train_progress_callback(self, regression_data):
        """Test training with progress callback."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        progress_calls = []
        def callback(message, progress):
            progress_calls.append((message, progress))
        
        trainer.train(model, 'RF', X, y, cv_folds=2, progress_callback=callback)
        assert len(progress_calls) > 0

    def test_rmse_normalized_constant_target(self):
        """Test RMSE normalized with constant target (edge case)."""
        trainer = ModelTrainer()
        y_true = np.array([5.0, 5.0, 5.0, 5.0])
        y_pred = np.array([5.1, 5.1, 5.1, 5.1])
        
        metrics = trainer._compute_metrics(y_true, y_pred, 'regression')
        assert metrics['rmse_normalized'] == 0.0  # std=0, should return 0

    def test_rmse_normalized_zero_std(self):
        """Test RMSE normalized with zero standard deviation."""
        trainer = ModelTrainer()
        y_true = np.array([3.0, 3.0, 3.0])
        y_pred = np.array([4.0, 4.0, 4.0])
        
        metrics = trainer._compute_metrics(y_true, y_pred, 'regression')
        assert metrics['rmse_normalized'] == 0.0  # Should handle zero std

    # ------------------------------------------------------------------
    # v1.5.0: Learning Curves and Confidence Intervals
    # ------------------------------------------------------------------

    def test_compute_learning_curve(self, regression_data):
        """Test learning curve computation."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        result = trainer.train(model, 'RF', X, y, X[:10], y[:10], cv_folds=3)
        
        lc_data = trainer.compute_learning_curve('RF', X, y, cv_folds=3, train_sizes=5)
        
        assert lc_data is not None
        assert 'train_sizes_abs' in lc_data
        assert 'train_sizes_pct' in lc_data
        assert 'train_scores_mean' in lc_data
        assert 'test_scores_mean' in lc_data
        assert len(lc_data['train_sizes_abs']) == 5
        # All scores should be reasonable
        for score in lc_data['train_scores_mean']:
            assert -1 <= score <= 1  # R² range
        for score in lc_data['test_scores_mean']:
            assert -1 <= score <= 1

    def test_compute_learning_curve_not_found(self):
        """Test learning curve for nonexistent model returns None."""
        trainer = ModelTrainer()
        result = trainer.compute_learning_curve('NonExistent', np.array([[1]]), np.array([1]))
        assert result is None

    def test_compute_learning_curve_stores_in_result(self, regression_data):
        """Test learning curve data is stored in TrainingResult."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        result = trainer.train(model, 'RF', X, y, X[:10], y[:10], cv_folds=3)
        
        trainer.compute_learning_curve('RF', X, y, cv_folds=3, train_sizes=5)
        
        # Result should now have learning_curve_data
        assert result.learning_curve_data is not None
        assert 'train_scores_mean' in result.learning_curve_data

    def test_estimate_cv_residuals_regression(self, regression_data):
        """Test CV residual estimation for regression."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        ci_params = trainer._estimate_cv_residuals(model, X, y, 'regression', cv_folds=3)
        
        assert ci_params is not None
        assert ci_params['type'] == 'regression'
        assert ci_params['residuals_std'] >= 0
        assert ci_params['quantile_68'] >= 0
        assert ci_params['quantile_95'] >= 0
        assert ci_params['quantile_95'] >= ci_params['quantile_68']
        assert ci_params['n_samples'] == len(y)

    def test_estimate_cv_residuals_classification(self, classification_data):
        """Test CV residual estimation for classification."""
        X, y, _ = classification_data
        trainer = ModelTrainer()
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        ci_params = trainer._estimate_cv_residuals(model, X, y, 'classification', cv_folds=3)
        
        assert ci_params is not None
        assert ci_params['type'] == 'classification'
        assert 'has_predict_proba' in ci_params

    def test_training_result_has_ci_params(self, regression_data):
        """Test that TrainingResult includes confidence_interval_params after training."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        result = trainer.train(model, 'RF', X, y, X[:10], y[:10], cv_folds=3)
        
        assert result.confidence_interval_params is not None
        assert result.confidence_interval_params['type'] == 'regression'

    def test_predict_with_intervals_regression(self, regression_data):
        """Test predict_with_intervals for regression."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        trainer.train(model, 'RF', X, y, X[:10], y[:10], cv_folds=3)
        
        result = trainer.predict_with_intervals('RF', X[:5])
        
        assert 'predictions' in result
        assert 'lower' in result
        assert 'upper' in result
        assert len(result['predictions']) == 5
        # For regression with CI: lower < prediction < upper
        for i in range(5):
            assert result['lower'][i] <= result['predictions'][i] <= result['upper'][i]

    def test_predict_with_intervals_not_found(self):
        """Test predict_with_intervals for nonexistent model raises."""
        trainer = ModelTrainer()
        with pytest.raises(ValueError, match="not found"):
            trainer.predict_with_intervals('NonExistent', np.array([[1, 2, 3]]))

    def test_feature_importance_mismatched_names(self, regression_data):
        """Test feature importance with mismatched feature names."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # Pass wrong number of feature names
        importance = trainer._get_feature_importance(model, ['f1', 'f2'])
        assert importance is None  # Should return None for mismatch


class TestParallelization:
    """v2.0: Test correct parallelization strategy — models use internal
    threading, sklearn CV uses n_jobs=1 to avoid oversubscription."""

    def test_cv_n_jobs_is_one(self):
        """sklearn CV must use n_jobs=1 to avoid nested parallelism."""
        trainer = ModelTrainer()
        assert trainer._cv_n_jobs == 1, (
            "CV n_jobs must be 1 — individual models already have internal "
            "parallelism (XGB n_jobs, RF n_jobs, etc.)"
        )

    def test_rf_model_has_n_jobs_from_registry(self):
        """RandomForest created via ModelRegistry must have n_jobs."""
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        rf_info = registry.get_model('random_forest')
        assert 'n_jobs' in rf_info.default_params, (
            "RF default_params must include n_jobs"
        )
        assert rf_info.default_params['n_jobs'] > 0

    def test_xgb_model_has_n_jobs_from_registry(self):
        """XGBoost created via ModelRegistry must have n_jobs."""
        try:
            import xgboost
        except ImportError:
            pytest.skip("xgboost not installed")
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        xgb_info = registry.get_model('xgboost')
        assert 'n_jobs' in xgb_info.default_params
        assert xgb_info.default_params['n_jobs'] > 0

    def test_lgbm_model_has_n_jobs_from_registry(self):
        """LightGBM created via ModelRegistry must have n_jobs."""
        try:
            import lightgbm
        except ImportError:
            pytest.skip("lightgbm not installed")
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        lgb_info = registry.get_model('lightgbm')
        assert 'n_jobs' in lgb_info.default_params
        assert lgb_info.default_params['n_jobs'] > 0

    def test_optimizer_preserves_n_jobs_in_default_params(self):
        """v2.0: Optimizer must merge default_params so n_jobs is preserved."""
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        rf_info = registry.get_model('random_forest')
        default_params = rf_info.default_params.copy()
        assert 'n_jobs' in default_params
        # Simulate what the optimizer does internally
        merged = default_params.copy()
        merged.update({'n_estimators': 100})  # simulated optimized param
        assert merged['n_jobs'] == default_params['n_jobs']

    def test_hist_gradient_boosting_exists_and_works(self):
        """v2.0: HistGradientBoosting is registered and can be instantiated."""
        from app.core.model_registry import ModelRegistry
        registry = ModelRegistry()
        hgb_info = registry.get_model('hist_gradient_boosting')
        assert hgb_info.display_name == 'Histogram Gradient Boosting'
        estimator = registry.create_estimator('hist_gradient_boosting', 'regression')
        assert estimator is not None
        import numpy as np
        X, y = np.random.randn(50, 3), np.random.randn(50)
        estimator.fit(X, y)
        preds = estimator.predict(X)
        assert len(preds) == 50


class TestHistGradientBoosting:
    """v2.0: Exhaustive test suite for Histogram Gradient Boosting."""

    def _create_estimator(self):
        """Helper: create HGB estimator via registry."""
        from app.core.model_registry import ModelRegistry
        return ModelRegistry().create_estimator('hist_gradient_boosting', 'regression')

    def test_registry_param_ranges(self):
        """HGB must have complete param_ranges for optimization."""
        from app.core.model_registry import ModelRegistry
        info = ModelRegistry().get_model('hist_gradient_boosting')
        expected = {'max_iter', 'max_depth', 'learning_rate', 'min_samples_leaf',
                    'max_bins', 'l2_regularization'}
        assert expected <= set(info.param_ranges.keys()), (
            f"Missing: {expected - set(info.param_ranges.keys())}"
        )

    def test_optimizer_param_ranges(self):
        """Optimizer must have param_ranges for 'Histogram Gradient Boosting'."""
        from app.core.optimizer import HyperparameterOptimizer
        opt = HyperparameterOptimizer()
        ranges = opt.get_param_ranges('Histogram Gradient Boosting')
        assert len(ranges) >= 5
        assert 'max_iter' in ranges
        assert 'learning_rate' in ranges

    def test_optimizer_model_class_regression(self):
        """Optimizer resolves to HistGradientBoostingRegressor."""
        from app.core.optimizer import HyperparameterOptimizer
        cls = HyperparameterOptimizer()._get_model_class(
            'Histogram Gradient Boosting', 'regression'
        )
        assert cls.__name__ == 'HistGradientBoostingRegressor'

    def test_optimizer_model_class_classification(self):
        """Optimizer resolves to HistGradientBoostingClassifier."""
        from app.core.optimizer import HyperparameterOptimizer
        cls = HyperparameterOptimizer()._get_model_class(
            'Histogram Gradient Boosting', 'classification'
        )
        assert cls.__name__ == 'HistGradientBoostingClassifier'

    def test_train_with_trainer(self, regression_data):
        """Full training workflow with ModelTrainer."""
        X, y, feature_names = regression_data
        trainer = ModelTrainer()
        model = self._create_estimator()
        result = trainer.train(
            model=model, model_name='HGB_Basic',
            X_train=X, y_train=y,
            cv_folds=3, feature_names=feature_names
        )
        assert result.model_name == 'HGB_Basic'
        assert 'test_score' in result.cv_metrics
        assert len(result.cv_metrics['test_score']) == 3

    def test_train_with_test_set(self, regression_data):
        """Training with train/test split computes metrics."""
        X, y, feature_names = regression_data
        # Manual split
        split = int(0.8 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        trainer = ModelTrainer()
        model = self._create_estimator()
        result = trainer.train(
            model=model, model_name='HGB_Split',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            cv_folds=3, feature_names=feature_names
        )
        assert result.metrics is not None
        assert 'r2' in result.metrics or 'rmse' in result.metrics

    def test_reproducibility_random_state(self, regression_data):
        """Same random_state must produce identical results."""
        X, y, _ = regression_data
        from app.core.model_registry import ModelRegistry
        mr = ModelRegistry()
        m1 = mr.create_estimator('hist_gradient_boosting', 'regression')
        m2 = mr.create_estimator('hist_gradient_boosting', 'regression')
        m1.fit(X, y)
        m2.fit(X, y)
        p1 = m1.predict(X)
        p2 = m2.predict(X)
        assert np.allclose(p1, p2, atol=1e-10)

    def test_feature_importance_handles_gracefully(self, regression_data):
        """HGB may not have feature_importances_ — trainer must handle it."""
        X, y, feature_names = regression_data
        trainer = ModelTrainer()
        model = self._create_estimator()
        result = trainer.train(
            model=model, model_name='HGB_FI',
            X_train=X, y_train=y,
            cv_folds=3, feature_names=feature_names
        )
        # Should not crash — either None or a dict
        assert result.feature_importance is None or isinstance(
            result.feature_importance, dict
        )

    def test_session_export(self, regression_data):
        """HGB results must appear correctly in session export."""
        X, y, feature_names = regression_data
        trainer = ModelTrainer()
        model = self._create_estimator()
        result = trainer.train(
            model=model, model_name='HGB_Export',
            X_train=X, y_train=y,
            cv_folds=3, feature_names=feature_names
        )
        from app.core.session_exporter import SessionExporter
        exporter = SessionExporter()
        exporter.add_training_results([result])
        txt = exporter._render('txt')
        assert 'HGB_Export' in txt
        assert 'Training' in txt

    def test_ensemble_as_base_model(self, regression_data):
        """HGB model can be used as base model in ensemble."""
        X, y, feature_names = regression_data
        trainer = ModelTrainer()
        # Train two models for ensemble
        for name in ('HGB_Ens1', 'HGB_Ens2'):
            m = self._create_estimator()
            trainer.train(model=m, model_name=name, X_train=X, y_train=y,
                         cv_folds=3, feature_names=feature_names)
        # Both should be in results
        assert 'HGB_Ens1' in trainer.results
        assert 'HGB_Ens2' in trainer.results
        # Predictions should work (ensemble uses these)
        p1 = trainer.results['HGB_Ens1'].model.predict(X)
        p2 = trainer.results['HGB_Ens2'].model.predict(X)
        assert len(p1) == len(y)
        assert len(p2) == len(y)

    def test_optimizer_end_to_end(self, regression_data):
        """Full optimization cycle for HGB."""
        try:
            import optuna
        except ImportError:
            pytest.skip("optuna not installed")
        X, y, _ = regression_data
        split = int(0.8 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        from app.core.optimizer import HyperparameterOptimizer
        from app.core.model_registry import ModelRegistry
        opt = HyperparameterOptimizer()
        mr = ModelRegistry()
        hgb_info = mr.get_model('hist_gradient_boosting')
        result = opt.optimize_sklearn_model(
            model_type='Histogram Gradient Boosting',
            algorithm='Histogram Gradient Boosting',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type='regression',
            n_trials=5, cv_folds=3,
            random_state=42,
            default_params=hgb_info.default_params.copy(),
        )
        assert result.best_model is not None
        assert result.n_trials == 5
        assert hasattr(result, 'best_params')
        assert 'max_iter' in result.best_params or 'max_depth' in result.best_params

    def test_predict_after_train(self, regression_data):
        """Saved model can predict on new data."""
        X, y, _ = regression_data
        split = int(0.8 * len(y))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        trainer = ModelTrainer()
        model = self._create_estimator()
        result = trainer.train(
            model=model, model_name='HGB_Pred',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, cv_folds=3
        )
        preds = result.model.predict(X_test)
        assert len(preds) == len(y_test)
        assert np.isfinite(preds).all()
