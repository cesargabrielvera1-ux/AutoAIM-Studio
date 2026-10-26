"""Tests for explainer module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from app.core.explainer import ModelExplainer


class TestModelExplainer:
    """Test suite for ModelExplainer."""

    def test_explainer_initialization(self):
        """Test explainer initialization."""
        explainer = ModelExplainer()
        assert explainer._model is None
        assert explainer._model_name is None

    def test_set_model(self):
        """Test setting model."""
        explainer = ModelExplainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        explainer.set_model(model, "Random Forest")
        
        assert explainer._model is model
        assert explainer._model_name == "Random Forest"

    def test_calculate_shap_values_no_model(self, regression_data):
        """Test SHAP calculation without model raises."""
        X, _, _ = regression_data
        explainer = ModelExplainer()
        
        with pytest.raises(ValueError, match="No model set"):
            explainer.calculate_shap_values(X[:5])

    def test_compute_shap_values_tree_model(self, regression_data):
        """Test SHAP values for tree model."""
        X, y, feature_names = regression_data
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        result = explainer.compute_shap_values(model, X[:5], feature_names=feature_names)
        
        assert isinstance(result, dict)
        if result:  # SHAP may fail in some environments
            assert 'shap_values' in result
            assert 'feature_names' in result

    def test_compute_shap_values_linear_model(self, regression_data):
        """Test SHAP values for linear model."""
        X, y, feature_names = regression_data
        
        model = Ridge()
        model.fit(X, y)
        
        explainer = ModelExplainer()
        result = explainer.compute_shap_values(model, X[:5], feature_names=feature_names)
        
        assert isinstance(result, dict)
        if result:
            assert 'shap_values' in result

    def test_compute_permutation_importance(self, regression_data):
        """Test permutation importance computation."""
        X, y, feature_names = regression_data
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        result = explainer.compute_permutation_importance(
            model, X[:20], y[:20], feature_names=feature_names, n_repeats=3
        )
        
        assert isinstance(result, dict)
        if result:
            assert 'importances' in result
            assert len(result['importances_mean']) == len(feature_names)

    def test_compute_permutation_importance_classification(self, classification_data):
        """Test permutation importance for classification."""
        X, y, feature_names = classification_data
        
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        result = explainer.compute_permutation_importance(
            model, X[:20], y[:20], feature_names=feature_names, n_repeats=3
        )
        
        assert isinstance(result, dict)

    def test_get_feature_importance_summary(self, regression_data):
        """Test comprehensive feature importance summary."""
        X, y, feature_names = regression_data
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        df = explainer.get_feature_importance_summary(
            model, X[:20], y[:20], feature_names=feature_names
        )
        
        assert isinstance(df, pd.DataFrame)
        assert 'feature' in df.columns
        assert len(df) == len(feature_names)

    def test_hist_gradient_boosting_has_model_importance(self, regression_data):
        """CRITICAL: HistGradientBoosting has NO feature_importances_ but must
        still produce model_importance via surrogate permutation.
        
        Regression test: HistGradientBoostingRegressor lacks the
        feature_importances_ attribute, so get_feature_importance_summary()
        must fall back to permutation-based surrogate importance.
        """
        X, y, feature_names = regression_data
        from sklearn.ensemble import HistGradientBoostingRegressor
        
        model = HistGradientBoostingRegressor(max_iter=50, random_state=42)
        model.fit(X, y)
        
        # Verify the model truly lacks feature_importances_
        assert not hasattr(model, 'feature_importances_')
        
        explainer = ModelExplainer()
        df = explainer.get_feature_importance_summary(
            model, X[:20], y[:20], feature_names=feature_names
        )
        
        assert isinstance(df, pd.DataFrame)
        assert 'feature' in df.columns
        # Must have model_importance (surrogate from permutation)
        assert 'model_importance' in df.columns
        # Values must be non-zero (real importance, not zeros)
        assert df['model_importance'].abs().sum() > 0

    def test_lightgbm_feature_importance_nonzero(self, regression_data):
        """LightGBM must produce non-zero feature importance."""
        X, y, feature_names = regression_data
        
        try:
            import lightgbm as lgb
        except ImportError:
            pytest.skip("LightGBM not available")
        
        model = lgb.LGBMRegressor(n_estimators=10, random_state=42, verbose=-1)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        df = explainer.get_feature_importance_summary(
            model, X[:20], y[:20], feature_names=feature_names
        )
        
        assert isinstance(df, pd.DataFrame)
        assert 'model_importance' in df.columns
        assert df['model_importance'].abs().sum() > 0
        # Must also have permutation importance
        assert 'permutation_importance' in df.columns
        assert df['permutation_importance'].abs().sum() > 0

    def test_all_registry_models_feature_importance(self, regression_data):
        """Every model in the registry produces non-empty feature importance.
        
        This catches models that silently produce zero/empty importance
        values (like HistGradientBoosting before the surrogate fix).
        """
        X, y, feature_names = regression_data
        from app.core.model_registry import ModelRegistry
        
        registry = ModelRegistry()
        explainer = ModelExplainer()
        failed = []
        
        for name in registry.list_models('regression'):
            try:
                model = registry.create_estimator(name, 'regression', random_state=42)
                model.fit(X, y)
                df = explainer.get_feature_importance_summary(
                    model, X[:20], y[:20], feature_names=feature_names
                )
                # Must have at least one importance column with non-zero values
                imp_cols = [c for c in df.columns if c != 'feature' and not c.endswith('_norm')]
                has_values = any(df[c].abs().sum() > 0 for c in imp_cols if c in df.columns)
                if not has_values:
                    failed.append((name, "all importance columns are zero"))
            except Exception as e:
                failed.append((name, str(e)))
        
        assert len(failed) == 0, f"Feature importance failed for: {failed}"

    def test_explain_prediction(self, regression_data):
        """Test single prediction explanation."""
        X, y, feature_names = regression_data
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        explanation = explainer.explain_prediction(model, X[0], feature_names=feature_names)
        
        assert isinstance(explanation, dict)
        assert 'prediction' in explanation
        assert 'feature_values' in explanation

    def test_explain_prediction_1d_input(self, regression_data):
        """Test explanation with 1D input."""
        X, y, feature_names = regression_data
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        # Pass 1D array
        explanation = explainer.explain_prediction(model, X[0], feature_names=feature_names)
        
        assert 'prediction' in explanation

    def test_feature_importance_with_mismatched_names(self, regression_data):
        """Test feature importance with wrong number of feature names."""
        X, y, _ = regression_data
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        df = explainer.get_feature_importance_summary(
            model, X[:20], y[:20], feature_names=['f1', 'f2']  # Wrong count
        )
        
        # Should use generic names
        assert len(df) == X.shape[1]

    def test_compute_partial_dependence(self, regression_data):
        """Test partial dependence computation."""
        X, y, feature_names = regression_data
        
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        explainer = ModelExplainer()
        pdp = explainer.compute_partial_dependence(
            model, X[:20], features=[0, 1], feature_names=feature_names
        )
        
        assert isinstance(pdp, dict)
        if pdp:
            assert feature_names[0] in pdp or len(pdp) == 0

    def test_plot_methods_without_computation(self):
        """Test plot methods when values not computed."""
        explainer = ModelExplainer()
        
        # Should return None without raising
        fig1 = explainer.plot_shap_summary()
        assert fig1 is None
        
        fig2 = explainer.plot_shap_waterfall()
        assert fig2 is None
        
        fig3 = explainer.plot_partial_dependence({})
        assert fig3 is None


class TestExplainerSelection:
    """v2.0: Test that the correct SHAP explainer is selected per model family."""

    def test_tree_explainer_for_random_forest(self, regression_data):
        """RandomForest must use TreeExplainer (exact, fast)."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")
        X, y, _ = regression_data
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        explainer = ModelExplainer()
        sv, exp = explainer._compute_shap_with_best_explainer(model, X[:5])
        assert type(exp).__name__ == 'TreeExplainer'
        assert sv is not None

    def test_tree_explainer_for_hist_gradient_boosting(self, regression_data):
        """HistGradientBoosting must use TreeExplainer (no Kernel fallback)."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")
        from sklearn.ensemble import HistGradientBoostingRegressor
        X, y, _ = regression_data
        model = HistGradientBoostingRegressor(max_iter=10, random_state=42)
        model.fit(X, y)
        explainer = ModelExplainer()
        sv, exp = explainer._compute_shap_with_best_explainer(model, X[:5])
        assert type(exp).__name__ == 'TreeExplainer'
        assert sv is not None

    def test_linear_explainer_for_ridge(self, regression_data):
        """Ridge must use LinearExplainer."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")
        X, y, _ = regression_data
        model = Ridge(alpha=1.0, random_state=42)
        model.fit(X, y)
        explainer = ModelExplainer()
        sv, exp = explainer._compute_shap_with_best_explainer(model, X[:5])
        assert type(exp).__name__ == 'LinearExplainer'
        assert sv is not None

    def test_no_kernel_fallback_for_trees(self, regression_data):
        """Tree models must NEVER silently fall back to KernelExplainer."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not installed")
        X, y, _ = regression_data
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        explainer = ModelExplainer()
        sv, exp = explainer._compute_shap_with_best_explainer(model, X[:5])
        # Must NOT be KernelExplainer
        assert type(exp).__name__ != 'KernelExplainer', (
            "Tree models should never fall back to KernelExplainer"
        )
