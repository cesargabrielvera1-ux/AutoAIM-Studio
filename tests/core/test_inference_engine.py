"""Tests for inference_engine module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.core.inference_engine import InferenceEngine, LegacyInferenceEngine
from app.core.model_saver import ModelSaver


class TestInferenceEngine:
    """Test suite for InferenceEngine."""

    @pytest.fixture
    def sklearn_bundle(self, tmp_path, regression_data):
        """Create a sklearn model bundle for testing."""
        X, y, feature_names = regression_data
        
        # Create and train model
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        # Save model
        saver = ModelSaver()
        output_path = saver.save_model(
            model=model,
            output_path=str(tmp_path / "sklearn_bundle"),
            model_name="test_rf",
            model_type="sklearn",
            algorithm="RandomForest",
            feature_names=feature_names,
            target_column="target",
            metrics={"r2": 0.95}
        )
        
        return output_path, feature_names, X, y

    def test_engine_initialization(self, sklearn_bundle):
        """Test inference engine initialization."""
        bundle_path, feature_names, _, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        assert engine.bundle_path == bundle_path
        assert engine.manifest is not None
        assert engine.manifest["model_name"] == "test_rf"

    def test_load_manifest(self, sklearn_bundle):
        """Test manifest loading."""
        bundle_path, _, _, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        assert engine._manifest is not None
        assert "feature_names" in engine._manifest
        assert "model_type" in engine._manifest

    def test_load_nonexistent_manifest(self, tmp_path):
        """Test loading bundle without manifest."""
        with pytest.raises(FileNotFoundError):
            InferenceEngine(tmp_path / "nonexistent")

    def test_sanitize_feature_names(self, sklearn_bundle):
        """Test sanitizing unnamed columns from feature names."""
        bundle_path, feature_names, _, _ = sklearn_bundle
        
        # Manually add unnamed columns to manifest
        import json
        manifest_path = bundle_path / "manifest.json"
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        manifest["feature_names"] = feature_names + ["Unnamed: 5"]
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        engine = InferenceEngine(bundle_path)
        # Unnamed columns should be removed
        assert "Unnamed: 5" not in engine.manifest["feature_names"]

    def test_predict_sklearn(self, sklearn_bundle):
        """Test prediction with sklearn model."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        df = pd.DataFrame(X[:5], columns=feature_names)
        result = engine.predict(df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns
        assert len(result) == 5

    def test_predict_without_loading(self, sklearn_bundle):
        """Test predict auto-loads model."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        # Don't explicitly call load()
        df = pd.DataFrame(X[:5], columns=feature_names)
        result = engine.predict(df)
        
        assert 'prediction' in result.columns

    def test_validate_input_valid(self, sklearn_bundle):
        """Test input validation with valid data."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        df = pd.DataFrame(X[:5], columns=feature_names)
        validation = engine.validate_input(df)
        
        assert validation["valid"] is True
        assert validation["missing_columns"] == []

    def test_validate_input_missing_columns(self, sklearn_bundle):
        """Test input validation with missing columns."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        # Drop one column
        df = pd.DataFrame(X[:5], columns=feature_names).drop(columns=[feature_names[0]])
        validation = engine.validate_input(df)
        
        assert validation["valid"] is False
        assert len(validation["missing_columns"]) == 1

    def test_validate_input_extra_columns(self, sklearn_bundle):
        """Test input validation with extra columns."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        df = pd.DataFrame(X[:5], columns=feature_names)
        df['extra_column'] = [1, 2, 3, 4, 5]
        validation = engine.validate_input(df)
        
        assert validation["valid"] is True  # Extra columns are ok
        assert 'extra_column' in validation["extra_columns"]

    def test_validate_input_unnamed_columns(self, sklearn_bundle):
        """Test validation strips unnamed columns."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        df = pd.DataFrame(X[:5], columns=feature_names)
        df['Unnamed: 99'] = [1, 2, 3, 4, 5]
        validation = engine.validate_input(df)
        
        # Unnamed columns should not count as extra
        assert 'Unnamed: 99' not in validation.get("extra_columns", [])

    def test_handle_missing_values(self, sklearn_bundle):
        """Test handling of missing values in input."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        df = pd.DataFrame(X[:5], columns=feature_names)
        df.iloc[0, 0] = np.nan  # Introduce NaN
        
        # Should handle NaN without raising
        result = engine.predict(df)
        assert len(result) == 5

    def test_check_version_compatibility(self, sklearn_bundle):
        """Test version compatibility check."""
        bundle_path, _, _, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        is_compatible, message = engine.check_version_compatibility()
        # Should be compatible with same version
        assert isinstance(is_compatible, bool)
        assert isinstance(message, str)

    def test_get_model_info(self, sklearn_bundle):
        """Test getting model info."""
        bundle_path, feature_names, _, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        info = engine.get_model_info()
        assert info["model_name"] == "test_rf"
        assert info["algorithm"] == "RandomForest"
        assert info["model_type"] == "sklearn"
        assert info["feature_count"] == len(feature_names)

    def test_is_loaded(self, sklearn_bundle):
        """Test is_loaded property."""
        bundle_path, _, _, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        assert engine.is_loaded is False
        engine.load()
        assert engine.is_loaded is True

    def test_predict_with_unnamed_in_df(self, sklearn_bundle):
        """Test prediction strips unnamed columns from input."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        df = pd.DataFrame(X[:5], columns=feature_names)
        df['Unnamed: 5'] = [1, 2, 3, 4, 5]
        
        result = engine.predict(df)
        assert 'prediction' in result.columns

    def test_predict_with_intervals_regression(self, sklearn_bundle):
        """Test predict_with_intervals for regression models."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        df = pd.DataFrame(X[:5], columns=feature_names)
        result = engine.predict_with_intervals(df)
        
        assert 'prediction' in result.columns
        assert 'lower_bound' in result.columns
        assert 'upper_bound' in result.columns
        assert len(result) == 5
        # If no CI params in manifest, bounds should be NaN
        if result['lower_bound'].isna().all():
            pass  # No CI params available — expected
        else:
            # Bounds should be reasonable
            for i in range(len(result)):
                assert result['lower_bound'].iloc[i] <= result['prediction'].iloc[i]
                assert result['prediction'].iloc[i] <= result['upper_bound'].iloc[i]

    def test_predict_with_intervals_no_ci_params(self, tmp_path):
        """Test predict_with_intervals when no CI params in manifest."""
        from sklearn.ensemble import RandomForestRegressor
        import json
        
        # Create a minimal bundle without CI params
        model = RandomForestRegressor(n_estimators=3, random_state=42)
        X = np.random.randn(20, 3)
        y = X.sum(axis=1)
        model.fit(X, y)
        
        bundle_dir = tmp_path / "no_ci_bundle"
        bundle_dir.mkdir()
        
        import joblib
        joblib.dump(model, bundle_dir / "pipeline.joblib")
        
        manifest = {
            "model_name": "test_no_ci",
            "model_type": "sklearn",
            "algorithm": "RandomForest",
            "feature_names": ["f1", "f2", "f3"],
            "target_column": "target",
            "creation_date": "2024-01-01",
            "autoaim_version": "2.0.0",
        }
        with open(bundle_dir / "manifest.json", 'w') as f:
            json.dump(manifest, f)
        
        engine = InferenceEngine(bundle_dir)
        df = pd.DataFrame(np.random.randn(3, 3), columns=["f1", "f2", "f3"])
        result = engine.predict_with_intervals(df)
        
        assert 'prediction' in result.columns
        assert 'lower_bound' in result.columns
        assert 'upper_bound' in result.columns
        # Should have NaN bounds when no CI params
        assert result['lower_bound'].isna().all()
        assert result['upper_bound'].isna().all()

    def test_predict_missing_required_raises(self, sklearn_bundle):
        """Test prediction with missing required columns raises."""
        bundle_path, feature_names, X, _ = sklearn_bundle
        engine = InferenceEngine(bundle_path)
        
        # Create DataFrame with wrong columns
        df = pd.DataFrame({'wrong_col': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Invalid input|Required columns not found"):
            engine.predict(df)


class TestLegacyInferenceEngine:
    """Test suite for LegacyInferenceEngine."""

    def test_legacy_initialization(self):
        """Test legacy engine initialization."""
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        engine = LegacyInferenceEngine(model, "/path/to/model.joblib")
        
        assert engine.is_loaded is True
        assert engine.manifest is not None
        assert engine.manifest["legacy"] is True

    def test_legacy_predict(self):
        """Test legacy prediction."""
        # Train a simple model
        X = np.random.randn(50, 3)
        y = X.sum(axis=1)
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X, y)
        
        engine = LegacyInferenceEngine(model, "/path/to/model.joblib")
        
        df = pd.DataFrame(X[:5], columns=['f1', 'f2', 'f3'])
        result = engine.predict(df)
        
        assert isinstance(result, pd.DataFrame)
        assert 'prediction' in result.columns
        assert len(result) == 5

    def test_legacy_validate_input(self):
        """Test legacy input validation."""
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        engine = LegacyInferenceEngine(model, "/path/to/model.joblib")
        
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
        validation = engine.validate_input(df)
        
        assert validation["valid"] is True
        assert "Legacy model" in validation["message"]

    def test_legacy_no_numeric_raises(self):
        """Test legacy prediction with no numeric columns raises."""
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        engine = LegacyInferenceEngine(model, "/path/to/model.joblib")
        
        df = pd.DataFrame({'a': ['x', 'y'], 'b': ['z', 'w']})
        
        with pytest.raises(ValueError, match="No numeric columns found"):
            engine.predict(df)

    def test_legacy_version_check(self):
        """Test legacy version compatibility."""
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        engine = LegacyInferenceEngine(model, "/path/to/model.joblib")
        
        is_compatible, message = engine.check_version_compatibility()
        assert is_compatible is True
        assert "skipped" in message

    def test_legacy_can_generate_features(self):
        """Test legacy feature generation check."""
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        engine = LegacyInferenceEngine(model, "/path/to/model.joblib")
        
        df_with_formula = pd.DataFrame({'formula': ['Fe2O3', 'SiO2']})
        can_gen, features = engine.can_generate_features(df_with_formula)
        assert can_gen is True

        df_without = pd.DataFrame({'other': [1, 2]})
        can_gen, features = engine.can_generate_features(df_without)
        assert can_gen is False
