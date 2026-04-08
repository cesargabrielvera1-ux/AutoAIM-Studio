"""Tests for model training and saving functionality."""

import pytest
import numpy as np
import json
from pathlib import Path


class TestModelTraining:
    """Test suite for model training operations."""
    
    def test_random_forest_training(self, sample_data):
        """Test training a simple Random Forest model."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, r2_score
        
        X, y = sample_data
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        rmse = mean_squared_error(y, y_pred, squared=False)
        r2 = r2_score(y, y_pred)
        
        # Verify model trained successfully
        assert model is not None
        assert len(y_pred) == len(y)
        assert rmse >= 0  # RMSE should be non-negative
        assert 0 <= r2 <= 1  # R2 should be between 0 and 1 for training data
    
    def test_model_has_predict_method(self, trained_rf_model, sample_data):
        """Test that trained model has predict method."""
        X, _ = sample_data
        
        # Verify predict method exists
        assert hasattr(trained_rf_model, 'predict')
        
        # Verify predict works
        predictions = trained_rf_model.predict(X)
        assert len(predictions) == len(X)
    
    def test_model_feature_importances(self, trained_rf_model):
        """Test that Random Forest provides feature importances."""
        # Random Forest should have feature_importances_ attribute
        assert hasattr(trained_rf_model, 'feature_importances_')
        
        importances = trained_rf_model.feature_importances_
        
        # Should have one importance per feature
        assert len(importances) == 10  # From sample_data fixture
        
        # Importances should sum to 1
        assert np.isclose(np.sum(importances), 1.0)
        
        # All importances should be non-negative
        assert np.all(importances >= 0)
    
    def test_manifest_json_creation(self, temp_directory):
        """Test creation of manifest.json for model bundle."""
        # Simulate creating a model bundle manifest
        manifest = {
            "model_type": "sklearn",
            "algorithm": "RandomForest",
            "model_name": "random_forest",
            "creation_date": "2026-04-07T00:00:00",
            "sklearn_version": "1.2.0",
            "feature_names": ["feature_1", "feature_2", "feature_3"],
            "target_column": "target",
            "metrics": {
                "rmse": 0.5,
                "mae": 0.4,
                "r2": 0.85
            },
            "is_neural_network": False
        }
        
        # Save manifest
        manifest_path = temp_directory / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Verify manifest was created
        assert manifest_path.exists()
        
        # Load and verify content
        with open(manifest_path, 'r') as f:
            loaded_manifest = json.load(f)
        
        assert loaded_manifest["model_type"] == "sklearn"
        assert loaded_manifest["algorithm"] == "RandomForest"
        assert "metrics" in loaded_manifest
        assert loaded_manifest["is_neural_network"] == False
    
    def test_model_bundle_structure(self, temp_directory, trained_rf_model):
        """Test complete model bundle structure."""
        import joblib
        
        # Create bundle directory
        bundle_dir = temp_directory / "model_bundle"
        bundle_dir.mkdir()
        
        # Save model
        model_path = bundle_dir / "pipeline.joblib"
        joblib.dump(trained_rf_model, model_path)
        
        # Save manifest
        manifest = {
            "model_type": "sklearn",
            "algorithm": "RandomForest",
            "model_filename": "pipeline.joblib",
            "preprocessor_filename": None,
            "feature_names": [f"feature_{i}" for i in range(10)],
            "target_column": "target"
        }
        
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Verify bundle structure
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "pipeline.joblib").exists()
        
        # Verify model can be loaded
        loaded_model = joblib.load(bundle_dir / "pipeline.joblib")
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
    
    def test_training_result_object(self, sample_data):
        """Test TrainingResult-like object creation."""
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        X, y = sample_data
        
        # Train model
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Create result object (simulating app structure)
        result = {
            'model': model,
            'model_name': 'random_forest',
            'problem_type': 'regression',
            'metrics': {
                'rmse': mean_squared_error(y, y_pred, squared=False),
                'mae': mean_absolute_error(y, y_pred),
                'r2': r2_score(y, y_pred)
            },
            'cv_metrics': {},
            'training_time': 0.5,
            'feature_importance': dict(zip(
                [f"feature_{i}" for i in range(10)],
                model.feature_importances_
            )),
            'is_neural_network': False
        }
        
        # Verify result structure
        assert result['model'] is not None
        assert result['model_name'] == 'random_forest'
        assert 'rmse' in result['metrics']
        assert 'mae' in result['metrics']
        assert 'r2' in result['metrics']
        assert result['is_neural_network'] == False
