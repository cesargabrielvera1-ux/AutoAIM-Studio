"""Tests for model inference functionality."""

import pytest
import numpy as np
import json
import joblib
from pathlib import Path


class TestInference:
    """Test suite for model inference operations."""
    
    def test_model_loading_from_bundle(self, temp_directory, trained_rf_model):
        """Test loading a model from a bundle."""
        # Create bundle
        bundle_dir = temp_directory / "test_bundle"
        bundle_dir.mkdir()
        
        # Save model
        model_path = bundle_dir / "pipeline.joblib"
        joblib.dump(trained_rf_model, model_path)
        
        # Save manifest
        manifest = {
            "model_type": "sklearn",
            "algorithm": "RandomForest",
            "model_filename": "pipeline.joblib",
            "feature_names": [f"feature_{i}" for i in range(10)],
            "target_column": "target"
        }
        
        manifest_path = bundle_dir / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        # Load manifest
        with open(manifest_path, 'r') as f:
            loaded_manifest = json.load(f)
        
        # Load model
        loaded_model = joblib.load(bundle_dir / loaded_manifest["model_filename"])
        
        # Verify model loaded correctly
        assert loaded_model is not None
        assert hasattr(loaded_model, 'predict')
    
    def test_prediction_with_loaded_model(self, temp_directory, trained_rf_model, sample_data):
        """Test making predictions with a loaded model."""
        X, _ = sample_data
        
        # Save and reload model
        model_path = temp_directory / "model.joblib"
        joblib.dump(trained_rf_model, model_path)
        loaded_model = joblib.load(model_path)
        
        # Make predictions
        predictions = loaded_model.predict(X)
        
        # Verify predictions
        assert len(predictions) == len(X)
        assert isinstance(predictions, np.ndarray)
    
    def test_feature_validation(self, sample_data):
        """Test validation of input features."""
        X, _ = sample_data
        
        # Expected features (from training)
        expected_features = 10
        
        # Verify input has correct number of features
        assert X.shape[1] == expected_features
        
        # Test with wrong number of features
        X_wrong = X[:, :5]  # Only 5 features
        assert X_wrong.shape[1] != expected_features
    
    def test_manifest_feature_names_match(self, temp_directory):
        """Test that manifest feature names match expected features."""
        expected_features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']
        
        manifest = {
            "feature_names": expected_features,
            "target_column": "target"
        }
        
        manifest_path = temp_directory / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f)
        
        # Load and verify
        with open(manifest_path, 'r') as f:
            loaded_manifest = json.load(f)
        
        assert loaded_manifest["feature_names"] == expected_features
        assert len(loaded_manifest["feature_names"]) == 5
    
    def test_inference_with_missing_features(self):
        """Test handling of missing features during inference."""
        # Expected features from training
        expected_features = ['a', 'b', 'c', 'd']
        
        # Input data with missing features
        input_features = ['a', 'b']  # Missing 'c' and 'd'
        
        # Detect missing features
        missing = set(expected_features) - set(input_features)
        
        assert 'c' in missing
        assert 'd' in missing
        assert len(missing) == 2
    
    def test_inference_with_extra_features(self):
        """Test handling of extra features during inference."""
        expected_features = ['a', 'b', 'c']
        input_features = ['a', 'b', 'c', 'd', 'e']  # Extra features
        
        # Detect extra features
        extra = set(input_features) - set(expected_features)
        
        assert 'd' in extra
        assert 'e' in extra
        assert len(extra) == 2
    
    def test_batch_prediction(self, trained_rf_model):
        """Test batch prediction with multiple samples."""
        # Create batch of samples
        n_samples = 50
        n_features = 10
        X_batch = np.random.randn(n_samples, n_features)
        
        # Make predictions
        predictions = trained_rf_model.predict(X_batch)
        
        # Verify batch prediction
        assert len(predictions) == n_samples
        assert isinstance(predictions, np.ndarray)
    
    def test_single_prediction(self, trained_rf_model):
        """Test single sample prediction."""
        # Single sample
        X_single = np.random.randn(1, 10)
        
        # Make prediction
        prediction = trained_rf_model.predict(X_single)
        
        # Verify single prediction
        assert len(prediction) == 1
        assert isinstance(prediction[0], (float, np.floating))
    
    def test_prediction_consistency(self, trained_rf_model, sample_data):
        """Test that predictions are consistent across multiple calls."""
        X, _ = sample_data
        
        # Make predictions multiple times
        pred1 = trained_rf_model.predict(X)
        pred2 = trained_rf_model.predict(X)
        pred3 = trained_rf_model.predict(X)
        
        # Verify consistency
        np.testing.assert_array_equal(pred1, pred2)
        np.testing.assert_array_equal(pred2, pred3)
