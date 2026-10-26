"""Tests for model_saver module."""

import pytest
import numpy as np
import json
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

import pandas as pd

from app.core.model_saver import ModelSaver, save_training_result
from app.core.trainer import ModelTrainer, TrainingResult
from app.core.data_manager import DataManager


class TestModelSaver:
    """Test suite for ModelSaver."""

    def test_saver_initialization(self):
        """Test ModelSaver initialization."""
        saver = ModelSaver()
        assert saver._manifest is None

    def test_save_sklearn_model(self, tmp_path):
        """Test saving sklearn model."""
        saver = ModelSaver()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        output_path = saver.save_model(
            model=model,
            output_path=str(tmp_path / "model_bundle"),
            model_name="test_model",
            model_type="sklearn",
            algorithm="RandomForest",
            feature_names=["f1", "f2", "f3"],
            target_column="target",
            preprocessor=None,
            metrics={"r2": 0.95}
        )
        
        assert output_path.exists()
        assert (output_path / "manifest.json").exists()
        assert (output_path / "pipeline.joblib").exists()

    def test_manifest_content(self, tmp_path):
        """Test manifest.json content."""
        saver = ModelSaver()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        output_path = saver.save_model(
            model=model,
            output_path=str(tmp_path / "model_bundle"),
            model_name="test_model",
            model_type="sklearn",
            algorithm="RandomForest",
            feature_names=["f1", "f2", "f3"],
            target_column="target",
            metrics={"r2": 0.95}
        )
        
        with open(output_path / "manifest.json", 'r') as f:
            manifest = json.load(f)
        
        assert manifest["model_name"] == "test_model"
        assert manifest["algorithm"] == "RandomForest"
        assert manifest["model_type"] == "sklearn"
        assert manifest["target_column"] == "target"
        assert manifest["feature_names"] == ["f1", "f2", "f3"]
        assert manifest["metrics"]["r2"] == 0.95
        assert "creation_date" in manifest

    def test_save_with_preprocessor(self, tmp_path):
        """Test saving with preprocessor."""
        saver = ModelSaver()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        preprocessor = StandardScaler()
        
        output_path = saver.save_model(
            model=model,
            output_path=str(tmp_path / "model_bundle"),
            model_name="test_model",
            model_type="sklearn",
            algorithm="RandomForest",
            feature_names=["f1", "f2"],
            target_column="target",
            preprocessor=preprocessor
        )
        
        # Check that pipeline was created with preprocessor
        assert (output_path / "pipeline.joblib").exists()

    def test_preprocessing_checksum(self, tmp_path):
        """Test preprocessing checksum computation."""
        saver = ModelSaver()
        preprocessor = StandardScaler()
        
        checksum = saver._compute_preprocessor_checksum(preprocessor)
        assert checksum is not None
        assert len(checksum) == 32  # MD5 hash length

    def test_preprocessing_checksum_none(self):
        """Test checksum with None preprocessor."""
        saver = ModelSaver()
        checksum = saver._compute_preprocessor_checksum(None)
        assert checksum is None

    def test_get_sklearn_version(self):
        """Test getting sklearn version."""
        saver = ModelSaver()
        version = saver._get_sklearn_version()
        assert version is not None
        assert isinstance(version, str)

    def test_get_pytorch_version(self):
        """Test getting PyTorch version."""
        saver = ModelSaver()
        version = saver._get_pytorch_version()
        # May or may not have PyTorch installed
        assert version is None or isinstance(version, str)


class TestSaveTrainingResult:
    """Test suite for save_training_result function."""

    def test_save_training_result(self, tmp_path, regression_data):
        """Test saving a training result."""
        X, y, feature_names = regression_data
        
        # Create trainer and train model
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        result = trainer.train(
            model, 'RF', X, y, X[:10], y[:10], 
            cv_folds=2, feature_names=feature_names
        )
        
        # Create data manager
        dm = DataManager()
        dm._data = pd.DataFrame({
            **{name: X[:, i] for i, name in enumerate(feature_names)},
            'target': y
        })
        dm._target_column = 'target'
        dm._feature_columns = feature_names
        dm._preprocessor = None
        
        output_path = save_training_result(
            result=result,
            output_path=str(tmp_path / "training_bundle"),
            data_manager=dm
        )
        
        assert output_path.exists()
        assert (output_path / "manifest.json").exists()

    def test_save_nn_result(self, tmp_path, regression_data):
        """Test saving NN training result."""
        X, y, feature_names = regression_data
        
        from app.core.nn_builder import NeuralNetworkBuilder
        
        # Create and train NN
        builder = NeuralNetworkBuilder()
        hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
        builder.create_architecture(X.shape[1], 1, hidden_layers)
        builder.build_model()
        builder.create_training_config(epochs=10, batch_size=16)
        
        split = int(0.8 * len(X))
        builder.fit(X[:split], y[:split], verbose=False)
        
        # Create result
        result = TrainingResult(
            model=builder,
            model_name='NN',
            problem_type='regression',
            metrics={'r2': 0.9},
            cv_metrics={'test_score': [0.85, 0.88]},
            training_time=1.0,
            is_neural_network=True
        )
        
        # Create data manager
        dm = DataManager()
        dm._data = pd.DataFrame({
            **{name: X[:, i] for i, name in enumerate(feature_names)},
            'target': y
        })
        dm._target_column = 'target'
        dm._feature_columns = feature_names
        
        output_path = save_training_result(
            result=result,
            output_path=str(tmp_path / "nn_bundle"),
            data_manager=dm
        )
        
        assert output_path.exists()
        assert (output_path / "manifest.json").exists()
        assert (output_path / "model.pt").exists()

    def test_model_card_generated(self, tmp_path):
        """Test that Model Card is generated and saved in the bundle."""
        saver = ModelSaver()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        card_info = {
            'n_samples': 100,
            'problem_type': 'regression',
            'intended_use': 'Test model for materials science',
            'random_state': 42,
            'model_params': {'n_estimators': 5, 'max_depth': None},
        }
        
        output_path = saver.save_model(
            model=model,
            output_path=str(tmp_path / "model_with_card"),
            model_name="test_model",
            model_type="sklearn",
            algorithm="RandomForest",
            feature_names=["f1", "f2", "f3"],
            target_column="target",
            metrics={"r2": 0.95, "rmse": 0.1},
            model_card_info=card_info,
        )
        
        # Check that MODEL_CARD.md exists
        card_file = output_path / "MODEL_CARD.md"
        assert card_file.exists()
        
        content = card_file.read_text()
        assert "# Model Card: test_model" in content
        assert "RandomForest" in content
        assert "f1" in content  # Feature names listed
        assert "r2" in content  # Metrics listed
        assert "0.950000" in content or "0.95" in content
        assert "## Intended Use" in content
        assert "## Limitations" in content
        assert "## Reproducibility" in content

    def test_model_card_without_info(self, tmp_path):
        """Test that Model Card is generated even without extra info."""
        saver = ModelSaver()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        output_path = saver.save_model(
            model=model,
            output_path=str(tmp_path / "model_minimal_card"),
            model_name="minimal_model",
            model_type="sklearn",
            algorithm="RF",
            feature_names=["a", "b"],
            target_column="y",
            model_card_info=None,
        )
        
        card_file = output_path / "MODEL_CARD.md"
        assert card_file.exists()
        content = card_file.read_text()
        assert "# Model Card: minimal_model" in content
        assert "## Intended Use" in content

    def test_model_card_with_ci(self, tmp_path):
        """Test Model Card includes confidence interval info."""
        saver = ModelSaver()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        output_path = saver.save_model(
            model=model,
            output_path=str(tmp_path / "model_ci_card"),
            model_name="ci_model",
            model_type="sklearn",
            algorithm="RF",
            feature_names=["f1"],
            target_column="y",
            metrics={"r2": 0.9},
            confidence_interval_params={
                'type': 'regression',
                'quantile_68': 0.15,
                'quantile_95': 0.30,
                'residuals_std': 0.15,
            },
            model_card_info={'n_samples': 50, 'problem_type': 'regression'},
        )
        
        card_file = output_path / "MODEL_CARD.md"
        content = card_file.read_text()
        assert "Confidence Interval" in content
        assert "0.15" in content or "0.30" in content

    def test_save_no_model_raises(self, tmp_path):
        """Test saving result without model raises ValueError."""
        # Create a result with no model
        class FakeResult:
            pass
        
        dm = DataManager()
        
        with pytest.raises(ValueError, match="no model or best_model attribute"):
            save_training_result(
                result=FakeResult(),
                output_path=str(tmp_path / "fail_bundle"),
                data_manager=dm
            )
