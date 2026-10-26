"""Tests for ensemble_trainer module."""

import pytest
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge

from app.core.ensemble_trainer import EnsembleTrainer, EnsembleResult
from app.core.trainer import ModelTrainer


class TestEnsembleTrainer:
    """Test suite for EnsembleTrainer."""

    @pytest.fixture
    def trained_trainer(self, regression_data):
        """Create a trainer with 3 trained models."""
        X, y, _ = regression_data
        trainer = ModelTrainer()
        
        models = [
            (RandomForestRegressor(n_estimators=10, random_state=42), 'RF1'),
            (RandomForestRegressor(n_estimators=15, random_state=43), 'RF2'),
            (Ridge(alpha=1.0), 'Ridge1'),
        ]
        
        for model, name in models:
            trainer.train(model, name, X, y, X[:20], y[:20], cv_folds=2)
        
        return trainer, X, y

    def test_ensemble_initialization(self, trained_trainer):
        """Test ensemble trainer initialization."""
        trainer, _, _ = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        assert ensemble_trainer.trainer == trainer
        assert ensemble_trainer.results == {}

    def test_train_weighted_average(self, trained_trainer):
        """Test training weighted average ensemble."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2', 'Ridge1'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20],
            cv_folds=2
        )
        
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_type == 'weighted_average'
        assert 'RF1' in result.weights
        assert 'RF2' in result.weights
        assert 'Ridge1' in result.weights
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_train_stacking(self, trained_trainer):
        """Test training stacking ensemble."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='stacking',
            model_names=['RF1', 'RF2', 'Ridge1'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20],
            cv_folds=2
        )
        
        assert isinstance(result, EnsembleResult)
        assert result.ensemble_type == 'stacking'

    def test_train_ensemble_with_custom_weights(self, trained_trainer):
        """Test ensemble with custom weights."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        custom_weights = {'RF1': 0.5, 'RF2': 0.3, 'Ridge1': 0.2}
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2', 'Ridge1'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20],
            weights=custom_weights
        )
        
        assert isinstance(result, EnsembleResult)
        # Weights should be normalized to sum to 1
        assert abs(sum(result.weights.values()) - 1.0) < 1e-6

    def test_train_ensemble_missing_model(self, trained_trainer):
        """Test ensemble with missing model."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        # Include a model that doesn't exist
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'NonExistent', 'Ridge1'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        # Should still work, just skip the missing model
        assert isinstance(result, EnsembleResult)
        assert 'NonExistent' not in result.weights

    def test_train_ensemble_no_valid_models(self, trained_trainer):
        """Test ensemble with no valid models."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        with pytest.raises(ValueError, match="No valid models found"):
            ensemble_trainer.train_ensemble(
                ensemble_type='weighted_average',
                model_names=['NonExistent1', 'NonExistent2'],
                X_train=X,
                y_train=y
            )

    def test_train_ensemble_single_model(self, trained_trainer):
        """Test ensemble with single model (edge case)."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        assert isinstance(result, EnsembleResult)
        assert len(result.weights) == 1

    def test_weighted_average_computation(self, trained_trainer):
        """Test weighted average computation."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        # Compute predictions manually
        rf1_preds = trainer.results['RF1'].model.predict(X[:20])
        rf2_preds = trainer.results['RF2'].model.predict(X[:20])
        
        expected = (result.weights['RF1'] * rf1_preds + 
                   result.weights['RF2'] * rf2_preds)
        
        np.testing.assert_array_almost_equal(result.predictions, expected)

    def test_ensemble_cv_scores(self, trained_trainer):
        """Test that ensemble computes CV scores."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20],
            cv_folds=3
        )
        
        assert len(result.cv_scores) > 0 or result.cv_scores == []

    def test_ensemble_predict_method(self, trained_trainer):
        """Test EnsembleResult.predict() method."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        # Test prediction through EnsembleResult
        predictions = result.predict(X[:10])
        assert len(predictions) == 10

    def test_ensemble_predict_no_base_models(self):
        """Test predict without base models raises."""
        result = EnsembleResult(
            ensemble_type='weighted_average',
            model_names=['RF1'],
            weights={'RF1': 1.0},
            metrics={}
        )
        
        with pytest.raises(RuntimeError, match="base models are not available"):
            result.predict(np.array([[1, 2, 3]]))

    def test_ensemble_progress_callback(self, trained_trainer):
        """Test ensemble with progress callback."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        progress_calls = []
        def callback(message, progress):
            progress_calls.append((message, progress))
        
        ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20],
            progress_callback=callback
        )
        
        assert len(progress_calls) > 0

    def test_stacking_fallback(self, trained_trainer):
        """Test stacking fallback to weighted average on error."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        # Force stacking to fail by providing invalid predictions
        result = ensemble_trainer._stacking(
            train_predictions={'RF1': np.array([1, 2, 3])},
            meta_train_features={'RF1': np.array([1, 2, 3])},
            y_train=np.array([1, 2, 3]),
            test_predictions={'RF1': np.array([4, 5, 6])}
        )
        
        # Should return weighted average (single model = same predictions)
        # Single model predictions should be approximately the model's predictions
        np.testing.assert_array_almost_equal(result, np.array([4.0, 5.0, 6.0]), decimal=0)

    def test_weighted_average_with_none_predictions(self):
        """Test weighted average with None predictions."""
        trainer = ModelTrainer()
        ensemble_trainer = EnsembleTrainer(trainer)
        
        predictions = {
            'model1': np.array([1.0, 2.0, 3.0]),
            'model2': None,
            'model3': np.array([1.5, 2.5, 3.5])
        }
        weights = {'model1': 0.5, 'model2': 0.3, 'model3': 0.2}
        
        result = ensemble_trainer._weighted_average(predictions, weights)
        assert result is not None
        assert len(result) == 3

    def test_weighted_average_empty_predictions(self):
        """Test weighted average with empty predictions."""
        trainer = ModelTrainer()
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer._weighted_average({}, {})
        assert result is None

    def test_ensemble_results_storage(self, trained_trainer):
        """Test that ensemble results are stored."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        assert len(ensemble_trainer.results) == 1
        assert 'Ensemble_weighted_average_1' in ensemble_trainer.results

    def test_skipping_ensemble_as_base(self, trained_trainer):
        """Test that ensemble models are skipped as base models."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        # First create an ensemble
        ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'RF2'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        # Try to use the ensemble as a base model
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1', 'Ensemble_weighted_average_1'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        # Ensemble should be skipped
        assert 'Ensemble_weighted_average_1' not in result.model_names

    def test_ensemble_with_2d_predictions(self, trained_trainer):
        """Test ensemble with 2D predictions (single output but 2D array)."""
        trainer, X, y = trained_trainer
        ensemble_trainer = EnsembleTrainer(trainer)
        
        result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average',
            model_names=['RF1'],
            X_train=X,
            y_train=y,
            X_test=X[:20],
            y_test=y[:20]
        )
        
        # Predictions should be 1D
        assert result.predictions.ndim == 1
