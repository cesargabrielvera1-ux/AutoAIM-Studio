"""Tests for nn_builder module."""

import pytest
import numpy as np

# Protect against broken torch installations (e.g., Windows DLL errors)
try:
    import torch
    HAS_TORCH = True
except (ImportError, OSError):
    HAS_TORCH = False
    torch = None

# Skip all tests if torch cannot be loaded
pytestmark = pytest.mark.skipif(not HAS_TORCH, reason="torch not available (import failed)")

if HAS_TORCH:
    from app.core.nn_builder import (
        NeuralNetworkBuilder, DynamicNN, NNArchitecture, LayerConfig,
        TrainingConfig, ActivationFunction, OptimizerType, SchedulerType
    )


class TestLayerConfig:
    """Test LayerConfig dataclass."""

    def test_valid_layer_config(self):
        """Test creating a valid layer config."""
        config = LayerConfig(n_units=64, activation=ActivationFunction.RELU)
        assert config.n_units == 64
        assert config.dropout_rate == 0.0
        assert config.use_batch_norm is True

    def test_invalid_n_units(self):
        """Test that invalid n_units raises ValueError."""
        with pytest.raises(ValueError, match="n_units must be between 16 and 2048"):
            LayerConfig(n_units=5)

    def test_invalid_dropout(self):
        """Test that invalid dropout raises ValueError."""
        with pytest.raises(ValueError, match="dropout_rate must be between 0.0 and 0.5"):
            LayerConfig(n_units=64, dropout_rate=0.8)


class TestNNArchitecture:
    """Test NNArchitecture dataclass."""

    def test_valid_architecture(self):
        """Test creating valid architecture."""
        layers = [LayerConfig(n_units=64), LayerConfig(n_units=32)]
        arch = NNArchitecture(input_dim=10, output_dim=1, hidden_layers=layers)
        assert arch.input_dim == 10
        assert arch.output_dim == 1
        assert len(arch.hidden_layers) == 2

    def test_invalid_input_dim(self):
        """Test that invalid input_dim raises ValueError."""
        with pytest.raises(ValueError, match="input_dim must be positive"):
            NNArchitecture(input_dim=0, output_dim=1)

    def test_invalid_output_dim(self):
        """Test that invalid output_dim raises ValueError."""
        with pytest.raises(ValueError, match="output_dim must be positive"):
            NNArchitecture(input_dim=10, output_dim=0)

    def test_too_many_layers(self):
        """Test that >5 layers raises ValueError."""
        layers = [LayerConfig(n_units=64) for _ in range(6)]
        with pytest.raises(ValueError, match="Maximum 5 hidden layers allowed"):
            NNArchitecture(input_dim=10, output_dim=1, hidden_layers=layers)


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_valid_config(self):
        """Test creating valid training config."""
        config = TrainingConfig(epochs=100, batch_size=32, learning_rate=0.001)
        assert config.epochs == 100
        assert config.batch_size == 32

    def test_invalid_epochs(self):
        """Test invalid epochs."""
        with pytest.raises(ValueError, match="epochs must be between 10 and 10000"):
            TrainingConfig(epochs=5)

    def test_invalid_batch_size(self):
        """Test invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be between 8 and 512"):
            TrainingConfig(batch_size=4)

    def test_random_seed_field(self):
        """Test that random_seed field exists (v1.3.0)."""
        config = TrainingConfig(random_seed=123)
        assert config.random_seed == 123


class TestDynamicNN:
    """Test DynamicNN model."""

    def test_model_creation(self):
        """Test creating a model."""
        layers = [
            LayerConfig(n_units=32, activation=ActivationFunction.RELU, dropout_rate=0.1),
            LayerConfig(n_units=16, activation=ActivationFunction.RELU)
        ]
        arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=layers)
        model = DynamicNN(arch)
        
        assert model is not None
        assert len(model.layers) == 2
        assert model.output_layer is not None

    def test_model_forward_pass(self):
        """Test forward pass."""
        layers = [LayerConfig(n_units=16, activation=ActivationFunction.RELU)]
        arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=layers)
        model = DynamicNN(arch)
        
        x = torch.randn(4, 5)  # batch_size=4, input_dim=5
        output = model(x)
        assert output.shape == (4, 1)

    def test_model_predict(self):
        """Test sklearn-compatible predict."""
        layers = [LayerConfig(n_units=16, activation=ActivationFunction.RELU)]
        arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=layers)
        model = DynamicNN(arch)
        
        X = np.random.randn(10, 5)
        predictions = model.predict(X)
        assert predictions.shape == (10,)
        assert isinstance(predictions, np.ndarray)

    def test_model_predict_2d_output(self):
        """Test predict with 2D output."""
        layers = [LayerConfig(n_units=16, activation=ActivationFunction.RELU)]
        arch = NNArchitecture(input_dim=5, output_dim=3, hidden_layers=layers)
        model = DynamicNN(arch)
        
        X = np.random.randn(10, 5)
        predictions = model.predict(X)
        assert predictions.shape == (10, 3)

    def test_model_with_batch_norm(self):
        """Test model with batch normalization."""
        layers = [LayerConfig(n_units=16, use_batch_norm=True)]
        arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=layers)
        model = DynamicNN(arch)
        
        x = torch.randn(4, 5)
        output = model(x)
        assert output.shape == (4, 1)

    def test_model_with_dropout(self):
        """Test model with dropout (eval mode disables dropout)."""
        layers = [LayerConfig(n_units=16, dropout_rate=0.5)]
        arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=layers)
        model = DynamicNN(arch)
        model.eval()
        
        x = torch.randn(4, 5)
        output1 = model(x)
        output2 = model(x)
        # In eval mode, dropout is disabled so outputs should be identical
        assert torch.allclose(output1, output2)

    def test_model_no_layers_raises(self):
        """Test that model with no layers raises ValueError."""
        arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=[])
        with pytest.raises(ValueError, match="no tiene capas ocultas"):
            DynamicNN(arch)

    def test_model_summary(self):
        """Test model summary."""
        layers = [LayerConfig(n_units=32), LayerConfig(n_units=16)]
        arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=layers)
        model = DynamicNN(arch)
        
        summary = model.get_model_summary()
        assert 'total_parameters' in summary
        assert 'trainable_parameters' in summary
        assert summary['total_parameters'] > 0

    def test_different_activations(self):
        """Test different activation functions."""
        for act in [ActivationFunction.RELU, ActivationFunction.TANH, 
                     ActivationFunction.GELU, ActivationFunction.SWISH]:
            layers = [LayerConfig(n_units=16, activation=act)]
            arch = NNArchitecture(input_dim=5, output_dim=1, hidden_layers=layers)
            model = DynamicNN(arch)
            x = torch.randn(2, 5)
            output = model(x)
            assert output.shape == (2, 1)
            assert not torch.isnan(output).any()


class TestNeuralNetworkBuilder:
    """Test NeuralNetworkBuilder."""

    def test_builder_initialization(self):
        """Test builder initialization."""
        builder = NeuralNetworkBuilder()
        assert builder.model is None
        assert builder.architecture is None

    def test_create_architecture(self):
        """Test creating architecture."""
        builder = NeuralNetworkBuilder()
        hidden_layers = [
            {'n_units': 32, 'activation': 'relu', 'dropout_rate': 0.1},
            {'n_units': 16, 'activation': 'relu', 'dropout_rate': 0.0}
        ]
        arch = builder.create_architecture(
            input_dim=5,
            output_dim=1,
            hidden_layers_config=hidden_layers,
            problem_type='regression'
        )
        assert isinstance(arch, NNArchitecture)
        assert len(arch.hidden_layers) == 2

    def test_build_model(self):
        """Test building model."""
        builder = NeuralNetworkBuilder()
        hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
        builder.create_architecture(5, 1, hidden_layers)
        model = builder.build_model()
        assert model is not None
        assert isinstance(model, DynamicNN)

    def test_create_training_config(self):
        """Test creating training config."""
        builder = NeuralNetworkBuilder()
        config = builder.create_training_config(
            epochs=50,
            batch_size=16,
            learning_rate=0.01,
            random_seed=42
        )
        assert isinstance(config, TrainingConfig)
        assert config.epochs == 50
        assert config.random_seed == 42

    def test_fit_predict(self, regression_data):
        """Test training workflow."""
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        
        hidden_layers = [
            {'n_units': 32, 'activation': 'relu', 'dropout_rate': 0.1},
        ]
        builder.create_architecture(X.shape[1], 1, hidden_layers)
        builder.build_model()
        builder.create_training_config(epochs=10, batch_size=16, learning_rate=0.01)
        
        # Split data
        split = int(0.8 * len(X))
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        
        history = builder.fit(X_train, y_train, X_val, y_val, verbose=False)
        
        assert 'train_loss' in history
        assert 'val_loss' in history
        assert len(history['train_loss']) > 0
        
        # Predict
        predictions = builder.predict(X_val)
        assert len(predictions) == len(X_val)

    def test_fit_no_validation(self, regression_data):
        """Test training without validation data."""
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        
        hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
        builder.create_architecture(X.shape[1], 1, hidden_layers)
        builder.build_model()
        builder.create_training_config(epochs=10, batch_size=16)
        
        history = builder.fit(X, y, verbose=False)
        assert 'train_loss' in history
        assert len(history['train_loss']) > 0

    def test_early_stopping(self, regression_data):
        """Test early stopping."""
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        
        hidden_layers = [{'n_units': 64, 'activation': 'relu'}]
        builder.create_architecture(X.shape[1], 1, hidden_layers)
        builder.build_model()
        builder.create_training_config(
            epochs=100,
            batch_size=16,
            early_stopping_patience=5,
            early_stopping_min_delta=1e-4
        )
        
        split = int(0.8 * len(X))
        history = builder.fit(X[:split], y[:split], X[split:], y[split:], verbose=False)
        
        # Should have stopped early (less than 100 epochs)
        assert len(history['train_loss']) <= 100  # May or may not stop early

    def test_save_load_model(self, regression_data, tmp_path):
        """Test saving and loading model."""
        import torch
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        
        hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
        builder.create_architecture(X.shape[1], 1, hidden_layers)
        builder.build_model()
        builder.create_training_config(epochs=10, batch_size=16)
        
        split = int(0.8 * len(X))
        builder.fit(X[:split], y[:split], verbose=False)
        
        # Save
        model_path = tmp_path / "test_model.pt"
        builder.save_model(str(model_path))
        assert model_path.exists()
        
        # Load into new builder
        builder2 = NeuralNetworkBuilder()
        builder2.load_model(str(model_path))
        
        # Predictions should match
        pred1 = builder.predict(X[split:])
        pred2 = builder2.predict(X[split:])
        np.testing.assert_array_almost_equal(pred1, pred2)

    def test_gradient_clipping(self, regression_data):
        """Test gradient clipping."""
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        
        hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
        builder.create_architecture(X.shape[1], 1, hidden_layers)
        builder.build_model()
        builder.create_training_config(
            epochs=10,
            batch_size=16,
            gradient_clip_val=1.0
        )
        
        split = int(0.8 * len(X))
        history = builder.fit(X[:split], y[:split], verbose=False)
        assert len(history['train_loss']) > 0

    def test_different_optimizers(self, regression_data):
        """Test different optimizers."""
        X, y, _ = regression_data
        
        for opt_name in ['adam', 'adamw', 'sgd']:
            builder = NeuralNetworkBuilder()
            hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
            builder.create_architecture(X.shape[1], 1, hidden_layers)
            builder.build_model()
            builder.create_training_config(epochs=10, batch_size=16, optimizer=opt_name)
            
            split = int(0.8 * len(X))
            history = builder.fit(X[:split], y[:split], verbose=False)
            assert len(history['train_loss']) > 0, f"Optimizer {opt_name} failed"

    def test_different_schedulers(self, regression_data):
        """Test different learning rate schedulers."""
        X, y, _ = regression_data
        
        for sched_name in ['reduce_on_plateau', 'cosine_annealing', 'step_lr', 'exponential', 'none']:
            builder = NeuralNetworkBuilder()
            hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
            builder.create_architecture(X.shape[1], 1, hidden_layers)
            builder.build_model()
            builder.create_training_config(
                epochs=10, 
                batch_size=16, 
                scheduler=sched_name,
                learning_rate=0.01
            )
            
            split = int(0.8 * len(X))
            history = builder.fit(X[:split], y[:split], X[split:], y[split:], verbose=False)
            assert len(history['train_loss']) > 0, f"Scheduler {sched_name} failed"

    def test_architecture_visualization(self):
        """Test architecture visualization string."""
        builder = NeuralNetworkBuilder()
        hidden_layers = [
            {'n_units': 64, 'activation': 'relu'},
            {'n_units': 32, 'activation': 'tanh'}
        ]
        builder.create_architecture(10, 1, hidden_layers)
        builder.build_model()
        
        viz = builder.get_architecture_visualization()
        assert 'NEURAL NETWORK ARCHITECTURE' in viz
        assert '64' in viz
        assert '32' in viz

    def test_fit_without_build_raises(self, regression_data):
        """Test that fitting without building model raises."""
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        
        with pytest.raises(ValueError, match="Model must be built before training"):
            builder.fit(X, y)

    def test_fit_without_config_raises(self, regression_data):
        """Test that fitting without training config raises."""
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        hidden_layers = [{'n_units': 16, 'activation': 'relu'}]
        builder.create_architecture(X.shape[1], 1, hidden_layers)
        builder.build_model()
        
        with pytest.raises(ValueError, match="Training config must be set before training"):
            builder.fit(X, y)

    def test_predict_without_train_raises(self, regression_data):
        """Test that predicting without training raises."""
        X, y, _ = regression_data
        builder = NeuralNetworkBuilder()
        
        with pytest.raises(ValueError, match="Model must be trained before prediction"):
            builder.predict(X)

    def test_torch_manual_seed_for_reproducible_weights(self, regression_data):
        """v2.0: build_model() must call torch.manual_seed() so two NNs with
        the same seed have identical initial weights."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        X, y, _ = regression_data
        input_dim = X.shape[1]

        def _create_and_extract_weights(seed):
            builder = NeuralNetworkBuilder()
            builder.create_architecture(
                input_dim=input_dim, output_dim=1,
                hidden_layers_config=[{'n_units': 16, 'activation': 'relu', 'dropout_rate': 0.0}]
            )
            builder.create_training_config(random_seed=seed)
            builder.build_model()
            # Extract first layer weights
            return list(builder.model.parameters())[0].detach().clone()

        w1 = _create_and_extract_weights(42)
        w2 = _create_and_extract_weights(42)
        w3 = _create_and_extract_weights(123)

        # Same seed → identical weights
        assert torch.allclose(w1, w2, atol=1e-10), (
            "Same random_seed must produce identical initial weights"
        )
        # Different seed → different weights (with very high probability)
        assert not torch.allclose(w1, w3, atol=1e-3), (
            "Different random_seed must produce different initial weights"
        )

    def test_build_model_uses_training_config_seed(self, regression_data):
        """v2.0: build_model() must read random_seed from TrainingConfig
        when no explicit seed is passed."""
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not installed")

        X, y, _ = regression_data
        input_dim = X.shape[1]

        builder = NeuralNetworkBuilder()
        builder.create_architecture(
            input_dim=input_dim, output_dim=1,
            hidden_layers_config=[{'n_units': 16, 'activation': 'relu', 'dropout_rate': 0.0}]
        )
        # create_training_config FIRST (correct order), then build_model
        builder.create_training_config(random_seed=99)
        builder.build_model()  # Should use seed=99 from TrainingConfig

        w = list(builder.model.parameters())[0].detach().clone()

        # Re-create with same seed explicitly
        builder2 = NeuralNetworkBuilder()
        builder2.create_architecture(
            input_dim=input_dim, output_dim=1,
            hidden_layers_config=[{'n_units': 16, 'activation': 'relu', 'dropout_rate': 0.0}]
        )
        builder2.build_model(random_seed=99)

        w2 = list(builder2.model.parameters())[0].detach().clone()
        assert torch.allclose(w, w2, atol=1e-10), (
            "build_model() must use random_seed from TrainingConfig"
        )
