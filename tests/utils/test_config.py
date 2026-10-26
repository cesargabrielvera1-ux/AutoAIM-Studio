"""Tests for config module."""

import pytest
import os
import tempfile
from pathlib import Path

from app.utils.config import (
    Config, AppConfig, ModelConfig, NNConfig, OptunaConfig, get_config
)


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_default_values(self):
        """Test default model config values."""
        config = ModelConfig()
        assert config.random_state == 42
        assert config.test_size == 0.2
        assert config.cv_folds == 5
        assert config.n_jobs == 0  # v2.0: 0 = use model's hardware-aware default
        assert config.early_stopping_rounds == 50


class TestNNConfig:
    """Test suite for NNConfig."""

    def test_default_values(self):
        """Test default NN config values."""
        config = NNConfig()
        assert config.max_epochs == 1000
        assert config.batch_size == 32
        assert config.learning_rate == 0.001
        assert config.dropout_rate == 0.2
        assert config.weight_decay == 1e-5
        assert config.patience == 50
        assert config.min_delta == 1e-4


class TestOptunaConfig:
    """Test suite for OptunaConfig."""

    def test_default_values(self):
        """Test default Optuna config values."""
        config = OptunaConfig()
        assert config.n_trials == 100
        assert config.timeout is None
        assert config.n_startup_trials == 10
        assert config.n_warmup_steps == 30
        assert config.pruning_interval == 10


class TestAppConfig:
    """Test suite for AppConfig."""

    def test_default_values(self):
        """Test default app config values."""
        config = AppConfig()
        assert config.app_name == "AutoAIM Studio"
        assert config.version == "2.0.0"
        assert config.theme == "dark"
        assert config.auto_save is True
        assert config.max_memory_gb == 8.0
        assert config.use_gpu is True

    def test_nested_configs(self):
        """Test nested configuration objects."""
        config = AppConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.nn, NNConfig)
        assert isinstance(config.optuna, OptunaConfig)


class TestConfig:
    """Test suite for Config manager."""

    def test_initialization(self):
        """Test config manager initialization."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_dir=tmp_dir)
            assert config.config_dir == Path(tmp_dir)
            assert isinstance(config.current, AppConfig)

    def test_config_file_created(self):
        """Test that config file is created."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_dir=tmp_dir)
            config.save()
            assert (Path(tmp_dir) / "config.yaml").exists()

    def test_load_existing_config(self):
        """Test loading existing config file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create initial config
            config1 = Config(config_dir=tmp_dir)
            config1.current.version = "2.0.0"
            config1.save()
            
            # Load config in new instance
            config2 = Config(config_dir=tmp_dir)
            assert config2.current.version == "2.0.0"

    def test_update(self):
        """Test updating configuration values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_dir=tmp_dir)
            config.update(theme='light')
            assert config.current.theme == 'light'

    def test_update_nested(self):
        """Test updating nested configuration values."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_dir=tmp_dir)
            config.update(**{'model.random_state': 123})
            assert config.current.model.random_state == 123

    def test_reset_to_defaults(self):
        """Test resetting to defaults."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_dir=tmp_dir)
            config.update(theme='light')
            assert config.current.theme == 'light'
            
            config.reset_to_defaults()
            assert config.current.theme == 'dark'

    def test_get_model_path(self):
        """Test getting model save path."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_dir=tmp_dir)
            path = config.get_model_path('test_model')
            assert path.name == 'test_model.joblib'

    def test_invalid_config_file(self):
        """Test handling of invalid config file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            # Create invalid config file
            config_file = Path(tmp_dir) / "config.yaml"
            config_file.write_text("invalid: yaml: content: [")
            
            # Should not raise, should use defaults
            config = Config(config_dir=tmp_dir)
            assert isinstance(config.current, AppConfig)

    def test_auto_save(self):
        """Test auto-save functionality."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = Config(config_dir=tmp_dir)
            config.current.auto_save = True
            config.update(theme='light')
            
            # Config should be saved
            config_file = Path(tmp_dir) / "config.yaml"
            assert config_file.exists()


class TestGetConfig:
    """Test suite for get_config function."""

    def test_singleton(self):
        """Test that get_config returns singleton."""
        config1 = get_config()
        config2 = get_config()
        assert config1 is config2

    def test_returns_config(self):
        """Test that get_config returns Config instance."""
        config = get_config()
        assert isinstance(config, Config)
