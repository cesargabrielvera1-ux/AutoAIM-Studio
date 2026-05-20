"""Configuration management for AutoAIM Studio."""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class ModelConfig:
    """Configuration for model training."""
    random_state: int = 42
    test_size: float = 0.2
    cv_folds: int = 5
    n_jobs: int = -1
    early_stopping_rounds: int = 50


@dataclass
class NNConfig:
    """Configuration for neural networks."""
    max_epochs: int = 1000
    batch_size: int = 32
    learning_rate: float = 0.001
    dropout_rate: float = 0.2
    weight_decay: float = 1e-5
    patience: int = 50
    min_delta: float = 1e-4


@dataclass
class OptunaConfig:
    """Configuration for hyperparameter optimization."""
    n_trials: int = 100
    timeout: Optional[int] = None
    n_startup_trials: int = 10
    n_warmup_steps: int = 30
    pruning_interval: int = 10


@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "AutoAIM Studio"
    full_name: str = "Auto Artificial Intelligence for Materials Studio"
    version: str = "1.1.0"
    theme: str = "dark"
    auto_save: bool = True
    auto_save_interval: int = 300
    default_output_dir: str = "./models"
    log_level: str = "INFO"
    max_memory_gb: float = 8.0
    use_gpu: bool = True
    model: ModelConfig = field(default_factory=ModelConfig)
    nn: NNConfig = field(default_factory=NNConfig)
    optuna: OptunaConfig = field(default_factory=OptunaConfig)


class Config:
    """Configuration manager with persistence."""
    
    CONFIG_FILENAME = "config.yaml"
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Directory to store configuration files.
                       Defaults to ~/.materials_automl
        """
        if config_dir is None:
            config_dir = os.path.expanduser("~/.materials_automl")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_file = self.config_dir / self.CONFIG_FILENAME
        
        self._config = AppConfig()
        self.load()
    
    @property
    def current(self) -> AppConfig:
        """Get current configuration."""
        return self._config
    
    def load(self) -> None:
        """Load configuration from file."""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.suffix == '.yaml' or self.config_file.suffix == '.yml':
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                if data:
                    self._config = self._dict_to_config(data)
            except Exception as e:
                print(f"Error loading config: {e}. Using defaults.")
    
    def save(self) -> None:
        """Save configuration to file."""
        try:
            data = self._config_to_dict(self._config)
            with open(self.config_file, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        except Exception as e:
            print(f"Error saving config: {e}")
    
    def reset_to_defaults(self) -> None:
        """Reset configuration to default values."""
        self._config = AppConfig()
        self.save()
    
    def update(self, **kwargs) -> None:
        """Update configuration values.
        
        Args:
            **kwargs: Configuration values to update
        """
        for key, value in kwargs.items():
            if hasattr(self._config, key):
                setattr(self._config, key, value)
            elif '.' in key:
                # Handle nested attributes like 'model.random_state'
                parts = key.split('.')
                obj = self._config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)
        
        if self._config.auto_save:
            self.save()
    
    def _config_to_dict(self, config: AppConfig) -> Dict[str, Any]:
        """Convert config dataclass to dictionary."""
        return asdict(config)
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to config dataclass."""
        try:
            model_config = ModelConfig(**data.get('model', {}))
            nn_config = NNConfig(**data.get('nn', {}))
            optuna_config = OptunaConfig(**data.get('optuna', {}))
            
            return AppConfig(
                app_name=data.get('app_name', 'Materials AutoML Studio'),
                version=data.get('version', '1.1.0'),
                theme=data.get('theme', 'dark'),
                auto_save=data.get('auto_save', True),
                auto_save_interval=data.get('auto_save_interval', 300),
                default_output_dir=data.get('default_output_dir', './models'),
                log_level=data.get('log_level', 'INFO'),
                max_memory_gb=data.get('max_memory_gb', 8.0),
                use_gpu=data.get('use_gpu', True),
                model=model_config,
                nn=nn_config,
                optuna=optuna_config
            )
        except Exception:
            return AppConfig()
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path for saving a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Path object for model file
        """
        output_dir = Path(self._config.default_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{model_name}.joblib"


# Global config instance
_config_instance: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance
