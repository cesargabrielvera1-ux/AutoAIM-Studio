"""Materials AutoML - Automated Machine Learning for Materials Science."""

__version__ = "1.0.0"
__author__ = "Materials AutoML Team"

from .core.data_manager import DataManager
from .core.model_registry import ModelRegistry
from .core.trainer import ModelTrainer
from .core.nn_builder import NeuralNetworkBuilder
from .core.optimizer import HyperparameterOptimizer

__all__ = [
    'DataManager',
    'ModelRegistry',
    'ModelTrainer',
    'NeuralNetworkBuilder',
    'HyperparameterOptimizer',
]
