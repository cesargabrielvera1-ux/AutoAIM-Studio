"""Core AutoAIM modules."""

from .data_manager import DataManager
from .trainer import ModelTrainer, TrainingResult
from .optimizer import HyperparameterOptimizer, OptimizationResult
from .nn_builder import NeuralNetworkBuilder, ActivationFunction, NNArchitecture, TrainingConfig
from .model_saver import save_training_result, ModelSaver
from .model_registry import ModelRegistry
from .feature_engineering import FeatureEngineer
from .domain_applicability import DomainApplicabilityAnalyzer
from .explainer import ModelExplainer
from .validator import MaterialsAwareValidator
from .inference_engine import InferenceEngine
from .ensemble_trainer import EnsembleTrainer, EnsembleResult
from .crystal_structure import (
    CrystalStructureLoader,
    CrystalStructureFeaturizer,
    CrystalStructureDatasetBuilder,
)

__all__ = [
    'DataManager',
    'ModelTrainer',
    'TrainingResult',
    'HyperparameterOptimizer',
    'OptimizationResult',
    'NeuralNetworkBuilder',
    'ActivationFunction',
    'NNArchitecture',
    'TrainingConfig',
    'save_training_result',
    'ModelSaver',
    'ModelRegistry',
    'FeatureEngineer',
    'DomainApplicabilityAnalyzer',
    'ModelExplainer',
    'MaterialsAwareValidator',
    'InferenceEngine',
    'EnsembleTrainer',
    'EnsembleResult',
    'CrystalStructureLoader',
    'CrystalStructureFeaturizer',
    'CrystalStructureDatasetBuilder',
]
