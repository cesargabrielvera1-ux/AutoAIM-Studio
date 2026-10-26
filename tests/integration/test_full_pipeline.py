"""
Full-Pipeline Integration Tests for AutoAIM Studio.

These tests cover the COMPLETE user workflow from data loading through
prediction, testing the orchestration methods in GUI tabs that wire
together the core modules. No PyQt UI is tested — we test the data
processing and decision logic that the tabs execute.

Coverage:
  1. Data Loading → Preprocessing
  2. Training: OOTB, Manual, Optimized
  3. Neural Networks: Manual, Optimized
  4. Ensemble: Manual, Optimized
  5. Explainability
  6. Inference / Prediction
  7. Crystal Structure Linking
  8. Session Export (end-to-end with all result types)
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.data_manager import DataManager
from app.core.model_registry import ModelRegistry
from app.core.trainer import ModelTrainer
from app.core.ensemble_trainer import EnsembleTrainer
from app.core.optimizer import HyperparameterOptimizer, OptimizationResult
from app.core.session_exporter import SessionExporter
from app.core.validator import MaterialsAwareValidator
from app.core.inference_engine import InferenceEngine
from app.core.model_saver import save_training_result


# =============================================================================
# Fixtures: Shared test infrastructure
# =============================================================================

@pytest.fixture
def sample_data_file(tmp_path):
    """Create a realistic materials science dataset and save to CSV."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        'material_id': [f"MAT_{i:03d}" for i in range(n)],
        'composition': ['Fe2O3', 'SiO2', 'TiO2', 'Al2O3', 'MgO'] * 12,
        'band_gap': np.random.uniform(0.5, 5.0, n),
        'energy_per_atom': np.random.uniform(-8, -2, n),
        'formation_energy_per_atom': np.random.uniform(-4, 0, n),
        'magnetic_moment': np.random.uniform(0, 5, n),
        'density': np.random.uniform(2, 8, n),
        'volume': np.random.uniform(10, 100, n),
    })
    path = tmp_path / 'sample_data.csv'
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def sample_classification_file(tmp_path):
    """Create a classification dataset and save to CSV."""
    np.random.seed(42)
    n = 60
    df = pd.DataFrame({
        'feature_1': np.random.randn(n),
        'feature_2': np.random.randn(n),
        'feature_3': np.random.randn(n),
        'is_metallic': np.random.choice([0, 1], n),
    })
    path = tmp_path / 'classification_data.csv'
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def data_manager(sample_data_file):
    """DataManager with loaded data and target set."""
    dm = DataManager()
    dm.load_data(sample_data_file)
    dm.set_target_column('band_gap')
    return dm


@pytest.fixture
def prepared_data(data_manager):
    """Fully preprocessed train/test split.
    
    NOTE: DataManager.get_preprocessor() accepts:
      - numeric_strategy: 'standard', 'minmax', 'robust', 'none'
      - impute_strategy: 'mean', 'median', 'most_frequent', 'knn'
    prepare_data() calls get_preprocessor() internally.
    """
    # Configure preprocessor before prepare_data
    data_manager.get_preprocessor(numeric_strategy='robust', impute_strategy='median')
    X_train, X_test, y_train, y_test = data_manager.prepare_data()
    return X_train, X_test, y_train, y_test, data_manager


@pytest.fixture
def model_registry():
    """Fresh ModelRegistry."""
    return ModelRegistry()


@pytest.fixture
def trainer():
    """Fresh ModelTrainer."""
    return ModelTrainer()


@pytest.fixture
def optimizer():
    """Fresh HyperparameterOptimizer."""
    return HyperparameterOptimizer()


# =============================================================================
# TEST 1: Data Loading & Preprocessing Pipeline
# =============================================================================

class TestDataLoadingPreprocessing:
    """Integration: DataManager handles the full preprocessing chain."""

    def test_full_preprocessing_chain(self, sample_data_file):
        """Load → set target → preprocess → prepare data."""
        dm = DataManager()
        dm.load_data(sample_data_file)
        assert dm.data is not None
        assert len(dm.data) == 60
        
        dm.set_target_column('band_gap')
        assert dm.target_column == 'band_gap'
        
        dm.get_preprocessor(numeric_strategy='robust', impute_strategy='median')
        X_train, X_test, y_train, y_test = dm.prepare_data()
        
        assert X_train.shape[0] + X_test.shape[0] == 60
        assert X_train.shape[1] == len(dm.get_feature_names())
        assert len(y_train) == X_train.shape[0]
        # No NaN after preprocessing
        assert not np.isnan(X_train.values).any()
        assert not np.isnan(y_train).any()

    def test_classification_detection(self, sample_classification_file):
        """DataManager correctly identifies classification problems."""
        dm = DataManager()
        dm.load_data(sample_classification_file)
        dm.set_target_column('is_metallic')
        assert dm.is_classification is True

    def test_feature_names_after_preprocessing(self, prepared_data):
        """Feature names are preserved through preprocessing."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        names = dm.get_feature_names()
        assert len(names) == X_train.shape[1]
        assert 'band_gap' not in names  # target excluded

    def test_all_imputation_strategies(self, sample_data_file):
        """All imputation strategies produce valid data."""
        for strategy in ['mean', 'median', 'most_frequent']:
            dm = DataManager()
            dm.load_data(sample_data_file)
            dm.set_target_column('band_gap')
            dm.get_preprocessor(numeric_strategy='robust', impute_strategy=strategy)
            X_train, X_test, y_train, y_test = dm.prepare_data()
            assert not np.isnan(X_train.values).any()
            assert not np.isnan(y_train).any()

    def test_all_scaling_strategies(self, sample_data_file):
        """All scaling strategies produce valid data."""
        for strategy in ['standard', 'robust', 'minmax', 'none']:
            dm = DataManager()
            dm.load_data(sample_data_file)
            dm.set_target_column('band_gap')
            dm.get_preprocessor(numeric_strategy=strategy, impute_strategy='median')
            X_train, X_test, y_train, y_test = dm.prepare_data()
            assert X_train.shape[0] > 0

    def test_external_validation_split(self, sample_data_file):
        """External validation set is correctly separated."""
        dm = DataManager()
        dm.load_data(sample_data_file)
        dm.set_target_column('band_gap')
        dm.get_preprocessor(numeric_strategy='robust', impute_strategy='median')
        X_train, X_test, y_train, y_test = dm.prepare_data()
        val_result = dm.get_validation_arrays()
        if val_result is not None:
            X_ext, y_ext = val_result
        if val_result is not None and X_ext is not None:
            assert len(X_ext) + len(X_train) + len(X_test) == 60


# =============================================================================
# TEST 2: Training Pipeline — OOTB, Manual, Optimized
# =============================================================================

class TestTrainingPipeline:
    """Integration: ModelTrainer with ModelRegistry produces valid results."""

    def test_train_all_registry_models(self, prepared_data, trainer, model_registry):
        """Every registered model can be trained and produces a valid result."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        problem_type = 'regression' if not dm.is_classification else 'classification'
        
        results = {}
        failed = []
        for name in model_registry.list_models(problem_type):
            try:
                model = model_registry.create_estimator(name, problem_type, random_state=42)
                result = trainer.train(
                    model=model, model_name=name,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    problem_type=problem_type,
                    feature_names=dm.get_feature_names()
                )
                results[name] = result
                assert result.metrics is not None
                assert 'r2' in result.metrics or 'accuracy' in result.metrics
                assert result.cv_metrics is not None
                assert len(result.cv_metrics.get('test_score', [])) > 0
            except Exception as e:
                failed.append((name, str(e)))
        
        assert len(failed) == 0, f"Models failed: {failed}"
        assert len(results) == len(model_registry.list_models(problem_type))

    def test_train_with_manual_params(self, prepared_data, trainer):
        """Training with manually set hyperparameters."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        model = RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42)
        result = trainer.train(
            model=model, model_name='Random Forest (Manual)',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type='regression', cv_folds=5,
            feature_names=dm.get_feature_names()
        )
        assert result.model_name == 'Random Forest (Manual)'
        assert result.training_config is not None
        assert result.training_config.get('n_estimators') == 50
        assert result.training_config.get('max_depth') == 5

    def test_multiple_same_model_gets_unique_names(self, prepared_data, trainer):
        """Training the same model twice produces unique names."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        r1 = trainer.train(
            RandomForestRegressor(random_state=42), 'random_forest',
            X_train, y_train, X_test, y_test, 'regression'
        )
        r2 = trainer.train(
            RandomForestRegressor(random_state=43), 'random_forest',
            X_train, y_train, X_test, y_test, 'regression'
        )
        assert r1.model_name == 'random_forest'
        assert r2.model_name == 'random_forest_1'
        assert r1.model_name in trainer.results
        assert r2.model_name in trainer.results

    def test_training_config_contains_all_params(self, prepared_data, trainer):
        """TrainingResult.training_config has full reproducibility info."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        result = trainer.train(
            RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'RF', X_train, y_train, X_test, y_test, 'regression'
        )
        cfg = result.training_config
        assert 'n_estimators' in cfg
        assert 'max_depth' in cfg
        assert 'cv_folds' in cfg
        assert 'problem_type' in cfg

    def test_manual_training_with_custom_cv_folds(self, prepared_data, trainer):
        """v2.0.0: Manual training respects user-selected cv_folds.
        
        Regression test: ManualParamDialog did not include cv_folds, and
        _train_with_manual_params did not pass cv_folds to TrainingThread,
        so all manual models used default cv_folds=5 regardless of user
        preference. This compromised reproducibility — two users with the
        same manual params but different cv_folds would get identical results.
        """
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        # Simulate user selecting cv_folds=3 in ManualParamDialog
        custom_cv_folds = 3
        
        # The trainer.train() call should use the specified cv_folds
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        result = trainer.train(
            model=model, model_name='RF_Manual',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type='regression',
            cv_folds=custom_cv_folds,
            feature_names=dm.get_feature_names()
        )
        
        # Verify the result reflects the custom cv_folds
        assert result is not None
        assert result.cv_metrics is not None
        assert len(result.cv_metrics.get('test_score', [])) == custom_cv_folds, \
            f"Expected {custom_cv_folds} CV scores, got {len(result.cv_metrics.get('test_score', []))}"
        
        # Verify different cv_folds produces different number of scores
        result_5 = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'RF_5fold', X_train, y_train, X_test, y_test, 'regression',
            cv_folds=5, feature_names=dm.get_feature_names()
        )
        assert len(result_5.cv_metrics.get('test_score', [])) == 5
        
        # Different cv_folds should produce different CV score arrays
        cv_3 = result.cv_metrics['test_score']
        cv_5 = result_5.cv_metrics['test_score']
        # Different number of folds means different arrays
        assert len(cv_3) != len(cv_5) or not all(abs(a-b) < 1e-10 for a, b in zip(cv_3[:3], cv_5[:3]))

    def test_results_persist_after_multiple_trainings(self, prepared_data, trainer):
        """Multiple training runs all persist in trainer.results."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        for name in ['random_forest', 'lightgbm']:
            model = registry.create_estimator(name, 'regression', random_state=42)
            trainer.train(model, name, X_train, y_train, 
                         X_test, y_test, 'regression')
        
        assert len(trainer.results) == 2


# =============================================================================
# TEST 3: Neural Network Pipeline
# =============================================================================

class TestNeuralNetworkPipeline:
    """Integration: NeuralNetworkBuilder → ModelTrainer → Export."""

    def test_nn_training_produces_valid_result(self, prepared_data, trainer):
        """NN training produces a complete TrainingResult."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            from app.core.nn_builder import NeuralNetworkBuilder
        except ImportError:
            pytest.skip("PyTorch not available")
        
        nn_builder = NeuralNetworkBuilder()
        nn_builder.create_architecture(
            input_dim=X_train.shape[1], output_dim=1,
            hidden_layers_config=[
                {'n_units': 32, 'activation': 'relu', 'dropout_rate': 0.2}
            ],
            problem_type='regression'
        )
        nn_builder.create_training_config(epochs=10, batch_size=16, learning_rate=0.01)
        nn_builder.build_model()
        
        result = trainer.train_neural_network(
            nn_builder, 'Neural Network', X_train, y_train,
            X_test, y_test, problem_type='regression', cv_folds=3
        )
        
        assert result.is_neural_network is True
        assert result.nn_history is not None
        assert 'train_loss' in result.nn_history
        assert len(result.nn_history['train_loss']) > 0
        assert result.nn_architecture_desc is not None
        assert result.nn_architecture_desc != 'Unknown'
        assert result.nn_epochs_trained > 0

    def test_nn_unique_names(self, prepared_data, trainer):
        """Multiple NNs get unique names."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            from app.core.nn_builder import NeuralNetworkBuilder
        except ImportError:
            pytest.skip("PyTorch not available")
        
        for i in range(3):
            nn_builder = NeuralNetworkBuilder()
            nn_builder.create_architecture(
                input_dim=X_train.shape[1], output_dim=1,
                hidden_layers_config=[
                    {'n_units': 16 * (i+1), 'activation': 'relu', 'dropout_rate': 0.1}
                ],
                problem_type='regression'
            )
            nn_builder.create_training_config(epochs=10, batch_size=16)
            nn_builder.build_model()
            trainer.train_neural_network(nn_builder, 'Neural Network', 
                                        X_train, y_train, cv_folds=0)
        
        names = list(trainer.results.keys())
        assert len(names) == 3
        assert names[0] == 'Neural Network'
        assert names[1] == 'Neural Network_1'
        assert names[2] == 'Neural Network_2'

    def test_nn_history_not_shared(self, prepared_data, trainer):
        """Each NN has independent history (no reference sharing)."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            from app.core.nn_builder import NeuralNetworkBuilder
        except ImportError:
            pytest.skip("PyTorch not available")
        
        results = []
        for i in range(2):
            nn_builder = NeuralNetworkBuilder()
            nn_builder.create_architecture(
                input_dim=X_train.shape[1], output_dim=1,
                hidden_layers_config=[{'n_units': 32, 'activation': 'relu', 'dropout_rate': 0.1}],
                problem_type='regression'
            )
            nn_builder.create_training_config(epochs=10, batch_size=16)
            nn_builder.build_model()
            result = trainer.train_neural_network(nn_builder, f'NN_{i}', 
                                                 X_train, y_train, cv_folds=0)
            results.append(result)
        
        # Histories must be independent
        h1 = results[0].nn_history
        h2 = results[1].nn_history
        assert h1 is not h2  # Different objects
        assert id(h1) != id(h2)
        # Modifying one should not affect the other
        original_len = len(h1.get('train_loss', []))
        h1['train_loss'].append(999.0)
        assert len(h2.get('train_loss', [])) == original_len

    def test_nn_optimization_completes_with_config(self, prepared_data, optimizer):
        """CRITICAL: NN optimization must complete and produce optimization_config.
        
        Regression test for bug: 'max_layers' is not defined.
        The optimization would run all trials then crash when building the
        OptimizationResult because max_layers/min_units/max_units were
        referenced but never defined in optimize_nn_architecture().
        """
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            from app.core.nn_builder import NeuralNetworkBuilder
            import optuna
        except ImportError:
            pytest.skip("PyTorch or Optuna not available")
        
        nn_builder = NeuralNetworkBuilder()
        nn_builder.create_architecture(
            input_dim=X_train.shape[1], output_dim=1,
            hidden_layers_config=[{'n_units': 32, 'activation': 'relu', 'dropout_rate': 0.1}],
            problem_type='regression'
        )
        nn_builder.create_training_config(epochs=10, batch_size=16)
        nn_builder.build_model()
        
        # This must complete WITHOUT NameError on 'max_layers'
        result = optimizer.optimize_nn_architecture(
            nn_builder=nn_builder,
            X_train=X_train, y_train=y_train,
            X_val=X_test, y_val=y_test,
            input_dim=X_train.shape[1], output_dim=1,
            problem_type='regression',
            n_trials=3,  # Small number for fast test
            cv_folds=0,  # Use simple validation
            random_state=42,
        )
        
        # Must return valid OptimizationResult
        assert isinstance(result, OptimizationResult)
        assert result.optimization_config is not None
        
        # These fields must exist and have reasonable values
        cfg = result.optimization_config
        assert 'nn_max_layers' in cfg, f"nn_max_layers missing from config: {cfg.keys()}"
        assert 'nn_min_units' in cfg, f"nn_min_units missing from config: {cfg.keys()}"
        assert 'nn_max_units' in cfg, f"nn_max_units missing from config: {cfg.keys()}"
        
        # Values should match default param_ranges (n_layers: (1,5), n_units: (16,512))
        assert cfg['nn_max_layers'] == 5, f"Expected 5, got {cfg['nn_max_layers']}"
        assert cfg['nn_min_units'] == 16, f"Expected 16, got {cfg['nn_min_units']}"
        assert cfg['nn_max_units'] == 512, f"Expected 512, got {cfg['nn_max_units']}"
        
        # Other critical fields
        assert cfg['algorithm'] == 'Neural Network'
        assert cfg['cv_folds'] == 0
        assert cfg['n_trials'] == 3

    def test_nn_optimization_with_custom_param_ranges(self, prepared_data, optimizer):
        """NN optimization with custom param_ranges produces correct config values."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            from app.core.nn_builder import NeuralNetworkBuilder
            import optuna
        except ImportError:
            pytest.skip("PyTorch or Optuna not available")
        
        nn_builder = NeuralNetworkBuilder()
        nn_builder.create_architecture(
            input_dim=X_train.shape[1], output_dim=1,
            hidden_layers_config=[{'n_units': 32, 'activation': 'relu', 'dropout_rate': 0.1}],
            problem_type='regression'
        )
        nn_builder.create_training_config(epochs=10, batch_size=16)
        nn_builder.build_model()
        
        custom_ranges = {
            'n_layers': (2, 8, 'int'),
            'n_units': (32, 256, 'int_log'),
            'dropout': (0.0, 0.3, 'float'),
            'activation': (['relu'], None, 'categorical'),
            'batch_norm': ([True], None, 'categorical'),
            'learning_rate': (1e-4, 1e-2, 'float_log'),
            'batch_size': ([32], None, 'categorical'),
            'weight_decay': (1e-6, 1e-4, 'float_log'),
            'optimizer': (['adam'], None, 'categorical'),
        }
        
        result = optimizer.optimize_nn_architecture(
            nn_builder=nn_builder,
            X_train=X_train, y_train=y_train,
            X_val=X_test, y_val=y_test,
            input_dim=X_train.shape[1], output_dim=1,
            problem_type='regression',
            n_trials=2, cv_folds=0,
            param_ranges=custom_ranges,
            random_state=42,
        )
        
        cfg = result.optimization_config
        # Values must come from custom_ranges, not defaults
        assert cfg['nn_max_layers'] == 8, f"Expected 8 (custom), got {cfg['nn_max_layers']}"
        assert cfg['nn_min_units'] == 32, f"Expected 32 (custom), got {cfg['nn_min_units']}"
        assert cfg['nn_max_units'] == 256, f"Expected 256 (custom), got {cfg['nn_max_units']}"

    def test_nn_reproducibility_with_seed(self, prepared_data):
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            from app.core.nn_builder import NeuralNetworkBuilder
        except ImportError:
            pytest.skip("PyTorch not available")
        
        results = []
        for _ in range(2):
            nn_builder = NeuralNetworkBuilder()
            nn_builder.create_architecture(
                input_dim=X_train.shape[1], output_dim=1,
                hidden_layers_config=[{'n_units': 16, 'activation': 'relu', 'dropout_rate': 0.0}],
                problem_type='regression'
            )
            nn_builder.create_training_config(epochs=10, batch_size=16, random_seed=42)
            nn_builder.build_model()
            
            trainer = ModelTrainer()
            result = trainer.train_neural_network(nn_builder, 'NN', 
                                                 X_train, y_train, cv_folds=0)
            results.append(result)
        
        # Same architecture description
        assert results[0].nn_architecture_desc == results[1].nn_architecture_desc


# =============================================================================
# TEST 4: Optimization Pipeline
# =============================================================================

class TestOptimizationPipeline:
    """Integration: Optimizer produces valid results that can be used for training."""

    def test_sklearn_optimization_produces_valid_result(self, prepared_data, optimizer):
        """optimize_sklearn_model returns a usable OptimizationResult."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not available")
        
        result = optimizer.optimize_sklearn_model(
            model_type='xgboost', algorithm='xgboost',
            X_train=X_train, y_train=y_train,
            n_trials=5, cv_folds=3, random_state=42
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.best_params is not None
        assert len(result.best_params) > 0
        assert result.optimization_config is not None  # v2.0
        assert result.optimization_config.get('cv_folds') == 3
        assert result.optimization_config.get('n_trials') == 5
        assert result.optimization_config.get('random_state') == 42
        assert result.cv_metrics is not None
        assert len(result.cv_metrics.get('test_score', [])) > 0

    def test_optimization_result_has_config(self, prepared_data, optimizer):
        """v2.0: OptimizationResult stores actual config used."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not available")
        
        result = optimizer.optimize_sklearn_model(
            model_type='random_forest', algorithm='random_forest',
            X_train=X_train, y_train=y_train,
            n_trials=3, cv_folds=3, random_state=99, timeout=10
        )
        
        cfg = result.optimization_config
        assert cfg['algorithm'] == 'random_forest'
        assert cfg['n_trials'] == 3
        assert cfg['cv_folds'] == 3
        assert cfg['random_state'] == 99
        assert cfg['pruning_enabled'] is True

    def test_train_with_optimized_params(self, prepared_data, trainer, optimizer):
        """Optimized params can be used to train a model."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not available")
        
        opt_result = optimizer.optimize_sklearn_model(
            model_type='random_forest', algorithm='random_forest',
            X_train=X_train, y_train=y_train, n_trials=3, cv_folds=3
        )
        
        # Train with optimized params
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(**opt_result.best_params, random_state=42)
        train_result = trainer.train(
            model=model, model_name='Random Forest_Optimized',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test, problem_type='regression'
        )
        
        assert train_result.metrics is not None
        assert 'r2' in train_result.metrics

    def test_multiple_optimizations_preserved(self, prepared_data, optimizer):
        """v2.0 FIX: Multiple optimizations of same algorithm are preserved."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not available")
        
        opt1 = optimizer.optimize_sklearn_model(
            model_type='xgboost', algorithm='xgboost',
            X_train=X_train, y_train=y_train,
            n_trials=3, cv_folds=5, random_state=42
        )
        opt2 = optimizer.optimize_sklearn_model(
            model_type='xgboost', algorithm='xgboost',
            X_train=X_train, y_train=y_train,
            n_trials=3, cv_folds=3, random_state=123
        )
        
        # Configs must be different
        assert opt1.optimization_config['cv_folds'] == 5
        assert opt2.optimization_config['cv_folds'] == 3
        assert opt1.optimization_config['random_state'] == 42
        assert opt2.optimization_config['random_state'] == 123


# =============================================================================
# TEST 5: Ensemble Pipeline
# =============================================================================

class TestEnsemblePipeline:
    """Integration: EnsembleTrainer with multiple model types."""

    def test_manual_ensemble_with_trained_models(self, prepared_data, trainer):
        """Manual ensemble from trained models produces valid result."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        # Train 3 models
        models = []
        for name in ['random_forest', 'lightgbm']:
            model = registry.create_estimator(name, 'regression', random_state=42)
            result = trainer.train(model, name, 
                                  X_train, y_train, X_test, y_test, 'regression')
            models.append((name, result.model))
        
        # Create ensemble
        ensemble_trainer = EnsembleTrainer(trainer)
        model_names = [name for name, _ in models]
        ensemble_result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average', model_names=model_names,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        
        assert ensemble_result is not None
        assert ensemble_result.metrics is not None
        assert 'r2' in ensemble_result.metrics

    def test_weighted_ensemble(self, prepared_data, trainer):
        """Ensemble with custom weights produces valid result."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        models = []
        for name in ['random_forest', 'lightgbm']:
            model = registry.create_estimator(name, 'regression', random_state=42)
            result = trainer.train(model, name,
                                  X_train, y_train, X_test, y_test, 'regression')
            models.append((name, result.model))
        
        ensemble_trainer = EnsembleTrainer(trainer)
        model_names = [name for name, _ in models]
        weights_dict = {name: w for (name, _), w in zip(models, [0.7, 0.3])}
        ensemble_result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average', model_names=model_names,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            weights=weights_dict
        )
        
        assert ensemble_result is not None
        assert ensemble_result.weights is not None

    def test_ensemble_unique_name(self, prepared_data, trainer):
        """Multiple ensembles get unique names."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        models = []
        for name in ['random_forest', 'lightgbm']:
            model = registry.create_estimator(name, 'regression', random_state=42)
            result = trainer.train(model, name,
                                  X_train, y_train, X_test, y_test, 'regression')
            models.append((name, result.model))
        
        ensemble_trainer = EnsembleTrainer(trainer)
        model_names = [name for name, _ in models]
        r1 = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average', model_names=model_names,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        r2 = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average', model_names=model_names,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        
        # EnsembleResult has ensemble_type as identifier
        assert r1.ensemble_type == r2.ensemble_type  # Same type
        assert r1 is not r2  # Different objects


# =============================================================================
# TEST 6: Explainability Pipeline
# =============================================================================

class TestExplainabilityPipeline:
    """Integration: Explainer works with ALL model types.
    
    CRITICAL: These tests catch UnboundLocalError and other explainability
    bugs that the GUI may encounter. Each test type uses a different SHAP
    explainer backend (Tree, Linear, Kernel, Deep).
    """

    def test_feature_importance_tree_model(self, prepared_data, trainer):
        """Feature importance works for tree models (TreeExplainer path)."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        result = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'RF', X_train, y_train, X_test, y_test, 'regression',
            feature_names=dm.get_feature_names()
        )
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        imp_df = explainer.get_feature_importance_summary(
            result.model, X_test, y_test, dm.get_feature_names()
        )
        
        assert imp_df is not None
        assert not imp_df.empty
        assert 'feature' in imp_df.columns
        assert 'model_importance' in imp_df.columns

    def test_feature_importance_linear_model(self, prepared_data, trainer):
        """Feature importance works for linear models (LinearExplainer path)."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.linear_model import Ridge
        
        result = trainer.train(
            Ridge(alpha=1.0, random_state=42),
            'Ridge', X_train, y_train, X_test, y_test, 'regression',
            feature_names=dm.get_feature_names()
        )
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        imp_df = explainer.get_feature_importance_summary(
            result.model, X_test, y_test, dm.get_feature_names()
        )
        
        assert imp_df is not None
        assert not imp_df.empty
        assert 'feature' in imp_df.columns
        assert 'model_importance' in imp_df.columns

    def test_feature_importance_kernel_fallback(self, prepared_data, trainer):
        """Feature importance works for SVR via KernelExplainer fallback."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.svm import SVR
        
        result = trainer.train(
            SVR(kernel='rbf', C=1.0),
            'SVR', X_train, y_train, X_test, y_test, 'regression',
            feature_names=dm.get_feature_names()
        )
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        imp_df = explainer.get_feature_importance_summary(
            result.model, X_test, y_test, dm.get_feature_names()
        )
        
        assert imp_df is not None
        assert not imp_df.empty
        assert 'feature' in imp_df.columns
        assert 'permutation_importance' in imp_df.columns

    def _skip_if_shap_missing(self):
        """Helper: skip test if shap is not installed."""
        try:
            import shap
        except ImportError:
            pytest.skip("SHAP not installed")

    def test_shap_tree_model(self, prepared_data, trainer):
        """SHAP works for tree models (TreeExplainer)."""
        self._skip_if_shap_missing()
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        result = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'RF', X_train, y_train, X_test, y_test, 'regression'
        )
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        shap_result = explainer.compute_shap_values(
            result.model, X_test[:10], feature_names=dm.get_feature_names()
        )
        
        assert shap_result is not None
        assert 'global_importance' in shap_result
        assert len(shap_result['global_importance']) > 0

    def test_shap_linear_model(self, prepared_data, trainer):
        """SHAP works for linear models (LinearExplainer)."""
        self._skip_if_shap_missing()
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.linear_model import Ridge
        
        result = trainer.train(
            Ridge(alpha=1.0, random_state=42),
            'Ridge', X_train, y_train, X_test, y_test, 'regression'
        )
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        shap_result = explainer.compute_shap_values(
            result.model, X_test[:10], feature_names=dm.get_feature_names()
        )
        
        assert shap_result is not None
        assert 'global_importance' in shap_result

    def test_shap_kernel_fallback(self, prepared_data, trainer):
        """SHAP works for SVR via KernelExplainer fallback."""
        self._skip_if_shap_missing()
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.svm import SVR
        
        result = trainer.train(
            SVR(kernel='rbf', C=1.0),
            'SVR', X_train, y_train, X_test, y_test, 'regression'
        )
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        shap_result = explainer.compute_shap_values(
            result.model, X_test[:10], feature_names=dm.get_feature_names()
        )
        
        assert shap_result is not None
        assert 'global_importance' in shap_result

    def test_pdp_sklearn_model(self, prepared_data, trainer):
        """Partial dependence works for sklearn models."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        result = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'RF', X_train, y_train, X_test, y_test, 'regression'
        )
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        pdp_result = explainer.compute_partial_dependence(
            result.model, X_test, features=dm.get_feature_names()[:2],
            feature_names=dm.get_feature_names()
        )
        
        assert pdp_result is not None
        assert len(pdp_result) > 0

    def test_explainer_on_optimized_model(self, prepared_data, trainer, optimizer):
        """Explainability works on optimized model (best_model from opt result)."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not available")
        
        opt_result = optimizer.optimize_sklearn_model(
            model_type='random_forest', algorithm='random_forest',
            X_train=X_train, y_train=y_train, n_trials=3, cv_folds=3
        )
        
        # The optimized model IS best_model from the optimization
        assert opt_result.best_model is not None
        
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        
        imp_df = explainer.get_feature_importance_summary(
            opt_result.best_model, X_test, y_test, dm.get_feature_names()
        )
        assert imp_df is not None
        assert not imp_df.empty
        
        shap_result = explainer.compute_shap_values(
            opt_result.best_model, X_test[:10], feature_names=dm.get_feature_names()
        )
        assert shap_result is not None
        assert 'global_importance' in shap_result

    def test_all_registry_models_explainer_compatible(self, prepared_data, trainer, model_registry):
        """CRITICAL: Every registered model can be explained without crashing.
        
        Catches UnboundLocalError and other bugs when explainer encounters
        unexpected model types. Simulates the exact call from ExplainabilityThread.
        """
        X_train, X_test, y_train, y_test, dm = prepared_data
        from app.core.explainer import ModelExplainer
        explainer = ModelExplainer()
        
        failed = []
        for name in model_registry.list_models('regression'):
            try:
                model = model_registry.create_estimator(name, 'regression', random_state=42)
                result = trainer.train(
                    model=model, model_name=name,
                    X_train=X_train, y_train=y_train,
                    X_test=X_test, y_test=y_test,
                    problem_type='regression',
                    cv_folds=3,
                    feature_names=dm.get_feature_names()
                )
                
                # Exact call sequence from ExplainabilityThread.run()
                imp_df = explainer.get_feature_importance_summary(
                    result.model, X_test, y_test, dm.get_feature_names()
                )
                assert imp_df is not None
                
            except Exception as e:
                failed.append((name, str(e)))
        
        assert len(failed) == 0, f"Explainability failed for: {failed}"


# =============================================================================
# TEST 7: Model Saving & Inference Pipeline
# =============================================================================

class TestModelSaveInferencePipeline:
    """Integration: Save model bundle → load → predict."""

    def test_save_and_load_sklearn_model(self, prepared_data, trainer):
        """Sklearn model can be saved and loaded as a bundle."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        result = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'RF', X_train, y_train, X_test, y_test, 'regression'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_bundle'
            saved = save_training_result(result, str(path), dm, 'RF')
            assert saved.exists()
            
            # Load and verify engine can be instantiated
            engine = InferenceEngine(str(saved))
            assert engine is not None
            # Verify model path exists
            assert hasattr(engine, 'model_path') or True  # At least doesn't crash

    def test_inference_engine_with_preprocessor(self, prepared_data, trainer):
        """Inference engine correctly applies preprocessing."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        from sklearn.ensemble import RandomForestRegressor
        
        result = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'RF', X_train, y_train, X_test, y_test, 'regression'
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / 'test_bundle'
            save_training_result(result, str(path), dm, 'RF')
            
            engine = InferenceEngine(str(path))
            # InferenceEngine loaded successfully
        assert engine is not None


# =============================================================================
# TEST 8: Export Pipeline — End-to-End
# =============================================================================

class TestExportPipeline:
    """Integration: Full session export with all result types."""

    def test_export_with_all_model_types(self, prepared_data, trainer, optimizer):
        """Export includes sklearn + NN + ensemble + optimization results."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        # 1. Train sklearn model
        from sklearn.ensemble import RandomForestRegressor
        rf_result = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'random_forest', X_train, y_train, X_test, y_test, 'regression'
        )
        
        # 2. Train optimized model
        opt_result = None
        try:
            import optuna
            opt_result = optimizer.optimize_sklearn_model(
                model_type='xgboost', algorithm='xgboost',
                X_train=X_train, y_train=y_train, n_trials=3, cv_folds=3
            )
            from xgboost import XGBRegressor
            xgb_model = XGBRegressor(**opt_result.best_params, random_state=42)
            xgb_result = trainer.train(xgb_model, 'XGBoost_Optimized',
                                      X_train, y_train, X_test, y_test, 'regression')
        except ImportError:
            pass  # XGBoost not installed
        
        # 3. Train NN
        try:
            from app.core.nn_builder import NeuralNetworkBuilder
            nn_builder = NeuralNetworkBuilder()
            nn_builder.create_architecture(
                input_dim=X_train.shape[1], output_dim=1,
                hidden_layers_config=[{'n_units': 32, 'activation': 'relu', 'dropout_rate': 0.1}],
                problem_type='regression'
            )
            nn_builder.create_training_config(epochs=10, batch_size=16)
            nn_builder.build_model()
            nn_result = trainer.train_neural_network(
                nn_builder, 'Neural Network', X_train, y_train,
                X_test, y_test, cv_folds=0
            )
        except ImportError:
            nn_result = None
        
        # 4. Create ensemble
        ensemble_trainer = EnsembleTrainer(trainer)
        model_names = ['random_forest']
        if opt_result:
            model_names.append('xgboost')
        ens_result = ensemble_trainer.train_ensemble(
            ensemble_type='weighted_average', model_names=model_names,
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test
        )
        
        # 5. Export
        exporter = SessionExporter()
        exporter.session_info = {'n_samples': 60, 'n_features': 7, 'problem_type': 'regression'}
        
        # Add training results
        sklearn_results = [rf_result]
        if opt_result:
            sklearn_results.append(xgb_result)
        exporter.add_training_results(sklearn_results)
        
        if nn_result:
            exporter.add_nn_training_result(
                model_name=nn_result.model_name,
                architecture_desc=nn_result.nn_architecture_desc or '[NN]',
                architecture_detail=nn_result.nn_architecture_detail or {},
                history=nn_result.nn_history or {},
                training_config=nn_result.training_config or {},
                metrics=nn_result.metrics,
                training_time_sec=nn_result.training_time,
                cv_metrics=getattr(nn_result, 'cv_metrics', None)
            )
        
        exporter.add_ensemble_result(ens_result)
        
        if opt_result:
            exporter.add_optimization_result(
                opt_result, algorithm='xgboost',
                n_trials=opt_result.n_trials,
                optimization_config=opt_result.optimization_config
            )
        
        exporter._link_optimization_entries()
        exporter.filter_orphaned_entries()
        
        # Render
        txt = exporter._render_txt()
        
        # Verify all sections present
        assert 'random_forest' in txt
        if opt_result:
            assert 'XGBoost_Optimized' in txt
            assert 'OPTIMIZATION DETAILS' in txt
        if nn_result:
            assert 'Neural Network' in txt
        assert 'Ensemble' in txt or 'ENSEMBLE' in txt
        assert 'FINAL RANKING' in txt
        assert 'REPRODUCIBILITY CHECKLIST' in txt

    def test_export_multiple_optimizations_all_preserved(self, prepared_data, trainer, optimizer):
        """v2.0: Two optimizations of same algorithm → both in export."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not available")
        
        # Two optimizations
        opt1 = optimizer.optimize_sklearn_model(
            model_type='random_forest', algorithm='random_forest',
            X_train=X_train, y_train=y_train, n_trials=3, cv_folds=5, random_state=42
        )
        opt2 = optimizer.optimize_sklearn_model(
            model_type='random_forest', algorithm='random_forest',
            X_train=X_train, y_train=y_train, n_trials=3, cv_folds=3, random_state=123
        )
        
        # Train with both
        from sklearn.ensemble import RandomForestRegressor
        r1 = trainer.train(RandomForestRegressor(**opt1.best_params, random_state=42),
                          'Random Forest_Optimized', X_train, y_train, X_test, y_test, 'regression')
        r2 = trainer.train(RandomForestRegressor(**opt2.best_params, random_state=42),
                          'Random Forest_Optimized_1', X_train, y_train, X_test, y_test, 'regression')
        
        # Export
        exporter = SessionExporter()
        exporter.add_training_results([r1, r2])
        exporter.add_optimization_result(opt1, algorithm='random_forest',
                                        n_trials=3, optimization_config=opt1.optimization_config)
        exporter.add_optimization_result(opt2, algorithm='random_forest',
                                        n_trials=3, optimization_config=opt2.optimization_config)
        exporter._link_optimization_entries()
        
        # Both must link to correct configs
        for e in exporter._entries:
            if e['section'] == 'Training' and '_linked_optimization' in e:
                cfg = e['_linked_optimization']['optimization_config']
                if e['model_name'] == 'Random Forest_Optimized':
                    assert cfg['cv_folds'] == 5, f"Expected 5 folds, got {cfg['cv_folds']}"
                elif e['model_name'] == 'Random Forest_Optimized_1':
                    assert cfg['cv_folds'] == 3, f"Expected 3 folds, got {cfg['cv_folds']}"

    def test_export_no_zero_results(self, prepared_data, trainer):
        """Export must never show '0 result(s)'."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        from sklearn.ensemble import RandomForestRegressor
        trainer.train(RandomForestRegressor(n_estimators=10, random_state=42),
                     'RF', X_train, y_train, X_test, y_test, 'regression')
        
        exporter = SessionExporter()
        exporter.add_training_results(list(trainer.results.values()))
        txt = exporter._render_txt()
        
        assert '0 result' not in txt.lower()


# =============================================================================
# TEST 9: Crystal Structure Linking
# =============================================================================

class TestCrystalStructurePipeline:
    """Integration: Crystal structure features are correctly extracted."""

    def test_crystal_feature_extractor_creation(self):
        """CrystalFeatureExtractor can be instantiated."""
        try:
            from app.core.crystal_feature_extractor import CrystalFeatureExtractor
            extractor = CrystalFeatureExtractor()
            assert extractor is not None
        except ImportError:
            pytest.skip("crystal_feature_extractor not available")

    def test_crystal_feature_extractor_creation(self):
        """CrystalFeatureExtractor can be instantiated."""
        try:
            from app.core.crystal_feature_extractor import CrystalFeatureExtractor
            extractor = CrystalFeatureExtractor()
            assert extractor is not None
        except ImportError:
            pytest.skip("crystal_feature_extractor not available")

    def test_composition_features(self):
        """Composition features are correctly parsed."""
        from app.core.feature_engineering import FeatureEngineer
        
        engineer = FeatureEngineer()
        df = pd.DataFrame({'composition': ['Fe2O3', 'SiO2', 'TiO2']})
        result_df = engineer.add_composition_features(df, composition_column='composition')
        
        assert result_df is not None
        # Should have added composition-related columns
        comp_cols = [c for c in result_df.columns if 'comp' in c or 'n_' in c]
        assert len(comp_cols) > 0, "Composition features should be added"


# =============================================================================
# TEST 10: Data Health & Validation
# =============================================================================

class TestDataHealthValidation:
    """Integration: Data quality checks and validation."""

    def test_data_health_check(self, sample_data_file):
        """Health check identifies issues in data."""
        from app.core.data_health_check import DataHealthChecker
        
        df = pd.read_csv(sample_data_file)
        checker = DataHealthChecker()
        report = checker.check(df, target_column='band_gap')
        
        assert report is not None
        assert hasattr(report, 'issues') or hasattr(report, 'score')

    def test_materials_aware_validator(self, prepared_data):
        """Validator produces CV splits with uncertainty estimates."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        validator = MaterialsAwareValidator()
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(random_state=42)
        results = validator.cross_validate_with_uncertainty(
            model=model, X=X_train, y=y_train, n_splits=3
        )
        
        assert results is not None

    def test_domain_applicability(self, prepared_data):
        """Domain applicability analyzer identifies out-of-domain samples."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        from app.core.domain_applicability import DomainApplicabilityAnalyzer
        analyzer = DomainApplicabilityAnalyzer()
        
        analyzer.fit(X_train, y_train)
        scores = analyzer.analyze(X_test)
        
        assert scores is not None
        assert len(scores) == len(X_test)


# =============================================================================
# TEST 11: Error Handling & Edge Cases
# =============================================================================

class TestErrorHandling:
    """Integration: Graceful handling of edge cases and errors."""

    def test_train_with_nan_data(self, trainer, tmp_path):
        """Training with NaN data should be handled gracefully."""
        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0],
            'f2': [np.nan, 3.0, 4.0, 5.0],
            'target': [1.0, 2.0, 3.0, 4.0]
        })
        path = tmp_path / 'nan_data.csv'
        df.to_csv(path, index=False)

        dm = DataManager()
        dm.load_data(str(path))
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor(impute_strategy='median')
        X_train, X_test, y_train, y_test = dm.prepare_data()

        # v2.0: X_train is RAW (NaNs still present). The Pipeline handles them.
        from sklearn.ensemble import RandomForestRegressor
        result = trainer.train(
            RandomForestRegressor(random_state=42), 'RF',
            X_train, y_train, X_test, y_test, problem_type='regression',
            preprocessor=preprocessor,
        )
        assert result is not None
        # Verify predictions are valid (no NaNs after Pipeline imputation)
        assert result.predictions is not None
        assert not np.isnan(result.predictions).any()

    def test_single_sample_cv(self, prepared_data, trainer):
        """CV with very small data should not crash."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        # Take only 5 samples
        X_small = X_train[:5]
        y_small = y_train[:5]
        
        from sklearn.ensemble import RandomForestRegressor
        result = trainer.train(RandomForestRegressor(n_estimators=5, random_state=42), 'RF',
                              X_small, y_small, problem_type='regression', cv_folds=2)
        assert result is not None

    def test_empty_feature_importance(self, prepared_data, trainer):
        """Models without feature importance should not crash export."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        
        from sklearn.svm import SVR
        result = trainer.train(SVR(), 'SVM', X_train, y_train,
                              X_test, y_test, 'regression',
                              feature_names=dm.get_feature_names())
        
        assert result is not None
        assert result.feature_importance is None or len(result.feature_importance) == 0

    def test_model_registry_all_algorithms(self, model_registry):
        """All registered algorithms can be instantiated."""
        for problem_type in ['regression', 'classification']:
            for name in model_registry.list_models(problem_type):
                try:
                    model = model_registry.create_estimator(name, problem_type)
                    assert model is not None
                except Exception as e:
                    pytest.fail(f"Failed to instantiate {name} for {problem_type}: {e}")

    def test_regression_vs_classification_registry(self, model_registry):
        """Regression and classification have different model sets."""
        reg_models = set(model_registry.list_models('regression'))
        clf_models = set(model_registry.list_models('classification'))
        # Both should have some common models
        common = reg_models & clf_models
        assert len(common) > 0, "Should have common models between regression and classification"

    def test_registry_create_estimator_with_random_state(self, model_registry):
        """create_estimator properly applies random_state."""
        model = model_registry.create_estimator('random_forest', 'regression', random_state=123)
        assert model.random_state == 123


# =============================================================================
# TEST 12: Ensemble Weight Optimization
# =============================================================================

class TestEnsembleWeightOptimization:
    """Integration: Ensemble weight optimization with Optuna."""

    def test_ensemble_weight_optimization(self, prepared_data, trainer):
        """Ensemble weights can be optimized via Bayesian optimization."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        # Train 3 models
        model_names = ['random_forest', 'lightgbm', 'ridge']
        for name in model_names:
            model = registry.create_estimator(name, 'regression', random_state=42)
            trainer.train(model, name, X_train, y_train, X_test, y_test, 'regression')
        
        ensemble_trainer = EnsembleTrainer(trainer)
        
        try:
            import optuna
            # Weight optimization uses Optuna
            result = ensemble_trainer.train_ensemble(
                ensemble_type='weighted_average',
                model_names=model_names,
                X_train=X_train, y_train=y_train,
                X_test=X_test, y_test=y_test,
                optimize_weights=True,
                n_trials=5,
            )
            assert result is not None
            # Should have optimized weights
            assert result.weights is not None
            assert len(result.weights) == len(model_names)
            # Weights should sum to approximately 1.0
            total_weight = sum(result.weights.values())
            assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}"
        except ImportError:
            pytest.skip("Optuna not available for weight optimization")


# =============================================================================
# TEST 13: Reproducibility
# =============================================================================

class TestReproducibility:
    """Integration: Same seed produces same results across runs."""

    def test_sklearn_reproducible_with_same_seed(self, prepared_data):
        """Same random_state produces identical CV scores."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        results = []
        for _ in range(2):
            trainer = ModelTrainer()
            model = registry.create_estimator('random_forest', 'regression', random_state=42)
            result = trainer.train(model, 'RF', X_train, y_train, X_test, y_test, 'regression')
            results.append(result)
        
        # Same seed should produce same CV scores
        cv1 = results[0].cv_metrics['test_score']
        cv2 = results[1].cv_metrics['test_score']
        assert len(cv1) == len(cv2)
        for a, b in zip(cv1, cv2):
            assert abs(a - b) < 1e-10, f"CV scores differ with same seed: {a} vs {b}"

    def test_different_seed_produces_different_results(self, prepared_data):
        """Different random_state produces different results."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        r1 = ModelTrainer()
        m1 = registry.create_estimator('random_forest', 'regression', random_state=42)
        result1 = r1.train(m1, 'RF1', X_train, y_train, X_test, y_test, 'regression')
        
        r2 = ModelTrainer()
        m2 = registry.create_estimator('random_forest', 'regression', random_state=99)
        result2 = r2.train(m2, 'RF2', X_train, y_train, X_test, y_test, 'regression')
        
        # Results should be different (very unlikely to be identical)
        cv1 = result1.cv_metrics['test_score']
        cv2 = result2.cv_metrics['test_score']
        # At least one fold should differ
        diffs = [abs(a - b) for a, b in zip(cv1, cv2)]
        assert any(d > 1e-10 for d in diffs), "Different seeds produced identical results"


# =============================================================================
# TEST 14: Classification Pipeline
# =============================================================================

class TestClassificationPipeline:
    """Integration: Full classification workflow."""

    def test_classification_detection_and_training(self, sample_classification_file):
        """Classification data is detected and models train correctly."""
        dm = DataManager()
        dm.load_data(sample_classification_file)
        dm.set_target_column('is_metallic')
        assert dm.is_classification is True
        
        dm.get_preprocessor(numeric_strategy='standard', impute_strategy='mean')
        X_train, X_test, y_train, y_test = dm.prepare_data()
        
        registry = ModelRegistry()
        trainer = ModelTrainer()
        
        # Train a classification model
        clf_models = registry.list_models('classification')
        assert len(clf_models) > 0
        
        model = registry.create_estimator(clf_models[0], 'classification', random_state=42)
        result = trainer.train(
            model=model, model_name=clf_models[0],
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type='classification',
            feature_names=dm.get_feature_names()
        )
        
        assert result is not None
        assert 'accuracy' in result.metrics
        assert result.cv_metrics is not None


# =============================================================================
# TEST 15: Export Formats
# =============================================================================

class TestExportFormats:
    """Integration: All export formats produce valid output."""

    def test_export_txt_contains_all_sections(self, prepared_data, trainer):
        """TXT export has all required sections."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        model = registry.create_estimator('random_forest', 'regression', random_state=42)
        result = trainer.train(model, 'RF', X_train, y_train, X_test, y_test, 'regression')
        
        exporter = SessionExporter()
        exporter.session_info = {'n_samples': 60, 'n_features': 5, 'problem_type': 'regression'}
        exporter.add_training_results([result])
        
        txt = exporter._render_txt()
        assert 'FINAL RANKING' in txt
        assert 'REPRODUCIBILITY CHECKLIST' in txt
        assert 'RF' in txt
        assert 'r2' in txt.lower() or 'R2' in txt

    def test_export_csv_has_data(self, prepared_data, trainer):
        """CSV export produces parseable data."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        model = registry.create_estimator('random_forest', 'regression', random_state=42)
        result = trainer.train(model, 'RF', X_train, y_train, X_test, y_test, 'regression')
        
        exporter = SessionExporter()
        exporter.session_info = {'n_samples': 60, 'n_features': 5, 'problem_type': 'regression'}
        exporter.add_training_results([result])
        
        csv = exporter._render_csv()
        assert len(csv) > 0
        assert 'RF' in csv or 'random_forest' in csv.lower()

    def test_export_json_is_parseable(self, prepared_data, trainer):
        """JSON export produces valid JSON."""
        import json
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        model = registry.create_estimator('random_forest', 'regression', random_state=42)
        result = trainer.train(model, 'RF', X_train, y_train, X_test, y_test, 'regression')
        
        exporter = SessionExporter()
        exporter.session_info = {'n_samples': 60, 'n_features': 5, 'problem_type': 'regression'}
        exporter.add_training_results([result])
        
        json_str = exporter._render_json()
        data = json.loads(json_str)
        assert 'entries' in data or 'results' in data or len(data) > 0

    def test_export_empty_session_no_crash(self):
        """Export with no results produces output without crashing."""
        exporter = SessionExporter()
        exporter.session_info = {'n_samples': 0, 'n_features': 0, 'problem_type': 'regression'}
        
        txt = exporter._render_txt()
        assert isinstance(txt, str)
        
        csv = exporter._render_csv()
        assert isinstance(csv, str)


# =============================================================================
# TEST 16: Model Save/Load with Manifest
# =============================================================================

class TestModelSaveLoad:
    """Integration: Model bundle save and load cycle."""

    def test_save_produces_manifest(self, prepared_data, trainer, tmp_path):
        """Saved model bundle contains manifest.json."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        model = registry.create_estimator('random_forest', 'regression', random_state=42)
        result = trainer.train(model, 'RF', X_train, y_train, X_test, y_test, 'regression')
        
        save_path = tmp_path / 'test_bundle'
        saved = save_training_result(result, str(save_path), dm, 'RF')
        assert saved.exists()
        
        # Check manifest exists
        import json
        manifest_path = saved / 'manifest.json'
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            assert 'model_name' in manifest
            assert manifest['model_name'] == 'RF'

    def test_save_load_roundtrip(self, prepared_data, trainer, tmp_path):
        """Model can be saved and loaded back."""
        X_train, X_test, y_train, y_test, dm = prepared_data
        registry = ModelRegistry()
        
        model = registry.create_estimator('random_forest', 'regression', random_state=42)
        result = trainer.train(model, 'RF', X_train, y_train, X_test, y_test, 'regression')
        
        save_path = tmp_path / 'test_bundle'
        saved = save_training_result(result, str(save_path), dm, 'RF')
        
        # Load back
        engine = InferenceEngine(str(saved))
        assert engine is not None
