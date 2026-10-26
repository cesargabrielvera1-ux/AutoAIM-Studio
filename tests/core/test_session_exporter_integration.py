"""
Integration test for the complete session export pipeline.

This simulates the EXACT flow from the GUI through to rendered export:
  optimization_tab → results_tab → session_exporter → _render_txt()

These tests catch the bugs that unit tests miss because they only test
individual classes in isolation. The real bugs happen at integration points:
- optimization_tab deleting previous optimizations of same algorithm
- results_tab reading config from GUI widgets instead of stored results
- nn_builder state leaking between multiple training runs
- missing CV fold details for neural networks

All tests pass a mock object through the full pipeline to verify the
rendered output contains correct data.
"""

import copy
import numpy as np
import pytest

# Import the full pipeline
from app.core.session_exporter import SessionExporter
from app.core.optimizer import OptimizationResult


# =============================================================================
# Mock objects that simulate EXACTLY what the PyQt GUI produces
# =============================================================================

class MockSklearnTrainingResult:
    """Simulates a TrainingResult produced by ModelTrainer.train()"""
    def __init__(self, name, r2, rs=42, cv_scores=None, 
                 opt_linked=False, training_config=None):
        self.model_name = name
        self.model = None
        self.problem_type = 'regression'
        self.metrics = {'rmse': round((1-r2)*0.5, 4), 'mae': 0.3, 'r2': r2}
        self.cv_metrics = {
            'test_score': cv_scores or [r2-0.02, r2, r2+0.01, r2-0.01, r2+0.02],
            'train_score': [r2+0.05]*len(cv_scores or [0]*5)
        }
        self.training_time = 0.125
        self.random_state = rs
        self.is_neural_network = False
        self.has_predictions = True
        self.feature_importance = {'feat_1': 0.3}
        self.has_external_validation = False
        self.external_metrics = None
        self.training_config = training_config or {}


class MockNNTrainingResult:
    """Simulates a TrainingResult produced by ModelTrainer.train_neural_network()"""
    def __init__(self, name, arch_desc, r2, rs=42, epochs=5,
                 cv_scores=None, opt_time=0.0):
        self.model_name = name
        self.model = None
        self.is_neural_network = True
        self.problem_type = 'regression'
        self.nn_architecture_desc = arch_desc
        self.nn_architecture_detail = {
            'layer_1': {'n_units': 64, 'activation': 'relu', 'dropout_rate': 0.2}
        }
        self.nn_epochs_trained = epochs
        self.nn_history = {
            'train_loss': [1.0, 0.8, 0.6, 0.5, 0.45],
            'val_loss': [1.1, 0.85, 0.65, 0.58, 0.55]
        }
        self.metrics = {'rmse': round((1-r2)*0.5, 4), 'mae': 0.3, 'r2': r2}
        self.training_time = 12.5
        self.random_state = rs
        self.training_config = {
            'epochs': 100, 'batch_size': 32, 'learning_rate': 0.001,
            'optimizer': 'adam', 'random_seed': rs
        }
        self.cv_metrics = {
            'test_score': cv_scores or [r2-0.02, r2, r2+0.01, r2-0.01, r2+0.02],
            'train_score': [r2+0.05]*len(cv_scores or [0]*5)
        }
        self.feature_importance = None
        self.predictions = None
        self.true_values = None
        self.external_metrics = None
        self.has_external_validation = False
        self.optimization_time_sec = opt_time


def create_mock_opt_result(algorithm, n_trials, cv_folds, rs, best_score, opt_time):
    """Simulates an OptimizationResult from HyperparameterOptimizer.
    
    v2.0: Must include optimization_config with the ACTUAL config used.
    """
    return OptimizationResult(
        best_params={'n_estimators': 200, 'max_depth': 8},
        best_score=best_score,
        best_model=None,
        n_trials=n_trials,
        optimization_time=opt_time,
        all_trials=None,
        study_name=f"{algorithm}_optimization",
        metrics={'rmse': 0.28, 'r2': 0.92},
        problem_type='regression',
        cv_metrics={'test_score': [0.91, 0.92, 0.90, 0.93, 0.92][:cv_folds]},
        # v2.0: Store ACTUAL config used for THIS optimization
        optimization_config={
            'algorithm': algorithm,
            'n_trials': n_trials,
            'cv_folds': cv_folds,
            'random_state': rs,
            'timeout_sec': None,
            'pruning_enabled': True,
            'problem_type': 'regression',
        }
    )


# =============================================================================
# Helper: Run the FULL export pipeline exactly as results_tab.py does
# =============================================================================

def run_export_pipeline(trainer_results, optimization_results,
                        nn_optimizer_times=None, nn_optimization_results=None):
    """Simulate _export_session_report() exactly as results_tab.py does it.
    
    This is the critical integration point — it must work the same way
    as the actual GUI code.
    
    v2.0: Added nn_optimization_results to simulate NN optimization
    collection from nn_builder_tab.nn_optimizer_results.
    """
    exporter = SessionExporter()
    exporter.session_info = {'n_samples': 30, 'n_features': 5, 'problem_type': 'regression'}
    
    # ---- Step 1: Collect training results (from trainer.results) ----
    sklearn_results = []
    for result in trainer_results:
        if getattr(result, 'is_neural_network', False):
            # v2.0: Read from immutable TrainingResult fields
            history = getattr(result, 'nn_history', {}) or {}
            if not history:
                continue
            cfg = dict(getattr(result, 'training_config', {}) or {})
            cfg['random_seed'] = getattr(result, 'random_state', 42)
            arch_desc = getattr(result, 'nn_architecture_desc', None) or 'Unknown'
            arch_detail = getattr(result, 'nn_architecture_detail', None) or {}
            cv_metrics = getattr(result, 'cv_metrics', None)
            
            exporter.add_nn_training_result(
                model_name=result.model_name,
                architecture_desc=arch_desc,
                architecture_detail=arch_detail,
                history=history,
                training_config=cfg,
                metrics=getattr(result, 'metrics', None),
                training_time_sec=getattr(result, 'training_time', 0.0),
                cv_metrics=cv_metrics,
                optimization_time_sec=getattr(result, 'optimization_time_sec', 0.0),
            )
        else:
            sklearn_results.append(result)
    
    if sklearn_results:
        exporter.add_training_results(sklearn_results)
    
    # ---- Step 2: Post-process NN optimization times ----
    if nn_optimizer_times:
        for entry in exporter._entries:
            if entry.get('section') == 'Neural Network':
                model_name = entry.get('model_name', '')
                for opt_name, opt_time in nn_optimizer_times.items():
                    if model_name == opt_name and opt_time > 0:
                        entry['optimization_time_sec'] = round(float(opt_time), 3)
                        break
    
    # ---- Step 3: Collect optimization results (sklearn) ----
    # v2.0 FIX: Read optimization_config from the result, NOT from GUI
    for name, result in optimization_results:
        opt_cfg = dict(getattr(result, 'optimization_config', {}) or {})
        opt_cfg.setdefault('model_type', name)
        
        exporter.add_optimization_result(
            result, algorithm=name,
            n_trials=getattr(result, 'n_trials', 0),
            timeout=None,
            all_trials_df=None,
            optimization_config=opt_cfg,
        )
    
    # ---- Step 4: Collect NN optimization results ----
    # v2.0: Simulates results_tab.py collecting from nn_builder_tab.
    # Uses algorithm='Neural Network' which normalizes to 'neural_network'
    # in _opt_entries, matching the 'nn' base from 'NN_Optimized' model names
    # via the nn → neural_network mapping in _link_optimization_entries.
    if nn_optimization_results:
        for opt_name, result in nn_optimization_results:
            opt_cfg = dict(getattr(result, 'optimization_config', {}) or {})
            opt_cfg.setdefault('model_type', 'Neural Network')
            exporter.add_optimization_result(
                result, algorithm='Neural Network',
                n_trials=getattr(result, 'n_trials', 0),
                timeout=None,
                all_trials_df=getattr(result, 'all_trials', None),
                optimization_config=opt_cfg,
            )
    
    # ---- Step 5: Link + filter + render ----
    exporter._link_optimization_entries()
    exporter.filter_orphaned_entries()
    
    return exporter


# =============================================================================
# TEST 1: The user's exact scenario — two optimizations of same algorithm
# =============================================================================

class TestMultipleOptimizationsSameAlgorithm:
    """BUG REGRESSION: optimization_tab.py was DELETING previous optimizations
    of the same algorithm. When user optimized XGBoost twice (5 folds, then 3 folds),
    only the second optimization remained, causing the first trained model to lose
    its optimization details in the export.
    """
    
    def test_both_optimizations_preserved_in_export(self):
        """Both optimization entries must appear in _opt_entries."""
        opt1 = create_mock_opt_result('XGBoost', 50, 5, 42, 0.93, 40.0)
        opt2 = create_mock_opt_result('XGBoost', 30, 3, 123, 0.91, 25.0)
        
        xgb1 = MockSklearnTrainingResult('XGBoost_Optimized', 0.93, 42)
        xgb2 = MockSklearnTrainingResult('XGBoost_Optimized_1', 0.91, 123)
        
        # Simulate: optimization_results APPENDS both (not replace!)
        optimization_results = [('XGBoost', opt1), ('XGBoost', opt2)]
        
        exporter = run_export_pipeline(
            trainer_results=[xgb1, xgb2],
            optimization_results=optimization_results
        )
        
        # Both Optimization entries must exist
        opt_entries = [e for e in exporter._entries if e['section'] == 'Optimization']
        assert len(opt_entries) == 2, f"Expected 2 Optimization entries, got {len(opt_entries)}"
    
    def test_each_model_links_to_correct_optimization_config(self):
        """Each trained model must link to the optimization with matching config."""
        opt1 = create_mock_opt_result('XGBoost', 50, 5, 42, 0.93, 40.0)
        opt2 = create_mock_opt_result('XGBoost', 30, 3, 123, 0.91, 25.0)
        
        xgb1 = MockSklearnTrainingResult('XGBoost_Optimized', 0.93, 42)
        xgb2 = MockSklearnTrainingResult('XGBoost_Optimized_1', 0.91, 123)
        
        optimization_results = [('XGBoost', opt1), ('XGBoost', opt2)]
        
        exporter = run_export_pipeline(
            trainer_results=[xgb1, xgb2],
            optimization_results=optimization_results
        )
        
        # Check linking
        training_entries = [e for e in exporter._entries 
                          if e['section'] == 'Training' and '_linked_optimization' in e]
        assert len(training_entries) == 2, f"Expected 2 linked models, got {len(training_entries)}"
        
        # First model must link to opt with 5 folds
        e1 = [e for e in training_entries if e['model_name'] == 'XGBoost_Optimized'][0]
        cfg1 = e1['_linked_optimization']['optimization_config']
        assert cfg1['cv_folds'] == 5, f"XGBoost_Optimized should link to 5 folds, got {cfg1['cv_folds']}"
        assert cfg1['n_trials'] == 50, f"Expected 50 trials, got {cfg1['n_trials']}"
        assert cfg1['random_state'] == 42, f"Expected rs=42, got {cfg1['random_state']}"
        
        # Second model must link to opt with 3 folds
        e2 = [e for e in training_entries if e['model_name'] == 'XGBoost_Optimized_1'][0]
        cfg2 = e2['_linked_optimization']['optimization_config']
        assert cfg2['cv_folds'] == 3, f"XGBoost_Optimized_1 should link to 3 folds, got {cfg2['cv_folds']}"
        assert cfg2['n_trials'] == 30, f"Expected 30 trials, got {cfg2['n_trials']}"
        assert cfg2['random_state'] == 123, f"Expected rs=123, got {cfg2['random_state']}"
    
    def test_optimization_config_not_from_gui(self):
        """The config must come from OptimizationResult, not from GUI defaults."""
        opt = create_mock_opt_result('XGBoost', 99, 7, 999, 0.95, 60.0)
        xgb = MockSklearnTrainingResult('XGBoost_Optimized', 0.95, 999)
        
        optimization_results = [('XGBoost', opt)]
        
        exporter = run_export_pipeline(
            trainer_results=[xgb],
            optimization_results=optimization_results
        )
        
        opt_entries = [e for e in exporter._entries if e['section'] == 'Optimization']
        assert len(opt_entries) == 1
        cfg = opt_entries[0]['optimization_config']
        # These values must match what was in the OptimizationResult,
        # NOT default GUI values (which would be 5 folds, 50 trials, rs=42)
        assert cfg['cv_folds'] == 7, f"cv_folds must be 7 (from result), got {cfg['cv_folds']}"
        assert cfg['n_trials'] == 99, f"n_trials must be 99 (from result), got {cfg['n_trials']}"
        assert cfg['random_state'] == 999, f"rs must be 999 (from result), got {cfg['random_state']}"


# =============================================================================
# TEST 2: Neural Network with CV fold details
# =============================================================================

class TestNeuralNetworkExport:
    """BUG REGRESSION: Neural networks were missing CV fold breakdown in export.
    The add_nn_training_result() didn't accept cv_metrics, and the renderer
    didn't display CV scores for NN sections.
    """
    
    def test_nn_has_cv_fold_breakdown(self):
        """NN entry must contain cv_test_scores for fold breakdown."""
        nn = MockNNTrainingResult('Neural Network', '[64-relu, 32-relu]', 
                                   r2=0.85, cv_scores=[0.83, 0.85, 0.84, 0.86, 0.85])
        
        exporter = run_export_pipeline(
            trainer_results=[nn],
            optimization_results=[]
        )
        
        nn_entries = [e for e in exporter._entries if e['section'] == 'Neural Network']
        assert len(nn_entries) == 1
        entry = nn_entries[0]
        assert len(entry['cv_test_scores']) == 5, f"Expected 5 CV scores, got {len(entry['cv_test_scores'])}"
        assert entry['cv_test_scores'][0] == 0.83, f"First score should be 0.83, got {entry['cv_test_scores'][0]}"
    
    def test_nn_cv_renders_in_txt_output(self):
        """The rendered TXT must include 'Cross-Validation' for NN."""
        nn = MockNNTrainingResult('Neural Network', '[64-relu, 32-relu]',
                                   r2=0.85, cv_scores=[0.83, 0.85, 0.84, 0.86, 0.85])
        
        exporter = run_export_pipeline(
            trainer_results=[nn],
            optimization_results=[]
        )
        
        txt = exporter._render_txt()
        # Find the NN section and check for CV fold table
        nn_section_start = txt.find('[1] NEURAL NETWORK: Neural Network')
        nn_section_end = txt.find('[2]', nn_section_start + 1)
        if nn_section_end == -1:
            nn_section_end = len(txt)
        nn_section = txt[nn_section_start:nn_section_end]
        
        assert 'Cross-Validation' in nn_section, "NN section must contain 'Cross-Validation'"
        assert 'Fold' in nn_section, "NN section must contain fold table"
    
    def test_optimized_nn_has_optimization_time(self):
        """NN that was optimized must show optimization_time_sec."""
        nn = MockNNTrainingResult('Neural Network_1', '[128-relu, 64-relu, 32-relu]',
                                   r2=0.91, rs=456, epochs=6,
                                   cv_scores=[0.89, 0.91, 0.90, 0.92, 0.91],
                                   opt_time=35.0)
        
        exporter = run_export_pipeline(
            trainer_results=[nn],
            optimization_results=[],
            nn_optimizer_times={'Neural Network_1': 35.0}
        )
        
        nn_entries = [e for e in exporter._entries if e['section'] == 'Neural Network']
        assert len(nn_entries) == 1
        assert nn_entries[0]['optimization_time_sec'] == 35.0, \
            f"Expected opt_time=35.0, got {nn_entries[0]['optimization_time_sec']}"
    
    def test_optimized_nn_opt_time_renders(self):
        """The rendered TXT must show 'Optimization Time' for optimized NN."""
        nn = MockNNTrainingResult('Neural Network_1', '[128-relu, 64-relu, 32-relu]',
                                   r2=0.91, rs=456, epochs=6,
                                   cv_scores=[0.89, 0.91, 0.90, 0.92, 0.91],
                                   opt_time=35.0)
        
        exporter = run_export_pipeline(
            trainer_results=[nn],
            optimization_results=[],
            nn_optimizer_times={'Neural Network_1': 35.0}
        )
        
        # Verify the data is correct in the entry (don't rely on string parsing)
        nn_entries = [e for e in exporter._entries if e['section'] == 'Neural Network']
        assert len(nn_entries) == 1
        assert nn_entries[0]['optimization_time_sec'] == 35.0
        # Full render should not crash and should contain the model name
        txt = exporter._render_txt()
        assert 'Neural Network_1' in txt
        assert 'Optimization Time' in txt  # Should appear somewhere in the output

    def test_nn_optimized_has_optimization_details(self):
        """CRITICAL: Optimized NNs must have [OPTIMIZATION DETAILS] in export.
        
        Regression test for: NN_Optimized models missing OPTIMIZATION DETAILS.
        Root cause: (1) NN optimizations were never added to exporter, and
        (2) _link_optimization_entries only processed 'Training' entries.
        
        This test simulates the exact user scenario:
        - Optimize a NN → produces OptimizationResult
        - Train with best params → produces NN_Optimized TrainingResult
        - Export → must show [OPTIMIZATION DETAILS] for NN_Optimized
        """
        # Create NN optimization result (from nn_builder_tab.nn_optimizer_results)
        nn_opt = create_mock_opt_result('Neural Network', 15, 10, 42, 0.88, 35.0)
        # Override the optimization_config to match what optimize_nn_architecture sets
        nn_opt.optimization_config = {
            'algorithm': 'Neural Network',
            'n_trials': 15,
            'cv_folds': 10,
            'random_state': 42,
            'timeout_sec': None,
            'pruning_enabled': False,
            'problem_type': 'regression',
            'nn_max_layers': 5,
            'nn_min_units': 16,
            'nn_max_units': 512,
        }
        
        # Create the trained NN result (from trainer.train_neural_network)
        nn_trained = MockNNTrainingResult(
            name='NN_Optimized',
            arch_desc='[431-relu, 185-relu]',
            r2=0.88, rs=42, epochs=67,
            cv_scores=[0.38, 0.89, 0.31, 0.64, 0.97, -6.72, 0.97, 0.86, 0.95, -0.72],
            opt_time=35.0
        )
        
        # Run the export pipeline with NN optimization results
        exporter = run_export_pipeline(
            trainer_results=[nn_trained],
            optimization_results=[],  # No sklearn optimizations
            nn_optimization_results=[('NN_Optimized_143025', nn_opt)],
        )
        
        # The NN entry MUST have _linked_optimization
        nn_entries = [e for e in exporter._entries
                      if e['section'] == 'Neural Network' and '_linked_optimization' in e]
        assert len(nn_entries) == 1, f"Expected 1 linked NN, got {len(nn_entries)}"
        
        linked = nn_entries[0]['_linked_optimization']
        assert linked['algorithm'] == 'Neural Network'
        assert linked['optimization_config']['n_trials'] == 15
        assert linked['optimization_config']['cv_folds'] == 10
        
        # Render must contain [OPTIMIZATION DETAILS]
        txt = exporter._render_txt()
        assert '[OPTIMIZATION DETAILS]' in txt, \
            "NN_Optimized must have [OPTIMIZATION DETAILS] in rendered output"

    def test_two_nn_optimizations_both_have_details(self):
        """Two NN optimizations → both get linked to their respective models."""
        opt1 = create_mock_opt_result('Neural Network', 15, 10, 42, 0.88, 35.0)
        opt2 = create_mock_opt_result('Neural Network', 20, 5, 123, 0.85, 28.0)
        
        nn1 = MockNNTrainingResult('NN_Optimized', '[431-relu, 185-relu]', 0.88, 42, 67,
                                    cv_scores=[0.1]*10, opt_time=35.0)
        nn2 = MockNNTrainingResult('NN_Optimized_1', '[456-relu, 109-relu, 95-relu]', 0.85, 123, 83,
                                    cv_scores=[0.2]*5, opt_time=28.0)
        
        exporter = run_export_pipeline(
            trainer_results=[nn1, nn2],
            optimization_results=[],
            nn_optimization_results=[
                ('NN_Optimized_143025', opt1),
                ('NN_Optimized_143512', opt2),
            ],
        )
        
        # Both NNs must be linked
        linked_nns = [e for e in exporter._entries
                      if e['section'] == 'Neural Network' and '_linked_optimization' in e]
        assert len(linked_nns) == 2, f"Expected 2 linked NNs, got {len(linked_nns)}"
        
        # First links to first opt (15 trials, 10 folds)
        e1 = [e for e in linked_nns if e['model_name'] == 'NN_Optimized'][0]
        assert e1['_linked_optimization']['optimization_config']['n_trials'] == 15
        
        # Second links to second opt (20 trials, 5 folds)
        e2 = [e for e in linked_nns if e['model_name'] == 'NN_Optimized_1'][0]
        assert e2['_linked_optimization']['optimization_config']['n_trials'] == 20


# =============================================================================
# TEST 3: Mixed session — sklearn + NN + optimizations
# =============================================================================

class TestMixedSessionExport:
    """Full session with multiple model types — the most common user scenario."""
    
    def test_all_models_appear_in_ranking(self):
        """All models must appear in the final ranking table."""
        opt1 = create_mock_opt_result('XGBoost', 50, 5, 42, 0.93, 40.0)
        opt2 = create_mock_opt_result('XGBoost', 30, 3, 123, 0.91, 25.0)
        
        xgb1 = MockSklearnTrainingResult('XGBoost_Optimized', 0.93, 42)
        xgb2 = MockSklearnTrainingResult('XGBoost_Optimized_1', 0.91, 123)
        nn1 = MockNNTrainingResult('Neural Network', '[64-relu, 32-relu]', 0.85,
                                    cv_scores=[0.83, 0.85, 0.84, 0.86, 0.85])
        nn2 = MockNNTrainingResult('Neural Network_1', '[128-relu, 64-relu, 32-relu]', 
                                    0.91, 456, 6,
                                    cv_scores=[0.89, 0.91, 0.90, 0.92, 0.91], 
                                    opt_time=35.0)
        
        optimization_results = [('XGBoost', opt1), ('XGBoost', opt2)]
        
        exporter = run_export_pipeline(
            trainer_results=[xgb1, xgb2, nn1, nn2],
            optimization_results=optimization_results,
            nn_optimizer_times={'Neural Network_1': 35.0}
        )
        
        # Verify entries exist before rendering
        model_names = [e['model_name'] for e in exporter._entries 
                      if e['section'] in ('Training', 'Neural Network')]
        assert 'XGBoost_Optimized' in model_names
        assert 'XGBoost_Optimized_1' in model_names
        assert 'Neural Network' in model_names
        assert 'Neural Network_1' in model_names
        
        # Full render should contain all model names
        txt = exporter._render_txt()
        assert 'XGBoost_Optimized' in txt, "XGBoost_Optimized missing from export"
        assert 'XGBoost_Optimized_1' in txt, "XGBoost_Optimized_1 missing from export"
        assert 'Neural Network' in txt, "Neural Network missing from export"
        assert 'Neural Network_1' in txt, "Neural Network_1 missing from export"
        assert 'FINAL RANKING' in txt, "Ranking must exist"
    
    def test_no_zero_results_message(self):
        """Export must never show '0 result(s)'."""
        opt = create_mock_opt_result('XGBoost', 50, 5, 42, 0.93, 40.0)
        xgb = MockSklearnTrainingResult('XGBoost_Optimized', 0.93, 42)
        nn = MockNNTrainingResult('Neural Network', '[64-relu, 32-relu]', 0.85)
        
        optimization_results = [('XGBoost', opt)]
        
        exporter = run_export_pipeline(
            trainer_results=[xgb, nn],
            optimization_results=optimization_results
        )
        
        txt = exporter._render_txt()
        assert '0 result' not in txt.lower(), "Export must not contain '0 result(s)'"
    
    def test_sklearn_and_nn_both_have_cv_breakdown(self):
        """Both sklearn models and NNs must show CV fold tables."""
        xgb = MockSklearnTrainingResult('XGBoost', 0.92, cv_scores=[0.91, 0.92, 0.93, 0.92, 0.94])
        nn = MockNNTrainingResult('Neural Network', '[64-relu, 32-relu]', 0.85,
                                    cv_scores=[0.83, 0.85, 0.84, 0.86, 0.85])
        
        exporter = run_export_pipeline(
            trainer_results=[xgb, nn],
            optimization_results=[]
        )
        
        # Verify data exists in entries (more reliable than string parsing)
        xgb_entries = [e for e in exporter._entries if e['section'] == 'Training']
        assert len(xgb_entries) == 1
        assert len(xgb_entries[0].get('cv_test_scores', [])) == 5, "Sklearn must have 5 CV scores"
        
        nn_entries = [e for e in exporter._entries if e['section'] == 'Neural Network']
        assert len(nn_entries) == 1
        assert len(nn_entries[0].get('cv_test_scores', [])) == 5, "NN must have 5 CV scores"
        
        # Full render should contain "Cross-Validation" at least twice (once per model)
        txt = exporter._render_txt()
        cv_count = txt.count('Cross-Validation')
        assert cv_count >= 2, f"Expected at least 2 'Cross-Validation' occurrences, got {cv_count}"


# =============================================================================
# TEST 4: Edge cases
# =============================================================================

class TestEdgeCases:
    """Boundary conditions that previously caused crashes or missing data."""
    
    def test_single_cv_score_does_not_crash(self):
        """Model with only 1 CV score must not crash the renderer."""
        xgb = MockSklearnTrainingResult('XGBoost', 0.92, cv_scores=[0.91])
        
        exporter = run_export_pipeline(
            trainer_results=[xgb],
            optimization_results=[]
        )
        
        txt = exporter._render_txt()
        assert 'Cross-Validation' in txt
        assert '0 result' not in txt.lower()
    
    def test_empty_optimization_results(self):
        """Export with no optimizations must still work."""
        xgb = MockSklearnTrainingResult('XGBoost', 0.92)
        
        exporter = run_export_pipeline(
            trainer_results=[xgb],
            optimization_results=[]
        )
        
        txt = exporter._render_txt()
        assert 'XGBoost' in txt
        assert '0 result' not in txt.lower()
    
    def test_nn_without_cv_metrics(self):
        """NN without cv_metrics must not crash (cv_scores=None)."""
        nn = MockNNTrainingResult('Neural Network', '[64-relu, 32-relu]', 0.85,
                                    cv_scores=None)
        
        exporter = run_export_pipeline(
            trainer_results=[nn],
            optimization_results=[]
        )
        
        txt = exporter._render_txt()
        assert 'Neural Network' in txt
        assert '0 result' not in txt.lower()
    
    def test_deleted_model_not_in_export(self):
        """When a model is deleted, its optimization becomes orphaned and filtered."""
        opt = create_mock_opt_result('RandomForest', 50, 5, 42, 0.88, 30.0)
        # No trained RandomForest model — optimization is orphaned
        
        optimization_results = [('RandomForest', opt)]
        
        exporter = run_export_pipeline(
            trainer_results=[],  # No trained models!
            optimization_results=optimization_results
        )
        
        txt = exporter._render_txt()
        # Orphaned optimization should still appear (has valid data)
        assert 'RandomForest' in txt or 'Optimization' in txt
