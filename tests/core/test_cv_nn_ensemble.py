"""Tests for extending the shared CV strategies to Neural Networks and Ensembles.

v2.0: NN training and Ensemble training/weight-optimization now go through the
shared cv_factory plan, supporting K-Fold, Stratified (with reserved bin),
Group K-Fold, LOO, and LOGO — consistent with sklearn regressors.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from app.core.cv_factory import CVConfig
from app.core.ensemble_trainer import EnsembleTrainer
from app.core.trainer import ModelTrainer

try:
    import torch  # noqa: F401
    from app.core.nn_builder import NeuralNetworkBuilder
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def make_data(n=80, seed=0, spike=False):
    rng = np.random.RandomState(seed)
    if spike:
        y = np.concatenate([np.zeros(n // 4), rng.rand(n - n // 4) * 95 + 5])
        rng.shuffle(y)
    else:
        y = rng.rand(n) * 100
    X = rng.rand(n, 3)
    groups = np.array(['Cu-O', 'Fe-O', 'Ni-O', 'Co-O'] * (n // 4))[:n]
    return X, y, groups


def make_trainer_with_bases(Xtr, ytr, Xte, yte):
    trainer = ModelTrainer()
    trainer.train(RandomForestRegressor(n_estimators=8, random_state=42), 'RF_A',
                  Xtr, ytr, Xte, yte, 'regression', cv_folds=5)
    trainer.train(RandomForestRegressor(n_estimators=8, random_state=1), 'RF_B',
                  Xtr, ytr, Xte, yte, 'regression', cv_folds=5)
    return trainer


# ── Ensemble training with the new strategies ───────────────────────────────

class TestEnsembleCVStrategies:
    def setup_method(self):
        X, y, groups = make_data()
        self.groups = groups
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            X, y, test_size=0.2, random_state=42)
        self.gtr = groups[:len(self.ytr)]
        self.trainer = make_trainer_with_bases(self.Xtr, self.ytr, self.Xte, self.yte)

    def _train(self, cfg):
        et = EnsembleTrainer(self.trainer)
        return et.train_ensemble('weighted_average', ['RF_A', 'RF_B'],
                                 self.Xtr, self.ytr, self.Xte, self.yte,
                                 cv_config=cfg, cv_groups=self.gtr)

    def test_kfold_default_unchanged(self):
        res = self._train(CVConfig('kfold', n_folds=5, random_state=42))
        assert res.cv_strategy == 'kfold'
        assert len(res.cv_scores) == 5
        assert res.cv_config_dict['strategy'] == 'kfold'

    def test_group_kfold(self):
        cfg = CVConfig('group', n_folds=4, group_source='chemical_system', random_state=42)
        res = self._train(cfg)
        assert res.cv_strategy == 'group'
        assert res.cv_metrics['group_info']['n_groups'] == 4
        assert len(res.cv_metrics['test_score']) == 4
        assert res.cv_metrics.get('fold_test_sizes')

    def test_group_no_leakage(self):
        cfg = CVConfig('group', n_folds=4, group_source='chemical_system',
                       groups=np.asarray(self.gtr), random_state=42)
        from app.core.cv_factory import build_cv_plan
        plan = build_cv_plan(cfg, self.Xtr, self.ytr, 'regression')
        g = np.asarray(self.gtr)
        for tr_idx, te_idx in plan.splitter.split(self.Xtr, groups=plan.split_labels):
            assert set(g[tr_idx]).isdisjoint(set(g[te_idx]))

    def test_logo_uses_oof(self):
        cfg = CVConfig('logo', group_source='chemical_system', random_state=42)
        res = self._train(cfg)
        assert res.cv_strategy == 'logo'
        assert res.cv_metrics['test_score_metric'] == 'mae'
        assert res.cv_metrics.get('r2_global') is not None
        # LOGO: one MAE value per group
        assert len(res.cv_metrics['test_score']) == res.cv_metrics['group_info']['n_groups']

    def test_stratified_with_reserved_bin(self):
        X, y, groups = make_data(spike=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        trainer = make_trainer_with_bases(Xtr, ytr, Xte, yte)
        et = EnsembleTrainer(trainer)
        cfg = CVConfig('stratified', n_folds=4, n_bins=4, special_value=0.0, random_state=42)
        res = et.train_ensemble('weighted_average', ['RF_A', 'RF_B'],
                                Xtr, ytr, Xte, yte, cv_config=cfg)
        assert res.cv_strategy == 'stratified'
        assert res.cv_config_dict['special_value'] == 0.0
        assert any('Reserved a stratum' in d
                   for d in res.cv_metrics.get('cv_diagnostics', []))


# ── Ensemble weight optimization with the new strategies ────────────────────

@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not available")
class TestEnsembleWeightOptCV:
    def test_optimize_weights_accepts_cv_config(self):
        X, y, groups = make_data()
        trainer = make_trainer_with_bases(X, y, None, None)
        et = EnsembleTrainer(trainer)
        cfg = CVConfig('group', n_folds=4, group_source='chemical_system',
                       groups=np.asarray(groups), random_state=42)
        res = et.optimize_weights(X, y, model_names=['RF_A', 'RF_B'],
                                  n_trials=3, cv_config=cfg, cv_groups=groups)
        assert res is not None
        assert res.cv_config_dict['group_source'] == 'chemical_system'


# ── Neural Network training with the new strategies ─────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestNNCVStrategies:
    def _make_nn(self, input_dim):
        nn = NeuralNetworkBuilder()
        nn.create_architecture(
            input_dim=input_dim, output_dim=1,
            hidden_layers_config=[{'n_units': 16, 'activation': 'relu',
                                   'dropout_rate': 0.0, 'use_batch_norm': False,
                                   'use_layer_norm': False}],
            problem_type='regression')
        nn.create_training_config(epochs=10, batch_size=16, learning_rate=0.01,
                                  optimizer='adam', scheduler='none', weight_decay=0.0,
                                  early_stopping_patience=10, random_seed=42)
        nn.build_model()
        return nn

    def setup_method(self):
        self.X, self.y, self.groups = make_data()
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)
        self.gtr = self.groups[:len(self.ytr)]
        self.trainer = ModelTrainer()

    def _train_nn(self, cfg):
        nn = self._make_nn(self.Xtr.shape[1])
        return self.trainer.train_neural_network(
            nn, 'NN', self.Xtr, self.ytr, self.Xte, self.yte, 'regression',
            cv_folds=cfg.n_folds, cv_config=cfg,
            cv_groups=self.gtr if cfg.strategy in ('group', 'logo') else None)

    def test_kfold_default(self):
        res = self._train_nn(CVConfig('kfold', n_folds=4, random_state=42))
        assert res.cv_strategy == 'kfold'
        assert len(res.cv_metrics['test_score']) == 4

    def test_stratified_with_reserved_bin(self):
        X, y, groups = make_data(spike=True)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        nn = self._make_nn(Xtr.shape[1])
        trainer = ModelTrainer()
        cfg = CVConfig('stratified', n_folds=4, n_bins=4, special_value=0.0, random_state=42)
        res = trainer.train_neural_network(nn, 'NN_s', Xtr, ytr, Xte, yte, 'regression',
                                           cv_folds=4, cv_config=cfg)
        assert res.cv_strategy == 'stratified'
        assert res.cv_config_dict['special_value'] == 0.0
        assert any('Reserved a stratum' in d
                   for d in res.cv_metrics.get('cv_diagnostics', []))

    def test_group_kfold(self):
        cfg = CVConfig('group', n_folds=4, group_source='chemical_system', random_state=42)
        res = self._train_nn(cfg)
        assert res.cv_strategy == 'group'
        assert res.cv_metrics['group_info']['n_groups'] == 4
        assert len(res.cv_metrics['test_score']) == 4

    def test_logo_uses_oof(self):
        cfg = CVConfig('logo', group_source='chemical_system', random_state=42)
        res = self._train_nn(cfg)
        assert res.cv_strategy == 'logo'
        assert res.cv_metrics['test_score_metric'] == 'mae'
        assert res.cv_metrics.get('r2_global') is not None

    def test_nn_cv_skipped_when_folds_1(self):
        """cv_folds=1 must not raise (regression: KFold(n_splits=1) is invalid)."""
        nn = self._make_nn(self.Xtr.shape[1])
        trainer = ModelTrainer()
        res = trainer.train_neural_network(nn, 'NN_1fold', self.Xtr, self.ytr,
                                           self.Xte, self.yte, 'regression', cv_folds=1)
        # CV skipped → empty or single-score test_score, but no exception.
        assert res is not None
