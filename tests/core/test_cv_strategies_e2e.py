"""End-to-end tests for the new CV strategies through the trainer and
optimizers, plus the DataManager group-alignment helper.

v2.0: stratified / group / logo through ModelTrainer.train and
GridSearchOptimizer.optimize_sklearn_model (Bayesian tests skip without optuna).
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from app.core.cv_factory import CVConfig
from app.core.data_manager import DataManager
from app.core.grouping import groups_from_chemical_system
from app.core.trainer import ModelTrainer

try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False


def make_df(n=40, seed=0):
    rng = np.random.RandomState(seed)
    formulas = ['YBa2Cu3O7', 'Fe2O3', 'Fe3O4', 'MgB2', 'NbTi',
                'SiO2', 'Al2O3', 'CaTiO3'] * (n // 8)
    df = pd.DataFrame({
        'formula': formulas,
        'feat1': rng.rand(n),
        'feat2': rng.rand(n),
        'target': rng.rand(n),
    })
    return df


def make_trainer_data(df, test_size=0.2, random_state=42):
    dm = DataManager()
    dm._data = df
    dm._target_column = 'target'
    dm._feature_columns = ['feat1', 'feat2']
    dm._is_classification = False
    Xtr, Xte, ytr, yte = dm.prepare_data(test_size=test_size, random_state=random_state)
    return dm, Xtr, Xte, ytr, yte


# ── DataManager group alignment ─────────────────────────────────────────────

class TestGroupAlignment:
    def test_groups_align_with_train_split(self):
        df = make_df()
        dm, Xtr, Xte, ytr, yte = make_trainer_data(df, test_size=0.2, random_state=42)
        labels_full, _ = groups_from_chemical_system(df['formula'])
        gtr = dm.get_train_group_labels(labels_full, test_size=0.2, random_state=42)
        assert len(gtr) == Xtr.shape[0]

    def test_groups_alignment_matches_manual_split(self):
        df = make_df()
        dm, Xtr, _, _, _ = make_trainer_data(df, test_size=0.2, random_state=42)
        labels_full, _ = groups_from_chemical_system(df['formula'])
        gtr = dm.get_train_group_labels(labels_full, test_size=0.2, random_state=42)
        train_idx, _ = train_test_split(np.arange(len(df)), test_size=0.2, random_state=42)
        assert list(gtr) == list(np.asarray(labels_full)[train_idx])

    def test_length_mismatch_raises(self):
        df = make_df()
        dm, _, _, _, _ = make_trainer_data(df)
        with pytest.raises(ValueError, match="does not match"):
            dm.get_train_group_labels(np.array(['A', 'B']), test_size=0.2, random_state=42)


# ── Trainer end-to-end with each strategy ──────────────────────────────────

class TestTrainerStrategies:
    def setup_method(self):
        self.df = make_df()
        self.dm, self.Xtr, self.Xte, self.ytr, self.yte = make_trainer_data(self.df)
        self.labels_full, _ = groups_from_chemical_system(self.df['formula'])
        self.groups_tr = self.dm.get_train_group_labels(
            self.labels_full, test_size=0.2, random_state=42)

    def _train(self, cfg):
        trainer = ModelTrainer()
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        return trainer.train(model, 'RF', self.Xtr, self.ytr, self.Xte, self.yte,
                             'regression', cv_folds=cfg.n_folds, cv_config=cfg)

    def test_stratified(self):
        cfg = CVConfig('stratified', n_folds=5, n_bins=5, random_state=42)
        res = self._train(cfg)
        assert res.cv_strategy == 'stratified'
        assert len(res.cv_metrics['test_score']) == 5
        assert res.cv_metrics.get('fold_test_sizes')
        assert res.cv_config_dict['strategy'] == 'stratified'
        assert res.cv_config_dict['n_bins'] == 5

    def test_group_kfold(self):
        cfg = CVConfig('group', n_folds=5, group_source='chemical_system',
                       groups=self.groups_tr, random_state=42)
        res = self._train(cfg)
        assert res.cv_strategy == 'group'
        assert len(res.cv_metrics['test_score']) == 5
        assert res.cv_metrics['group_info']['n_groups'] >= 3
        assert res.cv_metrics.get('fold_test_sizes')
        assert res.cv_config_dict['group_source'] == 'chemical_system'

    def test_logo_oof(self):
        cfg = CVConfig('logo', group_source='chemical_system',
                       groups=self.groups_tr, random_state=42)
        res = self._train(cfg)
        assert res.cv_strategy == 'logo'
        # LOGO regression uses OOF: per-group MAE + global R².
        assert res.cv_metrics['test_score_metric'] == 'mae'
        assert res.cv_metrics.get('r2_global') is not None
        n_groups = res.cv_metrics['group_info']['n_groups']
        assert len(res.cv_metrics['test_score']) == n_groups

    def test_group_no_leakage(self):
        """No group may appear in both train and test of any fold."""
        cfg = CVConfig('group', n_folds=5, group_source='chemical_system',
                       groups=self.groups_tr, random_state=42)
        from app.core.cv_factory import build_cv_plan
        plan = build_cv_plan(cfg, self.Xtr, self.ytr, 'regression')
        groups = np.asarray(self.groups_tr)
        for train_idx, test_idx in plan.splitter.split(self.Xtr, groups=plan.split_labels):
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert train_groups.isdisjoint(test_groups), "Group leaked across train/test!"


# ── GridSearch optimizer with grouped CV ────────────────────────────────────

class TestGridSearchStrategies:
    def setup_method(self):
        self.df = make_df()
        self.dm, self.Xtr, self.Xte, self.ytr, self.yte = make_trainer_data(self.df)
        self.labels_full, _ = groups_from_chemical_system(self.df['formula'])
        self.groups_tr = self.dm.get_train_group_labels(
            self.labels_full, test_size=0.2, random_state=42)

    def _optimize(self, cfg):
        from app.core.optimizer import GridSearchOptimizer
        gs = GridSearchOptimizer(resolution='coarse')
        return gs.optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            problem_type='regression',
            param_ranges={'n_estimators': (10, 30, 'int')},
            cv_folds=cfg.n_folds, random_state=42, cv_config=cfg)

    def test_group_gridsearch(self):
        cfg = CVConfig('group', n_folds=5, group_source='chemical_system',
                       groups=self.groups_tr, random_state=42)
        res = self._optimize(cfg)
        assert res.cv_strategy == 'group'
        assert res.cv_config_dict['group_source'] == 'chemical_system'
        assert res.cv_metrics['group_info']['n_groups'] >= 3
        assert len(res.cv_metrics['test_score']) == 5

    def test_stratified_gridsearch(self):
        cfg = CVConfig('stratified', n_folds=5, n_bins=5, random_state=42)
        res = self._optimize(cfg)
        assert res.cv_strategy == 'stratified'
        assert len(res.cv_metrics['test_score']) == 5

    def test_logo_gridsearch_uses_mae(self):
        cfg = CVConfig('logo', group_source='chemical_system',
                       groups=self.groups_tr, random_state=42)
        res = self._optimize(cfg)
        assert res.cv_strategy == 'logo'
        # OOF strategies optimize on neg-MAE, so scores are marked as MAE.
        assert res.cv_metrics.get('test_score_metric') == 'mae'


# ── Bayesian optimizer with grouped CV (skips without optuna) ──────────────

@pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not available")
class TestBayesianStrategies:
    def setup_method(self):
        self.df = make_df()
        self.dm, self.Xtr, self.Xte, self.ytr, self.yte = make_trainer_data(self.df)
        self.labels_full, _ = groups_from_chemical_system(self.df['formula'])
        self.groups_tr = self.dm.get_train_group_labels(
            self.labels_full, test_size=0.2, random_state=42)

    def test_group_bayesian(self):
        from app.core.optimizer import HyperparameterOptimizer
        cfg = CVConfig('group', n_folds=5, group_source='chemical_system',
                       groups=self.groups_tr, random_state=42)
        opt = HyperparameterOptimizer()
        res = opt.optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            problem_type='regression', n_trials=3,
            param_ranges={'n_estimators': (10, 30, 'int')},
            cv_folds=5, random_state=42, cv_config=cfg)
        assert res.cv_strategy == 'group'
        assert res.cv_config_dict['group_source'] == 'chemical_system'

    def test_logo_bayesian_uses_mae(self):
        from app.core.optimizer import HyperparameterOptimizer
        cfg = CVConfig('logo', group_source='chemical_system',
                       groups=self.groups_tr, random_state=42)
        opt = HyperparameterOptimizer()
        res = opt.optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            problem_type='regression', n_trials=3,
            param_ranges={'n_estimators': (10, 30, 'int')},
            cv_folds=5, random_state=42, cv_config=cfg)
        assert res.cv_metrics.get('test_score_metric') == 'mae'
