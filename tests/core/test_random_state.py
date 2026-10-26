"""Tests for Random State (RS) honoring across AutoAIM.

v2.0: The user-facing Random State must truly control the model's randomness,
the optimizer's search sequence, AND the CV fold shuffling. Covers:
  - create_estimator: same RS → identical predictions (stochastic models)
  - deterministic models (SVR, k-NN) never receive a random_state
  - CatBoost uses random_seed (not random_state) — regression for a silent bug
  - Grid Search optimizer: same RS → identical best params
  - CV fold shuffling follows cv_config.random_state
  - Neural Networks: same seed → identical predictions (weight init + shuffle)
"""

import numpy as np
import pytest

from app.core.model_registry import ModelRegistry
from app.core.optimizer import GridSearchOptimizer, _apply_random_state
from app.core.cv_factory import CVConfig, build_cv_plan

try:
    import optuna  # noqa: F401
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import catboost  # noqa: F401
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


def make_data(n=60, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 4)
    y = X[:, 0] * 5 + rng.rand(n) * 0.2
    return X, y


# ── _apply_random_state naming ──────────────────────────────────────────────

class TestApplyRandomState:
    def test_catboost_uses_random_seed(self):
        out = _apply_random_state({}, 'CatBoost', 99)
        assert out == {'random_seed': 99}
        assert 'random_state' not in out

    def test_tree_models_use_random_state(self):
        for algo in ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']:
            out = _apply_random_state({}, algo, 99)
            assert out == {'random_state': 99}

    def test_deterministic_models_get_no_seed(self):
        for algo in ['Support Vector Machine', 'k-Nearest Neighbors']:
            out = _apply_random_state({}, algo, 99)
            assert 'random_state' not in out and 'random_seed' not in out

    def test_does_not_mutate_input(self):
        src = {'n_estimators': 10}
        _apply_random_state(src, 'CatBoost', 5)
        assert 'random_seed' not in src  # original untouched


# ── create_estimator RS honoring ────────────────────────────────────────────

class TestCreateEstimatorRS:
    def setup_method(self):
        self.reg = ModelRegistry()
        self.X, self.y = make_data()

    @pytest.mark.parametrize("key", ['random_forest', 'gradient_boosting',
                                     'hist_gradient_boosting'])
    def test_same_rs_identical(self, key):
        m1 = self.reg.create_estimator(key, 'regression', random_state=42)
        m1.fit(self.X, self.y)
        m2 = self.reg.create_estimator(key, 'regression', random_state=42)
        m2.fit(self.X, self.y)
        assert np.allclose(m1.predict(self.X), m2.predict(self.X))

    def test_random_forest_diff_rs_differs(self):
        m1 = self.reg.create_estimator('random_forest', 'regression', random_state=42)
        m1.fit(self.X, self.y)
        m2 = self.reg.create_estimator('random_forest', 'regression', random_state=7)
        m2.fit(self.X, self.y)
        assert not np.allclose(m1.predict(self.X), m2.predict(self.X))

    def test_rs_param_actually_set(self):
        m = self.reg.create_estimator('random_forest', 'regression', random_state=77)
        assert m.get_params().get('random_state') == 77

    def test_deterministic_model_excludes_rs(self):
        m = self.reg.create_estimator('svr', 'regression', random_state=42)
        assert 'random_state' not in m.get_params()

    @pytest.mark.skipif(not HAS_CATBOOST, reason="catboost not available")
    def test_catboost_uses_random_seed(self):
        m = self.reg.create_estimator('catboost', 'regression', random_state=42)
        params = m.get_params()
        assert params.get('random_seed') == 42


# ── Optimizer RS honoring ───────────────────────────────────────────────────

class TestOptimizerRS:
    def setup_method(self):
        self.X, self.y = make_data()
        from sklearn.model_selection import train_test_split
        self.Xtr, self.Xte, self.ytr, self.yte = train_test_split(
            self.X, self.y, test_size=0.2, random_state=0)

    def test_gridsearch_same_rs_identical(self):
        gs = GridSearchOptimizer(resolution='coarse')
        r1 = gs.optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            'regression', param_ranges={'n_estimators': (10, 30, 'int'),
                                        'max_depth': (2, 6, 'int')},
            cv_folds=3, random_state=42)
        gs2 = GridSearchOptimizer(resolution='coarse')
        r2 = gs2.optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            'regression', param_ranges={'n_estimators': (10, 30, 'int'),
                                        'max_depth': (2, 6, 'int')},
            cv_folds=3, random_state=42)
        assert r1.best_params == r2.best_params

    def test_gridsearch_stores_rs(self):
        gs = GridSearchOptimizer(resolution='coarse')
        r = gs.optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            'regression', param_ranges={'n_estimators': (10, 30, 'int')},
            cv_folds=3, random_state=123)
        assert r.optimization_config.get('random_state') == 123

    @pytest.mark.skipif(not HAS_OPTUNA, reason="optuna not available")
    def test_bayesian_same_rs_identical(self):
        from app.core.optimizer import HyperparameterOptimizer
        o1 = HyperparameterOptimizer().optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            'regression', n_trials=5, param_ranges={'n_estimators': (10, 50, 'int')},
            cv_folds=3, random_state=42, pruning=False)
        o2 = HyperparameterOptimizer().optimize_sklearn_model(
            'sklearn', 'Random Forest', self.Xtr, self.ytr, self.Xte, self.yte,
            'regression', n_trials=5, param_ranges={'n_estimators': (10, 50, 'int')},
            cv_folds=3, random_state=42, pruning=False)
        assert o1.best_params == o2.best_params


# ── CV fold shuffling follows the configured RS ─────────────────────────────

class TestCVFoldRS:
    def setup_method(self):
        self.X, self.y = make_data(n=50)

    def _folds(self, rs):
        plan = build_cv_plan(CVConfig('kfold', n_folds=5, random_state=rs),
                             self.X, self.y, 'regression')
        return [tuple(sorted(te.tolist())) for _, te in plan.splitter.split(self.X)]

    def test_same_rs_same_folds(self):
        assert self._folds(42) == self._folds(42)

    def test_diff_rs_diff_folds(self):
        assert self._folds(42) != self._folds(7)


# ── Neural Network RS honoring ──────────────────────────────────────────────

@pytest.mark.skipif(not HAS_TORCH, reason="torch not available")
class TestNNRS:
    def _build_train(self, seed, Xtr, ytr, Xte, yte, epochs=15):
        from app.core.nn_builder import NeuralNetworkBuilder
        nn = NeuralNetworkBuilder()
        nn.create_architecture(
            input_dim=Xtr.shape[1], output_dim=1,
            hidden_layers_config=[{'n_units': 16, 'activation': 'relu',
                                   'dropout_rate': 0.0, 'use_batch_norm': False,
                                   'use_layer_norm': False}],
            problem_type='regression')
        nn.create_training_config(
            epochs=epochs, batch_size=16, learning_rate=0.01, optimizer='adam',
            scheduler='none', weight_decay=0.0, early_stopping_patience=50,
            random_seed=seed)
        nn.build_model()
        nn.fit(Xtr, ytr, Xte, yte, verbose=False)
        return nn.predict(Xte).flatten()

    def test_nn_same_seed_identical(self):
        X, y = make_data()
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        p1 = self._build_train(42, Xtr, ytr, Xte, yte)
        p2 = self._build_train(42, Xtr, ytr, Xte, yte)
        assert np.allclose(p1, p2)

    def test_nn_diff_seed_differs(self):
        X, y = make_data()
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
        p1 = self._build_train(42, Xtr, ytr, Xte, yte)
        p3 = self._build_train(7, Xtr, ytr, Xte, yte)
        assert not np.allclose(p1, p3)
