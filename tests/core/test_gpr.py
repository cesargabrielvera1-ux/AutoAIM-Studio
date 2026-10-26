"""Tests for Gaussian Process integration (v2.0).

GaussianProcessRegressor is the first model in AutoAIM with native
uncertainty quantification (return_std). Tests cover model creation,
reproducibility, optimization, prediction with uncertainty, SHAP
fallback to KernelExplainer, and manifest serialization.
"""

import numpy as np
import pytest

from app.core.model_registry import ModelRegistry
from app.core.trainer import ModelTrainer as Trainer, TrainingResult
from app.core.inference_engine import InferenceEngine
from app.core.model_saver import ModelSaver

try:
    from sklearn.gaussian_process import GaussianProcessRegressor
    HAS_GPR = True
except ImportError:
    HAS_GPR = False


def make_data(n=50, seed=0):
    rng = np.random.RandomState(seed)
    X = rng.rand(n, 4)
    y = X[:, 0] * 5 + rng.rand(n) * 0.2
    return X, y


@pytest.mark.skipif(not HAS_GPR, reason="sklearn GaussianProcess not available")
class TestGaussianProcess:
    def setup_method(self):
        self.reg = ModelRegistry()
        self.X, self.y = make_data()

    def test_model_registry_has_gp(self):
        models = self.reg.list_models('regression')
        assert 'gaussian_process' in models

    def test_create_estimator(self):
        gpr = self.reg.create_estimator('gaussian_process', 'regression', random_state=42)
        assert isinstance(gpr, GaussianProcessRegressor)
        assert gpr.get_params()['random_state'] == 42

    def test_param_ranges(self):
        info = self.reg.get_model('gaussian_process')
        assert 'alpha' in info.param_ranges
        assert 'n_restarts_optimizer' in info.param_ranges
        assert 'normalize_y' in info.param_ranges

    def test_reproducibility(self):
        """Same random_state → identical predictions."""
        gpr1 = self.reg.create_estimator('gaussian_process', 'regression', random_state=42)
        gpr1.fit(self.X, self.y)
        p1 = gpr1.predict(self.X)

        gpr2 = self.reg.create_estimator('gaussian_process', 'regression', random_state=42)
        gpr2.fit(self.X, self.y)
        p2 = gpr2.predict(self.X)

        assert np.allclose(p1, p2)

    def test_return_std(self):
        """GPR supports return_std=True after fitting."""
        gpr = self.reg.create_estimator('gaussian_process', 'regression', random_state=42)
        gpr.fit(self.X, self.y)
        mean, std = gpr.predict(self.X, return_std=True)
        assert mean.shape == (len(self.X),)
        assert std.shape == (len(self.X),)
        assert np.all(std >= 0)

    def test_trainer_captures_std(self):
        """Trainer captures prediction_std for GPR."""
        from sklearn.model_selection import train_test_split
        from sklearn.gaussian_process import GaussianProcessRegressor
        Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=0.3, random_state=0)

        trainer = Trainer()
        model = GaussianProcessRegressor(random_state=42, normalize_y=True)
        result = trainer.train(
            model, 'gaussian_process', Xtr, ytr, Xte, yte,
            'regression',
        )
        assert isinstance(result, TrainingResult)
        assert result.prediction_std is not None
        assert len(result.prediction_std) == len(yte)
        assert np.all(result.prediction_std >= 0)

    def test_inference_engine_std(self, tmp_path):
        """InferenceEngine adds prediction_std column for GPR bundles."""
        from sklearn.model_selection import train_test_split
        from sklearn.gaussian_process import GaussianProcessRegressor
        Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=0.3, random_state=0)

        trainer = Trainer()
        model = GaussianProcessRegressor(random_state=42, normalize_y=True)
        result = trainer.train(
            model, 'gaussian_process', Xtr, ytr, Xte, yte,
            'regression',
        )

        # Save bundle
        saver = ModelSaver()
        bundle = saver.save_model(
            model=result.model,
            output_path=tmp_path / "gp_bundle",
            model_name="test_gp",
            model_type="sklearn",
            algorithm="GaussianProcess",
            feature_names=[f"f{i}" for i in range(self.X.shape[1])],
            target_column="target",
            metrics=result.metrics,
        )

        # Load and predict
        engine = InferenceEngine(bundle)
        df = __import__('pandas', fromlist=['DataFrame']).DataFrame(Xte, columns=[f"f{i}" for i in range(self.X.shape[1])])
        pred_df = engine.predict(df)

        assert 'prediction' in pred_df.columns
        assert 'prediction_std' in pred_df.columns
        assert np.all(pred_df['prediction_std'].values >= 0)

    def test_manifest_has_gp_metadata(self, tmp_path):
        """Manifest contains gp_metadata for GPR models."""
        gpr = self.reg.create_estimator('gaussian_process', 'regression', random_state=42)
        gpr.fit(self.X, self.y)

        saver = ModelSaver()
        bundle = saver.save_model(
            model=gpr,
            output_path=tmp_path / "gp_meta",
            model_name="test_gp_meta",
            model_type="sklearn",
            algorithm="GaussianProcess",
            feature_names=[f"f{i}" for i in range(self.X.shape[1])],
            target_column="target",
        )

        import json
        manifest = json.loads((bundle / "manifest.json").read_text())
        gp_meta = manifest.get("gp_metadata")
        assert gp_meta is not None
        assert "kernel_str" in gp_meta
        assert "alpha" in gp_meta

    def test_optimizer_gridsearch(self):
        """GridSearchOptimizer works with GPR."""
        from app.core.optimizer import GridSearchOptimizer
        from sklearn.model_selection import train_test_split
        Xtr, Xte, ytr, yte = train_test_split(self.X, self.y, test_size=0.3, random_state=0)

        gs = GridSearchOptimizer(resolution='coarse')
        r = gs.optimize_sklearn_model(
            'sklearn', 'Gaussian Process', Xtr, ytr, Xte, yte,
            'regression',
            param_ranges={'alpha': (1e-10, 1e-5, 'float')},
            cv_folds=3, random_state=42,
        )
        assert r.best_model is not None
        assert 'alpha' in r.best_params

    def test_shap_kernel_explainer(self):
        """SHAP falls back to KernelExplainer for GPR."""
        try:
            import shap
        except ImportError:
            pytest.skip("shap not available")

        from app.core.explainer import Explainer
        gpr = self.reg.create_estimator('gaussian_process', 'regression', random_state=42)
        gpr.fit(self.X, self.y)

        explainer = Explainer()
        shap_values = explainer.compute_shap_values(
            gpr, self.X[:20], problem_type='regression', background_data=self.X[:10]
        )
        assert shap_values is not None
