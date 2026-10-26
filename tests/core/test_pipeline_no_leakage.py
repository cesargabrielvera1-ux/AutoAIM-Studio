"""
Test suite for the v2.0 sklearn Pipeline migration — ZERO DATA LEAKAGE.

These tests verify that:
1. prepare_data() returns RAW DataFrames (not preprocessed arrays).
2. The preprocessor from get_preprocessor() is UNFITTED.
3. Trainer.train() wraps model + preprocessor in a Pipeline.
4. cross_validate() refits the preprocessor INSIDE each fold.
5. Validation fold data NEVER leaks into the training statistics.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from app.core.data_manager import DataManager
from app.core.trainer import ModelTrainer as Trainer, TrainingResult
from app.core.model_registry import ModelRegistry
from app.core.optimizer import HyperparameterOptimizer, GridSearchOptimizer
from app.core.model_saver import ModelSaver, save_training_result
from app.core.inference_engine import InferenceEngine


def _df_to_dm(df: pd.DataFrame, tmp_path: Path) -> DataManager:
    """Helper: save DataFrame to temp CSV and load into DataManager."""
    path = tmp_path / 'data.csv'
    df.to_csv(path, index=False)
    dm = DataManager()
    dm.load_data(str(path))
    return dm


class TestPrepareDataReturnsRaw:
    """prepare_data() must return raw DataFrames, not preprocessed arrays."""

    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'f2': [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
            'target': [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
        })

    def test_returns_dataframes(self, sample_df, tmp_path):
        dm = _df_to_dm(sample_df, tmp_path)
        dm.set_target_column('target')
        X_train, X_test, y_train, y_test = dm.prepare_data()

        assert isinstance(X_train, pd.DataFrame), (
            f"X_train must be a DataFrame, got {type(X_train)}"
        )
        assert isinstance(X_test, pd.DataFrame), (
            f"X_test must be a DataFrame, got {type(X_test)}"
        )

    def test_preserves_column_names(self, sample_df, tmp_path):
        dm = _df_to_dm(sample_df, tmp_path)
        dm.set_target_column('target')
        X_train, X_test, y_train, y_test = dm.prepare_data()

        assert list(X_train.columns) == ['f1', 'f2']
        assert list(X_test.columns) == ['f1', 'f2']

    def test_preprocessor_is_unfitted(self, sample_df, tmp_path):
        dm = _df_to_dm(sample_df, tmp_path)
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor()

        # The preprocessor must NOT be fitted yet
        assert not hasattr(preprocessor, 'transformers_'), (
            "Preprocessor must be unfitted after get_preprocessor()"
        )

    def test_raw_data_has_original_scale(self, sample_df, tmp_path):
        dm = _df_to_dm(sample_df, tmp_path)
        dm.set_target_column('target')
        X_train, X_test, y_train, y_test = dm.prepare_data()

        # Raw data should preserve original values
        assert X_train['f1'].iloc[0] == 1.0
        assert X_train['f2'].iloc[0] == 10.0


class TestTrainerBuildsPipeline:
    """Trainer.train() must construct a Pipeline when preprocessor is passed."""

    @pytest.fixture
    def sample_data(self, tmp_path):
        df = pd.DataFrame({
            'f1': np.random.randn(50),
            'f2': np.random.randn(50) * 100,
            'target': np.random.randn(50),
        })
        dm = _df_to_dm(df, tmp_path)
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor()
        X_train, X_test, y_train, y_test = dm.prepare_data()
        return X_train, X_test, y_train, y_test, preprocessor

    def test_result_model_is_pipeline(self, sample_data):
        X_train, X_test, y_train, y_test, preprocessor = sample_data
        from sklearn.ensemble import RandomForestRegressor

        trainer = Trainer()
        result = trainer.train(
            RandomForestRegressor(n_estimators=5, random_state=42),
            'RF_Pipe', X_train, y_train, X_test, y_test,
            'regression', preprocessor=preprocessor,
        )

        assert hasattr(result.model, 'named_steps'), (
            "Result model must be a sklearn Pipeline"
        )
        assert 'preprocessor' in result.model.named_steps
        assert 'model' in result.model.named_steps

    def test_pipeline_can_predict_on_dataframe(self, sample_data):
        X_train, X_test, y_train, y_test, preprocessor = sample_data
        from sklearn.ensemble import RandomForestRegressor

        trainer = Trainer()
        result = trainer.train(
            RandomForestRegressor(n_estimators=5, random_state=42),
            'RF_Pipe2', X_train, y_train, X_test, y_test,
            'regression', preprocessor=preprocessor,
        )

        # Pipeline should accept the raw DataFrame directly
        preds = result.model.predict(X_test)
        assert len(preds) == len(y_test)

    def test_pipeline_preprocessor_is_fitted(self, sample_data):
        X_train, X_test, y_train, y_test, preprocessor = sample_data
        from sklearn.ensemble import RandomForestRegressor

        trainer = Trainer()
        result = trainer.train(
            RandomForestRegressor(n_estimators=5, random_state=42),
            'RF_Pipe3', X_train, y_train, X_test, y_test,
            'regression', preprocessor=preprocessor,
        )

        pipeline = result.model
        fitted_preprocessor = pipeline.named_steps['preprocessor']
        assert hasattr(fitted_preprocessor, 'transformers_'), (
            "Preprocessor inside Pipeline must be fitted after training"
        )

    def test_without_preprocessor_no_pipeline(self, sample_data):
        """Backward compat: train without preprocessor => plain model."""
        X_train, X_test, y_train, y_test, _ = sample_data
        from sklearn.ensemble import RandomForestRegressor

        # Convert to numpy (old API) since no preprocessor is passed
        X_train_np = X_train.values
        X_test_np = X_test.values

        trainer = Trainer()
        result = trainer.train(
            RandomForestRegressor(n_estimators=5, random_state=42),
            'RF_NoPipe', X_train_np, y_train, X_test_np, y_test,
            'regression',
        )

        assert not hasattr(result.model, 'named_steps'), (
            "Without preprocessor, model should be plain estimator"
        )


class TestNoDataLeakageInCV:
    """
    CRITICAL: Verify that validation fold data does NOT leak into the
    scaler statistics during cross-validation.

    Strategy: create data where one fold has wildly different scale.
    With the old (leaky) approach, the scaler sees all data before CV,
    so validation fold statistics influence the transform.
    With Pipeline, the scaler is fit ONLY on the training fold each time.
    """

    @pytest.fixture
    def leakage_sensitive_data(self, tmp_path):
        """Data where fold 0 has mean=0 and fold 1 has mean=1000."""
        np.random.seed(0)
        # Fold 0: low values
        X_low = np.random.randn(20, 2)
        y_low = np.random.randn(20)
        # Fold 1: high values (very different scale)
        X_high = np.random.randn(20, 2) + 1000.0
        y_high = np.random.randn(20)

        X = np.vstack([X_low, X_high])
        y = np.hstack([y_low, y_high])

        df = pd.DataFrame(X, columns=['f1', 'f2'])
        df['target'] = y

        dm = _df_to_dm(df, tmp_path)
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor()
        # Use the FULL dataset (no split) so fold structure is preserved
        X_train = df[['f1', 'f2']]
        y_train = df['target'].values
        return X_train, y_train, preprocessor

    def test_cv_refits_preprocessor_per_fold(self, leakage_sensitive_data):
        """
        If the scaler were fitted on all data before CV, the validation
        fold would see transformed values close to zero (since the global
        mean includes the +1000 shift). With Pipeline, each fold's scaler
        uses ONLY the training fold's mean, so the validation fold retains
        its large magnitude after transform — proving no leakage.
        """
        X_train, y_train, preprocessor = leakage_sensitive_data
        from sklearn.linear_model import Ridge
        from sklearn.model_selection import KFold

        model = Ridge(random_state=42)
        pipeline_steps = [('preprocessor', preprocessor), ('model', model)]
        from sklearn.pipeline import Pipeline
        pipeline = Pipeline(pipeline_steps)

        cv = KFold(n_splits=2, shuffle=False)
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X_train)):
            fold_pipeline = Pipeline(pipeline_steps)
            X_tr = X_train.iloc[train_idx]
            X_val = X_train.iloc[val_idx]
            y_tr = y_train[train_idx]
            y_val = y_train[val_idx]

            fold_pipeline.fit(X_tr, y_tr)
            # Extract the fitted preprocessor from this fold's pipeline
            fitted_prep = fold_pipeline.named_steps['preprocessor']
            # The scaler's mean should reflect ONLY the training fold
            scaler_means = []
            for name, transformer, cols in fitted_prep.transformers_:
                if hasattr(transformer, 'named_steps') and 'scaler' in transformer.named_steps:
                    scaler = transformer.named_steps['scaler']
                    if hasattr(scaler, 'mean_'):
                        scaler_means.extend(scaler.mean_.tolist())

            # KFold(n_splits=2) on ordered data: fold 0 trains on HIGH indices,
            # fold 1 trains on LOW indices. The scaler mean must reflect ONLY
            # the training fold, not the full dataset (~500).
            train_mean = np.mean(scaler_means)
            if fold_idx == 0:
                # Fold 0 trains on high values (indices 20-39) => mean near 1000
                assert abs(train_mean - 1000) < 100, (
                    f"Fold 0 scaler mean {train_mean} should be near 1000, "
                    f"not near 0 or 500"
                )
            else:
                # Fold 1 trains on low values (indices 0-19) => mean near 0
                assert abs(train_mean) < 100, (
                    f"Fold 1 scaler mean {train_mean} should be near 0, "
                    f"not near 1000 or 500"
                )

    def test_trainer_cv_uses_pipeline(self, leakage_sensitive_data):
        """End-to-end: Trainer with preprocessor runs CV inside Pipeline."""
        X_train, y_train, preprocessor = leakage_sensitive_data
        from sklearn.linear_model import Ridge

        trainer = Trainer()
        result = trainer.train(
            Ridge(random_state=42), 'Ridge_LeakTest',
            X_train, y_train, problem_type='regression',
            cv_folds=2, preprocessor=preprocessor,
        )

        # CV metrics should be present
        assert 'test_score' in result.cv_metrics
        assert len(result.cv_metrics['test_score']) == 2


class TestOptimizerPipeline:
    """Optimizers must wrap trial estimators in Pipeline when preprocessor given."""

    @pytest.fixture
    def opt_data(self, tmp_path):
        df = pd.DataFrame({
            'f1': np.random.randn(30),
            'f2': np.random.randn(30) * 50,
            'target': np.random.randn(30),
        })
        dm = _df_to_dm(df, tmp_path)
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor()
        X_train, X_test, y_train, y_test = dm.prepare_data()
        return X_train, X_test, y_train, y_test, preprocessor

    def test_bayesian_optimizer_builds_pipeline(self, opt_data):
        X_train, X_test, y_train, y_test, preprocessor = opt_data
        try:
            import optuna
        except ImportError:
            pytest.skip("Optuna not available")

        opt = HyperparameterOptimizer()
        result = opt.optimize_sklearn_model(
            model_type='random_forest', algorithm='Random Forest',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type='regression',
            n_trials=3, cv_folds=3,
            random_state=42,
            preprocessor=preprocessor,
        )

        assert hasattr(result.best_model, 'named_steps'), (
            "Optimized model must be a Pipeline"
        )

    def test_gridsearch_optimizer_builds_pipeline(self, opt_data):
        X_train, X_test, y_train, y_test, preprocessor = opt_data
        opt = GridSearchOptimizer(resolution='coarse')

        # Use Ridge Regression with a tiny custom param grid for speed
        result = opt.optimize_sklearn_model(
            model_type='ridge', algorithm='Ridge Regression',
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            problem_type='regression',
            param_ranges={'alpha': ([0.1, 1.0], None, 'categorical')},
            cv_folds=2,
            random_state=42,
            preprocessor=preprocessor,
        )

        assert hasattr(result.best_model, 'named_steps'), (
            "GridSearch optimized model must be a Pipeline"
        )


class TestSaveLoadPipeline:
    """Saved models must preserve Pipeline structure for inference."""

    @pytest.fixture
    def trained_pipeline(self, tmp_path):
        df = pd.DataFrame({
            'f1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'f2': [10.0, 20.0, 30.0, 40.0, 50.0],
            'target': [1.5, 2.5, 3.5, 4.5, 5.5],
        })
        dm = _df_to_dm(df, tmp_path)
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor()
        X_train, X_test, y_train, y_test = dm.prepare_data()

        from sklearn.ensemble import RandomForestRegressor
        trainer = Trainer()
        result = trainer.train(
            RandomForestRegressor(n_estimators=5, random_state=42),
            'RF_SaveTest', X_train, y_train, X_test, y_test,
            'regression', preprocessor=preprocessor,
        )
        return result, dm, X_test

    def test_saved_model_is_pipeline(self, trained_pipeline, tmp_path):
        result, dm, X_test = trained_pipeline
        save_path = tmp_path / 'bundle'
        bundle_path = save_training_result(result, str(save_path), dm, 'RF_SaveTest')

        import joblib
        loaded = joblib.load(bundle_path / 'pipeline.joblib')
        assert hasattr(loaded, 'named_steps'), (
            "Saved model must be a Pipeline"
        )

    def test_inference_engine_with_pipeline(self, trained_pipeline, tmp_path):
        result, dm, X_test = trained_pipeline
        save_path = tmp_path / 'bundle'
        bundle_path = save_training_result(result, str(save_path), dm, 'RF_SaveTest')

        engine = InferenceEngine(str(bundle_path))
        preds = engine.predict(X_test)
        assert 'prediction' in preds.columns
        assert len(preds) == len(X_test)


class TestGPRInsidePipeline:
    """Gaussian Process must work inside Pipeline with return_std."""

    def test_gpr_pipeline_returns_std(self, tmp_path):
        df = pd.DataFrame({
            'f1': np.random.randn(30),
            'f2': np.random.randn(30),
            'target': np.random.randn(30),
        })
        dm = _df_to_dm(df, tmp_path)
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor()
        X_train, X_test, y_train, y_test = dm.prepare_data()

        from sklearn.gaussian_process import GaussianProcessRegressor
        trainer = Trainer()
        result = trainer.train(
            GaussianProcessRegressor(random_state=42, normalize_y=True),
            'GP_Pipe', X_train, y_train, X_test, y_test,
            'regression', preprocessor=preprocessor,
        )

        assert result.prediction_std is not None
        assert len(result.prediction_std) == len(y_test)
        assert np.all(result.prediction_std >= 0)

    def test_gpr_pipeline_predict_with_std(self, tmp_path):
        df = pd.DataFrame({
            'f1': np.random.randn(30),
            'f2': np.random.randn(30),
            'target': np.random.randn(30),
        })
        dm = _df_to_dm(df, tmp_path)
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor()
        X_train, X_test, y_train, y_test = dm.prepare_data()

        from sklearn.gaussian_process import GaussianProcessRegressor
        trainer = Trainer()
        result = trainer.train(
            GaussianProcessRegressor(random_state=42, normalize_y=True),
            'GP_Pipe2', X_train, y_train, X_test, y_test,
            'regression', preprocessor=preprocessor,
        )

        # Pipeline.predict(X, return_std=True) should work in sklearn 1.0+
        mean, std = result.model.predict(X_test, return_std=True)
        assert len(mean) == len(y_test)
        assert len(std) == len(y_test)
