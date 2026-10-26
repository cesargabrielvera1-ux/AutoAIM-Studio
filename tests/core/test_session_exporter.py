"""Tests for session_exporter module — exhaustive scientific completeness audit."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.core.session_exporter import (
    SessionExporter, _safe_dict, _fmt, _extract_random_state,
    _extract_model_params, _collect_all_random_states
)


# ------------------------------------------------------------------
# Fakes that mimic real AutoAIM result objects
# ------------------------------------------------------------------

class FakeModel:
    """A fake sklearn-like model with get_params()."""
    def __init__(self, **params):
        self._params = params
    def get_params(self, deep=True):
        return self._params.copy()


class FakeTrainingResult:
    def __init__(self, model_name, metrics, cv_metrics, problem_type="regression",
                 training_time=1.0, feature_importance=None, model=None,
                 is_neural_network=False, nn_history=None):
        self.model_name = model_name
        self.metrics = metrics
        self.cv_metrics = cv_metrics
        self.problem_type = problem_type
        self.training_time = training_time
        self.feature_importance = feature_importance
        self.model = model
        self.is_neural_network = is_neural_network
        self.nn_history = nn_history
        self.predictions = None


class FakeOptimizationResult:
    def __init__(self, best_params, best_score, n_trials, optimization_time=10.0,
                 all_trials=None, study_name="", cv_metrics=None, problem_type="regression"):
        self.best_params = best_params
        self.best_score = best_score
        self.n_trials = n_trials
        self.optimization_time = optimization_time
        self.all_trials = all_trials
        self.study_name = study_name
        self.cv_metrics = cv_metrics or {}
        self.metrics = {}
        self.problem_type = problem_type
        self.best_model = None


class FakeEnsembleResult:
    def __init__(self, ensemble_type, model_names, weights, metrics,
                 training_time=2.0, cv_scores=None, optimized_weights=None,
                 optimization_trials=0, problem_type='regression'):
        self.model_name = f"Ensemble_{ensemble_type}"
        self.ensemble_type = ensemble_type
        self.model_names = model_names
        self.weights = weights
        self.metrics = metrics
        self.training_time = training_time
        self.cv_scores = cv_scores or []
        self.optimized_weights = optimized_weights
        self.optimization_trials = optimization_trials
        self.problem_type = problem_type
        self.predictions = None
        self.base_models = {}


# ------------------------------------------------------------------
# Helper tests
# ------------------------------------------------------------------

class TestSafeDict:
    def test_none(self): assert _safe_dict(None) == {}
    def test_plain(self): assert _safe_dict({"a": 1}) == {"a": 1}
    def test_top_n(self):
        d = {"a": 10, "b": 5, "c": 20, "d": 1}
        r = _safe_dict(d, top_n=2)
        assert len(r) == 2
        assert "c" in r and "a" in r


class TestFmt:
    def test_none(self): assert _fmt(None) == "N/A"
    def test_float(self): assert "0.500000" in _fmt(0.5)
    def test_string(self): assert _fmt("hello") == "hello"


class TestExtractRandomState:
    def test_none_model(self): assert _extract_random_state(None) == 42  # v2.0: guaranteed default
    def test_random_state(self):
        m = FakeModel(random_state=42)
        assert _extract_random_state(m) == 42
    def test_random_seed(self):
        m = FakeModel(random_seed=99)
        assert _extract_random_state(m) == 99
    def test_no_rs(self):
        m = FakeModel(C=1.0)
        assert _extract_random_state(m) == 42  # v2.0: guaranteed default, never None


class TestExtractModelParams:
    def test_none(self): assert _extract_model_params(None) == {}
    def test_basic(self):
        m = FakeModel(n_estimators=100, max_depth=5, random_state=42)
        p = _extract_model_params(m)
        assert p['n_estimators'] == 100
        assert p['max_depth'] == 5
        assert p['random_state'] == 42


class TestCollectAllRandomStates:
    def test_mixed(self):
        entries = [
            {"section": "Training", "model_name": "RF", "random_state": 42},
            {"section": "NN", "model_name": "NN1", "random_state": None,
             "training_config": {"random_seed": 123}},
        ]
        result = _collect_all_random_states(entries)
        assert "Training/RF" in result and result["Training/RF"] == 42
        assert "NN/NN1" in result and result["NN/NN1"] == 123


# ------------------------------------------------------------------
# SessionExporter tests
# ------------------------------------------------------------------

class TestSessionExporterEmpty:
    def test_empty_txt(self):
        exporter = SessionExporter(session_info={"dataset": "test.csv"})
        txt = exporter._render("txt")
        assert "AutoAIM Studio v2.0.0" in txt
        assert "test.csv" in txt
        assert "REPRODUCIBILITY" in txt
        assert "End of Report" in txt

    def test_empty_json(self):
        exporter = SessionExporter()
        j = json.loads(exporter._render("json"))
        assert j["autoaim_version"] == "2.0.0"
        assert j["entries"] == []


class TestSessionExporterTraining:
    def test_training_with_model_params(self):
        """CRITICAL: Model params (including random_state) must be captured."""
        model = FakeModel(n_estimators=200, max_depth=7, random_state=42, C=None)
        result = FakeTrainingResult(
            model_name="RF_optimized",
            metrics={"r2": 0.9123, "rmse": 0.1234, "mae": 0.0987},
            cv_metrics={"test_score": [0.90, 0.91, 0.92, 0.91, 0.91],
                        "train_score": [0.95, 0.96, 0.95, 0.96, 0.95]},
            model=model,
            feature_importance={"feat_a": 0.5, "feat_b": 0.3, "feat_c": 0.2},
        )
        exporter = SessionExporter()
        exporter.add_training_results([result])

        entry = exporter._entries[0]
        assert entry["random_state"] == 42
        assert entry["model_type"] == "FakeModel"
        assert entry["model_params"]["n_estimators"] == 200
        assert entry["model_params"]["max_depth"] == 7
        assert entry["cv_test_scores"] == [0.9, 0.91, 0.92, 0.91, 0.91]
        assert entry["cv_train_scores"] == [0.95, 0.96, 0.95, 0.96, 0.95]
        assert entry["cv_test_mean"] is not None
        assert entry["cv_n_folds"] == 5

        # Verify feature importance top-N
        assert len(entry["feature_importance"]) <= 15
        assert "feat_a" in entry["feature_importance"]

    def test_training_txt_includes_cv_per_fold(self):
        """CRITICAL: CV must show per-fold train+test scores."""
        result = FakeTrainingResult(
            model_name="SVR",
            metrics={"r2": 0.85},
            cv_metrics={"test_score": [0.83, 0.87, 0.85], "train_score": [0.88, 0.90, 0.89]},
            model=FakeModel(C=1.0, kernel="rbf"),
        )
        exporter = SessionExporter()
        exporter.add_training_results([result])
        txt = exporter._render("txt")
        # Should contain per-fold breakdown
        assert "Fold" in txt
        assert "Train Score" in txt or "train" in txt.lower()
        assert "0.83" in txt
        assert "0.87" in txt

    def test_training_no_model(self):
        """TrainingResult without model should still work."""
        result = FakeTrainingResult(model_name="Ridge", metrics={"r2": 0.7}, cv_metrics={})
        exporter = SessionExporter()
        exporter.add_training_results([result])
        entry = exporter._entries[0]
        assert entry["random_state"] == 42  # v2.0: guaranteed default, never None
        assert entry["model_params"] == {}

    def test_training_classification(self):
        result = FakeTrainingResult(
            model_name="RF_clf", metrics={"accuracy": 0.92, "f1": 0.90},
            cv_metrics={"test_score": [0.90, 0.93, 0.92]},
            problem_type="classification",
        )
        exporter = SessionExporter()
        exporter.add_training_results([result])
        entry = exporter._entries[0]
        assert entry["problem_type"] == "classification"


class TestSessionExporterOptimization:
    def test_optimization_with_all_trials(self):
        """CRITICAL: All individual trials must be recorded."""
        import pandas as pd
        trials_df = pd.DataFrame([
            {"number": 0, "value": 0.80, "params": {"n_estimators": 10}},
            {"number": 1, "value": 0.85, "params": {"n_estimators": 50}},
            {"number": 2, "value": 0.90, "params": {"n_estimators": 200}},
        ])
        opt_result = FakeOptimizationResult(
            best_params={"n_estimators": 200, "max_depth": 7},
            best_score=0.90,
            n_trials=50,
            all_trials=trials_df,
            study_name="xgb_optimization",
            cv_metrics={"test_score": [0.89, 0.90, 0.91]},
        )
        model = FakeModel(random_state=123)
        opt_result.best_model = model

        exporter = SessionExporter()
        exporter.add_optimization_result(opt_result, algorithm="XGBoost", n_trials=50)

        entry = exporter._entries[0]
        assert entry["random_state"] == 123
        assert entry["study_name"] == "xgb_optimization"
        assert entry["n_trials"] == 50
        assert entry["n_trials_recorded"] == 3
        assert len(entry["all_trials"]) == 3
        assert entry["all_trials"][2]["value"] == 0.90
        assert entry["cv_test_scores"] == [0.89, 0.90, 0.91]

    def test_optimization_txt_shows_trials(self):
        trials_df = __import__('pandas').DataFrame([
            {"number": i, "value": 0.5 + i * 0.01} for i in range(10)
        ])
        opt_result = FakeOptimizationResult(
            best_params={"C": 1.0}, best_score=0.95, n_trials=10, all_trials=trials_df,
        )
        exporter = SessionExporter()
        exporter.add_optimization_result(opt_result, algorithm="SVR")
        txt = exporter._render("txt")
        assert "Trial History" in txt
        assert "Score" in txt

    def test_optimization_no_trials(self):
        """Optimization without all_trials should still work."""
        opt_result = FakeOptimizationResult(
            best_params={"C": 1.0}, best_score=0.90, n_trials=10,
        )
        exporter = SessionExporter()
        exporter.add_optimization_result(opt_result, algorithm="SVR")
        entry = exporter._entries[0]
        assert entry["all_trials"] == []
        assert entry["n_trials_recorded"] == 0


class TestSessionExporterNN:
    def test_nn_full_history_preserved(self):
        """CRITICAL: Full epoch history must be preserved, not truncated."""
        history = {
            "train_loss": [0.5 - i * 0.01 for i in range(100)],
            "val_loss": [0.6 - i * 0.008 for i in range(100)],
            "train_metric": [0.5 + i * 0.005 for i in range(100)],
            "val_metric": [0.4 + i * 0.004 for i in range(100)],
        }
        exporter = SessionExporter()
        exporter.add_nn_training_result(
            model_name="NN_Best",
            architecture_desc="[64 (relu), 32 (relu)]",
            architecture_detail={"layer_1": "64 units, relu, dropout=0.1",
                                 "layer_2": "32 units, relu"},
            history=history,
            training_config={"epochs": 100, "learning_rate": 0.001,
                             "batch_size": 32, "optimizer": "adam",
                             "random_seed": 42, "scheduler": "reduce_on_plateau",
                             "early_stopping_patience": 10, "gradient_clip_val": 1.0},
        )
        entry = exporter._entries[0]
        assert entry["epochs_trained"] == 100
        # History must contain ALL epochs, not truncated
        assert len(entry["history"]["train_loss"]) == 100
        assert len(entry["history"]["val_loss"]) == 100
        # Stats computed
        assert entry["history_stats"]["best_epoch"] == 100  # monotonically decreasing
        assert "final_train_loss" in entry["history_stats"]
        assert "mean_gap_train_val" in entry["history_stats"]
        # Architecture detail
        assert entry["architecture_detail"]["layer_1"] == "64 units, relu, dropout=0.1"

    def test_nn_txt_shows_full_history_table(self):
        history = {"train_loss": [0.5, 0.4, 0.3, 0.2, 0.1]}
        exporter = SessionExporter()
        exporter.add_nn_training_result(
            model_name="NN_Small", architecture_desc="[16]",
            history=history,
            training_config={"epochs": 5, "random_seed": 42},
        )
        txt = exporter._render("txt")
        assert "Full Epoch History" in txt
        assert "5 epochs" in txt
        # All 5 epochs should appear (n_epochs <= 25)
        assert "1" in txt and "5" in txt

    def test_nn_training_config_complete(self):
        """CRITICAL: Every training config field must be preserved."""
        cfg = {
            "epochs": 500, "learning_rate": 0.001, "batch_size": 64,
            "optimizer": "adamw", "scheduler": "cosine_annealing",
            "early_stopping_patience": 25, "early_stopping_min_delta": 1e-5,
            "gradient_clip_val": 1.5, "weight_decay": 0.01, "random_seed": 99,
        }
        exporter = SessionExporter()
        exporter.add_nn_training_result(
            model_name="NN_FullConfig", architecture_desc="[128, 64, 32]",
            history={"train_loss": [0.5, 0.3, 0.1]}, training_config=cfg,
        )
        entry = exporter._entries[0]
        for key in cfg:
            assert entry["training_config"][key] == cfg[key]


class TestSessionExporterEnsemble:
    def test_ensemble_complete(self):
        """CRITICAL: Both manual and optimized weights, all base models."""
        result = FakeEnsembleResult(
            ensemble_type="weighted_average",
            model_names=["RF", "XGB", "SVR"],
            weights={"RF": 0.3, "XGB": 0.5, "SVR": 0.2},
            optimized_weights={"RF": 0.25, "XGB": 0.55, "SVR": 0.20},
            metrics={"r2": 0.93, "rmse": 0.10},
            cv_scores=[0.92, 0.93, 0.94],
            optimization_trials=100,
            problem_type="regression",
        )
        exporter = SessionExporter()
        exporter.add_ensemble_result(result)
        entry = exporter._entries[0]
        assert entry["weights"] == {"RF": 0.3, "XGB": 0.5, "SVR": 0.2}
        assert entry["optimized_weights"] == {"RF": 0.25, "XGB": 0.55, "SVR": 0.20}
        assert entry["optimization_trials"] == 100
        assert entry["cv_scores"] == [0.92, 0.93, 0.94]
        assert entry["problem_type"] == "regression"

    def test_ensemble_no_optimized_weights(self):
        result = FakeEnsembleResult(
            ensemble_type="stacking", model_names=["RF"],
            weights={"RF": 1.0}, metrics={"r2": 0.88},
        )
        exporter = SessionExporter()
        exporter.add_ensemble_result(result)
        entry = exporter._entries[0]
        assert entry["optimized_weights"] == {}


class TestSessionExporterRanking:
    def test_ranking_sorted_descending(self):
        exporter = SessionExporter()
        exporter.add_training_results([
            FakeTrainingResult("Bad", {"r2": 0.50}, {}),
            FakeTrainingResult("Good", {"r2": 0.95}, {}),
            FakeTrainingResult("Mid", {"r2": 0.75}, {}),
        ])
        ranking = exporter._build_ranking()
        assert len(ranking) == 3
        assert ranking[0]["primary_metric"] == 0.95
        assert ranking[1]["primary_metric"] == 0.75
        assert ranking[2]["primary_metric"] == 0.50

    def test_ranking_with_accuracy(self):
        exporter = SessionExporter()
        exporter.add_training_results([
            FakeTrainingResult("C1", {"accuracy": 0.88}, {}, problem_type="classification"),
        ])
        ranking = exporter._build_ranking()
        assert ranking[0]["primary_metric"] == 0.88


class TestSessionExporterFileExport:
    def test_export_txt_file(self, tmp_path):
        exporter = SessionExporter(session_info={"dataset": "test.csv"})
        exporter.add_training_results([
            FakeTrainingResult("RF", {"r2": 0.88, "rmse": 0.12},
                               {"test_score": [0.86, 0.90]},
                               model=FakeModel(random_state=42, n_estimators=100)),
        ])
        out = tmp_path / "report.txt"
        written = exporter.export(str(out), "txt")
        assert Path(written).exists()
        content = Path(written).read_text()
        assert "REPRODUCIBILITY" in content
        assert "random_state" in content.lower() or "Random State" in content
        assert "0.880000" in content or "0.88" in content
        assert "REPRODUCIBILITY CHECKLIST" in content

    def test_export_md_file(self, tmp_path):
        exporter = SessionExporter()
        exporter.add_training_results([FakeTrainingResult("RF", {"r2": 0.9}, {})])
        out = tmp_path / "report.md"
        exporter.export(str(out))
        content = Path(out).read_text()
        assert "# AutoAIM Studio v2.0.0" in content
        assert "## [1] Training: RF" in content

    def test_export_csv_file(self, tmp_path):
        exporter = SessionExporter()
        exporter.add_training_results([
            FakeTrainingResult("RF", {"r2": 0.9, "rmse": 0.1}, {},
                               model=FakeModel(random_state=42)),
        ])
        out = tmp_path / "report.csv"
        exporter.export(str(out))
        content = Path(out).read_text()
        assert "RF" in content
        assert "0.9" in content
        assert "42" in content  # random_state column

    def test_export_json_file(self, tmp_path):
        exporter = SessionExporter(session_info={"n_samples": 100})
        exporter.set_preprocessing_config({"imputation": "mean", "scaling": "standard"})
        exporter.set_feature_names(["f1", "f2", "f3"])
        exporter.add_training_results([
            FakeTrainingResult("RF", {"r2": 0.9}, {},
                               model=FakeModel(random_state=42)),
        ])
        out = tmp_path / "report.json"
        exporter.export(str(out))
        data = json.loads(Path(out).read_text())
        assert data["autoaim_version"] == "2.0.0"
        assert data["session_info"]["n_samples"] == 100
        assert data["preprocessing_config"]["imputation"] == "mean"
        assert data["feature_names"] == ["f1", "f2", "f3"]
        assert data["reproducibility_notes"]["all_random_states"]
        assert len(data["entries"]) == 1
        assert len(data["ranking"]) == 1

    def test_export_auto_detect_format(self, tmp_path):
        exporter = SessionExporter()
        exporter.add_training_results([FakeTrainingResult("RF", {"r2": 0.9}, {})])
        out = tmp_path / "report.md"  # extension determines format
        exporter.export(str(out))  # fmt=None → auto-detect
        content = Path(out).read_text()
        assert "# AutoAIM Studio" in content


class TestSessionExporterReproducibilityChecklist:
    def test_checklist_in_txt(self):
        exporter = SessionExporter(session_info={"data_file": "data.csv"})
        exporter.set_preprocessing_config({"scaling": "standard"})
        exporter.add_training_results([
            FakeTrainingResult("RF", {"r2": 0.9}, {},
                               model=FakeModel(random_state=42, n_estimators=100)),
        ])
        exporter.add_nn_training_result(
            "NN1", "[64]", history={"train_loss": [0.5, 0.3]},
            training_config={"random_seed": 123, "epochs": 100},
        )
        txt = exporter._render("txt")
        assert "REPRODUCIBILITY CHECKLIST" in txt
        assert "data.csv" in txt
        assert "random_state=42" in txt
        assert "random_seed=123" in txt
        assert "[ ]" in txt  # checklist boxes


class TestSessionExporterSessionMetadata:
    def test_hardware_info(self):
        exporter = SessionExporter()
        exporter.set_hardware_info("CPU: 8 cores, 16 GB RAM")
        exporter.add_training_results([FakeTrainingResult("RF", {"r2": 0.9}, {})])
        txt = exporter._render("txt")
        assert "CPU: 8 cores" in txt

    def test_feature_names(self):
        exporter = SessionExporter()
        exporter.set_feature_names(["a", "b", "c", "d"])
        exporter.add_training_results([FakeTrainingResult("RF", {"r2": 0.9}, {})])
        txt = exporter._render("txt")
        assert "a, b, c, d" in txt

    def test_many_feature_names_truncated(self):
        exporter = SessionExporter()
        exporter.set_feature_names([f"feat_{i}" for i in range(30)])
        txt = exporter._render("txt")
        assert "feat_0" in txt
        assert "and 10 more" in txt


class TestSessionExporterMixed:
    def test_all_types_together(self):
        """CRITICAL: All entry types in one report with correct ranking."""
        exporter = SessionExporter(session_info={"dataset": "bandgaps.csv", "n_samples": 500})
        exporter.set_preprocessing_config({"imputation": "mean", "scaling": "standard"})
        exporter.set_feature_names(["f1", "f2"])

        # Training
        exporter.add_training_results([
            FakeTrainingResult("RF", {"r2": 0.85, "rmse": 0.12},
                               {"test_score": [0.83, 0.87]},
                               model=FakeModel(random_state=42, n_estimators=100)),
        ])
        # Optimization
        opt = FakeOptimizationResult(
            best_params={"n_estimators": 200}, best_score=0.91, n_trials=50,
        )
        opt.best_model = FakeModel(random_state=99)
        exporter.add_optimization_result(opt, algorithm="XGBoost", n_trials=50)
        # NN
        exporter.add_nn_training_result(
            "NN_1", "[64, 32]",
            history={"train_loss": [0.5, 0.3, 0.1], "val_loss": [0.6, 0.35, 0.15]},
            training_config={"random_seed": 77, "epochs": 100},
        )
        # Ensemble
        exporter.add_ensemble_result(FakeEnsembleResult(
            "weighted_average", ["RF", "XGB"], {"RF": 0.5, "XGB": 0.5},
            {"r2": 0.93, "rmse": 0.08},
        ))

        assert len(exporter._entries) == 4
        txt = exporter._render("txt")
        assert "RF" in txt
        assert "XGBoost" in txt
        assert "NN_1" in txt
        assert "Ensemble" in txt

        ranking = exporter._build_ranking()
        assert len(ranking) == 4
        # Ensemble with 0.93 should be first
        assert ranking[0]["primary_metric"] == 0.93

        # JSON should have all metadata
        json_str = exporter._render("json")
        data = json.loads(json_str)
        assert data["session_info"]["dataset"] == "bandgaps.csv"
        assert data["preprocessing_config"]["imputation"] == "mean"
        assert data["feature_names"] == ["f1", "f2"]


# ------------------------------------------------------------------
# v2.0: Optimization linking tests
# ------------------------------------------------------------------

class TestSessionExporterOptimizationLinking:
    """Test that optimization details are linked inline with trained models."""

    def _create_linked_scenario(self):
        """Create an exporter with a Training entry and matching Optimization entry."""
        exporter = SessionExporter()
        # Training result with optimized parameters
        exporter.add_training_results([
            FakeTrainingResult(
                model_name="XGBoost_Optimized",
                metrics={"r2": 0.95, "rmse": 0.08},
                cv_metrics={"test_score": [0.94, 0.95, 0.96]},
                model=FakeModel(random_state=42, n_estimators=200, max_depth=7),
            ),
        ])
        # Optimization result that produced those params
        opt = FakeOptimizationResult(
            best_params={"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05},
            best_score=0.95,
            n_trials=50,
            optimization_time=15.5,
        )
        opt.best_model = FakeModel(random_state=42)
        exporter.add_optimization_result(
            opt, algorithm="XGBoost", n_trials=50,
            optimization_config={"n_trials": 50, "cv_folds": 5, "random_state": 42, "pruning_enabled": True},
        )
        return exporter

    def test_linking_matches_by_name(self):
        """Training 'XGBoost_Optimized' should link with Optimization 'XGBoost'."""
        exporter = self._create_linked_scenario()
        exporter._link_optimization_entries()

        training_entry = exporter._entries[0]
        assert training_entry["section"] == "Training"
        assert "_linked_optimization" in training_entry
        linked = training_entry["_linked_optimization"]
        assert linked["section"] == "Optimization"
        assert linked["algorithm"] == "XGBoost"
        assert linked["n_trials"] == 50

    def test_no_link_for_non_optimized_model(self):
        """A plain Training model without optimization should not have a link."""
        exporter = SessionExporter()
        exporter.add_training_results([
            FakeTrainingResult("RandomForest", {"r2": 0.88}, {}),
        ])
        exporter.add_optimization_result(
            FakeOptimizationResult({"C": 1.0}, 0.90, 10), algorithm="SVR", n_trials=10,
        )
        exporter._link_optimization_entries()

        training_entry = exporter._entries[0]
        assert training_entry["model_name"] == "RandomForest"
        assert "_linked_optimization" not in training_entry

    def test_txt_renders_optimization_inline(self):
        """The TXT report must include optimization details within the Training entry."""
        exporter = self._create_linked_scenario()
        exporter._link_optimization_entries()  # v2.0: trigger linking before render
        txt = exporter._render("txt")

        assert "[OPTIMIZATION DETAILS" in txt
        assert "Trials           : 50" in txt
        assert "Best Score       : 0.950000" in txt
        assert "n_estimators" in txt
        assert "max_depth" in txt
        # Best params should appear
        assert "200" in txt  # n_estimators value
        assert "7" in txt or " pruning_enabled" in txt  # either params or config

    def test_md_renders_optimization_inline(self):
        """The MD report must include optimization details within the Training entry."""
        exporter = self._create_linked_scenario()
        exporter._link_optimization_entries()  # v2.0: trigger linking before render
        md = exporter._render("md")

        assert "### Optimization Details" in md
        assert "Trials:" in md
        assert "50" in md
        assert "Best Parameters" in md

    def test_linking_preserves_optimization_entry(self):
        """The standalone Optimization entry should still exist in the report."""
        exporter = self._create_linked_scenario()
        exporter._link_optimization_entries()

        sections = [e["section"] for e in exporter._entries]
        assert sections.count("Optimization") == 1
        assert sections.count("Training") == 1

    def test_random_forest_with_space_in_name(self):
        """'Random Forest_Optimized' should link with 'Random Forest'."""
        exporter = SessionExporter()
        exporter.add_training_results([
            FakeTrainingResult(
                model_name="Random Forest_Optimized",
                metrics={"r2": 0.92}, cv_metrics={},
                model=FakeModel(random_state=10),
            ),
        ])
        opt = FakeOptimizationResult({"n_estimators": 100}, 0.92, 30)
        exporter.add_optimization_result(opt, algorithm="Random Forest", n_trials=30)
        exporter._link_optimization_entries()

        training_entry = exporter._entries[0]
        assert "_linked_optimization" in training_entry
        assert training_entry["_linked_optimization"]["algorithm"] == "Random Forest"

    def test_multiple_models_only_links_correct_one(self):
        """With multiple training models, only the matching one gets linked."""
        exporter = SessionExporter()
        exporter.add_training_results([
            FakeTrainingResult("SVR", {"r2": 0.80}, {}),
            FakeTrainingResult("XGBoost_Optimized", {"r2": 0.95}, {}),
        ])
        opt = FakeOptimizationResult({"lr": 0.1}, 0.95, 50)
        exporter.add_optimization_result(opt, algorithm="XGBoost", n_trials=50)
        exporter._link_optimization_entries()

        svr_entry = [e for e in exporter._entries if e["model_name"] == "SVR"][0]
        xgb_entry = [e for e in exporter._entries if e["model_name"] == "XGBoost_Optimized"][0]

        assert "_linked_optimization" not in svr_entry
        assert "_linked_optimization" in xgb_entry

    def test_linking_after_collection_order_gui_real(self):
        """v2.0: Linking works regardless of collection order.

        In the real GUI (_export_session_report), Training is collected
        BEFORE Optimization. The linking must still work because
        _link_optimization_entries() runs AFTER all data is collected,
        using _opt_entries which was populated during add_optimization_result().
        """
        exporter = SessionExporter()

        # 1. Add TRAINING first (GUI order — this is how _export_session_report does it)
        exporter.add_training_results([
            FakeTrainingResult(
                model_name="Gradient Boosting_Optimized",
                metrics={"r2": 0.95, "rmse": 0.08},
                cv_metrics={"test_score": [0.94, 0.95, 0.96]},
                model=FakeModel(random_state=42, n_estimators=200, max_depth=7),
            ),
        ])

        # 2. Add optimization AFTER (GUI order)
        opt = FakeOptimizationResult(
            best_params={"n_estimators": 200, "max_depth": 7, "learning_rate": 0.05},
            best_score=0.95,
            n_trials=50,
            optimization_time=15.5,
        )
        opt.best_model = FakeModel(random_state=42)
        exporter.add_optimization_result(
            opt, algorithm="Gradient Boosting", n_trials=50,
            optimization_config={"n_trials": 50, "cv_folds": 5, "random_state": 42},
        )

        # 3. At this point, Training entry has NO link yet
        training_entry = exporter._entries[0]
        assert training_entry["section"] == "Training"
        assert "_linked_optimization" not in training_entry  # Not linked yet!

        # 4. Call _link_optimization_entries() (this is what export() does)
        exporter._link_optimization_entries()

        # 5. Now it should be linked
        assert "_linked_optimization" in training_entry
        linked = training_entry["_linked_optimization"]
        assert linked["algorithm"] == "Gradient Boosting"
        assert linked["n_trials"] == 50
        assert linked["best_params"]["n_estimators"] == 200

        # 6. Verify inline rendering in TXT
        txt = exporter._render("txt")
        assert "[OPTIMIZATION DETAILS" in txt
        assert "Gradient Boosting" in txt
        assert "Trials           : 50" in txt

    def test_full_user_workflow_simulation(self):
        """v2.0: Simulate the real GUI workflow (Training BEFORE Optimization):
        train 2 optimized models, then collect optimization data.
        Both should show optimization details inline after linking."""
        exporter = SessionExporter(session_info={"dataset": "test.csv", "n_samples": 100})

        # GUI order: Training results collected FIRST
        exporter.add_training_results([
            FakeTrainingResult(
                model_name="Gradient Boosting_Optimized",
                metrics={"r2": 0.736, "rmse": 1.285},
                cv_metrics={"test_score": [0.26, 0.09, 0.57, 0.26, 0.32]},
                model=FakeModel(random_state=42, n_estimators=187, max_depth=8),
                training_time=0.343,
            ),
            FakeTrainingResult(
                model_name="XGBoost_Optimized",
                metrics={"r2": 0.85, "rmse": 0.92},
                cv_metrics={"test_score": [0.83, 0.84, 0.85, 0.86, 0.84]},
                model=FakeModel(random_state=42, n_estimators=300, max_depth=6),
                training_time=0.512,
            ),
        ])

        # GUI order: Optimization results collected AFTER
        opt_gb = FakeOptimizationResult(
            best_params={"n_estimators": 187, "max_depth": 8, "learning_rate": 0.014},
            best_score=0.736, n_trials=50, optimization_time=12.3,
        )
        exporter.add_optimization_result(
            opt_gb, algorithm="Gradient Boosting", n_trials=50,
            optimization_config={"cv_folds": 5, "random_state": 42, "pruning_enabled": True},
        )

        opt_xgb = FakeOptimizationResult(
            best_params={"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1},
            best_score=0.85, n_trials=50, optimization_time=18.7,
        )
        exporter.add_optimization_result(
            opt_xgb, algorithm="XGBoost", n_trials=50,
            optimization_config={"cv_folds": 5, "random_state": 42, "pruning_enabled": True},
        )

        # CRITICAL: _link_optimization_entries() must be called AFTER all collection
        exporter._link_optimization_entries()

        # Both should be linked
        entries = {e["model_name"]: e for e in exporter._entries if e["section"] == "Training"}
        assert "_linked_optimization" in entries["Gradient Boosting_Optimized"]
        assert "_linked_optimization" in entries["XGBoost_Optimized"]

        # Verify inline optimization details for both
        txt = exporter._render("txt")
        opt_detail_count = txt.count("[OPTIMIZATION DETAILS")
        assert opt_detail_count == 2, f"Expected 2 OPTIMIZATION DETAILS sections, got {opt_detail_count}"

        assert "Gradient Boosting" in txt
        assert "XGBoost" in txt
        assert "50" in txt  # trials count

        # v2.0: Linked Optimization entries are NOT shown as standalone sections
        # (they appear inline within their Training entry only)
        opt_count = txt.count("] OPTIMIZATION:")
        assert opt_count == 0, f"Expected 0 standalone OPTIMIZATION entries, got {opt_count}"
