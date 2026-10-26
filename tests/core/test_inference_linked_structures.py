"""
Tests for InferenceEngine linked-structure support.

Verifies that models trained with linked crystal/molecule features can be
used for prediction without crashing, by either:
  a) Providing structure files via predict_with_linked_structures(), or
  b) Getting an informative error that explains what to do.

This prevents the "crash asegurado" when users try to predict with models
that were trained on linked features but provide data without them.

v2.0: Unified flow — composition is auto-derived from structure files.
No separate Magpie/composition step. Both crystal and molecule linking
include periodic properties (electronegativity, atomic mass) directly.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.inference_engine import InferenceEngine
from app.core.model_saver import ModelSaver


# =============================================================================
# TEST 1: Detection of linked-structure models
# =============================================================================

class TestLinkedStructureDetection:
    """InferenceEngine correctly detects models that require linked structures."""

    def test_plain_model_not_linked(self, tmp_path):
        """A model without linked features is NOT a linked model."""
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t1",
            model_name="Plain", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'f2'], target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        assert engine.is_structure_linked_model() is False
        reqs = engine.get_linked_structure_requirements()
        assert reqs['requires_linking'] is False
        assert reqs['structure_type'] is None

    def test_crystal_model_detected(self, tmp_path):
        """A model with crystal_* features IS a linked crystal model."""
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t2",
            model_name="Crystal", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'crystal_density', 'crystal_lattice_a'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        assert engine.is_structure_linked_model() is True
        reqs = engine.get_linked_structure_requirements()
        assert reqs['structure_type'] == 'crystal'
        assert reqs['missing_feature_count'] == 2

    def test_molecule_model_detected(self, tmp_path):
        """A model with mol_* features IS a linked molecule model."""
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t3",
            model_name="Molecule", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'mol_n_atoms', 'mol_mass'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        assert engine.is_structure_linked_model() is True
        reqs = engine.get_linked_structure_requirements()
        assert reqs['structure_type'] == 'molecule'
        assert reqs['missing_feature_count'] == 2

    def test_combined_model_detected(self, tmp_path):
        """A model with BOTH crystal_ and mol_ features is detected."""
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t4",
            model_name="Hybrid", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'crystal_density', 'mol_n_atoms'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        assert engine.is_structure_linked_model() is True
        reqs = engine.get_linked_structure_requirements()
        assert 'crystal_' in reqs['linked_prefixes']
        assert 'mol_' in reqs['linked_prefixes']


# =============================================================================
# TEST 2: validate_input with missing linked features
# =============================================================================

class TestValidateInputWithLinkedFeatures:
    """validate_input() provides informative feedback for missing linked features."""

    def test_plain_missing_features_no_linking(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t5",
            model_name="Plain", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'f2'], target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        val = engine.validate_input(pd.DataFrame({'f1': [1.0]}))
        assert val['valid'] is False
        assert val['needs_linking_action'] is False

    def test_crystal_missing_signals_linking_needed(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t6",
            model_name="Crystal", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'crystal_density', 'crystal_lattice_a'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        val = engine.validate_input(pd.DataFrame({'f1': [1.0]}))
        assert val['needs_linking_action'] is True
        assert 'crystal_density' in val['missing_linked_features']

    def test_molecule_missing_signals_linking_needed(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t7",
            model_name="Molecule", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'mol_n_atoms', 'mol_mass'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        val = engine.validate_input(pd.DataFrame({'f1': [1.0]}))
        assert val['needs_linking_action'] is True
        assert 'mol_n_atoms' in val['missing_linked_features']

    def test_mixed_missing_some_linked_some_plain(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t8",
            model_name="Crystal", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'f2', 'crystal_density'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        val = engine.validate_input(pd.DataFrame({'f1': [1.0]}))
        assert val['needs_linking_action'] is False

    def test_all_features_present_is_valid(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t9",
            model_name="Crystal", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'crystal_density'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        val = engine.validate_input(pd.DataFrame({'f1': [1.0], 'crystal_density': [2.5]}))
        assert val['valid'] is True


# =============================================================================
# TEST 3: predict() gives informative error for linked models
# =============================================================================

class TestPredictInformativeError:
    """predict() gives actionable error instead of cryptic crash."""

    def test_predict_crystal_informative_error(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t10",
            model_name="Crystal", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'crystal_density', 'crystal_lattice_a'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        with pytest.raises(ValueError, match="predict_with_linked_structures"):
            engine.predict(pd.DataFrame({'f1': [1.0]}))

    def test_predict_molecule_informative_error(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t11",
            model_name="Molecule", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'mol_n_atoms', 'mol_mass'],
            target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        with pytest.raises(ValueError, match="predict_with_linked_structures"):
            engine.predict(pd.DataFrame({'f1': [1.0]}))


# =============================================================================
# TEST 4: predict_with_linked_structures() — molecule (.xyz)
# =============================================================================

class TestPredictWithLinkedStructuresMolecule:
    """predict_with_linked_structures() with .xyz files."""

    def _write_xyz(self, tmp_path: Path, name: str, n_atoms: int,
                   elements: list, coords: list) -> Path:
        p = tmp_path / f"{name}.xyz"
        lines = [str(n_atoms), name]
        for elem, (x, y, z) in zip(elements, coords):
            lines.append(f"{elem}  {x:.4f}  {y:.4f}  {z:.4f}")
        p.write_text("\n".join(lines))
        return p

    def test_molecule_link_and_predict(self, tmp_path):
        from app.core.molecule_featurizer import MoleculeFeaturizer
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        for name, elems, coords in [
            ("water", ["O", "H", "H"],
             [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            ("co2", ["C", "O", "O"],
             [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
        ]:
            self._write_xyz(xyz_dir, name, len(elems), elems, coords)

        featurizer = MoleculeFeaturizer()
        df_ids = pd.DataFrame({"mol_id": ["water", "co2"]})
        feat_df = featurizer.featurize_files([xyz_dir / "water.xyz", xyz_dir / "co2.xyz"])
        mol_cols = [c for c in feat_df.columns if c != "mol_source_file"]

        feature_names = ["base"] + mol_cols
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        X = np.random.randn(10, len(feature_names))
        y = np.random.randn(10)
        model = RandomForestRegressor(n_estimators=3, random_state=42)
        model.fit(X, y)
        joblib.dump(model, bundle / "pipeline.joblib")
        (bundle / "manifest.json").write_text(json.dumps({
            "model_name": "Test", "model_type": "sklearn", "algorithm": "rf",
            "creation_date": "2024-01-01", "sklearn_version": "1.5.0",
            "feature_names": feature_names, "target_column": "y",
            "metrics": {}, "is_neural_network": False,
            "linked_feature_prefixes": ["mol_"],
            "requires_structure_linking": True,
        }))

        engine = InferenceEngine(bundle)
        result = engine.predict_with_linked_structures(
            pd.DataFrame({"base": [1.0, 2.0], "mol_id": ["water", "co2"]}),
            id_column="mol_id", structure_folder=str(xyz_dir),
        )
        assert "prediction" in result.columns
        assert len(result) == 2

    def test_auto_detect_structure_type(self, tmp_path):
        from app.core.molecule_featurizer import MoleculeFeaturizer
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        self._write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        featurizer = MoleculeFeaturizer()
        feat_df = featurizer.featurize_file(xyz_dir / "water.xyz")
        mol_cols = [c for c in feat_df.columns if c != "mol_source_file"]

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(np.random.randn(5, len(mol_cols) + 1), np.random.randn(5))
        joblib.dump(model, bundle / "pipeline.joblib")
        (bundle / "manifest.json").write_text(json.dumps({
            "model_name": "T", "model_type": "sklearn", "algorithm": "rf",
            "feature_names": ["base"] + mol_cols, "target_column": "y",
            "metrics": {}, "is_neural_network": False,
            "linked_feature_prefixes": ["mol_"],
            "requires_structure_linking": True,
        }))

        engine = InferenceEngine(bundle)
        result = engine.predict_with_linked_structures(
            pd.DataFrame({"base": [1.0], "mol_id": ["water"]}),
            id_column="mol_id", structure_folder=str(xyz_dir),
        )
        assert "prediction" in result.columns

    def test_explicit_structure_type_override(self, tmp_path):
        from app.core.molecule_featurizer import MoleculeFeaturizer
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        self._write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                        [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        featurizer = MoleculeFeaturizer()
        feat_df = featurizer.featurize_file(xyz_dir / "water.xyz")
        mol_cols = [c for c in feat_df.columns if c != "mol_source_file"]

        bundle = tmp_path / "bundle"
        bundle.mkdir()
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(np.random.randn(5, len(mol_cols) + 1), np.random.randn(5))
        joblib.dump(model, bundle / "pipeline.joblib")
        (bundle / "manifest.json").write_text(json.dumps({
            "model_name": "T", "model_type": "sklearn", "algorithm": "rf",
            "feature_names": ["base"] + mol_cols, "target_column": "y",
            "metrics": {}, "is_neural_network": False,
            "linked_feature_prefixes": ["mol_"],
            "requires_structure_linking": True,
        }))

        engine = InferenceEngine(bundle)
        result = engine.predict_with_linked_structures(
            pd.DataFrame({"base": [1.0], "mol_id": ["water"]}),
            id_column="mol_id", structure_folder=str(xyz_dir),
            structure_type="molecule",
        )
        assert "prediction" in result.columns

    def test_missing_id_column_raises(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t",
            model_name="Mol", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'mol_n_atoms'], target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        with pytest.raises(ValueError, match="not found"):
            engine.predict_with_linked_structures(
                pd.DataFrame({"f1": [1.0]}), "bad_id", str(tmp_path)
            )

    def test_empty_structure_folder_raises(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t",
            model_name="Mol", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'mol_n_atoms'], target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        with pytest.raises(RuntimeError, match="No structure files matched"):
            engine.predict_with_linked_structures(
                pd.DataFrame({"f1": [1.0], "mol_id": ["x"]}),
                "mol_id", str(xyz_dir),
            )

    def test_plain_model_raises_on_linked_call(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t",
            model_name="Plain", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'f2'], target_column='y', preprocessor=None,
        )
        engine = InferenceEngine(result)
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            engine.predict_with_linked_structures(
                pd.DataFrame({"id": ["a"], "f1": [1.0]}),
                "id", str(tmp_path),
            )


# =============================================================================
# TEST 5: End-to-end with real ModelSaver
# =============================================================================

class TestEndToEndWithRealSaver:
    """Verify ModelSaver correctly flags linked features in manifest."""

    def test_saver_detects_no_linked_features(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t1",
            model_name="Plain", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'f2'], target_column='y', preprocessor=None,
        )
        manifest = json.loads((result / "manifest.json").read_text())
        assert manifest.get("requires_structure_linking") is False
        assert manifest.get("linked_feature_prefixes") == []

    def test_saver_detects_crystal_features(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t2",
            model_name="Crystal", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'crystal_density', 'crystal_lattice_a'],
            target_column='y', preprocessor=None,
        )
        manifest = json.loads((result / "manifest.json").read_text())
        assert manifest.get("requires_structure_linking") is True
        assert 'crystal_' in manifest.get("linked_feature_prefixes", [])

    def test_saver_detects_molecule_features(self, tmp_path):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=tmp_path / "t3",
            model_name="Molecule", model_type="sklearn", algorithm="rf",
            feature_names=['f1', 'mol_n_atoms', 'mol_mass'],
            target_column='y', preprocessor=None,
        )
        manifest = json.loads((result / "manifest.json").read_text())
        assert manifest.get("requires_structure_linking") is True
        assert 'mol_' in manifest.get("linked_feature_prefixes", [])
