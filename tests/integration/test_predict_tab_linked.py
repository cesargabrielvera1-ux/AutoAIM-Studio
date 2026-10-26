"""
Integration tests for Predict Tab linked-structure flows.

Tests the 2 main cases:
  1. Molecule structures (.xyz linking) — CSV with IDs + .xyz files
  2. Crystal structures (.cif linking) — CSV with IDs + .cif files

Plus cross-cutting tests for error handling, edge cases, and UI detection.

v2.0 Unified flow: composition is auto-derived from structure files during
linking. Both CrystalFeatureExtractor and MoleculeFeatureLinker now include
periodic properties (electronegativity, atomic mass, etc.) directly from the
structure — no separate Magpie/composition step needed.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.inference_engine import InferenceEngine
from app.core.model_saver import ModelSaver
from app.core.molecule_feature_linker import MoleculeFeatureLinker


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _write_xyz(tmp_path: Path, name: str, n_atoms: int,
               elements: list, coords: list) -> Path:
    """Write a minimal .xyz file."""
    p = tmp_path / f"{name}.xyz"
    lines = [str(n_atoms), name]
    for elem, (x, y, z) in zip(elements, coords):
        lines.append(f"{elem}  {x:.4f}  {y:.4f}  {z:.4f}")
    p.write_text("\n".join(lines))
    return p


def _create_fake_cif(tmp_path: Path, material_id: str) -> Path:
    """Create a minimal valid CIF file."""
    cif_path = tmp_path / f"{material_id}.cif"
    cif_content = f"""data_{material_id}
_chemical_formula_sum "Fe2 O3"
_cell_length_a 5.0000
_cell_length_b 5.0000
_cell_length_c 5.0000
_cell_angle_alpha 90.0000
_cell_angle_beta 90.0000
_cell_angle_gamma 90.0000
_cell_volume 125.0000
_space_group_IT_number 227
loop_
_atom_site_label
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Fe1 0.1250 0.1250 0.1250
Fe2 0.8750 0.8750 0.8750
O1 0.5000 0.5000 0.5000
O2 0.2500 0.2500 0.2500
O3 0.7500 0.7500 0.7500
"""
    cif_path.write_text(cif_content)
    return cif_path


def _create_model_bundle(tmp_path: Path, feature_names: list,
                         linked_prefixes: list) -> Path:
    """Create a real model bundle trained on the given features."""
    from sklearn.ensemble import RandomForestRegressor
    import joblib

    bundle = tmp_path / "model_bundle"
    bundle.mkdir()

    n_features = len(feature_names)
    X_train = np.random.randn(20, n_features)
    y_train = np.random.randn(20)
    model = RandomForestRegressor(n_estimators=5, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, bundle / "pipeline.joblib")

    saver = ModelSaver()
    saver.save_model(
        model=model,
        output_path=bundle,
        model_name="TestLinkedModel",
        model_type="sklearn",
        algorithm="random_forest",
        feature_names=feature_names,
        target_column="target",
        preprocessor=None,
    )
    return bundle


def _get_mol_feature_names(tmp_path: Path, xyz_files: list) -> list:
    """Get actual molecule feature names by linking sample files."""
    linker = MoleculeFeatureLinker()
    df_ids = pd.DataFrame({"mol_id": [p.stem for p in xyz_files]})
    linked_df, _ = linker.link_structures(
        df_ids, "mol_id", str(xyz_files[0].parent)
    )
    return [c for c in linked_df.columns
            if (c.startswith("mol_") or c.startswith("has_")) and c != "mol_id"]


# =============================================================================
# CASE 1: Molecule structures (.xyz linking)
# User has: CSV with properties + folder with .xyz files + ID column
# =============================================================================

class TestMoleculeLinkedPrediction:
    """Predict with models trained on linked molecular features."""

    def test_molecule_linked_prediction(self, tmp_path):
        """Full flow: CSV with IDs + .xyz files → link → predict."""
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        for name, elements, coords in [
            ("water", ["O", "H", "H"],
             [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            ("co2", ["C", "O", "O"],
             [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
        ]:
            _write_xyz(xyz_dir, name, len(elements), elements, coords)

        mol_features = _get_mol_feature_names(tmp_path, list(xyz_dir.glob("*.xyz")))
        feature_names = ["base_feature"] + mol_features
        bundle = _create_model_bundle(tmp_path, feature_names, ["mol_"])

        engine = InferenceEngine(bundle)
        assert engine.is_structure_linked_model() is True
        reqs = engine.get_linked_structure_requirements()
        assert reqs["structure_type"] == "molecule"

        df_pred = pd.DataFrame({
            "base_feature": [1.0, 2.0],
            "mol_id": ["water", "co2"],
        })

        val = engine.validate_input(df_pred)
        assert val["needs_linking_action"] is True

        result = engine.predict_with_linked_structures(
            df=df_pred, id_column="mol_id", structure_folder=str(xyz_dir),
        )
        assert "prediction" in result.columns
        assert len(result) == 2

    def test_molecule_informative_error(self, tmp_path):
        """Without linking, predict() gives informative error."""
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        _write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                   [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        mol_features = _get_mol_feature_names(tmp_path, list(xyz_dir.glob("*.xyz")))
        feature_names = ["base_feature"] + mol_features
        bundle = _create_model_bundle(tmp_path, feature_names, ["mol_"])

        engine = InferenceEngine(bundle)
        df_pred = pd.DataFrame({"base_feature": [1.0], "mol_id": ["water"]})

        with pytest.raises(ValueError, match="predict_with_linked_structures"):
            engine.predict(df_pred)


# =============================================================================
# CASE 2: Crystal structures (.cif linking)
# User has: CSV with properties + folder with .cif files + ID column
# =============================================================================

class TestCrystalLinkedPrediction:
    """Predict with models trained on linked crystal features."""

    def test_crystal_linked_prediction(self, tmp_path):
        """Full flow: CSV with IDs + .cif files → link → predict."""
        pytest.importorskip("pymatgen", reason="pymatgen not available")

        cif_dir = tmp_path / "cifs"
        cif_dir.mkdir()
        for mid in ["MAT_001", "MAT_002"]:
            _create_fake_cif(cif_dir, mid)

        from app.core.crystal_feature_extractor import CrystalFeatureExtractor
        linker = CrystalFeatureExtractor()
        df_ids = pd.DataFrame({"material_id": ["MAT_001", "MAT_002"]})
        linked_df, _ = linker.link_structures(df_ids, "material_id", str(cif_dir))
        crystal_features = [c for c in linked_df.columns if c.startswith("crystal_")]

        feature_names = ["base_feature"] + crystal_features
        bundle = _create_model_bundle(tmp_path, feature_names, ["crystal_"])

        engine = InferenceEngine(bundle)
        assert engine.is_structure_linked_model() is True
        reqs = engine.get_linked_structure_requirements()
        assert reqs["structure_type"] == "crystal"

        df_pred = pd.DataFrame({
            "base_feature": [1.0, 2.0],
            "material_id": ["MAT_001", "MAT_002"],
        })

        result = engine.predict_with_linked_structures(
            df=df_pred, id_column="material_id", structure_folder=str(cif_dir),
        )
        assert "prediction" in result.columns
        assert len(result) == 2

    def test_crystal_informative_error(self, tmp_path):
        """Without linking, predict() gives informative error."""
        pytest.importorskip("pymatgen", reason="pymatgen not available")

        cif_dir = tmp_path / "cifs"
        cif_dir.mkdir()
        _create_fake_cif(cif_dir, "MAT_001")

        from app.core.crystal_feature_extractor import CrystalFeatureExtractor
        linker = CrystalFeatureExtractor()
        df_ids = pd.DataFrame({"material_id": ["MAT_001"]})
        linked_df, _ = linker.link_structures(df_ids, "material_id", str(cif_dir))
        crystal_features = [c for c in linked_df.columns if c.startswith("crystal_")]

        feature_names = ["base_feature"] + crystal_features
        bundle = _create_model_bundle(tmp_path, feature_names, ["crystal_"])

        engine = InferenceEngine(bundle)
        df_pred = pd.DataFrame({"base_feature": [1.0], "material_id": ["MAT_001"]})

        with pytest.raises(ValueError, match="predict_with_linked_structures"):
            engine.predict(df_pred)


# =============================================================================
# Cross-cutting tests
# =============================================================================

class TestLinkedPredictionEdgeCases:
    """Edge cases and error handling for linked prediction."""

    def test_plain_model_not_linked(self, tmp_path):
        """Plain model (no linked features) works normally."""
        feature_names = ["feature_1", "feature_2", "feature_3"]
        bundle = _create_model_bundle(tmp_path, feature_names, [])

        engine = InferenceEngine(bundle)
        assert engine.is_structure_linked_model() is False

        df_pred = pd.DataFrame({"feature_1": [1.0], "feature_2": [2.0], "feature_3": [3.0]})
        result = engine.predict(df_pred)
        assert "prediction" in result.columns

    def test_missing_id_column_raises(self, tmp_path):
        """predict_with_linked_structures raises on missing ID column."""
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        _write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                   [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        mol_features = _get_mol_feature_names(tmp_path, list(xyz_dir.glob("*.xyz")))
        feature_names = ["base_feature"] + mol_features
        bundle = _create_model_bundle(tmp_path, feature_names, ["mol_"])

        engine = InferenceEngine(bundle)
        df_pred = pd.DataFrame({"base_feature": [1.0]})

        with pytest.raises(ValueError, match="not found"):
            engine.predict_with_linked_structures(
                df_pred, "bad_id", str(xyz_dir)
            )

    def test_empty_structure_folder_raises(self, tmp_path):
        """predict_with_linked_structures raises when no files match."""
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()  # Empty

        df_pred = pd.DataFrame({"base_feature": [1.0], "mol_id": ["unknown"]})
        feature_names = ["base_feature", "mol_n_atoms"]
        bundle = _create_model_bundle(tmp_path, feature_names, ["mol_"])

        engine = InferenceEngine(bundle)
        with pytest.raises(RuntimeError, match="No structure files matched"):
            engine.predict_with_linked_structures(
                df_pred, "mol_id", str(xyz_dir)
            )

    def test_partial_match_predictable(self, tmp_path):
        """Partial match (some missing structures) still produces predictions."""
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        _write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                   [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        mol_features = _get_mol_feature_names(tmp_path, list(xyz_dir.glob("*.xyz")))
        feature_names = ["base_feature"] + mol_features
        bundle = _create_model_bundle(tmp_path, feature_names, ["mol_"])

        engine = InferenceEngine(bundle)
        df_pred = pd.DataFrame({
            "base_feature": [1.0, 2.0],
            "mol_id": ["water", "co2"],
        })

        result = engine.predict_with_linked_structures(
            df_pred, "mol_id", str(xyz_dir)
        )
        assert "prediction" in result.columns
        assert len(result) == 2

    def test_manifest_flags_linked_features(self, tmp_path):
        """ModelSaver correctly flags linked features in manifest."""
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        _write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                   [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        mol_features = _get_mol_feature_names(tmp_path, list(xyz_dir.glob("*.xyz")))
        feature_names = ["base_feature"] + mol_features
        bundle = _create_model_bundle(tmp_path, feature_names, ["mol_"])

        manifest = json.loads((bundle / "manifest.json").read_text())
        assert manifest.get("requires_structure_linking") is True
        assert "mol_" in manifest.get("linked_feature_prefixes", [])
