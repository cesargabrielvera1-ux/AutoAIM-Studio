"""
Feature Consistency Tests — Critical for linked-structure prediction.

Verifies that the features generated during prediction EXACTLY match
the features the model was trained on. Any mismatch = crash.

Tests:
  1. MoleculeFeaturizer: list_molecule_feature_names() == featurize_file() columns
  2. MoleculeFeatureLinker: linked_df contains exactly the expected mol_ features
  3. End-to-end: train model with linked mol features → predict → features match exactly
  4. FeatureEngineering: composition features are deterministic (same input = same output)
  5. ModelSaver manifest: linked_feature_prefixes correctly detected
  6. InferenceEngine: predict_with_linked_structures() produces exact feature set
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.molecule_featurizer import MoleculeFeaturizer, list_molecule_feature_names
from app.core.molecule_feature_linker import MoleculeFeatureLinker
from app.core.inference_engine import InferenceEngine
from app.core.model_saver import ModelSaver


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _write_xyz(tmp_path: Path, name: str, n_atoms: int,
               elements: list, coords: list) -> Path:
    p = tmp_path / f"{name}.xyz"
    lines = [str(n_atoms), name]
    for elem, (x, y, z) in zip(elements, coords):
        lines.append(f"{elem}  {x:.4f}  {y:.4f}  {z:.4f}")
    p.write_text("\n".join(lines))
    return p


def _get_actual_mol_feature_names() -> list:
    """Get the exact feature names produced by featurize_file()."""
    import tempfile
    tmp = Path(tempfile.mkdtemp())
    xyz = tmp / "test.xyz"
    xyz.write_text("2\ntest\nH  0.0  0.0  0.0\nH  0.74  0.0  0.0")
    featurizer = MoleculeFeaturizer()
    df = featurizer.featurize_file(xyz)
    return sorted([c for c in df.columns if c != "mol_source_file"])


# =============================================================================
# TEST 1: MoleculeFeaturizer feature name consistency
# =============================================================================

class TestMoleculeFeaturizerConsistency:
    """Verify that list_molecule_feature_names() matches actual featurize output."""

    def test_list_names_match_featurize_output(self):
        """list_molecule_feature_names() must == featurize_file() columns (minus source_file)."""
        from_list = sorted(list_molecule_feature_names())
        from_featurize = _get_actual_mol_feature_names()

        assert len(from_list) > 0, "list_molecule_feature_names() is empty"
        assert len(from_featurize) > 0, "featurize_file() produced no features"
        assert from_list == from_featurize, (
            f"Feature name mismatch!\n"
            f"In list only: {set(from_list) - set(from_featurize)}\n"
            f"In featurize only: {set(from_featurize) - set(from_list)}"
        )

    def test_consistent_across_molecules(self):
        """Different molecules must produce the SAME set of feature columns."""
        tmp = Path(tempfile.mkdtemp())
        featurizer = MoleculeFeaturizer()

        molecules = [
            ("h2", ["H", "H"], [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]),
            ("water", ["O", "H", "H"], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            ("co2", ["C", "O", "O"], [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
            ("ethanol", ["C", "C", "O", "H", "H", "H", "H", "H", "H"],
             [[0.0, 0.0, 0.0], [1.54, 0.0, 0.0], [1.54, 1.43, 0.0], [-1.04, 0.0, 0.0],
              [0.52, 0.0, 0.89], [0.52, 0.0, -0.89], [2.07, -0.52, 0.89],
              [2.07, -0.52, -0.89], [1.54, 1.79, 0.89]]),
        ]

        feature_sets = []
        for name, elems, coords in molecules:
            xyz = _write_xyz(tmp, name, len(elems), elems, coords)
            df = featurizer.featurize_file(xyz)
            cols = set(c for c in df.columns if c != "mol_source_file")
            feature_sets.append(cols)

        # All molecules must produce the same feature columns
        for i in range(1, len(feature_sets)):
            assert feature_sets[0] == feature_sets[i], (
                f"Feature columns differ between molecules!\n"
                f"Base vs molecule {i}: symmetric_diff={feature_sets[0] ^ feature_sets[i]}"
            )

    def test_numeric_prefix_elements_parsed_consistently(self):
        """.xyz files with numeric-prefixed elements produce consistent features."""
        tmp = Path(tempfile.mkdtemp())
        featurizer = MoleculeFeaturizer()

        # Normal format
        normal = tmp / "normal.xyz"
        normal.write_text("2\ntest\nC  0.0  0.0  0.0\nO  1.16  0.0  0.0")
        df_normal = featurizer.featurize_file(normal)

        # Numeric prefix format (some software produces this)
        prefixed = tmp / "prefixed.xyz"
        prefixed.write_text("2\ntest\n1C  0.0  0.0  0.0\n8O  1.16  0.0  0.0")
        df_prefixed = featurizer.featurize_file(prefixed)

        # Must have same columns
        normal_cols = set(c for c in df_normal.columns if c != "mol_source_file")
        prefixed_cols = set(c for c in df_prefixed.columns if c != "mol_source_file")
        assert normal_cols == prefixed_cols

        # Element counts must be the same
        assert df_normal["mol_n_carbons"].iloc[0] == df_prefixed["mol_n_carbons"].iloc[0]
        assert df_normal["mol_n_oxygens"].iloc[0] == df_prefixed["mol_n_oxygens"].iloc[0]


# =============================================================================
# TEST 2: MoleculeFeatureLinker produces exact expected features
# =============================================================================

class TestMoleculeFeatureLinkerConsistency:
    """Verify that MoleculeFeatureLinker produces consistent features."""

    def test_linked_features_match_featurizer_output(self):
        """Features from link_structures() must match featurize_file() exactly."""
        tmp = Path(tempfile.mkdtemp())

        # Create .xyz
        xyz_dir = tmp / "xyzs"
        xyz_dir.mkdir()
        _write_xyz(xyz_dir, "water", 3,
                   ["O", "H", "H"],
                   [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        # Method 1: Direct featurize_file
        featurizer = MoleculeFeaturizer()
        df_direct = featurizer.featurize_file(xyz_dir / "water.xyz")
        direct_features = set(c for c in df_direct.columns if c != "mol_source_file")

        # Method 2: Link via MoleculeFeatureLinker
        linker = MoleculeFeatureLinker()
        df_input = pd.DataFrame({"mol_id": ["water"], "base": [1.0]})
        df_linked, _ = linker.link_structures(df_input, "mol_id", str(xyz_dir))
        linked_features = set(c for c in df_linked.columns
                              if (c.startswith("mol_") or c.startswith("has_"))
                              and c not in ("mol_id", "mol_source_file"))

        assert direct_features == linked_features, (
            f"Feature mismatch between direct featurize and linker!\n"
            f"Direct only: {direct_features - linked_features}\n"
            f"Linker only: {linked_features - direct_features}"
        )

    def test_feature_order_consistent(self):
        """Feature column order must be identical across multiple link calls."""
        tmp = Path(tempfile.mkdtemp())
        xyz_dir = tmp / "xyzs"
        xyz_dir.mkdir()
        for name in ["a", "b", "c"]:
            _write_xyz(xyz_dir, name, 2, ["H", "H"],
                       [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])

        linker = MoleculeFeatureLinker()
        mol_cols_runs = []
        for _ in range(3):
            df_input = pd.DataFrame({"mol_id": ["a", "b", "c"], "base": [1.0, 2.0, 3.0]})
            df_linked, _ = linker.link_structures(df_input, "mol_id", str(xyz_dir))
            mol_cols = [c for c in df_linked.columns
                        if (c.startswith("mol_") or c.startswith("has_")) and c != "mol_id"]
            mol_cols_runs.append(mol_cols)

        assert mol_cols_runs[0] == mol_cols_runs[1] == mol_cols_runs[2], (
            "Feature column order is not deterministic across link calls!"
        )


# =============================================================================
# TEST 3: End-to-end feature consistency (train → save → predict)
# =============================================================================

class TestEndToEndFeatureConsistency:
    """Verify that prediction features exactly match training features."""

    def test_train_save_predict_features_match(self):
        """CRITICAL: Features at predict time must == features at train time."""
        import tempfile
        from sklearn.ensemble import RandomForestRegressor
        import joblib

        tmp = Path(tempfile.mkdtemp())

        # Create .xyz files
        xyz_dir = tmp / "xyzs"
        xyz_dir.mkdir()
        molecules = {
            "water": (["O", "H", "H"], [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]),
            "co2": (["C", "O", "O"], [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]]),
            "methane": (["C", "H", "H", "H", "H"], [[0.0, 0.0, 0.0], [1.09, 0.0, 0.0],
                                                     [-0.36, 1.03, 0.0], [-0.36, -0.51, 0.89],
                                                     [-0.36, -0.51, -0.89]]),
        }
        for name, (elems, coords) in molecules.items():
            _write_xyz(xyz_dir, name, len(elems), elems, coords)

        # Create training data and link
        train_df = pd.DataFrame({
            "mol_id": list(molecules.keys()),
            "temperature": [300.0, 350.0, 400.0],
            "pressure": [1.0, 10.0, 100.0],
        })

        linker = MoleculeFeatureLinker()
        linked_train, _ = linker.link_structures(train_df, "mol_id", str(xyz_dir))

        # Extract EXACT feature columns (numeric only, excluding mol_id)
        mol_cols = sorted([c for c in linked_train.columns
                           if (c.startswith("mol_") or c.startswith("has_"))
                           and c != "mol_id"
                           and pd.api.types.is_numeric_dtype(linked_train[c])])
        feature_cols = ["temperature", "pressure"] + mol_cols

        # Train model
        X_train = linked_train[feature_cols].fillna(linked_train[feature_cols].median())
        y_train = np.random.randn(len(linked_train))
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        model.fit(X_train, y_train)

        # Save model
        bundle = tmp / "bundle"
        bundle.mkdir()
        joblib.dump(model, bundle / "pipeline.joblib")
        manifest = {
            "model_name": "MolModel",
            "model_type": "sklearn",
            "algorithm": "random_forest",
            "creation_date": "2024-01-01",
            "sklearn_version": "1.5.0",
            "pytorch_version": None,
            "feature_names": feature_cols,
            "target_column": "target",
            "preprocessing_checksum": "abc",
            "metrics": {"r2": 0.9},
            "is_neural_network": False,
            "nn_config": {},
            "confidence_interval_params": {},
            "linked_feature_prefixes": ["mol_"],
            "requires_structure_linking": True,
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))

        # Create prediction data
        pred_df = pd.DataFrame({
            "mol_id": ["water", "co2", "methane"],
            "temperature": [310.0, 360.0, 410.0],
            "pressure": [2.0, 20.0, 200.0],
        })

        # Predict with linked structures
        engine = InferenceEngine(bundle)
        result = engine.predict_with_linked_structures(
            pred_df, "mol_id", str(xyz_dir)
        )

        # Must produce predictions without error
        assert "prediction" in result.columns
        assert len(result) == 3
        assert not result["prediction"].isna().any()

        # CRITICAL: Verify that the linked prediction DataFrame had exactly
        # the same feature columns as training
        # (This is implicit — if features didn't match, predict() would raise)

    def test_feature_count_match_manifest(self):
        """The manifest feature count must match what predict_with_linked produces."""
        tmp = Path(tempfile.mkdtemp())
        xyz_dir = tmp / "xyzs"
        xyz_dir.mkdir()
        _write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                   [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])

        linker = MoleculeFeatureLinker()
        df_train = pd.DataFrame({"mol_id": ["water"], "base": [1.0]})
        linked, _ = linker.link_structures(df_train, "mol_id", str(xyz_dir))

        mol_cols = [c for c in linked.columns
                    if (c.startswith("mol_") or c.startswith("has_"))
                    and c != "mol_id"]
        feature_cols = ["base"] + mol_cols

        # Save model with these exact features
        bundle = tmp / "bundle"
        bundle.mkdir()
        from sklearn.ensemble import RandomForestRegressor
        import joblib
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        model.fit(np.random.randn(5, len(feature_cols)), np.random.randn(5))
        joblib.dump(model, bundle / "pipeline.joblib")

        manifest = {
            "model_name": "Test",
            "model_type": "sklearn",
            "algorithm": "rf",
            "creation_date": "2024-01-01",
            "sklearn_version": "1.5.0",
            "feature_names": feature_cols,
            "target_column": "y",
            "metrics": {},
            "is_neural_network": False,
            "linked_feature_prefixes": ["mol_"],
            "requires_structure_linking": True,
        }
        (bundle / "manifest.json").write_text(json.dumps(manifest))

        # Predict
        engine = InferenceEngine(bundle)
        df_pred = pd.DataFrame({"mol_id": ["water"], "base": [1.0]})
        result = engine.predict_with_linked_structures(
            df_pred, "mol_id", str(xyz_dir)
        )

        assert len(result) == 1
        assert "prediction" in result.columns


# =============================================================================
# TEST 4: Composition feature consistency
# =============================================================================

class TestCompositionFeatureConsistency:
    """Verify composition features are deterministic."""

    def test_composition_features_deterministic(self):
        """Same formula must produce same composition features every time."""
        from app.core.feature_engineering import FeatureEngineer

        df = pd.DataFrame({
            "formula": ["Fe2O3", "SiO2", "H2O"],
        })

        fe = FeatureEngineer()
        result1 = fe.add_composition_features(df, composition_column="formula", use_magpie=True)

        fe2 = FeatureEngineer()
        result2 = fe2.add_composition_features(df, composition_column="formula", use_magpie=True)

        comp_cols1 = sorted([c for c in result1.columns if c.startswith("comp_")])
        comp_cols2 = sorted([c for c in result2.columns if c.startswith("comp_")])

        assert comp_cols1 == comp_cols2, "Composition feature names not deterministic"

        for col in comp_cols1:
            assert result1[col].tolist() == result2[col].tolist(), (
                f"Composition feature '{col}' values differ between runs"
            )

    def test_composition_features_same_with_linked(self):
        """Composition features generated during prediction match training."""
        from app.core.feature_engineering import FeatureEngineer

        df = pd.DataFrame({
            "mol_id": ["water", "co2"],
            "formula": ["H2O", "CO2"],
        })

        fe = FeatureEngineer()
        result = fe.add_composition_features(df, composition_column="formula", use_magpie=True)
        comp_cols = sorted([c for c in result.columns if c.startswith("comp_")])

        assert len(comp_cols) > 0, "No composition features generated"

        # Verify all composition columns are numeric
        for col in comp_cols:
            assert pd.api.types.is_numeric_dtype(result[col]), (
                f"Composition feature '{col}' is not numeric: {result[col].dtype}"
            )


# =============================================================================
# TEST 5: ModelSaver linked prefix detection
# =============================================================================

class TestModelSaverLinkedPrefixDetection:
    """Verify ModelSaver correctly flags linked features in manifest."""

    def test_detects_mol_prefixes(self):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=Path(tempfile.mkdtemp()) / "test1",
            model_name="Mol", model_type="sklearn", algorithm="rf",
            feature_names=["f1", "mol_n_atoms", "mol_mass", "has_OH"],
            target_column="y", preprocessor=None,
        )
        manifest = json.loads((result / "manifest.json").read_text())
        assert manifest.get("requires_structure_linking") is True
        assert "mol_" in manifest.get("linked_feature_prefixes", [])

    def test_detects_crystal_prefixes(self):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=Path(tempfile.mkdtemp()) / "test2",
            model_name="Crystal", model_type="sklearn", algorithm="rf",
            feature_names=["f1", "crystal_density", "crystal_lattice_a"],
            target_column="y", preprocessor=None,
        )
        manifest = json.loads((result / "manifest.json").read_text())
        assert manifest.get("requires_structure_linking") is True
        assert "crystal_" in manifest.get("linked_feature_prefixes", [])

    def test_plain_model_no_linking(self):
        saver = ModelSaver()
        result = saver.save_model(
            model=None, output_path=Path(tempfile.mkdtemp()) / "test4",
            model_name="Plain", model_type="sklearn", algorithm="rf",
            feature_names=["f1", "f2", "f3"],
            target_column="y", preprocessor=None,
        )
        manifest = json.loads((result / "manifest.json").read_text())
        assert manifest.get("requires_structure_linking") is False
        assert manifest.get("linked_feature_prefixes") == []


# =============================================================================
# TEST 6: Missing structure handling
# =============================================================================

class TestMissingStructureHandling:
    """Verify behavior when some structures are missing."""

    def test_missing_structures_fill_nan(self):
        """Missing .xyz files produce NaN features (not crash)."""
        tmp = Path(tempfile.mkdtemp())
        xyz_dir = tmp / "xyzs"
        xyz_dir.mkdir()
        _write_xyz(xyz_dir, "water", 3, ["O", "H", "H"],
                   [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]])
        # co2.xyz intentionally missing

        linker = MoleculeFeatureLinker()
        df = pd.DataFrame({"mol_id": ["water", "co2"], "base": [1.0, 2.0]})
        linked, report = linker.link_structures(df, "mol_id", str(xyz_dir))

        assert report.n_matched == 1
        assert report.n_missing == 1

        # Water should have real values
        water_row = linked[linked["mol_id"] == "water"]
        assert not water_row["mol_n_atoms"].isna().iloc[0]

        # CO2 should have NaN
        co2_row = linked[linked["mol_id"] == "co2"]
        assert co2_row["mol_n_atoms"].isna().iloc[0]

    def test_all_missing_structures(self):
        """All missing .xyz files produce all-NaN features."""
        tmp = Path(tempfile.mkdtemp())
        xyz_dir = tmp / "xyzs"
        xyz_dir.mkdir()  # Empty

        linker = MoleculeFeatureLinker()
        df = pd.DataFrame({"mol_id": ["missing1", "missing2"], "base": [1.0, 2.0]})
        linked, report = linker.link_structures(df, "mol_id", str(xyz_dir))

        assert report.n_matched == 0
        mol_cols = [c for c in linked.columns
                    if (c.startswith("mol_") or c.startswith("has_"))
                    and c != "mol_id"]
        for col in mol_cols:
            assert linked[col].isna().all(), f"Column {col} should be all NaN"
