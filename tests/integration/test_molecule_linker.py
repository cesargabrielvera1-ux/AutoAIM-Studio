"""
Integration tests for MoleculeFeatureLinker.

Parallels the CrystalFeatureExtractor test suite to ensure both linkers
behave identically from the user's perspective.

Coverage:
  1. ID candidate detection
  2. XYZ availability check
  3. Full linking pipeline (all matched, partial, none)
  4. Report generation
  5. Edge cases (nonexistent folder, duplicate IDs, invalid columns)
  6. Integration with DataManager + training pipeline
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.molecule_feature_linker import (
    MoleculeFeatureLinker,
    MoleculeLinkReport,
)
from app.core.data_manager import DataManager
from app.core.trainer import ModelTrainer


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def _write_xyz(tmp_path: Path, name: str, n_atoms: int, elements: list,
               coords: list) -> Path:
    """Write a minimal .xyz file and return its path."""
    p = tmp_path / f"{name}.xyz"
    lines = [str(n_atoms), name]
    for elem, (x, y, z) in zip(elements, coords):
        lines.append(f"{elem}  {x:.4f}  {y:.4f}  {z:.4f}")
    p.write_text("\n".join(lines))
    return p


def _make_dataset(tmp_path: Path, n: int = 20) -> pd.DataFrame:
    """Create a synthetic dataset with molecule IDs and a target."""
    solvents = ['water', 'ethanol', 'acetone', 'dmso', 'hexane']
    df = pd.DataFrame({
        'molecule_id': [f"MOL_{i:03d}" for i in range(n)],
        'solvent': [solvents[i % len(solvents)] for i in range(n)],
        'temperature': np.random.uniform(250, 400, n),
        'target_property': np.random.randn(n),
    })
    return df


def _create_xyz_folder(tmp_path: Path, ids: list) -> Path:
    """Create a folder with .xyz files for the given IDs."""
    xyz_dir = tmp_path / "xyzs"
    xyz_dir.mkdir()
    for mid in ids:
        _write_xyz(
            xyz_dir, mid, 3,
            ['C', 'H', 'O'],
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
        )
    return xyz_dir


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def linker():
    return MoleculeFeatureLinker()


# =============================================================================
# TEST 1: ID Candidate Detection
# =============================================================================

class TestIDCandidateDetection:
    """Parallels CrystalFeatureExtractor.detect_id_candidates."""

    def test_detects_unique_id_column(self, tmp_path):
        df = _make_dataset(tmp_path)
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)

        assert len(candidates) > 0
        assert candidates[0][0] == 'molecule_id'

    def test_skips_target_like_columns(self, tmp_path):
        df = pd.DataFrame({
            'target': [1, 2, 3, 4, 5],
            'y': [1, 2, 3, 4, 5],
            'label': [1, 2, 3, 4, 5],
            'real_id': ['A', 'B', 'C', 'D', 'E'],
        })
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)

        names = [c[0] for c in candidates]
        assert 'target' not in names
        assert 'y' not in names
        assert 'label' not in names
        assert 'real_id' in names

    def test_low_uniqueness_excluded(self, tmp_path):
        df = pd.DataFrame({
            'mostly_same': ['A'] * 10 + ['B'],
            'unique': [f"X{i}" for i in range(11)],
        })
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)

        names = [c[0] for c in candidates]
        assert 'unique' in names
        # mostly_same has < 50% uniqueness, should be excluded
        assert 'mostly_same' not in names


# =============================================================================
# TEST 2: XYZ Availability Check
# =============================================================================

class TestXYZAvailability:
    """Parallels CrystalFeatureExtractor.check_cif_availability."""

    def test_all_matched(self, tmp_path):
        df = _make_dataset(tmp_path, n=5)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(5)])
        linker = MoleculeFeatureLinker()
        result = linker.check_xyz_availability(df, 'molecule_id', str(xyz_dir))

        assert len(result['matched']) == 5
        assert len(result['missing']) == 0
        assert result['n_xyz_files'] == 5

    def test_partial_match(self, tmp_path):
        df = _make_dataset(tmp_path, n=10)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(3)])
        linker = MoleculeFeatureLinker()
        result = linker.check_xyz_availability(df, 'molecule_id', str(xyz_dir))

        assert len(result['matched']) == 3
        assert len(result['missing']) == 7

    def test_no_xyz_folder(self, tmp_path):
        df = _make_dataset(tmp_path, n=3)
        linker = MoleculeFeatureLinker()
        result = linker.check_xyz_availability(df, 'molecule_id', str(tmp_path / "nonexistent"))

        assert len(result['matched']) == 0
        assert 'error' in result

    def test_duplicate_ids_reported(self, tmp_path):
        df = _make_dataset(tmp_path, n=5)
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(5)])
        linker = MoleculeFeatureLinker()
        result = linker.check_xyz_availability(df, 'molecule_id', str(xyz_dir))

        assert len(result['duplicate_ids']) == 2


# =============================================================================
# TEST 3: Full Linking Pipeline
# =============================================================================

class TestFullLinkingPipeline:
    """Parallels CrystalFeatureExtractor.link_structures tests."""

    def test_all_molecules_linked(self, tmp_path):
        df = _make_dataset(tmp_path, n=10)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(10)])
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'molecule_id', str(xyz_dir))

        assert report.n_matched == 10
        assert report.n_missing == 0
        assert len(report.errors) == 0
        # Molecule features should be added
        assert any(c.startswith('mol_') for c in result_df.columns)
        assert 'mol_n_atoms' in result_df.columns
        assert 'mol_mass' in result_df.columns

    def test_partial_match_with_nan_fill(self, tmp_path):
        df = _make_dataset(tmp_path, n=10)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(4)])
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'molecule_id', str(xyz_dir))

        assert report.n_matched == 4
        assert report.n_missing == 6
        # First 4 should have real values, rest should be NaN
        assert not result_df['mol_n_atoms'].iloc[:4].isna().any()
        assert result_df['mol_n_atoms'].iloc[4:].isna().all()

    def test_no_match_returns_error(self, tmp_path):
        df = _make_dataset(tmp_path, n=5)
        xyz_dir = tmp_path / "empty_xyz"
        xyz_dir.mkdir()
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'molecule_id', str(xyz_dir))

        assert report.n_matched == 0
        assert len(report.errors) > 0
        assert "No .xyz files matched" in report.errors[0]

    def test_nonexistent_folder(self, tmp_path):
        df = _make_dataset(tmp_path, n=3)
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'molecule_id', str(tmp_path / "nope"))

        assert report.n_matched == 0
        assert len(report.errors) > 0
        assert "does not exist" in report.errors[0]

    def test_missing_id_column(self, tmp_path):
        df = _make_dataset(tmp_path, n=3)
        xyz_dir = _create_xyz_folder(tmp_path, ["MOL_000"])
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'nonexistent_column', str(xyz_dir))

        assert len(report.errors) > 0
        assert "not found" in report.errors[0]

    def test_duplicate_ids_warning(self, tmp_path):
        df = _make_dataset(tmp_path, n=5)
        df = pd.concat([df, df.iloc[:2]], ignore_index=True)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(5)])
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'molecule_id', str(xyz_dir))

        assert report.n_duplicate_ids == 2
        assert len(report.warnings) > 0

    def test_report_text_output(self, tmp_path):
        df = _make_dataset(tmp_path, n=5)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(5)])
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'molecule_id', str(xyz_dir))
        text = report.to_text()

        assert "Molecule Structure Link Report" in text
        assert "molecule_id" in text
        assert str(report.n_matched) in text
        assert f"{report.match_rate*100:.1f}%" in text

    def test_original_columns_preserved(self, tmp_path):
        df = _make_dataset(tmp_path, n=3)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(3)])
        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'molecule_id', str(xyz_dir))

        assert 'solvent' in result_df.columns
        assert 'temperature' in result_df.columns
        assert 'target_property' in result_df.columns
        assert len(result_df) == len(df)

    def test_consistent_feature_count(self, tmp_path):
        """CRITICAL: Same number of mol_ columns regardless of which .xyz files match."""
        df1 = _make_dataset(tmp_path, n=3)
        df2 = _make_dataset(tmp_path, n=3)

        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(3)])
        linker = MoleculeFeatureLinker()

        result1, _ = linker.link_structures(df1, 'molecule_id', str(xyz_dir))
        linker.clear()
        result2, _ = linker.link_structures(df2, 'molecule_id', str(xyz_dir))

        mol_cols_1 = sorted([c for c in result1.columns if c.startswith('mol_')])
        mol_cols_2 = sorted([c for c in result2.columns if c.startswith('mol_')])
        assert mol_cols_1 == mol_cols_2

    def test_feature_values_differ_across_molecules(self, tmp_path):
        """Different molecules should produce different feature values."""
        df = pd.DataFrame({
            'mol_id': ['small', 'large'],
        })
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()

        # H2 — small molecule
        _write_xyz(xyz_dir, 'small', 2,
                   ['H', 'H'],
                   [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]])

        # CO2 — larger molecule
        _write_xyz(xyz_dir, 'large', 3,
                   ['C', 'O', 'O'],
                   [[0.0, 0.0, 0.0], [1.16, 0.0, 0.0], [-1.16, 0.0, 0.0]])

        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'mol_id', str(xyz_dir))

        assert result_df['mol_n_atoms'].iloc[0] == 2.0
        assert result_df['mol_n_atoms'].iloc[1] == 3.0
        assert result_df['mol_mass'].iloc[1] > result_df['mol_mass'].iloc[0]

    def test_is_linked_state(self, tmp_path):
        linker = MoleculeFeatureLinker()
        assert not linker.is_linked()

        df = _make_dataset(tmp_path, n=2)
        xyz_dir = _create_xyz_folder(tmp_path, ["MOL_000", "MOL_001"])
        linker.link_structures(df, 'molecule_id', str(xyz_dir))
        assert linker.is_linked()

        linker.clear()
        assert not linker.is_linked()

    def test_get_id_column(self, tmp_path):
        df = _make_dataset(tmp_path, n=2)
        xyz_dir = _create_xyz_folder(tmp_path, ["MOL_000", "MOL_001"])
        linker = MoleculeFeatureLinker()
        linker.link_structures(df, 'molecule_id', str(xyz_dir))

        assert linker.get_id_column() == 'molecule_id'

    def test_molecule_features_df_property(self, tmp_path):
        df = _make_dataset(tmp_path, n=3)
        xyz_dir = _create_xyz_folder(tmp_path, [f"MOL_{i:03d}" for i in range(3)])
        linker = MoleculeFeatureLinker()
        linker.link_structures(df, 'molecule_id', str(xyz_dir))

        assert linker.molecule_features_df is not None
        assert len(linker.molecule_features_df) == 3

    def test_feature_names_property(self, tmp_path):
        linker = MoleculeFeatureLinker()
        names = linker.feature_names

        assert len(names) > 50  # Should have many features
        assert all(n.startswith('mol_') or n.startswith('has_') for n in names)
        assert 'mol_n_atoms' in names
        assert 'mol_mass' in names


# =============================================================================
# TEST 4: End-to-End with DataManager + Training
# =============================================================================

class TestEndToEndWithTraining:
    """Full workflow: link .xyz → DataManager → train model."""

    def test_molecule_linked_data_can_train_model(self, tmp_path):
        """The complete workflow from .xyz linking to model training."""
        # Create dataset with molecule IDs
        n = 20
        df = pd.DataFrame({
            'mol_id': [f"MOL_{i:03d}" for i in range(n)],
            'experimental_temp': np.random.uniform(200, 400, n),
            'band_gap': np.random.uniform(1.0, 5.0, n),
        })
        data_path = tmp_path / "molecular_data.csv"
        df.to_csv(data_path, index=False)

        # Create .xyz files for all molecules
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        for i in range(n):
            _write_xyz(
                xyz_dir, f"MOL_{i:03d}", 3,
                ['C', 'H', 'O'],
                [[0.0, 0.0, 0.0], [1.0 + i*0.01, 0.0, 0.0], [0.0, 1.0, 0.0]]
            )

        # Step 1: Link molecular features
        linker = MoleculeFeatureLinker()
        linked_df, report = linker.link_structures(df, 'mol_id', str(xyz_dir))

        assert report.n_matched == n
        assert any(c.startswith('mol_') for c in linked_df.columns)

        # Step 2: Save linked dataset
        linked_path = tmp_path / "linked_data.csv"
        linked_df.to_csv(linked_path, index=False)

        # Step 3: Load through DataManager
        dm = DataManager()
        dm.load_data(str(linked_path))
        dm.set_target_column('band_gap')
        dm.get_preprocessor(numeric_strategy='standard', impute_strategy='mean')
        X_train, X_test, y_train, y_test = dm.prepare_data()

        assert X_train.shape[0] > 0
        # v2.0: X_train is RAW — NaNs are handled by the Pipeline during training.
        # Should have both original and molecular features
        assert X_train.shape[1] > 2  # More than just experimental_temp

        # Step 4: Train a model with Pipeline (preprocessor handles NaNs)
        from sklearn.ensemble import RandomForestRegressor
        trainer = ModelTrainer()
        preprocessor = dm.get_preprocessor()
        result = trainer.train(
            RandomForestRegressor(n_estimators=10, random_state=42),
            'RF_MolLinked', X_train, y_train, X_test, y_test, 'regression',
            feature_names=dm.get_feature_names(),
            preprocessor=preprocessor,
        )

        assert result is not None
        assert 'r2' in result.metrics

    def test_partial_linking_still_trainable(self, tmp_path):
        """Even with missing .xyz files, the dataset remains trainable."""
        n = 15
        df = pd.DataFrame({
            'mol_id': [f"MOL_{i:03d}" for i in range(n)],
            'feature_a': np.random.randn(n),
            'target': np.random.randn(n),
        })

        # Only create .xyz for half
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()
        for i in range(n // 2):
            _write_xyz(
                xyz_dir, f"MOL_{i:03d}", 2,
                ['H', 'H'],
                [[0.0, 0.0, 0.0], [0.74, 0.0, 0.0]]
            )

        linker = MoleculeFeatureLinker()
        linked_df, report = linker.link_structures(df, 'mol_id', str(xyz_dir))

        assert report.n_matched == n // 2
        assert report.n_missing == n - n // 2

        # Save and load
        linked_path = tmp_path / "partial_linked.csv"
        linked_df.to_csv(linked_path, index=False)

        dm = DataManager()
        dm.load_data(str(linked_path))
        dm.set_target_column('target')
        preprocessor = dm.get_preprocessor(numeric_strategy='standard', impute_strategy='mean')
        X_train, X_test, y_train, y_test = dm.prepare_data()

        # v2.0: X_train is RAW — NaNs are handled inside the Pipeline during training.
        # Trainable with Pipeline wrapper
        from sklearn.ensemble import RandomForestRegressor
        trainer = ModelTrainer()
        result = trainer.train(
            RandomForestRegressor(n_estimators=5, random_state=42),
            'RF', X_train, y_train, problem_type='regression',
            preprocessor=preprocessor,
        )
        assert result is not None

    def test_linker_with_numeric_prefix_elements(self, tmp_path):
        """.xyz files with numeric-prefixed elements (e.g. '1C') parse correctly."""
        df = pd.DataFrame({
            'mol_id': ['ethane'],
            'target': [1.0],
        })
        xyz_dir = tmp_path / "xyzs"
        xyz_dir.mkdir()

        # Write .xyz with numeric prefixes (some software produces these)
        p = xyz_dir / "ethane.xyz"
        p.write_text("\n".join([
            "2",
            "Ethane test",
            "1C  0.0000  0.0000  0.0000",
            "2C  1.5400  0.0000  0.0000",
        ]))

        linker = MoleculeFeatureLinker()
        result_df, report = linker.link_structures(df, 'mol_id', str(xyz_dir))

        assert report.n_matched == 1
        assert result_df['mol_n_carbons'].iloc[0] == 2.0
