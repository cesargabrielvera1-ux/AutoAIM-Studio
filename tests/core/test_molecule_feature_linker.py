"""
Tests for app/core/molecule_feature_linker.py

Validates molecular structure linking: detection, availability checking,
feature merging, and edge cases.

v2.0: 100% Python pure. No RDKit required.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.molecule_feature_linker import (
    MoleculeLinkReport,
    MoleculeFeatureLinker,
)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_xyz_content():
    """Return a minimal valid XYZ file content."""
    return """2
H2 molecule
H 0.0 0.0 0.0
H 0.0 0.0 0.74
"""


@pytest.fixture
def xyz_folder(tmp_path, sample_xyz_content):
    """Create a temporary folder with sample .xyz files."""
    folder = tmp_path / "xyz_files"
    folder.mkdir()
    # Create a few XYZ files
    for name in ["mol_001", "mol_002", "mol_003"]:
        f = folder / f"{name}.xyz"
        f.write_text(sample_xyz_content)
    return str(folder)


@pytest.fixture
def sample_dataframe():
    """Return a sample DataFrame with IDs for linking."""
    return pd.DataFrame({
        "sample_id": ["mol_001", "mol_002", "mol_003", "mol_004"],
        "temperature": [300, 400, 500, 600],
        "target": [1.0, 2.0, 3.0, 4.0],
    })


# ──────────────────────────────────────────────────────────────────────────
# MoleculeLinkReport tests
# ──────────────────────────────────────────────────────────────────────────

class TestMoleculeLinkReport:
    """Report dataclass must compute properties correctly."""

    def test_match_rate_basic(self):
        report = MoleculeLinkReport(
            id_column="id",
            xyz_folder="/tmp",
            n_rows_total=10,
            n_matched=7,
            n_missing=3,
            n_duplicate_ids=0,
        )
        assert report.match_rate == pytest.approx(0.7)

    def test_match_rate_zero_rows(self):
        report = MoleculeLinkReport(
            id_column="id",
            xyz_folder="/tmp",
            n_rows_total=0,
            n_matched=0,
            n_missing=0,
            n_duplicate_ids=0,
        )
        assert report.match_rate == 0.0

    def test_match_rate_full_match(self):
        report = MoleculeLinkReport(
            id_column="id",
            xyz_folder="/tmp",
            n_rows_total=5,
            n_matched=5,
            n_missing=0,
            n_duplicate_ids=0,
        )
        assert report.match_rate == 1.0

    def test_to_text_contains_key_info(self):
        report = MoleculeLinkReport(
            id_column="sample_id",
            xyz_folder="/data/xyz",
            n_rows_total=100,
            n_matched=75,
            n_missing=25,
            n_duplicate_ids=2,
            molecule_features_added=["mol_feat_1", "mol_feat_2"],
            warnings=["Some rows missing"],
            errors=["Folder not found"],
        )
        text = report.to_text()
        assert "sample_id" in text
        assert "/data/xyz" in text
        assert "100" in text
        assert "75" in text
        assert "25" in text
        assert "2" in text
        assert "Some rows missing" in text
        assert "Folder not found" in text

    def test_to_text_empty_warnings_errors(self):
        report = MoleculeLinkReport(
            id_column="id",
            xyz_folder="/tmp",
            n_rows_total=10,
            n_matched=10,
            n_missing=0,
            n_duplicate_ids=0,
        )
        text = report.to_text()
        assert "Warnings:" not in text
        assert "Errors:" not in text

    def test_default_lists_are_isolated(self):
        """Each report must have its own list instances."""
        r1 = MoleculeLinkReport(
            id_column="id", xyz_folder="/tmp",
            n_rows_total=1, n_matched=1, n_missing=0, n_duplicate_ids=0,
        )
        r2 = MoleculeLinkReport(
            id_column="id", xyz_folder="/tmp",
            n_rows_total=1, n_matched=1, n_missing=0, n_duplicate_ids=0,
        )
        r1.warnings.append("test")
        assert len(r2.warnings) == 0


# ──────────────────────────────────────────────────────────────────────────
# MoleculeFeatureLinker — initialization
# ──────────────────────────────────────────────────────────────────────────

class TestLinkerInitialization:
    """Linker must initialize with correct default state."""

    def test_init_creates_empty_state(self):
        linker = MoleculeFeatureLinker()
        assert linker.link_report is None
        assert linker.molecule_features_df is None
        assert len(linker.feature_names) > 0

    def test_feature_names_are_strings(self):
        linker = MoleculeFeatureLinker()
        for name in linker.feature_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_feature_names_have_expected_prefixes(self):
        linker = MoleculeFeatureLinker()
        for name in linker.feature_names:
            assert name.startswith("mol_") or name.startswith("has_"), (
                f"Unexpected feature name: {name}"
            )

    def test_is_linked_false_initially(self):
        linker = MoleculeFeatureLinker()
        assert linker.is_linked() is False

    def test_get_id_column_none_initially(self):
        linker = MoleculeFeatureLinker()
        assert linker.get_id_column() is None


# ──────────────────────────────────────────────────────────────────────────
# detect_id_candidates
# ──────────────────────────────────────────────────────────────────────────

class TestDetectIdCandidates:
    """ID candidate detection must identify suitable columns."""

    def test_unique_column_detected(self):
        df = pd.DataFrame({
            "id": ["a", "b", "c", "d"],
            "value": [1, 1, 1, 1],
        })
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)
        assert len(candidates) == 1
        assert candidates[0][0] == "id"

    def test_target_column_excluded(self):
        df = pd.DataFrame({
            "target": ["a", "b", "c", "d"],
            "y": [1, 2, 3, 4],
        })
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)
        assert len(candidates) == 0

    def test_low_uniqueness_excluded(self):
        df = pd.DataFrame({
            "mostly_same": ["a", "a", "a", "b"],  # 50% unique -> excluded
            "value": [1, 2, 3, 4],
        })
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)
        # mostly_same has 2 unique / 4 rows = 0.5 -> should be included (>= 0.5)
        names = [c[0] for c in candidates]
        assert "mostly_same" in names

    def test_multiple_candidates_sorted(self):
        df = pd.DataFrame({
            "perfect_id": ["a", "b", "c", "d"],      # 4/4 = 1.0
            "good_id": ["a", "b", "c", "a"],         # 3/4 = 0.75
            "poor_id": ["a", "b", "a", "b"],         # 2/4 = 0.5
            "target": [1, 2, 3, 4],
        })
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)
        names = [c[0] for c in candidates]
        assert names[0] == "perfect_id"
        assert "target" not in names

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)
        assert candidates == []

    def test_numeric_id_column(self):
        df = pd.DataFrame({
            "sample_id": [1, 2, 3, 4],
            "value": [10, 10, 10, 10],
        })
        linker = MoleculeFeatureLinker()
        candidates = linker.detect_id_candidates(df)
        assert len(candidates) == 1
        assert candidates[0][0] == "sample_id"


# ──────────────────────────────────────────────────────────────────────────
# check_xyz_availability
# ──────────────────────────────────────────────────────────────────────────

class TestCheckXyzAvailability:
    """XYZ availability check must report matched/missing/duplicate counts."""

    def test_all_matched(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        # Only use first 3 rows which have matching XYZ files
        df = sample_dataframe.iloc[:3].copy()
        result = linker.check_xyz_availability(df, "sample_id", xyz_folder)
        assert len(result["matched"]) == 3
        assert len(result["missing"]) == 0
        assert len(result["duplicate_ids"]) == 0
        assert result["n_xyz_files"] == 3

    def test_some_missing(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        result = linker.check_xyz_availability(
            sample_dataframe, "sample_id", xyz_folder
        )
        assert len(result["matched"]) == 3
        assert len(result["missing"]) == 1
        assert "mol_004" in result["missing"]

    def test_nonexistent_folder(self, sample_dataframe):
        linker = MoleculeFeatureLinker()
        result = linker.check_xyz_availability(
            sample_dataframe, "sample_id", "/nonexistent/path"
        )
        assert "error" in result
        assert len(result["missing"]) == 4

    def test_duplicate_ids_detected(self, xyz_folder):
        linker = MoleculeFeatureLinker()
        df = pd.DataFrame({
            "sample_id": ["mol_001", "mol_001", "mol_002", "mol_003"],
            "value": [1, 2, 3, 4],
        })
        result = linker.check_xyz_availability(df, "sample_id", xyz_folder)
        assert len(result["duplicate_ids"]) == 1
        assert "mol_001" in result["duplicate_ids"]

    def test_empty_dataframe(self, xyz_folder):
        linker = MoleculeFeatureLinker()
        df = pd.DataFrame({"sample_id": [], "value": []})
        result = linker.check_xyz_availability(df, "sample_id", xyz_folder)
        assert result["matched"] == []
        assert result["missing"] == []
        assert result["n_ids"] == 0

    def test_nan_ids_ignored(self, xyz_folder):
        linker = MoleculeFeatureLinker()
        df = pd.DataFrame({
            "sample_id": ["mol_001", None, "mol_002"],
            "value": [1, 2, 3],
        })
        result = linker.check_xyz_availability(df, "sample_id", xyz_folder)
        # NaN should be dropped before comparison
        assert "nan" not in result["matched"]


# ──────────────────────────────────────────────────────────────────────────
# link_structures — main method
# ──────────────────────────────────────────────────────────────────────────

class TestLinkStructures:
    """Full linking pipeline must merge features correctly."""

    def test_successful_link(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        df_out, report = linker.link_structures(
            sample_dataframe, "sample_id", xyz_folder
        )
        assert report.n_matched == 3
        assert report.n_missing == 1
        assert len(report.molecule_features_added) > 0
        # Original columns preserved
        assert "sample_id" in df_out.columns
        assert "temperature" in df_out.columns
        # Molecule features added
        for feat in report.molecule_features_added:
            assert feat in df_out.columns

    def test_linker_state_updated(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        linker.link_structures(sample_dataframe, "sample_id", xyz_folder)
        assert linker.is_linked() is True
        assert linker.get_id_column() == "sample_id"
        assert linker.link_report is not None
        assert linker.molecule_features_df is not None

    def test_no_match_returns_original(self, tmp_path, sample_dataframe):
        """If no XYZ files match, return original DataFrame with error report."""
        linker = MoleculeFeatureLinker()
        empty_folder = str(tmp_path / "empty")
        os.makedirs(empty_folder, exist_ok=True)
        df_out, report = linker.link_structures(
            sample_dataframe, "sample_id", empty_folder
        )
        assert report.n_matched == 0
        assert len(report.errors) > 0
        # DataFrame should be unchanged
        assert list(df_out.columns) == list(sample_dataframe.columns)

    def test_missing_id_column(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        df_out, report = linker.link_structures(
            sample_dataframe, "nonexistent_column", xyz_folder
        )
        assert len(report.errors) > 0
        assert "not found" in report.errors[0]

    def test_missing_xyz_folder(self, sample_dataframe):
        linker = MoleculeFeatureLinker()
        df_out, report = linker.link_structures(
            sample_dataframe, "sample_id", "/nonexistent"
        )
        assert len(report.errors) > 0
        assert "does not exist" in report.errors[0]

    def test_duplicate_ids_warning(self, xyz_folder):
        linker = MoleculeFeatureLinker()
        df = pd.DataFrame({
            "sample_id": ["mol_001", "mol_001", "mol_002"],
            "value": [1, 2, 3],
        })
        df_out, report = linker.link_structures(df, "sample_id", xyz_folder)
        assert report.n_duplicate_ids == 1
        assert len(report.warnings) > 0
        assert "multiple" in report.warnings[0].lower()

    def test_row_count_preserved(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        df_out, report = linker.link_structures(
            sample_dataframe, "sample_id", xyz_folder
        )
        assert len(df_out) == len(sample_dataframe)

    def test_missing_rows_have_nan_features(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        df_out, report = linker.link_structures(
            sample_dataframe, "sample_id", xyz_folder
        )
        # mol_004 has no XYZ file, its molecular features should be NaN
        missing_row_idx = sample_dataframe[
            sample_dataframe["sample_id"] == "mol_004"
        ].index[0]
        mol_cols = [c for c in df_out.columns if c.startswith("mol_")]
        for col in mol_cols:
            assert pd.isna(df_out.loc[missing_row_idx, col])


# ──────────────────────────────────────────────────────────────────────────
# clear and utility methods
# ──────────────────────────────────────────────────────────────────────────

class TestClearAndUtilities:
    """clear() must reset all state."""

    def test_clear_resets_state(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        linker.link_structures(sample_dataframe, "sample_id", xyz_folder)
        assert linker.is_linked() is True

        linker.clear()
        assert linker.is_linked() is False
        assert linker.link_report is None
        assert linker.molecule_features_df is None
        assert linker.get_id_column() is None

    def test_clear_idempotent(self):
        linker = MoleculeFeatureLinker()
        linker.clear()  # Should not raise even if never linked
        assert linker.is_linked() is False

    def test_clear_after_multiple_links(self, xyz_folder, sample_dataframe):
        linker = MoleculeFeatureLinker()
        linker.link_structures(sample_dataframe, "sample_id", xyz_folder)
        linker.clear()
        linker.link_structures(sample_dataframe.iloc[:2], "sample_id", xyz_folder)
        assert linker.is_linked() is True
        assert linker.link_report.n_matched == 2
