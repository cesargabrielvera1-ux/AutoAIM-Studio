"""Comprehensive tests for DatasetExporter (v2.0 dataset export feature).

Covers:
  1. Feature breakdown detection (crystal, molecular, functional group, Magpie)
  2. Filename suggestion logic
  3. CSV export with/without index
  4. Export summary generation
  5. Edge cases: empty DataFrame, missing directory, None input
  6. Integration: round-trip export → read-back
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.dataset_exporter import DatasetExporter


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def plain_df():
    """Simple tabular dataset with no featurized columns."""
    np.random.seed(42)
    return pd.DataFrame({
        'material_id': [f"MAT_{i:03d}" for i in range(10)],
        'band_gap': np.random.uniform(0.5, 5.0, 10),
        'energy': np.random.uniform(-8, -2, 10),
        'density': np.random.uniform(2, 8, 10),
    })


@pytest.fixture
def crystal_df():
    """Dataset with crystal structure features (v2.0 unified naming)."""
    np.random.seed(42)
    return pd.DataFrame({
        'formula': ['Fe2O3', 'SiO2'] * 5,
        'band_gap': np.random.uniform(0.5, 5.0, 10),
        'crystal_density': np.random.uniform(3, 8, 10),
        'crystal_lattice_volume': np.random.uniform(100, 500, 10),
        'crystal_lattice_a': np.random.uniform(3, 10, 10),
        'crystal_lattice_b': np.random.uniform(3, 10, 10),
        'crystal_lattice_c': np.random.uniform(3, 10, 10),
    })


@pytest.fixture
def molecular_df():
    """Dataset with molecular structure features."""
    np.random.seed(42)
    return pd.DataFrame({
        'molecule_name': ['water', 'methane', 'ethanol'] * 4,
        'target': np.random.uniform(0, 1, 12),
        'mol_n_atoms': [3, 5, 9] * 4,
        'mol_n_bonds': [2, 4, 8] * 4,
        'mol_mean_atomic_mass': [18.0, 16.0, 46.0] * 4,
        'has_oh_group': [1, 0, 1] * 4,
        'has_nh_group': [0, 0, 1] * 4,
    })


@pytest.fixture
def mixed_df():
    """Dataset with crystal + molecular + Magpie features (v2.0 unified naming)."""
    np.random.seed(42)
    return pd.DataFrame({
        'formula': ['Fe2O3', 'SiO2'] * 5,
        'band_gap': np.random.uniform(0.5, 5.0, 10),
        'crystal_density': np.random.uniform(3, 8, 10),
        'crystal_lattice_a': np.random.uniform(3, 10, 10),
        'mol_n_atoms': np.random.randint(1, 50, 10),
        'mol_mean_atomic_mass': np.random.uniform(10, 100, 10),
        'has_oh_group': np.random.randint(0, 2, 10),
        'has_cooh_group': np.random.randint(0, 2, 10),
        'atomic_Number_mean': np.random.uniform(10, 40, 10),
        'avg_Electronegativity': np.random.uniform(1.0, 3.0, 10),
        'ThermalConductivity_mean': np.random.uniform(0.1, 5.0, 10),
    })


@pytest.fixture
def indexed_df():
    """Dataset with a meaningful string index."""
    return pd.DataFrame({
        'band_gap': [1.5, 2.3, 3.1],
        'energy': [-5.2, -4.1, -3.8],
    }, index=['MP-1', 'MP-2', 'MP-3']).rename_axis('material_id')


# =============================================================================
# Test Class 1: Feature Breakdown Detection
# =============================================================================

class TestFeatureBreakdown:
    """Test build_feature_breakdown for all feature category combinations."""

    def test_plain_dataset_shows_tabular(self, plain_df):
        """Plain dataset with no special features shows 'tabular features'."""
        result = DatasetExporter.build_feature_breakdown(plain_df)
        assert len(result) == 1
        assert "tabular" in result[0].lower()

    def test_crystal_features_detected(self, crystal_df):
        """Crystal features are counted correctly."""
        result = DatasetExporter.build_feature_breakdown(crystal_df)
        crystal_line = [r for r in result if "Crystal" in r]
        assert len(crystal_line) == 1
        assert "5" in crystal_line[0]  # 5 crystal_ columns

    def test_molecular_features_detected(self, molecular_df):
        """Molecular features are counted correctly."""
        result = DatasetExporter.build_feature_breakdown(molecular_df)
        mol_line = [r for r in result if "Molecular" in r]
        has_line = [r for r in result if "Functional" in r]
        assert len(mol_line) == 1
        assert "3" in mol_line[0]  # 3 mol_ columns
        assert len(has_line) == 1
        assert "2" in has_line[0]  # 2 has_ columns

    def test_mixed_features_all_categories(self, mixed_df):
        """Mixed dataset shows all feature categories."""
        result = DatasetExporter.build_feature_breakdown(mixed_df)
        categories = [r.strip().split(":")[0].replace("•", "").strip() for r in result]
        assert "Crystal" in categories
        assert "Molecular" in categories
        assert "Functional groups" in categories
        assert "Composition (Magpie)" in categories

    def test_magpie_prefix_variants(self):
        """All Magpie prefixes are detected."""
        df = pd.DataFrame({
            'band_gap': [1.0, 2.0],
            'atomic_Number_mean': [26.0, 14.0],
            'avg_Electronegativity': [2.2, 1.9],
            'ThermalConductivity_mean': [0.8, 1.2],
            'BulkModulus_mean': [100, 200],
            'ShearModulus_mean': [50, 80],
            'band_center_Fe': [-2.5, -3.1],
            'fractional_Fe': [0.4, 0.33],
            'weight_fraction_O': [0.6, 0.67],
        })
        result = DatasetExporter.build_feature_breakdown(df)
        magpie_line = [r for r in result if "Magpie" in r]
        assert len(magpie_line) == 1
        # 8 Magpie columns (band_gap is NOT a Magpie feature)
        assert "8" in magpie_line[0]

    def test_no_false_positives_for_partial_matches(self):
        """Column names that contain but don't start with prefixes are NOT counted."""
        df = pd.DataFrame({
            'my_crystal_data': [1, 2],    # contains 'crystal' but starts with 'my_'
            'my_molecular_weight': [3, 4], # contains 'mol' but starts with 'my_'
            'hash_value': [5, 6],          # contains 'has' but starts with 'hash'
        })
        result = DatasetExporter.build_feature_breakdown(df)
        assert len(result) == 1
        assert "tabular" in result[0].lower()

    def test_empty_dataframe_falls_back_to_tabular(self):
        """Empty DataFrame falls back gracefully."""
        df = pd.DataFrame()
        result = DatasetExporter.build_feature_breakdown(df)
        assert len(result) == 1
        assert "tabular" in result[0].lower()


# =============================================================================
# Test Class 2: Filename Suggestion
# =============================================================================

class TestSuggestFilename:
    """Test suggest_filename for various input scenarios."""

    def test_with_file_path(self):
        """Filename derived from original file path."""
        result = DatasetExporter.suggest_filename("/data/my_experiment.csv")
        assert result == "my_experiment_featurized.csv"

    def test_with_relative_path(self):
        """Filename derived from relative path."""
        result = DatasetExporter.suggest_filename("data/results.xlsx")
        assert result == "results_featurized.csv"

    def test_with_none(self):
        """Default filename when no path given."""
        result = DatasetExporter.suggest_filename(None)
        assert result == "autoaim_dataset.csv"

    def test_with_empty_string(self):
        """Empty string falls back to default filename (falsy check)."""
        result = DatasetExporter.suggest_filename("")
        assert result == "autoaim_dataset.csv"

    def test_strips_path_correctly(self, tmp_path):
        """Correctly strips directory and extension from various paths."""
        for filename in ["data.csv", "my data.xlsx", "results_final.json"]:
            path = str(tmp_path / filename)
            result = DatasetExporter.suggest_filename(path)
            expected = Path(filename).stem + "_featurized.csv"
            assert result == expected


# =============================================================================
# Test Class 3: CSV Export Core Logic
# =============================================================================

class TestExportToCsv:
    """Test export_to_csv with various DataFrame configurations."""

    def test_basic_export_no_index(self, plain_df, tmp_path):
        """Plain DataFrame exports without index."""
        output = tmp_path / "output.csv"
        rows, cols = DatasetExporter.export_to_csv(plain_df, str(output))
        assert rows == 10
        assert cols == 4
        assert output.exists()
        # Read back and verify
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 10
        assert list(df_read.columns) == list(plain_df.columns)

    def test_export_with_named_index(self, indexed_df, tmp_path):
        """DataFrame with named index exports with index."""
        output = tmp_path / "indexed.csv"
        rows, cols = DatasetExporter.export_to_csv(indexed_df, str(output))
        assert rows == 3
        assert cols == 2
        df_read = pd.read_csv(str(output))
        assert 'material_id' in df_read.columns
        assert set(df_read['material_id']) == {'MP-1', 'MP-2', 'MP-3'}

    def test_export_with_string_index_no_name(self, tmp_path):
        """DataFrame with string index but no name exports with index."""
        df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]}, index=['x', 'y'])
        output = tmp_path / "str_index.csv"
        DatasetExporter.export_to_csv(df, str(output))
        df_read = pd.read_csv(str(output))
        # Index should be present as first column
        assert len(df_read.columns) == 3  # index + a + b

    def test_export_numeric_index_no_name(self, plain_df, tmp_path):
        """DataFrame with default numeric index exports without index."""
        output = tmp_path / "no_index.csv"
        DatasetExporter.export_to_csv(plain_df, str(output))
        df_read = pd.read_csv(str(output))
        assert 'Unnamed' not in str(df_read.columns)
        # Should NOT have an extra index column
        assert len(df_read.columns) == len(plain_df.columns)

    def test_crystal_features_round_trip(self, crystal_df, tmp_path):
        """Crystal features survive round-trip export → read."""
        output = tmp_path / "crystal.csv"
        DatasetExporter.export_to_csv(crystal_df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 10
        crystal_cols = [c for c in df_read.columns if c.startswith('crystal_')]
        assert len(crystal_cols) == 5

    def test_molecular_features_round_trip(self, molecular_df, tmp_path):
        """Molecular features survive round-trip export → read."""
        output = tmp_path / "molecular.csv"
        DatasetExporter.export_to_csv(molecular_df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 12
        mol_cols = [c for c in df_read.columns if c.startswith('mol_')]
        has_cols = [c for c in df_read.columns if c.startswith('has_')]
        assert len(mol_cols) == 3
        assert len(has_cols) == 2

    def test_mixed_features_round_trip(self, mixed_df, tmp_path):
        """Mixed feature types all survive round-trip."""
        output = tmp_path / "mixed.csv"
        DatasetExporter.export_to_csv(mixed_df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 10
        assert len(df_read.columns) == len(mixed_df.columns)
        for col in mixed_df.columns:
            assert col in df_read.columns

    def test_empty_dataframe_raises(self, tmp_path):
        """Empty DataFrame raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            DatasetExporter.export_to_csv(pd.DataFrame(), str(tmp_path / "empty.csv"))

    def test_none_dataframe_raises(self, tmp_path):
        """None DataFrame raises ValueError or AttributeError."""
        with pytest.raises((ValueError, AttributeError)):
            DatasetExporter.export_to_csv(None, str(tmp_path / "none.csv"))

    def test_nonexistent_directory_raises(self, plain_df):
        """Nonexistent output directory raises ValueError."""
        with pytest.raises(ValueError, match="Directory"):
            DatasetExporter.export_to_csv(plain_df, "/nonexistent/path/file.csv")

    def test_overwrite_existing_file(self, plain_df, tmp_path):
        """Export can overwrite an existing file."""
        output = tmp_path / "existing.csv"
        output.write_text("old content")
        DatasetExporter.export_to_csv(plain_df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 10


# =============================================================================
# Test Class 4: Export Summary
# =============================================================================

class TestExportSummary:
    """Test get_export_summary for complete export reporting."""

    def test_summary_structure(self, plain_df, tmp_path):
        """Summary contains all expected keys."""
        output = str(tmp_path / "summary.csv")
        summary = DatasetExporter.get_export_summary(plain_df, output)
        assert set(summary.keys()) == {"file", "rows", "columns", "breakdown"}
        assert summary["rows"] == "10"
        assert summary["columns"] == "4"
        assert summary["file"] == output

    def test_summary_breakdown_content(self, crystal_df, tmp_path):
        """Summary breakdown reflects actual features."""
        output = str(tmp_path / "crystal_summary.csv")
        summary = DatasetExporter.get_export_summary(crystal_df, output)
        assert "Crystal: 5" in summary["breakdown"]

    def test_summary_for_mixed(self, mixed_df, tmp_path):
        """Summary for mixed dataset includes all categories."""
        output = str(tmp_path / "mixed_summary.csv")
        summary = DatasetExporter.get_export_summary(mixed_df, output)
        assert "Crystal" in summary["breakdown"]
        assert "Molecular" in summary["breakdown"]
        assert "Functional groups" in summary["breakdown"]
        assert "Composition (Magpie)" in summary["breakdown"]

    def test_summary_numeric_values(self, molecular_df, tmp_path):
        """Summary numeric values are correct."""
        output = str(tmp_path / "mol_summary.csv")
        summary = DatasetExporter.get_export_summary(molecular_df, output)
        assert summary["rows"] == "12"
        assert summary["columns"] == "7"


# =============================================================================
# Test Class 5: Edge Cases & Defensive Behavior
# =============================================================================

class TestEdgeCases:
    """Test edge cases and defensive behavior."""

    def test_single_row_export(self, tmp_path):
        """Single-row DataFrame exports correctly."""
        df = pd.DataFrame({'a': [1], 'b': [2]})
        output = tmp_path / "single.csv"
        rows, cols = DatasetExporter.export_to_csv(df, str(output))
        assert rows == 1
        assert cols == 2
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 1

    def test_single_column_export(self, tmp_path):
        """Single-column DataFrame exports correctly."""
        df = pd.DataFrame({'feature': [1, 2, 3, 4, 5]})
        output = tmp_path / "single_col.csv"
        rows, cols = DatasetExporter.export_to_csv(df, str(output))
        assert rows == 5
        assert cols == 1

    def test_dataframe_with_nan_values(self, tmp_path):
        """DataFrame with NaN values exports correctly."""
        df = pd.DataFrame({
            'a': [1.0, np.nan, 3.0],
            'b': [np.nan, 2.0, 3.0],
        })
        output = tmp_path / "with_nan.csv"
        DatasetExporter.export_to_csv(df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 3
        assert df_read['a'].isna().sum() == 1

    def test_dataframe_with_special_characters_in_values(self, tmp_path):
        """DataFrame with special characters exports correctly."""
        df = pd.DataFrame({
            'name': ['Fe2O3', 'SiO2,Al2O3', 'CaTiO3\nMgO'],
            'value': [1.5, 2.3, 3.1],
        })
        output = tmp_path / "special.csv"
        DatasetExporter.export_to_csv(df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read) == 3

    def test_very_long_column_names(self, tmp_path):
        """DataFrame with very long column names exports correctly."""
        df = pd.DataFrame({
            'this_is_a_very_long_column_name_that_exceeds_typical_limits': [1, 2, 3],
            'another_extremely_long_column_name_for_testing_purposes': [4, 5, 6],
        })
        output = tmp_path / "long_names.csv"
        DatasetExporter.export_to_csv(df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read.columns) == 2

    def test_unicode_in_column_names_and_values(self, tmp_path):
        """Unicode content in columns and values exports correctly."""
        df = pd.DataFrame({
            'α_parameter': [0.1, 0.2, 0.3],
            'β_energy': [1.5, 2.5, 3.5],
        })
        output = tmp_path / "unicode.csv"
        DatasetExporter.export_to_csv(df, str(output))
        df_read = pd.read_csv(str(output))
        assert len(df_read.columns) == 2
        assert len(df_read) == 3

    def test_datetime_index_exports_with_name(self, tmp_path):
        """DataFrame with DatetimeIndex and name exports with index."""
        dates = pd.date_range('2024-01-01', periods=3, freq='D')
        df = pd.DataFrame({'value': [1, 2, 3]}, index=dates)
        df.index.name = 'date'
        output = tmp_path / "datetime_index.csv"
        DatasetExporter.export_to_csv(df, str(output))
        df_read = pd.read_csv(str(output))
        assert 'date' in df_read.columns
        assert len(df_read) == 3

    def test_all_feature_prefixes_present_in_constants(self):
        """Verify that the class constants cover expected prefixes."""
        assert 'crystal_' in DatasetExporter.CRYSTAL_PREFIXES
        assert 'mol_' in DatasetExporter.MOLECULAR_PREFIXES
        assert 'has_' in DatasetExporter.FUNCTIONAL_GROUP_PREFIXES
        assert 'atomic_' in DatasetExporter.MAGPIE_PREFIXES
        assert 'avg_' in DatasetExporter.MAGPIE_PREFIXES
        assert 'Number' in DatasetExporter.MAGPIE_PREFIXES
        assert 'Electronegativity' in DatasetExporter.MAGPIE_PREFIXES
        assert 'ThermalConductivity' in DatasetExporter.MAGPIE_PREFIXES
        assert 'BulkModulus' in DatasetExporter.MAGPIE_PREFIXES
        assert 'ShearModulus' in DatasetExporter.MAGPIE_PREFIXES
        assert 'band_center_' in DatasetExporter.MAGPIE_PREFIXES
        assert 'fractional_' in DatasetExporter.MAGPIE_PREFIXES
        assert 'weight_fraction_' in DatasetExporter.MAGPIE_PREFIXES
