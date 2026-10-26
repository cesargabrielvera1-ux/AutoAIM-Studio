"""Tests for crystal_feature_extractor module (AutoAIM v2.0)."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.crystal_feature_extractor import CrystalFeatureExtractor, LinkReport

try:
    import pymatgen
    HAS_Pymatgen = True
except ImportError:
    HAS_Pymatgen = False


class TestLinkReport:
    """Test the LinkReport dataclass."""

    def test_basic_creation(self):
        r = LinkReport(
            id_column="material_id",
            cif_folder="/test/cifs",
            n_rows_total=10,
            n_matched=8,
            n_missing=2,
            n_duplicate_ids=0,
        )
        assert r.id_column == "material_id"
        assert r.match_rate == 0.8

    def test_match_rate_zero_rows(self):
        r = LinkReport(
            id_column="id", cif_folder="/test",
            n_rows_total=0, n_matched=0, n_missing=0, n_duplicate_ids=0,
        )
        assert r.match_rate == 0.0

    def test_to_text_contains_key_info(self):
        r = LinkReport(
            id_column="mp_id", cif_folder="/cifs",
            n_rows_total=5, n_matched=4, n_missing=1, n_duplicate_ids=0,
            crystal_features_added=["crystal_density", "crystal_lattice_volume"],
            composition_features_replaced=True,
        )
        text = r.to_text()
        assert "mp_id" in text
        assert "/cifs" in text
        assert "4" in text
        assert "2" in text  # 2 crystal features
        assert "True" in text  # composition_features_replaced


class TestCrystalFeatureExtractor:
    """Test the CrystalFeatureExtractor class."""

    def test_init(self):
        ex = CrystalFeatureExtractor()
        assert not ex.is_linked()
        assert ex.link_report is None
        assert ex.crystal_features_df is None

    def test_detect_id_candidates(self):
        ex = CrystalFeatureExtractor()
        df = pd.DataFrame({
            'material_id': ['mp-1', 'mp-2', 'mp-3'],
            'formula': ['TiO2', 'Fe2O3', 'SiO2'],
            'band_gap': [3.0, 2.1, 4.5],
        })
        candidates = ex.detect_id_candidates(df)
        # material_id should be top candidate (fully unique)
        assert len(candidates) >= 2
        assert candidates[0][0] == 'material_id'  # Most unique

    def test_detect_id_candidates_excludes_target_like(self):
        ex = CrystalFeatureExtractor()
        df = pd.DataFrame({
            'target': [1, 2, 3],
            'label': ['a', 'b', 'c'],
            'id_col': ['x', 'y', 'z'],
        })
        candidates = ex.detect_id_candidates(df)
        names = [c[0] for c in candidates]
        assert 'target' not in names
        assert 'label' not in names
        assert 'id_col' in names

    def test_detect_id_candidates_low_uniqueness_filtered(self):
        ex = CrystalFeatureExtractor()
        df = pd.DataFrame({
            'almost_constant': ['A', 'A', 'A', 'B'],  # 50% unique, borderline
            'band_gap': [1.0, 2.0, 3.0, 4.0],
        })
        candidates = ex.detect_id_candidates(df)
        names = [c[0] for c in candidates]
        # band_gap should be included (fully unique)
        assert 'band_gap' in names

    def test_check_cif_availability_missing_folder(self):
        ex = CrystalFeatureExtractor()
        df = pd.DataFrame({'id': ['mp-1', 'mp-2']})
        result = ex.check_cif_availability(df, 'id', '/nonexistent/folder')
        assert 'error' in result
        assert result['matched'] == []
        assert len(result['missing']) == 2

    def test_check_cif_availability_with_matches(self, tmp_path):
        ex = CrystalFeatureExtractor()
        # Create fake CIF files
        (tmp_path / "mp-1.cif").write_text("fake cif")
        (tmp_path / "mp-3.cif").write_text("fake cif")

        df = pd.DataFrame({'id': ['mp-1', 'mp-2', 'mp-3']})
        result = ex.check_cif_availability(df, 'id', str(tmp_path))

        assert len(result['matched']) == 2
        assert 'mp-1' in result['matched']
        assert 'mp-3' in result['matched']
        assert len(result['missing']) == 1
        assert 'mp-2' in result['missing']
        assert result['n_cif_files'] == 2

    def test_check_cif_availability_detects_duplicates(self, tmp_path):
        ex = CrystalFeatureExtractor()
        (tmp_path / "TiO2.cif").write_text("fake cif")

        df = pd.DataFrame({
            'formula': ['TiO2', 'TiO2', 'Fe2O3'],
            'bg': [3.0, 3.2, 2.1],
        })
        result = ex.check_cif_availability(df, 'formula', str(tmp_path))

        assert len(result['duplicate_ids']) == 1
        assert 'TiO2' in result['duplicate_ids']
        assert len(result['matched']) == 2  # Two TiO2 rows match
        assert len(result['missing']) == 1  # Fe2O3 missing

    def test_link_structures_no_pymatgen(self):
        """When pymatgen is not available, should return original df with error."""
        ex = CrystalFeatureExtractor()
        ex._has_pymatgen = False
        df = pd.DataFrame({'id': ['mp-1'], 'bg': [3.0]})
        df_out, report = ex.link_structures(df, 'id', '/fake/folder')
        assert len(df_out) == 1
        assert len(report.errors) > 0

    def test_link_structures_missing_id_column(self):
        ex = CrystalFeatureExtractor()
        df = pd.DataFrame({'bg': [3.0]})
        df_out, report = ex.link_structures(df, 'nonexistent', '/fake')
        assert len(df_out) == 1
        assert any("not found" in e for e in report.errors)

    def test_link_structures_missing_folder(self):
        ex = CrystalFeatureExtractor()
        df = pd.DataFrame({'id': ['mp-1'], 'bg': [3.0]})
        df_out, report = ex.link_structures(df, 'id', '/nonexistent')
        assert len(df_out) == 1
        assert any("does not exist" in e for e in report.errors)

    @pytest.mark.skipif(not HAS_Pymatgen, reason="pymatgen not installed")
    def test_crystal_feature_names_defined(self):
        """Verify that the expected crystal feature names are defined."""
        ex = CrystalFeatureExtractor()
        names = ex.crystal_feature_names
        # v2.0 UNIFIED: All 149 features from CrystalStructureFeaturizer with crystal_ prefix
        assert 'crystal_density' in names
        assert 'crystal_lattice_volume' in names
        assert 'crystal_lattice_a' in names
        assert 'crystal_lattice_b' in names
        assert 'crystal_lattice_c' in names
        assert 'crystal_n_elements' in names
        assert 'crystal_n_sites' in names
        assert 'crystal_spacegroup_number' in names
        assert 'crystal_volume_per_atom' in names
        assert 'crystal_packing_fraction' in names
        assert 'crystal_mean_atomic_number' in names
        assert 'crystal_mean_electronegativity' in names
        assert 'crystal_mean_ionization_energy' in names
        assert 'crystal_mean_group' in names
        assert 'crystal_mean_period' in names
        assert 'crystal_mean_melting_point' in names
        assert 'crystal_mean_thermal_conductivity' in names
        # Magpie features (prefixed with crystal_magpie_comp_)
        assert 'crystal_magpie_comp_atomic_number_mean' in names
        assert 'crystal_magpie_comp_atomic_number_std' in names
        # Advanced structural features
        assert 'crystal_mean_bond_length' in names
        assert 'crystal_mean_coordination_number' in names
        assert 'crystal_rdf_peak_1' in names
        assert 'crystal_structure_complexity_index' in names
        assert 'crystal_lattice_anisotropy' in names
        # Total: 149 features (10 lattice + 13 comp + 3 struct + 2 derived + 106 magpie + 15 advanced)
        assert len(names) == 149

    def test_clear(self):
        ex = CrystalFeatureExtractor()
        ex._link_report = LinkReport('id', '/cifs', 5, 5, 0, 0)
        ex._id_column = 'test'
        ex.clear()
        assert ex.link_report is None
        assert not ex.is_linked()
        assert not hasattr(ex, '_id_column')

    def test_get_id_column(self):
        ex = CrystalFeatureExtractor()
        assert ex.get_id_column() is None
        ex._id_column = 'mp_id'
        assert ex.get_id_column() == 'mp_id'
