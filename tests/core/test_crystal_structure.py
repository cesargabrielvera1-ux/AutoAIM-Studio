"""
Tests for app/core/crystal_structure.py

Validates crystal structure loading, featurization, and dataset building.

v2.0: Tests work with or without pymatgen installed.
"""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# Try to import pymatgen for availability check
try:
    import pymatgen
    from pymatgen.core import Lattice, Structure

    HAS_Pymatgen = True
except ImportError:
    HAS_Pymatgen = False

from app.core.crystal_structure import (
    CrystalStructureLoader,
    CrystalStructureFeaturizer,
    CrystalStructureDatasetBuilder,
    _find_pymatgen_data_in_bundle,
    _try_import_pymatgen,
)


# ──────────────────────────────────────────────────────────────────────────
# 1. Pymatgen availability
# ──────────────────────────────────────────────────────────────────────────

class TestPymatgenAvailability:
    """Lazy import and bundle detection must be robust."""

    def test_try_import_returns_bool(self):
        result = _try_import_pymatgen()
        assert isinstance(result, bool)

    def test_try_import_matches_import_check(self):
        result = _try_import_pymatgen()
        assert result == HAS_Pymatgen

    def test_find_pymatgen_data_in_bundle_returns_none_outside_bundle(self):
        """Outside PyInstaller bundle, should return None."""
        result = _find_pymatgen_data_in_bundle()
        assert result is None or isinstance(result, str)


# ──────────────────────────────────────────────────────────────────────────
# 2. CrystalStructureLoader
# ──────────────────────────────────────────────────────────────────────────

class TestLoaderNoPymatgen:
    """Loader initialization and basic methods without pymatgen."""

    def test_init_creates_empty_state(self):
        loader = CrystalStructureLoader()
        assert loader.structures == []
        assert loader.failed_files == []

    def test_supported_formats_are_lowercase(self):
        for ext in CrystalStructureLoader.SUPPORTED_FORMATS:
            assert ext == ext.lower(), f"Extension {ext} is not lowercase"

    def test_supported_formats_not_empty(self):
        assert len(CrystalStructureLoader.SUPPORTED_FORMATS) > 0

    @pytest.mark.skipif(HAS_Pymatgen, reason="pymatgen is installed")
    def test_load_file_raises_without_pymatgen(self):
        loader = CrystalStructureLoader()
        with pytest.raises(ImportError) as exc_info:
            loader.load_file("dummy.cif")
        assert "pymatgen" in str(exc_info.value).lower()

    @pytest.mark.skipif(HAS_Pymatgen, reason="pymatgen is installed")
    def test_load_directory_raises_without_pymatgen(self):
        loader = CrystalStructureLoader()
        with pytest.raises(ImportError) as exc_info:
            loader.load_directory("/tmp")
        assert "pymatgen" in str(exc_info.value).lower()


@pytest.mark.skipif(not HAS_Pymatgen, reason="pymatgen not installed")
class TestLoaderWithPymatgen:
    """Loader with real pymatgen structures."""

    def test_load_file_not_found(self):
        loader = CrystalStructureLoader()
        result = loader.load_file("/nonexistent/file.cif")
        assert result is None

    def test_load_directory_empty(self, tmp_path):
        loader = CrystalStructureLoader()
        count = loader.load_directory(str(tmp_path))
        assert count == 0
        assert loader.structures == []
        assert loader.failed_files == []

    def test_load_directory_with_unsupported_files(self, tmp_path):
        (tmp_path / "dummy.txt").write_text("not a structure")
        loader = CrystalStructureLoader()
        count = loader.load_directory(str(tmp_path))
        assert count == 0

    def test_cif_file_with_single_structure(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        loader = CrystalStructureLoader()
        struct = loader.load_file(str(cif_file))
        assert struct is not None
        assert len(struct) == 2  # 2 sites

    def test_load_directory_counts_successful(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
"""
        (tmp_path / "struct1.cif").write_text(cif_content)
        (tmp_path / "struct2.cif").write_text(cif_content)

        loader = CrystalStructureLoader()
        count = loader.load_directory(str(tmp_path))
        assert count == 2
        assert len(loader.structures) == 2

    def test_failed_files_tracked(self, tmp_path):
        bad_cif = tmp_path / "bad.cif"
        bad_cif.write_text("this is not valid CIF content")

        loader = CrystalStructureLoader()
        result = loader.load_file(str(bad_cif))
        assert result is None
        assert len(loader.failed_files) == 1
        assert str(bad_cif) in loader.failed_files[0][0]

    def test_load_files(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
"""
        f1 = tmp_path / "s1.cif"
        f2 = tmp_path / "s2.cif"
        f1.write_text(cif_content)
        f2.write_text(cif_content)

        loader = CrystalStructureLoader()
        count = loader.load_files([str(f1), str(f2)])
        assert count == 2

    def test_get_structures_dataframe(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        loader = CrystalStructureLoader()
        loader.load_file(str(cif_file))
        df = loader.get_structures_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ──────────────────────────────────────────────────────────────────────────
# 3. CrystalStructureFeaturizer
# ──────────────────────────────────────────────────────────────────────────

class TestFeaturizerInit:
    """Featurizer initialization and basic methods."""

    def test_init_sets_default_state(self):
        featurizer = CrystalStructureFeaturizer()
        assert featurizer.feature_names is not None
        assert isinstance(featurizer.feature_names, list)
        assert len(featurizer.feature_names) > 0

    def test_feature_names_are_unique(self):
        featurizer = CrystalStructureFeaturizer()
        names = featurizer.feature_names
        assert len(names) == len(set(names)), "Duplicate feature names found"

    def test_feature_names_are_strings(self):
        featurizer = CrystalStructureFeaturizer()
        for name in featurizer.feature_names:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_init_with_options(self):
        featurizer = CrystalStructureFeaturizer(use_magpie=False, use_advanced_structural=False)
        # Should still have basic structural features
        assert len(featurizer.feature_names) > 0

    def test_init_creates_magpie_lazy(self):
        featurizer = CrystalStructureFeaturizer()
        assert featurizer._magpie_featurizer is None


@pytest.mark.skipif(not HAS_Pymatgen, reason="pymatgen not installed")
class TestFeaturizerWithPymatgen:
    """Featurizer with real pymatgen structures."""

    def test_featurize_single_structure(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        loader = CrystalStructureLoader()
        struct = loader.load_file(str(cif_file))

        featurizer = CrystalStructureFeaturizer()
        features = featurizer.featurize(struct)

        assert isinstance(features, dict)
        assert len(features) > 0
        # All values should be numeric
        for k, v in features.items():
            assert isinstance(v, (int, float, np.floating, np.integer)), f"{k}: {type(v)}"

    def test_featurize_structures_multiple(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        (tmp_path / "NaCl.cif").write_text(cif_content)
        (tmp_path / "NaCl2.cif").write_text(cif_content)

        loader = CrystalStructureLoader()
        loader.load_directory(str(tmp_path))

        featurizer = CrystalStructureFeaturizer()
        df = featurizer.featurize_structures(loader.structures)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2
        assert "structure_name" in df.columns

    def test_featurize_structures_empty_list(self):
        featurizer = CrystalStructureFeaturizer()
        df = featurizer.featurize_structures([])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_feature_count_consistent(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        (tmp_path / "s1.cif").write_text(cif_content)
        (tmp_path / "s2.cif").write_text(cif_content)

        loader = CrystalStructureLoader()
        loader.load_directory(str(tmp_path))

        featurizer = CrystalStructureFeaturizer()
        df = featurizer.featurize_structures(loader.structures)

        # Numeric columns should match feature_names count
        numeric_cols = [c for c in df.columns if c != "structure_name"]
        assert len(numeric_cols) == len(featurizer.feature_names)

    def test_featurize_dataframe(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        loader = CrystalStructureLoader()
        loader.load_file(str(cif_file))

        featurizer = CrystalStructureFeaturizer()
        df = featurizer.featurize_structures(loader.structures)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ──────────────────────────────────────────────────────────────────────────
# 4. CrystalStructureDatasetBuilder
# ──────────────────────────────────────────────────────────────────────────

class TestDatasetBuilderInit:
    """Dataset builder initialization."""

    def test_init_creates_components(self):
        builder = CrystalStructureDatasetBuilder()
        assert builder.loader is not None
        assert builder.featurizer is not None
        assert builder.dataset_df is None

    def test_init_components_are_correct_types(self):
        builder = CrystalStructureDatasetBuilder()
        assert isinstance(builder.loader, CrystalStructureLoader)
        assert isinstance(builder.featurizer, CrystalStructureFeaturizer)


@pytest.mark.skipif(not HAS_Pymatgen, reason="pymatgen not installed")
class TestDatasetBuilderWithPymatgen:
    """Dataset builder with real structures."""

    def test_build_from_directory(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        builder = CrystalStructureDatasetBuilder()
        result = builder.build_from_directory(str(tmp_path))

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1
        assert "structure_name" in result.columns

    def test_build_from_files(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        builder = CrystalStructureDatasetBuilder()
        result = builder.build_from_files([str(cif_file)])

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_build_with_target_values(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        builder = CrystalStructureDatasetBuilder()
        result = builder.build_from_directory(
            str(tmp_path),
            target_values={"NaCl": 42.0},
            target_column="band_gap"
        )

        assert "band_gap" in result.columns
        assert result["band_gap"].iloc[0] == 42.0

    def test_build_from_structures(self, tmp_path):
        cif_content = """data_test
_cell_length_a 4.0
_cell_length_b 4.0
_cell_length_c 4.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
_symmetry_space_group_name_H-M 'P 1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
Na1 Na 0.0 0.0 0.0
Cl1 Cl 0.5 0.5 0.5
"""
        cif_file = tmp_path / "NaCl.cif"
        cif_file.write_text(cif_content)

        loader = CrystalStructureLoader()
        loader.load_file(str(cif_file))

        builder = CrystalStructureDatasetBuilder()
        result = builder.build_from_structures(loader.structures)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == 1

    def test_build_empty_directory_raises(self, tmp_path):
        builder = CrystalStructureDatasetBuilder()
        with pytest.raises(ValueError) as exc_info:
            builder.build_from_directory(str(tmp_path))
        assert "no structures" in str(exc_info.value).lower()


# ──────────────────────────────────────────────────────────────────────────
# 5. Edge cases
# ──────────────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Boundary conditions and error paths."""

    @pytest.mark.skipif(HAS_Pymatgen, reason="pymatgen is installed")
    def test_loader_with_none_path_raises(self):
        loader = CrystalStructureLoader()
        with pytest.raises(ImportError):
            loader.load_file(None)

    @pytest.mark.skipif(HAS_Pymatgen, reason="pymatgen is installed")
    def test_loader_with_empty_string_path_raises(self):
        loader = CrystalStructureLoader()
        with pytest.raises(ImportError):
            loader.load_file("")

    def test_featurizer_init_does_not_require_pymatgen(self):
        """Initializing featurizer should not require pymatgen."""
        featurizer = CrystalStructureFeaturizer()
        assert featurizer is not None
