"""
Comprehensive tests for MoleculeFeaturizer.

Three levels validated:
  Level 1 — Composition (element counts, mass, ratios)
  Level 2 — Geometry 3D (COM, Rg, bbox, inertia, shape)
  Level 3 — Topology (bonds, angles, rings, functional groups)

All tests use Python pure (no RDKit, no external deps).
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.core.molecule_featurizer import (
    MoleculeFeaturizer,
    list_molecule_feature_names,
    ATOMIC_MASSES,
    COVALENT_RADII,
)


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────

@pytest.fixture
def featurizer():
    return MoleculeFeaturizer()


def _write_xyz(tmp_path: Path, lines: list) -> Path:
    """Helper: write an .xyz file and return its path."""
    p = tmp_path / "test_mol.xyz"
    p.write_text("\n".join(lines))
    return p


def _h2_xyz(tmp_path: Path) -> Path:
    """H2 molecule — minimal valid .xyz"""
    return _write_xyz(tmp_path, [
        "2",
        "H2 molecule",
        "H  0.0000  0.0000  0.0000",
        "H  0.7400  0.0000  0.0000",
    ])


def _co2_xyz(tmp_path: Path) -> Path:
    """CO2 (O=C=O) — linear molecule with double bonds"""
    return _write_xyz(tmp_path, [
        "3",
        "CO2 linear",
        "C  0.0000  0.0000  0.0000",
        "O  1.1600  0.0000  0.0000",
        "O -1.1600  0.0000  0.0000",
    ])


def _water_xyz(tmp_path: Path) -> Path:
    """Water — bent molecule with OH groups"""
    return _write_xyz(tmp_path, [
        "3",
        "Water",
        "O  0.0000   0.0000  0.0000",
        "H  0.9600   0.0000  0.0000",
        "H -0.2400   0.9300  0.0000",
    ])


def _benzene_xyz(tmp_path: Path) -> Path:
    """Benzene — planar ring with 6 C and 6 H"""
    # C-C bond ~1.40 Å, C-H bond ~1.08 Å, planar ring
    r_c = 1.40
    r_h = 2.48  # distance from center to H
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    lines = ["12", "Benzene C6H6"]
    for a in angles:
        cx = r_c * np.cos(a)
        cy = r_c * np.sin(a)
        lines.append(f"C  {cx:.4f}  {cy:.4f}  0.0000")
        hx = r_h * np.cos(a)
        hy = r_h * np.sin(a)
        lines.append(f"H  {hx:.4f}  {hy:.4f}  0.0000")
    return _write_xyz(tmp_path, lines)


def _ethanol_xyz(tmp_path: Path) -> Path:
    """Ethanol CH3-CH2-OH — has OH functional group"""
    return _write_xyz(tmp_path, [
        "9",
        "Ethanol",
        "C  -1.2600   0.0000  0.0000",   # C1 (methyl)
        "H  -1.2600   0.3600  1.0300",
        "H  -1.2600   0.3600 -1.0300",
        "H  -2.3100   0.0000  0.0000",
        "C   0.0000   0.0000  0.0000",   # C2 (methylene)
        "H   0.3700   1.0300  0.0000",
        "H   0.3700  -0.5200  0.8900",
        "O   0.7000  -0.7000  0.0000",   # Oxygen
        "H   1.6600  -0.5000  0.0000",   # Hydroxyl H
    ])


def _methane_xyz(tmp_path: Path) -> Path:
    """Methane — tetrahedral, saturated"""
    a = 0.629  # half edge of tetrahedron inscribed in unit sphere
    return _write_xyz(tmp_path, [
        "5",
        "Methane CH4",
        "C  0.0000   0.0000   0.0000",
        f"H  {a:.4f}   {a:.4f}   {a:.4f}",
        f"H  {a:.4f}  -{a:.4f}  -{a:.4f}",
        f"H -{a:.4f}   {a:.4f}  -{a:.4f}",
        f"H -{a:.4f}  -{a:.4f}   {a:.4f}",
    ])


# ──────────────────────────────────────────────────────────────────
# Level 1 — Composition
# ──────────────────────────────────────────────────────────────────

class TestComposition:
    """Level 1: element counts, mass, ratios, diversity."""

    def test_h2_counts(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_h2_xyz(tmp_path))
        assert df['mol_n_atoms'].iloc[0] == 2
        assert df['mol_n_hydrogens'].iloc[0] == 2
        assert df['mol_n_heavy_atoms'].iloc[0] == 0
        assert df['mol_n_carbons'].iloc[0] == 0

    def test_co2_counts(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_co2_xyz(tmp_path))
        assert df['mol_n_atoms'].iloc[0] == 3
        assert df['mol_n_carbons'].iloc[0] == 1
        assert df['mol_n_oxygens'].iloc[0] == 2
        assert df['mol_n_hydrogens'].iloc[0] == 0

    def test_water_counts(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        assert df['mol_n_atoms'].iloc[0] == 3
        assert df['mol_n_oxygens'].iloc[0] == 1
        assert df['mol_n_hydrogens'].iloc[0] == 2

    def test_benzene_counts(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_benzene_xyz(tmp_path))
        assert df['mol_n_atoms'].iloc[0] == 12
        assert df['mol_n_carbons'].iloc[0] == 6
        assert df['mol_n_hydrogens'].iloc[0] == 6

    def test_molecular_mass(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_co2_xyz(tmp_path))
        expected_mass = ATOMIC_MASSES['C'] + 2 * ATOMIC_MASSES['O']
        assert abs(df['mol_mass'].iloc[0] - expected_mass) < 0.1

    def test_hc_ratio(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_ethanol_xyz(tmp_path))
        # C2H6O → H/C = 6/2 = 3
        assert abs(df['mol_hc_ratio'].iloc[0] - 3.0) < 0.1

    def test_saturation(self, featurizer, tmp_path):
        # Methane: CH4 → fully saturated → saturation_index ≈ 4/(2*1+2) = 1
        df_sat = featurizer.featurize_file(_methane_xyz(tmp_path))
        assert df_sat['mol_saturation_index'].iloc[0] >= 1.0

        # CO2: no H → unsaturated
        df_unsat = featurizer.featurize_file(_co2_xyz(tmp_path))
        assert df_unsat['mol_saturation_index'].iloc[0] == 0.0

    def test_elemental_diversity(self, featurizer, tmp_path):
        # H2: only one element → diversity = 0
        df_h2 = featurizer.featurize_file(_h2_xyz(tmp_path))
        assert df_h2['mol_elemental_diversity'].iloc[0] == pytest.approx(0.0, abs=1e-6)

        # Ethanol: C, H, O → diversity > 0
        df_et = featurizer.featurize_file(_ethanol_xyz(tmp_path))
        assert df_et['mol_elemental_diversity'].iloc[0] > 0.5

    def test_no_metals(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        assert df['mol_n_metals'].iloc[0] == 0


# ──────────────────────────────────────────────────────────────────
# Level 2 — Geometry 3D
# ──────────────────────────────────────────────────────────────────

class TestGeometry:
    """Level 2: COM, Rg, bbox, inertia, shape descriptors."""

    def test_radius_of_gyration_positive(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_benzene_xyz(tmp_path))
        assert df['mol_radius_gyration'].iloc[0] > 0

    def test_bbox_positive(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_co2_xyz(tmp_path))
        assert df['mol_bbox_x'].iloc[0] > 0  # O=C=O is ~2.32 Å wide
        assert df['mol_bbox_volume'].iloc[0] > 0

    def test_linear_molecule_asphericity(self, featurizer, tmp_path):
        """CO2 is linear → high asphericity (rod-like)."""
        df = featurizer.featurize_file(_co2_xyz(tmp_path))
        assert df['mol_asphericity'].iloc[0] > 0.5

    def test_spherical_molecule_low_asphericity(self, featurizer, tmp_path):
        """Methane is roughly spherical → low asphericity."""
        df = featurizer.featurize_file(_methane_xyz(tmp_path))
        assert df['mol_asphericity'].iloc[0] < 1.0

    def test_inertia_eigenvalues_ordered(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        i1 = df['mol_inertia_1'].iloc[0]
        i2 = df['mol_inertia_2'].iloc[0]
        i3 = df['mol_inertia_3'].iloc[0]
        assert i1 >= i2 >= i3

    def test_pairwise_distances(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_h2_xyz(tmp_path))
        # H-H bond length is 0.74 Å
        assert abs(df['mol_mean_pairwise_dist'].iloc[0] - 0.74) < 0.1
        assert abs(df['mol_max_pairwise_dist'].iloc[0] - 0.74) < 0.1

    def test_density_atoms(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        assert df['mol_density_atoms_per_a3'].iloc[0] > 0

    def test_com_reasonable(self, featurizer, tmp_path):
        """For H2 symmetric at origin, COM should be near origin."""
        df = featurizer.featurize_file(_h2_xyz(tmp_path))
        assert abs(df['mol_com_x'].iloc[0]) < 0.5
        assert abs(df['mol_com_y'].iloc[0]) < 0.01
        assert abs(df['mol_com_z'].iloc[0]) < 0.01


# ──────────────────────────────────────────────────────────────────
# Level 3 — Topology
# ──────────────────────────────────────────────────────────────────

class TestTopology:
    """Level 3: bonds, angles, rings, functional groups."""

    def test_h2_single_bond(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_h2_xyz(tmp_path))
        assert df['mol_n_bonds_estimated'].iloc[0] == 1
        assert abs(df['mol_mean_bond_length'].iloc[0] - 0.74) < 0.1

    def test_co2_two_double_bonds(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_co2_xyz(tmp_path))
        # C=O counted as bonds
        assert df['mol_n_bonds_estimated'].iloc[0] == 2

    def test_water_two_bonds(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        assert df['mol_n_bonds_estimated'].iloc[0] == 2

    def test_benzene_ring_detection(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_benzene_xyz(tmp_path))
        assert df['mol_n_rings_6'].iloc[0] >= 1, "Benzene should have at least one 6-membered ring"

    def test_ethanol_no_rings(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_ethanol_xyz(tmp_path))
        assert df['mol_n_rings'].iloc[0] == 0

    def test_bond_angle_exists(self, featurizer, tmp_path):
        """Water has one bond angle ~104.5°"""
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        assert df['mol_mean_bond_angle'].iloc[0] > 80
        assert df['mol_mean_bond_angle'].iloc[0] < 180

    def test_functional_group_oh_water(self, featurizer, tmp_path):
        """Water has two O-H bonds → has_OH = 2"""
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        assert df['has_OH'].iloc[0] == 2

    def test_functional_group_oh_ethanol(self, featurizer, tmp_path):
        """Ethanol has one O-H (hydroxyl) → has_OH = 1"""
        df = featurizer.featurize_file(_ethanol_xyz(tmp_path))
        # Ethanol has one OH group
        assert df['has_OH'].iloc[0] >= 1

    def test_carbonyl_co2(self, featurizer, tmp_path):
        """CO2 has two C=O → has_C_double_O = 2"""
        df = featurizer.featurize_file(_co2_xyz(tmp_path))
        assert df['has_C_double_O'].iloc[0] == 2

    def test_coordination_numbers(self, featurizer, tmp_path):
        """Methane: C has coordination 4, each H has 1."""
        df = featurizer.featurize_file(_methane_xyz(tmp_path))
        assert df['mol_max_coordination'].iloc[0] == 4
        assert df['mol_n_tetravalent_atoms'].iloc[0] == 1

    def test_terminal_atoms(self, featurizer, tmp_path):
        """CO2: both O atoms are terminal (coordination=1)."""
        df = featurizer.featurize_file(_co2_xyz(tmp_path))
        assert df['mol_n_terminal_atoms'].iloc[0] == 2

    def test_bond_histogram_bins(self, featurizer, tmp_path):
        """Bond histogram should have 10 bins."""
        df = featurizer.featurize_file(_ethanol_xyz(tmp_path))
        for i in range(10):
            assert f'mol_bondlen_hist_{i}' in df.columns

    def test_angle_histogram_bins(self, featurizer, tmp_path):
        """Angle histogram should have 10 bins."""
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        for i in range(10):
            assert f'mol_angle_hist_{i}' in df.columns

    def test_ring_size_histograms(self, featurizer, tmp_path):
        """Ring size histograms for 3-8 member rings."""
        df = featurizer.featurize_file(_benzene_xyz(tmp_path))
        for size in range(3, 9):
            assert f'mol_n_rings_{size}' in df.columns


# ──────────────────────────────────────────────────────────────────
# Batch operations
# ──────────────────────────────────────────────────────────────────

class TestBatchOperations:
    """Multiple files and DataFrame integration."""

    def test_featurize_files(self, featurizer, tmp_path):
        paths = [_h2_xyz(tmp_path), _co2_xyz(tmp_path), _water_xyz(tmp_path)]
        df = featurizer.featurize_files(paths)
        assert len(df) == 3
        assert set(df['mol_source_file']) == {'test_mol'}

    def test_featurize_dataframe(self, featurizer, tmp_path):
        # Create DataFrame with xyz references
        h2 = _h2_xyz(tmp_path)
        df_input = pd.DataFrame({
            'id': ['mol1'],
            'xyz_path': [str(h2)],
        })
        df_out = featurizer.featurize_dataframe(df_input, xyz_column='xyz_path')
        assert len(df_out) == 1
        assert 'mol_n_atoms' in df_out.columns
        assert df_out['mol_n_atoms'].iloc[0] == 2

    def test_list_feature_names(self):
        names = list_molecule_feature_names()
        assert len(names) > 50  # Should have many features
        # Most features start with mol_; functional group flags use has_ prefix
        assert all(n.startswith('mol_') or n.startswith('has_') for n in names)
        # Key features should be present
        assert 'mol_n_atoms' in names
        assert 'mol_mass' in names
        assert 'mol_radius_gyration' in names
        assert 'mol_n_bonds_estimated' in names
        assert 'has_OH' in names


# ──────────────────────────────────────────────────────────────────
# Edge cases & error handling
# ──────────────────────────────────────────────────────────────────

class TestEdgeCases:
    """Robustness: malformed files, single atoms, empty inputs."""

    def test_single_atom(self, featurizer, tmp_path):
        """Single atom (no bonds, no angles)."""
        p = _write_xyz(tmp_path, [
            "1",
            "Single Ar",
            "Ar  0.0  0.0  0.0",
        ])
        df = featurizer.featurize_file(p)
        assert df['mol_n_atoms'].iloc[0] == 1
        assert df['mol_n_bonds_estimated'].iloc[0] == 0
        assert df['mol_mean_bond_angle'].iloc[0] == 0

    def test_numeric_prefix_element(self, featurizer, tmp_path):
        """Some .xyz files have numeric prefixes like '1C' → should parse as 'C'."""
        p = _write_xyz(tmp_path, [
            "2",
            "Ethane",
            "1C  0.0000  0.0000  0.0000",
            "2C  1.5400  0.0000  0.0000",
        ])
        df = featurizer.featurize_file(p)
        assert df['mol_n_carbons'].iloc[0] == 2

    def test_nonexistent_file_raises(self, featurizer, tmp_path):
        with pytest.raises(FileNotFoundError):
            featurizer.featurize_file(tmp_path / "does_not_exist.xyz")

    def test_empty_file_handled(self, featurizer, tmp_path):
        """Empty/malformed .xyz should raise ValueError, not crash."""
        p = _write_xyz(tmp_path, [""])
        with pytest.raises((ValueError, IndexError)):
            featurizer.featurize_file(p)

    def test_all_features_present_for_every_molecule(self, featurizer, tmp_path):
        """CRITICAL: Every molecule should produce the SAME set of columns."""
        molecules = [
            _h2_xyz(tmp_path),
            _co2_xyz(tmp_path),
            _water_xyz(tmp_path),
            _benzene_xyz(tmp_path),
            _ethanol_xyz(tmp_path),
            _methane_xyz(tmp_path),
        ]
        dfs = [featurizer.featurize_file(p) for p in molecules]
        all_cols = set(dfs[0].columns)
        for i, df in enumerate(dfs[1:], 1):
            missing = all_cols - set(df.columns)
            extra = set(df.columns) - all_cols
            assert not missing, f"Molecule {i} missing columns: {missing}"
            assert not extra, f"Molecule {i} extra columns: {extra}"

    def test_no_nans_in_core_features(self, featurizer, tmp_path):
        """Core features (levels 1+2) should never be NaN."""
        df = featurizer.featurize_file(_h2_xyz(tmp_path))
        core_cols = [
            'mol_n_atoms', 'mol_mass', 'mol_hc_ratio',
            'mol_radius_gyration', 'mol_bbox_volume',
            'mol_com_x', 'mol_com_y', 'mol_com_z',
        ]
        for col in core_cols:
            assert not pd.isna(df[col].iloc[0]), f"{col} is NaN"


# ──────────────────────────────────────────────────────────────────
# Feature count validation
# ──────────────────────────────────────────────────────────────────

class TestFeatureCount:
    """Verify total feature count is in the expected range."""

    def test_total_feature_count(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_ethanol_xyz(tmp_path))
        # Exclude source_file column
        feat_cols = [c for c in df.columns if c != 'mol_source_file']
        n_features = len(feat_cols)
        # v2.0: 149 features (115 original + 5 periodic + 19 oxidation states + 10 detailed)
        assert n_features >= 80, f"Expected >=80 features, got {n_features}"
        assert n_features <= 155, f"Expected <=155 features, got {n_features}"
        print(f"\nTotal molecular features: {n_features}")

    def test_level1_feature_count(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        level1_cols = [c for c in df.columns if c.startswith('mol_n_') or c in [
            'mol_mass', 'mol_mass_heavy', 'mol_hc_ratio', 'mol_oc_ratio',
            'mol_nc_ratio', 'mol_on_ratio', 'mol_sc_ratio', 'mol_pc_ratio',
            'mol_elemental_diversity', 'mol_saturation_index', 'mol_unsaturation_degree',
        ]]
        assert len(level1_cols) >= 18

    def test_level2_feature_count(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_water_xyz(tmp_path))
        level2_cols = [c for c in df.columns if any(k in c for k in [
            'com_', 'radius_gyration', 'bbox_', 'inertia', 'asphericity',
            'spherical_defect', 'shape_oblate', 'pairwise_dist', 'density_',
        ])]
        assert len(level2_cols) >= 15

    def test_level3_feature_count(self, featurizer, tmp_path):
        df = featurizer.featurize_file(_ethanol_xyz(tmp_path))
        level3_cols = [c for c in df.columns if any(k in c for k in [
            'bond', 'coordination', 'terminal', 'bivalent', 'trivalent',
            'tetravalent', 'angle', 'ring', 'has_', 'carbonyl', 'alcohol',
            'double_bonds',
        ])]
        assert len(level3_cols) >= 30
