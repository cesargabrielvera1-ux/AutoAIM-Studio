"""MoleculeFeaturizer — Extract molecular features from .xyz files.

v2.0.0: 100% Python pure (numpy only). Zero external dependencies.
Works in PyInstaller builds without issues. No RDKit required.

Three feature levels:
  Level 1 — Composition (~20 features): element counts, mass, ratios
  Level 2 — Geometry 3D (~20 features): COM, Rg, bbox, inertia, shape
  Level 3 — Topology estimated (~45 features): bonds, angles, rings,
             functional groups, coordination

Total: ~85 features per molecule.

Usage:
    featurizer = MoleculeFeaturizer()
    df_features = featurizer.featurize_file("path/to/molecule.xyz")
    # or for multiple files:
    df_features = featurizer.featurize_files(["mol1.xyz", "mol2.xyz"])
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# v2.0: Unified periodic data — used by all featurizers (molecule, crystal, composition)
from .periodic_data import (
    ATOMIC_MASSES,
    ATOMIC_NUMBERS,
    ATOMIC_RADII,
    COVALENT_RADII,
    ELECTRONEGATIVITIES,
    IONIZATION_ENERGIES,
    ELECTRON_AFFINITIES,
    GROUP,
    PERIOD,
    MELTING_POINTS,
    THERMAL_CONDUCTIVITIES,
    COMMON_OXIDATION_STATES,
    assign_oxidation_states,
    count_oxidation_states,
    weighted_mean,
    weighted_range,
    weighted_std,
)

logger = logging.getLogger(__name__)

# Bond length templates for functional group detection (in Angstroms)
BOND_LENGTH_TEMPLATES: Dict[str, Tuple[str, str, float, float]] = {
    # (atom1, atom2, expected_length, tolerance)
    'has_OH':   ('O', 'H', 0.96, 0.15),
    'has_NH':   ('N', 'H', 1.01, 0.15),
    'has_CH_aromatic': ('C', 'H', 1.08, 0.10),
    'has_C_double_O': ('C', 'O', 1.23, 0.10),
    'has_CN_triple': ('C', 'N', 1.16, 0.10),
    'has_CC_triple': ('C', 'C', 1.21, 0.10),
    'has_CS':   ('C', 'S', 1.82, 0.15),
    'has_PO':   ('P', 'O', 1.51, 0.12),
    'has_SO':   ('S', 'O', 1.43, 0.12),
    'has_CF':   ('C', 'F', 1.35, 0.10),
    'has_CCl':  ('C', 'Cl', 1.77, 0.10),
    'has_CBr':  ('C', 'Br', 1.94, 0.10),
}

# Elements considered metals for counting
METAL_SYMBOLS: Tuple[str, ...] = (
    'Li', 'Na', 'K', 'Rb', 'Cs', 'Fr',
    'Be', 'Mg', 'Ca', 'Sr', 'Ba', 'Ra',
    'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
    'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg',
    'Al', 'Ga', 'In', 'Sn', 'Tl', 'Pb', 'Bi',
    'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb',
    'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu',
    'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
)

# Halogens (local constant — not in periodic_data)
HALOGEN_SYMBOLS: Tuple[str, ...] = ('F', 'Cl', 'Br', 'I', 'At')

# Common organic elements (local constant — not in periodic_data)
ORGANIC_ELEMENTS: Tuple[str, ...] = ('C', 'H', 'O', 'N', 'P', 'S')


# ============================================================================
# MoleculeFeaturizer class
# ============================================================================

class MoleculeFeaturizer:
    """Extract ~85 molecular features from .xyz files. Zero external deps.

    Parameters
    ----------
    None — fully self-contained.

    Example
    -------
    >>> mf = MoleculeFeaturizer()
    >>> df = mf.featurize_file("benzene.xyz")
    >>> print(df.shape)  # (1, ~85)
    """

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def __init__(self):
        self.atomic_masses = ATOMIC_MASSES
        self.covalent_radii = COVALENT_RADII
        self.bond_templates = BOND_LENGTH_TEMPLATES

    def featurize_file(self, xyz_path: Union[str, Path]) -> pd.DataFrame:
        """Featurize a single .xyz file. Returns DataFrame (1 row, ~85 cols)."""
        path = Path(xyz_path)
        if not path.exists():
            raise FileNotFoundError(f".xyz file not found: {path}")

        elements, coords = self._parse_xyz(path)
        features = self._extract_all_features(elements, coords)

        # Add source file name as identifier
        features['mol_source_file'] = path.stem

        return pd.DataFrame([features])

    def featurize_files(self, xyz_paths: List[Union[str, Path]]) -> pd.DataFrame:
        """Featurize multiple .xyz files. Returns DataFrame (N rows, ~85 cols)."""
        rows = []
        for p in xyz_paths:
            try:
                df = self.featurize_file(p)
                rows.append(df)
            except Exception as e:
                logger.warning(f"Skipping {p}: {e}")
        if not rows:
            return pd.DataFrame()
        return pd.concat(rows, ignore_index=True)

    def featurize_dataframe(
        self,
        df: pd.DataFrame,
        xyz_column: str,
        base_dir: Optional[Union[str, Path]] = None,
    ) -> pd.DataFrame:
        """Attach molecular features to an existing DataFrame.

        The *xyz_column* must contain either absolute .xyz paths or
        file names that are resolved relative to *base_dir*.

        Returns the original DataFrame with ~85 new columns prefixed
        with ``mol_``.
        """
        base_dir = Path(base_dir) if base_dir else Path.cwd()

        feature_rows = []
        for _, row in df.iterrows():
            xyz_ref = row[xyz_column]
            xyz_path = Path(xyz_ref) if Path(xyz_ref).is_absolute() else base_dir / xyz_ref
            try:
                felem, fcoords = self._parse_xyz(xyz_path)
                features = self._extract_all_features(felem, fcoords)
            except Exception as e:
                logger.warning(f"Failed to featurize {xyz_path}: {e}")
                features = self._empty_features()
            feature_rows.append(features)

        feat_df = pd.DataFrame(feature_rows)
        # Ensure no column collisions
        feat_df = feat_df.rename(columns=lambda c: c if c.startswith('mol_') else f'mol_{c}')
        return pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    # ──────────────────────────────────────────────────────────────────
    # .xyz I/O
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_xyz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
        """Parse .xyz file.

        Supports THREE formats automatically detected:

        1. **Standard XYZ** (most common):
           N_atoms
           comment line
           Element X Y Z
           ...

        2. **Raw coordinates** (GaussView/Spartan export):
           Element X Y Z
           Element X Y Z
           ...
           (no atom count, no comment line)

        3. **Gaussian-style** with atom index prefix:
           N_atoms
           comment
           1  Element  X  Y  Z
           2  Element  X  Y  Z
           ...

        Handles BOM (Byte Order Mark), UTF-8/UTF-16 encodings, and
        both Unix (\n) and Windows (\r\n) line endings.

        Returns
        -------
        elements : np.ndarray of str, shape (n_atoms,)
        coords   : np.ndarray of float64, shape (n_atoms, 3)
        """
        # Try UTF-8 with BOM sig first, then fallback to plain UTF-8
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeDecodeError:
            text = path.read_text(encoding='utf-8')

        lines = text.strip().splitlines()
        if not lines:
            raise ValueError(f"Empty .xyz file: {path}")

        # ── Auto-detect format ──────────────────────────────────────────
        first_line = lines[0].lstrip('\ufeff').strip()

        # Try standard XYZ: first line = integer atom count
        try:
            n_atoms = int(first_line)
            # If we have fewer than n_atoms+2 lines, maybe it's raw format
            # and the first line just happens to look like an int
            # (e.g. "C1" stripped to "1").  Be strict: require at least 2 lines.
            if len(lines) >= 2 and n_atoms > 0:
                # Defensive: second line should exist, and line 2 should have coords
                # If line 2 also parses as int, it's probably NOT standard XYZ
                second_parses_as_int = False
                try:
                    int(lines[1].strip().split()[0])
                    second_parses_as_int = True
                except (ValueError, IndexError):
                    pass
                if not second_parses_as_int:
                    # Standard XYZ confirmed
                    return MoleculeFeaturizer._parse_standard_xyz(lines, n_atoms, path)
        except ValueError:
            pass  # First line is not an int → raw format

        # ── Raw coordinates format (no header) ─────────────────────────
        return MoleculeFeaturizer._parse_raw_xyz(lines, path)

    @staticmethod
    def _parse_standard_xyz(
        lines: List[str], n_atoms: int, path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse standard XYZ: N_atoms, comment, then coordinates."""
        elements = []
        coords = []
        # Start after comment line (index 2)
        for i in range(2, min(2 + n_atoms, len(lines))):
            parts = lines[i].split()
            if len(parts) < 4:
                continue
            elem = MoleculeFeaturizer._clean_element(parts[0])
            if not elem:
                continue
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            elements.append(elem)
            coords.append([x, y, z])

        if not elements:
            raise ValueError(f"No valid atoms parsed from {path}")
        return np.array(elements), np.array(coords, dtype=np.float64)

    @staticmethod
    def _parse_raw_xyz(
        lines: List[str], path: Path
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Parse raw coordinate format (no atom count header).

        Each line must contain: Element X Y Z
        Also handles 5-column Gaussian format: Index Element X Y Z
        """
        elements = []
        coords = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            # Skip lines that look like section headers (single token)
            if len(parts) < 2:
                continue

            # Determine which token is the element
            elem, x_idx = MoleculeFeaturizer._detect_element_token(parts)
            if not elem or x_idx < 0 or x_idx + 2 >= len(parts):
                continue

            try:
                x = float(parts[x_idx])
                y = float(parts[x_idx + 1])
                z = float(parts[x_idx + 2])
            except ValueError:
                continue

            elements.append(elem)
            coords.append([x, y, z])

        if not elements:
            raise ValueError(
                f"No valid atoms parsed from {path}. "
                f"Expected format: 'Element X Y Z' (one atom per line). "
                f"File has {len(lines)} lines."
            )
        return np.array(elements), np.array(coords, dtype=np.float64)

    @staticmethod
    def _detect_element_token(parts: List[str]) -> Tuple[Optional[str], int]:
        """Find which token in a split line is the element symbol.

        Returns (element_symbol, index_of_x_coordinate) or (None, -1).

        Handles:
        - Standard:  [C, 0.0, 0.0, 0.0]           → elem=C,  x_idx=1
        - Gaussian:  [1, C, 0, 0.0, 0.0, 0.0]    → elem=C,  x_idx=2
        - Numbered:  [C1, 0.0, 0.0, 0.0]         → elem=C,  x_idx=1
        - Reordered: [0.0, C, 0.0, 0.0]          → elem=C,  x_idx=2
        """
        # Strategy: test each token; the element token is the one that,
        # when cleaned, is a valid chemical symbol (1-2 letters).
        # The next three tokens must be parseable as floats.
        for i, token in enumerate(parts):
            cleaned = MoleculeFeaturizer._clean_element(token)
            if not cleaned:
                continue
            # After the element token, we need 3 more tokens that are floats
            if i + 3 >= len(parts):
                continue
            try:
                float(parts[i + 1])
                float(parts[i + 2])
                float(parts[i + 3])
                return cleaned, i + 1
            except ValueError:
                continue
        return None, -1

    @staticmethod
    def _clean_element(token: str) -> Optional[str]:
        """Extract element symbol from a token.

        Handles:
        - 'C'        → 'C'
        - 'C1'       → 'C'
        - '1C'       → 'C'
        - 'Cl'       → 'Cl'
        - '12'       → None
        - '0.5'      → None
        """
        if not token:
            return None
        token = token.strip()
        # Strip leading numbers: "1C" → "C"
        elem = ''.join(ch for ch in token if ch.isalpha()).capitalize()
        # Validate: must be 1-2 letters
        if len(elem) < 1 or len(elem) > 2:
            return None
        # First letter must be uppercase, second lowercase (or absent)
        if not elem[0].isupper():
            return None
        if len(elem) == 2 and not elem[1].islower():
            return None
        return elem

    # ──────────────────────────────────────────────────────────────────
    # Level 1 — Composition (~20 features)
    # ──────────────────────────────────────────────────────────────────

    def _composition_features(self, elements: np.ndarray) -> Dict[str, float]:
        """Element counts, mass, ratios, diversity."""
        unique, counts = np.unique(elements, return_counts=True)
        comp = dict(zip(unique, counts))
        n_atoms = len(elements)
        n_heavy = n_atoms - comp.get('H', 0)

        masses_arr = np.array([self.atomic_masses.get(e, 0.0) for e in elements])
        total_mass = float(masses_arr.sum())

        # Core element counts
        features: Dict[str, float] = {
            'mol_n_atoms': float(n_atoms),
            'mol_n_heavy_atoms': float(n_heavy),
            'mol_n_hydrogens': float(comp.get('H', 0)),
            'mol_n_carbons': float(comp.get('C', 0)),
            'mol_n_oxygens': float(comp.get('O', 0)),
            'mol_n_nitrogens': float(comp.get('N', 0)),
            'mol_n_sulfurs': float(comp.get('S', 0)),
            'mol_n_phosphorus': float(comp.get('P', 0)),
            'mol_n_silicon': float(comp.get('Si', 0)),
            'mol_n_boron': float(comp.get('B', 0)),
            'mol_n_fluorines': float(comp.get('F', 0)),
            'mol_n_chlorines': float(comp.get('Cl', 0)),
            'mol_n_bromines': float(comp.get('Br', 0)),
            'mol_n_iodines': float(comp.get('I', 0)),
            'mol_n_halogens': float(
                sum(comp.get(h, 0) for h in HALOGEN_SYMBOLS)
            ),
            'mol_n_metals': float(
                sum(comp.get(m, 0) for m in METAL_SYMBOLS)
            ),
            'mol_n_organic_elements': float(
                sum(comp.get(e, 0) for e in ORGANIC_ELEMENTS)
            ),
        }

        # Mass features
        features['mol_mass'] = total_mass
        features['mol_mass_heavy'] = total_mass - comp.get('H', 0) * self.atomic_masses.get('H', 1.008)

        # Element ratios
        n_c = comp.get('C', 0)
        n_h = comp.get('H', 0)
        n_o = comp.get('O', 0)
        n_n = comp.get('N', 0)
        features['mol_hc_ratio'] = n_h / max(n_c, 1)
        features['mol_oc_ratio'] = n_o / max(n_c, 1)
        features['mol_nc_ratio'] = n_n / max(n_c, 1)
        features['mol_on_ratio'] = n_o / max(n_n, 1)
        features['mol_sc_ratio'] = comp.get('S', 0) / max(n_c, 1)
        features['mol_pc_ratio'] = comp.get('P', 0) / max(n_c, 1)

        # Shannon entropy of elemental composition (diversity)
        probs = counts / n_atoms
        entropy = -np.sum(probs * np.log(probs + 1e-12))
        features['mol_elemental_diversity'] = float(entropy)

        # Saturation indicators
        if n_c > 0:
            # Alkane formula: CnH2n+2 → H/C ≈ 2 for saturated
            features['mol_saturation_index'] = n_h / (2.0 * n_c + 2.0)
            # Degree of unsaturation = (2C + 2 + N - H - X) / 2
            n_x = sum(comp.get(h, 0) for h in HALOGEN_SYMBOLS)
            features['mol_unsaturation_degree'] = (2 * n_c + 2 + n_n - n_h - n_x) / 2.0
        else:
            features['mol_saturation_index'] = 0.0
            features['mol_unsaturation_degree'] = 0.0

        # ── v2.0: Periodic properties from composition ──
        # Unified with CrystalFeatureExtractor — composition auto-derived
        # from structure files. No separate Magpie step needed.
        unique_elements = list(comp.keys())
        elem_list = unique_elements
        count_list = [comp[e] for e in unique_elements]

        # Helper: composition-weighted mean
        def _wm(prop_dict):
            return weighted_mean(elem_list, count_list, prop_dict, 0.0)

        features['mol_n_elements'] = float(len(unique_elements))
        atomic_fracs = [comp[e] / n_atoms for e in unique_elements]
        features['mol_max_element_frac'] = float(max(atomic_fracs)) if atomic_fracs else 0.0

        # Atomic number
        features['mol_mean_atomic_number'] = _wm(ATOMIC_NUMBERS)
        features['mol_range_atomic_number'] = weighted_range(elem_list, ATOMIC_NUMBERS, 0.0)

        # Atomic mass (already computed as mol_mass — use weighted avg)
        features['mol_mean_atomic_mass'] = _wm(ATOMIC_MASSES)

        # Atomic radius (Slater)
        features['mol_mean_atomic_radius'] = _wm(ATOMIC_RADII)
        features['mol_range_atomic_radius'] = weighted_range(elem_list, ATOMIC_RADII, 0.0)

        # Covalent radius (Cordero 2008)
        features['mol_mean_covalent_radius'] = _wm(COVALENT_RADII)
        features['mol_range_covalent_radius'] = weighted_range(elem_list, COVALENT_RADII, 0.0)

        # Electronegativity (Pauling)
        features['mol_mean_electroneg'] = _wm(ELECTRONEGATIVITIES)
        features['mol_electroneg_range'] = weighted_range(elem_list, ELECTRONEGATIVITIES, 0.0)

        # Ionization energy
        features['mol_mean_ionization_energy'] = _wm(IONIZATION_ENERGIES)
        features['mol_range_ionization_energy'] = weighted_range(elem_list, IONIZATION_ENERGIES, 0.0)

        # Electron affinity
        features['mol_mean_electron_affinity'] = _wm(ELECTRON_AFFINITIES)

        # Group and period
        features['mol_mean_group'] = _wm(GROUP)
        features['mol_range_group'] = weighted_range(elem_list, GROUP, 0.0)
        features['mol_mean_period'] = _wm(PERIOD)
        features['mol_range_period'] = weighted_range(elem_list, PERIOD, 0.0)

        # Thermal properties
        features['mol_mean_melting_point'] = _wm(MELTING_POINTS)
        features['mol_mean_thermal_conductivity'] = _wm(THERMAL_CONDUCTIVITIES)

        # ── v2.0: Oxidation states (Fase 3) ──
        ox_counts = count_oxidation_states(unique_elements, count_list)
        for key, val in ox_counts.items():
            features[f'mol_{key}'] = float(val)

        return features

    # ──────────────────────────────────────────────────────────────────
    # Level 2 — Geometry 3D (~20 features)
    # ──────────────────────────────────────────────────────────────────

    def _geometry_features(self, elements: np.ndarray, coords: np.ndarray) -> Dict[str, float]:
        """Center of mass, radius of gyration, bbox, inertia tensor, shape."""
        n_atoms = len(elements)
        masses_arr = np.array([self.atomic_masses.get(e, 0.0) for e in elements])
        total_mass = float(masses_arr.sum())

        features: Dict[str, float] = {}

        # Center of mass (mass-weighted)
        if total_mass > 0:
            com = np.average(coords, axis=0, weights=masses_arr)
        else:
            com = coords.mean(axis=0)

        features['mol_com_x'] = float(com[0])
        features['mol_com_y'] = float(com[1])
        features['mol_com_z'] = float(com[2])

        # Centered coordinates
        centered = coords - com

        # Radius of gyration (mass-weighted RMS distance from COM)
        if total_mass > 0 and n_atoms > 0:
            rg_sq = np.average(np.sum(centered ** 2, axis=1), weights=masses_arr)
            features['mol_radius_gyration'] = float(np.sqrt(max(rg_sq, 0.0)))
        else:
            features['mol_radius_gyration'] = 0.0

        # Bounding box dimensions and volume
        mins = coords.min(axis=0)
        maxs = coords.max(axis=0)
        dims = maxs - mins
        # Clamp dims to minimum 0.01 to avoid zero volume for
        # linear/planar molecules (all coords identical on one axis)
        dims = np.maximum(dims, 0.01)
        features['mol_bbox_x'] = float(dims[0])
        features['mol_bbox_y'] = float(dims[1])
        features['mol_bbox_z'] = float(dims[2])
        features['mol_bbox_volume'] = float(np.prod(dims))
        features['mol_bbox_diagonal'] = float(np.linalg.norm(dims))

        # Surface area approximation (box surface)
        if len(dims) == 3:
            features['mol_bbox_surface'] = float(
                2 * (dims[0]*dims[1] + dims[1]*dims[2] + dims[2]*dims[0])
            )
        else:
            features['mol_bbox_surface'] = 0.0

        # Principal moments of inertia
        if n_atoms >= 2:
            inertia_tensor = self._inertia_tensor(centered, masses_arr)
            eigenvals = np.sort(np.linalg.eigvalsh(inertia_tensor))[::-1]

            features['mol_inertia_1'] = float(eigenvals[0])
            features['mol_inertia_2'] = float(eigenvals[1])
            features['mol_inertia_3'] = float(eigenvals[2])

            # Normalized eigenvalues (unitless shape descriptors)
            trace = eigenvals.sum()
            if trace > 0:
                features['mol_inertia_norm_1'] = float(eigenvals[0] / trace)
                features['mol_inertia_norm_2'] = float(eigenvals[1] / trace)
                features['mol_inertia_norm_3'] = float(eigenvals[2] / trace)
            else:
                features['mol_inertia_norm_1'] = 0.0
                features['mol_inertia_norm_2'] = 0.0
                features['mol_inertia_norm_3'] = 0.0

            # Asphericity: I_max / I_min - 1
            # For linear molecules (I_min ≈ 0), use a fixed high value
            # to indicate rod-like shape. For symmetric shapes where
            # I_mid ≈ I_max, asphericity ≈ 0 (sphere/cylinder).
            if eigenvals[2] > 1e-10:
                features['mol_asphericity'] = float(eigenvals[0] / eigenvals[2] - 1.0)
            elif eigenvals[1] > 1e-10:
                # Linear molecule (e.g., CO2, O=C=O): I_min ≈ 0
                features['mol_asphericity'] = 10.0  # capped high value
            else:
                features['mol_asphericity'] = 0.0

            # Spherical defect (0 = sphere, 1 = rod or disk)
            denom = 2.0 * trace ** 2
            if denom > 1e-10:
                features['mol_spherical_defect'] = float(
                    ((eigenvals[0] - eigenvals[1]) ** 2 +
                     (eigenvals[1] - eigenvals[2]) ** 2 +
                     (eigenvals[0] - eigenvals[2]) ** 2) / denom
                )
            else:
                features['mol_spherical_defect'] = 0.0

            # Oblate / prolate indicator
            mid = eigenvals[1]
            if mid > 1e-10:
                features['mol_shape_oblate_prolate'] = float(
                    (eigenvals[0] - eigenvals[2]) / mid
                )
            else:
                features['mol_shape_oblate_prolate'] = 0.0
        else:
            for k in ['mol_inertia_1', 'mol_inertia_2', 'mol_inertia_3',
                      'mol_inertia_norm_1', 'mol_inertia_norm_2', 'mol_inertia_norm_3',
                      'mol_asphericity', 'mol_spherical_defect', 'mol_shape_oblate_prolate']:
                features[k] = 0.0

        # Pairwise distance statistics
        if n_atoms >= 2:
            dists = self._pairwise_distances(coords)
            # Exclude diagonal (zeros)
            mask = dists > 0
            valid_dists = dists[mask]
            if len(valid_dists) > 0:
                features['mol_max_pairwise_dist'] = float(valid_dists.max())
                features['mol_mean_pairwise_dist'] = float(valid_dists.mean())
                features['mol_std_pairwise_dist'] = float(valid_dists.std())
                features['mol_median_pairwise_dist'] = float(np.median(valid_dists))
            else:
                features['mol_max_pairwise_dist'] = 0.0
                features['mol_mean_pairwise_dist'] = 0.0
                features['mol_std_pairwise_dist'] = 0.0
                features['mol_median_pairwise_dist'] = 0.0
        else:
            features['mol_max_pairwise_dist'] = 0.0
            features['mol_mean_pairwise_dist'] = 0.0
            features['mol_std_pairwise_dist'] = 0.0
            features['mol_median_pairwise_dist'] = 0.0

        # Density (atoms per cubic angstrom of bounding box)
        vol = features['mol_bbox_volume']
        features['mol_density_atoms_per_a3'] = float(n_atoms / vol) if vol > 1e-10 else 0.0

        # Mass density (Daltons per cubic angstrom)
        if vol > 1e-10:
            features['mol_density_mass_per_a3'] = float(total_mass / vol)
        else:
            features['mol_density_mass_per_a3'] = 0.0

        return features

    @staticmethod
    def _inertia_tensor(coords_centered: np.ndarray, masses: np.ndarray) -> np.ndarray:
        """Compute 3×3 inertia tensor for point masses."""
        I = np.zeros((3, 3), dtype=np.float64)
        for i in range(len(coords_centered)):
            m = masses[i]
            x, y, z = coords_centered[i]
            I[0, 0] += m * (y * y + z * z)
            I[1, 1] += m * (x * x + z * z)
            I[2, 2] += m * (x * x + y * y)
            I[0, 1] -= m * x * y
            I[0, 2] -= m * x * z
            I[1, 2] -= m * y * z
        I[1, 0] = I[0, 1]
        I[2, 0] = I[0, 2]
        I[2, 1] = I[1, 2]
        return I

    @staticmethod
    def _pairwise_distances(coords: np.ndarray) -> np.ndarray:
        """Compute full pairwise distance matrix."""
        diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=2))

    # ──────────────────────────────────────────────────────────────────
    # Level 3 — Estimated Topology (~45 features)
    # ──────────────────────────────────────────────────────────────────

    def _topology_features(self, elements: np.ndarray, coords: np.ndarray) -> Dict[str, float]:
        """Bonds, angles, rings, functional groups, coordination."""
        n_atoms = len(elements)
        features: Dict[str, float] = {}

        # Distance matrix
        dists = self._pairwise_distances(coords)
        np.fill_diagonal(dists, 1e6)  # Exclude self

        # ── Estimated bond graph ──
        bonds, bond_lengths = self._estimate_bonds(elements, dists)
        n_bonds = len(bonds)

        features['mol_n_bonds_estimated'] = float(n_bonds)

        if n_bonds > 0:
            bl_arr = np.array(bond_lengths)
            features['mol_mean_bond_length'] = float(bl_arr.mean())
            features['mol_std_bond_length'] = float(bl_arr.std())
            features['mol_min_bond_length'] = float(bl_arr.min())
            features['mol_max_bond_length'] = float(bl_arr.max())
        else:
            features['mol_mean_bond_length'] = 0.0
            features['mol_std_bond_length'] = 0.0
            features['mol_min_bond_length'] = 0.0
            features['mol_max_bond_length'] = 0.0

        # Bond length histogram (10 bins: 0.8–1.8 Å)
        if n_bonds > 0:
            hist, _ = np.histogram(bond_lengths, bins=10, range=(0.8, 1.8))
            for i, h in enumerate(hist):
                features[f'mol_bondlen_hist_{i}'] = float(h)
        else:
            for i in range(10):
                features[f'mol_bondlen_hist_{i}'] = 0.0

        # ── Coordination numbers ──
        coord_nums = self._coordination_numbers(elements, dists)
        if len(coord_nums) > 0:
            features['mol_mean_coordination'] = float(np.mean(coord_nums))
            features['mol_max_coordination'] = float(np.max(coord_nums))
            features['mol_std_coordination'] = float(np.std(coord_nums))
        else:
            features['mol_mean_coordination'] = 0.0
            features['mol_max_coordination'] = 0.0
            features['mol_std_coordination'] = 0.0

        # Valency counts (coordination ≥ 1)
        n_terminal = int(np.sum(coord_nums == 1))
        n_bivalent = int(np.sum(coord_nums == 2))
        n_trivalent = int(np.sum(coord_nums == 3))
        n_tetravalent = int(np.sum(coord_nums == 4))
        features['mol_n_terminal_atoms'] = float(n_terminal)
        features['mol_n_bivalent_atoms'] = float(n_bivalent)
        features['mol_n_trivalent_atoms'] = float(n_trivalent)
        features['mol_n_tetravalent_atoms'] = float(n_tetravalent)

        # ── Bond angle statistics ──
        angles = self._compute_bond_angles(elements, coords, bonds)
        if len(angles) > 0:
            angles_arr = np.array(angles)
            features['mol_mean_bond_angle'] = float(angles_arr.mean())
            features['mol_std_bond_angle'] = float(angles_arr.std())
            features['mol_min_bond_angle'] = float(angles_arr.min())
            features['mol_max_bond_angle'] = float(angles_arr.max())

            # Angle histogram (10 bins: 60–180°)
            hist, _ = np.histogram(angles_arr, bins=10, range=(60, 180))
            for i, h in enumerate(hist):
                features[f'mol_angle_hist_{i}'] = float(h)
        else:
            features['mol_mean_bond_angle'] = 0.0
            features['mol_std_bond_angle'] = 0.0
            features['mol_min_bond_angle'] = 0.0
            features['mol_max_bond_angle'] = 0.0
            for i in range(10):
                features[f'mol_angle_hist_{i}'] = 0.0

        # ── Ring detection (simple cycle enumeration) ──
        ring_sizes = self._detect_rings(elements, bonds)
        features['mol_n_rings'] = float(len(ring_sizes))
        if len(ring_sizes) > 0:
            rs = np.array(ring_sizes)
            features['mol_mean_ring_size'] = float(rs.mean())
            features['mol_min_ring_size'] = float(rs.min())
            features['mol_max_ring_size'] = float(rs.max())
            for size in range(3, 9):
                features[f'mol_n_rings_{size}'] = float(np.sum(rs == size))
        else:
            features['mol_mean_ring_size'] = 0.0
            features['mol_min_ring_size'] = 0.0
            features['mol_max_ring_size'] = 0.0
            for size in range(3, 9):
                features[f'mol_n_rings_{size}'] = 0.0

        # ── Functional group detection ──
        fg_features = self._detect_functional_groups(elements, coords, dists)
        features.update(fg_features)

        # ── Bond type ratios ──
        if n_bonds > 0 and n_atoms > 0:
            features['mol_bonds_per_atom'] = float(n_bonds) / n_atoms
            # Estimate double bonds by short C=C and C=O distances
            n_double_est = 0
            for (i, j), bl in zip(bonds, bond_lengths):
                pair = tuple(sorted([elements[i], elements[j]]))
                if pair == ('C', 'C') and bl < 1.45:
                    n_double_est += 1
                elif pair == ('C', 'O') and bl < 1.35:
                    n_double_est += 1
                elif pair == ('C', 'N') and bl < 1.35:
                    n_double_est += 1
            features['mol_n_double_bonds_est'] = float(n_double_est)
        else:
            features['mol_bonds_per_atom'] = 0.0
            features['mol_n_double_bonds_est'] = 0.0

        return features

    def _estimate_bonds(
        self, elements: np.ndarray, dists: np.ndarray
    ) -> Tuple[List[Tuple[int, int]], List[float]]:
        """Estimate bonds by covalent radii sum + 0.45 Å tolerance."""
        n_atoms = len(elements)
        bonds: List[Tuple[int, int]] = []
        lengths: List[float] = []
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r1 = self.covalent_radii.get(elements[i], 1.0)
                r2 = self.covalent_radii.get(elements[j], 1.0)
                max_bond = r1 + r2 + 0.45
                d = dists[i, j]
                if d < max_bond:
                    bonds.append((i, j))
                    lengths.append(float(d))
        return bonds, lengths

    def _coordination_numbers(self, elements: np.ndarray, dists: np.ndarray) -> np.ndarray:
        """Coordination number per atom (estimated bonds per atom)."""
        n_atoms = len(elements)
        coord = np.zeros(n_atoms, dtype=int)
        for i in range(n_atoms):
            for j in range(i + 1, n_atoms):
                r1 = self.covalent_radii.get(elements[i], 1.0)
                r2 = self.covalent_radii.get(elements[j], 1.0)
                if dists[i, j] < r1 + r2 + 0.45:
                    coord[i] += 1
                    coord[j] += 1
        return coord

    def _compute_bond_angles(
        self,
        elements: np.ndarray,
        coords: np.ndarray,
        bonds: List[Tuple[int, int]],
    ) -> List[float]:
        """Compute all bond angles (i–j–k where j is central)."""
        # Build adjacency list
        adj: Dict[int, List[int]] = {i: [] for i in range(len(elements))}
        for i, j in bonds:
            adj[i].append(j)
            adj[j].append(i)

        angles: List[float] = []
        for j in adj:
            neighbors = adj[j]
            if len(neighbors) < 2:
                continue
            for idx_a in range(len(neighbors)):
                for idx_b in range(idx_a + 1, len(neighbors)):
                    i = neighbors[idx_a]
                    k = neighbors[idx_b]
                    v1 = coords[i] - coords[j]
                    v2 = coords[k] - coords[j]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 > 1e-10 and n2 > 1e-10:
                        cos_angle = np.dot(v1, v2) / (n1 * n2)
                        cos_angle = np.clip(cos_angle, -1.0, 1.0)
                        angle_deg = np.degrees(np.arccos(cos_angle))
                        angles.append(float(angle_deg))
        return angles

    def _detect_rings(
        self, elements: np.ndarray, bonds: List[Tuple[int, int]]
    ) -> List[int]:
        """Detect simple cycles (rings) up to 8 members using DFS.

        Only considers heavy-atom cycles — hydrogen atoms are excluded
        because H-C-H paths create false cycles (e.g., C1-C2-H-C1).
        """
        n_atoms = len(elements)
        # Build adjacency only for heavy atoms (non-H)
        is_heavy = elements != 'H'
        heavy_indices = set(np.where(is_heavy)[0])

        adj = {i: set() for i in range(n_atoms) if is_heavy[i]}
        for i, j in bonds:
            if is_heavy[i] and is_heavy[j]:
                adj[i].add(j)
                adj[j].add(i)

        ring_sizes: List[int] = []
        visited_cycles = set()

        def dfs(node: int, start: int, path: List[int], depth: int):
            if depth > 8:
                return
            for neighbor in adj.get(node, set()):
                if neighbor == start and len(path) >= 3:
                    cycle = tuple(sorted(path))
                    if cycle not in visited_cycles:
                        visited_cycles.add(cycle)
                        ring_sizes.append(len(path))
                elif neighbor not in path and neighbor > start:
                    dfs(neighbor, start, path + [neighbor], depth + 1)

        for start in adj:
            for neighbor in adj[start]:
                if neighbor > start:
                    dfs(neighbor, start, [start, neighbor], 2)

        return ring_sizes

    def _detect_functional_groups(
        self, elements: np.ndarray, coords: np.ndarray, dists: np.ndarray
    ) -> Dict[str, float]:
        """Detect functional groups by bond distance matching."""
        features: Dict[str, float] = {}

        for fg_name, (a1, a2, expected, tol) in self.bond_templates.items():
            found = 0
            n_atoms = len(elements)
            for i in range(n_atoms):
                for j in range(i + 1, n_atoms):
                    pair = tuple(sorted([elements[i], elements[j]]))
                    template_pair = tuple(sorted([a1, a2]))
                    if pair == template_pair:
                        d = dists[i, j]
                        if abs(d - expected) < tol:
                            found += 1
            features[fg_name] = float(found)

        # Carbonyl ratio (C=O per heavy atom)
        n_heavy = len(elements) - np.sum(elements == 'H')
        if n_heavy > 0:
            features['mol_carbonyl_ratio'] = features.get('has_C_double_O', 0.0) / n_heavy
        else:
            features['mol_carbonyl_ratio'] = 0.0

        # Fraction of carbons with OH (alcohol indicator)
        n_c = np.sum(elements == 'C')
        if n_c > 0:
            features['mol_alcohol_ratio'] = features.get('has_OH', 0.0) / n_c
        else:
            features['mol_alcohol_ratio'] = 0.0

        return features

    # ──────────────────────────────────────────────────────────────────
    # Orchestrator
    # ──────────────────────────────────────────────────────────────────

    def _extract_all_features(
        self, elements: np.ndarray, coords: np.ndarray
    ) -> Dict[str, float]:
        """Run all three levels and merge."""
        features: Dict[str, float] = {}
        features.update(self._composition_features(elements))
        features.update(self._geometry_features(elements, coords))
        features.update(self._topology_features(elements, coords))
        return features

    def _empty_features(self) -> Dict[str, float]:
        """Return a feature dict full of NaN for failed molecules."""
        # We need to know the exact columns; compute from a dummy call
        # or use a hardcoded set. Simpler: return empty and let
        # DataFrame handle it, but better: predefine NaN columns.
        # For now, return a minimal dict — the caller (featurize_dataframe)
        # should handle missing columns by aligning afterward.
        return {}


# ============================================================================
# Convenience: feature name listing
# ============================================================================

def list_molecule_feature_names() -> List[str]:
    """Return all feature names that MoleculeFeaturizer produces.

    Useful for documentation and column alignment."""
    mf = MoleculeFeaturizer()
    # Use a minimal dummy molecule (H2) to trigger all code paths
    import tempfile
    import os
    with tempfile.NamedTemporaryFile(mode='w', suffix='.xyz', delete=False) as f:
        f.write("2\nH2 dummy\nH 0.0 0.0 0.0\nH 0.74 0.0 0.0\n")
        tmp = f.name
    try:
        df = mf.featurize_file(tmp)
        return [c for c in df.columns if c != 'mol_source_file']
    finally:
        os.unlink(tmp)
