"""Crystal structure loading and featurization using pymatgen (NO matminer).

This module handles loading of crystallographic files (CIF, POSCAR, XYZ, etc.)
and extracting tabular features using only pymatgen, without any dependency on
matminer or other problematic libraries.

The extracted features are pure numeric vectors that feed directly into the
existing AutoAIM training pipeline (XGBoost, LightGBM, Neural Networks, etc.).
"""

import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..utils.logger import LoggerMixin


# ---------------------------------------------------------------------------
# pymatgen availability check (handles PyInstaller missing-data-files issue)
# ---------------------------------------------------------------------------
_PMG_AVAILABLE = False
_PMG_IMPORT_ERROR = None

def _find_pymatgen_data_in_bundle():
    """
    Buscar archivos de datos de pymatgen dentro del bundle de PyInstaller.
    Retorna la ruta al directorio que contiene el directorio 'core/' con los datos,
    o None si no se encuentra.
    """
    import os as _os
    
    _meipass = getattr(sys, '_MEIPASS', None)
    _exe_dir = _os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else None
    
    # Lugares donde buscar, en orden de prioridad
    _search_roots = []
    if _meipass:
        _search_roots.append(_meipass)
    if _exe_dir and _exe_dir != _meipass:
        _search_roots.append(_exe_dir)
        # También buscar en _internal/ subdirectorio (PyInstaller 6.x onedir)
        _internal = _os.path.join(_exe_dir, '_internal')
        if _os.path.exists(_internal):
            _search_roots.append(_internal)
    
    # Subdirectorios donde podria estar pymatgen.
    # pymatgen VA PRIMERO porque el xcopy post-build copia el paquete completo ahi.
    # pymatgen_core es para versiones split-package anteriores.
    _pmg_subdirs = ['pymatgen', 'pymatgen_core', 'pymatgen-core']
    
    # Buscar periodic_table.json.gz
    for _root in _search_roots:
        for _subdir in _pmg_subdirs:
            _candidate = _os.path.join(_root, _subdir)
            _pt = _os.path.join(_candidate, 'core', 'periodic_table.json.gz')
            if _os.path.exists(_pt):
                return _candidate  # Este es el directorio correcto (contiene core/)
        
        # Busqueda recursiva (mas lenta pero mas exhaustiva)
        for _rootpath, _dirs, _files in _os.walk(_root):
            if 'periodic_table.json.gz' in _files:
                # El archivo esta en algun subdirectorio
                _parent = _rootpath
                # Si esta en un subdir 'core/', el PMG_HOME es el padre
                if _os.path.basename(_parent) == 'core':
                    _parent = _os.path.dirname(_parent)
                # Verificar que sea realmente pymatgen
                _pt_check = _os.path.join(_parent, 'core', 'periodic_table.json.gz')
                if _os.path.exists(_pt_check):
                    return _parent
    
    return None


def _try_import_pymatgen():
    """Lazy import of pymatgen - call this inside methods, not at module level."""
    global _PMG_AVAILABLE, _PMG_IMPORT_ERROR
    if _PMG_AVAILABLE:
        return True
    
    # PyInstaller runtime: ensure pymatgen can find its data files
    if getattr(sys, 'frozen', False):
        import os as _os
        
        _meipass = getattr(sys, '_MEIPASS', None)
        _pmg_data_dir = None
        
        if _meipass:
            # Buscar los datos de pymatgen en el bundle
            _pmg_data_dir = _find_pymatgen_data_in_bundle()
            
            if _pmg_data_dir:
                # Encontramos los datos - configurar PMG_HOME
                _os.environ['PMG_HOME'] = _pmg_data_dir
            else:
                # No encontramos los datos en el bundle
                # Intentar: usar fallback dirs
                _exe_dir = _os.path.dirname(sys.executable)
                _fallback_dirs = [
                    _os.path.join(_exe_dir, 'pymatgen'),
                    _os.path.join(_exe_dir, '_internal', 'pymatgen'),
                    _meipass,  # Usar MEIPASS directamente como ultimo recurso
                ]
                for _fd in _fallback_dirs:
                    if _os.path.exists(_fd):
                        _os.environ['PMG_HOME'] = _fd
                        break
                else:
                    # Nada funciono - intentar crear estructura minima
                    _os.environ['PMG_HOME'] = _meipass if _meipass else _exe_dir
    
    try:
        import pymatgen
        from pymatgen.core import Structure, Lattice, Composition
        _PMG_AVAILABLE = True
        return True
    except Exception as e:
        _PMG_IMPORT_ERROR = str(e)
        if getattr(sys, 'frozen', False):
            _meipass = getattr(sys, '_MEIPASS', 'unknown')
            _pmg_home = _os.environ.get('PMG_HOME', 'NOT SET') if '_os' in dir() else 'NOT SET'
            _PMG_IMPORT_ERROR += (
                f"\n\n[PyInstaller Debug] _MEIPASS={_meipass}\n"
                f"[PyInstaller Debug] PMG_HOME={_pmg_home}\n"
                f"[PyInstaller Debug] Error type: {type(e).__name__}\n"
                f"\nPara arreglar:\n"
                f"1. Asegurate de que build_windows_onedir.bat tenga:\n"
                f"   --collect-all pymatgen ^\n"
                f"   --collect-all pymatgen-core ^\n"
                f"2. O ejecuta directamente con el .spec:\n"
                f"   pyinstaller materials_automl_win.spec"
            )
        return False

# NOTE: Do NOT import pymatgen here. Use _try_import_pymatgen() inside methods.
# This prevents PyInstaller crashes when pymatgen data files are missing.


class CrystalStructureLoader(LoggerMixin):
    """Load crystal structure files (CIF, POSCAR, XYZ) using pymatgen.
    
    No dependency on matminer. Uses only pymatgen.core.Structure.
    """
    
    # Supported file extensions
    SUPPORTED_FORMATS = {
        '.cif': 'CIF',
        '.poscar': 'POSCAR',
        '.contcar': 'CONTCAR',
        '.vasp': 'POSCAR',
        '.xyz': 'XYZ',
        '.extxyz': 'XYZ',
        '.json': 'pymatgen_json',
    }
    
    def __init__(self):
        """Initialize loader."""
        self.structures = []  # List of (name, Structure) tuples
        self.failed_files = []  # List of (filepath, error) tuples
    
    @staticmethod
    def _check_pymatgen():
        """Raise informative error if pymatgen is not available."""
        if not _try_import_pymatgen():
            raise ImportError(
                f"pymatgen is required for crystal structure support but could not be imported.\n"
                f"Error: {_PMG_IMPORT_ERROR}\n\n"
                f"To fix:\n"
                f"1. Install pymatgen: pip install pymatgen>=2023.9.10\n"
                f"2. If using PyInstaller: add to your .spec:\n"
                f"   from PyInstaller.utils.hooks import collect_data_files\n"
                f"   datas += collect_data_files('pymatgen_core')\n"
                f"   datas += collect_data_files('pymatgen')"
            )
    
    def load_file(self, file_path: str) -> Optional['Structure']:
        """Load a single structure file.
        
        Args:
            file_path: Path to structure file (CIF, POSCAR, XYZ, etc.)
            
        Returns:
            pymatgen Structure object or None if loading failed
        """
        self._check_pymatgen()
        
        fp = Path(file_path)
        if not fp.exists():
            self.logger.error(f"File not found: {file_path}")
            return None
        
        ext = fp.suffix.lower()
        fname = fp.stem
        
        try:
            if ext == '.cif':
                from pymatgen.io.cif import CifParser
                parser = CifParser(file_path)
                structures = parser.get_structures(primitive=False)
                struct = structures[0] if structures else None
            
            elif ext in ('.poscar', '.contcar', '.vasp'):
                from pymatgen.io.vasp import Poscar
                poscar = Poscar.from_file(file_path)
                struct = poscar.structure
            
            elif ext == '.xyz':
                from pymatgen.io.xyz import XYZ
                xyz = XYZ.from_file(file_path)
                struct = xyz.all_molecules[0] if xyz.all_molecules else None
                # Convert molecule to structure if needed
                if struct is not None and not hasattr(struct, 'lattice'):
                    from pymatgen.core import Lattice, Structure as PymatgenStructure
                    # Create a cubic lattice large enough for the molecule
                    coords = struct.cart_coords
                    max_dim = np.max(coords) - np.min(coords)
                    padding = 10.0  # 10 Angstrom padding
                    lattice_param = max_dim + 2 * padding
                    lattice = Lattice.cubic(lattice_param)
                    # Convert to Structure
                    species = [str(site.specie) for site in struct]
                    cart_coords = struct.cart_coords + lattice_param / 2
                    struct = PymatgenStructure(
                        lattice=lattice,
                        species=species,
                        coords=cart_coords,
                        coords_are_cartesian=True
                    )
            
            elif ext == '.json':
                struct = Structure.from_file(file_path)
            
            else:
                # Try pymatgen generic loader
                struct = Structure.from_file(file_path)
            
            if struct is not None:
                self.logger.info(f"Loaded structure: {fname} ({len(struct)} sites)")
                return struct
            else:
                self.logger.error(f"No structures found in {file_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to load {file_path}: {e}")
            self.failed_files.append((file_path, str(e)))
            return None
    
    def load_directory(self, directory: str) -> int:
        """Load all structure files from a directory.
        
        Args:
            directory: Path to directory containing structure files
            
        Returns:
            Number of structures successfully loaded
        """
        self.structures = []
        self.failed_files = []
        
        dir_path = Path(directory)
        if not dir_path.is_dir():
            self.logger.error(f"Not a directory: {directory}")
            return 0
        
        # Find all supported files
        files = []
        for ext in self.SUPPORTED_FORMATS.keys():
            files.extend(dir_path.glob(f"*{ext}"))
            files.extend(dir_path.glob(f"*{ext.upper()}"))
        
        # Also check for POSCAR files without extension
        files.extend(dir_path.glob("POSCAR*"))
        files.extend(dir_path.glob("CONTCAR*"))
        
        # Deduplicate while preserving order
        seen = set()
        unique_files = []
        for f in files:
            if f.name not in seen:
                seen.add(f.name)
                unique_files.append(f)
        
        self.logger.info(f"Found {len(unique_files)} structure files in {directory}")
        
        for fp in unique_files:
            struct = self.load_file(str(fp))
            if struct is not None:
                self.structures.append((fp.stem, struct))
        
        if self.failed_files:
            self.logger.warning(f"Failed to load {len(self.failed_files)} files")
        
        self.logger.info(f"Successfully loaded {len(self.structures)} structures")
        return len(self.structures)
    
    def load_files(self, file_paths: List[str]) -> int:
        """Load multiple structure files.
        
        Args:
            file_paths: List of file paths
            
        Returns:
            Number of structures successfully loaded
        """
        self.structures = []
        self.failed_files = []
        
        for fp in file_paths:
            struct = self.load_file(fp)
            if struct is not None:
                name = Path(fp).stem
                self.structures.append((name, struct))
        
        return len(self.structures)
    
    def get_structures_dataframe(self) -> pd.DataFrame:
        """Get a DataFrame with basic info about loaded structures.
        
        Returns:
            DataFrame with columns: name, formula, n_sites, n_elements, volume
        """
        if not self.structures:
            return pd.DataFrame()
        
        rows = []
        for name, struct in self.structures:
            rows.append({
                'structure_name': name,
                'formula': struct.formula.replace(' ', ''),
                'formula_pretty': struct.composition.reduced_formula,
                'n_sites': len(struct),
                'n_elements': len(struct.composition.elements),
                'volume': struct.volume,
                'density': struct.density,
            })
        
        return pd.DataFrame(rows)


class CrystalStructureFeaturizer(LoggerMixin):
    """Extract tabular numeric features from crystal structures using pymatgen.
    
    NO dependency on matminer. All features are computed directly from
    pymatgen Structure and Composition objects.
    
    Includes:
      - ~25 crystallographic descriptors (lattice, structure, composition)
      - 106 Magpie-style composition descriptors (reuses existing featurizer)
      - ~15 advanced structural descriptors (RDF, coordination, bonding)
    
    Total: ~145 features per structure, ready for AutoAIM training pipeline.
    """
    
    def __init__(self, use_magpie: bool = True, use_advanced_structural: bool = True):
        """Initialize featurizer.
        
        Args:
            use_magpie: Whether to include 106 Magpie composition features
            use_advanced_structural: Whether to include advanced structural features
        """
        self.use_magpie = use_magpie
        self.use_advanced_structural = use_advanced_structural
        self.feature_names = []
        self._magpie_featurizer = None  # Lazy init
        self._build_feature_name_list()
    
    def _build_feature_name_list(self):
        """Build the list of feature names that will be generated."""
        # --- 1. Lattice features (~10) ---
        lattice_features = [
            'lattice_a', 'lattice_b', 'lattice_c',
            'lattice_alpha', 'lattice_beta', 'lattice_gamma',
            'lattice_volume', 'lattice_abc_ratio',
            'lattice_is_cubic', 'lattice_is_hexagonal',
        ]
        
        # --- 2. Composition-weighted features (~13) ---
        comp_features = [
            'n_sites', 'n_elements',
            'avg_atomic_number', 'avg_atomic_mass',
            'avg_electronegativity', 'avg_ionization_energy',
            'avg_electron_affinity', 'avg_covalent_radius',
            'avg_atomic_radius', 'avg_density',
            'electronegativity_variance',
            'radius_variance', 'mass_variance',
        ]
        
        # --- 3. Structural features (~3) ---
        struct_features = [
            'volume_per_atom', 'packing_fraction',
            'spacegroup_number',
        ]
        
        # --- 4. Derived features (~2) ---
        derived_features = [
            'density', 'total_electrons',
        ]
        
        names = lattice_features + comp_features + struct_features + derived_features
        
        # --- 5. Magpie composition features (106) ---
        if self.use_magpie:
            from .feature_engineering import MagpieFeatures
            magpie = MagpieFeatures(composition_column='formula')
            magpie_names = magpie.get_feature_names_out()
            names.extend([f"magpie_{n}" for n in magpie_names])
        
        # --- 6. Advanced structural features (~15) ---
        if self.use_advanced_structural:
            adv_features = [
                'mean_bond_length', 'std_bond_length',
                'min_bond_length', 'max_bond_length',
                'mean_coordination_number',
                'max_coordination_number',
                'n_bonds_per_atom',
                'rdf_peak_1', 'rdf_peak_2', 'rdf_peak_3',
                'rdf_peak_height_1', 'rdf_peak_height_2',
                'structure_complexity_index',
                'lattice_anisotropy',
                'surface_area_estimate',
            ]
            names.extend(adv_features)
        
        self.feature_names = names
    
    def featurize(self, structure: 'Structure') -> Dict[str, float]:
        """Extract all features from a single structure.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Dictionary of feature_name -> float value
        """
        features = {}
        
        # ==================== 1. LATTICE FEATURES (~10) ====================
        lattice = structure.lattice
        features['lattice_a'] = float(lattice.a)
        features['lattice_b'] = float(lattice.b)
        features['lattice_c'] = float(lattice.c)
        features['lattice_alpha'] = float(lattice.alpha)
        features['lattice_beta'] = float(lattice.beta)
        features['lattice_gamma'] = float(lattice.gamma)
        features['lattice_volume'] = float(lattice.volume)
        
        abc_mean = (lattice.a + lattice.b + lattice.c) / 3.0
        features['lattice_abc_ratio'] = float(
            abc_mean / max(lattice.a, lattice.b, lattice.c)
        ) if max(lattice.a, lattice.b, lattice.c) > 0 else 0.0
        
        features['lattice_is_cubic'] = 1.0 if self._is_cubic(lattice) else 0.0
        features['lattice_is_hexagonal'] = 1.0 if self._is_hexagonal(lattice) else 0.0
        
        # ==================== 2. COMPOSITION FEATURES (~13) ====================
        composition = structure.composition
        elements = composition.elements
        features['n_sites'] = float(len(structure))
        features['n_elements'] = float(len(elements))
        
        atomic_numbers, atomic_masses, electronegativities = [], [], []
        ionization_energies, electron_affinities = [], []
        covalent_radii, atomic_radii, densities = [], [], []
        fractions = []
        
        for element in elements:
            fraction = float(composition.get_atomic_fraction(element))
            fractions.append(fraction)
            atomic_numbers.append(float(element.Z))
            atomic_masses.append(float(element.atomic_mass))
            try:
                en = float(element.X)
            except (TypeError, ValueError):
                en = 0.0
            electronegativities.append(en)
            try:
                ie = float(element.ionization_energy)
            except (TypeError, ValueError):
                ie = 0.0
            ionization_energies.append(ie)
            try:
                ea = float(element.electron_affinity)
            except (TypeError, ValueError):
                ea = 0.0
            electron_affinities.append(ea)
            try:
                cr = float(element.data.get('Covalent radius', 0.0))
            except (TypeError, ValueError):
                cr = 0.0
            covalent_radii.append(cr)
            try:
                ar = float(element.data.get('Atomic radius', 0.0))
            except (TypeError, ValueError):
                ar = 0.0
            atomic_radii.append(ar)
            try:
                d = float(element.data.get('Density of solid', 0.0))
            except (TypeError, ValueError):
                d = 0.0
            densities.append(d)
        
        fractions_arr = np.array(fractions)
        if fractions_arr.sum() > 0:
            fractions_arr = fractions_arr / fractions_arr.sum()
        
        features['avg_atomic_number'] = float(np.average(atomic_numbers, weights=fractions_arr))
        features['avg_atomic_mass'] = float(np.average(atomic_masses, weights=fractions_arr))
        features['avg_electronegativity'] = float(np.average(electronegativities, weights=fractions_arr))
        features['avg_ionization_energy'] = float(np.average(ionization_energies, weights=fractions_arr))
        features['avg_electron_affinity'] = float(np.average(electron_affinities, weights=fractions_arr))
        features['avg_covalent_radius'] = float(np.average(covalent_radii, weights=fractions_arr))
        features['avg_atomic_radius'] = float(np.average(atomic_radii, weights=fractions_arr))
        features['avg_density'] = float(np.average(densities, weights=fractions_arr))
        features['electronegativity_variance'] = float(np.var(electronegativities))
        features['radius_variance'] = float(np.var(atomic_radii))
        features['mass_variance'] = float(np.var(atomic_masses))
        
        # ==================== 3. STRUCTURAL FEATURES (~3) ====================
        features['volume_per_atom'] = float(structure.volume / len(structure))
        features['density'] = float(structure.density)
        features['total_electrons'] = float(composition.total_electrons)
        features['packing_fraction'] = float(self._estimate_packing_fraction(structure))
        
        try:
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            sga = SpacegroupAnalyzer(structure, symprec=0.1)
            features['spacegroup_number'] = float(sga.get_space_group_number())
        except Exception:
            features['spacegroup_number'] = 0.0
        
        # ==================== 4. MAGPIE COMPOSITION (106) ====================
        if self.use_magpie:
            magpie_features = self._featurize_magpie(structure)
            for k, v in magpie_features.items():
                features[f'magpie_{k}'] = v
        
        # ==================== 5. ADVANCED STRUCTURAL (~15) ====================
        if self.use_advanced_structural:
            adv_features = self._featurize_advanced_structural(structure)
            features.update(adv_features)
        
        return features
    
    def _featurize_magpie(self, structure: 'Structure') -> Dict[str, float]:
        """Extract Magpie composition features using the existing custom featurizer.
        
        Reuses MagpieFeatures from feature_engineering.py — 106 descriptors
        computed from the structure's chemical formula.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Dictionary of magpie_feature_name -> float value
        """
        try:
            from .feature_engineering import MagpieFeatures
        except ImportError:
            return {}
        
        try:
            # Lazy init Magpie featurizer
            if self._magpie_featurizer is None:
                self._magpie_featurizer = MagpieFeatures(composition_column='formula')
            
            # Create a DataFrame with the structure's formula
            formula = structure.composition.reduced_formula
            df = pd.DataFrame({'formula': [formula]})
            
            # Compute Magpie features
            magpie_features = self._magpie_featurizer.transform(df)
            
            # Convert to dict (single row)
            result = {}
            for col in magpie_features.columns:
                # Remove 'comp_' prefix that MagpieFeatures adds
                clean_name = col.replace('comp_', '')
                val = magpie_features[col].iloc[0]
                try:
                    result[clean_name] = float(val)
                except (TypeError, ValueError):
                    result[clean_name] = 0.0
            
            return result
            
        except Exception as e:
            self.logger.warning(f"Magpie featurization failed: {e}")
            # Return zeros for all expected Magpie features
            if self._magpie_featurizer is None:
                try:
                    self._magpie_featurizer = MagpieFeatures(composition_column='formula')
                except Exception:
                    return {}
            names = self._magpie_featurizer.get_feature_names_out()
            return {n.replace('comp_', ''): 0.0 for n in names}
    
    def _featurize_advanced_structural(self, structure: 'Structure') -> Dict[str, float]:
        """Extract advanced structural features: bonds, coordination, RDF.
        
        Args:
            structure: pymatgen Structure object
            
        Returns:
            Dictionary of advanced feature_name -> float value
        """
        features = {}
        
        try:
            # --- Bond length statistics ---
            bond_lengths = []
            coordination_numbers = []
            
            # Get all unique element pairs for neighbor finding
            try:
                from pymatgen.analysis.local_env import CrystalNN
                cnn = CrystalNN()
            except ImportError:
                cnn = None
            
            for i, site in enumerate(structure):
                try:
                    if cnn:
                        # Use CrystalNN for coordination environment
                        info = cnn.get_nn_info(structure, i)
                        cn = len(info)
                        coordination_numbers.append(cn)
                        for n in info:
                            bond_lengths.append(n['weight'])
                    else:
                        # Fallback: simple distance-based neighbors
                        neighbors = structure.get_neighbors(site, r=3.5)
                        cn = len(neighbors)
                        coordination_numbers.append(cn)
                        for n_site, dist in neighbors:
                            bond_lengths.append(float(dist))
                except Exception:
                    coordination_numbers.append(0)
            
            if bond_lengths:
                bond_arr = np.array(bond_lengths)
                features['mean_bond_length'] = float(np.mean(bond_arr))
                features['std_bond_length'] = float(np.std(bond_arr))
                features['min_bond_length'] = float(np.min(bond_arr))
                features['max_bond_length'] = float(np.max(bond_arr))
            else:
                features['mean_bond_length'] = 0.0
                features['std_bond_length'] = 0.0
                features['min_bond_length'] = 0.0
                features['max_bond_length'] = 0.0
            
            if coordination_numbers:
                cn_arr = np.array(coordination_numbers)
                features['mean_coordination_number'] = float(np.mean(cn_arr))
                features['max_coordination_number'] = float(np.max(cn_arr))
            else:
                features['mean_coordination_number'] = 0.0
                features['max_coordination_number'] = 0.0
            
            n_atoms = len(structure)
            features['n_bonds_per_atom'] = float(len(bond_lengths) / n_atoms) if n_atoms > 0 else 0.0
            
            # --- Radial Distribution Function (RDF) peaks ---
            rdf_features = self._compute_rdf_peaks(structure)
            features.update(rdf_features)
            
            # --- Complexity index ---
            features['structure_complexity_index'] = float(
                len(structure.composition.elements) * np.log1p(len(structure))
            )
            
            # --- Lattice anisotropy ---
            abc = np.array(structure.lattice.abc)
            features['lattice_anisotropy'] = float(
                np.std(abc) / np.mean(abc) if np.mean(abc) > 0 else 0.0
            )
            
            # --- Surface area estimate (from lattice parameters) ---
            a, b, c = structure.lattice.abc
            features['surface_area_estimate'] = float(2 * (a*b + b*c + a*c))
            
        except Exception as e:
            self.logger.warning(f"Advanced structural featurization failed: {e}")
            # Return zeros
            for name in ['mean_bond_length', 'std_bond_length', 'min_bond_length',
                         'max_bond_length', 'mean_coordination_number',
                         'max_coordination_number', 'n_bonds_per_atom',
                         'rdf_peak_1', 'rdf_peak_2', 'rdf_peak_3',
                         'rdf_peak_height_1', 'rdf_peak_height_2',
                         'structure_complexity_index', 'lattice_anisotropy',
                         'surface_area_estimate']:
                features[name] = 0.0
        
        return features
    
    def _compute_rdf_peaks(self, structure: 'Structure') -> Dict[str, float]:
        """Compute Radial Distribution Function peak positions and heights.
        
        Uses a simple histogram of interatomic distances.
        
        Args:
            structure: pymatgen Structure
            
        Returns:
            Dictionary with rdf_peak_1,2,3 and rdf_peak_height_1,2
        """
        try:
            # Collect all pairwise distances up to a cutoff
            all_dists = []
            cutoff = 10.0  # Angstroms
            
            for i in range(min(len(structure), 50)):  # Sample first 50 sites for speed
                neighbors = structure.get_neighbors(structure[i], r=cutoff)
                for _, dist in neighbors:
                    if dist > 0.5:  # Exclude self
                        all_dists.append(dist)
            
            if not all_dists:
                return {
                    'rdf_peak_1': 0.0, 'rdf_peak_2': 0.0, 'rdf_peak_3': 0.0,
                    'rdf_peak_height_1': 0.0, 'rdf_peak_height_2': 0.0
                }
            
            # Histogram
            hist, bin_edges = np.histogram(all_dists, bins=50, range=(0, cutoff))
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            
            # Find peaks (local maxima)
            peaks = []
            for i in range(1, len(hist) - 1):
                if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > 0:
                    peaks.append((bin_centers[i], hist[i]))
            
            # Sort by height
            peaks.sort(key=lambda x: x[1], reverse=True)
            
            result = {}
            for i in range(3):
                if i < len(peaks):
                    result[f'rdf_peak_{i+1}'] = float(peaks[i][0])
                else:
                    result[f'rdf_peak_{i+1}'] = 0.0
            
            for i in range(2):
                if i < len(peaks):
                    result[f'rdf_peak_height_{i+1}'] = float(peaks[i][1])
                else:
                    result[f'rdf_peak_height_{i+1}'] = 0.0
            
            return result
            
        except Exception:
            return {
                'rdf_peak_1': 0.0, 'rdf_peak_2': 0.0, 'rdf_peak_3': 0.0,
                'rdf_peak_height_1': 0.0, 'rdf_peak_height_2': 0.0
            }
    
    def featurize_structures(self, structures: List[Tuple[str, 'Structure']]) -> pd.DataFrame:
        """Extract features from multiple structures.
        
        Args:
            structures: List of (name, Structure) tuples
            
        Returns:
            DataFrame with one row per structure, columns = feature names
        """
        if not structures:
            return pd.DataFrame(columns=self.feature_names)
        
        rows = []
        for name, struct in structures:
            try:
                features = self.featurize(struct)
                features['structure_name'] = name  # Keep name for reference
                rows.append(features)
            except Exception as e:
                self.logger.warning(f"Failed to featurize {name}: {e}")
                # Add row with zeros
                zero_features = {f: 0.0 for f in self.feature_names}
                zero_features['structure_name'] = name
                rows.append(zero_features)
        
        df = pd.DataFrame(rows)
        return df
    
    def featurize_dataframe(self, df: pd.DataFrame, structure_column: str = 'structure') -> pd.DataFrame:
        """Featurize a DataFrame that contains pymatgen Structure objects.
        
        Args:
            df: DataFrame with a column containing Structure objects
            structure_column: Name of column with Structure objects
            
        Returns:
            DataFrame with original data + crystal features
        """
        if structure_column not in df.columns:
            raise ValueError(f"Column '{structure_column}' not found in DataFrame")
        
        features_list = []
        for idx, struct in df[structure_column].items():
            if struct is not None:
                try:
                    features = self.featurize(struct)
                    features['__index__'] = idx
                    features_list.append(features)
                except Exception as e:
                    self.logger.warning(f"Failed to featurize row {idx}: {e}")
        
        if not features_list:
            return df
        
        features_df = pd.DataFrame(features_list)
        features_df.set_index('__index__', inplace=True)
        
        # Prefix column names to avoid conflicts
        features_df.columns = [f'xtal_{col}' if col != '__index__' else col 
                               for col in features_df.columns]
        
        # Drop structure column from original and merge
        result = df.drop(columns=[structure_column]).join(features_df, how='left')
        
        self.logger.info(
            f"Added {len(features_df.columns)} crystal structure features. "
            f"Shape: {df.shape} -> {result.shape}"
        )
        
        return result
    
    @staticmethod
    def _is_cubic(lattice) -> bool:
        """Check if lattice is approximately cubic."""
        a, b, c = lattice.abc
        alpha, beta, gamma = lattice.angles
        tol = 0.5
        return (abs(a - b) < tol and abs(b - c) < tol and 
                abs(alpha - 90) < tol and abs(beta - 90) < tol and abs(gamma - 90) < tol)
    
    @staticmethod
    def _is_hexagonal(lattice) -> bool:
        """Check if lattice is approximately hexagonal."""
        a, b, c = lattice.abc
        alpha, beta, gamma = lattice.angles
        tol = 0.5
        return (abs(a - b) < tol and abs(gamma - 120) < tol and 
                abs(alpha - 90) < tol and abs(beta - 90) < tol)
    
    @staticmethod
    def _estimate_packing_fraction(structure) -> float:
        """Estimate packing fraction using atomic radii.
        
        Uses covalent radii from pymatgen data.
        Returns packing fraction (0-1).
        """
        try:
            total_atom_volume = 0.0
            for site in structure:
                element = site.specie
                try:
                    radius = float(element.data.get('Covalent radius', 0.0))  # in Angstroms
                except (TypeError, ValueError):
                    radius = 0.0
                
                if radius > 0:
                    # Volume of sphere
                    atom_volume = (4.0 / 3.0) * np.pi * (radius ** 3)
                    total_atom_volume += atom_volume
            
            cell_volume = structure.volume
            if cell_volume > 0:
                return min(total_atom_volume / cell_volume, 1.0)
            return 0.0
        except Exception:
            return 0.0


class CrystalStructureDatasetBuilder(LoggerMixin):
    """Build a complete dataset from crystal structures for AutoAIM training.
    
    Combines structure loading, featurization, and target property assignment
    into a single pipeline that produces a DataFrame ready for DataManager.
    """
    
    def __init__(self):
        """Initialize builder."""
        self.loader = CrystalStructureLoader()
        self.featurizer = CrystalStructureFeaturizer()
        self.dataset_df = None
    
    def build_from_directory(
        self,
        directory: str,
        target_values: Optional[Dict[str, float]] = None,
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """Build dataset from a directory of structure files.
        
        Args:
            directory: Path to directory with CIF/POSCAR/XYZ files
            target_values: Optional dict mapping structure_name -> target value
            target_column: Name for the target column
            
        Returns:
            DataFrame ready for DataManager with crystal features + target
        """
        n_loaded = self.loader.load_directory(directory)
        if n_loaded == 0:
            raise ValueError(f"No structures loaded from {directory}")
        
        return self._build_dataset(target_values, target_column)
    
    def build_from_files(
        self,
        file_paths: List[str],
        target_values: Optional[Dict[str, float]] = None,
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """Build dataset from a list of structure files.
        
        Args:
            file_paths: List of paths to structure files
            target_values: Optional dict mapping structure_name -> target value
            target_column: Name for the target column
            
        Returns:
            DataFrame ready for DataManager with crystal features + target
        """
        n_loaded = self.loader.load_files(file_paths)
        if n_loaded == 0:
            raise ValueError("No structures loaded from provided files")
        
        return self._build_dataset(target_values, target_column)
    
    def build_from_structures(
        self,
        structures: List[Tuple[str, 'Structure']],
        target_values: Optional[Dict[str, float]] = None,
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """Build dataset from already-loaded structures.
        
        Args:
            structures: List of (name, Structure) tuples
            target_values: Optional dict mapping structure_name -> target value
            target_column: Name for the target column
            
        Returns:
            DataFrame ready for DataManager
        """
        self.loader.structures = structures
        return self._build_dataset(target_values, target_column)
    
    def _build_dataset(
        self,
        target_values: Optional[Dict[str, float]] = None,
        target_column: str = 'target'
    ) -> pd.DataFrame:
        """Internal: build dataset from loaded structures.
        
        Returns a DataFrame with:
          - 'structure_name' as a regular column (needed for merging with target CSV)
          - Only numeric feature columns (lattice, composition, magpie, etc.)
          - Target column if provided
        
        NOTE: The caller (data_tab._load_crystal_structures) is responsible
        for setting structure_name as index and filtering to numeric-only
        features before passing to DataManager.
        """
        # Featurize all structures
        features_df = self.featurizer.featurize_structures(self.loader.structures)
        
        if features_df.empty:
            raise ValueError("No features extracted from structures")
        
        # Remove any non-numeric columns (like formula_pretty, file paths)
        # but KEEP structure_name as a column for merging
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
        keep_cols = ['structure_name'] + numeric_cols
        keep_cols = [c for c in keep_cols if c in features_df.columns]
        features_df = features_df[keep_cols].copy()
        
        # Verify we have the expected ~154 numeric features
        expected_n = len(self.featurizer.feature_names)
        actual_numeric = len([c for c in features_df.columns if c != 'structure_name'])
        if actual_numeric != expected_n:
            self.logger.warning(
                f"Feature count mismatch: expected {expected_n} numeric, got {actual_numeric}. "
                f"Extra columns may have been mixed in."
            )
        
        # Add target values if provided
        if target_values:
            features_df[target_column] = features_df['structure_name'].map(target_values)
            n_missing = features_df[target_column].isna().sum()
            if n_missing > 0:
                self.logger.warning(
                    f"{n_missing} structures have no target value and will be dropped"
                )
                features_df = features_df.dropna(subset=[target_column])
        
        self.dataset_df = features_df
        self.logger.info(
            f"Dataset built: {len(features_df)} structures, "
            f"{actual_numeric} numeric features + structure_name column"
        )
        
        return features_df
