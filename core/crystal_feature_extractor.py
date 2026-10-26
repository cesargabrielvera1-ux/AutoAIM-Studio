"""
Crystal Feature Extractor for AutoAIM Studio v2.0.

Links crystal structure files (CIF) with tabular datasets via an ID column,
extracting structural descriptors and merging them into the dataset for training.

v2.0 UNIFIED: Uses CrystalStructureFeaturizer internally so that both
"direct CIF loading" and "CIF-to-CSV linking" produce the SAME 149 features.
The linking workflow simply prefixes all feature names with 'crystal_'.
"""

import os
import glob
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class LinkReport:
    """Report of the crystal structure linking process."""
    id_column: str
    cif_folder: str
    n_rows_total: int
    n_matched: int
    n_missing: int
    n_duplicate_ids: int
    crystal_features_added: List[str] = field(default_factory=list)
    composition_features_replaced: bool = False
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_rows_total if self.n_rows_total > 0 else 0.0

    def to_text(self) -> str:
        lines = [
            "Crystal Structure Link Report",
            "=" * 40,
            f"ID Column          : {self.id_column}",
            f"CIF Folder         : {self.cif_folder}",
            f"Total rows         : {self.n_rows_total}",
            f"Matched CIFs       : {self.n_matched} ({self.match_rate*100:.1f}%)",
            f"Missing CIFs       : {self.n_missing}",
            f"Duplicate IDs      : {self.n_duplicate_ids}",
            f"Crystal features   : {len(self.crystal_features_added)}",
            f"Comp. replaced     : {self.composition_features_replaced}",
        ]
        if self.warnings:
            lines.extend(["", "Warnings:"] + [f"  - {w}" for w in self.warnings])
        if self.errors:
            lines.extend(["", "Errors:"] + [f"  - {e}" for e in self.errors])
        return "\n".join(lines)


class CrystalFeatureExtractor:
    """Extracts crystal structure features from CIF files and merges with datasets.

    v2.0 UNIFIED: Uses CrystalStructureFeaturizer internally so that both
    direct CIF loading and CIF-to-CSV linking produce the same 149 features.
    """

    def __init__(self):
        self._link_report: Optional[LinkReport] = None
        self._crystal_features_df: Optional[pd.DataFrame] = None
        self._has_pymatgen = False
        self._crystal_feature_names: List[str] = []
        self._featurizer = None
        try:
            from pymatgen.core import Structure
            from .crystal_structure import CrystalStructureFeaturizer
            self._has_pymatgen = True
            self.Structure = Structure
            # Build unified feature name list with crystal_ prefix
            featurizer = CrystalStructureFeaturizer(
                use_magpie=True, use_advanced_structural=True
            )
            self._crystal_feature_names = [
                f'crystal_{name}' for name in featurizer.get_feature_names()
            ]
            self._featurizer = featurizer
        except ImportError:
            logger.warning("pymatgen not available. Crystal feature extraction disabled.")

    @property
    def has_pymatgen(self) -> bool:
        return self._has_pymatgen

    @property
    def link_report(self) -> Optional[LinkReport]:
        return self._link_report

    @property
    def crystal_features_df(self) -> Optional[pd.DataFrame]:
        return self._crystal_features_df

    @property
    def crystal_feature_names(self) -> List[str]:
        """Return the list of crystal feature names (with crystal_ prefix)."""
        return self._crystal_feature_names.copy()

    def detect_id_candidates(self, df: pd.DataFrame) -> List[Tuple[str, int]]:
        """Detect columns that could serve as ID for linking with CIFs.

        Returns list of (column_name, uniqueness_score) sorted by score.
        Uniqueness score = number of unique values / total rows (1.0 = fully unique).
        """
        candidates = []
        for col in df.columns:
            # Skip obvious non-ID columns
            if col.lower() in ('target', 'y', 'label', 'class', 'index'):
                continue
            n_unique = df[col].nunique()
            score = n_unique / len(df)
            # Favor columns that are mostly unique and string-like
            if score >= 0.5:  # At least 50% unique
                candidates.append((col, n_unique, score))
        # Sort by uniqueness score descending
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(c[0], c[1]) for c in candidates]

    def check_cif_availability(self, df: pd.DataFrame, id_column: str,
                                cif_folder: str) -> Dict[str, Any]:
        """Check which IDs have corresponding CIF files.

        Returns dict with 'matched', 'missing', 'duplicate_ids'.
        """
        cif_path = Path(cif_folder)
        if not cif_path.exists():
            return {'matched': [], 'missing': list(df[id_column].dropna().unique()),
                    'duplicate_ids': [], 'error': 'Folder does not exist'}

        # Get all .cif files in folder
        cif_files = {f.stem for f in cif_path.glob('*.cif')}

        ids = df[id_column].dropna().astype(str).tolist()
        matched = [mid for mid in ids if mid in cif_files]
        missing = [mid for mid in ids if mid not in cif_files]

        # Check for duplicate IDs
        id_counts = df[id_column].value_counts()
        duplicate_ids = id_counts[id_counts > 1].index.tolist()

        return {
            'matched': matched,
            'missing': missing,
            'duplicate_ids': duplicate_ids,
            'n_cif_files': len(cif_files),
            'n_ids': len(set(ids)),
        }

    def _detect_artificial_cell(self, struct) -> Tuple[bool, str]:
        """Detect if a structure appears to be a molecule in an artificial box.
        
        v2.0: This detects common patterns of non-crystalline structures
        placed in a simulation box, which produces physically meaningless
        crystal descriptors.
        
        Returns:
            (is_artificial, reason_string)
        """
        lattice = struct.lattice
        n_sites = len(struct)
        
        if n_sites == 0:
            return True, "Empty structure"
        
        # Signal 1: Perfect cubic box (a=b=c, all angles 90) — common for isolated molecules
        is_cubic_box = (
            abs(lattice.a - lattice.b) < 0.01 and
            abs(lattice.a - lattice.c) < 0.01 and
            abs(lattice.alpha - 90.0) < 0.1 and
            abs(lattice.beta - 90.0) < 0.1 and
            abs(lattice.gamma - 90.0) < 0.1
        )
        
        # Signal 2: Unrealistically low density (< 0.01 g/cm³) — molecule in huge box
        density = struct.density if struct.density else 0.0
        
        # Signal 3: Volume per atom is very large (> 1000 Å³/atom)
        vol_per_atom = lattice.volume / n_sites
        
        reasons = []
        if is_cubic_box and vol_per_atom > 500:
            reasons.append(
                f"perfect cubic simulation box ({lattice.a:.1f}x{lattice.b:.1f}x{lattice.c:.1f} Å) "
                f"with very low density ({density:.4f} g/cm³) — appears to be an isolated molecule, "
                f"not a periodic crystal"
            )
        elif density < 0.001:
            reasons.append(
                f"extremely low density ({density:.4f} g/cm³) — "
                f"the unit cell is likely a simulation box, not a real crystal"
            )
        elif vol_per_atom > 1000:
            reasons.append(
                f"very large volume per atom ({vol_per_atom:.0f} Å³) — "
                f"typical of isolated molecules in artificial containers"
            )
        
        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def extract_cif_features(self, cif_path: str) -> Optional[Dict[str, float]]:
        """Extract structural features from a single CIF file.

        v2.0 UNIFIED: Uses CrystalStructureFeaturizer internally to produce
        the same 149 features as direct CIF loading. Feature names are prefixed
        with 'crystal_' for the linking workflow.

        Returns dict of feature_name -> value, or None if extraction fails.
        Also detects non-crystalline structures (isolated molecules) and
        logs a warning with explanation.
        """
        if not self._has_pymatgen or self._featurizer is None:
            return None
        try:
            struct = self.Structure.from_file(cif_path)
            
            # v2.0: Detect isolated molecules in artificial boxes
            is_artificial, reason = self._detect_artificial_cell(struct)
            if is_artificial:
                logger.warning(
                    f"Non-crystalline structure detected in {cif_path}: {reason}. "
                    f"Crystal descriptors will be numerically valid but physically meaningless. "
                    f"This tool is designed for periodic crystalline materials with defined unit cells. "
                    f"For isolated organic molecules, consider using molecular descriptors (e.g., RDKit) instead."
                )
            
            # v2.0 UNIFIED: Use CrystalStructureFeaturizer for consistent features
            features = self._featurizer.featurize(struct)
            
            # Prefix all feature names with 'crystal_' for the linking workflow
            renamed = {f'crystal_{k}': float(v) for k, v in features.items()}
            
            return renamed
        except Exception as e:
            logger.debug(f"Failed to extract features from {cif_path}: {e}")
            return None

    def scan_for_noncrystalline(self, df: pd.DataFrame, id_column: str,
                                 cif_folder: str) -> List[Tuple[str, str]]:
        """v2.0: Scan CIF files for non-crystalline structures (isolated molecules).

        Returns list of (material_id, reason) for structures that appear to be
        molecules in artificial simulation boxes rather than periodic crystals.

        This should be called BEFORE link_structures to warn the user.
        """
        if not self._has_pymatgen:
            return []

        cif_path = Path(cif_folder)
        non_crystalline = []

        # v2.0 FIX: Only iterate over unique IDs that have matching CIF files
        # instead of scanning the entire DataFrame row-by-row.
        unique_ids = df[id_column].dropna().astype(str).unique()
        for mid in unique_ids:
            cif_file = cif_path / f"{mid}.cif"
            if not cif_file.exists():
                continue
            try:
                struct = self.Structure.from_file(str(cif_file))
                is_artificial, reason = self._detect_artificial_cell(struct)
                if is_artificial:
                    non_crystalline.append((mid, reason))
            except Exception as exc:
                logger.debug(f"Failed to parse CIF for artificial cell detection ({cif_file}): {exc}")
                pass  # Failed to parse — will be handled elsewhere

        return non_crystalline

    def link_structures(self, df: pd.DataFrame, id_column: str,
                        cif_folder: str,
                        replace_composition_features: bool = True) -> Tuple[pd.DataFrame, LinkReport]:
        """Link crystal structures to the dataset and merge features.

        Args:
            df: The dataset DataFrame
            id_column: Column name to use as ID for matching with CIF filenames
            cif_folder: Path to folder containing .cif files
            replace_composition_features: If True, removes existing comp_* columns
                (from previous Magpie featurization) to avoid duplicates

        Returns:
            Tuple of (modified DataFrame, LinkReport)
        """
        cif_path = Path(cif_folder)
        df_out = df.copy()

        report = LinkReport(
            id_column=id_column,
            cif_folder=cif_folder,
            n_rows_total=len(df),
            n_matched=0,
            n_missing=0,
            n_duplicate_ids=0,
        )

        # Validate
        if id_column not in df.columns:
            report.errors.append(f"ID column '{id_column}' not found in dataset")
            return df_out, report

        if not cif_path.exists():
            report.errors.append(f"CIF folder '{cif_folder}' does not exist")
            return df_out, report

        # Check CIF availability
        availability = self.check_cif_availability(df, id_column, cif_folder)
        report.n_matched = len(availability['matched'])
        report.n_missing = len(availability['missing'])
        report.n_duplicate_ids = len(availability['duplicate_ids'])

        if availability['duplicate_ids']:
            report.warnings.append(
                f"{len(availability['duplicate_ids'])} IDs appear multiple times: "
                f"{', '.join(str(d) for d in availability['duplicate_ids'][:5])}"
                f"{'...' if len(availability['duplicate_ids']) > 5 else ''}. "
                f"Multiple rows with the same ID will receive identical crystal features."
            )

        if report.n_matched == 0:
            report.errors.append(
                f"No CIF files matched. Found {availability['n_cif_files']} CIF files "
                f"but none match the '{id_column}' values. "
                f"Ensure CIF filenames match the '{id_column}' values exactly."
            )
            return df_out, report

        if report.n_missing > 0:
            report.warnings.append(
                f"{report.n_missing} rows have no matching CIF file "
                f"(0.0 will be filled in crystal features)."
            )

        # Extract features for each matched CIF
        feature_names = self._crystal_feature_names
        features_list = []
        for idx, row in df.iterrows():
            mid = str(row[id_column]) if pd.notna(row[id_column]) else ''
            cif_file = cif_path / f"{mid}.cif"
            if cif_file.exists():
                feats = self.extract_cif_features(str(cif_file))
                if feats:
                    feats[id_column] = mid  # Keep ID for merge verification
                    features_list.append(feats)
                else:
                    features_list.append({k: 0.0 for k in feature_names})
            else:
                features_list.append({k: 0.0 for k in feature_names})

        # Create features DataFrame
        self._crystal_features_df = pd.DataFrame(features_list)
        report.crystal_features_added = feature_names.copy()

        # Handle composition feature replacement
        comp_cols = [c for c in df_out.columns if c.startswith('comp_')]
        if comp_cols and replace_composition_features:
            df_out = df_out.drop(columns=comp_cols)
            report.composition_features_replaced = True
            report.warnings.append(
                f"Removed {len(comp_cols)} existing composition feature columns "
                f"(comp_*) to avoid duplication with CIF-derived features."
            )

        # Add crystal features to the output DataFrame
        for col in feature_names:
            if col in self._crystal_features_df.columns:
                df_out[col] = self._crystal_features_df[col].values
            else:
                df_out[col] = 0.0

        # Mark ID column as non-feature (not for training)
        # This is handled by the data_manager via _id_column marker
        self._id_column = id_column

        self._link_report = report
        logger.info(f"Crystal linking complete: {report.n_matched}/{report.n_rows_total} matched")
        return df_out, report

    def get_id_column(self) -> Optional[str]:
        """Return the ID column used for linking, if any."""
        return getattr(self, '_id_column', None)

    def is_linked(self) -> bool:
        """Return True if crystal structures have been linked."""
        return self._link_report is not None and self._link_report.n_matched > 0

    def clear(self):
        """Clear the link state."""
        self._link_report = None
        self._crystal_features_df = None
        if hasattr(self, '_id_column'):
            delattr(self, '_id_column')

    def get_structure_info(self, cif_path: str) -> Dict[str, Any]:
        """Get summary info about a crystal structure for display.

        Returns dict with basic metadata (not full feature extraction).
        """
        if not self._has_pymatgen:
            return {'error': 'pymatgen not available'}
        try:
            struct = self.Structure.from_file(cif_path)
            comp = struct.composition
            elements = list(comp.element_composition.keys())
            return {
                'density': comp.density,
                'volume': struct.volume,
                'n_elements': len(elements),
                'elements': [str(el) for el in elements],
                'formula': comp.reduced_formula,
                'spacegroup': struct.get_space_group_info(),
                'n_features': len(self._crystal_feature_names),
            }
        except Exception as e:
            return {'error': str(e)}
