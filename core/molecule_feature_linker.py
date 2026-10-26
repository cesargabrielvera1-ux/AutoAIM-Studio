"""
Molecule Feature Linker for AutoAIM Studio v2.0.

Links molecular structure files (.xyz) with tabular datasets via an ID column,
extracting molecular descriptors (composition, geometry, topology)
and merging them into the dataset for training.

This module bridges the gap between tabular data and molecular structure
information, enabling models to learn from both compositional and
structural features simultaneously — analogous to CrystalFeatureExtractor
for CIF files.

v2.0: 100% Python pure (numpy only). Zero external dependencies.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .molecule_featurizer import MoleculeFeaturizer, list_molecule_feature_names


logger = logging.getLogger(__name__)


# ============================================================================
# MoleculeLinkReport — mirrors CrystalFeatureExtractor.LinkReport
# ============================================================================

@dataclass
class MoleculeLinkReport:
    """Report of the molecule structure linking process."""
    id_column: str
    xyz_folder: str
    n_rows_total: int
    n_matched: int
    n_missing: int
    n_duplicate_ids: int
    molecule_features_added: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    @property
    def match_rate(self) -> float:
        return self.n_matched / self.n_rows_total if self.n_rows_total > 0 else 0.0

    def to_text(self) -> str:
        lines = [
            "Molecule Structure Link Report",
            "=" * 40,
            f"ID Column            : {self.id_column}",
            f"XYZ Folder           : {self.xyz_folder}",
            f"Total rows           : {self.n_rows_total}",
            f"Matched XYZs         : {self.n_matched} ({self.match_rate*100:.1f}%)",
            f"Missing XYZs         : {self.n_missing}",
            f"Duplicate IDs        : {self.n_duplicate_ids}",
            f"Molecule features    : {len(self.molecule_features_added)}",
        ]
        if self.warnings:
            lines.extend(["", "Warnings:"] + [f"  - {w}" for w in self.warnings])
        if self.errors:
            lines.extend(["", "Errors:"] + [f"  - {e}" for e in self.errors])
        return "\n".join(lines)


# ============================================================================
# MoleculeFeatureLinker — mirrors CrystalFeatureExtractor API
# ============================================================================

class MoleculeFeatureLinker:
    """Links molecular structure files (.xyz) with tabular datasets.

    Extracts ~85 molecular features per .xyz file and merges them into the
    dataset. Follows the same API pattern as CrystalFeatureExtractor for
    consistency.

    v2.0: 100% Python pure (numpy only). No RDKit, no pymatgen required.
    """

    def __init__(self):
        self._link_report: Optional[MoleculeLinkReport] = None
        self._molecule_features_df: Optional[pd.DataFrame] = None
        self._featurizer = MoleculeFeaturizer()
        # Cache the feature names from a dummy call
        self._feature_names: List[str] = list_molecule_feature_names()

    @property
    def link_report(self) -> Optional[MoleculeLinkReport]:
        return self._link_report

    @property
    def molecule_features_df(self) -> Optional[pd.DataFrame]:
        return self._molecule_features_df

    @property
    def feature_names(self) -> List[str]:
        """List of molecule feature names that will be added."""
        return list(self._feature_names)

    # ──────────────────────────────────────────────────────────────────
    # ID candidate detection (mirrors CrystalFeatureExtractor)
    # ──────────────────────────────────────────────────────────────────

    def detect_id_candidates(self, df: pd.DataFrame) -> List[Tuple[str, int]]:
        """Detect columns that could serve as ID for linking with .xyz files.

        Returns list of (column_name, n_unique) sorted by uniqueness score.
        """
        candidates = []
        for col in df.columns:
            if col.lower() in ('target', 'y', 'label', 'class', 'index'):
                continue
            n_unique = df[col].nunique()
            score = n_unique / len(df)
            if score >= 0.5:
                candidates.append((col, n_unique, score))
        candidates.sort(key=lambda x: x[2], reverse=True)
        return [(c[0], c[1]) for c in candidates]

    # ──────────────────────────────────────────────────────────────────
    # XYZ availability check (mirrors check_cif_availability)
    # ──────────────────────────────────────────────────────────────────

    def check_xyz_availability(
        self, df: pd.DataFrame, id_column: str, xyz_folder: str
    ) -> Dict[str, Any]:
        """Check which IDs have corresponding .xyz files.

        Returns dict with 'matched', 'missing', 'duplicate_ids'.
        """
        xyz_path = Path(xyz_folder)
        if not xyz_path.exists():
            return {
                'matched': [],
                'missing': list(df[id_column].dropna().astype(str).unique()),
                'duplicate_ids': [],
                'error': 'Folder does not exist'
            }

        xyz_files = {f.stem for f in xyz_path.glob('*.xyz')}

        ids = df[id_column].dropna().astype(str).tolist()
        matched = [mid for mid in ids if mid in xyz_files]
        missing = [mid for mid in ids if mid not in xyz_files]

        id_counts = df[id_column].value_counts()
        duplicate_ids = id_counts[id_counts > 1].index.tolist()

        return {
            'matched': matched,
            'missing': missing,
            'duplicate_ids': duplicate_ids,
            'n_xyz_files': len(xyz_files),
            'n_ids': len(set(ids)),
        }

    # ──────────────────────────────────────────────────────────────────
    # Main link method (mirrors link_structures)
    # ──────────────────────────────────────────────────────────────────

    def link_structures(
        self,
        df: pd.DataFrame,
        id_column: str,
        xyz_folder: str
    ) -> Tuple[pd.DataFrame, MoleculeLinkReport]:
        """Link molecular structures to the dataset and merge features.

        Args:
            df: The dataset DataFrame
            id_column: Column name to use as ID for matching with .xyz filenames
            xyz_folder: Path to folder containing .xyz files

        Returns:
            Tuple of (modified DataFrame, MoleculeLinkReport)
        """
        xyz_path = Path(xyz_folder)
        df_out = df.copy()

        report = MoleculeLinkReport(
            id_column=id_column,
            xyz_folder=xyz_folder,
            n_rows_total=len(df),
            n_matched=0,
            n_missing=0,
            n_duplicate_ids=0,
        )

        # Validate
        if id_column not in df.columns:
            report.errors.append(f"ID column '{id_column}' not found in dataset")
            return df_out, report

        if not xyz_path.exists():
            report.errors.append(f"XYZ folder '{xyz_folder}' does not exist")
            return df_out, report

        # Check XYZ availability
        availability = self.check_xyz_availability(df, id_column, xyz_folder)
        report.n_matched = len(availability['matched'])
        report.n_missing = len(availability['missing'])
        report.n_duplicate_ids = len(availability['duplicate_ids'])

        if availability['duplicate_ids']:
            report.warnings.append(
                f"{len(availability['duplicate_ids'])} IDs appear multiple times: "
                f"{', '.join(str(d) for d in availability['duplicate_ids'][:5])}"
                f"{'...' if len(availability['duplicate_ids']) > 5 else ''}. "
                f"Multiple rows with the same ID will receive identical molecule features."
            )

        if report.n_matched == 0:
            report.errors.append(
                f"No .xyz files matched. Found {availability['n_xyz_files']} .xyz files "
                f"but none match the '{id_column}' values. "
                f"Ensure .xyz filenames match the '{id_column}' values exactly."
            )
            return df_out, report

        if report.n_missing > 0:
            report.warnings.append(
                f"{report.n_missing} rows have no matching .xyz file "
                f"(NaN will be filled in molecule features)."
            )

        # Extract features for each row
        features_list: List[Dict[str, float]] = []
        for idx, row in df.iterrows():
            mid = str(row[id_column]) if pd.notna(row[id_column]) else ''
            xyz_file = xyz_path / f"{mid}.xyz"
            if xyz_file.exists():
                try:
                    feat_df = self._featurizer.featurize_file(xyz_file)
                    # Drop source_file column, keep only numeric features
                    feat_row = {
                        c: float(feat_df[c].iloc[0])
                        for c in feat_df.columns
                        if c != 'mol_source_file'
                        and (c.startswith('mol_') or c.startswith('has_'))
                    }
                    features_list.append(feat_row)
                except Exception as e:
                    logger.warning(f"Failed to featurize {xyz_file}: {e}")
                    features_list.append(self._empty_features())
            else:
                features_list.append(self._empty_features())

        # Create features DataFrame
        self._molecule_features_df = pd.DataFrame(features_list)
        report.molecule_features_added = [
            c for c in self._molecule_features_df.columns
            if c.startswith('mol_') or c.startswith('has_')
        ]

        # Add molecule features to output DataFrame (use concat to avoid fragmentation)
        mol_cols = [c for c in self._molecule_features_df.columns
                    if c.startswith('mol_') or c.startswith('has_')]
        if mol_cols:
            df_out = pd.concat([
                df_out.reset_index(drop=True),
                self._molecule_features_df[mol_cols].reset_index(drop=True)
            ], axis=1)

        # Mark ID column
        self._id_column = id_column

        self._link_report = report
        logger.info(
            f"Molecule linking complete: {report.n_matched}/{report.n_rows_total} matched, "
            f"{len(report.molecule_features_added)} features added"
        )
        return df_out, report

    # ──────────────────────────────────────────────────────────────────
    # Utility methods (mirror CrystalFeatureExtractor)
    # ──────────────────────────────────────────────────────────────────

    def get_id_column(self) -> Optional[str]:
        """Return the ID column used for linking, if any."""
        return getattr(self, '_id_column', None)

    def is_linked(self) -> bool:
        """Return True if molecule structures have been linked."""
        return self._link_report is not None and self._link_report.n_matched > 0

    def clear(self):
        """Clear the link state."""
        self._link_report = None
        self._molecule_features_df = None
        if hasattr(self, '_id_column'):
            delattr(self, '_id_column')

    def _empty_features(self) -> Dict[str, float]:
        """Return a feature dict full of NaN for missing molecules."""
        return {name: np.nan for name in self._feature_names}
