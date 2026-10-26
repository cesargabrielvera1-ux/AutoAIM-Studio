"""Dataset export utilities for AutoAIM Studio.

v2.0: Provides a clean, testable interface for exporting featurized datasets
to CSV. Shared by the Data Tab and Predict Tab.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


class DatasetExporter:
    """Export featurized datasets to CSV with feature breakdown reporting."""

    # Feature prefixes by category
    CRYSTAL_PREFIXES = ("crystal_",)
    MOLECULAR_PREFIXES = ("mol_",)
    FUNCTIONAL_GROUP_PREFIXES = ("has_",)
    MAGPIE_PREFIXES = (
        "atomic_", "avg_", "Number", "Electronegativity",
        "ThermalConductivity", "BulkModulus", "ShearModulus",
        "band_center_", "fractional_", "weight_fraction_",
    )

    @staticmethod
    def build_feature_breakdown(df: pd.DataFrame) -> List[str]:
        """Build a human-readable breakdown of feature categories in the dataset.

        Args:
            df: The featurized DataFrame.

        Returns:
            List of strings like "  • Crystal: 45" describing each category.
        """
        breakdown = []
        n_crystal = sum(1 for c in df.columns if c.startswith(DatasetExporter.CRYSTAL_PREFIXES))
        n_mol = sum(1 for c in df.columns if c.startswith(DatasetExporter.MOLECULAR_PREFIXES))
        n_has = sum(1 for c in df.columns if c.startswith(DatasetExporter.FUNCTIONAL_GROUP_PREFIXES))
        n_magpie = sum(
            1 for c in df.columns
            if any(c.startswith(p) for p in DatasetExporter.MAGPIE_PREFIXES)
        )

        if n_crystal > 0:
            breakdown.append(f"  • Crystal: {n_crystal}")
        if n_mol > 0:
            breakdown.append(f"  • Molecular: {n_mol}")
        if n_has > 0:
            breakdown.append(f"  • Functional groups: {n_has}")
        if n_magpie > 0:
            breakdown.append(f"  • Composition (Magpie): {n_magpie}")
        if not breakdown:
            breakdown.append("  • Original tabular features")

        return breakdown

    @staticmethod
    def suggest_filename(file_path: Optional[str]) -> str:
        """Suggest an export filename based on the original file path.

        Args:
            file_path: Original dataset file path, or None.

        Returns:
            Suggested filename like "my_data_featurized.csv" or "autoaim_dataset.csv".
        """
        if file_path:
            return f"{Path(file_path).stem}_featurized.csv"
        return "autoaim_dataset.csv"

    @staticmethod
    def export_to_csv(df: pd.DataFrame, file_path: str) -> Tuple[int, int]:
        """Export a DataFrame to CSV with intelligent index handling.

        Args:
            df: DataFrame to export.
            file_path: Destination CSV path.

        Returns:
            Tuple of (n_rows, n_cols) written.

        Raises:
            ValueError: If df is empty or file_path is invalid.
            OSError: If file cannot be written.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty — nothing to export.")

        path = Path(file_path)
        if not path.parent.exists():
            raise ValueError(f"Directory does not exist: {path.parent}")

        # Determine whether to include index
        has_meaningful_index = (
            df.index.name is not None
            or (len(df) > 0 and isinstance(df.index[0], str))
        )

        if has_meaningful_index and df.index.name:
            df.to_csv(file_path, index=True, index_label=df.index.name)
        elif has_meaningful_index:
            df.to_csv(file_path, index=True)
        else:
            df.to_csv(file_path, index=False)

        return len(df), len(df.columns)

    @staticmethod
    def get_export_summary(df: pd.DataFrame, file_path: str) -> Dict[str, str]:
        """Build a complete export summary for display to the user.

        Args:
            df: The exported DataFrame.
            file_path: Path where the file was saved.

        Returns:
            Dict with keys: file, rows, columns, breakdown (newline-joined string).
        """
        n_rows, n_cols = len(df), len(df.columns)
        breakdown_lines = DatasetExporter.build_feature_breakdown(df)

        return {
            "file": str(file_path),
            "rows": str(n_rows),
            "columns": str(n_cols),
            "breakdown": "\n".join(breakdown_lines),
        }
