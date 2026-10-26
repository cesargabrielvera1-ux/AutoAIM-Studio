"""Group-label generation for grouped cross-validation.

v2.0: Provides the four grouping strategies for Group K-Fold / Leave-One-Group-Out:
  - Chemical system   (set of elements present, e.g. all Y-Ba-Cu-O together)
  - Exact/reduced formula (same stoichiometry together)
  - Clustering        (agglomerative clustering on element fractions)
  - Arbitrary column  (any user-selected categorical column: batch, lab, ...)

All functions are pure (no GUI, no global state) and reuse CompositionParser
so no new third-party dependency is introduced (matminer is NOT required).

Every generator returns:
    (labels, info)
where ``labels`` is a ``np.ndarray`` of per-sample group labels (dtype object
or int) aligned with the input rows, and ``info`` is a dict of diagnostics:
    {
        'n_groups': int,
        'group_sizes': {label: int, ...},
        'min_size': int, 'max_size': int, 'mean_size': float,
        'n_singletons': int,
        'source': str,
    }

Errors raise ``GroupingError`` (a ValueError subclass) with a user-facing,
English message suitable for display in a QMessageBox.
"""

from __future__ import annotations

from math import gcd
from functools import reduce
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .feature_engineering import CompositionParser
from ..utils.logger import LoggerMixin


class GroupingError(ValueError):
    """Raised when group labels cannot be generated for CV."""
    pass


# ── Composition column detection ────────────────────────────────────────────

def detect_composition_column(df: pd.DataFrame) -> Optional[str]:
    """Detect the most likely composition/formula column in a DataFrame.

    Scores string columns by the fraction of values that parse as a valid
    chemical formula (via CompositionParser.is_valid_formula). Column-name
    hints ('composition', 'formula', 'chemical') get a small bonus.

    Returns:
        Column name, or None if no column looks like a composition column.
    """
    if df is None or df.empty:
        return None

    best_col: Optional[str] = None
    best_score = 0.0

    for col in df.columns:
        series = df[col]
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue
        sample = series.dropna().head(50).astype(str)
        if len(sample) == 0:
            continue
        frac_valid = float(np.mean([CompositionParser.is_valid_formula(v) for v in sample]))
        score = frac_valid
        name = str(col).lower()
        if any(h in name for h in ('composition', 'formula', 'chemical', 'compound')):
            score += 0.15  # name hint bonus
        if frac_valid >= 0.6 and score > best_score:
            best_score = score
            best_col = col

    return best_col


# ── Element-count helpers ───────────────────────────────────────────────────

def _integer_counts(formula: str, tol: float = 0.03, max_mult: int = 60) -> Optional[Dict[str, int]]:
    """Recover integer stoichiometry from CompositionParser fractions.

    CompositionParser.parse_formula normalizes to fractions, losing integer
    counts. We find the smallest multiplier that makes every fraction
    (near-)integer, which recovers the subscripts (Fe3O4 → Fe:3, O:4).
    Returns None if no clean integer ratio is found within ``max_mult``.
    """
    frac = CompositionParser.parse_formula(formula)
    if not frac:
        return None
    for m in range(1, max_mult + 1):
        scaled = {el: f * m for el, f in frac.items()}
        if all(abs(v - round(v)) <= tol for v in scaled.values()):
            return {el: max(1, int(round(v))) for el, v in scaled.items()}
    return None


def _reduced_formula(formula: str) -> Optional[str]:
    """Return the reduced (simplest-integer-ratio) formula, or None if invalid.

    Examples: 'Fe3O4' → 'Fe3O4'; 'Y2Ba4Cu6O14' → 'YBa2Cu3O7' (so different
    multiples of the same stoichiometry group together).
    """
    counts = _integer_counts(formula)
    if not counts:
        return None
    g = reduce(gcd, counts.values())
    if g <= 0:
        g = 1
    reduced = {el: counts[el] // g for el in sorted(counts)}
    parts = [el if n == 1 else f"{el}{n}" for el, n in sorted(reduced.items())]
    return ''.join(parts)


def _build_info(labels: np.ndarray, source: str) -> Dict:
    """Compute group diagnostics from a label array."""
    labels = np.asarray(labels)
    uniq, counts = np.unique(labels, return_counts=True)
    sizes = {str(u): int(c) for u, c in zip(uniq, counts)}
    return {
        'n_groups': int(len(uniq)),
        'group_sizes': sizes,
        'min_size': int(counts.min()) if len(counts) else 0,
        'max_size': int(counts.max()) if len(counts) else 0,
        'mean_size': float(counts.mean()) if len(counts) else 0.0,
        'n_singletons': int((counts == 1).sum()) if len(counts) else 0,
        'source': source,
    }


def _require_compositions(compositions: pd.Series, source: str) -> List[str]:
    """Validate a composition series and return the list of formula strings.

    Raises GroupingError if too many entries cannot be parsed.
    """
    if compositions is None:
        raise GroupingError(
            f"{source}: no composition column was provided. "
            "Load a dataset with a chemical-formula column first."
        )
    formulas = compositions.fillna('').astype(str).tolist()
    n_bad = sum(1 for f in formulas if not CompositionParser.is_valid_formula(f))
    if n_bad == len(formulas) and len(formulas) > 0:
        raise GroupingError(
            f"{source}: none of the {len(formulas)} values look like valid chemical "
            "formulas (e.g. 'Fe2O3'). Check that you selected the composition column."
        )
    if n_bad > 0:
        raise GroupingError(
            f"{source}: {n_bad} sample(s) do not parse as valid chemical formulas. "
            "Grouping requires every sample to have a valid composition. "
            "Please clean the column or choose a different grouping source."
        )
    return formulas


# ── 1. Chemical system ──────────────────────────────────────────────────────

def groups_from_chemical_system(compositions: pd.Series) -> Tuple[np.ndarray, Dict]:
    """Group by the set of elements present (leave-one-chemical-system-out).

    All compounds sharing the same element set (e.g. any Y-Ba-Cu-O
    stoichiometry) land in the same group regardless of proportions.
    """
    formulas = _require_compositions(compositions, "Chemical-system grouping")
    labels = []
    for f in formulas:
        frac = CompositionParser.parse_formula(f)
        labels.append('-'.join(sorted(frac.keys())))
    arr = np.array(labels, dtype=object)
    return arr, _build_info(arr, 'chemical_system')


# ── 2. Exact / reduced formula ──────────────────────────────────────────────

def groups_from_formula(compositions: pd.Series, reduced: bool = True) -> Tuple[np.ndarray, Dict]:
    """Group by chemical formula.

    Args:
        compositions: series of formula strings.
        reduced: if True (default) use the reduced formula so that
            'YBa2Cu3O7' and 'Y2Ba4Cu6O14' group together; if False, group by
            the exact (normalized) formula string.
    """
    formulas = _require_compositions(compositions, "Formula grouping")
    labels = []
    for f in formulas:
        if reduced:
            red = _reduced_formula(f)
            labels.append(red if red is not None else f.strip())
        else:
            labels.append(f.strip())
    arr = np.array(labels, dtype=object)
    return arr, _build_info(arr, 'formula_reduced' if reduced else 'formula_exact')


# ── 3. Clustering on element fractions ─────────────────────────────────────

def groups_from_clustering(compositions: pd.Series, n_clusters: int = 8) -> Tuple[np.ndarray, Dict]:
    """Group by agglomerative clustering over element-fraction vectors.

    Uses the same element fractions already computed for featurization, then
    sklearn AgglomerativeClustering (deterministic — no random seed needed,
    which keeps CV reproducible).
    """
    if n_clusters < 2:
        raise GroupingError("Clustering grouping needs at least 2 clusters.")
    formulas = _require_compositions(compositions, "Clustering grouping")

    fracs = [CompositionParser.parse_formula(f) for f in formulas]
    elements = sorted({el for fr in fracs for el in fr.keys()})
    if not elements:
        raise GroupingError("Clustering grouping: no elements could be parsed.")

    n = len(fracs)
    if n_clusters > n:
        raise GroupingError(
            f"Clustering grouping: n_clusters ({n_clusters}) exceeds the number of "
            f"samples ({n}). Reduce the cluster count."
        )

    # Build element-fraction matrix (n_samples x n_elements)
    X = np.zeros((n, len(elements)), dtype=float)
    col = {el: i for i, el in enumerate(elements)}
    for i, fr in enumerate(fracs):
        for el, v in fr.items():
            X[i, col[el]] = v

    try:
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=n_clusters)
        labels = clustering.fit_predict(X)
    except Exception as e:
        raise GroupingError(f"Clustering grouping failed: {e}")

    arr = np.array([f"cluster_{int(c)}" for c in labels], dtype=object)
    info = _build_info(arr, 'clustering')
    info['n_clusters'] = int(n_clusters)
    info['n_elements'] = int(len(elements))
    return arr, info


# ── 4. Arbitrary user column ───────────────────────────────────────────────

def groups_from_column(df: pd.DataFrame, column: str) -> Tuple[np.ndarray, Dict]:
    """Group by an arbitrary user-selected column (batch, lab, method, ...).

    Raises GroupingError on NaN values (we never silently drop samples).
    """
    if df is None or column not in df.columns:
        raise GroupingError(
            f"Column grouping: column '{column}' was not found in the dataset."
        )
    series = df[column]
    n_nan = int(series.isna().sum())
    if n_nan > 0:
        raise GroupingError(
            f"Column grouping: {n_nan} sample(s) have no value in column "
            f"'{column}'. Grouping requires every sample to have a group label. "
            "Please fill or remove those rows first."
        )
    labels = series.astype(str).tolist()
    arr = np.array(labels, dtype=object)
    info = _build_info(arr, 'column')
    info['column'] = column
    return arr, info


# ── Unified dispatcher ──────────────────────────────────────────────────────

class GroupGenerator(LoggerMixin):
    """Thin convenience wrapper that dispatches to the grouping strategies."""

    SOURCES = ('chemical_system', 'formula', 'cluster', 'column')

    def generate(
        self,
        source: str,
        df: Optional[pd.DataFrame] = None,
        compositions: Optional[pd.Series] = None,
        column: Optional[str] = None,
        n_clusters: int = 8,
        reduced: bool = True,
    ) -> Tuple[np.ndarray, Dict]:
        """Generate group labels for the requested source.

        Either ``compositions`` (a Series of formulas) or ``df``+composition
        detection is used for chemical sources; ``df``+``column`` for 'column'.
        """
        if source == 'column':
            if df is None:
                raise GroupingError("Column grouping requires the dataset DataFrame.")
            return groups_from_column(df, column)

        # Chemical sources need a composition series.
        if compositions is None:
            if df is None:
                raise GroupingError("Grouping requires compositions or a DataFrame.")
            col = detect_composition_column(df)
            if col is None:
                raise GroupingError(
                    "No composition/formula column was detected in the dataset. "
                    "Chemical grouping sources need a column of chemical formulas."
                )
            compositions = df[col]

        if source == 'chemical_system':
            return groups_from_chemical_system(compositions)
        if source == 'formula':
            return groups_from_formula(compositions, reduced=reduced)
        if source == 'cluster':
            return groups_from_clustering(compositions, n_clusters=n_clusters)

        raise GroupingError(f"Unknown grouping source: '{source}'")
