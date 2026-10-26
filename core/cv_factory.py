"""Single source of truth for cross-validation splitter construction.

v2.0: Centralizes what used to be built independently in trainer.py and
optimizer.py (x2). Supports five strategies:

    'kfold'      Standard K-Fold (classification uses StratifiedKFold on classes)
    'stratified' Quantile-stratified K-Fold for regression (target binned into
                 quantiles so every fold preserves the target distribution)
    'group'      Group K-Fold (a user-defined group never crosses train/test)
    'loo'        Leave-One-Out (each sample left out once; folds = N)
    'logo'       Leave-One-Group-Out (each group left out once; folds = n_groups)

The factory returns a :class:`CVPlan` describing exactly how to run CV:
    - splitter:      sklearn-compatible splitter (has .split / .get_n_splits)
    - split_labels:  labels to pass as ``groups=`` to sklearn CV helpers
                     (None unless a grouped strategy is used)
    - scoring:       scoring string for cross_val_score / GridSearchCV
    - use_oof:       True → use out-of-fold predictions + global metric
                     (required when a test fold can hold a single sample,
                     where R² is mathematically undefined)
    - n_splits_actual: real number of folds produced
    - diagnostics:   human-readable list of fallbacks/warnings applied
    - group_info:    grouping diagnostics (for grouped strategies)

Design rules enforced here (single decision point, not scattered):
  * Any REGRESSION strategy where a test fold may hold < 2 samples uses the
    out-of-fold path (global R² + per-unit MAE). R² needs >= 2 points.
  * Classification always uses the per-fold path (accuracy is defined for 1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..utils.logger import LoggerMixin


# ── Configuration ───────────────────────────────────────────────────────────

@dataclass
class CVConfig:
    """User-facing cross-validation configuration (strategy + parameters)."""
    strategy: str = 'kfold'            # kfold | stratified | group | loo | logo
    n_folds: int = 5
    n_bins: int = 5                    # quantile bins for 'stratified'
    # v2.0: reserved bin for a degenerate target value ('stratified' only).
    # When special_value is set, samples with |y - special_value| <=
    # special_tolerance form their own stratum; the rest are quantile-binned
    # into (n_bins - 1) bins. None disables the reserved bin (default).
    special_value: Optional[float] = None
    special_tolerance: float = 0.0     # 0.0 = exact match (float precision)
    group_source: Optional[str] = None # chemical_system | formula | cluster | column
    n_clusters: int = 8                # for group_source == 'cluster'
    group_column: Optional[str] = None # for group_source == 'column'
    reduced_formula: bool = True       # for group_source == 'formula'
    random_state: int = 42

    # Resolved at build time (not user-set)
    groups: Optional[np.ndarray] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize (without the group labels array) for export / manifests."""
        return {
            'strategy': self.strategy,
            'n_folds': self.n_folds,
            'n_bins': self.n_bins,
            'special_value': self.special_value,
            'special_tolerance': self.special_tolerance,
            'group_source': self.group_source,
            'n_clusters': self.n_clusters,
            'group_column': self.group_column,
            'reduced_formula': self.reduced_formula,
            'random_state': self.random_state,
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> 'CVConfig':
        """Rebuild a CVConfig from a serialized dict (e.g. OptimizationResult).
        Missing keys fall back to defaults; unknown keys are ignored."""
        if not d:
            return cls()
        return cls(
            strategy=d.get('strategy', 'kfold'),
            n_folds=d.get('n_folds', 5),
            n_bins=d.get('n_bins', 5),
            special_value=d.get('special_value'),
            special_tolerance=d.get('special_tolerance', 0.0),
            group_source=d.get('group_source'),
            n_clusters=d.get('n_clusters', 8),
            group_column=d.get('group_column'),
            reduced_formula=d.get('reduced_formula', True),
            random_state=d.get('random_state', 42),
        )


@dataclass
class CVPlan:
    """Everything needed to execute CV consistently across trainer/optimizers."""
    splitter: Any
    split_labels: Optional[np.ndarray]
    scoring: str
    use_oof: bool
    n_splits_actual: int
    diagnostics: List[str] = field(default_factory=list)
    group_info: Optional[Dict] = None
    strategy: str = 'kfold'


# ── Quantile-stratified splitter (regression) ───────────────────────────────

class QuantileStratifiedSplitter:
    """sklearn-compatible splitter that stratifies regression folds by target
    quantiles. The bin labels are precomputed here, so the splitter can ignore
    the (continuous) y that sklearn CV helpers pass to ``split``.

    Binning degeneracies are handled upstream by ``_quantile_bins``; this class
    assumes ``y_binned`` is already valid (every bin has >= n_splits members).
    """

    def __init__(self, n_splits: int, y_binned: np.ndarray, random_state: int = 42):
        from sklearn.model_selection import StratifiedKFold
        self._n_splits = int(n_splits)
        self._y_binned = np.asarray(y_binned)
        self._skf = StratifiedKFold(n_splits=self._n_splits, shuffle=True,
                                    random_state=random_state)

    def split(self, X, y=None, groups=None):
        # Ignore the continuous y — stratify on the precomputed quantile bins.
        return self._skf.split(X, self._y_binned)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self._n_splits


def _plain_quantile_bins(y: np.ndarray, n_bins: int, n_splits: int,
                         diagnostics: List[str]) -> np.ndarray:
    """Bin a continuous target into quantiles robust to ties.

    Ensures every resulting bin has at least ``n_splits`` members so that
    StratifiedKFold can place >= 1 sample of each bin in every fold. Reduces
    the number of bins (with a diagnostic) when ties or small N require it.
    """
    y = np.asarray(y, dtype=float)
    n_unique = len(np.unique(y))
    if n_unique < 2:
        raise ValueError(
            "Target is constant — quantile stratification is impossible. "
            "Use standard K-Fold."
        )

    # Cap bins by what the data can support: need >= n_splits per bin.
    max_bins_by_count = max(2, len(y) // max(1, n_splits))
    q = int(min(n_bins, n_unique, max_bins_by_count))

    # Reduce until qcut succeeds and min bin count >= n_splits.
    while q >= 2:
        try:
            binned = pd.qcut(y, q=q, labels=False, duplicates='drop')
        except Exception:
            q -= 1
            continue
        binned = np.asarray(binned)
        counts = np.bincount(binned)
        if len(counts) >= 2 and counts.min() >= n_splits:
            if q < n_bins:
                diagnostics.append(
                    f"Quantile bins reduced from {n_bins} to {q} "
                    f"(ties or limited data) so each fold sees every bin."
                )
            return binned
        q -= 1

    # Last resort: median split (2 bins) if it satisfies the constraint.
    binned = (y > np.median(y)).astype(int)
    if np.bincount(binned).min() >= n_splits:
        diagnostics.append("Quantile stratification fell back to a median (2-bin) split.")
        return binned

    raise ValueError(
        "Could not build quantile bins where every fold sees every bin "
        f"(N={len(y)}, folds={n_splits}). Use standard K-Fold instead."
    )


def _quantile_bins(y: np.ndarray, n_bins: int, n_splits: int,
                   diagnostics: List[str],
                   special_value: Optional[float] = None,
                   special_tolerance: float = 0.0) -> np.ndarray:
    """Bin a target into quantiles, optionally reserving a bin for a special value.

    v2.0: When ``special_value`` is set, every sample with
    ``|y - special_value| <= special_tolerance`` is assigned to a dedicated
    stratum (label 0), and the remaining samples are quantile-binned into
    ``n_bins - 1`` bins (labels 1..q). This isolates a degenerate point mass
    (e.g. Tc = 0 K for non-superconductors, Eg = 0 eV for metals) so every fold
    carries the same proportion of it.

    Falls back to plain quantile binning (with a diagnostic) if the special
    value is absent from the target.
    """
    y = np.asarray(y, dtype=float)

    # No reserved bin → plain quantile binning.
    if special_value is None:
        return _plain_quantile_bins(y, n_bins, n_splits, diagnostics)

    special_mask = np.abs(y - special_value) <= special_tolerance
    n_special = int(special_mask.sum())

    if n_special == 0:
        diagnostics.append(
            f"Reserved bin value {special_value} not found in the target; "
            "using plain quantile stratification."
        )
        return _plain_quantile_bins(y, n_bins, n_splits, diagnostics)

    if n_special < n_splits:
        raise ValueError(
            f"Cannot reserve a bin for {special_value}: only {n_special} sample(s) "
            f"match, but {n_splits} folds require at least {n_splits}. Increase the "
            "tolerance, reduce folds, or use plain quantile stratification."
        )

    # Bin the non-special samples into (n_bins - 1) quantile bins.
    y_rest = y[~special_mask]
    rest_bins = _plain_quantile_bins(y_rest, max(1, n_bins - 1), n_splits, diagnostics)

    # Combine: special stratum = 0, quantile bins = 1..q (shifted by 1).
    binned = np.zeros(len(y), dtype=int)
    binned[special_mask] = 0
    binned[~special_mask] = rest_bins + 1

    diagnostics.append(
        f"Reserved a stratum for target = {special_value} "
        f"({n_special} sample(s), {100.0 * n_special / len(y):.1f}%), stratified into every fold; "
        f"the remaining {len(y) - n_special} samples were binned into "
        f"{len(np.unique(rest_bins))} quantile(s)."
    )
    return binned


def detect_point_mass(y: np.ndarray, n_folds: int = 5) -> Optional[Dict[str, Any]]:
    """Detect a degenerate point mass worth a reserved stratification bin.

    A point mass is a single exact target value repeated far more than expected
    under a continuous distribution (exact ties have ~0 probability beyond
    measurement precision). This is the "property absent" pattern common in
    materials data (Tc = 0 K, Eg = 0 eV, E_hull = 0, activity = 0, ...).

    Trigger rule (two floors, both must hold):
      * Stratifiability floor: count >= 2 * n_folds (each fold gets >= 2).
      * Salience floor: count / N >= 5% (below this, the spike barely distorts CV).

    Returns a dict ``{'value', 'count', 'fraction'}`` for the strongest
    candidate, or None if no value qualifies.
    """
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < max(2, n_folds):
        return None

    # Round to a coarse precision to collapse float noise into exact ties.
    rounded = np.round(y, decimals=8)
    values, counts = np.unique(rounded, return_counts=True)
    if len(values) < 2:
        return None

    order = np.argsort(counts)[::-1]
    top_value, top_count = values[order[0]], int(counts[order[0]])

    # A point mass should dominate: top value far more frequent than runner-up.
    # (For a continuous target, distinct values have comparably small counts.)
    second_count = int(counts[order[1]]) if len(order) > 1 else 0

    min_count = max(2 * n_folds, int(np.ceil(0.05 * n)))
    if top_count >= min_count and top_count >= 2 * max(1, second_count):
        return {
            'value': float(top_value),
            'count': top_count,
            'fraction': top_count / n,
        }
    return None


# ── The factory ─────────────────────────────────────────────────────────────

class CVFactory(LoggerMixin):
    """Builds CV plans. One instance is cheap; methods are stateless."""

    STRATEGIES = ('kfold', 'stratified', 'group', 'loo', 'logo')

    # ── public API ──────────────────────────────────────────────────────────

    def build_plan(
        self,
        config: CVConfig,
        X: np.ndarray,
        y: np.ndarray,
        problem_type: str = 'regression',
    ) -> CVPlan:
        """Build a CVPlan for the given config and data.

        Args:
            config: CVConfig (groups already resolved for grouped strategies).
            X: feature matrix (only used for length here).
            y: target vector.
            problem_type: 'regression' or 'classification'.
        """
        from sklearn.model_selection import (
            KFold, StratifiedKFold, GroupKFold, LeaveOneOut, LeaveOneGroupOut,
        )

        n = len(y)
        strategy = config.strategy
        diagnostics: List[str] = []
        is_reg = problem_type == 'regression'

        if strategy not in self.STRATEGIES:
            raise ValueError(f"Unknown CV strategy: '{strategy}'")

        # ── K-Fold ─────────────────────────────────────────────────────────
        if strategy == 'kfold':
            if is_reg:
                splitter = KFold(n_splits=config.n_folds, shuffle=True,
                                 random_state=config.random_state)
            else:
                # Classification: stratify by class label (statistically correct).
                splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True,
                                           random_state=config.random_state)
            return CVPlan(splitter=splitter, split_labels=None,
                          scoring=self._scoring(is_reg, use_oof=False),
                          use_oof=False, n_splits_actual=config.n_folds,
                          diagnostics=diagnostics, strategy=strategy)

        # ── Quantile-stratified K-Fold (regression only) ───────────────────
        if strategy == 'stratified':
            if not is_reg:
                # Classification is already class-stratified under 'kfold'.
                diagnostics.append(
                    "Quantile stratification applies to regression only; "
                    "classification already stratifies by class under K-Fold."
                )
                splitter = StratifiedKFold(n_splits=config.n_folds, shuffle=True,
                                           random_state=config.random_state)
                return CVPlan(splitter=splitter, split_labels=None,
                              scoring=self._scoring(False, False), use_oof=False,
                              n_splits_actual=config.n_folds,
                              diagnostics=diagnostics, strategy=strategy)
            # v2.0: optional reserved bin for a degenerate target value
            # (e.g. Tc = 0 K, Eg = 0 eV). Isolates the point mass so every fold
            # carries the same proportion of it.
            y_binned = _quantile_bins(
                y, config.n_bins, config.n_folds, diagnostics,
                special_value=config.special_value,
                special_tolerance=config.special_tolerance,
            )
            splitter = QuantileStratifiedSplitter(config.n_folds, y_binned,
                                                  random_state=config.random_state)
            return CVPlan(splitter=splitter, split_labels=None,
                          scoring=self._scoring(True, False), use_oof=False,
                          n_splits_actual=config.n_folds,
                          diagnostics=diagnostics, strategy=strategy)

        # ── Leave-One-Out ──────────────────────────────────────────────────
        if strategy == 'loo':
            splitter = LeaveOneOut()
            use_oof = is_reg  # R² undefined for 1-sample test folds
            return CVPlan(splitter=splitter, split_labels=None,
                          scoring=self._scoring(is_reg, use_oof),
                          use_oof=use_oof, n_splits_actual=n,
                          diagnostics=diagnostics, strategy=strategy)

        # ── Grouped strategies need group labels ───────────────────────────
        groups = config.groups
        if groups is None:
            raise ValueError(
                f"CV strategy '{strategy}' requires group labels, but none were provided."
            )
        groups = np.asarray(groups)
        n_groups = len(np.unique(groups))

        if strategy == 'group':
            if n_groups < config.n_folds:
                raise ValueError(
                    f"Grouping produced {n_groups} group(s) but {config.n_folds} folds "
                    f"were requested. Reduce folds or choose a different grouping."
                )
            splitter = GroupKFold(n_splits=config.n_folds)
            # A fold's test size is the sum of its member groups' sizes. If any
            # fold could hold a single sample and this is regression, use OOF.
            fold_sizes = self._group_fold_test_sizes(groups, config.n_folds)
            use_oof = bool(is_reg and fold_sizes is not None and min(fold_sizes) < 2)
            if use_oof:
                diagnostics.append(
                    "Some group fold holds a single sample; using out-of-fold "
                    "predictions (global R² + per-group MAE) because R² needs ≥2 points."
                )
            return CVPlan(splitter=splitter, split_labels=groups,
                          scoring=self._scoring(is_reg, use_oof),
                          use_oof=use_oof, n_splits_actual=config.n_folds,
                          diagnostics=diagnostics,
                          group_info=self._group_info(groups, config), strategy=strategy)

        if strategy == 'logo':
            if n_groups < 3:
                raise ValueError(
                    f"Leave-One-Group-Out needs at least 3 groups, but only "
                    f"{n_groups} were found. Use Group K-Fold or a different grouping."
                )
            splitter = LeaveOneGroupOut()
            use_oof = is_reg  # a left-out group may hold a single sample
            return CVPlan(splitter=splitter, split_labels=groups,
                          scoring=self._scoring(is_reg, use_oof),
                          use_oof=use_oof, n_splits_actual=n_groups,
                          diagnostics=diagnostics,
                          group_info=self._group_info(groups, config), strategy=strategy)

        # Unreachable (strategies validated above)
        raise ValueError(f"Unhandled CV strategy: '{strategy}'")

    # ── scoring ─────────────────────────────────────────────────────────────

    @staticmethod
    def _scoring(is_regression: bool, use_oof: bool) -> str:
        """Return the sklearn scoring string.

        Regression on the OOF path (single-sample folds) uses MAE, which is
        defined for one sample; R² is not. Everything else uses R²/accuracy.
        """
        if is_regression:
            return 'neg_mean_absolute_error' if use_oof else 'r2'
        return 'accuracy'

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _group_fold_test_sizes(groups: np.ndarray, n_splits: int) -> Optional[List[int]]:
        """Approximate the test size of each GroupKFold fold.

        sklearn assigns whole groups to folds greedily by size; we reproduce
        the resulting fold test sizes exactly by running a throwaway split.
        """
        try:
            from sklearn.model_selection import GroupKFold
            n = len(groups)
            dummy = np.zeros((n, 1))
            gkf = GroupKFold(n_splits=n_splits)
            sizes = [len(test_idx) for _, test_idx in gkf.split(dummy, groups=groups)]
            return sizes
        except Exception:
            return None

    @staticmethod
    def _group_info(groups: np.ndarray, config: CVConfig) -> Dict:
        uniq, counts = np.unique(groups, return_counts=True)
        return {
            'n_groups': int(len(uniq)),
            'min_size': int(counts.min()),
            'max_size': int(counts.max()),
            'mean_size': float(counts.mean()),
            'n_singletons': int((counts == 1).sum()),
            'group_source': config.group_source,
            'group_column': config.group_column,
            'n_clusters': config.n_clusters if config.group_source == 'cluster' else None,
        }


# ── Module-level convenience ────────────────────────────────────────────────

_FACTORY = CVFactory()


def build_cv_plan(config: CVConfig, X, y, problem_type: str = 'regression') -> CVPlan:
    """Build a CVPlan (module-level convenience wrapper)."""
    return _FACTORY.build_plan(config, X, y, problem_type=problem_type)


# ── Validation layer (shared by GUI and backend — defense in depth) ────────

# Severity levels
BLOCK = 'block'   # Cannot run — hard stop
WARN = 'warn'     # Risky — require explicit user confirmation
INFO = 'info'     # Informational only


def validate_cv_config(
    config: CVConfig,
    n_samples: int,
    problem_type: str = 'regression',
    group_info: Optional[Dict] = None,
    y: Optional[np.ndarray] = None,
) -> List[Dict[str, str]]:
    """Validate a CV config against the dataset.

    Returns a list of issues, each ``{'severity': ..., 'message': ...}``.
    An empty list means the configuration is valid with no caveats.

    This function is intentionally GUI-free: both the widgets (before
    launching a thread) and the backend call it, so the rules live in one
    place. ``group_info`` is the diagnostics dict returned by a grouping
    function (n_groups, min/max sizes, n_singletons); it may be None if
    grouping has not been resolved yet. ``y`` is the target vector, needed to
    validate a reserved-bin (special value) for quantile stratification.
    """
    issues: List[Dict[str, str]] = []
    strategy = config.strategy
    is_reg = problem_type == 'regression'

    def add(sev, msg):
        issues.append({'severity': sev, 'message': msg})

    if n_samples < 2:
        add(BLOCK, "Dataset is too small for cross-validation (N < 2).")
        return issues

    # ── K-Fold / Stratified ────────────────────────────────────────────────
    if strategy in ('kfold', 'stratified'):
        if config.n_folds < 2:
            add(BLOCK, "CV folds must be at least 2.")
        if config.n_folds > n_samples:
            add(BLOCK,
                f"CV folds ({config.n_folds}) exceed the number of samples "
                f"({n_samples}). Reduce folds or use Leave-One-Out.")
        if strategy == 'stratified' and is_reg:
            if n_samples < config.n_folds * 2:
                add(WARN,
                    f"Stratified K-Fold with {config.n_folds} folds on only "
                    f"{n_samples} samples leaves very small folds. Consider "
                    "fewer folds or Leave-One-Out.")
            # v2.0: reserved-bin (special value) validation — data-grounded.
            if config.special_value is not None and y is not None:
                y_arr = np.asarray(y, dtype=float)
                n_special = int((np.abs(y_arr - config.special_value)
                                 <= config.special_tolerance).sum())
                pct = 100.0 * n_special / len(y_arr)
                if n_special == 0:
                    add(INFO,
                        f"The reserved value {config.special_value} does not occur "
                        "in the target. Plain quantile stratification will be used.")
                elif n_special < config.n_folds:
                    add(BLOCK,
                        f"Cannot reserve a bin for {config.special_value}: only "
                        f"{n_special} sample(s) match, but {config.n_folds} folds "
                        f"require at least {config.n_folds} (one per fold). Increase "
                        "the tolerance, reduce folds, or use plain stratification.")
                elif pct > 50.0:
                    add(WARN,
                        f"The reserved value {config.special_value} matches "
                        f"{n_special} of {len(y_arr)} samples ({pct:.1f}%) — it is the "
                        "majority. Only a few quantile bins remain for the continuous "
                        "part, which reduces the benefit of stratification.")
                elif pct < 5.0:
                    add(WARN,
                        f"The reserved value {config.special_value} matches only "
                        f"{n_special} of {len(y_arr)} samples ({pct:.1f}%). A reserved "
                        "bin is most useful for a dominant point mass (e.g. Tc = 0 K "
                        "for non-superconductors, Eg = 0 eV for metals). Confirm this "
                        "value is a meaningful 'property absent' marker in your data.")

    # ── LOO ────────────────────────────────────────────────────────────────
    if strategy == 'loo':
        if n_samples > 100:
            add(BLOCK,
                f"Leave-One-Out CV is only available for datasets with ≤ 100 "
                f"samples (yours has {n_samples}). It would train {n_samples} "
                "models. Please use K-Fold instead.")
        elif n_samples > 30:
            add(WARN,
                f"Leave-One-Out will train {n_samples} models (one per sample), "
                "which may take significant time. K-Fold is generally "
                "recommended for N > 30. Proceed only if you accept the cost.")

    # ── Grouped strategies ─────────────────────────────────────────────────
    if strategy in ('group', 'logo'):
        if group_info is None:
            add(BLOCK, "Group labels have not been generated yet.")
        else:
            n_groups = group_info.get('n_groups', 0)
            min_size = group_info.get('min_size', 0)
            max_size = group_info.get('max_size', 0)
            n_singletons = group_info.get('n_singletons', 0)

            if n_groups < 2:
                add(BLOCK,
                    f"Grouping produced only {n_groups} group. At least 2 "
                    "distinct groups are required.")

            if strategy == 'group':
                if n_groups < config.n_folds:
                    add(BLOCK,
                        f"Grouping produced {n_groups} group(s) but "
                        f"{config.n_folds} folds were requested. Reduce folds "
                        "to ≤ the number of groups, or change the grouping.")
            if strategy == 'logo':
                if n_groups < 3:
                    add(BLOCK,
                        f"Leave-One-Group-Out needs at least 3 groups, but only "
                        f"{n_groups} were found. Use Group K-Fold instead.")
                if n_groups > 100:
                    add(BLOCK,
                        f"Leave-One-Group-Out would train {n_groups} models "
                        "(one per group). This is only supported for ≤ 100 "
                        "groups. Use Group K-Fold with fewer folds instead.")
                elif n_groups > 30:
                    add(WARN,
                        f"Leave-One-Group-Out will train {n_groups} models "
                        "(one per group), which may take significant time. "
                        "Group K-Fold with fewer folds is recommended. Proceed "
                        "only if you accept the cost.")

            # Imbalance warning (any grouped strategy)
            if n_groups >= 2 and min_size > 0 and max_size / max(1, min_size) > 10:
                add(WARN,
                    f"Groups are highly imbalanced (largest={max_size}, "
                    f"smallest={min_size} samples). Some folds will be much "
                    "larger than others, which can make CV scores uneven.")

            # Singleton groups
            if n_singletons > 0:
                add(INFO,
                    f"{n_singletons} group(s) contain a single sample. In "
                    "regression these folds use MAE (R² needs ≥ 2 points).")

    return issues


def has_blocking_issue(issues: List[Dict[str, str]]) -> bool:
    """True if any issue is a hard block."""
    return any(i['severity'] == BLOCK for i in issues)


# ── Shared CV metrics assembly (for non-sklearn models: NN, ensembles) ──────

def assemble_cv_metrics(
    *,
    y: np.ndarray,
    plan: CVPlan,
    problem_type: str,
    oof_preds: Optional[np.ndarray] = None,
    fold_test_scores: Optional[List[float]] = None,
    fold_train_scores: Optional[List[float]] = None,
    fold_sizes: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Assemble a cv_metrics dict consistently for models that can't use
    sklearn's cross_validate (neural networks and ensembles retrain / recombine
    per fold manually).

    This centralizes the out-of-fold vs per-fold decision so NN and ensemble CV
    produce exactly the same structure as the sklearn path in
    ``ModelTrainer._cross_validate``:

      * OOF path (plan.use_oof, regression): global R² over all out-of-fold
        predictions + per-unit MAE (per sample for LOO, per group for LOGO).
      * Per-fold path: per-fold test/train scores (R² or accuracy), computed by
        the caller.

    Args:
        y: target vector for the data being cross-validated.
        plan: the CVPlan from build_cv_plan.
        problem_type: 'regression' or 'classification'.
        oof_preds: full-length out-of-fold predictions (required for OOF path).
        fold_test_scores: per-fold test scores (per-fold path).
        fold_train_scores: per-fold train scores (per-fold path).
        fold_sizes: optional per-fold test sizes (uneven group folds).
    """
    from sklearn.metrics import r2_score

    is_reg = problem_type == 'regression'
    y = np.asarray(y)

    out: Dict[str, Any] = {
        'cv_diagnostics': list(plan.diagnostics),
        'fold_test_sizes': fold_sizes or [],
    }
    if plan.group_info:
        out['group_info'] = plan.group_info

    # ── OOF path: single-sample-capable folds in regression ────────────────
    if plan.use_oof and is_reg:
        if oof_preds is None:
            raise ValueError("OOF path requires out-of-fold predictions.")
        oof_preds = np.asarray(oof_preds)
        out['r2_global'] = float(r2_score(y, oof_preds))
        abs_err = np.abs(y - oof_preds)
        if plan.strategy == 'loo':
            out['test_score'] = abs_err.tolist()
            out['fold_test_sizes'] = [1] * len(y)
        else:
            # LOGO / grouped OOF: one MAE per group.
            groups = np.asarray(plan.split_labels)
            test_scores, sizes = [], []
            for g in pd.unique(groups):
                mask = groups == g
                test_scores.append(float(np.mean(abs_err[mask])))
                sizes.append(int(mask.sum()))
            out['test_score'] = test_scores
            out['fold_test_sizes'] = sizes
        out['test_score_metric'] = 'mae'
        out['train_score'] = []
        return out

    # ── Per-fold path ───────────────────────────────────────────────────────
    out['test_score'] = list(fold_test_scores or [])
    out['train_score'] = list(fold_train_scores or [])
    return out
