"""Reusable cross-validation configuration widget.

v2.0: Single shared component that centralizes ALL cross-validation controls
(strategy, folds, quantile bins, group source, clusters, group column) and the
associated info label. It is embedded identically in:

  * Training Tab
  * Optimization Tab
  * Manual Parameter Dialog

This eliminates the triplicated LOO-CV UI logic that previously lived in each
of those three places, and adds the new strategies (quantile-stratified,
Group K-Fold, Leave-One-Group-Out) without any code duplication.

The widget is GUI-only: it builds a :class:`~app.core.cv_factory.CVConfig`
and resolves group labels through :class:`~app.core.grouping.GroupGenerator`.
Validation is performed by the host tab via ``cv_factory.validate_cv_config``.

Fonts follow the app-wide convention: controls inherit the global tooltip
style (15 pt, wrapped); the inline info label uses 8 pt orange (#ff9800).
"""

from __future__ import annotations

from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

from PyQt6.QtWidgets import (
    QWidget, QGridLayout, QLabel, QComboBox, QSpinBox, QDoubleSpinBox, QCheckBox,
)
from PyQt6.QtCore import pyqtSignal

from ..core.cv_factory import CVConfig, detect_point_mass
from ..core.grouping import GroupGenerator, GroupingError, detect_composition_column


class CVConfigWidget(QWidget):
    """Cross-validation configuration controls with conditional visibility."""

    configChanged = pyqtSignal()

    # Human-readable strategy labels (index order matches _STRATEGY_KEYS).
    _STRATEGY_KEYS = ['kfold', 'stratified', 'group', 'loo', 'logo']
    _STRATEGY_LABELS = [
        "K-Fold",
        "Stratified K-Fold (quantiles)",
        "Group K-Fold",
        "Leave-One-Out (LOO)",
        "Leave-One-Group-Out (LOGO)",
    ]

    _GROUP_SOURCE_KEYS = ['chemical_system', 'formula', 'cluster', 'column']
    _GROUP_SOURCE_LABELS = [
        "Chemical system (element set)",
        "Exact formula",
        "Clustering (advanced)",
        "Dataset column",
    ]

    def __init__(self, parent: Optional[QWidget] = None, show_title: bool = False):
        super().__init__(parent)
        self._generator = GroupGenerator()
        self._df: Optional[pd.DataFrame] = None
        self._composition_col: Optional[str] = None
        self._problem_type = 'regression'
        self._target: Optional[np.ndarray] = None  # for point-mass detection
        self._build_ui()
        self._on_strategy_changed()  # set initial visibility

    # ── UI construction ─────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(4)
        row = 0

        # CV Strategy
        layout.addWidget(QLabel("CV Strategy:"), row, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(self._STRATEGY_LABELS)
        self.strategy_combo.setToolTip(
            "K-Fold: standard random folds. Recommended for most datasets (N > 30).\n"
            "Stratified (quantiles): each fold preserves the target distribution. "
            "Recommended for skewed targets (regression only).\n"
            "Group K-Fold: samples from the same group never cross the train/test "
            "boundary. Prevents chemical or batch data leakage.\n"
            "LOO: leave each sample out once. For very small datasets (N ≤ 30).\n"
            "LOGO: leave each group out once. For grouped data with few groups."
        )
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        layout.addWidget(self.strategy_combo, row, 1)
        row += 1

        # CV Folds
        self.folds_label = QLabel("CV Folds:")
        layout.addWidget(self.folds_label, row, 0)
        self.folds_spin = QSpinBox()
        self.folds_spin.setRange(2, 10)
        self.folds_spin.setValue(5)
        self.folds_spin.setToolTip(
            "Number of cross-validation folds. Each model is trained K times on "
            "different train/validation splits. Higher = more robust estimate, "
            "but takes K times longer."
        )
        self.folds_spin.valueChanged.connect(self.configChanged)
        layout.addWidget(self.folds_spin, row, 1)
        row += 1

        # Quantile bins (stratified only)
        self.bins_label = QLabel("Quantile Bins:")
        self.bins_spin = QSpinBox()
        self.bins_spin.setRange(2, 20)
        self.bins_spin.setValue(5)
        self.bins_spin.setToolTip(
            "Number of target quantiles used for stratification. Each bin must "
            "have at least as many samples as folds. Reduced automatically if "
            "the target has many repeated values."
        )
        self.bins_spin.valueChanged.connect(self.configChanged)
        layout.addWidget(self.bins_label, row, 0)
        layout.addWidget(self.bins_spin, row, 1)
        row += 1

        # v2.0: Reserved bin for a degenerate target value (stratified only).
        # Isolates a point mass (e.g. Tc = 0 K, Eg = 0 eV) into its own stratum
        # so every fold carries the same proportion of it.
        self.special_check = QCheckBox("Reserve a bin for an exact value")
        self.special_check.setToolTip(
            "Enable to isolate one exact target value into its own stratum. "
            "Use this for a degenerate 'property absent' marker such as "
            "Tc = 0 K (non-superconductors) or Eg = 0 eV (metals). "
            "Every fold will then contain the same proportion of that value."
        )
        self.special_check.toggled.connect(self._on_special_toggled)
        layout.addWidget(self.special_check, row, 0, 1, 2)
        row += 1

        self.special_value_label = QLabel("Special Value:")
        self.special_value_spin = QDoubleSpinBox()
        self.special_value_spin.setRange(-1e9, 1e9)
        self.special_value_spin.setDecimals(6)
        self.special_value_spin.setValue(0.0)
        self.special_value_spin.setToolTip(
            "The exact target value to isolate (e.g. 0.0). All samples with "
            "|target - value| <= tolerance are placed in the reserved bin."
        )
        self.special_value_spin.valueChanged.connect(self.configChanged)
        layout.addWidget(self.special_value_label, row, 0)
        layout.addWidget(self.special_value_spin, row, 1)
        row += 1

        self.special_tol_label = QLabel("Tolerance:")
        self.special_tol_spin = QDoubleSpinBox()
        self.special_tol_spin.setRange(0.0, 1e6)
        self.special_tol_spin.setDecimals(6)
        self.special_tol_spin.setValue(0.0)
        self.special_tol_spin.setToolTip(
            "How close a value must be to count as the special value. "
            "0.0 = exact match (recommended). Increase only if your data stores "
            "the value as 'below detection threshold' (e.g. 0.01 instead of 0)."
        )
        self.special_tol_spin.valueChanged.connect(self.configChanged)
        layout.addWidget(self.special_tol_label, row, 0)
        layout.addWidget(self.special_tol_spin, row, 1)
        row += 1

        # v2.0: detection suggestion label (shown when a point mass is found).
        self.special_suggest_label = QLabel("")
        self.special_suggest_label.setStyleSheet("color: #00bcd4; font-size: 8pt;")
        self.special_suggest_label.setWordWrap(True)
        layout.addWidget(self.special_suggest_label, row, 0, 1, 2)
        row += 1

        # Group source (group / logo only)
        self.group_source_label = QLabel("Group By:")
        self.group_source_combo = QComboBox()
        self.group_source_combo.addItems(self._GROUP_SOURCE_LABELS)
        self.group_source_combo.setToolTip(
            "How to define the groups that must not cross the train/test split.\n"
            "Chemical system: same set of elements (e.g. all Y-Ba-Cu-O) — the "
            "standard anti-leakage split for materials data.\n"
            "Exact formula: same stoichiometry together.\n"
            "Clustering: agglomerative clusters over element fractions (advanced).\n"
            "Dataset column: any column (batch, lab, synthesis method, ...)."
        )
        self.group_source_combo.currentIndexChanged.connect(self._on_group_source_changed)
        layout.addWidget(self.group_source_label, row, 0)
        layout.addWidget(self.group_source_combo, row, 1)
        row += 1

        # Cluster count (cluster source only)
        self.clusters_label = QLabel("Clusters:")
        self.clusters_spin = QSpinBox()
        self.clusters_spin.setRange(2, 50)
        self.clusters_spin.setValue(8)
        self.clusters_spin.setToolTip(
            "Number of clusters for the agglomerative clustering of element "
            "fractions. Deterministic (no random seed). Choose a value well "
            "below the number of samples."
        )
        self.clusters_spin.valueChanged.connect(self.configChanged)
        layout.addWidget(self.clusters_label, row, 0)
        layout.addWidget(self.clusters_spin, row, 1)
        row += 1

        # Dataset column (column source only)
        self.column_label = QLabel("Group Column:")
        self.column_combo = QComboBox()
        self.column_combo.setToolTip(
            "Column whose equal values define a group. Every sample must have a "
            "value (rows with missing group labels are rejected). Examples: "
            "synthesis batch, laboratory, source, year."
        )
        self.column_combo.currentIndexChanged.connect(self.configChanged)
        layout.addWidget(self.column_label, row, 0)
        layout.addWidget(self.column_combo, row, 1)
        row += 1

        # Info label (8 pt orange, wrapped) — always present.
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #ff9800; font-size: 8pt;")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label, row, 0, 1, 2)

    # ── Public API ──────────────────────────────────────────────────────────

    def set_dataframe(self, df: Optional[pd.DataFrame]):
        """Provide the loaded dataset so the widget can populate the group
        column combo and enable/disable chemical grouping sources."""
        self._df = df
        self._composition_col = detect_composition_column(df) if df is not None else None

        # Populate the arbitrary-column combo with all columns.
        self.column_combo.blockSignals(True)
        self.column_combo.clear()
        if df is not None:
            for c in df.columns:
                self.column_combo.addItem(str(c))
        self.column_combo.blockSignals(False)

        self._update_group_source_availability()

    def set_problem_type(self, problem_type: str):
        """Enable/disable strategies that only apply to regression."""
        self._problem_type = problem_type
        # Quantile stratification is regression-only.
        idx = self._STRATEGY_KEYS.index('stratified')
        model = self.strategy_combo.model()
        item = model.item(idx)
        if problem_type == 'classification':
            item.setEnabled(False)
            if self.strategy_combo.currentIndex() == idx:
                self.strategy_combo.setCurrentIndex(0)
        else:
            item.setEnabled(True)
        self._on_strategy_changed()

    def current_strategy(self) -> str:
        return self._STRATEGY_KEYS[self.strategy_combo.currentIndex()]

    def current_group_source(self) -> str:
        return self._GROUP_SOURCE_KEYS[self.group_source_combo.currentIndex()]

    def set_target_values(self, y: Optional[np.ndarray]):
        """Provide the target vector so the widget can detect a degenerate point
        mass (e.g. many samples with target = 0) and suggest a reserved bin."""
        self._target = None if y is None else np.asarray(y, dtype=float)
        self._update_special_suggestion()

    def get_config(self) -> CVConfig:
        """Build a CVConfig from the current control values (groups unresolved)."""
        strategy = self.current_strategy()
        is_grouped = strategy in ('group', 'logo')
        source = self.current_group_source() if is_grouped else None
        # v2.0: reserved bin only applies to stratified regression, and only
        # when the user enabled the checkbox.
        use_special = (
            strategy == 'stratified'
            and self._problem_type == 'regression'
            and self.special_check.isChecked()
        )
        return CVConfig(
            strategy=strategy,
            n_folds=self.folds_spin.value(),
            n_bins=self.bins_spin.value(),
            special_value=(self.special_value_spin.value() if use_special else None),
            special_tolerance=(self.special_tol_spin.value() if use_special else 0.0),
            group_source=source,
            n_clusters=self.clusters_spin.value(),
            group_column=(self.column_combo.currentText()
                          if (is_grouped and source == 'column') else None),
            reduced_formula=True,
        )

    def resolve_groups(self) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """Resolve group labels over the FULL dataset for grouped strategies.

        Returns (groups, group_info) or (None, None) for non-grouped
        strategies. Raises GroupingError with a user-facing message on failure.
        """
        config = self.get_config()
        if config.strategy not in ('group', 'logo'):
            return None, None
        if self._df is None:
            raise GroupingError("No dataset loaded. Load data before grouped CV.")

        compositions = None
        if config.group_source in ('chemical_system', 'formula', 'cluster'):
            if self._composition_col is None:
                raise GroupingError(
                    "No composition/formula column was detected in the dataset. "
                    "Chemical grouping sources need a column of chemical formulas."
                )
            compositions = self._df[self._composition_col]

        groups, info = self._generator.generate(
            source=config.group_source,
            df=self._df,
            compositions=compositions,
            column=config.group_column,
            n_clusters=config.n_clusters,
            reduced=config.reduced_formula,
        )
        return groups, info

    # ── Internal slots ──────────────────────────────────────────────────────

    def _on_strategy_changed(self):
        strategy = self.current_strategy()
        is_stratified = strategy == 'stratified'
        is_grouped = strategy in ('group', 'logo')
        is_loo_like = strategy in ('loo', 'logo')

        # Folds spinner: disabled for leave-one-out strategies.
        self.folds_spin.setEnabled(not is_loo_like)
        self.folds_label.setEnabled(not is_loo_like)

        # Conditional rows.
        for w in (self.bins_label, self.bins_spin):
            w.setVisible(is_stratified)
        # v2.0: reserved-bin controls visible only for stratified regression.
        show_special = is_stratified and self._problem_type == 'regression'
        self.special_check.setVisible(show_special)
        self._update_special_row_visibility()
        if show_special:
            self._update_special_suggestion()
        else:
            self.special_suggest_label.setText("")
        for w in (self.group_source_label, self.group_source_combo):
            w.setVisible(is_grouped)
        self._on_group_source_changed(emit=False)

        # Info text.
        info = {
            'kfold': "",
            'stratified': ("ℹ Each fold preserves the target distribution. "
                           "Recommended for skewed targets (regression)."),
            'group': ("ℹ Samples from the same group never cross the train/test "
                      "boundary. Prevents chemical/batch data leakage."),
            'loo': ("ℹ With LOO-CV, each sample is left out once. The number of "
                    "folds is automatically N (number of samples); the CV Folds "
                    "setting is ignored."),
            'logo': ("ℹ With LOGO, each group is left out once. The number of "
                     "folds equals the number of groups; the CV Folds setting "
                     "is ignored."),
        }[strategy]
        self.info_label.setText(info)

        self.configChanged.emit()

    def _on_group_source_changed(self, emit: bool = True):
        if self.current_strategy() not in ('group', 'logo'):
            self.clusters_label.setVisible(False)
            self.clusters_spin.setVisible(False)
            self.column_label.setVisible(False)
            self.column_combo.setVisible(False)
        else:
            source = self.current_group_source()
            self.clusters_label.setVisible(source == 'cluster')
            self.clusters_spin.setVisible(source == 'cluster')
            self.column_label.setVisible(source == 'column')
            self.column_combo.setVisible(source == 'column')
        if emit:
            self.configChanged.emit()

    # ── Reserved-bin (special value) helpers ────────────────────────────────

    def _on_special_toggled(self, checked: bool):
        """Show/hide the special value + tolerance rows and re-suggest."""
        self._update_special_row_visibility()
        self._update_special_suggestion()
        self.configChanged.emit()

    def _update_special_row_visibility(self):
        """Value/tolerance rows are only editable when the checkbox is on."""
        enabled = self.special_check.isChecked()
        for w in (self.special_value_label, self.special_value_spin,
                  self.special_tol_label, self.special_tol_spin):
            w.setVisible(self.special_check.isVisible() and enabled)
            w.setEnabled(enabled)

    def _update_special_suggestion(self):
        """Detect a degenerate point mass in the target and suggest reserving a
        bin for it (guidance, not forced). Only when stratified + regression."""
        if (self.current_strategy() != 'stratified'
                or self._problem_type != 'regression'
                or self._target is None):
            self.special_suggest_label.setText("")
            return
        hit = detect_point_mass(self._target, n_folds=self.folds_spin.value())
        if hit is None:
            self.special_suggest_label.setText("")
            return
        # Suggest only if the user hasn't already enabled the checkbox.
        if self.special_check.isChecked():
            self.special_suggest_label.setText("")
            return
        self.special_suggest_label.setText(
            f"💡 Detected {hit['count']} samples with target = {hit['value']:g} "
            f"({100.0*hit['fraction']:.1f}%). This usually means 'property absent' "
            f"(e.g. non-superconductor, metal). Consider enabling 'Reserve a bin "
            f"for an exact value' so every fold carries the same proportion."
        )

    def _update_group_source_availability(self):
        """Disable chemical grouping sources when no composition column exists."""
        has_comp = self._composition_col is not None
        model = self.group_source_combo.model()
        for key, label in zip(self._GROUP_SOURCE_KEYS, self._GROUP_SOURCE_LABELS):
            idx = self._GROUP_SOURCE_KEYS.index(key)
            item = model.item(idx)
            if key in ('chemical_system', 'formula', 'cluster'):
                item.setEnabled(has_comp)
                if not has_comp:
                    item.setToolTip("No composition/formula column detected in the dataset.")
                else:
                    item.setToolTip("")
