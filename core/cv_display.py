"""Shared rendering of cross-validation breakdowns.

v2.0: Centralizes the per-fold / per-unit CV display that was previously
duplicated across the Training tab, Optimization tab (x2) and Results tab.
One function produces the breakdown for every strategy:

    kfold      → Train/Test score per fold + Mean/Std
    stratified → same as K-Fold (header notes the quantile bins)
    group      → Train/Test score per fold + Test Size column (uneven folds)
    loo        → R² (global) + per-sample MAE + Mean/Std
    logo       → R² (global) + per-group MAE + Mean/Std

Two output formats are supported: 'html' (for Results / Training Details
popups) and 'text' (for the Optimization details panel and session export).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


_STRATEGY_HEADERS = {
    'kfold': 'K-Fold',
    'stratified': 'Stratified K-Fold',
    'group': 'Group K-Fold',
    'loo': 'Leave-One-Out',
    'logo': 'Leave-One-Group-Out',
}


def cv_strategy_label(cv_strategy: str, cv_config_dict: Optional[Dict[str, Any]] = None) -> str:
    """Public wrapper: human-readable CV strategy label (with parameters)."""
    return _strategy_label(cv_strategy, cv_config_dict)


def _strategy_label(cv_strategy: str, cv_config_dict: Optional[Dict[str, Any]]) -> str:
    """Human-readable CV strategy label, with parameters when relevant."""
    base = _STRATEGY_HEADERS.get(cv_strategy, 'K-Fold')
    cfg = cv_config_dict or {}
    if cv_strategy == 'stratified':
        bins = cfg.get('n_bins')
        if bins:
            base += f" ({bins} quantile bins)"
        # v2.0: note a reserved bin for a degenerate target value.
        sv = cfg.get('special_value')
        if sv is not None:
            base += f" + reserved bin for {sv:g}"
    elif cv_strategy in ('group', 'logo'):
        src = cfg.get('group_source')
        src_label = {
            'chemical_system': 'chemical system',
            'formula': 'formula',
            'cluster': 'clustering',
            'column': f"column '{cfg.get('group_column')}'" if cfg.get('group_column') else 'column',
        }.get(src)
        if src_label:
            base += f" (by {src_label})"
    return base


def format_cv_details(
    cv_metrics: Optional[Dict[str, Any]],
    cv_strategy: str = 'kfold',
    cv_config_dict: Optional[Dict[str, Any]] = None,
    fmt: str = 'html',
) -> str:
    """Return a formatted CV breakdown for display.

    Args:
        cv_metrics: the result's cv_metrics dict (may be None/empty).
        cv_strategy: 'kfold' | 'stratified' | 'group' | 'loo' | 'logo'.
        cv_config_dict: serialized CVConfig (for parameter-aware headers).
        fmt: 'html' or 'text'.

    Returns:
        Formatted string (empty if there are no CV scores to show).
    """
    cv_metrics = cv_metrics or {}
    test_scores = cv_metrics.get('test_score', []) or []
    train_scores = cv_metrics.get('train_score', []) or []
    if not test_scores:
        return ""

    test_metric = cv_metrics.get('test_score_metric', 'r2')
    is_oof = test_metric == 'mae'
    r2_global = cv_metrics.get('r2_global')
    fold_sizes = cv_metrics.get('fold_test_sizes', []) or []
    diagnostics = cv_metrics.get('cv_diagnostics', []) or []

    label = _strategy_label(cv_strategy, cv_config_dict)

    if fmt == 'text':
        return _render_text(label, cv_strategy, is_oof, test_scores, train_scores,
                            r2_global, fold_sizes, diagnostics)
    return _render_html(label, cv_strategy, is_oof, test_scores, train_scores,
                        r2_global, fold_sizes, diagnostics)


# ── HTML rendering ──────────────────────────────────────────────────────────

def _render_html(label, cv_strategy, is_oof, test_scores, train_scores,
                 r2_global, fold_sizes, diagnostics) -> str:
    out = f"<h4>Cross-Validation ({label}):</h4>"
    for d in diagnostics:
        out += f"<p style='color:#ff9800;'><i>⚠ {d}</i></p>"

    if is_oof:
        unit = "Group" if cv_strategy == 'logo' else "Sample"
        if r2_global is not None:
            out += f"<p><b>R² (global):</b> {r2_global:.4f}</p>"
        out += f"<p><i>Per-{unit.lower()} Mean Absolute Error (MAE):</i></p>"
        out += "<table border='1' cellpadding='5'>"
        out += f"<tr><th>{unit}</th><th>MAE</th></tr>"
        for i, s in enumerate(test_scores):
            out += f"<tr><td>{unit} {i+1}</td><td>{s:.6f}</td></tr>"
        out += (f"<tr><td><b>Mean MAE ± Std</b></td>"
                f"<td><b>{np.mean(test_scores):.6f} ± {np.std(test_scores):.6f}</b></td></tr>")
        out += "</table>"
        return out

    # Per-fold path. Show a Test Size column when folds are uneven (group CV).
    show_size = bool(fold_sizes) and len(set(fold_sizes)) > 1
    out += "<table border='1' cellpadding='5'>"
    header = "<tr><th>Fold</th><th>Train Score</th><th>Test Score</th>"
    if show_size:
        header += "<th>Test Size</th>"
    header += "</tr>"
    out += header
    for i, test_s in enumerate(test_scores, 1):
        train_s = train_scores[i-1] if i - 1 < len(train_scores) else None
        train_str = f"{train_s:.6f}" if train_s is not None else "N/A"
        row = f"<tr><td>Fold {i}</td><td>{train_str}</td><td>{test_s:.6f}</td>"
        if show_size:
            size = fold_sizes[i-1] if i - 1 < len(fold_sizes) else None
            row += f"<td>{size if size is not None else '—'}</td>"
        row += "</tr>"
        out += row
    # Mean / Std rows.
    train_mean = np.mean(train_scores) if train_scores else None
    test_mean = np.mean(test_scores)
    train_std = np.std(train_scores) if train_scores else None
    test_std = np.std(test_scores)
    mean_row = (f"<tr><td><b>Mean</b></td>"
                f"<td><b>{train_mean:.6f}</b></td>" if train_mean is not None else
                "<tr><td><b>Mean</b></td><td><b>N/A</b></td>")
    mean_row += f"<td><b>{test_mean:.6f}</b></td>"
    std_row = (f"<tr><td><b>Std</b></td>"
               f"<td><b>{train_std:.6f}</b></td>" if train_std is not None else
               "<tr><td><b>Std</b></td><td><b>N/A</b></td>")
    std_row += f"<td><b>{test_std:.6f}</b></td>"
    if show_size:
        mean_row += "<td></td>"
        std_row += "<td></td>"
    out += mean_row + "</tr>" + std_row + "</tr>"
    out += "</table>"
    return out


# ── Plain-text rendering ────────────────────────────────────────────────────

def _render_text(label, cv_strategy, is_oof, test_scores, train_scores,
                 r2_global, fold_sizes, diagnostics) -> str:
    lines: List[str] = []
    lines.append(f"Cross-Validation ({label}):")
    for d in diagnostics:
        lines.append(f"  ⚠ {d}")

    if is_oof:
        unit = "Group" if cv_strategy == 'logo' else "Sample"
        if r2_global is not None:
            lines.append(f"  R² (global): {r2_global:.6f}")
        lines.append(f"  Per-{unit.lower()} MAE:")
        for i, s in enumerate(test_scores, 1):
            lines.append(f"    {unit} {i:2d}: {s:.6f}")
        lines.append(f"    Mean MAE: {np.mean(test_scores):.6f}")
        lines.append(f"    Std MAE:  {np.std(test_scores):.6f}")
        return "\n".join(lines) + "\n"

    show_size = bool(fold_sizes) and len(set(fold_sizes)) > 1
    if show_size:
        lines.append(f"  {'Fold':>6} | {'Train Score':>12} | {'Test Score':>12} | {'Test Size':>9}")
        lines.append(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}")
    else:
        lines.append(f"  {'Fold':>6} | {'Train Score':>12} | {'Test Score':>12}")
        lines.append(f"  {'-'*6}-+-{'-'*12}-+-{'-'*12}")
    for i, test_s in enumerate(test_scores, 1):
        train_s = train_scores[i-1] if i - 1 < len(train_scores) else None
        train_str = f"{train_s:12.6f}" if train_s is not None else "         N/A"
        row = f"  {i:6d} | {train_str} | {test_s:12.6f}"
        if show_size:
            size = fold_sizes[i-1] if i - 1 < len(fold_sizes) else None
            row += f" | {size:9d}" if size is not None else " |         —"
        lines.append(row)
    train_mean = np.mean(train_scores) if train_scores else None
    test_mean = np.mean(test_scores)
    train_std = np.std(train_scores) if train_scores else None
    test_std = np.std(test_scores)
    mean_row = (f"  {'Mean':>6} | {train_mean:12.6f} | {test_mean:12.6f}" if train_mean is not None
                else f"  {'Mean':>6} | {'N/A':>12} | {test_mean:12.6f}")
    std_row = (f"  {'Std':>6} | {train_std:12.6f} | {test_std:12.6f}" if train_std is not None
               else f"  {'Std':>6} | {'N/A':>12} | {test_std:12.6f}")
    if show_size:
        mean_row += " |          "
        std_row += " |          "
    lines.append(mean_row)
    lines.append(std_row)
    return "\n".join(lines) + "\n"
