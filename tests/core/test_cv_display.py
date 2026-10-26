"""Tests for app.core.cv_display — the shared CV breakdown formatter.

v2.0: one renderer for kfold / stratified / group / loo / logo, in both HTML
(Results/Training Details) and text (Optimization panel / session export).
"""

import numpy as np
import pytest

from app.core.cv_display import cv_strategy_label, format_cv_details


# ── Strategy labels ─────────────────────────────────────────────────────────

class TestStrategyLabel:
    def test_kfold(self):
        assert cv_strategy_label('kfold') == 'K-Fold'

    def test_stratified_with_bins(self):
        assert 'quantile' in cv_strategy_label('stratified', {'n_bins': 5})

    def test_group_chemical_system(self):
        label = cv_strategy_label('group', {'group_source': 'chemical_system'})
        assert 'Group K-Fold' in label and 'chemical system' in label

    def test_logo(self):
        assert 'Leave-One-Group-Out' in cv_strategy_label('logo')

    def test_loo(self):
        assert 'Leave-One-Out' in cv_strategy_label('loo')


# ── HTML rendering ──────────────────────────────────────────────────────────

class TestHTML:
    def test_kfold_table(self):
        cm = {'test_score': [0.9, 0.91, 0.92], 'train_score': [0.99, 0.99, 0.99],
              'fold_test_sizes': [10, 10, 10]}
        html = format_cv_details(cm, 'kfold', fmt='html')
        assert 'Cross-Validation' in html
        assert 'Train Score' in html and 'Test Score' in html
        assert '<b>Mean</b>' in html and '<b>Std</b>' in html

    def test_uneven_folds_show_test_size(self):
        cm = {'test_score': [0.9, 0.91], 'train_score': [0.99, 0.99],
              'fold_test_sizes': [12, 8]}
        html = format_cv_details(cm, 'group', {'group_source': 'chemical_system'}, fmt='html')
        assert 'Test Size' in html

    def test_even_folds_hide_test_size(self):
        cm = {'test_score': [0.9, 0.91], 'train_score': [0.99, 0.99],
              'fold_test_sizes': [10, 10]}
        html = format_cv_details(cm, 'kfold', fmt='html')
        assert 'Test Size' not in html

    def test_loo_oof_html(self):
        cm = {'test_score': [0.02, 0.01, 0.03], 'test_score_metric': 'mae',
              'r2_global': 0.95}
        html = format_cv_details(cm, 'loo', fmt='html')
        assert 'R² (global)' in html
        assert 'Sample' in html

    def test_logo_uses_group_unit(self):
        cm = {'test_score': [0.1, 0.2], 'test_score_metric': 'mae', 'r2_global': 0.9}
        html = format_cv_details(cm, 'logo', fmt='html')
        assert 'Group' in html

    def test_diagnostics_shown_html(self):
        cm = {'test_score': [0.9, 0.9], 'train_score': [0.99, 0.99],
              'cv_diagnostics': ['bins reduced']}
        html = format_cv_details(cm, 'stratified', {'n_bins': 5}, fmt='html')
        assert 'bins reduced' in html

    def test_empty_returns_empty(self):
        assert format_cv_details({}, 'kfold', fmt='html') == ""
        assert format_cv_details(None, 'kfold', fmt='html') == ""


# ── Text rendering ──────────────────────────────────────────────────────────

class TestText:
    def test_kfold_text(self):
        cm = {'test_score': [0.9, 0.91], 'train_score': [0.99, 0.99],
              'fold_test_sizes': [10, 10]}
        txt = format_cv_details(cm, 'kfold', fmt='text')
        assert 'Cross-Validation (K-Fold)' in txt
        assert 'Mean' in txt and 'Std' in txt

    def test_group_text_test_size(self):
        cm = {'test_score': [0.9, 0.91], 'train_score': [0.99, 0.99],
              'fold_test_sizes': [12, 8]}
        txt = format_cv_details(cm, 'group', {'group_source': 'chemical_system'}, fmt='text')
        assert 'Test Size' in txt
        assert '12' in txt and '8' in txt

    def test_loo_text(self):
        cm = {'test_score': [0.02, 0.01], 'test_score_metric': 'mae', 'r2_global': 0.9}
        txt = format_cv_details(cm, 'loo', fmt='text')
        assert 'Leave-One-Out' in txt
        assert 'Per-sample MAE' in txt
        assert 'R² (global)' in txt

    def test_logo_text_per_group(self):
        cm = {'test_score': [0.1, 0.2], 'test_score_metric': 'mae', 'r2_global': 0.9}
        txt = format_cv_details(cm, 'logo', fmt='text')
        assert 'Per-group MAE' in txt
