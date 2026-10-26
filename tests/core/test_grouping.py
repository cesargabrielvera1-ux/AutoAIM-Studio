"""Tests for app.core.grouping — the four group-label sources for grouped CV.

v2.0: Covers chemical-system, formula (reduced/exact), clustering, and
arbitrary-column grouping, plus composition-column detection and edge cases.
"""

import numpy as np
import pandas as pd
import pytest

from app.core.grouping import (
    GroupingError,
    GroupGenerator,
    detect_composition_column,
    groups_from_chemical_system,
    groups_from_clustering,
    groups_from_column,
    groups_from_formula,
    _reduced_formula,
)


# ── Composition column detection ────────────────────────────────────────────

class TestDetectCompositionColumn:
    def test_detects_formula_column(self):
        df = pd.DataFrame({
            'formula': ['Fe2O3', 'SiO2', 'Al2O3'],
            'value': [1.0, 2.0, 3.0],
        })
        assert detect_composition_column(df) == 'formula'

    def test_detects_composition_named_column(self):
        df = pd.DataFrame({
            'composition': ['NaCl', 'KCl', 'LiF'],
            'x': [1, 2, 3],
        })
        assert detect_composition_column(df) == 'composition'

    def test_returns_none_without_composition(self):
        df = pd.DataFrame({'a': [1, 2, 3], 'b': ['x', 'y', 'z']})
        assert detect_composition_column(df) is None

    def test_ignores_numeric_columns(self):
        df = pd.DataFrame({'n': [123, 456, 789]})
        assert detect_composition_column(df) is None


# ── Reduced formula helper ──────────────────────────────────────────────────

class TestReducedFormula:
    def test_simple(self):
        assert _reduced_formula('Fe2O3') == 'Fe2O3'

    def test_reduces_multiple(self):
        # Y2Ba4Cu6O14 is 2x YBa2Cu3O7 — both must map to the same reduced form.
        assert _reduced_formula('Y2Ba4Cu6O14') == _reduced_formula('YBa2Cu3O7')

    def test_fe3o4(self):
        assert _reduced_formula('Fe3O4') == 'Fe3O4'

    def test_invalid_returns_none(self):
        assert _reduced_formula('') is None
        assert _reduced_formula(None) is None


# ── Chemical system ─────────────────────────────────────────────────────────

class TestChemicalSystem:
    def test_groups_by_element_set(self):
        comps = pd.Series(['YBa2Cu3O7', 'Y2Ba4Cu6O14', 'Fe2O3', 'Fe3O4'])
        labels, info = groups_from_chemical_system(comps)
        # Both Y-Ba-Cu-O stoichiometries share a system; Fe2O3 and Fe3O4 share.
        assert labels[0] == labels[1]
        assert labels[2] == labels[3]
        assert labels[0] != labels[2]
        assert info['n_groups'] == 2

    def test_info_diagnostics(self):
        comps = pd.Series(['Fe2O3', 'Fe2O3', 'SiO2'])
        labels, info = groups_from_chemical_system(comps)
        assert info['n_groups'] == 2
        assert info['max_size'] == 2
        assert info['min_size'] == 1
        assert info['n_singletons'] == 1

    def test_invalid_formula_raises(self):
        comps = pd.Series(['Fe2O3', 'not_a_formula!!', 'SiO2'])
        with pytest.raises(GroupingError):
            groups_from_chemical_system(comps)


# ── Formula grouping ────────────────────────────────────────────────────────

class TestFormulaGrouping:
    def test_reduced_groups_multiples(self):
        comps = pd.Series(['YBa2Cu3O7', 'Y2Ba4Cu6O14', 'Fe2O3'])
        labels, info = groups_from_formula(comps, reduced=True)
        assert labels[0] == labels[1]   # same reduced formula
        assert labels[0] != labels[2]
        assert info['n_groups'] == 2

    def test_exact_distinguishes_multiples(self):
        comps = pd.Series(['YBa2Cu3O7', 'Y2Ba4Cu6O14'])
        labels, info = groups_from_formula(comps, reduced=False)
        assert labels[0] != labels[1]
        assert info['n_groups'] == 2


# ── Clustering ──────────────────────────────────────────────────────────────

class TestClustering:
    def test_deterministic(self):
        comps = pd.Series(['Fe2O3', 'Fe3O4', 'SiO2', 'Al2O3', 'NaCl', 'KCl'] * 3)
        l1, i1 = groups_from_clustering(comps, n_clusters=3)
        l2, i2 = groups_from_clustering(comps, n_clusters=3)
        # Agglomerative clustering is deterministic (no random seed).
        assert list(l1) == list(l2)
        assert i1['n_groups'] == i2['n_groups']

    def test_cluster_count(self):
        comps = pd.Series(['Fe2O3', 'Fe3O4', 'SiO2', 'Al2O3', 'NaCl', 'KCl',
                           'LiF', 'MgO', 'CaO', 'ZnO'])
        labels, info = groups_from_clustering(comps, n_clusters=3)
        assert info['n_groups'] <= 3
        assert info['n_clusters'] == 3

    def test_too_many_clusters_raises(self):
        comps = pd.Series(['Fe2O3', 'SiO2'])
        with pytest.raises(GroupingError):
            groups_from_clustering(comps, n_clusters=10)

    def test_invalid_n_clusters_raises(self):
        comps = pd.Series(['Fe2O3', 'SiO2'])
        with pytest.raises(GroupingError):
            groups_from_clustering(comps, n_clusters=1)


# ── Arbitrary column ────────────────────────────────────────────────────────

class TestColumnGrouping:
    def test_groups_by_column(self):
        df = pd.DataFrame({
            'batch': ['A', 'A', 'B', 'B', 'B'],
            'x': [1, 2, 3, 4, 5],
        })
        labels, info = groups_from_column(df, 'batch')
        assert info['n_groups'] == 2
        assert info['group_sizes'] == {'A': 2, 'B': 3}
        assert info['column'] == 'batch'

    def test_nan_raises(self):
        df = pd.DataFrame({'batch': ['A', None, 'B']})
        with pytest.raises(GroupingError, match="no value"):
            groups_from_column(df, 'batch')

    def test_missing_column_raises(self):
        df = pd.DataFrame({'x': [1, 2, 3]})
        with pytest.raises(GroupingError):
            groups_from_column(df, 'nonexistent')


# ── Unified dispatcher ──────────────────────────────────────────────────────

class TestGroupGenerator:
    def setup_method(self):
        self.df = pd.DataFrame({
            'formula': ['YBa2Cu3O7', 'Fe2O3', 'Fe3O4', 'MgB2', 'NbTi', 'SiO2'],
            'lab': ['L1', 'L1', 'L2', 'L2', 'L3', 'L3'],
        })
        self.gen = GroupGenerator()

    def test_chemical_system_via_df(self):
        labels, info = self.gen.generate('chemical_system', df=self.df)
        assert info['n_groups'] >= 4

    def test_column_source(self):
        labels, info = self.gen.generate('column', df=self.df, column='lab')
        assert info['n_groups'] == 3

    def test_cluster_source(self):
        labels, info = self.gen.generate('cluster', df=self.df, n_clusters=2)
        assert info['n_groups'] <= 2

    def test_unknown_source_raises(self):
        with pytest.raises(GroupingError):
            self.gen.generate('bogus', df=self.df)

    def test_column_source_requires_df(self):
        with pytest.raises(GroupingError):
            self.gen.generate('column', df=None, column='lab')
