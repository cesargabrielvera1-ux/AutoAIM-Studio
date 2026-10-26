"""Tests for the reserved-bin (special value) feature of quantile stratification.

v2.0: isolating a degenerate point mass (e.g. Tc = 0 K, Eg = 0 eV) into its own
stratum so every fold carries the same proportion of it. Covers the binner,
point-mass detection, validation rules, serialization, and end-to-end training.
"""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from app.core.cv_factory import (
    BLOCK, INFO, WARN,
    CVConfig,
    build_cv_plan,
    detect_point_mass,
    has_blocking_issue,
    validate_cv_config,
)
from app.core.trainer import ModelTrainer


def make_target_with_spike(n_special=30, n_rest=70, value=0.0, seed=0):
    rng = np.random.RandomState(seed)
    y = np.concatenate([np.full(n_special, value), rng.rand(n_rest) * 95 + 5])
    rng.shuffle(y)
    return y


# ── Reserved bin in the plan ────────────────────────────────────────────────

class TestReservedBinPlan:
    def test_special_value_isolated_as_stratum(self):
        y = make_target_with_spike(n_special=20, n_rest=80, value=0.0)
        X = np.random.RandomState(0).rand(100, 4)
        plan = build_cv_plan(
            CVConfig('stratified', n_folds=5, n_bins=5, special_value=0.0),
            X, y, 'regression')
        bins = plan.splitter._y_binned
        # Special stratum is label 0 and holds exactly the 20 zeros.
        assert (bins == 0).sum() == 20
        assert any('Reserved a stratum' in d for d in plan.diagnostics)

    def test_special_value_evenly_distributed_across_folds(self):
        y = make_target_with_spike(n_special=25, n_rest=75, value=0.0)
        X = np.random.RandomState(0).rand(100, 4)
        plan = build_cv_plan(
            CVConfig('stratified', n_folds=5, n_bins=5, special_value=0.0),
            X, y, 'regression')
        bins = plan.splitter._y_binned
        dummy = np.zeros((100, 1))
        expected = 0.25
        for _, te in plan.splitter.split(dummy):
            prop = (bins[te] == 0).mean()
            assert abs(prop - expected) < 0.10  # each fold ~25% zeros

    def test_remaining_samples_binned_into_n_minus_1(self):
        y = make_target_with_spike(n_special=25, n_rest=75, value=0.0)
        X = np.random.RandomState(0).rand(100, 4)
        plan = build_cv_plan(
            CVConfig('stratified', n_folds=5, n_bins=5, special_value=0.0),
            X, y, 'regression')
        bins = plan.splitter._y_binned
        # 1 reserved stratum + (5-1)=4 quantile bins.
        assert len(np.unique(bins)) == 5

    def test_tolerance_widens_match(self):
        rng = np.random.RandomState(1)
        # Zeros stored as 0.0 and 0.01 (below-threshold).
        y = np.concatenate([np.zeros(15), np.full(15, 0.01), rng.rand(70) * 95 + 5])
        X = rng.rand(100, 4)
        plan = build_cv_plan(
            CVConfig('stratified', n_folds=5, n_bins=5,
                     special_value=0.0, special_tolerance=0.02),
            X, y, 'regression')
        bins = plan.splitter._y_binned
        assert (bins == 0).sum() == 30  # both 0.0 and 0.01 captured

    def test_absent_value_falls_back_with_diagnostic(self):
        y = np.random.RandomState(0).rand(100) * 100  # no zeros
        X = np.random.RandomState(0).rand(100, 4)
        plan = build_cv_plan(
            CVConfig('stratified', n_folds=5, n_bins=5, special_value=0.0),
            X, y, 'regression')
        assert any('not found' in d for d in plan.diagnostics)

    def test_too_few_special_raises(self):
        y = np.concatenate([np.zeros(3), np.random.RandomState(0).rand(97) * 100])
        X = np.random.RandomState(0).rand(100, 4)
        with pytest.raises(ValueError, match="Cannot reserve a bin"):
            build_cv_plan(
                CVConfig('stratified', n_folds=5, n_bins=5, special_value=0.0),
                X, y, 'regression')


# ── Point-mass detection ────────────────────────────────────────────────────

class TestDetectPointMass:
    def test_detects_dominant_zero_spike(self):
        y = make_target_with_spike(n_special=30, n_rest=70, value=0.0)
        hit = detect_point_mass(y, n_folds=5)
        assert hit is not None
        assert hit['value'] == 0.0
        assert hit['count'] == 30
        assert abs(hit['fraction'] - 0.30) < 1e-6

    def test_detects_nonzero_value(self):
        y = make_target_with_spike(n_special=25, n_rest=75, value=1.5)
        hit = detect_point_mass(y, n_folds=5)
        assert hit is not None and hit['value'] == 1.5

    def test_no_detection_on_continuous(self):
        y = np.random.RandomState(0).rand(200) * 100
        assert detect_point_mass(y, n_folds=5) is None

    def test_below_salience_floor_not_detected(self):
        # 3 of 100 = 3% < 5% salience floor (and < 2*folds=10 count floor).
        y = make_target_with_spike(n_special=3, n_rest=97, value=0.0)
        assert detect_point_mass(y, n_folds=5) is None

    def test_below_count_floor_not_detected(self):
        # 5 of 20 = 25% but 5 < 2*5=10 count floor.
        y = make_target_with_spike(n_special=5, n_rest=15, value=0.0)
        assert detect_point_mass(y, n_folds=5) is None

    def test_too_small_dataset(self):
        y = np.array([0.0, 1.0, 2.0])
        assert detect_point_mass(y, n_folds=5) is None


# ── Validation rules ────────────────────────────────────────────────────────

class TestSpecialValueValidation:
    def _issues(self, cfg, y):
        return validate_cv_config(cfg, len(y), 'regression', y=y)

    def test_block_when_special_fewer_than_folds(self):
        y = make_target_with_spike(n_special=3, n_rest=97, value=0.0)
        issues = self._issues(CVConfig('stratified', n_folds=5, special_value=0.0), y)
        assert has_blocking_issue(issues)
        assert any('Cannot reserve a bin' in i['message'] for i in issues)

    def test_warn_when_special_is_majority(self):
        y = make_target_with_spike(n_special=60, n_rest=40, value=0.0)
        issues = self._issues(CVConfig('stratified', n_folds=5, special_value=0.0), y)
        assert not has_blocking_issue(issues)
        assert any(i['severity'] == WARN and 'majority' in i['message'] for i in issues)

    def test_warn_when_special_small_fraction(self):
        # 8 of 200 = 4% — stratifiable (>=5 folds) but below the 5% salience floor.
        y = make_target_with_spike(n_special=8, n_rest=192, value=0.0)
        issues = self._issues(CVConfig('stratified', n_folds=5, special_value=0.0), y)
        assert not has_blocking_issue(issues)
        assert any(i['severity'] == WARN and 'property absent' in i['message'] for i in issues)

    def test_info_when_value_absent(self):
        y = np.random.RandomState(0).rand(100) * 100  # no zeros
        issues = self._issues(CVConfig('stratified', n_folds=5, special_value=0.0), y)
        assert any(i['severity'] == INFO and 'does not occur' in i['message'] for i in issues)

    def test_ok_for_good_point_mass(self):
        y = make_target_with_spike(n_special=30, n_rest=70, value=0.0)
        issues = self._issues(CVConfig('stratified', n_folds=5, special_value=0.0), y)
        assert issues == []

    def test_no_special_value_no_rules(self):
        y = make_target_with_spike(n_special=30, n_rest=70, value=0.0)
        issues = self._issues(CVConfig('stratified', n_folds=5), y)
        assert issues == []


# ── Serialization round-trip ────────────────────────────────────────────────

class TestSpecialValueSerialization:
    def test_roundtrip(self):
        cfg = CVConfig('stratified', n_folds=4, n_bins=6,
                       special_value=0.0, special_tolerance=0.01)
        cfg2 = CVConfig.from_dict(cfg.to_dict())
        assert cfg2.special_value == 0.0
        assert cfg2.special_tolerance == 0.01
        assert cfg2.n_bins == 6

    def test_default_none(self):
        cfg = CVConfig.from_dict({})
        assert cfg.special_value is None
        assert cfg.special_tolerance == 0.0


# ── End-to-end training ─────────────────────────────────────────────────────

class TestSpecialValueEndToEnd:
    def test_trainer_with_reserved_bin(self):
        y = make_target_with_spike(n_special=30, n_rest=70, value=0.0)
        X = np.random.RandomState(0).rand(100, 4)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        trainer = ModelTrainer()
        cfg = CVConfig('stratified', n_folds=5, n_bins=5, special_value=0.0, random_state=42)
        res = trainer.train(RandomForestRegressor(n_estimators=10, random_state=42),
                            'Tc', Xtr, ytr, Xte, yte, 'regression',
                            cv_folds=5, cv_config=cfg)
        assert res.cv_strategy == 'stratified'
        assert len(res.cv_metrics['test_score']) == 5
        assert res.cv_config_dict['special_value'] == 0.0
        assert any('Reserved a stratum' in d
                   for d in res.cv_metrics.get('cv_diagnostics', []))

    def test_trainer_without_special_value_unchanged(self):
        y = make_target_with_spike(n_special=30, n_rest=70, value=0.0)
        X = np.random.RandomState(0).rand(100, 4)
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
        trainer = ModelTrainer()
        cfg = CVConfig('stratified', n_folds=5, n_bins=5, random_state=42)
        res = trainer.train(RandomForestRegressor(n_estimators=10, random_state=42),
                            'Tc2', Xtr, ytr, Xte, yte, 'regression',
                            cv_folds=5, cv_config=cfg)
        assert res.cv_config_dict['special_value'] is None
        # No reserved-bin diagnostic without a special value.
        assert not any('Reserved a stratum' in d
                       for d in res.cv_metrics.get('cv_diagnostics', []))
