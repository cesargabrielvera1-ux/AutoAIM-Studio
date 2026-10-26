"""Tests for validator module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from app.core.validator import MaterialsAwareValidator, UncertaintyEstimator


class TestMaterialsAwareValidator:
    """Test suite for MaterialsAwareValidator."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = MaterialsAwareValidator()
        assert 'kfold' in validator.available_strategies
        assert 'stratified' in validator.available_strategies
        assert 'loco' in validator.available_strategies
        assert 'loo' in validator.available_strategies

    def test_get_cv_splitter_kfold(self):
        """Test getting KFold splitter."""
        validator = MaterialsAwareValidator()
        cv = validator.get_cv_splitter('kfold', n_splits=5, random_state=42)
        
        assert cv is not None
        assert cv.get_n_splits() == 5

    def test_get_cv_splitter_stratified(self):
        """Test getting stratified splitter."""
        validator = MaterialsAwareValidator()
        y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 1, 1])
        cv = validator.get_cv_splitter('stratified', n_splits=3, y=y, random_state=42)
        
        assert cv is not None

    def test_get_cv_splitter_stratified_no_y(self):
        """Test stratified splitter without y raises."""
        validator = MaterialsAwareValidator()
        with pytest.raises(ValueError, match="requires target variable"):
            validator.get_cv_splitter('stratified', n_splits=3)

    def test_get_cv_splitter_loco_no_composition(self):
        """Test LOCO without composition raises."""
        validator = MaterialsAwareValidator()
        with pytest.raises(ValueError, match="requires composition column or groups"):
            validator.get_cv_splitter('loco', n_splits=3)

    def test_get_cv_splitter_group(self):
        """Test group splitter."""
        validator = MaterialsAwareValidator()
        compositions = pd.Series(['Fe2O3', 'SiO2', 'Fe2O3', 'Al2O3'])
        cv = validator.get_cv_splitter('group', n_splits=2, composition_column=compositions)
        
        assert cv is not None

    def test_get_cv_splitter_loo(self):
        """Test Leave-One-Out splitter."""
        validator = MaterialsAwareValidator()
        cv = validator.get_cv_splitter('loo')
        
        assert cv is not None
        # LOO needs data to know n_splits - just verify it was created
        assert cv is not None

    def test_get_cv_splitter_unknown(self):
        """Test unknown strategy raises."""
        validator = MaterialsAwareValidator()
        with pytest.raises(ValueError, match="Unknown strategy"):
            validator.get_cv_splitter('unknown', n_splits=3)

    def test_create_chemical_system_groups(self):
        """Test chemical system group creation."""
        validator = MaterialsAwareValidator()
        compositions = pd.Series(['Fe2O3', 'Fe2O3', 'SiO2', 'Al2O3'])
        groups = validator._create_chemical_system_groups(compositions)
        
        assert len(groups) == 4
        assert groups[0] == groups[1]  # Fe2O3 and Fe2O3 same group
        assert groups[0] != groups[2]  # Fe2O3 and SiO2 different groups

    def test_parse_elements(self):
        """Test chemical formula parsing."""
        validator = MaterialsAwareValidator()
        
        elements = validator._parse_elements('Fe2O3')
        assert 'Fe' in elements
        assert 'O' in elements
        
        elements = validator._parse_elements('SiO2')
        assert 'Si' in elements
        assert 'O' in elements

    def test_parse_elements_invalid(self):
        """Test parsing invalid formula."""
        validator = MaterialsAwareValidator()
        elements = validator._parse_elements(None)
        assert elements == []

    def test_recommend_strategy_small_dataset(self):
        """Test strategy recommendation for small dataset."""
        validator = MaterialsAwareValidator()
        strategy = validator.recommend_strategy(n_samples=30)
        assert strategy == 'loo'

    def test_recommend_strategy_medium_with_compositions(self):
        """Test strategy recommendation with compositions."""
        validator = MaterialsAwareValidator()
        strategy = validator.recommend_strategy(n_samples=500, has_compositions=True)
        assert strategy == 'loco'

    def test_recommend_strategy_large(self):
        """Test strategy recommendation for large dataset."""
        validator = MaterialsAwareValidator()
        strategy = validator.recommend_strategy(n_samples=2000, has_compositions=False)
        assert strategy == 'kfold'

    def test_cross_validate_with_uncertainty(self, regression_data):
        """Test CV with uncertainty estimation."""
        X, y, _ = regression_data
        validator = MaterialsAwareValidator()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        results = validator.cross_validate_with_uncertainty(
            model, X, y, strategy='kfold', n_splits=3, random_state=42
        )
        
        assert 'fold_scores' in results
        assert 'mean_score' in results
        assert 'overall_r2' in results
        assert len(results['fold_scores']) == 3

    def test_cross_validate_with_predictions(self, regression_data):
        """Test CV returning out-of-fold predictions."""
        X, y, _ = regression_data
        validator = MaterialsAwareValidator()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        results = validator.cross_validate_with_uncertainty(
            model, X, y, strategy='kfold', n_splits=3,
            return_predictions=True, random_state=42
        )
        
        assert 'oof_predictions' in results
        assert len(results['oof_predictions']) == len(y)

    def test_cluster_based_split(self, regression_data):
        """Test cluster-based split."""
        X, _, _ = regression_data
        validator = MaterialsAwareValidator()
        
        clusters = validator.cluster_based_split(X, n_clusters=3, random_state=42)
        
        assert len(clusters) == len(X)
        assert len(np.unique(clusters)) == 3

    def test_temporal_split(self):
        """Test temporal split."""
        validator = MaterialsAwareValidator()
        timestamps = pd.Series(pd.date_range('2020-01-01', periods=10))
        
        train_idx, test_idx = validator.temporal_split(timestamps, train_fraction=0.7)
        
        assert len(train_idx) == 7
        assert len(test_idx) == 3


class TestUncertaintyEstimator:
    """Test suite for UncertaintyEstimator."""

    def test_estimate_from_cv(self, regression_data):
        """Test uncertainty estimation from CV results."""
        X, y, _ = regression_data
        validator = MaterialsAwareValidator()
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        cv_results = validator.cross_validate_with_uncertainty(
            model, X, y, strategy='kfold', n_splits=3,
            return_predictions=True, random_state=42
        )
        
        estimator = UncertaintyEstimator()
        # Add y_true for residual-based estimation
        cv_results['y_true'] = y
        
        uncertainty = estimator.estimate_from_cv(cv_results)
        
        assert 'mean' in uncertainty
        assert 'std' in uncertainty
        assert 'ci_lower' in uncertainty
        assert 'ci_upper' in uncertainty

    def test_estimate_from_cv_no_predictions(self):
        """Test estimation without predictions raises."""
        estimator = UncertaintyEstimator()
        
        with pytest.raises(ValueError, match="must include out-of-fold predictions"):
            estimator.estimate_from_cv({})

    def test_bootstrap_uncertainty(self, regression_data):
        """Test bootstrap uncertainty estimation."""
        X, y, _ = regression_data
        model = RandomForestRegressor(n_estimators=5, random_state=42)
        
        estimator = UncertaintyEstimator()
        X_train, y_train = X[:80], y[:80]
        X_test = X[80:]
        
        result = estimator.bootstrap_uncertainty(
            model, X_train, y_train, X_test, n_bootstrap=10
        )
        
        assert 'mean' in result
        assert 'std' in result
        assert 'ci_lower' in result
        assert 'ci_upper' in result
        assert result['all_predictions'].shape[0] == 10
