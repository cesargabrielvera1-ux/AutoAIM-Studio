"""Tests for feature_engineering module."""

import pytest
import numpy as np
import pandas as pd

from app.core.feature_engineering import (
    CompositionParser, MagpieFeatures, CompositionFeatures,
    FeatureEngineer
)


class TestCompositionParser:
    """Test suite for CompositionParser."""

    def test_parse_simple_formula(self):
        """Test parsing simple formula."""
        parser = CompositionParser()
        comp = parser.parse_formula('Fe2O3')
        
        assert 'Fe' in comp
        assert 'O' in comp
        assert abs(comp['Fe'] - 0.4) < 0.01  # 2 Fe out of 5 atoms
        assert abs(comp['O'] - 0.6) < 0.01   # 3 O out of 5 atoms

    def test_parse_single_element(self):
        """Test parsing single element."""
        parser = CompositionParser()
        comp = parser.parse_formula('Si')
        
        assert 'Si' in comp
        assert comp['Si'] == 1.0

    def test_parse_binary_oxide(self):
        """Test parsing binary oxide."""
        parser = CompositionParser()
        comp = parser.parse_formula('SiO2')
        
        assert 'Si' in comp
        assert 'O' in comp
        assert abs(comp['Si'] - 1/3) < 0.01
        assert abs(comp['O'] - 2/3) < 0.01

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        parser = CompositionParser()
        comp = parser.parse_formula('')
        assert comp == {}

    def test_parse_none(self):
        """Test parsing None."""
        parser = CompositionParser()
        comp = parser.parse_formula(None)
        assert comp == {}

    def test_is_valid_formula(self):
        """Test formula validation."""
        parser = CompositionParser()
        
        assert parser.is_valid_formula('Fe2O3') is True
        assert parser.is_valid_formula('SiO2') is True
        assert parser.is_valid_formula('Fe') is True
        assert parser.is_valid_formula('') is False
        assert parser.is_valid_formula(None) is False


class TestMagpieFeatures:
    """Test suite for MagpieFeatures."""

    def test_initialization(self):
        """Test MagpieFeatures initialization."""
        mf = MagpieFeatures(composition_column='formula')
        assert mf.composition_column == 'formula'

    def test_fit(self):
        """Test fit method."""
        mf = MagpieFeatures(composition_column='formula')
        df = pd.DataFrame({'formula': ['Fe2O3', 'SiO2']})
        
        result = mf.fit(df)
        assert result is mf  # Should return self

    def test_transform(self):
        """Test transform method."""
        mf = MagpieFeatures(composition_column='formula')
        df = pd.DataFrame({'formula': ['Fe2O3', 'SiO2', 'Al2O3']})
        
        mf.fit(df)
        features = mf.transform(df)
        
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == 3  # 3 samples
        assert features.shape[1] > 0   # Should have features
        # All columns should start with 'comp_'
        assert all(col.startswith('comp_') for col in features.columns)

    def test_transform_missing_column(self):
        """Test transform with missing column raises."""
        mf = MagpieFeatures(composition_column='formula')
        df = pd.DataFrame({'other_column': [1, 2]})
        
        with pytest.raises(ValueError, match="Column 'formula' not found"):
            mf.transform(df)

    def test_invalid_formula_returns_zeros(self):
        """Test that invalid formula returns zeros or NaN."""
        mf = MagpieFeatures(composition_column='formula')
        df = pd.DataFrame({'formula': ['INVALID', 'XXX']})
        
        mf.fit(df)
        features = mf.transform(df)
        
        # Should return a DataFrame of correct shape
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == 2
        # For completely unparseable formulas, values should be 0 or NaN
        assert not (features.abs() > 1e6).any().any()  # No crazy large values

    def test_feature_names(self):
        """Test getting feature names."""
        mf = MagpieFeatures(composition_column='formula')
        names = mf._get_feature_names()
        
        assert len(names) > 0
        assert 'atomic_number_mean' in names
        assert 'n_elements' in names
        assert 'fractional_entropy' in names

    def test_get_feature_names_out(self):
        """Test get_feature_names_out method."""
        mf = MagpieFeatures(composition_column='formula')
        names = mf.get_feature_names_out()
        
        assert all(name.startswith('comp_') for name in names)

    def test_known_properties(self):
        """Test that known element properties are available."""
        mf = MagpieFeatures(composition_column='formula')
        
        assert 'Fe' in mf.ELEMENT_PROPERTIES
        assert 'O' in mf.ELEMENT_PROPERTIES
        assert 'Si' in mf.ELEMENT_PROPERTIES
        assert 'Al' in mf.ELEMENT_PROPERTIES
        
        props = mf.ELEMENT_PROPERTIES['Fe']
        assert 'atomic_number' in props
        assert props['atomic_number'] == 26

    def test_compute_entropy(self):
        """Test entropy computation."""
        mf = MagpieFeatures(composition_column='formula')
        
        # Single element -> 0 entropy
        entropy1 = mf._compute_entropy(np.array([1.0]))
        assert entropy1 == 0.0
        
        # Two equal elements -> max entropy for binary
        entropy2 = mf._compute_entropy(np.array([0.5, 0.5]))
        assert entropy2 > 0


class TestCompositionFeatures:
    """Test suite for CompositionFeatures."""

    def test_initialization(self):
        """Test initialization."""
        cf = CompositionFeatures(composition_column='formula')
        assert cf.composition_column == 'formula'
        assert cf.use_magpie is True
        # matminer should be forced to False
        assert cf.use_matminer is False

    def test_transform(self):
        """Test transform."""
        cf = CompositionFeatures(composition_column='formula')
        df = pd.DataFrame({'formula': ['Fe2O3', 'SiO2', 'Al2O3']})
        
        cf.fit(df)
        features = cf.transform(df)
        
        assert isinstance(features, pd.DataFrame)
        assert features.shape[0] == 3


class TestFeatureEngineer:
    """Test suite for FeatureEngineer."""

    def test_initialization(self):
        """Test FeatureEngineer initialization."""
        fe = FeatureEngineer()
        assert fe._transformers == {}

    def test_add_composition_features(self):
        """Test adding composition features."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'formula': ['Fe2O3', 'SiO2', 'Al2O3'],
            'property': [1.0, 2.0, 3.0]
        })
        
        result = fe.add_composition_features(df, composition_column='formula')
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3
        assert result.shape[1] > df.shape[1]  # Should have more columns
        # Original columns should be preserved
        assert 'formula' in result.columns
        assert 'property' in result.columns

    def test_add_composition_features_missing_column(self):
        """Test adding features with missing column raises."""
        fe = FeatureEngineer()
        df = pd.DataFrame({'other': [1, 2]})
        
        with pytest.raises(ValueError, match="Column 'formula' not found"):
            fe.add_composition_features(df, composition_column='formula')

    def test_add_polynomial_features(self):
        """Test adding polynomial features."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0],
            'c': ['x', 'y', 'z']  # Non-numeric, should be ignored
        })
        
        result = fe.add_polynomial_features(df, degree=2)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 3
        assert result.shape[1] > df.shape[1]
        # Original columns should be preserved
        assert 'a' in result.columns
        assert 'b' in result.columns

    def test_add_polynomial_features_specific_columns(self):
        """Test polynomial features with specific columns."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'a': [1.0, 2.0, 3.0],
            'b': [4.0, 5.0, 6.0],
        })
        
        result = fe.add_polynomial_features(df, columns=['a'], degree=2)
        
        # Should only use 'a' column
        assert 'a' in result.columns

    def test_add_statistical_features(self):
        """Test adding statistical features."""
        fe = FeatureEngineer()
        df = pd.DataFrame({
            'group': ['A', 'A', 'B', 'B'],
            'value': [1.0, 2.0, 3.0, 4.0]
        })
        
        result = fe.add_statistical_features(df, group_column='group', agg_column='value')
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape[0] == 4
        # Should have added aggregation columns
        assert any('mean' in col for col in result.columns)
