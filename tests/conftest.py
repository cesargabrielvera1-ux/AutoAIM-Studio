"""Shared fixtures for all tests."""

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression, make_classification


@pytest.fixture
def regression_data():
    """Create a simple regression dataset."""
    X, y = make_regression(
        n_samples=100,
        n_features=5,
        noise=10,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


@pytest.fixture
def classification_data():
    """Create a simple classification dataset."""
    X, y = make_classification(
        n_samples=100,
        n_features=5,
        n_classes=2,
        n_informative=3,
        random_state=42
    )
    feature_names = [f"feature_{i}" for i in range(X.shape[1])]
    return X, y, feature_names


@pytest.fixture
def small_data():
    """Create a very small dataset for fast tests."""
    X, y = make_regression(
        n_samples=20,
        n_features=3,
        noise=5,
        random_state=42
    )
    return X, y


@pytest.fixture
def sample_dataframe():
    """Create a sample DataFrame with numeric and categorical columns."""
    np.random.seed(42)
    df = pd.DataFrame({
        'num_a': np.random.randn(50),
        'num_b': np.random.randn(50),
        'num_c': np.random.randint(0, 10, 50),
        'cat_d': np.random.choice(['A', 'B', 'C'], 50),
        'formula': np.random.choice(['Fe2O3', 'SiO2', 'Al2O3', 'TiO2'], 50),
        'target': np.random.randn(50),
    })
    return df


@pytest.fixture
def sample_dataframe_with_missing():
    """Create a sample DataFrame with missing values."""
    np.random.seed(42)
    df = pd.DataFrame({
        'feature_a': np.random.randn(50),
        'feature_b': np.random.randn(50),
        'feature_c': np.random.randn(50),
        'target': np.random.randn(50),
    })
    # Introduce some missing values
    df.loc[0:4, 'feature_a'] = np.nan
    df.loc[10:12, 'feature_b'] = np.nan
    return df
