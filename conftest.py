"""Pytest configuration and shared fixtures for AutoAIM Studio tests."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path


@pytest.fixture
def sample_data():
    """Generate sample training data for testing.
    
    Returns:
        Tuple of (X, y) where X is a 100x10 feature matrix
        and y is a continuous target variable.
    """
    np.random.seed(42)
    X = np.random.randn(100, 10)
    # Create target with known relationship to features
    y = (
        2.0 * X[:, 0] +
        1.5 * X[:, 1] +
        0.5 * X[:, 2] +
        np.random.randn(100) * 0.1
    )
    return X, y


@pytest.fixture
def sample_dataframe():
    """Generate sample DataFrame with features and target.
    
    Returns:
        DataFrame with 50 rows, 5 feature columns, and 1 target column.
    """
    np.random.seed(42)
    return pd.DataFrame({
        'feature_1': np.random.randn(50),
        'feature_2': np.random.randn(50),
        'feature_3': np.random.randn(50),
        'feature_4': np.random.randn(50),
        'feature_5': np.random.randn(50),
        'target': np.random.randn(50)
    })


@pytest.fixture
def sample_dataframe_with_formula():
    """Generate sample DataFrame with chemical formulas.
    
    Returns:
        DataFrame with chemical formulas and target values.
    """
    return pd.DataFrame({
        'formula': ['Fe2O3', 'SiO2', 'Al2O3', 'TiO2', 'MgO', 'CaO', 'Na2O', 'K2O'],
        'target': [5.2, 3.1, 4.5, 6.8, 2.9, 3.7, 4.1, 3.5]
    })


@pytest.fixture
def temp_directory(tmp_path):
    """Provide a temporary directory for test outputs.
    
    Returns:
        Path object pointing to a temporary directory.
    """
    return tmp_path


@pytest.fixture
def trained_rf_model(sample_data):
    """Provide a pre-trained Random Forest model for testing.
    
    Returns:
        Trained RandomForestRegressor instance.
    """
    from sklearn.ensemble import RandomForestRegressor
    X, y = sample_data
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X, y)
    return model
