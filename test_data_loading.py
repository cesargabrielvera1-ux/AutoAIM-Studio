"""Tests for data loading functionality."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import os


class TestDataLoading:
    """Test suite for data loading operations."""
    
    def test_csv_loading(self, sample_dataframe, temp_directory):
        """Test loading data from CSV file."""
        # Save sample data to CSV
        csv_path = temp_directory / "test_data.csv"
        sample_dataframe.to_csv(csv_path, index=False)
        
        # Load the CSV
        loaded_df = pd.read_csv(csv_path)
        
        # Verify loaded data matches original
        assert len(loaded_df) == len(sample_dataframe)
        assert list(loaded_df.columns) == list(sample_dataframe.columns)
        pd.testing.assert_frame_equal(loaded_df, sample_dataframe)
    
    def test_excel_loading(self, sample_dataframe, temp_directory):
        """Test loading data from Excel file."""
        # Save sample data to Excel
        excel_path = temp_directory / "test_data.xlsx"
        sample_dataframe.to_excel(excel_path, index=False)
        
        # Load the Excel file
        loaded_df = pd.read_excel(excel_path)
        
        # Verify loaded data matches original
        assert len(loaded_df) == len(sample_dataframe)
        assert list(loaded_df.columns) == list(sample_dataframe.columns)
    
    def test_column_detection(self, sample_dataframe):
        """Test automatic detection of column types."""
        # Create a DataFrame with mixed types
        df = pd.DataFrame({
            'numeric_1': [1.0, 2.0, 3.0, 4.0, 5.0],
            'numeric_2': [10, 20, 30, 40, 50],
            'categorical': ['A', 'B', 'A', 'C', 'B'],
            'target': [1.5, 2.3, 3.1, 3.9, 4.7]
        })
        
        # Detect column types
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Verify detection
        assert 'numeric_1' in numerical_cols
        assert 'numeric_2' in numerical_cols
        assert 'target' in numerical_cols
        assert 'categorical' in categorical_cols
    
    def test_missing_value_detection(self):
        """Test detection of missing values in data."""
        df = pd.DataFrame({
            'feature_1': [1.0, 2.0, np.nan, 4.0, 5.0],
            'feature_2': [10, 20, 30, 40, 50],
            'target': [1.5, np.nan, 3.1, 3.9, 4.7]
        })
        
        # Check missing values
        missing_counts = df.isnull().sum()
        
        assert missing_counts['feature_1'] == 1
        assert missing_counts['feature_2'] == 0
        assert missing_counts['target'] == 1
    
    def test_data_shape_validation(self, sample_data):
        """Test validation of data shapes."""
        X, y = sample_data
        
        # Verify X and y have same number of samples
        assert X.shape[0] == y.shape[0]
        
        # Verify X is 2D and y is 1D
        assert len(X.shape) == 2
        assert len(y.shape) == 1
    
    def test_target_column_selection(self, sample_dataframe):
        """Test selecting target column from DataFrame."""
        target_col = 'target'
        
        # Verify target column exists
        assert target_col in sample_dataframe.columns
        
        # Extract target
        y = sample_dataframe[target_col].values
        
        # Verify target is numeric
        assert np.issubdtype(y.dtype, np.number)
    
    def test_train_test_split(self, sample_data):
        """Test train/test split functionality."""
        from sklearn.model_selection import train_test_split
        
        X, y = sample_data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Verify split sizes
        assert X_train.shape[0] == 80
        assert X_test.shape[0] == 20
        assert y_train.shape[0] == 80
        assert y_test.shape[0] == 20
        
        # Verify feature dimensions preserved
        assert X_train.shape[1] == X.shape[1]
        assert X_test.shape[1] == X.shape[1]
