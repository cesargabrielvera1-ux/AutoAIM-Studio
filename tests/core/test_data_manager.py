"""Tests for data_manager module."""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from app.core.data_manager import DataManager, DatasetInfo, ColumnInfo


class TestDataManager:
    """Test suite for DataManager."""

    def test_initialization(self):
        """Test DataManager initialization."""
        dm = DataManager()
        assert dm.data is None
        assert dm.target_column is None
        assert dm.feature_columns == []
        assert dm.is_classification is False

    def test_load_csv(self, tmp_path):
        """Test loading CSV file."""
        df = pd.DataFrame({
            'feature_a': [1, 2, 3, 4, 5],
            'feature_b': [5, 4, 3, 2, 1],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        dm = DataManager()
        info = dm.load_data(str(csv_path))
        
        assert isinstance(info, DatasetInfo)
        assert info.n_samples == 5
        assert info.n_features == 3
        assert dm.data is not None
        assert len(dm.data) == 5

    def test_load_csv_with_unnamed_columns(self, tmp_path):
        """Test loading CSV with trailing unnamed columns (bug fix)."""
        df = pd.DataFrame({
            'feature_a': [1, 2, 3],
            'target': [1.0, 2.0, 3.0],
            'Unnamed: 2': [np.nan, np.nan, np.nan]
        })
        csv_path = tmp_path / "test_unnamed.csv"
        df.to_csv(csv_path, index=False)
        
        dm = DataManager()
        info = dm.load_data(str(csv_path))
        
        # Should drop unnamed columns
        assert 'Unnamed: 2' not in dm.data.columns
        assert info.n_features == 2

    def test_load_nonexistent_file(self):
        """Test loading nonexistent file raises FileNotFoundError."""
        dm = DataManager()
        with pytest.raises(FileNotFoundError):
            dm.load_data("/nonexistent/path/file.csv")

    def test_load_excel(self, tmp_path):
        """Test loading Excel file."""
        df = pd.DataFrame({
            'feature_a': [1, 2, 3],
            'target': [1.0, 2.0, 3.0]
        })
        excel_path = tmp_path / "test.xlsx"
        df.to_excel(excel_path, index=False)
        
        dm = DataManager()
        info = dm.load_data(str(excel_path))
        
        assert isinstance(info, DatasetInfo)
        assert info.n_samples == 3

    def test_set_target_column(self):
        """Test setting target column."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'feature_a': [1, 2, 3],
            'feature_b': [4, 5, 6],
            'target': [1.0, 2.0, 3.0]
        })
        
        dm.set_target_column('target')
        
        assert dm.target_column == 'target'
        assert 'feature_a' in dm.feature_columns
        assert 'feature_b' in dm.feature_columns
        assert 'target' not in dm.feature_columns

    def test_set_target_column_nonexistent(self):
        """Test setting nonexistent target column."""
        dm = DataManager()
        dm._data = pd.DataFrame({'feature_a': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="Column 'nonexistent' not found"):
            dm.set_target_column('nonexistent')

    def test_set_target_column_no_data(self):
        """Test setting target without data."""
        dm = DataManager()
        with pytest.raises(ValueError, match="No data loaded"):
            dm.set_target_column('target')

    def test_set_target_classification(self):
        """Test classification detection."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'feature': [1, 2, 3, 4],
            'target': [0, 1, 0, 1]  # Binary integer target
        })
        
        dm.set_target_column('target')
        assert dm.is_classification is True

    def test_set_target_regression(self):
        """Test regression detection."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'feature': [1, 2, 3, 4],
            'target': [1.5, 2.3, 3.7, 4.2]  # Float target
        })
        
        dm.set_target_column('target')
        assert dm.is_classification is False

    def test_numeric_only_features(self):
        """Test that only numeric columns are used as features (bug fix v1.2.0)."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'numeric_feature': [1.0, 2.0, 3.0],
            'string_feature': ['a', 'b', 'c'],
            'structure_name': ['s1', 's2', 's3'],
            'target': [1.0, 2.0, 3.0]
        })
        
        dm.set_target_column('target')
        
        # Only numeric_feature should be in feature_columns
        assert 'numeric_feature' in dm.feature_columns
        assert 'string_feature' not in dm.feature_columns
        assert 'structure_name' not in dm.feature_columns

    def test_get_preprocessor(self):
        """Test creating preprocessor."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'num_a': [1.0, 2.0, 3.0, 4.0],
            'num_b': [4.0, 3.0, 2.0, 1.0],
            'target': [1.0, 2.0, 3.0, 4.0]
        })
        dm.set_target_column('target')
        
        preprocessor = dm.get_preprocessor()
        assert preprocessor is not None

    def test_prepare_data(self):
        """Test data preparation."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'num_a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num_b': [5.0, 4.0, 3.0, 2.0, 1.0],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        dm.set_target_column('target')
        
        X_train, X_test, y_train, y_test = dm.prepare_data(
            test_size=0.4,
            random_state=42
        )
        
        assert X_train.shape[0] == 3  # 60% of 5
        assert X_test.shape[0] == 2   # 40% of 5
        assert y_train.shape[0] == 3
        assert y_test.shape[0] == 2

    def test_prepare_data_without_target(self):
        """Test prepare_data without target set."""
        dm = DataManager()
        dm._data = pd.DataFrame({'feature': [1, 2, 3]})
        
        with pytest.raises(ValueError, match="target column must be set"):
            dm.prepare_data()

    def test_get_feature_names(self):
        """Test getting feature names."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'num_a': [1.0, 2.0, 3.0],
            'num_b': [4.0, 5.0, 6.0],
            'target': [1.0, 2.0, 3.0]
        })
        dm.set_target_column('target')
        
        feature_names = dm.get_feature_names()
        assert 'num_a' in feature_names
        assert 'num_b' in feature_names
        assert 'target' not in feature_names

    def test_composition_column_detection(self):
        """Test composition column detection."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'formula': ['Fe2O3', 'SiO2', 'Al2O3'],
            'property': [1.0, 2.0, 3.0]
        })
        
        dm._dataset_info = dm._analyze_dataset()
        comp_cols = dm.get_composition_columns()
        assert 'formula' in comp_cols

    def test_analyze_dataset(self):
        """Test dataset analysis."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'numeric': [1.0, 2.0, 3.0, np.nan],
            'categorical': ['A', 'B', 'A', 'B'],
            'target': [1.0, 2.0, 3.0, 4.0]
        })
        
        info = dm._analyze_dataset()
        assert isinstance(info, DatasetInfo)
        assert info.n_samples == 4
        assert info.has_missing_values is True
        assert 'numeric' in info.columns
        assert 'categorical' in info.columns

    def test_column_info(self):
        """Test ColumnInfo dataclass."""
        col = ColumnInfo(
            name='test_col',
            dtype='float64',
            is_numeric=True,
            is_categorical=False,
            is_composition=False
        )
        assert col.name == 'test_col'
        assert col.is_numeric is True
        assert col.is_target is False

    def test_save_load_preprocessor(self, tmp_path):
        """Test saving and loading preprocessor."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'num_a': [1.0, 2.0, 3.0, 4.0],
            'num_b': [4.0, 3.0, 2.0, 1.0],
            'target': [1.0, 2.0, 3.0, 4.0]
        })
        dm.set_target_column('target')
        dm.prepare_data()
        
        # Save preprocessor
        prep_path = tmp_path / "preprocessor.joblib"
        dm.save_preprocessor(str(prep_path))
        assert prep_path.exists()
        
        # Load preprocessor
        dm2 = DataManager()
        dm2.load_preprocessor(str(prep_path))
        assert dm2._preprocessor is not None

    def test_load_preprocessor_not_fitted(self):
        """Test saving preprocessor that isn't fitted."""
        dm = DataManager()
        with pytest.raises(ValueError, match="Preprocessor not fitted"):
            dm.save_preprocessor("/tmp/test.joblib")

    def test_is_composition_column(self):
        """Test composition column detection."""
        dm = DataManager()
        
        series = pd.Series(['Fe2O3', 'SiO2', 'Al2O3'], name='formula')
        assert dm._is_composition_column(series) is True
        
        series2 = pd.Series([1.0, 2.0, 3.0], name='feature')
        assert dm._is_composition_column(series2) is False

    def test_get_preprocessed_feature_count(self):
        """Test getting preprocessed feature count."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'num_a': [1.0, 2.0, 3.0],
            'num_b': [4.0, 5.0, 6.0],
            'target': [1.0, 2.0, 3.0]
        })
        dm.set_target_column('target')
        
        count = dm.get_preprocessed_feature_count()
        assert count == 2  # 2 numeric features

    def test_prepare_data_with_validation(self, tmp_path):
        """Test prepare_data with external validation data."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'num_a': [1.0, 2.0, 3.0, 4.0, 5.0],
            'num_b': [5.0, 4.0, 3.0, 2.0, 1.0],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        dm.set_target_column('target')
        
        # Create validation CSV
        val_df = pd.DataFrame({
            'num_a': [6.0, 7.0],
            'num_b': [1.0, 2.0],
            'target': [6.0, 7.0]
        })
        val_path = tmp_path / "validation.csv"
        val_df.to_csv(val_path, index=False)
        dm.load_validation_data(str(val_path))
        
        X_train, X_val, y_train, y_val = dm.prepare_data(use_validation=True)
        
        assert X_val.shape[0] == 2
        assert y_val.shape[0] == 2

    def test_dataset_info_memory(self):
        """Test dataset memory calculation."""
        dm = DataManager()
        dm._data = pd.DataFrame({
            'feature': [1.0] * 1000,
            'target': [2.0] * 1000
        })
        
        info = dm._analyze_dataset()
        assert info.memory_usage_mb > 0

    def test_empty_dataframe_analysis(self):
        """Test analysis of empty DataFrame."""
        dm = DataManager()
        
        with pytest.raises(ValueError, match="No data loaded"):
            dm._analyze_dataset()

    def test_external_validation_preprocessing(self, tmp_path):
        """v1.5.0: Verify get_validation_arrays() applies preprocessor.
        
        This is a critical test: without preprocessing, models trained on
        scaled data produce garbage predictions on raw validation data,
        resulting in negative R2 values.
        """
        dm = DataManager()
        
        # Training data with very different scales
        train_df = pd.DataFrame({
            'feature_a': [1000.0, 2000.0, 3000.0, 4000.0, 5000.0],
            'feature_b': [0.1, 0.2, 0.3, 0.4, 0.5],
            'target': [1.0, 2.0, 3.0, 4.0, 5.0]
        })
        train_path = tmp_path / "train.csv"
        train_df.to_csv(train_path, index=False)
        dm.load_data(str(train_path))
        dm.set_target_column('target')
        
        # Validation data with same scale pattern
        val_df = pd.DataFrame({
            'feature_a': [6000.0, 7000.0],
            'feature_b': [0.6, 0.7],
            'target': [6.0, 7.0]
        })
        val_path = tmp_path / "val.csv"
        val_df.to_csv(val_path, index=False)
        dm.load_validation_data(str(val_path))
        
        # Prepare data (fits preprocessor on training)
        X_train, X_test, y_train, y_test = dm.prepare_data(use_validation=False)
        
        # v2.0: get_validation_arrays() now returns RAW DataFrames.
        # Preprocessing happens inside the sklearn Pipeline during training
        # and inference — this eliminates data leakage in CV.
        X_val, y_val = dm.get_validation_arrays()

        assert X_val is not None
        assert y_val is not None

        # X_val is a raw DataFrame (same shape as original features)
        assert X_val.shape[1] == X_test.shape[1], (
            f"Feature mismatch: val has {X_val.shape[1]} features, "
            f"test has {X_test.shape[1]}."
        )

        # v2.0: Validation data is returned RAW — the scaler lives inside
        # the Pipeline and is applied during inference, not here.
        # Raw feature_a = 6000 should still be ~6000 in the raw DataFrame.
        assert np.abs(X_val.iloc[0, 0]) > 1000, (
            f"Validation data should be RAW in v2.0: got {X_val.iloc[0, 0]}. "
            f"The preprocessor now lives inside the Pipeline."
        )
