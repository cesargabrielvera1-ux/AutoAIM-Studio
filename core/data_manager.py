"""Data management and preprocessing for Materials AutoML."""

import re
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from ..utils.logger import LoggerMixin, log_exception


@dataclass
class ColumnInfo:
    """Information about dataset columns."""
    name: str
    dtype: str
    is_numeric: bool
    is_categorical: bool
    is_composition: bool
    is_target: bool = False
    null_count: int = 0
    unique_count: int = 0
    sample_values: List[Any] = field(default_factory=list)


@dataclass
class DatasetInfo:
    """Complete dataset information."""
    n_samples: int
    n_features: int
    columns: Dict[str, ColumnInfo]
    target_column: Optional[str] = None
    has_missing_values: bool = False
    memory_usage_mb: float = 0.0


class DataManager(LoggerMixin):
    """Manage data loading, validation, and preprocessing."""
    
    # Patterns for detecting chemical compositions
    COMPOSITION_PATTERNS = [
        r'^[A-Z][a-z]?[0-9]*(?:[A-Z][a-z]?[0-9]*)*$',  # Simple formulas like Fe2O3, SiO2
        r'^[A-Z][a-z]?$',  # Single elements like Fe, O
        r'^composition$',  # Column named 'composition'
        r'^formula$',  # Column named 'formula'
        r'^chemical_formula$',
    ]
    
    def __init__(self):
        """Initialize data manager."""
        self._data: Optional[pd.DataFrame] = None
        self._validation_data: Optional[pd.DataFrame] = None
        self._dataset_info: Optional[DatasetInfo] = None
        self._target_column: Optional[str] = None
        self._feature_columns: List[str] = []
        self._preprocessor: Optional[Pipeline] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._is_classification: bool = False
        self._processed_feature_names: Optional[List[str]] = None
        
    @property
    def data(self) -> Optional[pd.DataFrame]:
        """Get current dataset."""
        return self._data
    
    @property
    def validation_data(self) -> Optional[pd.DataFrame]:
        """Get validation dataset."""
        return self._validation_data
    
    @property
    def dataset_info(self) -> Optional[DatasetInfo]:
        """Get dataset information."""
        return self._dataset_info
    
    @property
    def target_column(self) -> Optional[str]:
        """Get target column name."""
        return self._target_column
    
    @property
    def feature_columns(self) -> List[str]:
        """Get feature column names."""
        return self._feature_columns
    
    @property
    def is_classification(self) -> bool:
        """Check if problem is classification."""
        return self._is_classification
    
    def load_data(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ) -> DatasetInfo:
        """Load data from file.
        
        Args:
            file_path: Path to data file
            file_type: File type (csv, excel, etc.). Auto-detected if None.
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Dataset information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Auto-detect file type
        if file_type is None:
            file_type = file_path.suffix.lower()
        
        try:
            if file_type in ['.csv', 'csv']:
                self._data = pd.read_csv(file_path, index_col=False, **kwargs)
            elif file_type in ['.xlsx', '.xls', 'excel']:
                self._data = pd.read_excel(file_path, **kwargs)
            elif file_type in ['.json', 'json']:
                self._data = pd.read_json(file_path, **kwargs)
            elif file_type in ['.parquet', 'parquet']:
                self._data = pd.read_parquet(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Strip trailing unnamed columns created by extra commas in CSV/Excel
            unnamed_cols = [c for c in self._data.columns if str(c).startswith('Unnamed:')]
            if unnamed_cols:
                self._data = self._data.drop(columns=unnamed_cols)
                self.logger.warning(f"Dropped {len(unnamed_cols)} trailing unnamed columns: {unnamed_cols}")
            
            self.logger.info(f"Loaded {len(self._data)} rows, {len(self._data.columns)} columns from {file_path}")
            
            # Analyze dataset
            self._dataset_info = self._analyze_dataset()
            
            return self._dataset_info
            
        except Exception as e:
            log_exception(self.logger, e, f"Error loading data from {file_path}")
            raise
    
    def load_validation_data(
        self,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs
    ) -> DatasetInfo:
        """Load external validation dataset.
        
        Args:
            file_path: Path to validation data file
            file_type: File type
            **kwargs: Additional arguments
            
        Returns:
            Dataset information
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if file_type is None:
            file_type = file_path.suffix.lower()
        
        try:
            if file_type in ['.csv', 'csv']:
                self._validation_data = pd.read_csv(file_path, index_col=False, **kwargs)
            elif file_type in ['.xlsx', '.xls', 'excel']:
                self._validation_data = pd.read_excel(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")
            
            # Strip trailing unnamed columns
            unnamed_cols = [c for c in self._validation_data.columns if str(c).startswith('Unnamed:')]
            if unnamed_cols:
                self._validation_data = self._validation_data.drop(columns=unnamed_cols)
                self.logger.warning(f"Dropped {len(unnamed_cols)} trailing unnamed columns from validation: {unnamed_cols}")
            
            self.logger.info(f"Loaded validation data: {len(self._validation_data)} rows, {len(self._validation_data.columns)} columns")
            
            return self._analyze_dataset(self._validation_data)
            
        except Exception as e:
            log_exception(self.logger, e, f"Error loading validation data")
            raise
    
    def _analyze_dataset(self, data: Optional[pd.DataFrame] = None) -> DatasetInfo:
        """Analyze dataset and extract column information.
        
        Args:
            data: DataFrame to analyze. Uses self._data if None.
            
        Returns:
            Dataset information
        """
        if data is None:
            data = self._data
        
        if data is None:
            raise ValueError("No data loaded")
        
        columns = {}
        has_missing = False
        
        for col in data.columns:
            dtype = str(data[col].dtype)
            null_count = data[col].isnull().sum()
            unique_count = data[col].nunique()
            
            if null_count > 0:
                has_missing = True
            
            # Detect column type
            is_numeric = pd.api.types.is_numeric_dtype(data[col])
            is_categorical = False
            is_composition = False
            
            if is_numeric:
                # Trust pandas dtype: numeric columns stay numeric.
                # Do NOT force numeric columns to categorical based on unique count.
                # Columns with repeated numeric values (e.g. 0, 18.9) are still numeric.
                pass  # Keep is_numeric=True as detected by pandas
            else:
                # Check if it's categorical
                if pd.api.types.is_categorical_dtype(data[col]) or unique_count <= 20:
                    is_categorical = True
                
                # Check if it's a composition column
                if self._is_composition_column(data[col]):
                    is_composition = True
            
            # Sample values (non-null)
            sample_values = data[col].dropna().head(5).tolist()
            
            columns[col] = ColumnInfo(
                name=col,
                dtype=dtype,
                is_numeric=is_numeric,
                is_categorical=is_categorical,
                is_composition=is_composition,
                null_count=null_count,
                unique_count=unique_count,
                sample_values=sample_values
            )
        
        memory_usage = data.memory_usage(deep=True).sum() / (1024 ** 2)
        
        return DatasetInfo(
            n_samples=len(data),
            n_features=len(data.columns),
            columns=columns,
            has_missing_values=has_missing,
            memory_usage_mb=memory_usage
        )
    
    def _is_composition_column(self, series: pd.Series) -> bool:
        """Check if column contains chemical compositions.
        
        Args:
            series: pandas Series to check
            
        Returns:
            True if column appears to contain compositions
        """
        # Check column name
        col_name = series.name.lower()
        if any(pattern in col_name for pattern in ['composition', 'formula', 'chemical']):
            return True
        
        # Check sample values
        sample_values = series.dropna().head(20).astype(str)
        matches = 0
        
        for value in sample_values:
            for pattern in self.COMPOSITION_PATTERNS:
                if re.match(pattern, value):
                    matches += 1
                    break
        
        # If more than 70% match, consider it a composition column
        if len(sample_values) > 0 and matches / len(sample_values) > 0.7:
            return True
        
        return False
    
    def set_target_column(self, column: str) -> None:
        """Set the target column for prediction.
        
        Args:
            column: Name of target column
        """
        if self._data is None:
            raise ValueError("No data loaded")
        
        if column not in self._data.columns:
            raise ValueError(f"Column '{column}' not found in dataset")
        
        self._target_column = column
        self._feature_columns = [c for c in self._data.columns if c != column]
        
        # Determine if classification or regression
        target_series = self._data[column]
        if pd.api.types.is_numeric_dtype(target_series):
            # If integer with few unique values, likely classification
            if (pd.api.types.is_integer_dtype(target_series) and 
                target_series.nunique() <= 20):
                self._is_classification = True
            else:
                self._is_classification = False
        else:
            self._is_classification = True
        
        # Update dataset info
        if self._dataset_info:
            self._dataset_info.target_column = column
            for col_info in self._dataset_info.columns.values():
                col_info.is_target = (col_info.name == column)
        
        self.logger.info(f"Target column set to '{column}' "
                        f"({'classification' if self._is_classification else 'regression'})")
    
    def get_composition_columns(self) -> List[str]:
        """Get list of columns identified as chemical compositions.
        
        Returns:
            List of column names
        """
        if self._dataset_info is None:
            return []
        
        return [
            col for col, info in self._dataset_info.columns.items()
            if info.is_composition
        ]
    
    def get_preprocessor(
        self,
        numeric_strategy: str = 'standard',
        categorical_strategy: str = 'onehot',
        impute_strategy: str = 'mean',
        handle_unknown: str = 'ignore'
    ) -> ColumnTransformer:
        """Create preprocessing pipeline.
        
        Args:
            numeric_strategy: Scaling strategy ('standard', 'minmax', 'robust', 'none')
            categorical_strategy: Encoding strategy ('onehot', 'label', 'none')
            impute_strategy: Imputation strategy ('mean', 'median', 'most_frequent', 'knn')
            handle_unknown: How to handle unknown categories ('ignore', 'error')
            
        Returns:
            Configured ColumnTransformer
        """
        if self._data is None or self._target_column is None:
            raise ValueError("Data and target column must be set before creating preprocessor")
        
        # Separate numeric and categorical columns (excluding target)
        numeric_cols = []
        categorical_cols = []
        
        for col in self._feature_columns:
            # FIX: Usar dataset_info si está disponible, sino inferir del DataFrame
            if self._dataset_info is not None and col in self._dataset_info.columns:
                col_info = self._dataset_info.columns[col]
                if col_info.is_numeric:
                    numeric_cols.append(col)
                elif col_info.is_categorical:
                    categorical_cols.append(col)
            else:
                # Inferir tipo de dato del DataFrame
                if pd.api.types.is_numeric_dtype(self._data[col]):
                    numeric_cols.append(col)
                else:
                    categorical_cols.append(col)
        
        transformers = []
        
        # Numeric transformer
        if numeric_cols:
            numeric_steps = []
            
            # Imputation
            if impute_strategy == 'knn':
                numeric_steps.append(('imputer', KNNImputer(n_neighbors=5)))
            else:
                numeric_steps.append(('imputer', SimpleImputer(strategy=impute_strategy)))
            
            # Scaling
            if numeric_strategy == 'standard':
                numeric_steps.append(('scaler', StandardScaler()))
            elif numeric_strategy == 'minmax':
                numeric_steps.append(('scaler', MinMaxScaler()))
            elif numeric_strategy == 'robust':
                numeric_steps.append(('scaler', RobustScaler()))
            
            numeric_pipeline = Pipeline(steps=numeric_steps)
            transformers.append(('numeric', numeric_pipeline, numeric_cols))
        
        # Categorical transformer
        if categorical_cols:
            categorical_steps = []
            
            # Imputation
            categorical_steps.append(
                ('imputer', SimpleImputer(strategy='most_frequent'))
            )
            
            # Encoding
            if categorical_strategy == 'onehot':
                categorical_steps.append(
                    ('encoder', OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False))
                )
            elif categorical_strategy == 'label':
                # Label encoding will be done separately
                pass
            
            categorical_pipeline = Pipeline(steps=categorical_steps)
            transformers.append(('categorical', categorical_pipeline, categorical_cols))
        
        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            verbose_feature_names_out=False
        )
        
        self._preprocessor = preprocessor
        return preprocessor
    
    def prepare_data(
        self,
        test_size: float = 0.2,
        random_state: int = 42,
        use_validation: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """Prepare data for training.
        
        Args:
            test_size: Fraction of data for testing
            random_state: Random seed
            use_validation: Whether to use external validation data
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test) or 
            (X_train, X_val, y_train, y_val) if use_validation=True
        """
        if self._data is None or self._target_column is None:
            raise ValueError("Data and target column must be set")
        
        from sklearn.model_selection import train_test_split
        
        X = self._data[self._feature_columns]
        y = self._data[self._target_column]
        
        # Encode target if classification
        if self._is_classification and not pd.api.types.is_numeric_dtype(y):
            self._label_encoder = LabelEncoder()
            y = self._label_encoder.fit_transform(y)
        
        if use_validation and self._validation_data is not None:
            X_train = X
            y_train = y
            
            X_test = self._validation_data[self._feature_columns]
            y_test = self._validation_data[self._target_column]
            
            if self._label_encoder is not None:
                y_test = self._label_encoder.transform(y_test)
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
        
        # Fit preprocessor and transform
        # FIX: Siempre recrear preprocessor para asegurar que use las feature_columns actuales
        self._preprocessor = None
        self.get_preprocessor()
        
        X_train_processed = self._preprocessor.fit_transform(X_train)
        X_test_processed = self._preprocessor.transform(X_test)
        
        # FIX: Guardar los nombres de features procesados
        try:
            self._processed_feature_names = self._preprocessor.get_feature_names_out().tolist()
        except Exception:
            self._processed_feature_names = self._feature_columns
        
        return X_train_processed, X_test_processed, y_train.values, y_test.values
    
    def get_feature_names(self) -> List[str]:
        """Get feature names after preprocessing.
        
        Returns:
            List of feature names
        """
        # FIX: Usar nombres procesados si están disponibles
        if hasattr(self, '_processed_feature_names') and self._processed_feature_names:
            return self._processed_feature_names
        
        # Fallback: usar _feature_columns
        return self._feature_columns
    
    def get_preprocessed_feature_count(self) -> int:
        """Get the number of features after preprocessing.
        
        This returns the actual number of input features that the model will see,
        including one-hot encoded categorical variables.
        
        Returns:
            Number of features after preprocessing
        """
        # If we have processed feature names, use that count
        if hasattr(self, '_processed_feature_names') and self._processed_feature_names:
            return len(self._processed_feature_names)
        
        # If preprocessor exists and is fitted, try to get output shape
        if self._preprocessor is not None:
            try:
                # Create a dummy sample to get output shape
                import numpy as np
                dummy = np.zeros((1, len(self._feature_columns)))
                output = self._preprocessor.transform(dummy)
                return output.shape[1]
            except Exception as e:
                self.logger.warning(f"Could not get preprocessor output shape: {e}")
        
        # Fallback: return raw feature count (will be wrong if one-hot encoding is used)
        return len(self._feature_columns)
    
    def save_preprocessor(self, path: Union[str, Path]) -> None:
        """Save fitted preprocessor to file.
        
        Args:
            path: Path to save preprocessor
        """
        if self._preprocessor is None:
            raise ValueError("Preprocessor not fitted")
        
        import joblib
        joblib.dump(self._preprocessor, path)
        self.logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path: Union[str, Path]) -> None:
        """Load preprocessor from file.
        
        Args:
            path: Path to preprocessor file
        """
        import joblib
        self._preprocessor = joblib.load(path)
        self.logger.info(f"Preprocessor loaded from {path}")
