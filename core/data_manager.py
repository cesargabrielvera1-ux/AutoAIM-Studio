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
    is_id_column: bool = False  # v2.0: ID column for crystal structure linking
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
        self._id_column: Optional[str] = None  # v2.0: ID column for crystal linking
        self._feature_columns: List[str] = []
        self._preprocessor: Optional[Pipeline] = None
        self._label_encoder: Optional[LabelEncoder] = None
        self._is_classification: bool = False
        self._processed_feature_names: Optional[List[str]] = None
        # v2.0: Crystal feature extractor state
        self._crystal_extractor: Optional[Any] = None
        self._crystal_features_active: bool = False
        # v2.0: Store original file path for export and restore operations
        self._file_path: Optional[str] = None
        
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
    
    def get_column_info(self) -> Optional[DatasetInfo]:
        """Re-analyze and return current column information.
        
        This is a convenience wrapper around _analyze_dataset() that is safe
        to call after any data modification (link, unlink, preprocessing).
        Returns the refreshed DatasetInfo or None if no data is loaded.
        """
        if self._data is None:
            return None
        self._dataset_info = self._analyze_dataset()
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
    
    @property
    def id_column(self) -> Optional[str]:
        """v2.0: Get ID column name for crystal structure linking."""
        return self._id_column
    
    @property
    def crystal_features_active(self) -> bool:
        """v2.0: Check if crystal structure features are linked."""
        return self._crystal_features_active
    
    def set_id_column(self, column: Optional[str]) -> None:
        """v2.0: Set or clear the ID column for crystal structure linking.
        
        The ID column is excluded from feature columns automatically.
        """
        if column == self._id_column:
            return
        self._id_column = column
        if column and self._data is not None:
            self._update_feature_columns()
        self.logger.info(f"ID column set to: {column}")
    
    def set_crystal_extractor(self, extractor: Optional[Any]) -> None:
        """v2.0: Set the crystal feature extractor instance."""
        self._crystal_extractor = extractor
        self._crystal_features_active = extractor is not None and extractor.is_linked()

    def _update_feature_columns(self) -> None:
        """Recalculate feature columns excluding ID column and target.

        v2.0: Called after set_id_column to ensure the ID column is not
        included in the feature set for training.
        """
        if self._data is None:
            return
        numeric_cols = self._data.select_dtypes(include=[np.number]).columns.tolist()
        exclude = {self._target_column, self._id_column}
        self._feature_columns = [c for c in numeric_cols if c not in exclude]
        self.logger.info(
            f"Feature columns updated: {len(self._feature_columns)} features. "
            f"Excluded: target={self._target_column}, id={self._id_column}"
        )

    def unset_crystal_link(self) -> None:
        """v2.0: Remove crystal structure link and restore state.

        Clears the ID column, removes crystal features from the dataset,
        and resets the crystal extractor state.
        """
        if not self._crystal_features_active or self._data is None:
            return

        # Remove crystal feature columns
        crystal_cols = [c for c in self._data.columns if c.startswith('crystal_')]
        if crystal_cols:
            self._data = self._data.drop(columns=crystal_cols)
            self.logger.info(f"Removed {len(crystal_cols)} crystal feature columns")

        # Clear ID column marker (restore as regular column)
        old_id = self._id_column
        self._id_column = None

        # Recalculate feature columns (ID column may now be a feature again)
        self._update_feature_columns()
        self._processed_feature_names = None

        # Clear extractor state
        self._crystal_extractor = None
        self._crystal_features_active = False

        self.logger.info(f"Crystal link unset. ID column '{old_id}' restored.")

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
        
        # v2.0: Store original file path for export and restore
        self._file_path = str(file_path)
        
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
            
            # Auto-featurize validation data if training data has comp_* columns
            self._sync_validation_features()
            
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
                if isinstance(data[col].dtype, pd.CategoricalDtype) or unique_count <= 20:
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

        # v2.0: n_features should be the actual feature columns (usable for training),
        # NOT the total number of DataFrame columns. Exclude target and ID columns.
        exclude = {self._target_column, self._id_column}
        n_feature_cols = len([c for c in data.columns if c not in exclude])

        return DatasetInfo(
            n_samples=len(data),
            n_features=n_feature_cols,
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
    
    def _sync_validation_features(self) -> None:
        """Apply same featurization to validation data as training data.
        
        If training data has Magpie composition features (comp_*) but
        validation data doesn't, auto-apply Magpie featurizer if the
        validation data has a 'formula' column.
        """
        if self._data is None or self._validation_data is None:
            return
        
        training_cols = set(self._data.columns)
        val_cols = set(self._validation_data.columns)
        
        # Check if training has comp_* columns that validation is missing
        missing_comp = [c for c in training_cols 
                       if c.startswith('comp_') and c not in val_cols]
        
        if not missing_comp:
            return  # All good, no missing featurized columns
        
        if 'formula' not in self._validation_data.columns:
            self.logger.warning(
                f"Validation data missing {len(missing_comp)} featurized columns "
                f"(comp_*) and has no 'formula' column to regenerate them. "
                f"Please either: (1) include 'formula' in validation CSV, or "
                f"(2) pre-featurize validation data with same featurizer."
            )
            return
        
        # Auto-apply Magpie featurizer to validation data
        try:
            from .feature_engineering import MagpieFeatures
            magpie = MagpieFeatures(composition_column='formula')
            self._validation_data = magpie.transform(self._validation_data)
            new_cols = [c for c in self._validation_data.columns 
                       if c.startswith('comp_')]
            self.logger.info(
                f"Auto-featurized validation data: {len(new_cols)} Magpie "
                f"columns added (was missing {len(missing_comp)})."
            )
        except Exception as e:
            self.logger.warning(
                f"Could not auto-featurize validation data: {e}. "
                f"Validation may fail if column mismatch persists."
            )
    
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
        
        # v1.2.0 FIX: Use ONLY numeric columns as features.
        # String columns (structure_name, file paths, composition strings,
        # etc.) must NOT be included in _feature_columns. If they are,
        # the OneHotEncoder in get_preprocessor() will create one feature
        # per unique string value, exploding the feature count from ~154
        # to 600+ and making prediction impossible on new data.
        # v2.0: Also exclude the ID column used for crystal structure linking.
        numeric_cols = self._data.select_dtypes(include=[np.number]).columns.tolist()
        id_col = self._id_column
        self._feature_columns = [
            c for c in numeric_cols 
            if c != column and c != id_col
        ]
        
        # Log what was excluded
        excluded = [c for c in self._data.columns if c not in self._feature_columns and c != column]
        if excluded:
            self.logger.info(
                f"Excluded {len(excluded)} non-numeric columns from features: {excluded}"
            )
        
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
        
        self.logger.info(
            f"Target column set to '{column}' "
            f"({'classification' if self._is_classification else 'regression'}), "
            f"features: {len(self._feature_columns)} numeric columns"
        )
    
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

        # v2.0 CRITICAL FIX: Use integer column indices instead of column names.
        # This makes the ColumnTransformer compatible with BOTH pandas DataFrames
        # and numpy arrays. When passed a DataFrame, sklearn selects by position
        # (iloc) when indices are integers. When passed an array, integer indices
        # work natively. This eliminates the incompatibility between Pipeline
        # (which receives DataFrames from tabs) and SHAP / explainability code
        # (which works with numpy arrays).
        col_to_idx = {col: i for i, col in enumerate(self._feature_columns)}
        numeric_idx = [col_to_idx[c] for c in numeric_cols]
        categorical_idx = [col_to_idx[c] for c in categorical_cols]

        # Numeric transformer
        if numeric_idx:
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
            transformers.append(('numeric', numeric_pipeline, numeric_idx))

        # Categorical transformer
        if categorical_idx:
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
            transformers.append(('categorical', categorical_pipeline, categorical_idx))

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
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
        """Prepare data for training.

        v2.0 CRITICAL FIX: Returns RAW DataFrames (NOT preprocessed arrays) to
        eliminate data leakage in cross-validation. The preprocessor is created
        but NOT fitted here. Each CV fold will fit the scaler independently via
        sklearn Pipeline, ensuring the validation fold never leaks into the
        training statistics.

        Args:
            test_size: Fraction of data for testing
            random_state: Random seed
            use_validation: Whether to use external validation data

        Returns:
            Tuple of (X_train_raw_df, X_test_raw_df, y_train_arr, y_test_arr)
            where X_* are pandas DataFrames with original feature column names.
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

        # v2.0: Create the preprocessor but DO NOT fit it here.
        # Fitting inside each CV fold (via Pipeline) is the only way to avoid
        # data leakage. The preprocessor is stored so tabs can pass it to
        # trainer/optimizer/ensemble for Pipeline construction.
        self._preprocessor = None
        self.get_preprocessor()

        # v2.0: Store processed feature names for reference (e.g., model cards,
        # export). We do a one-off fit_transform on a dummy sample just to
        # discover the output shape / names — this does NOT affect training.
        try:
            _tmp = self._preprocessor.fit_transform(X_train.head(5))
            self._processed_feature_names = (
                self._preprocessor.get_feature_names_out().tolist()
            )
            # Reset so the preprocessor is truly unfitted for training
            self._preprocessor = None
            self.get_preprocessor()
        except Exception:
            self._processed_feature_names = self._feature_columns

        # Convert y to numpy arrays safely.
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)

        # v2.0: Return DataFrames (raw) so sklearn Pipeline can select columns
        # by name inside each CV fold.
        return X_train, X_test, y_train_arr, y_test_arr
    
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

    def get_composition_series(self) -> Optional[pd.Series]:
        """Return the detected composition/formula column as a Series.

        v2.0: Used to build chemical-system / formula / clustering groups for
        grouped cross-validation. Returns None if no composition column is
        detected among the raw columns.
        """
        if self._data is None:
            return None
        from .grouping import detect_composition_column
        col = detect_composition_column(self._data)
        return self._data[col] if col is not None else None

    def get_train_group_labels(
        self,
        group_labels_full: np.ndarray,
        test_size: float,
        random_state: int,
    ) -> np.ndarray:
        """Return group labels aligned with the X_train rows from prepare_data.

        v2.0: Grouped CV runs on the training set only, so group labels (built
        over the FULL dataset) must be restricted to the training rows. This
        recomputes the exact same train/test split that prepare_data performed:
        sklearn's train_test_split is fully deterministic given the number of
        samples and random_state, independent of the data content, so splitting
        an index array of the same length yields the identical partition.

        Args:
            group_labels_full: group label for EVERY row of the loaded dataset.
            test_size: same value passed to prepare_data.
            random_state: same value passed to prepare_data.

        Returns:
            Group labels for the training rows only.
        """
        if self._data is None:
            raise ValueError("No data loaded")
        labels = np.asarray(group_labels_full)
        n = len(self._data)
        if len(labels) != n:
            raise ValueError(
                f"Group labels length ({len(labels)}) does not match the number "
                f"of dataset rows ({n})."
            )
        from sklearn.model_selection import train_test_split
        idx = np.arange(n)
        train_idx, _ = train_test_split(idx, test_size=test_size, random_state=random_state)
        return labels[train_idx]
    
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
    
    def validate_columns(self) -> dict:
        """Validate that validation data has the same feature columns as training data.
        
        v1.5.0: Compares feature columns between training and validation datasets.
        Returns a detailed report for user-facing warnings.
        
        Returns:
            dict with keys: 'valid' (bool), 'missing_in_val' (list),
            'extra_in_val' (list), 'train_features' (int), 'val_features' (int),
            'message' (str)
        """
        result = {
            'valid': True,
            'missing_in_val': [],
            'extra_in_val': [],
            'train_features': 0,
            'val_features': 0,
            'message': '',
        }
        
        if self._data is None or self._validation_data is None:
            result['message'] = 'No training or validation data loaded.'
            return result
        
        train_features = set(self._feature_columns)
        val_cols = set(self._validation_data.columns)
        val_features = val_cols - {self._target_column} if self._target_column else val_cols
        
        result['train_features'] = len(train_features)
        result['val_features'] = len(val_features)
        
        missing = sorted(train_features - val_features)
        extra = sorted(val_features - train_features)
        
        result['missing_in_val'] = missing
        result['extra_in_val'] = extra
        
        if missing:
            result['valid'] = False
            result['message'] = (
                f"Validation data is missing {len(missing)} feature column(s) "
                f"present in training data: {', '.join(missing[:5])}"
                f"{'...' if len(missing) > 5 else ''}. "
                f"External validation metrics will not be available. "
                f"Ensure both datasets were featurized with the same method."
            )
        elif extra:
            result['message'] = (
                f"Validation data has {len(extra)} extra column(s) not in training: "
                f"{', '.join(extra[:5])}{'...' if len(extra) > 5 else ''}. "
                f"These will be ignored. Validation is aligned."
            )
        else:
            result['message'] = (
                f"Column validation passed: {len(train_features)} feature columns "
                f"match between training and validation datasets."
            )
        
        return result
    
    def get_validation_arrays(self, apply_preprocessing: bool = True) -> Optional[tuple]:
        """Get validation data as (X_val_df, y_val_arr).

        v2.0 FIX: Returns a RAW DataFrame by default. With sklearn Pipeline,
        the preprocessor lives inside the model, so validation data must be
        passed RAW to `pipeline.predict(X_val_df)`. The Pipeline applies the
        fitted preprocessor internally. This eliminates the previous leakage
        where the preprocessor was fitted on the full train set before CV.

        Args:
            apply_preprocessing: DEPRECATED — kept for API compatibility but
                ignored. Preprocessing now happens inside the Pipeline.

        Returns:
            Tuple of (X_val_df, y_val_arr) or None if unavailable.
        """
        if self._validation_data is None or self._target_column is None:
            return None
        try:
            X_val = self._validation_data[self._feature_columns]
            y_val = self._validation_data[self._target_column]
            return X_val, np.asarray(y_val)
        except Exception:
            return None
    
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
