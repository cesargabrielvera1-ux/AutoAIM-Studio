"""Inference Engine for standalone model prediction.

Supports both sklearn pipelines and PyTorch TorchScript models with lazy loading.
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import warnings

import numpy as np
import pandas as pd

from ..utils.logger import LoggerMixin


class InferenceEngine(LoggerMixin):
    """Inference engine for loading and running saved models.
    
    Supports lazy loading of dependencies (torch, sklearn, joblib).
    """
    
    def __init__(self, model_bundle_path: Union[str, Path]):
        """Initialize inference engine.
        
        Args:
            model_bundle_path: Path to directory containing manifest.json and model files
        """
        self.bundle_path = Path(model_bundle_path)
        self.manifest_path = self.bundle_path / "manifest.json"
        
        # Manifest data (loaded immediately)
        self._manifest: Optional[Dict[str, Any]] = None
        
        # Model components (loaded lazily)
        self._model: Optional[Any] = None
        self._preprocessor: Optional[Any] = None
        self._model_loaded: bool = False
        
        # Version info for validation
        self._loaded_sklearn_version: Optional[str] = None
        self._loaded_pytorch_version: Optional[str] = None
        
        # Load manifest immediately
        self._load_manifest()
    
    def _load_manifest(self) -> None:
        """Load manifest.json from bundle."""
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {self.manifest_path}")
        
        with open(self.manifest_path, 'r') as f:
            self._manifest = json.load(f)
        
        self.logger.info(f"Loaded manifest for model: {self._manifest.get('model_name', 'unknown')}")

        # Sanitize feature names to remove CSV artifact columns
        self._sanitize_feature_names()

    def _sanitize_feature_names(self) -> None:
        """Remove 'Unnamed: xxx' columns from manifest feature_names.

        These columns are artifacts from CSV files with trailing commas
        and should never be used as features.
        """
        if self._manifest is None:
            return
        feature_names = self._manifest.get("feature_names", [])
        if not feature_names:
            return
        clean_names = [c for c in feature_names if not str(c).startswith("Unnamed:")]
        dropped = len(feature_names) - len(clean_names)
        if dropped > 0:
            self._manifest["feature_names"] = clean_names
            self.logger.warning(
                f"Removed {dropped} 'Unnamed' artifact columns from feature_names. "
                f"Clean feature count: {len(clean_names)}"
            )

    def load(self) -> None:
        """Load model and preprocessor (lazy loading).
        
        This method performs lazy imports to avoid loading unnecessary dependencies.
        """
        if self._model_loaded:
            return
        
        if self._manifest is None:
            raise RuntimeError("Manifest not loaded")
        
        model_type = self._manifest.get("model_type")
        
        try:
            if model_type == "sklearn":
                self._load_sklearn_model()
            elif model_type == "pytorch":
                self._load_pytorch_model()
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            self._model_loaded = True
            self.logger.info(f"Model loaded successfully: {self._manifest.get('model_name')}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model: {e}")
    
    def _load_sklearn_model(self) -> None:
        """Load sklearn model with lazy import."""
        import joblib
        import sklearn
        
        self._loaded_sklearn_version = sklearn.__version__
        
        model_filename = self._manifest.get("model_filename", "pipeline.joblib")
        model_path = self.bundle_path / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self._model = joblib.load(model_path)
        
        # For sklearn, preprocessor is usually part of the pipeline
        # But check if there's a separate preprocessor
        preprocessor_filename = self._manifest.get("preprocessor_filename")
        if preprocessor_filename:
            preprocessor_path = self.bundle_path / preprocessor_filename
            if preprocessor_path.exists():
                self._preprocessor = joblib.load(preprocessor_path)
        
        self.logger.info(f"Sklearn model loaded (sklearn {self._loaded_sklearn_version})")
    
    def _load_pytorch_model(self) -> None:
        """Load PyTorch model with lazy import."""
        import torch
        import joblib
        
        self._loaded_pytorch_version = torch.__version__
        
        model_filename = self._manifest.get("model_filename", "model.pt")
        model_path = self.bundle_path / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load TorchScript model
        self._model = torch.jit.load(model_path)
        self._model.eval()
        
        # Load preprocessor (required for PyTorch)
        preprocessor_filename = self._manifest.get("preprocessor_filename")
        if preprocessor_filename:
            preprocessor_path = self.bundle_path / preprocessor_filename
            if preprocessor_path.exists():
                self._preprocessor = joblib.load(preprocessor_path)
            else:
                self.logger.warning(f"Preprocessor file not found: {preprocessor_path}")
        else:
            self.logger.warning("No preprocessor specified for PyTorch model")
        
        self.logger.info(f"PyTorch model loaded (torch {self._loaded_pytorch_version})")
    
    def validate_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data against model requirements.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Validation report dictionary with:
            - valid: bool - whether input is valid
            - missing_columns: list - columns required but not present
            - extra_columns: list - columns present but not required
            - type_issues: dict - columns with type mismatches
            - message: str - human-readable summary
        """
        if self._manifest is None:
            raise RuntimeError("Manifest not loaded")
        
        required_features = set(self._manifest.get("feature_names", []))
        actual_columns = set(df.columns)
        
        # Check for missing columns
        missing_columns = list(required_features - actual_columns)
        
        # Check for extra columns
        extra_columns = list(actual_columns - required_features)

        # Separate Unnamed artifact columns from truly extra columns
        unnamed_cols = [c for c in extra_columns if str(c).startswith("Unnamed:")]
        real_extra = [c for c in extra_columns if not str(c).startswith("Unnamed:")]
        if unnamed_cols:
            extra_columns = real_extra  # Report only real extra columns
        
        # Check for type issues (basic)
        type_issues = {}
        for col in required_features & actual_columns:
            if col in df.columns:
                # Check if numeric column has non-numeric data
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Check if it's a string that might be convertible
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except:
                        type_issues[col] = f"Non-numeric type: {df[col].dtype}"
        
        # Determine validity
        valid = len(missing_columns) == 0 and len(type_issues) == 0
        
        # Create message
        messages = []
        if missing_columns:
            messages.append(f"Missing columns: {missing_columns}")
        if extra_columns:
            messages.append(f"Extra columns (will be ignored): {extra_columns}")
        if type_issues:
            messages.append(f"Type issues: {list(type_issues.keys())}")
        
        if valid and not extra_columns:
            message = "All required features present"
        elif valid:
            message = "All required features present (some extra columns will be ignored)"
        else:
            message = "; ".join(messages)
        
        return {
            "valid": valid,
            "missing_columns": missing_columns,
            "extra_columns": extra_columns,
            "type_issues": type_issues,
            "message": message,
            "required_count": len(required_features),
            "actual_count": len(actual_columns),
            "matched_count": len(required_features & actual_columns)
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on input data.
        
        Args:
            df: Input DataFrame with required features
            
        Returns:
            DataFrame with predictions added as 'prediction' column
        """
        if not self._model_loaded:
            self.load()
        
        # Validate input
        validation = self.validate_input(df)
        if not validation["valid"]:
            raise ValueError(f"Invalid input: {validation['message']}")
        
        # Strip any "Unnamed" artifact columns from the input DataFrame
        unnamed_in_df = [c for c in df.columns if str(c).startswith("Unnamed:")]
        if unnamed_in_df:
            df = df.drop(columns=unnamed_in_df)
            self.logger.warning(f"Dropped {len(unnamed_in_df)} 'Unnamed' artifact columns from input: {unnamed_in_df}")

        # Select only required features in correct order
        feature_names = self._manifest.get("feature_names", [])
        
        # Defensive: filter out any remaining "Unnamed" artifact columns
        # (for models saved before the data_manager fix)
        clean_feature_names = [c for c in feature_names if not str(c).startswith("Unnamed:")]
        unnamed_in_features = [c for c in feature_names if str(c).startswith("Unnamed:")]
        if unnamed_in_features:
            self.logger.warning(
                f"Model has {len(unnamed_in_features)} 'Unnamed' artifact columns in feature_names. "
                f"They will be ignored."
            )
            feature_names = clean_feature_names
        
        # Check all requested features exist in the DataFrame
        missing = [c for c in feature_names if c not in df.columns]
        if missing:
            raise ValueError(
                f"Required columns not found in input data: {missing}\n\n"
                f"This can happen when a model was trained on a CSV with trailing commas "
                f"that created 'Unnamed' columns, or when the input data is missing features. "
                f"Please retrain the model with clean data or provide the missing columns."
            )
        
        X = df[feature_names].copy()
        
        # Handle NaN values
        X = self._handle_missing_values(X)
        
        # Make prediction based on model type
        model_type = self._manifest.get("model_type")
        
        if model_type == "sklearn":
            predictions = self._predict_sklearn(X)
        elif model_type == "pytorch":
            predictions = self._predict_pytorch(X)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create result DataFrame
        result = df.copy()
        result['prediction'] = predictions
        
        return result
    
    def _predict_sklearn(self, X: pd.DataFrame) -> np.ndarray:
        """Make prediction with sklearn model."""
        # Model is a pipeline that includes preprocessing
        predictions = self._model.predict(X)
        return predictions
    
    def _predict_pytorch(self, X: pd.DataFrame) -> np.ndarray:
        """Make prediction with PyTorch model."""
        import torch
        
        # Apply preprocessor if available
        if self._preprocessor is not None:
            X_processed = self._preprocessor.transform(X)
        else:
            X_processed = X.values
        
        # Convert to tensor
        X_tensor = torch.tensor(X_processed, dtype=torch.float32)
        
        # Make prediction
        with torch.no_grad():
            predictions = self._model(X_tensor)
        
        # Convert to numpy
        predictions = predictions.numpy()
        
        # Flatten if needed
        if predictions.ndim > 1 and predictions.shape[1] == 1:
            predictions = predictions.flatten()
        
        return predictions
    
    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in input data."""
        # Check for NaN values
        nan_count = X.isna().sum().sum()
        
        if nan_count > 0:
            self.logger.warning(f"Input contains {nan_count} NaN values. Attempting imputation...")
            
            # For numeric columns, fill with median
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if X[col].isna().any():
                    median_val = X[col].median()
                    X[col] = X[col].fillna(median_val)
                    self.logger.info(f"Filled NaN in '{col}' with median: {median_val}")
        
        return X
    
    def check_version_compatibility(self) -> Tuple[bool, str]:
        """Check if current library versions are compatible with model.
        
        Returns:
            Tuple of (is_compatible, message)
        """
        if self._manifest is None:
            return False, "Manifest not loaded"
        
        warnings_list = []
        
        # Check sklearn version
        manifest_sklearn = self._manifest.get("sklearn_version")
        if manifest_sklearn and manifest_sklearn != "unknown":
            try:
                import sklearn
                current_sklearn = sklearn.__version__
                
                # Parse versions
                manifest_major_minor = ".".join(manifest_sklearn.split(".")[:2])
                current_major_minor = ".".join(current_sklearn.split(".")[:2])
                
                if manifest_major_minor != current_major_minor:
                    diff = abs(float(current_major_minor) - float(manifest_major_minor))
                    if diff >= 0.2:
                        warnings_list.append(
                            f"scikit-learn version mismatch: model={manifest_sklearn}, "
                            f"current={current_sklearn}. Major version differences may cause issues."
                        )
            except ImportError:
                warnings_list.append("scikit-learn not installed")
        
        # Check PyTorch version
        manifest_pytorch = self._manifest.get("pytorch_version")
        if manifest_pytorch and manifest_pytorch != "unknown":
            try:
                import torch
                current_pytorch = torch.__version__
                
                # Parse versions (PyTorch version format: 2.0.1+cu118)
                manifest_version = manifest_pytorch.split("+")[0]
                current_version = current_pytorch.split("+")[0]
                
                manifest_major_minor = ".".join(manifest_version.split(".")[:2])
                current_major_minor = ".".join(current_version.split(".")[:2])
                
                if manifest_major_minor != current_major_minor:
                    warnings_list.append(
                        f"PyTorch version mismatch: model={manifest_pytorch}, "
                        f"current={current_pytorch}. Version differences may cause issues."
                    )
            except ImportError:
                warnings_list.append("PyTorch not installed")
        
        if warnings_list:
            return False, "\n".join(warnings_list)
        
        return True, "Versions compatible"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get human-readable model information.
        
        Returns:
            Dictionary with model info
        """
        if self._manifest is None:
            return {"error": "Manifest not loaded"}
        
        return {
            "model_name": self._manifest.get("model_name", "Unknown"),
            "algorithm": self._manifest.get("algorithm", "Unknown"),
            "model_type": self._manifest.get("model_type", "Unknown"),
            "creation_date": self._manifest.get("creation_date", "Unknown"),
            "sklearn_version": self._manifest.get("sklearn_version", "N/A"),
            "pytorch_version": self._manifest.get("pytorch_version", "N/A"),
            "feature_count": len(self._manifest.get("feature_names", [])),
            "target_column": self._manifest.get("target_column", "Unknown"),
            "metrics": self._manifest.get("metrics", {}),
            "is_neural_network": self._manifest.get("is_neural_network", False)
        }
    
    def can_generate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check if missing features can be auto-generated.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (can_generate, list of generatable features)
        """
        if self._manifest is None:
            return False, []
        
        required_features = set(self._manifest.get("feature_names", []))
        
        # If no required features specified (legacy model), check if we have formula column
        if not required_features:
            has_formula = any(col.lower() in ['formula', 'composition', 'chem_formula'] 
                             for col in df.columns)
            if has_formula:
                # Can generate Magpie features
                return True, ['magpie_features']
            return False, []
        
        actual_columns = set(df.columns)
        missing_features = required_features - actual_columns
        
        if not missing_features:
            return False, []
        
        # Check if missing features are composition descriptors (formerly Magpie)
        generatable = []
        comp_prefixes = ['comp_', 'magpie_', 'atomic_', 'covalent_', 'electronegativity_', 
                        'ionization_', 'electron_', 'thermal_', 'density_', 'boiling_',
                        'melting_', 'group_', 'period_']
        
        for feat in missing_features:
            # Check if feature looks like a composition descriptor
            if any(feat.startswith(prefix) for prefix in comp_prefixes):
                generatable.append(feat)
        
        # Check if we have a formula/composition column to generate from
        has_formula = any(col.lower() in ['formula', 'composition', 'chem_formula'] 
                         for col in actual_columns)
        
        return len(generatable) > 0 and has_formula, generatable
    
    def generate_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Auto-generate missing Magpie-like features from formula column.
        
        Args:
            df: Input DataFrame with formula/composition column
            
        Returns:
            DataFrame with generated features added
        """
        # Find formula column
        formula_col = None
        for col in df.columns:
            if col.lower() in ['formula', 'composition', 'chem_formula']:
                formula_col = col
                break
        
        if formula_col is None:
            raise ValueError("No formula/composition column found for feature generation")
        
        # Import feature engineering
        from .feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        result = fe.add_composition_features(
            df,
            composition_column=formula_col,
            use_magpie=True,
            use_matminer=False
        )
        
        self.logger.info(f"Generated {len(result.columns) - len(df.columns)} features from {formula_col}")
        
        return result
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        """Get manifest data."""
        return self._manifest


class LegacyInferenceEngine(LoggerMixin):
    """Inference engine for legacy models without manifest.
    
    Supports .joblib and .pkl files saved with older versions of the software.
    """
    
    def __init__(self, model: Any, file_path: str):
        """Initialize legacy inference engine.
        
        Args:
            model: Loaded model object
            file_path: Path to the model file
        """
        self._model = model
        self._file_path = file_path
        self._model_loaded = True
        
        # Create a basic manifest for compatibility
        self._manifest = {
            "model_type": "sklearn" if not hasattr(model, 'forward') else "pytorch",
            "algorithm": type(model).__name__,
            "model_name": Path(file_path).stem,
            "creation_date": "Unknown (legacy model)",
            "sklearn_version": "Unknown",
            "pytorch_version": None,
            "feature_names": [],  # Will be inferred from input
            "target_column": "Unknown",
            "preprocessing_checksum": "null",
            "metrics": {},
            "is_neural_network": hasattr(model, 'forward'),
            "nn_config": {},
            "legacy": True
        }
    
    def load(self) -> None:
        """Model is already loaded in constructor."""
        pass
    
    def validate_input(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate input data - for legacy models, accept any numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        return {
            "valid": len(numeric_cols) > 0,
            "missing_columns": [],
            "extra_columns": [],
            "type_issues": {},
            "message": f"Legacy model - will use {len(numeric_cols)} numeric columns",
            "required_count": "Unknown",
            "actual_count": len(df.columns),
            "matched_count": len(numeric_cols)
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions using legacy model."""
        # Select only numeric columns
        X = df.select_dtypes(include=[np.number])
        
        if X.empty:
            raise ValueError("No numeric columns found in input data")
        
        # Handle NaN values
        X = X.fillna(X.median())
        
        # Make prediction
        if hasattr(self._model, 'predict'):
            predictions = self._model.predict(X)
        elif hasattr(self._model, 'forward'):
            # PyTorch model
            import torch
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
            with torch.no_grad():
                predictions = self._model(X_tensor).numpy()
            if predictions.ndim > 1:
                predictions = predictions.flatten()
        else:
            raise ValueError("Model does not have predict or forward method")
        
        # Create result DataFrame
        result = df.copy()
        result['prediction'] = predictions
        
        return result
    
    def check_version_compatibility(self) -> Tuple[bool, str]:
        """Legacy models skip version checking."""
        return True, "Legacy model - version checking skipped"
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get basic model info."""
        return {
            "model_name": self._manifest["model_name"],
            "algorithm": self._manifest["algorithm"],
            "model_type": self._manifest["model_type"],
            "creation_date": self._manifest["creation_date"],
            "sklearn_version": "N/A (legacy)",
            "pytorch_version": "N/A (legacy)",
            "feature_count": "Unknown",
            "target_column": "Unknown",
            "metrics": {},
            "is_neural_network": self._manifest["is_neural_network"],
            "legacy": True
        }
    
    def can_generate_features(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Check if Magpie features can be generated from formula column."""
        # Check if there's a formula/composition column
        has_formula = any(col.lower() in ['formula', 'composition', 'chem_formula'] 
                         for col in df.columns)
        
        if has_formula:
            return True, ['magpie_features']
        return False, []
    
    def generate_missing_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate Magpie features from formula column."""
        # Find formula column
        formula_col = None
        for col in df.columns:
            if col.lower() in ['formula', 'composition', 'chem_formula']:
                formula_col = col
                break
        
        if formula_col is None:
            raise ValueError("No formula/composition column found for feature generation")
        
        # Import feature engineering
        from .feature_engineering import FeatureEngineer
        
        fe = FeatureEngineer()
        result = fe.add_composition_features(
            df,
            composition_column=formula_col,
            use_magpie=True,
            use_matminer=False
        )
        
        self.logger.info(f"Generated {len(result.columns) - len(df.columns)} Magpie features from {formula_col}")
        
        return result
    
    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model_loaded
    
    @property
    def manifest(self) -> Optional[Dict[str, Any]]:
        """Get manifest data."""
        return self._manifest
