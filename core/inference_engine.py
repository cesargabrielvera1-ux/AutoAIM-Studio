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
from .ood_detector import OodDetector, format_ood_warning


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
        # v2.0: Out-of-distribution detector (loaded lazily, None for legacy bundles)
        self._ood_detector: Optional[OodDetector] = None
        
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

            # v2.0: Load OOD detector if present in bundle
            ood_path = self.bundle_path / "ood_detector.joblib"
            if ood_path.exists():
                try:
                    self._ood_detector = OodDetector.load(str(ood_path))
                    self.logger.info(
                        f"OOD detector loaded (threshold={self._ood_detector.threshold_:.3f}, "
                        f"n_train={self._ood_detector.n_train_samples_})"
                    )
                except Exception as e:
                    self.logger.warning(f"Failed to load OOD detector: {e}")

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
        import torch.nn as nn
        
        self._loaded_pytorch_version = torch.__version__
        
        model_filename = self._manifest.get("model_filename", "model.pt")
        model_path = self.bundle_path / model_filename
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # v1.3.0 FIX: Support both TorchScript and full nn.Module formats.
        # TorchScript is saved with torch.jit.save() and loaded with torch.jit.load().
        # Full model is saved with torch.save(nn.Module) and loaded with torch.load().
        model_format = self._manifest.get("model_format", "torchscript")
        
        # PyTorch 2.6+ changed weights_only default to True; we use False for our trusted checkpoints
        _torch_load_kwargs = {"map_location": "cpu"}
        try:
            # Only add weights_only if PyTorch supports it (2.6+)
            import torch.serialization
            _torch_load_kwargs["weights_only"] = False
        except (AttributeError, TypeError):
            pass
        
        try:
            if model_format == "full_model":
                # Load full nn.Module (saved when TorchScript conversion failed)
                self._model = torch.load(model_path, **_torch_load_kwargs)
                self._model.eval()
                self._model.cpu()
                self.logger.info("PyTorch full nn.Module loaded (CPU)")
            else:
                # Default: try TorchScript first
                self._model = torch.jit.load(model_path, map_location='cpu')
                self._model.eval()
                self.logger.info("PyTorch TorchScript model loaded (CPU)")
        except Exception as e:
            # Fallback: if TorchScript fails, try full model load
            self.logger.warning(f"Primary load method failed: {e}. Trying alternative...")
            try:
                self._model = torch.load(model_path, **_torch_load_kwargs)
                if isinstance(self._model, nn.Module):
                    self._model.eval()
                    self._model.cpu()
                    self.logger.info("PyTorch full nn.Module loaded via fallback (CPU)")
                else:
                    raise ValueError(f"Loaded object is not a nn.Module: {type(self._model)}")
            except Exception as e2:
                raise RuntimeError(f"Failed to load PyTorch model. Both TorchScript and full model loading failed. "
                                   f"TorchScript error: {e}. Full model error: {e2}")
        
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
            - requires_structure_linking: bool - model needs crystal/molecule files
            - missing_linked_features: list - linked features that are missing
            - needs_linking_action: bool - user must provide structure files
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

        # v2.0: Detect missing linked-structure features (crystal_*, mol_*, has_*)
        linked_prefixes = self._manifest.get("linked_feature_prefixes", [])
        missing_linked = []
        if linked_prefixes and missing_columns:
            for prefix in linked_prefixes:
                if prefix in ('crystal_', 'mol_'):
                    missing_linked.extend([
                        c for c in missing_columns if c.startswith(prefix)
                    ])
            # has_* features are functional-group flags generated alongside
            # mol_* features by MoleculeFeaturizer — treat them as linked too
            if 'mol_' in linked_prefixes:
                missing_linked.extend([
                    c for c in missing_columns
                    if c.startswith('has_') and c not in missing_linked
                ])

        # Check for type issues (basic)
        type_issues = {}
        for col in required_features & actual_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col], errors='raise')
                    except Exception:
                        type_issues[col] = f"Non-numeric type: {df[col].dtype}"

        # v2.0: If ONLY linked features are missing, signal that linking is needed
        # rather than a hard failure — the user can call predict_with_linked_structures()
        needs_linking_action = len(missing_linked) > 0 and len(missing_columns) == len(missing_linked)
        all_required_present = len(missing_columns) == 0

        # Determine validity
        if needs_linking_action:
            # Not strictly "valid" but actionable — caller can use predict_with_linked_structures()
            valid = False
        else:
            valid = all_required_present and len(type_issues) == 0

        # Build message
        messages = []
        if missing_linked and needs_linking_action:
            structure_type = (
                "crystal structure (CIF)" if 'crystal_' in linked_prefixes
                else "molecular structure (.xyz)" if 'mol_' in linked_prefixes
                else "structure"
            )
            messages.append(
                f"This model requires linked {structure_type} features that are "
                f"missing from the input data: {missing_linked}. "
                f"Use predict_with_linked_structures() to automatically link "
                f"structure files before prediction."
            )
        elif missing_columns:
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
            "matched_count": len(required_features & actual_columns),
            "requires_structure_linking": self._manifest.get("requires_structure_linking", False),
            "missing_linked_features": missing_linked,
            "needs_linking_action": needs_linking_action,
        }
    
    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions on input data.

        v2.0: GaussianProcessRegressor produces a native 'prediction_std'
        column when the loaded model is a GPR. All other models return only
        the 'prediction' column.
        
        Args:
            df: Input DataFrame with required features
            
        Returns:
            DataFrame with predictions added as 'prediction' column,
            and 'prediction_std' when the model is a Gaussian Process.
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
        
        # v2.0: Detect GaussianProcessRegressor for native uncertainty.
        # When the saved model is a Pipeline, drill into the last step.
        _actual_model = self._model
        if hasattr(self._model, 'named_steps') and self._model.named_steps:
            _actual_model = list(self._model.named_steps.values())[-1]
        is_gpr = (
            model_type == "sklearn"
            and hasattr(self._model, 'predict')
            and 'GaussianProcessRegressor' in type(_actual_model).__name__
        )

        if model_type == "sklearn":
            if is_gpr:
                # GPR: get both mean and std in one call
                mean, std = self._predict_sklearn_with_std(X)
            else:
                predictions = self._predict_sklearn(X)
        elif model_type == "pytorch":
            predictions = self._predict_pytorch(X)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create result DataFrame
        result = df.copy()
        if is_gpr:
            result['prediction'] = mean
            result['prediction_std'] = std
        else:
            result['prediction'] = predictions
        
        return result

    def predict_with_ood(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with out-of-distribution (OOD) flags.

        v2.0: Returns the same columns as ``predict()``, plus three OOD
        columns when an applicability-domain detector was bundled with the
        model:

        * ``ood_distance`` – Euclidean distance to the nearest training
          neighbour in standardised feature space.
        * ``ood_flag``     – ``True`` if the sample lies outside the
          applicability domain (distance > threshold).
        * ``ood_warning``  – Human-readable explanation (empty string for
          in-domain samples).

        If the model bundle was created before v2.0 and has no OOD detector,
        the three columns are still present but filled with ``NaN`` /
        ``False`` / ``""`` so that downstream code (GUI table, CSV export)
        does not need branching logic.

        Args:
            df: Input DataFrame with required features

        Returns:
            DataFrame with ``prediction`` (+ ``prediction_std`` for GP)
            and the three OOD columns described above.
        """
        # 1. Base predictions (same as predict())
        result = self.predict(df)

        # 2. OOD detection
        if self._ood_detector is None:
            # Legacy bundle — fill with neutral values so the GUI table
            # can always expect these columns.
            result['ood_distance'] = np.nan
            result['ood_flag'] = False
            result['ood_warning'] = ""
            return result

        # Re-extract the feature matrix (same path as predict())
        feature_names = self._manifest.get("feature_names", [])
        clean_feature_names = [c for c in feature_names if not str(c).startswith("Unnamed:")]
        X = df[clean_feature_names].copy()
        X = self._handle_missing_values(X)
        X_np = np.asarray(X, dtype=np.float64)

        distances, flags = self._ood_detector.predict(X_np)

        result['ood_distance'] = distances
        result['ood_flag'] = flags
        result['ood_warning'] = [
            format_ood_warning(d, self._ood_detector.threshold_,
                               self._ood_detector.threshold_percentile)
            if flag else ""
            for d, flag in zip(distances, flags)
        ]

        # Log summary
        n_ood = int(flags.sum())
        if n_ood > 0:
            self.logger.warning(
                f"{n_ood}/{len(flags)} samples flagged as out-of-distribution "
                f"(threshold={self._ood_detector.threshold_:.3f})."
            )

        return result

    def predict_with_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Make predictions with confidence intervals.
        
        v1.5.0: Adds 'lower_bound' and 'upper_bound' columns based on
        CV residual analysis stored in the model bundle.
        
        Args:
            df: Input DataFrame with required features
            
        Returns:
            DataFrame with 'prediction', 'lower_bound', 'upper_bound' columns
        """
        result = self.predict(df)
        
        # Get CI params from manifest
        ci_params = self._manifest.get("confidence_interval_params", {})
        if not ci_params:
            result['lower_bound'] = np.nan
            result['upper_bound'] = np.nan
            return result
        
        preds = result['prediction'].values
        
        if ci_params.get('type') == 'regression':
            margin = ci_params.get('quantile_95', 2.0 * ci_params.get('residuals_std', 0))
            result['lower_bound'] = preds - margin
            result['upper_bound'] = preds + margin
        else:
            result['lower_bound'] = np.nan
            result['upper_bound'] = np.nan
        
        return result
    
    def _predict_sklearn(self, X: pd.DataFrame) -> np.ndarray:
        """Make prediction with sklearn model."""
        # If the model is a Pipeline with a ColumnTransformer preprocessor,
        # we MUST pass the DataFrame so columns can be selected by name.
        # Only convert to numpy for plain estimators trained on arrays.
        if hasattr(self._model, 'named_steps') and 'preprocessor' in self._model.named_steps:
            predictions = self._model.predict(X)
        else:
            X_np = X.values if hasattr(X, 'values') else X
            predictions = self._model.predict(X_np)
        return predictions

    def _predict_sklearn_with_std(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make prediction with a GaussianProcessRegressor, returning mean + std.

        v2.0: GPR is the only sklearn model that natively supports uncertainty
        quantification via return_std=True. This helper is called by predict()
        when the loaded model is detected as a GPR.
        """
        if hasattr(self._model, 'named_steps') and 'preprocessor' in self._model.named_steps:
            # Pipeline: sklearn Pipeline.predict does NOT forward return_std.
            # We must call predict on the final estimator directly.
            # Step 1: transform through the pipeline up to the final estimator
            Xt = X
            for name, step in self._model.named_steps.items():
                if name == self._model.steps[-1][0]:
                    # Last step is the estimator — call with return_std
                    mean, std = step.predict(Xt, return_std=True)
                    return mean, std
                else:
                    Xt = step.transform(Xt)
            # Fallback: should never reach here
            return self._model.predict(X), np.zeros(len(X))
        else:
            X_np = X.values if hasattr(X, 'values') else X
            mean, std = self._model.predict(X_np, return_std=True)
            return mean, std
    
    def _predict_pytorch(self, X: pd.DataFrame) -> np.ndarray:
        """Make prediction with PyTorch model."""
        import torch
        
        # Apply preprocessor if available
        if self._preprocessor is not None:
            X_processed = self._preprocessor.transform(X)
        else:
            X_processed = X.values
        
        # Convert to tensor
        # v2.0 FIX: np.asarray() ensures sparse matrices / DataFrames become dense arrays.
        X_tensor = torch.tensor(np.asarray(X_processed), dtype=torch.float32)
        
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
        
        # v2.0: CV provenance — how the model was validated during training.
        cv_label = None
        cv_cfg = self._manifest.get("cv_config") or {}
        if cv_cfg.get('strategy'):
            try:
                from .cv_display import cv_strategy_label
                cv_label = cv_strategy_label(cv_cfg['strategy'], cv_cfg)
            except Exception:
                cv_label = cv_cfg['strategy']

        # v2.0: Gaussian Process uncertainty metadata
        gp_meta = self._manifest.get("gp_metadata")
        uncertainty_info = None
        if gp_meta:
            uncertainty_info = {
                "has_native_uncertainty": True,
                "kernel": gp_meta.get('kernel_str', 'N/A'),
                "alpha": gp_meta.get('alpha'),
                "uncertainty_column": "prediction_std",
                "note": "Prediction standard deviation is the native posterior uncertainty of the Gaussian Process. Lower values = higher confidence.",
            }

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
            "is_neural_network": self._manifest.get("is_neural_network", False),
            # v2.0: e.g. "Group K-Fold (by chemical system)" — None if unknown.
            "cv_validation": cv_label,
            # v2.0: Gaussian Process native uncertainty info
            "uncertainty": uncertainty_info,
        }
    
    # ── v2.0: Linked-structure support ────────────────────────────

    def is_structure_linked_model(self) -> bool:
        """Return True if this model requires linked structure features.

        Use this in the UI to show/hide structure-linking controls.
        """
        if self._manifest is None:
            return False
        return self._manifest.get("requires_structure_linking", False)

    def get_linked_structure_requirements(self) -> Dict[str, Any]:
        """Get information about the structure linking this model needs.

        Returns dict with:
          - requires_linking: bool
          - linked_prefixes: list of feature prefixes (e.g. ['crystal_'])
          - structure_type: 'crystal', 'molecule', or None
          - missing_feature_count: int (how many linked features are expected)
        """
        if self._manifest is None:
            return {
                "requires_linking": False,
                "linked_prefixes": [],
                "structure_type": None,
                "missing_feature_count": 0,
            }
        prefixes = self._manifest.get("linked_feature_prefixes", [])
        feature_names = self._manifest.get("feature_names", [])
        linked_count = sum(
            1 for f in feature_names
            if any(f.startswith(p) for p in prefixes if p not in ('composition',))
        )
        structure_type = None
        if 'crystal_' in prefixes:
            structure_type = 'crystal'
        elif 'mol_' in prefixes:
            structure_type = 'molecule'

        return {
            "requires_linking": self._manifest.get("requires_structure_linking", False),
            "linked_prefixes": prefixes,
            "structure_type": structure_type,
            "missing_feature_count": linked_count,
        }

    def predict_with_linked_structures(
        self,
        df: pd.DataFrame,
        id_column: str,
        structure_folder: str,
        structure_type: Optional[str] = None,
    ) -> pd.DataFrame:
        """Link structure files to the dataset, then predict.

        This is the **recommended** way to predict with models that were
        trained on linked crystal or molecule features. It automatically:
          1. Detects the structure type (crystal vs molecule) from the manifest
          2. Links the appropriate structure files (CIF or .xyz)
          3. Runs prediction on the fully-featured dataset

        Args:
            df: Input DataFrame with an ID column
            id_column: Column name to match with structure filenames
            structure_folder: Path to folder with .cif or .xyz files
            structure_type: 'crystal' or 'molecule' (auto-detected if None)

        Returns:
            DataFrame with predictions (same as predict())

        Raises:
            ValueError: If structure_type cannot be auto-detected
            RuntimeError: If linking fails (no files matched, etc.)
        """
        if not self._model_loaded:
            self.load()

        # Auto-detect structure type from manifest
        if structure_type is None:
            reqs = self.get_linked_structure_requirements()
            structure_type = reqs.get("structure_type")
            if structure_type is None:
                raise ValueError(
                    "Cannot auto-detect structure type. "
                    "This model does not appear to require linked structures, "
                    "or the manifest lacks linked_feature_prefixes. "
                    "Please specify structure_type='crystal' or 'molecule'."
                )

        # Validate ID column exists
        if id_column not in df.columns:
            raise ValueError(
                f"ID column '{id_column}' not found in input data. "
                f"Available columns: {list(df.columns)}"
            )

        folder = Path(structure_folder)
        if not folder.exists():
            raise ValueError(f"Structure folder not found: {structure_folder}")

        # Link structures
        self.logger.info(
            f"Linking {structure_type} structures from '{folder}' "
            f"using ID column '{id_column}'"
        )

        if structure_type == 'crystal':
            try:
                from .crystal_feature_extractor import CrystalFeatureExtractor
            except ImportError as e:
                raise RuntimeError(
                    "Crystal structure support not available. "
                    "Install pymatgen: pip install pymatgen-core"
                ) from e
            linker = CrystalFeatureExtractor()
            linked_df, report = linker.link_structures(
                df, id_column, str(folder)
            )

        elif structure_type == 'molecule':
            from .molecule_feature_linker import MoleculeFeatureLinker
            linker = MoleculeFeatureLinker()
            linked_df, report = linker.link_structures(
                df, id_column, str(folder)
            )

        else:
            raise ValueError(
                f"Unknown structure_type: '{structure_type}'. "
                f"Use 'crystal' or 'molecule'."
            )

        # Log linking results
        self.logger.info(
            f"Linking complete: {report.n_matched}/{report.n_rows_total} matched, "
            f"{report.n_missing} missing"
        )
        if report.warnings:
            for w in report.warnings:
                self.logger.warning(f"Link warning: {w}")

        if report.n_matched == 0:
            raise RuntimeError(
                f"No structure files matched. "
                f"Ensure filenames in '{folder}' match the '{id_column}' values."
            )

        # v2.0: Composition is auto-derived from structure files during linking.
        # Both CrystalFeatureExtractor and MoleculeFeatureLinker now include
        # periodic properties (electronegativity, atomic mass, etc.) directly
        # from the structure — no separate Magpie step needed.

        # Now predict on the linked dataset
        return self.predict(linked_df)

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
    
    def predict_from_structures(
        self,
        structure_dir: Optional[str] = None,
        structure_files: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Make predictions from crystal structure files.
        
        Loads structure files (CIF, POSCAR, XYZ), featurizes them with the
        CrystalStructureFeaturizer, and runs prediction using the trained model.
        
        This is the PREDICTION counterpart to training on crystal structures.
        It extracts the same ~154 numeric features that were used during training,
        then calls predict() with only those numeric features.
        
        Args:
            structure_dir: Directory containing structure files
            structure_files: List of individual structure file paths
            
        Returns:
            DataFrame with columns [structure_name, prediction] plus any
            structure metadata
        """
        # Lazy import crystal structure module
        try:
            from .crystal_structure import CrystalStructureLoader, CrystalStructureFeaturizer
        except ImportError as e:
            raise RuntimeError(
                "Crystal structure support not available. "
                "Install pymatgen: pip install pymatgen-core"
            ) from e
        
        # Load structures
        loader = CrystalStructureLoader()
        if structure_dir:
            n = loader.load_directory(structure_dir)
        elif structure_files:
            n = loader.load_files(structure_files)
        else:
            raise ValueError("Must provide structure_dir or structure_files")
        
        if n == 0:
            raise ValueError("No crystal structures loaded")
        
        self.logger.info(f"Loaded {n} structures for prediction")
        
        # Featurize
        featurizer = CrystalStructureFeaturizer()
        features_df = featurizer.featurize_structures(loader.structures)
        
        # Set structure_name as index (matching training behavior)
        if 'structure_name' in features_df.columns:
            features_df = features_df.set_index('structure_name')
        
        # Keep ONLY numeric columns (matching training behavior)
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df = features_df[numeric_cols].astype(float)
        
        self.logger.info(
            f"Featurized: {len(features_df)} structures, "
            f"{len(features_df.columns)} numeric features"
        )
        
        # Run prediction using numeric features only
        result = self.predict(features_df)
        
        # structure_name is preserved as the index
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
        
        # Make prediction — convert to numpy to avoid UserWarning
        if hasattr(self._model, 'predict'):
            predictions = self._model.predict(X.values)
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
