"""Model saver with unified format for standalone inference.

Supports both sklearn pipelines and PyTorch TorchScript models.
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import warnings

import numpy as np
import pandas as pd

from ..utils.logger import LoggerMixin


class ModelSaver(LoggerMixin):
    """Save models in unified format for standalone inference.
    
    For sklearn models: saves complete pipeline (preprocessor + model) as joblib
    For PyTorch models: saves TorchScript model + separate preprocessor as joblib
    """
    
    def __init__(self):
        """Initialize model saver."""
        self._manifest: Optional[Dict[str, Any]] = None
    
    def save_model(
        self,
        model: Any,
        output_path: Union[str, Path],
        model_name: str,
        model_type: str,
        algorithm: str,
        feature_names: List[str],
        target_column: str,
        preprocessor: Optional[Any] = None,
        metrics: Optional[Dict[str, float]] = None,
        is_neural_network: bool = False,
        nn_config: Optional[Dict] = None
    ) -> Path:
        """Save model in unified format.
        
        Args:
            model: Trained model (sklearn or PyTorch)
            output_path: Directory path to save model bundle
            model_name: Name of the model
            model_type: 'sklearn' or 'pytorch'
            algorithm: Algorithm name (e.g., 'RandomForest', 'NeuralNetwork')
            feature_names: List of feature column names
            target_column: Name of target column
            preprocessor: Fitted preprocessor (for sklearn, can be None if in pipeline)
            metrics: Dictionary of model metrics
            is_neural_network: Whether this is a neural network
            nn_config: Neural network configuration (if applicable)
            
        Returns:
            Path to saved model bundle directory
        """
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Get version info (lazy imports)
        sklearn_version = self._get_sklearn_version()
        pytorch_version = self._get_pytorch_version()
        
        # Compute preprocessing checksum
        preprocessing_checksum = self._compute_preprocessor_checksum(preprocessor)
        
        # Create manifest
        self._manifest = {
            "model_type": model_type,
            "algorithm": algorithm,
            "model_name": model_name,
            "creation_date": datetime.now().isoformat(),
            "sklearn_version": sklearn_version,
            "pytorch_version": pytorch_version if model_type == "pytorch" else None,
            "feature_names": feature_names,
            "target_column": target_column,
            "preprocessing_checksum": preprocessing_checksum,
            "metrics": metrics or {},
            "is_neural_network": is_neural_network,
            "nn_config": nn_config or {}
        }
        
        # Save model based on type
        if model_type == "pytorch" or is_neural_network:
            self._save_pytorch_model(model, output_path, preprocessor)
        else:
            self._save_sklearn_model(model, output_path, preprocessor)
        
        # Save manifest
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {output_path}")
        return output_path
    
    def _save_sklearn_model(
        self,
        model: Any,
        output_path: Path,
        preprocessor: Optional[Any]
    ) -> None:
        """Save sklearn model with optional preprocessor.
        
        If preprocessor is provided, creates a pipeline [preprocessor, model].
        Otherwise, saves just the model.
        """
        import joblib
        from sklearn.pipeline import Pipeline
        
        # If preprocessor exists, create a unified pipeline
        if preprocessor is not None:
            # Check if model is already a pipeline
            if hasattr(model, 'named_steps'):
                # Model is already a pipeline, save as-is
                pipeline = model
            else:
                # Create new pipeline with preprocessor and model
                pipeline = Pipeline([
                    ('preprocessor', preprocessor),
                    ('model', model)
                ])
        else:
            pipeline = model
        
        # Save pipeline
        model_path = output_path / "pipeline.joblib"
        joblib.dump(pipeline, model_path)
        
        self._manifest["model_filename"] = "pipeline.joblib"
        self._manifest["preprocessor_filename"] = None
        
        self.logger.info(f"Sklearn pipeline saved to {model_path}")
    
    def _save_pytorch_model(
        self,
        model: Any,
        output_path: Path,
        preprocessor: Optional[Any]
    ) -> None:
        """Save PyTorch model as TorchScript with separate preprocessor.
        
        Args:
            model: PyTorch model (nn.Module or NeuralNetworkBuilder)
            output_path: Directory to save model
            preprocessor: Fitted preprocessor for feature transformation
        """
        import torch
        import joblib
        
        # Handle NeuralNetworkBuilder wrapper
        if hasattr(model, '_model') and hasattr(model, 'architecture'):
            # This is a NeuralNetworkBuilder, extract the actual model
            pytorch_model = model._model
            architecture = model.architecture
        elif hasattr(model, 'state_dict'):
            # This is already a PyTorch nn.Module
            pytorch_model = model
            architecture = None
        else:
            raise ValueError(f"Unknown model type: {type(model)}")
        
        # Save model as TorchScript
        pytorch_model.eval()
        
        # Try to convert to TorchScript
        try:
            # Create example input for tracing
            if architecture:
                example_input = torch.randn(1, architecture.input_dim)
            else:
                # Try to infer input size from first layer
                first_layer = None
                for module in pytorch_model.modules():
                    if isinstance(module, torch.nn.Linear):
                        first_layer = module
                        break
                if first_layer:
                    example_input = torch.randn(1, first_layer.in_features)
                else:
                    raise ValueError("Cannot determine input dimension for TorchScript conversion")
            
            # Trace the model
            traced_model = torch.jit.trace(pytorch_model, example_input)
            
            # Save traced model
            model_path = output_path / "model.pt"
            traced_model.save(str(model_path))
            
            self._manifest["model_filename"] = "model.pt"
            self.logger.info(f"PyTorch model saved as TorchScript to {model_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not convert to TorchScript: {e}. Saving state dict instead.")
            # Fallback: save state dict
            model_path = output_path / "model_state.pt"
            torch.save(pytorch_model.state_dict(), model_path)
            self._manifest["model_filename"] = "model_state.pt"
            self.logger.info(f"PyTorch state dict saved to {model_path}")
        
        # Save preprocessor separately if provided
        if preprocessor is not None:
            preprocessor_path = output_path / "preprocessor.joblib"
            joblib.dump(preprocessor, preprocessor_path)
            self._manifest["preprocessor_filename"] = "preprocessor.joblib"
            self.logger.info(f"Preprocessor saved to {preprocessor_path}")
        else:
            self._manifest["preprocessor_filename"] = None
    
    def _get_sklearn_version(self) -> Optional[str]:
        """Get sklearn version."""
        try:
            import sklearn
            return sklearn.__version__
        except ImportError:
            return None
    
    def _get_pytorch_version(self) -> Optional[str]:
        """Get PyTorch version."""
        try:
            import torch
            return torch.__version__
        except ImportError:
            return None
    
    def _compute_preprocessor_checksum(self, preprocessor: Optional[Any]) -> Optional[str]:
        """Compute checksum of preprocessor for validation."""
        if preprocessor is None:
            return None
        
        try:
            import joblib
            import io
            
            # Serialize preprocessor to bytes
            buffer = io.BytesIO()
            joblib.dump(preprocessor, buffer)
            buffer.seek(0)
            
            # Compute MD5 hash
            md5_hash = hashlib.md5(buffer.read()).hexdigest()
            return md5_hash
        except Exception as e:
            self.logger.warning(f"Could not compute preprocessor checksum: {e}")
            return None


def save_training_result(
    result: Any,
    output_path: Union[str, Path],
    data_manager: Any,
    model_name: Optional[str] = None,
    save_format: str = "bundle"  # "bundle" or "legacy"
) -> Path:
    """Convenience function to save a training result.
    
    Args:
        result: TrainingResult or OptimizationResult
        output_path: Directory to save model
        data_manager: DataManager with preprocessor info
        model_name: Optional custom model name
        save_format: "bundle" for manifest.json format, "legacy" for joblib/pickle
        
    Returns:
        Path to saved model bundle
    """
    import joblib
    from pathlib import Path
    
    # Extract model from result (handle both TrainingResult and OptimizationResult)
    if hasattr(result, 'model') and result.model is not None:
        model = result.model
        is_nn = getattr(result, 'is_neural_network', False)
        algorithm = getattr(result, 'model_name', 'model')
    elif hasattr(result, 'best_model') and result.best_model is not None:
        model = result.best_model
        is_nn = False  # Optimized models are not NN
        algorithm = model_name or 'optimized_model'
    else:
        raise ValueError("Result object has no model or best_model attribute")
    
    metrics = getattr(result, 'metrics', {})
    
    # Legacy format (joblib/pickle only)
    if save_format == "legacy":
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if is_nn:
            # Save NN as PyTorch model
            model_path = output_path / f"{algorithm}.pt"
            model.save_model(str(model_path))
        else:
            # Save sklearn model as joblib
            model_path = output_path / f"{algorithm}.joblib"
            joblib.dump(model, model_path)
        
        return model_path
    
    # Bundle format (manifest.json + model files)
    saver = ModelSaver()
    
    model_type = "pytorch" if is_nn else "sklearn"
    
    # Get feature names and target from data manager
    feature_names = data_manager.get_feature_names()
    target_column = data_manager._target_column
    
    # Get preprocessor
    preprocessor = getattr(data_manager, '_preprocessor', None)
    
    # Get NN config if applicable
    nn_config = None
    if is_nn and hasattr(model, 'architecture'):
        arch = model.architecture
        first_layer = arch.hidden_layers[0] if arch.hidden_layers else None
        nn_config = {
            "input_dim": arch.input_dim,
            "hidden_layers": [
                {
                    "n_units": layer.n_units,
                    "activation": layer.activation.value if layer.activation else "relu",
                    "dropout_rate": layer.dropout_rate,
                    "use_batch_norm": layer.use_batch_norm,
                    "use_layer_norm": layer.use_layer_norm
                }
                for layer in arch.hidden_layers
            ],
            "output_dim": arch.output_dim,
            "dropout_rate": first_layer.dropout_rate if first_layer else 0.0,
            "activation": first_layer.activation.value if first_layer else "relu",
            "use_batch_norm": first_layer.use_batch_norm if first_layer else True,
            "use_layer_norm": first_layer.use_layer_norm if first_layer else False
        }
    
    name = model_name or algorithm
    
    return saver.save_model(
        model=model,
        output_path=output_path,
        model_name=name,
        model_type=model_type,
        algorithm=algorithm,
        feature_names=feature_names,
        target_column=target_column,
        preprocessor=preprocessor,
        metrics=metrics,
        is_neural_network=is_nn,
        nn_config=nn_config
    )
