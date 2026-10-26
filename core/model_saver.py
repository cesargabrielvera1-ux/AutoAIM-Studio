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
        nn_config: Optional[Dict] = None,
        # v1.5.0: Confidence interval parameters
        confidence_interval_params: Optional[Dict] = None,
        # v1.5.0: Model card information
        model_card_info: Optional[Dict[str, Any]] = None,
        # v2.0: Cross-validation configuration used during training (for
        # provenance in the Predict tab and session exports).
        cv_config: Optional[Dict[str, Any]] = None,
        # v2.0: Out-of-distribution detector for applicability-domain warnings
        ood_detector: Optional[Any] = None,
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
        
        # v2.0: Detect linked-structure features. When linking is used
        # (crystal_* or mol_*), composition is auto-derived from the
        # structure files — no separate "composition" prefix needed.
        linked_prefixes = []
        if feature_names:
            if any(f.startswith('crystal_') for f in feature_names):
                linked_prefixes.append('crystal_')
            if any(f.startswith('mol_') for f in feature_names):
                linked_prefixes.append('mol_')
            # has_* features are functional-group flags generated alongside
            # mol_* by MoleculeFeaturizer — ensure mol_ prefix is present
            if any(f.startswith('has_') for f in feature_names):
                if 'mol_' not in linked_prefixes:
                    linked_prefixes.append('mol_')

        # v2.0: Gaussian Process metadata — kernel and alpha are invaluable
        # for reproducibility and model provenance.
        # When model is a Pipeline, drill into the last step (the actual estimator).
        gp_metadata = None
        _gp_model = model
        if hasattr(model, 'named_steps') and model.named_steps:
            _gp_model = list(model.named_steps.values())[-1]
        if hasattr(_gp_model, 'kernel_') or 'GaussianProcess' in type(_gp_model).__name__:
            try:
                gp_metadata = {
                    'kernel_str': str(getattr(_gp_model, 'kernel_', getattr(_gp_model, 'kernel', 'unknown'))),
                    'alpha': float(getattr(_gp_model, 'alpha', 1e-10)),
                    'log_marginal_likelihood_value': float(_gp_model.log_marginal_likelihood_value_) if hasattr(_gp_model, 'log_marginal_likelihood_value_') else None,
                }
            except Exception:
                pass

        # Create manifest
        self._manifest = {
            "model_type": model_type,
            "algorithm": algorithm,
            "model_name": model_name,
            "creation_date": datetime.now().isoformat(),
            "autoaim_version": "2.0.0",
            "sklearn_version": sklearn_version,
            "pytorch_version": pytorch_version if model_type == "pytorch" else None,
            "feature_names": feature_names,
            "target_column": target_column,
            "preprocessing_checksum": preprocessing_checksum,
            "metrics": metrics or {},
            "is_neural_network": is_neural_network,
            "nn_config": nn_config or {},
            "confidence_interval_params": confidence_interval_params or {},
            "linked_feature_prefixes": linked_prefixes,
            "requires_structure_linking": len(linked_prefixes) > 0,
            # v2.0: CV provenance (how the model was validated during training)
            "cv_config": cv_config or {},
            # v2.0: Gaussian Process native uncertainty metadata
            "gp_metadata": gp_metadata,
        }

        # Save model based on type
        if model_type == "pytorch" or is_neural_network:
            self._save_pytorch_model(model, output_path, preprocessor)
        else:
            self._save_sklearn_model(model, output_path, preprocessor)

        # v2.0: Save out-of-distribution detector and merge its metadata
        if ood_detector is not None:
            try:
                ood_path = output_path / "ood_detector.joblib"
                ood_detector.save(str(ood_path))
                self._manifest.update(ood_detector.get_metadata())
            except Exception as e:
                self.logger.warning(f"Could not save OOD detector: {e}")

        # Save manifest
        manifest_path = output_path / "manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(self._manifest, f, indent=2, default=str)
        
        # v1.5.0: Generate and save Model Card
        model_card = self._generate_model_card(model_card_info)
        if model_card:
            card_path = output_path / "MODEL_CARD.md"
            card_path.write_text(model_card, encoding='utf-8')
            self.logger.info(f"Model Card saved to {card_path}")
        
        self.logger.info(f"Model saved to {output_path}")
        return output_path
    
    def _generate_model_card(self, info: Optional[Dict[str, Any]]) -> Optional[str]:
        """Generate a Model Card in Markdown format.
        
        Model Cards are a transparency tool for ML models, documenting
        intended use, performance, limitations, and ethical considerations.
        
        v1.5.0: Auto-generated from model metadata and training info.
        """
        if info is None:
            info = {}
        
        m = self._manifest
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        lines = []
        lines.append(f"# Model Card: {m.get('model_name', 'Unknown')}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # Model Details
        lines.append("## Model Details")
        lines.append("")
        lines.append(f"- **Model Name:** {m.get('model_name', 'N/A')}")
        lines.append(f"- **Algorithm:** {m.get('algorithm', 'N/A')}")
        lines.append(f"- **Model Type:** {m.get('model_type', 'N/A')}")
        lines.append(f"- **Version:** {m.get('autoaim_version', '2.0.0')}")
        lines.append(f"- **Creation Date:** {m.get('creation_date', now)}")
        lines.append(f"- **Library:** {m.get('sklearn_version', 'N/A')}")
        lines.append("")
        
        # Intended Use
        lines.append("## Intended Use")
        lines.append("")
        purpose = info.get('intended_use', 
            "This model was trained for materials science property prediction. "
            "It should be used to predict material properties based on compositional "
            "and/or structural features."
        )
        lines.append(purpose)
        lines.append("")
        
        # Factors
        lines.append("## Factors")
        lines.append("")
        lines.append("### Dataset Characteristics")
        n_samples = info.get('n_samples', 'N/A')
        n_features = len(m.get('feature_names', []))
        target = m.get('target_column', 'N/A')
        lines.append(f"- **Training Samples:** {n_samples}")
        lines.append(f"- **Input Features:** {n_features}")
        lines.append(f"- **Target Column:** {target}")
        lines.append(f"- **Problem Type:** {info.get('problem_type', 'regression')}")
        if info.get('preprocessing'):
            lines.append(f"- **Preprocessing:** {info['preprocessing']}")
        lines.append("")
        
        # v2.0: Gaussian Process — native uncertainty quantification section
        gp = m.get("gp_metadata")
        if gp:
            lines.append("### Uncertainty Quantification (Gaussian Process)")
            lines.append("")
            lines.append(f"- **Kernel:** `{gp.get('kernel_str', 'N/A')}`")
            lines.append(f"- **Alpha (noise level):** {gp.get('alpha', 'N/A')}")
            lml = gp.get('log_marginal_likelihood_value')
            if lml is not None:
                lines.append(f"- **Log Marginal Likelihood:** {lml:.4f}")
            lines.append("- **Uncertainty type:** Native posterior standard deviation (±1σ)")
            lines.append("- **Note:** Prediction standard deviation is available via the Predict tab")
            lines.append("")

        # Metrics
        lines.append("## Metrics")
        lines.append("")
        metrics = m.get('metrics', {})
        if metrics:
            lines.append("### Test Set Performance")
            lines.append("")
            for k, v in metrics.items():
                if isinstance(v, float):
                    lines.append(f"- **{k}:** {v:.6f}")
                else:
                    lines.append(f"- **{k}:** {v}")
            lines.append("")
        
        # Confidence Intervals
        ci = m.get('confidence_interval_params', {})
        if ci and ci.get('type'):
            lines.append("### Prediction Confidence Intervals")
            lines.append("")
            if ci.get('type') == 'regression':
                lines.append(f"- **Method:** CV residual quantiles ({ci.get('n_samples', 'N/A')} samples)")
                lines.append(f"- **68% Interval (±):** {ci.get('quantile_68', 'N/A')}")
                lines.append(f"- **95% Interval (±):** {ci.get('quantile_95', 'N/A')}")
                lines.append(f"- **Residual Std:** {ci.get('residuals_std', 'N/A')}")
            lines.append("")
        
        # Feature Names
        feature_names = m.get('feature_names', [])
        if feature_names:
            lines.append("## Input Features")
            lines.append("")
            lines.append("The model expects the following input features (in order):")
            lines.append("")
            for i, fn in enumerate(feature_names, 1):
                lines.append(f"{i}. `{fn}`")
            lines.append("")
        
        # Limitations
        lines.append("## Limitations")
        lines.append("")
        # v2.0: Peer-review compliant disclaimer + OOD description
        _ood_txt = ""
        if m.get('ood_enabled'):
            _ood_txt = (
                "- **Applicability Domain (AD):** An out-of-distribution detector is "
                f"enabled (method={m.get('ood_method', 'N/A')}, "
                f"threshold_percentile={m.get('ood_threshold_percentile', 'N/A')}). "
                "Predictions for samples flagged as OOD are extrapolations and must be "
                "interpreted with extreme caution.\n"
            )
        limitations = info.get('limitations',
            "**These predictions are model estimates for screening and prioritisation, "
            "NOT experimentally validated properties.**\n\n"
            "- Predictions are only reliable for materials similar to those in the training set.\n"
            "- Extrapolation beyond the training domain may produce unreliable results.\n"
            + _ood_txt +
            "- The model does not account for experimental conditions (temperature, pressure) "
            "unless included as features.\n"
            "- Missing values in input features must be handled before prediction."
        )
        lines.append(limitations)
        lines.append("")
        
        # Reproducibility
        lines.append("## Reproducibility")
        lines.append("")
        lines.append("To reproduce this model:\n")
        lines.append(f"1. Use AutoAIM Studio v{m.get('autoaim_version', '2.0.0')}")
        lines.append(f"2. Load the same training data ({info.get('data_file', 'see session report')})")
        if info.get('random_state'):
            lines.append(f"3. Set random_state={info['random_state']}")
        if info.get('model_params'):
            lines.append("4. Use these model parameters:")
            for k, v in info['model_params'].items():
                lines.append(f"   - `{k}`: `{v}`")
        lines.append("")
        
        # Citation
        lines.append("## Citation")
        lines.append("")
        lines.append(
            "If you use this model in your research, please cite the AutoAIM Studio "
            "session report that generated it, and include the model name and creation date."
        )
        lines.append("")
        lines.append("---")
        lines.append("")
        lines.append(f"*Model Card generated by AutoAIM Studio v{m.get('autoaim_version', '2.0.0')} on {now}*")
        
        return "\n".join(lines)
    
    def _save_sklearn_model(
        self,
        model: Any,
        output_path: Path,
        preprocessor: Optional[Any]
    ) -> None:
        """Save sklearn model with optional preprocessor.

        v2.0 FIX: Only wraps the model in a Pipeline if the preprocessor is
        actually fitted. In v2.0, data_manager._preprocessor is typically
        UNFITTED (fitting happens inside each CV fold via Pipeline). Saving an
        unfitted preprocessor alongside a fitted model would break inference.
        """
        import joblib
        from sklearn.pipeline import Pipeline

        # v2.0: Check if preprocessor is fitted before using it.
        preprocessor_fitted = False
        if preprocessor is not None:
            try:
                # sklearn transformers have a check_is_fitted utility, but we
                # can also detect fitted state by looking for fitted attributes.
                if hasattr(preprocessor, 'transformers_'):
                    preprocessor_fitted = True
                elif hasattr(preprocessor, 'mean_') or hasattr(preprocessor, 'scale_'):
                    preprocessor_fitted = True
                elif hasattr(preprocessor, 'feature_names_in_'):
                    preprocessor_fitted = True
            except Exception:
                pass

        # If preprocessor exists AND is fitted, create a unified pipeline
        if preprocessor is not None and preprocessor_fitted:
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
        
        # v1.3.0 FIX: Move model to CPU before TorchScript conversion.
        # If model is on GPU (CUDA) and example_input is on CPU (default),
        # torch.jit.trace fails with "Expected all tensors to be on the same device".
        original_device = next(pytorch_model.parameters()).device
        pytorch_model = pytorch_model.cpu()
        
        # Try to convert to TorchScript
        try:
            # Create example input for tracing (both model and input now on CPU)
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
            self._manifest["model_format"] = "torchscript"
            self.logger.info(f"PyTorch model saved as TorchScript to {model_path}")
            
        except Exception as e:
            self.logger.warning(f"Could not convert to TorchScript: {e}. Saving full model instead.")
            # v1.3.0 FIX: Save the full nn.Module on CPU instead of just state_dict.
            # torch.load() can restore it as a proper model for inference.
            model_path = output_path / "model.pt"
            torch.save(pytorch_model.cpu(), model_path)
            self._manifest["model_filename"] = "model.pt"
            self._manifest["model_format"] = "full_model"
            self.logger.info(f"PyTorch full model saved to {model_path}")
        
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
    # v2.0: Support disk-cached models — load from cache if model was offloaded.
    model = None
    is_nn = False
    if hasattr(result, 'model') and result.model is not None:
        model = result.model
        is_nn = getattr(result, 'is_neural_network', False)
        algorithm = getattr(result, 'model_name', 'model')
    elif hasattr(result, 'is_cached') and result.is_cached and result.model_cache_path:
        from .model_cache import get_model_cache
        cache = get_model_cache()
        model = cache.load(result.model_cache_path, is_neural_network=result.is_neural_network)
        is_nn = getattr(result, 'is_neural_network', False)
        algorithm = getattr(result, 'model_name', 'model')
    elif hasattr(result, 'best_model') and result.best_model is not None:
        model = result.best_model
        is_nn = False  # Optimized models are not NN
        algorithm = model_name or 'optimized_model'

    if model is None:
        raise ValueError(
            "Result object has no model available (neither in RAM nor in disk cache). "
            "The model may have been pruned or the cache file deleted."
        )
    
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
    
    # v1.5.0: Get confidence interval params from result
    ci_params = getattr(result, 'confidence_interval_params', None)
    
    # v1.5.0: Build Model Card info
    dm = data_manager
    card_info = {
        'n_samples': len(dm.data) if dm.data is not None else 'N/A',
        'problem_type': getattr(result, 'problem_type', 'regression'),
        'preprocessing': str(getattr(dm, '_preprocessor', None)),
        'random_state': (ci_params or {}).get('n_samples'),
        'model_params': getattr(result, 'model_params', {}),
    }
    # Extract random_state from model if available
    try:
        if hasattr(model, 'get_params'):
            params = model.get_params()
            if 'random_state' in params:
                card_info['random_state'] = params['random_state']
            elif 'random_seed' in params:
                card_info['random_state'] = params['random_seed']
    except Exception:
        pass
    
    # v2.0: CV provenance — carry the CV configuration used during training
    # (or optimization) into the bundle manifest for the Predict tab.
    cv_config = getattr(result, 'cv_config_dict', None)
    if cv_config is None:
        # OptimizationResult: build a minimal config from its stored fields.
        strat = getattr(result, 'cv_strategy', None)
        if strat:
            cv_config = {'strategy': strat}

    # v2.0: Out-of-distribution detector trained on X_train
    ood_detector = getattr(result, 'ood_detector', None)

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
        nn_config=nn_config,
        confidence_interval_params=ci_params,
        model_card_info=card_info,
        cv_config=cv_config,
        ood_detector=ood_detector,
    )
