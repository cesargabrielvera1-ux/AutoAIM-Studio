"""Neural Network builder with visual architecture configuration."""

from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..utils.logger import LoggerMixin
from ..utils.hardware_detector import get_hardware_detector


class ActivationFunction(Enum):
    """Available activation functions."""
    RELU = 'relu'
    LEAKY_RELU = 'leaky_relu'
    TANH = 'tanh'
    SIGMOID = 'sigmoid'
    GELU = 'gelu'
    ELU = 'elu'
    SWISH = 'swish'


class OptimizerType(Enum):
    """Available optimizers."""
    ADAM = 'adam'
    ADAMW = 'adamw'
    SGD = 'sgd'
    RMSPROP = 'rmsprop'
    ADAGRAD = 'adagrad'


class SchedulerType(Enum):
    """Available learning rate schedulers."""
    REDUCE_ON_PLATEAU = 'reduce_on_plateau'
    COSINE_ANNEALING = 'cosine_annealing'
    STEPLR = 'step_lr'
    EXPONENTIAL = 'exponential'
    NONE = 'none'


@dataclass
class LayerConfig:
    """Configuration for a neural network layer."""
    n_units: int
    activation: ActivationFunction = ActivationFunction.RELU
    dropout_rate: float = 0.0
    use_batch_norm: bool = True
    use_layer_norm: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not 16 <= self.n_units <= 2048:
            raise ValueError(f"n_units must be between 16 and 2048, got {self.n_units}")
        if not 0.0 <= self.dropout_rate <= 0.5:
            raise ValueError(f"dropout_rate must be between 0.0 and 0.5, got {self.dropout_rate}")


@dataclass
class NNArchitecture:
    """Complete neural network architecture configuration."""
    input_dim: int
    output_dim: int
    hidden_layers: List[LayerConfig] = field(default_factory=list)
    output_activation: Optional[ActivationFunction] = None
    problem_type: str = 'regression'  # 'regression' or 'classification'
    
    def __post_init__(self):
        """Validate architecture."""
        if self.input_dim <= 0:
            raise ValueError(f"input_dim must be positive, got {self.input_dim}")
        if self.output_dim <= 0:
            raise ValueError(f"output_dim must be positive, got {self.output_dim}")
        if len(self.hidden_layers) > 5:
            raise ValueError(f"Maximum 5 hidden layers allowed, got {len(self.hidden_layers)}")


@dataclass
class TrainingConfig:
    """Configuration for neural network training."""
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 0.001
    optimizer: OptimizerType = OptimizerType.ADAM
    scheduler: SchedulerType = SchedulerType.REDUCE_ON_PLATEAU
    weight_decay: float = 1e-5
    l1_reg: float = 0.0
    l2_reg: float = 0.0
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 1e-4
    gradient_clip_val: float = 1.0
    random_seed: int = 42
    use_mixed_precision: bool = False
    
    def __post_init__(self):
        """Validate configuration."""
        if not 10 <= self.epochs <= 10000:
            raise ValueError(f"epochs must be between 10 and 10000, got {self.epochs}")
        if not 8 <= self.batch_size <= 512:
            raise ValueError(f"batch_size must be between 8 and 512, got {self.batch_size}")
        if not 1e-5 <= self.learning_rate <= 1e-1:
            raise ValueError(f"learning_rate must be between 1e-5 and 1e-1, got {self.learning_rate}")


class DynamicNN(nn.Module, LoggerMixin):
    """Dynamic neural network with configurable architecture."""
    
    def __init__(self, architecture: NNArchitecture):
        """Initialize neural network.
        
        Args:
            architecture: Network architecture configuration
        """
        # FIX: Inicializar nn.Module explicitamente antes que nada
        nn.Module.__init__(self)
        LoggerMixin.__init__(self)
        
        self.architecture = architecture
        
        # FIX: Inicializar ModuleLists ANTES de agregar capas
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.output_layer = None
        
        # FIX: Validar arquitectura antes de construir
        if not self.architecture.hidden_layers:
            raise ValueError(
                "La arquitectura no tiene capas ocultas. "
                "Agrega al menos una capa oculta."
            )
        
        self._build_network()
        
        # Initialize weights
        self._initialize_weights()
        
        # FIX: Verificar parametros despues de toda la inicializacion
        self._validate_parameters()
    
    def _build_network(self) -> None:
        """Build the neural network layers."""
        prev_dim = self.architecture.input_dim
        
        # FIX: Asegurar que ModuleLists esten vacios antes de construir
        # Usar del en lugar de clear() para compatibilidad con PyInstaller/Windows
        if len(self.layers) > 0:
            del self.layers[:]
        if len(self.batch_norms) > 0:
            del self.batch_norms[:]
        if len(self.layer_norms) > 0:
            del self.layer_norms[:]
        if len(self.dropouts) > 0:
            del self.dropouts[:]
        
        # Hidden layers
        for i, layer_config in enumerate(self.architecture.hidden_layers):
            # Linear layer
            linear = nn.Linear(prev_dim, layer_config.n_units)
            self.layers.append(linear)
            
            # Batch normalization
            if layer_config.use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(layer_config.n_units))
            else:
                self.batch_norms.append(None)
            
            # Layer normalization
            if layer_config.use_layer_norm:
                self.layer_norms.append(nn.LayerNorm(layer_config.n_units))
            else:
                self.layer_norms.append(None)
            
            # Dropout
            if layer_config.dropout_rate > 0:
                self.dropouts.append(nn.Dropout(layer_config.dropout_rate))
            else:
                self.dropouts.append(None)
            
            prev_dim = layer_config.n_units
        
        # Output layer
        self.output_layer = nn.Linear(prev_dim, self.architecture.output_dim)
        
        self.logger.info(
            f"Built neural network: {self.architecture.input_dim} -> "
            f"{' -> '.join(str(l.n_units) for l in self.architecture.hidden_layers)} -> "
            f"{self.architecture.output_dim}"
        )
    
    def _validate_parameters(self) -> None:
        """Validate that the model has trainable parameters."""
        # FIX: Verificar que el modelo tiene parametros entrenables
        model_params = list(self.parameters())
        if len(model_params) == 0:
            raise ValueError(
                "El modelo no tiene parametros entrenables. "
                "Verifica que las capas se hayan agregado correctamente."
            )
        
        # FIX: Verificar que output_layer tiene parametros
        if self.output_layer is None:
            raise ValueError("La capa de salida no fue inicializada.")
        
        total_params = sum(p.numel() for p in model_params)
        self.logger.info(f"Model has {total_params:,} parameters")
    
    def _initialize_weights(self) -> None:
        """Initialize network weights."""
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
        # Output layer
        nn.init.xavier_normal_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0)
    
    def _get_activation(self, activation: ActivationFunction) -> nn.Module:
        """Get activation function module.
        
        Args:
            activation: Activation function type
            
        Returns:
            Activation module
        """
        activations = {
            ActivationFunction.RELU: nn.ReLU(),
            ActivationFunction.LEAKY_RELU: nn.LeakyReLU(0.1),
            ActivationFunction.TANH: nn.Tanh(),
            ActivationFunction.SIGMOID: nn.Sigmoid(),
            ActivationFunction.GELU: nn.GELU(),
            ActivationFunction.ELU: nn.ELU(),
            ActivationFunction.SWISH: nn.SiLU(),  # Swish is SiLU in PyTorch
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor
            
        Returns:
            Output tensor
        """
        # Hidden layers
        for i, layer_config in enumerate(self.architecture.hidden_layers):
            x = self.layers[i](x)
            
            # Batch normalization
            if self.batch_norms[i] is not None:
                x = self.batch_norms[i](x)
            
            # Layer normalization
            if self.layer_norms[i] is not None:
                x = self.layer_norms[i](x)
            
            # Activation
            x = self._get_activation(layer_config.activation)(x)
            
            # Dropout
            if self.dropouts[i] is not None:
                x = self.dropouts[i](x)
        
        # Output layer
        x = self.output_layer(x)
        
        # Output activation if specified
        if self.architecture.output_activation:
            x = self._get_activation(self.architecture.output_activation)(x)
        
        return x
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions (sklearn-compatible interface).
        
        Args:
            X: Input features as numpy array
            
        Returns:
            Predictions as numpy array
        """
        import torch
        
        self.eval()  # Set to evaluation mode
        
        # Convert to tensor
        if not isinstance(X, torch.Tensor):
            X_tensor = torch.FloatTensor(X)
        else:
            X_tensor = X
        
        # Move to same device as model
        device = next(self.parameters()).device
        X_tensor = X_tensor.to(device)
        
        with torch.no_grad():
            predictions = self.forward(X_tensor)
        
        # Convert back to numpy
        predictions_np = predictions.cpu().numpy()
        
        # Flatten if single output
        if predictions_np.shape[1] == 1:
            predictions_np = predictions_np.flatten()
        
        return predictions_np
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get model architecture summary.
        
        Returns:
            Dictionary with model information
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        layer_info = []
        for i, layer_config in enumerate(self.architecture.hidden_layers):
            layer_info.append({
                'layer': i + 1,
                'units': layer_config.n_units,
                'activation': layer_config.activation.value,
                'dropout': layer_config.dropout_rate,
                'batch_norm': layer_config.use_batch_norm,
            })
        
        return {
            'input_dim': self.architecture.input_dim,
            'output_dim': self.architecture.output_dim,
            'n_hidden_layers': len(self.architecture.hidden_layers),
            'hidden_layers': layer_info,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'problem_type': self.architecture.problem_type,
        }


class NeuralNetworkBuilder(LoggerMixin):
    """Builder for creating and training neural networks."""
    
    def __init__(self):
        """Initialize neural network builder."""
        self._hardware = get_hardware_detector()
        self._device = self._hardware.get_torch_device()
        self._model: Optional[DynamicNN] = None
        self._architecture: Optional[NNArchitecture] = None
        self._training_config: Optional[TrainingConfig] = None
        self._history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
    
    @property
    def device(self) -> torch.device:
        """Get computation device."""
        return self._device
    
    @property
    def model(self) -> Optional[DynamicNN]:
        """Get current model."""
        return self._model
    
    @property
    def architecture(self) -> Optional[NNArchitecture]:
        """Get current architecture."""
        return self._architecture
    
    @property
    def history(self) -> Dict[str, List[float]]:
        """Get training history."""
        return self._history
    
    def create_architecture(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers_config: List[Dict[str, Any]],
        problem_type: str = 'regression',
        output_activation: Optional[str] = None
    ) -> NNArchitecture:
        """Create neural network architecture from configuration.
        
        Args:
            input_dim: Number of input features
            output_dim: Number of output units
            hidden_layers_config: List of layer configurations
            problem_type: 'regression' or 'classification'
            output_activation: Output activation function name
            
        Returns:
            Architecture configuration
        """
        hidden_layers = []
        for config in hidden_layers_config:
            layer = LayerConfig(
                n_units=config['n_units'],
                activation=ActivationFunction(config.get('activation', 'relu')),
                dropout_rate=config.get('dropout_rate', 0.0),
                use_batch_norm=config.get('use_batch_norm', True),
                use_layer_norm=config.get('use_layer_norm', False)
            )
            hidden_layers.append(layer)
        
        output_act = None
        if output_activation:
            output_act = ActivationFunction(output_activation)
        
        self._architecture = NNArchitecture(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            output_activation=output_act,
            problem_type=problem_type
        )
        
        return self._architecture
    
    def build_model(self) -> DynamicNN:
        """Build the neural network model.
        
        Returns:
            Built model
        """
        if self._architecture is None:
            raise ValueError("Architecture must be created before building model")
        
        self._model = DynamicNN(self._architecture)
        self._model.to(self._device)
        
        return self._model
    
    def create_training_config(
        self,
        epochs: int = 100,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        optimizer: str = 'adam',
        scheduler: str = 'reduce_on_plateau',
        weight_decay: float = 1e-5,
        l1_reg: float = 0.0,
        l2_reg: float = 0.0,
        early_stopping_patience: int = 50,
        early_stopping_min_delta: float = 1e-4,
        gradient_clip_val: float = 1.0,
        random_seed: int = 42
    ) -> TrainingConfig:
        """Create training configuration.
        
        Args:
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            optimizer: Optimizer type
            scheduler: Learning rate scheduler
            weight_decay: Weight decay (L2)
            l1_reg: L1 regularization
            l2_reg: L2 regularization
            early_stopping_patience: Patience for early stopping
            early_stopping_min_delta: Minimum delta for early stopping
            gradient_clip_val: Max gradient norm for clipping (0 to disable)
            random_seed: Random seed for reproducible CV folds
            
        Returns:
            Training configuration
        """
        self._training_config = TrainingConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            optimizer=OptimizerType(optimizer),
            scheduler=SchedulerType(scheduler),
            weight_decay=weight_decay,
            l1_reg=l1_reg,
            l2_reg=l2_reg,
            early_stopping_patience=early_stopping_patience,
            early_stopping_min_delta=early_stopping_min_delta,
            gradient_clip_val=gradient_clip_val,
            random_seed=random_seed
        )
        
        return self._training_config
    
    def _get_optimizer(self, model_parameters) -> optim.Optimizer:
        """Create optimizer.
        
        Args:
            model_parameters: Model parameters
            
        Returns:
            Optimizer
        """
        config = self._training_config
        
        optimizers = {
            OptimizerType.ADAM: optim.Adam(
                model_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            ),
            OptimizerType.ADAMW: optim.AdamW(
                model_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            ),
            OptimizerType.SGD: optim.SGD(
                model_parameters,
                lr=config.learning_rate,
                momentum=0.9,
                weight_decay=config.weight_decay
            ),
            OptimizerType.RMSPROP: optim.RMSprop(
                model_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            ),
            OptimizerType.ADAGRAD: optim.Adagrad(
                model_parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay
            ),
        }
        
        return optimizers.get(config.optimizer, optim.Adam(model_parameters, lr=config.learning_rate))
    
    def _get_scheduler(self, optimizer: optim.Optimizer) -> Optional[Any]:
        """Create learning rate scheduler.
        
        Args:
            optimizer: Optimizer
            
        Returns:
            Scheduler or None
        """
        config = self._training_config
        
        schedulers = {
            SchedulerType.REDUCE_ON_PLATEAU: optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10
                # FIX: Eliminado 'verbose' - no soportado en PyTorch >= 2.0
            ),
            SchedulerType.COSINE_ANNEALING: optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.epochs,
                eta_min=1e-6
            ),
            SchedulerType.STEPLR: optim.lr_scheduler.StepLR(
                optimizer,
                step_size=30,
                gamma=0.1
            ),
            SchedulerType.EXPONENTIAL: optim.lr_scheduler.ExponentialLR(
                optimizer,
                gamma=0.95
            ),
        }
        
        if config.scheduler == SchedulerType.NONE:
            return None
        
        return schedulers.get(config.scheduler)
    
    def _compute_loss(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Compute loss with regularization.
        
        Args:
            predictions: Model predictions
            targets: True targets
            
        Returns:
            Loss value
        """
        config = self._training_config
        
        # Base loss
        if self._architecture.problem_type == 'regression':
            loss = nn.MSELoss()(predictions, targets)
        else:
            if self._architecture.output_dim == 1:
                loss = nn.BCEWithLogitsLoss()(predictions, targets)
            else:
                loss = nn.CrossEntropyLoss()(predictions, targets)
        
        # L1 regularization
        if config.l1_reg > 0:
            l1_reg = 0
            for param in self._model.parameters():
                l1_reg += torch.norm(param, 1)
            loss += config.l1_reg * l1_reg
        
        # L2 regularization (already handled by weight_decay in optimizer)
        if config.l2_reg > 0:
            l2_reg = 0
            for param in self._model.parameters():
                l2_reg += torch.norm(param, 2)
            loss += config.l2_reg * l2_reg
        
        return loss
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        verbose: bool = True,
        progress_callback: Optional[Callable[[int, float, float], None]] = None
    ) -> Dict[str, List[float]]:
        """Train the neural network.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            verbose: Whether to print progress
            progress_callback: Callback function(epoch, train_loss, val_loss)
            
        Returns:
            Training history
        """
        if self._model is None:
            raise ValueError("Model must be built before training")
        if self._training_config is None:
            raise ValueError("Training config must be set before training")
        
        config = self._training_config
        
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train).to(self._device)
        y_train_tensor = torch.FloatTensor(y_train).to(self._device)
        
        if y_train_tensor.dim() == 1:
            y_train_tensor = y_train_tensor.unsqueeze(1)
        
        # Create data loader
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0
        )
        
        # Validation data
        if X_val is not None and y_val is not None:
            X_val_tensor = torch.FloatTensor(X_val).to(self._device)
            y_val_tensor = torch.FloatTensor(y_val).to(self._device)
            if y_val_tensor.dim() == 1:
                y_val_tensor = y_val_tensor.unsqueeze(1)
        else:
            X_val_tensor = None
            y_val_tensor = None
        
        # Optimizer and scheduler
        # FIX: Verificacion exhaustiva de parametros antes de crear el optimizador
        if self._model is None:
            raise ValueError("El modelo no ha sido construido. Llama a build_model() primero.")
        
        model_params = list(self._model.parameters())
        if len(model_params) == 0:
            raise ValueError(
                "El modelo no tiene parametros entrenables. "
                "Esto puede ocurrir si:\n"
                "1. No se agregaron capas ocultas\n"
                "2. El modelo no se inicializo correctamente\n"
                "3. Hay un problema con la herencia de nn.Module"
            )
        
        # FIX: Verificar que los parametros requieren gradiente
        trainable_params = [p for p in model_params if p.requires_grad]
        if len(trainable_params) == 0:
            raise ValueError("Ningun parametro del modelo requiere gradiente.")
        
        optimizer = self._get_optimizer(trainable_params)
        scheduler = self._get_scheduler(optimizer)
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        self._history = {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []}
        
        for epoch in range(config.epochs):
            # Training
            self._model.train()
            train_losses = []
            
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                
                predictions = self._model(batch_X)
                loss = self._compute_loss(predictions, batch_y)
                
                loss.backward()
                
                # Gradient clipping
                if config.gradient_clip_val > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self._model.parameters(),
                        config.gradient_clip_val
                    )
                
                optimizer.step()
                train_losses.append(loss.item())
            
            avg_train_loss = np.mean(train_losses)
            self._history['train_loss'].append(avg_train_loss)
            
            # Validation
            if X_val_tensor is not None:
                self._model.eval()
                with torch.no_grad():
                    val_predictions = self._model(X_val_tensor)
                    val_loss = self._compute_loss(val_predictions, y_val_tensor).item()
                self._history['val_loss'].append(val_loss)
            else:
                val_loss = None
            
            # Update scheduler
            if scheduler is not None:
                if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_loss if val_loss is not None else avg_train_loss)
                else:
                    scheduler.step()
            
            # Early stopping
            if val_loss is not None:
                if val_loss < best_val_loss - config.early_stopping_min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= config.early_stopping_patience:
                    if verbose:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                    break
            
            # Progress callback
            if progress_callback:
                progress_callback(epoch + 1, avg_train_loss, val_loss if val_loss else 0)
            
            if verbose and (epoch + 1) % 10 == 0:
                msg = f"Epoch {epoch + 1}/{config.epochs} - Train Loss: {avg_train_loss:.6f}"
                if val_loss is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                self.logger.info(msg)
        
        return self._history
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predictions
        """
        if self._model is None:
            raise ValueError("Model must be trained before prediction")
        
        self._model.eval()
        X_tensor = torch.FloatTensor(X).to(self._device)
        
        with torch.no_grad():
            predictions = self._model(X_tensor)
        
        return predictions.cpu().numpy()
    
    def save_model(self, path: str) -> None:
        """Save model to file.
        
        Args:
            path: Path to save model
        """
        if self._model is None:
            raise ValueError("No model to save")
        
        checkpoint = {
            'model_state_dict': self._model.state_dict(),
            'architecture': {
                'input_dim': self._architecture.input_dim,
                'output_dim': self._architecture.output_dim,
                'hidden_layers': [
                    {
                        'n_units': l.n_units,
                        'activation': l.activation.value,
                        'dropout_rate': l.dropout_rate,
                        'use_batch_norm': l.use_batch_norm,
                    }
                    for l in self._architecture.hidden_layers
                ],
                'output_activation': self._architecture.output_activation.value if self._architecture.output_activation else None,
                'problem_type': self._architecture.problem_type,
            },
            'training_config': {
                'epochs': self._training_config.epochs,
                'batch_size': self._training_config.batch_size,
                'learning_rate': self._training_config.learning_rate,
                'optimizer': self._training_config.optimizer.value,
                'scheduler': self._training_config.scheduler.value,
            },
            'history': self._history,
        }
        
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str) -> None:
        """Load model from file.
        
        Args:
            path: Path to model file
        """
        checkpoint = torch.load(path, map_location=self._device)
        
        # Recreate architecture
        arch_dict = checkpoint['architecture']
        hidden_layers = [
            {
                'n_units': l['n_units'],
                'activation': l['activation'],
                'dropout_rate': l['dropout_rate'],
                'use_batch_norm': l['use_batch_norm'],
            }
            for l in arch_dict['hidden_layers']
        ]
        
        self.create_architecture(
            input_dim=arch_dict['input_dim'],
            output_dim=arch_dict['output_dim'],
            hidden_layers_config=hidden_layers,
            problem_type=arch_dict['problem_type'],
            output_activation=arch_dict['output_activation']
        )
        
        self.build_model()
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._history = checkpoint.get('history', {'train_loss': [], 'val_loss': [], 'train_metric': [], 'val_metric': []})
        
        self.logger.info(f"Model loaded from {path}")
    
    def get_architecture_visualization(self) -> str:
        """Get ASCII visualization of network architecture.
        
        Returns:
            ASCII string representation
        """
        if self._architecture is None:
            return "No architecture defined"
        
        lines = []
        lines.append("=" * 60)
        lines.append("NEURAL NETWORK ARCHITECTURE")
        lines.append("=" * 60)
        lines.append("")
        
        # Input layer
        lines.append(f"  Input Layer: {self._architecture.input_dim} neurons")
        lines.append("       |")
        lines.append("       v")
        
        # Hidden layers
        for i, layer in enumerate(self._architecture.hidden_layers):
            lines.append(f"  Hidden Layer {i+1}: {layer.n_units} neurons")
            lines.append(f"    - Activation: {layer.activation.value}")
            lines.append(f"    - Dropout: {layer.dropout_rate}")
            lines.append(f"    - Batch Norm: {layer.use_batch_norm}")
            lines.append("       |")
            lines.append("       v")
        
        # Output layer
        lines.append(f"  Output Layer: {self._architecture.output_dim} neurons")
        if self._architecture.output_activation:
            lines.append(f"    - Activation: {self._architecture.output_activation.value}")
        
        lines.append("")
        lines.append("=" * 60)
        
        # Model summary
        if self._model:
            summary = self._model.get_model_summary()
            lines.append(f"Total Parameters: {summary['total_parameters']:,}")
            lines.append(f"Trainable Parameters: {summary['trainable_parameters']:,}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
