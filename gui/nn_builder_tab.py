"""Neural Network Builder tab."""

import logging
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QCheckBox, QTextEdit, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QSplitter, QListWidget,
    QGridLayout, QFrame, QScrollArea, QDialog, QDialogButtonBox,
    QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.nn_builder import NeuralNetworkBuilder, ActivationFunction
from ..core.optimizer import HyperparameterOptimizer
from ..core.cv_factory import CVConfig, validate_cv_config, has_blocking_issue
from ..core.grouping import GroupingError
from .cv_config_widget import CVConfigWidget

logger = logging.getLogger(__name__)


def _safe_get_parent_attr(widget, attr_path, default=None):
    """Safely get a nested attribute from the widget's main window.
    
    v2.0 FIX: After addTab(), widget.parent() returns the QTabWidget — NOT
    the MainWindow. We first try the _main_window reference stored during
    __init__, then fall back to walking the entire parent chain until we find
    an object that has the first attribute in attr_path.
    """
    if widget is None:
        return default
    
    parts = attr_path.split('.')
    first_attr = parts[0]
    
    # v2.0: Strategy 1 — use the _main_window reference stored in __init__
    # before addTab() reparented the widget.
    obj = getattr(widget, '_main_window', None)
    if obj is not None and hasattr(obj, first_attr):
        pass  # Fast path: _main_window has the attribute
    else:
        # Strategy 2 — walk the full parent chain looking for an object
        # that has the first attribute. This covers addTab() reparenting.
        try:
            current = widget.parent()
        except (TypeError, AttributeError):
            return default
        
        obj = None
        visited = set()  # Guard against circular references
        while current is not None:
            obj_id = id(current)
            if obj_id in visited:
                break
            visited.add(obj_id)
            if hasattr(current, first_attr):
                obj = current
                break
            try:
                current = current.parent()
            except (TypeError, AttributeError):
                break
        
        if obj is None:
            return default
    
    # Traverse the rest of the attribute path
    try:
        for part in parts:
            if part.endswith('()'):
                obj = getattr(obj, part[:-2])()
            else:
                if not hasattr(obj, part):
                    return default
                obj = getattr(obj, part)
        return obj
    except Exception:
        return default


class NNTrainingThread(QThread):
    """Thread for training neural network."""
    progress = pyqtSignal(int, float, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, nn_builder, X_train, y_train, X_val, y_val, cv_folds=1):
        super().__init__()
        self.nn_builder = nn_builder
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.cv_folds = cv_folds
    
    def run(self):
        try:
            def progress_callback(epoch, train_loss, val_loss):
                self.progress.emit(epoch, train_loss, val_loss)
            
            history = self.nn_builder.fit(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                verbose=False,
                progress_callback=progress_callback
            )
            self.finished.emit(history)
        except Exception as e:
            self.error.emit(str(e))


class NNOptimizationThread(QThread):
    """Thread for optimizing neural network hyperparameters."""
    progress = pyqtSignal(int, float, dict)
    status = pyqtSignal(str)  # v1.3.0: Status messages for long operations
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, optimizer, nn_builder, X_train, y_train, X_val, y_val,
                 input_dim, output_dim, param_ranges, n_trials, cv_folds=1,
                 opt_epochs=200, cv_epochs=100, opt_patience=30, cv_patience=20,
                 random_state=42):
        super().__init__()
        self.optimizer = optimizer
        self.nn_builder = nn_builder
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.param_ranges = param_ranges
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.opt_epochs = opt_epochs
        self.cv_epochs = cv_epochs
        self.opt_patience = opt_patience
        self.cv_patience = cv_patience
        self.random_state = random_state
    
    def run(self):
        try:
            self.status.emit("Starting NN optimization...")
            
            def progress_callback(trial_num, score, params):
                self.progress.emit(trial_num, score, params)
            
            result = self.optimizer.optimize_nn_architecture(
                nn_builder=self.nn_builder,
                X_train=self.X_train,
                y_train=self.y_train,
                X_val=self.X_val,
                y_val=self.y_val,
                input_dim=self.input_dim,
                output_dim=self.output_dim,
                param_ranges=self.param_ranges,
                n_trials=self.n_trials,
                cv_folds=self.cv_folds,
                progress_callback=progress_callback,
                opt_epochs=self.opt_epochs,
                cv_epochs=self.cv_epochs,
                opt_patience=self.opt_patience,
                cv_patience=self.cv_patience,
                random_state=self.random_state
            )
            self.status.emit("Compiling optimization results...")
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class NNParameterRangeDialog(QDialog):
    """Dialog for editing neural network optimization parameter ranges."""
    
    def __init__(self, param_ranges, parent=None):
        super().__init__(parent)
        self.param_ranges = param_ranges
        self.custom_ranges = {}
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Edit Neural Network Optimization Ranges")
        self.setMinimumWidth(550)
        self.setMinimumHeight(500)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Modify the ranges for each hyperparameter. These ranges will be used by Optuna "
            "during neural network architecture optimization."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        scroll_widget = QWidget()
        scroll_layout = QGridLayout(scroll_widget)
        
        scroll_layout.addWidget(QLabel("<b>Parameter</b>"), 0, 0)
        scroll_layout.addWidget(QLabel("<b>Min Value</b>"), 0, 1)
        scroll_layout.addWidget(QLabel("<b>Max Value</b>"), 0, 2)
        scroll_layout.addWidget(QLabel("<b>Type</b>"), 0, 3)
        
        row = 1
        for param_name, (min_val, max_val, param_type) in self.param_ranges.items():
            name_label = QLabel(f"{param_name}")
            scroll_layout.addWidget(name_label, row, 0)
            
            if param_type == 'int':
                min_spin = QSpinBox()
                min_spin.setRange(1, 10000)
                min_spin.setValue(int(min_val))
                max_spin = QSpinBox()
                max_spin.setRange(1, 10000)
                max_spin.setValue(int(max_val))
            elif param_type == 'categorical':
                min_spin = QTextEdit()
                min_spin.setMaximumHeight(60)
                choices = min_val if isinstance(min_val, (list, tuple)) else [min_val]
                min_spin.setPlainText(", ".join(map(str, choices)))
                max_spin = QLabel("N/A")
            elif param_type == 'float_log':
                min_spin = QDoubleSpinBox()
                min_spin.setRange(1e-10, 1.0)
                min_spin.setValue(float(min_val))
                min_spin.setDecimals(8)
                max_spin = QDoubleSpinBox()
                max_spin.setRange(1e-10, 1.0)
                max_spin.setValue(float(max_val))
                max_spin.setDecimals(8)
            elif param_type == 'int_log':
                min_spin = QSpinBox()
                min_spin.setRange(1, 10000)
                min_spin.setValue(int(min_val))
                max_spin = QSpinBox()
                max_spin.setRange(1, 10000)
                max_spin.setValue(int(max_val))
            else:
                min_spin = QDoubleSpinBox()
                min_spin.setValue(float(min_val))
                min_spin.setDecimals(6)
                max_spin = QDoubleSpinBox()
                max_spin.setValue(float(max_val))
                max_spin.setDecimals(6)
            
            scroll_layout.addWidget(min_spin, row, 1)
            scroll_layout.addWidget(max_spin, row, 2)
            scroll_layout.addWidget(QLabel(f"({param_type})"), row, 3)
            
            self.custom_ranges[param_name] = {
                'min_spin': min_spin,
                'max_spin': max_spin,
                'type': param_type,
                'original_min': min_val,
                'original_max': max_val
            }
            
            row += 1
        
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def get_custom_ranges(self):
        """Get the custom ranges set by user."""
        result = {}
        for param_name, widgets in self.custom_ranges.items():
            param_type = widgets['type']
            
            if param_type == 'categorical':
                min_text = widgets['min_spin'].toPlainText()
                choices = [x.strip() for x in min_text.split(',') if x.strip()]
                result[param_name] = (choices, choices, 'categorical')
            else:
                if hasattr(widgets['min_spin'], 'value'):
                    min_val = widgets['min_spin'].value()
                else:
                    min_val = widgets['original_min']
                if hasattr(widgets['max_spin'], 'value'):
                    max_val = widgets['max_spin'].value()
                else:
                    max_val = widgets['original_max']
                
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                
                result[param_name] = (min_val, max_val, param_type)
        
        return result


class NNBuilderTab(QWidget):
    """Tab for building and training neural networks."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent  # v2.0 FIX: self.parent overwrites PyQt parent()
        self.nn_builder = NeuralNetworkBuilder()
        self.layer_configs = []
        
        # Results storage
        self.nn_training_results = []  # List of (name, TrainingResult) tuples
        self.nn_optimizer_results = []  # List of (name, OptimizationResult) tuples
        
        # Optimization
        self.nn_optimizer = HyperparameterOptimizer()
        self.nn_custom_param_ranges = {}
        self.nn_selected_opt_params = {}  # v1.4.0: param_name -> bool
        self.nn_cv_folds = 5
        self.last_nn_opt_result = None  # Store last optimization result
        self._best_params_applied = False  # v1.3.0: True after Apply Best + before Train
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Architecture Configuration
        arch_group = QGroupBox("Architecture Configuration")
        arch_layout = QVBoxLayout(arch_group)
        
        # Add layer controls
        layer_controls = QHBoxLayout()
        
        layer_controls.addWidget(QLabel("Units:"))
        self.units_spin = QSpinBox()
        self.units_spin.setRange(1, 2048)
        self.units_spin.setToolTip("Number of neurons in this hidden layer. Start with 32-128 and decrease in deeper layers.")
        self.units_spin.setValue(128)
        self.units_spin.setToolTip(
            "Number of neurons in this layer. More neurons = higher capacity to learn "
            "complex patterns, but may overfit. Typical: 64-512 per hidden layer."
        )
        layer_controls.addWidget(self.units_spin)
        
        layer_controls.addWidget(QLabel("Activation:"))
        self.activation_combo = QComboBox()
        self.activation_combo.setToolTip("Activation function: relu (fast, default), tanh (smooth gradients), gelu (modern, often better), swish (self-gated)")
        self.activation_combo.addItems(['relu', 'leaky_relu', 'tanh', 'gelu', 'sigmoid', 'elu'])
        self.activation_combo.setToolTip(
            "Non-linear function applied to each neuron's output. "
            "relu: fast default, good for most problems. "
            "tanh: zero-centered, good for balanced data. "
            "gelu: smooth, state-of-the-art for deep networks. "
            "sigmoid: only for output layer (binary classification)."
        )
        layer_controls.addWidget(self.activation_combo)
        
        layer_controls.addWidget(QLabel("Dropout:"))
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setSingleStep(0.05)
        self.dropout_spin.setToolTip(
            "Fraction of neurons randomly disabled during training. "
            "Prevents overfitting by forcing the network to not rely on single neurons. "
            "Typical: 0.2-0.5. 0 = no dropout."
        )
        layer_controls.addWidget(self.dropout_spin)
        
        arch_layout.addLayout(layer_controls)
        
        # Batch norm and layer norm
        norm_layout = QHBoxLayout()
        self.batch_norm_check = QCheckBox("Batch Normalization")
        self.batch_norm_check.setToolTip("Normalizes layer inputs during training. Speeds up convergence and reduces sensitivity to initialization. Recommended.")
        self.batch_norm_check.setChecked(True)
        self.batch_norm_check.setToolTip(
            "Normalizes layer inputs across the batch. Speeds up training, "
            "allows higher learning rates, and reduces sensitivity to initialization. "
            "Recommended for most networks."
        )
        norm_layout.addWidget(self.batch_norm_check)
        
        self.layer_norm_check = QCheckBox("Layer Normalization")
        self.layer_norm_check.setToolTip(
            "Normalizes across features within each sample. Better than batch norm "
            "for small batches and recurrent networks. Use alone or combined with batch norm."
        )
        norm_layout.addWidget(self.layer_norm_check)
        
        arch_layout.addLayout(norm_layout)
        
        # Add layer button
        self.add_layer_btn = QPushButton("Add Layer")
        self.add_layer_btn.setToolTip("Add a hidden layer to the neural network. Start with 1-2 layers for most problems.")
        self.add_layer_btn.clicked.connect(self._add_layer)
        arch_layout.addWidget(self.add_layer_btn)
        
        # Layer list
        self.layer_list = QListWidget()
        arch_layout.addWidget(self.layer_list)
        
        # Remove layer button
        self.remove_layer_btn = QPushButton("Remove Selected Layer")
        self.remove_layer_btn.setToolTip("Remove the currently selected hidden layer from the architecture.")
        self.remove_layer_btn.clicked.connect(self._remove_layer)
        arch_layout.addWidget(self.remove_layer_btn)
        
        left_layout.addWidget(arch_group)
        
        # Training Configuration (compact 4x2 grid)
        train_group = QGroupBox("Training Configuration")
        train_layout = QGridLayout(train_group)
        
        # Row 0: Epochs + Batch Size
        train_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 10000)
        self.epochs_spin.setValue(500)
        self.epochs_spin.setSingleStep(50)
        self.epochs_spin.setToolTip(
            "Maximum training epochs (full passes through the training data). "
            "Early stopping will likely halt before this limit. "
            "Typical: 200-1000 depending on dataset size."
        )
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Batch Size:"), 0, 2)
        self.batch_size_combo = QComboBox()
        self.batch_size_combo.addItems(['8', '16', '32', '64', '128', '256'])
        self.batch_size_combo.setCurrentText('32')
        self.batch_size_combo.setToolTip(
            "Number of samples processed together before updating weights. "
            "Larger = faster training but more memory. Smaller = noisier gradients but may generalize better. "
            "Typical: 16-128. Reduce if you get out-of-memory errors."
        )
        train_layout.addWidget(self.batch_size_combo, 0, 3)
        
        # Row 1: Learning Rate + Optimizer
        train_layout.addWidget(QLabel("Learning Rate:"), 1, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setToolTip(
            "Step size for weight updates. Too high = unstable training. "
            "Too low = very slow convergence. Typical: 0.0001 - 0.01. "
            "With scheduler, start moderate (0.001) and let it decay."
        )
        train_layout.addWidget(self.lr_spin, 1, 1)
        
        train_layout.addWidget(QLabel("Optimizer:"), 1, 2)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['adam', 'adamw', 'rmsprop'])
        self.optimizer_combo.setToolTip(
            "Algorithm for updating neural network weights. "
            "adam: adaptive, works well for most problems (recommended). "
            "adamw: Adam with proper weight decay (often better generalization). "
            "rmsprop: good for recurrent networks and non-stationary objectives."
        )
        train_layout.addWidget(self.optimizer_combo, 1, 3)
        
        # Row 2: Weight Decay + Patience
        train_layout.addWidget(QLabel("Weight Decay:"), 2, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.setDecimals(6)
        self.weight_decay_spin.setToolTip(
            "L2 regularization for network weights. Penalizes large weights, "
            "encouraging the model to use all inputs moderately. Prevents overfitting."
        )
        train_layout.addWidget(self.weight_decay_spin, 2, 1)
        
        train_layout.addWidget(QLabel("Patience:"), 2, 2)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(5, 500)
        self.patience_spin.setValue(50)
        self.patience_spin.setToolTip(
            "Epochs to wait before stopping if validation loss doesn't improve. "
            "Higher = more chance to escape local minima. Lower = faster stopping. "
            "Typical: 20-100."
        )
        train_layout.addWidget(self.patience_spin, 2, 3)
        
        # Row 3: Random Seed (CV Folds now lives in the Cross-Validation section
        # via the shared CVConfigWidget; we alias it below for compatibility).
        train_layout.addWidget(QLabel("Random Seed:"), 3, 0)
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setToolTip(
            "Random seed for reproducible NN training. "
            "Same seed = same weight initialization and data shuffling."
        )
        train_layout.addWidget(self.random_seed_spin, 3, 1)
        
        # v1.3.0: Row 4: LR Scheduler + Early Stopping Min Delta
        train_layout.addWidget(QLabel("Scheduler:"), 4, 0)
        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems([
            'reduce_on_plateau', 'cosine_annealing', 'step_lr', 'exponential', 'none'
        ])
        self.scheduler_combo.setCurrentText('reduce_on_plateau')
        self.scheduler_combo.setToolTip(
            "Learning rate scheduler. cosine_annealing: smooth decay; "
            "reduce_on_plateau: drop when metrics stagnate; "
            "step_lr: periodic drops; exponential: continuous decay; none: fixed LR"
        )
        train_layout.addWidget(self.scheduler_combo, 4, 1)
        
        train_layout.addWidget(QLabel("Min Delta:"), 4, 2)
        self.min_delta_spin = QDoubleSpinBox()
        self.min_delta_spin.setRange(1e-6, 0.1)
        self.min_delta_spin.setValue(1e-4)
        self.min_delta_spin.setDecimals(6)
        self.min_delta_spin.setSingleStep(1e-4)
        self.min_delta_spin.setToolTip(
            "Minimum change in validation loss to qualify as improvement. "
            "Smaller = more sensitive (stops later); Larger = less sensitive (stops earlier)"
        )
        train_layout.addWidget(self.min_delta_spin, 4, 3)
        
        # v1.3.0: Row 5: Gradient Clipping
        train_layout.addWidget(QLabel("Grad Clip:"), 5, 0)
        self.grad_clip_spin = QDoubleSpinBox()
        self.grad_clip_spin.setRange(0.0, 10.0)
        self.grad_clip_spin.setValue(1.0)
        self.grad_clip_spin.setDecimals(2)
        self.grad_clip_spin.setSingleStep(0.1)
        self.grad_clip_spin.setToolTip(
            "Maximum gradient norm for clipping. 0 = disabled. "
            "Prevents exploding gradients in deep networks."
        )
        train_layout.addWidget(self.grad_clip_spin, 5, 1)
        
        left_layout.addWidget(train_group)

        # v2.0: Cross-validation configuration (shared widget). NNs now support
        # the same five strategies as sklearn models. Kept compact; the left
        # panel is scrollable so nothing is lost on smaller screens.
        cv_group = QGroupBox("Cross-Validation")
        cv_group_layout = QVBoxLayout(cv_group)
        cv_group_layout.setContentsMargins(4, 4, 4, 4)
        self.cv_widget = CVConfigWidget()
        self.cv_widget.set_problem_type(self._get_problem_type())
        # Backward-compatible alias: code that reads self.cv_folds_spin.value()
        # now reads the shared widget's folds spinner.
        self.cv_folds_spin = self.cv_widget.folds_spin
        cv_group_layout.addWidget(self.cv_widget)
        left_layout.addWidget(cv_group)

        # Hyperparameter Optimization (compact layout)
        opt_group = QGroupBox("Hyperparameter Optimization")
        opt_layout = QGridLayout(opt_group)
        
        # Row 0: Trials + CV Folds side by side
        opt_layout.addWidget(QLabel("Trials:"), 0, 0)
        self.opt_trials_spin = QSpinBox()
        self.opt_trials_spin.setRange(10, 200)
        self.opt_trials_spin.setValue(50)
        self.opt_trials_spin.setSingleStep(10)
        self.opt_trials_spin.setToolTip(
            "Number of Bayesian optimization trials. Each trial tests a different "
            "architecture + hyperparameter combination. More = better results but MUCH longer."
        )
        opt_layout.addWidget(self.opt_trials_spin, 0, 1)
        
        opt_layout.addWidget(QLabel("CV Folds:"), 0, 2)
        self.opt_cv_folds_spin = QSpinBox()
        self.opt_cv_folds_spin.setRange(1, 10)
        self.opt_cv_folds_spin.setValue(5)
        self.opt_cv_folds_spin.setToolTip(
            "K-fold CV for evaluating each trial. Higher = more reliable scores "
            "but each trial takes Kx longer. Use fewer folds for large datasets."
        )
        opt_layout.addWidget(self.opt_cv_folds_spin, 0, 3)
        
        # Row 1-2: Advanced settings in 2x2 grid (compact labels)
        opt_layout.addWidget(QLabel("Opt.Ep:"), 1, 0)
        self.opt_epochs_spin = QSpinBox()
        self.opt_epochs_spin.setRange(10, 2000)
        self.opt_epochs_spin.setValue(200)
        self.opt_epochs_spin.setSingleStep(10)
        self.opt_epochs_spin.setToolTip("Epochs per trial (simple validation)")
        opt_layout.addWidget(self.opt_epochs_spin, 1, 1)
        
        opt_layout.addWidget(QLabel("CV Ep:"), 1, 2)
        self.cv_epochs_spin = QSpinBox()
        self.cv_epochs_spin.setRange(10, 1000)
        self.cv_epochs_spin.setValue(100)
        self.cv_epochs_spin.setSingleStep(10)
        self.cv_epochs_spin.setToolTip("Epochs per fold (K-fold CV)")
        opt_layout.addWidget(self.cv_epochs_spin, 1, 3)
        
        opt_layout.addWidget(QLabel("Opt.Pat:"), 2, 0)
        self.opt_patience_spin = QSpinBox()
        self.opt_patience_spin.setRange(5, 200)
        self.opt_patience_spin.setValue(30)
        self.opt_patience_spin.setToolTip("Early stopping patience (validation)")
        opt_layout.addWidget(self.opt_patience_spin, 2, 1)
        
        opt_layout.addWidget(QLabel("CV Pat:"), 2, 2)
        self.cv_patience_spin = QSpinBox()
        self.cv_patience_spin.setRange(5, 100)
        self.cv_patience_spin.setValue(20)
        self.cv_patience_spin.setToolTip("Early stopping patience (CV fold)")
        opt_layout.addWidget(self.cv_patience_spin, 2, 3)
        
        # Row 3: Random Seed for optimization
        opt_layout.addWidget(QLabel("Random Seed:"), 3, 0)
        self.opt_random_seed_spin = QSpinBox()
        self.opt_random_seed_spin.setRange(0, 9999)
        self.opt_random_seed_spin.setValue(42)
        self.opt_random_seed_spin.setToolTip("Random seed for reproducible NN optimization")
        opt_layout.addWidget(self.opt_random_seed_spin, 3, 1, 1, 3)
        
        # v1.4.0: Compact 3-button toolbar in 1 row (was 2 rows)
        btn_style = "QPushButton{padding:3px 6px;font-size:8pt;}"
        
        self.select_nn_params_btn = QPushButton("Select Params")
        self.select_nn_params_btn.setStyleSheet(btn_style)
        self.select_nn_params_btn.setToolTip("Choose which parameters to optimize")
        self.select_nn_params_btn.clicked.connect(self._select_nn_opt_params)
        opt_layout.addWidget(self.select_nn_params_btn, 4, 0)
        
        self.edit_nn_ranges_btn = QPushButton("Edit Ranges")
        self.edit_nn_ranges_btn.setStyleSheet(btn_style)
        self.edit_nn_ranges_btn.clicked.connect(self._edit_nn_parameter_ranges)
        opt_layout.addWidget(self.edit_nn_ranges_btn, 4, 1)
        
        self.reset_nn_ranges_btn = QPushButton("Reset")
        self.reset_nn_ranges_btn.setStyleSheet(btn_style)
        self.reset_nn_ranges_btn.clicked.connect(self._reset_nn_parameter_ranges)
        self.reset_nn_ranges_btn.setEnabled(False)
        opt_layout.addWidget(self.reset_nn_ranges_btn, 4, 2)
        
        # Status label (compact, replaces the separate 'nn_selected_label')
        self.nn_ranges_label = QLabel("Default")
        self.nn_ranges_label.setStyleSheet("color: #888888; font-size: 7pt;")
        self.nn_ranges_label.setToolTip("Current parameter range status")
        opt_layout.addWidget(self.nn_ranges_label, 4, 3)
        
        left_layout.addWidget(opt_group)
        
        # v1.3.0: Action buttons in a compact horizontal grid (3 columns)
        actions_group = QGroupBox("Actions")
        actions_layout = QGridLayout(actions_group)
        actions_layout.setSpacing(4)
        
        # v2.0: Info row — strategy and CV labels (compact, non-overlapping)
        info_layout = QHBoxLayout()
        info_layout.setSpacing(8)
        strategy_label = QLabel("Strategy: Bayesian (Optuna)")
        strategy_label.setStyleSheet("color: #00bcd4; font-size: 8pt;")
        strategy_label.setToolTip(
            "Neural Network architecture search uses Bayesian optimization only.\n"
            "Grid Search is not available because NN hyperparameter spaces are\n"
            "non-rectangular (variable architecture sizes) and require gradient-based\n"
            "training that cannot be evaluated with standard GridSearchCV."
        )
        info_layout.addWidget(strategy_label)

        # v2.0: NNs now support the full CV strategy set (see the Cross-Validation
        # section above). Show a short pointer instead of the old "K-Fold only".
        cv_label = QLabel("CV: see Cross-Validation section")
        cv_label.setStyleSheet("color: #00bcd4; font-size: 8pt;")
        cv_label.setToolTip(
            "Neural Networks now support the same cross-validation strategies as\n"
            "sklearn models: K-Fold, Stratified (quantiles, with optional reserved\n"
            "bin), Group K-Fold, LOO, and LOGO. Configure them in the\n"
            "Cross-Validation section above. Note: LOO/LOGO train one network per\n"
            "sample/group — expensive for NNs; use them only on small datasets."
        )
        info_layout.addWidget(cv_label)
        info_layout.addStretch()
        actions_layout.addLayout(info_layout, 0, 0, 1, 3)

        # Buttons row
        self.optimize_btn = QPushButton("Auto-Optimize")
        self.optimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                font-weight: bold;
                padding: 6px;
                font-size: 9pt;
            }
        """)
        self.optimize_btn.setToolTip("Bayesian optimization of NN hyperparameters")
        self.optimize_btn.clicked.connect(self._optimize_hyperparameters)
        actions_layout.addWidget(self.optimize_btn, 1, 0)
        
        self.apply_best_params_btn = QPushButton("Apply Best")
        self.apply_best_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A148C;
                color: white;
                font-weight: bold;
                padding: 6px;
                font-size: 9pt;
            }
            QPushButton:disabled {
                background-color: #9E9E9E;
                color: #E0E0E0;
            }
        """)
        self.apply_best_params_btn.clicked.connect(self._apply_best_nn_parameters)
        self.apply_best_params_btn.setEnabled(False)
        self.apply_best_params_btn.setToolTip("Run Auto-Optimize first to enable this button")
        actions_layout.addWidget(self.apply_best_params_btn, 1, 1)
        
        self.train_btn = QPushButton("Train NN")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                padding: 6px;
                font-size: 10pt;
            }
        """)
        self.train_btn.setToolTip("Train neural network with current configuration")
        self.train_btn.clicked.connect(self._train_network)
        actions_layout.addWidget(self.train_btn, 1, 2)
        
        left_layout.addWidget(actions_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Quick architecture presets
        presets_group = QGroupBox("Quick Presets")
        presets_layout = QHBoxLayout(presets_group)
        
        self.small_preset_btn = QPushButton("Small")
        self.small_preset_btn.clicked.connect(lambda: self._load_preset('small'))
        presets_layout.addWidget(self.small_preset_btn)
        
        self.medium_preset_btn = QPushButton("Medium")
        self.medium_preset_btn.clicked.connect(lambda: self._load_preset('medium'))
        presets_layout.addWidget(self.medium_preset_btn)
        
        self.large_preset_btn = QPushButton("Large")
        self.large_preset_btn.clicked.connect(lambda: self._load_preset('large'))
        presets_layout.addWidget(self.large_preset_btn)
        
        left_layout.addWidget(presets_group)

        # v2.0: Make the left (configuration) panel scrollable so that all
        # controls — including the new Cross-Validation section and the action
        # buttons — remain reachable on typical laptop and desktop screens.
        # Nothing is ever cut off; the panel scrolls vertically when needed.
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)
        
        # v1.4.0: Right panel — compact 4-panel layout with limited heights
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setSpacing(4)
        right_layout.setContentsMargins(4, 4, 4, 4)
        
        # Create right-panel widgets if not already created
        if not hasattr(self, 'progress_text'):
            self.progress_text = QTextEdit()
            self.progress_text.setReadOnly(True)
        if not hasattr(self, 'architecture_text'):
            self.architecture_text = QTextEdit()
            self.architecture_text.setReadOnly(True)
        if not hasattr(self, 'results_table'):
            self.results_table = QTableWidget()
            self.results_table.setColumnCount(6)
            self.results_table.setHorizontalHeaderLabels([
                "Epoch", "Train Loss", "Val Loss", "Train Metric", "Val Metric", "Time"
            ])
        if not hasattr(self, 'model_info_text'):
            self.model_info_text = QTextEdit()
            self.model_info_text.setReadOnly(True)
        
        # v2.0: Compact right panel — reduced to ~25% of screen
        # with smaller minimum heights for better left-panel visibility.
        panel_configs = [
            ("Training Progress", "progress_text", 60, 2),    # min_h, stretch
            ("Architecture Preview", "architecture_text", 40, 1),
            ("Training Results", "results_table", 60, 2),
            ("Model Information", "model_info_text", 40, 1),
        ]
        for grp_name, widget_name, min_h, stretch in panel_configs:
            grp = QGroupBox(grp_name)
            vl = QVBoxLayout(grp)
            vl.setContentsMargins(2, 2, 2, 2)
            vl.setSpacing(2)
            w = getattr(self, widget_name)
            w.setMinimumHeight(min_h)
            # Allow text widgets to expand but prefer compact sizing
            if isinstance(w, QTextEdit):
                w.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
            vl.addWidget(w, stretch=1)
            right_layout.addWidget(grp, stretch=stretch)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([750, 250])  # v2.0: Right panel ~25% of screen
        layout.addWidget(splitter)

    def _get_problem_type(self) -> str:
        """Return 'classification' or 'regression' based on the loaded dataset.

        Defaults to 'regression' when no data is loaded yet."""
        data_manager = _safe_get_parent_attr(self, 'data_tab.get_data_manager()')
        if data_manager is None or getattr(data_manager, 'data', None) is None:
            return 'regression'
        return 'classification' if getattr(data_manager, '_is_classification', False) else 'regression'

    def _get_cv_config(self) -> CVConfig:
        """Return the full CVConfig from the shared CV widget."""
        return self.cv_widget.get_config()

    def _resolve_cv_groups(self, cv_config: CVConfig, data_manager,
                           test_size: float, random_state: int) -> tuple:
        """Resolve group labels for grouped CV, aligned to the training rows.

        Returns (groups_train, group_info); both None for non-grouped strategies.
        Raises GroupingError with a user-facing message on failure.
        """
        if cv_config.strategy not in ('group', 'logo'):
            return None, None
        groups_full, group_info = self.cv_widget.resolve_groups()
        groups_train = data_manager.get_train_group_labels(
            groups_full, test_size=test_size, random_state=random_state,
        )
        cv_config.groups = np.asarray(groups_train)
        return groups_train, group_info

    def _load_preset(self, preset):
        """Load a quick architecture preset."""
        self.layer_list.clear()
        self.layer_configs.clear()
        
        presets = {
            'small': [(64, 'relu', 0.2), (32, 'relu', 0.1)],
            'medium': [(256, 'relu', 0.3), (128, 'relu', 0.2), (64, 'relu', 0.1)],
            'large': [(512, 'relu', 0.4), (256, 'relu', 0.3), (128, 'relu', 0.2), (64, 'relu', 0.1)]
        }
        
        for units, activation, dropout in presets.get(preset, []):
            config = {
                'n_units': units,
                'activation': activation,
                'dropout_rate': dropout,
                'use_batch_norm': True,
                'use_layer_norm': False
            }
            self.layer_configs.append(config)
            self.layer_list.addItem(
                f"{units} units, {activation}, dropout={dropout}"
            )
        
        self._update_architecture_preview()
    
    def _add_layer(self):
        """Add a layer to the architecture."""
        config = {
            'n_units': self.units_spin.value(),
            'activation': self.activation_combo.currentText(),
            'dropout_rate': self.dropout_spin.value(),
            'use_batch_norm': self.batch_norm_check.isChecked(),
            'use_layer_norm': self.layer_norm_check.isChecked()
        }
        
        self.layer_configs.append(config)
        self.layer_list.addItem(
            f"{config['n_units']} units, {config['activation']}, "
            f"dropout={config['dropout_rate']}"
        )
        
        self._update_architecture_preview()
    
    def _remove_layer(self):
        """Remove selected layer."""
        current_row = self.layer_list.currentRow()
        if current_row >= 0:
            self.layer_list.takeItem(current_row)
            self.layer_configs.pop(current_row)
            self._update_architecture_preview()
    
    def _update_architecture_preview(self):
        """Update architecture preview text."""
        if not self.layer_configs:
            self.architecture_text.setPlainText("No layers added yet.")
            return
        
        preview = "Neural Network Architecture:\n"
        preview += "=" * 40 + "\n"
        
        for i, config in enumerate(self.layer_configs):
            preview += f"Layer {i+1}:\n"
            preview += f"  Units: {config['n_units']}\n"
            preview += f"  Activation: {config['activation']}\n"
            preview += f"  Dropout: {config['dropout_rate']}\n"
            preview += f"  Batch Norm: {config['use_batch_norm']}\n"
            preview += f"  Layer Norm: {config['use_layer_norm']}\n"
            preview += "\n"
        
        self.architecture_text.setPlainText(preview)
    
    def _select_nn_opt_params(self):
        """v1.4.0: Open dialog to select which NN parameters to optimize."""
        # Default NN parameters and their descriptions
        nn_params = {
            'n_layers': "Number of hidden layers (1-5). High impact on capacity.",
            'n_units': "Neurons per layer (16-512). High impact on expressiveness.",
            'dropout': "Dropout rate (0.0-0.5). High impact on overfitting prevention.",
            'activation': "Activation function. Medium impact.",
            'batch_norm': "Use batch normalization. Low-medium impact.",
            'learning_rate': "Learning rate (1e-5 to 1e-2). High impact on convergence.",
            'batch_size': "Batch size (16, 32, 64, 128). Medium impact.",
            'weight_decay': "Weight decay / L2 regularization. Medium impact.",
            'optimizer': "Optimizer (adam, adamw). Low-medium impact.",
            'gradient_clip_val': "Gradient clipping (0.0-5.0). Medium impact on stability.",
        }
        
        # Recommended (high-impact) subset
        recommended = {'n_layers', 'n_units', 'dropout', 'learning_rate', 'batch_size'}
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Select NN Parameters to Optimize")
        dialog.setMinimumWidth(420)
        dialog.setMinimumHeight(350)
        layout = QVBoxLayout(dialog)
        
        info = QLabel("Checked parameters will be optimized. Unchecked use defaults.")
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Buttons
        btn_layout = QHBoxLayout()
        for label, slot in [("Select All", lambda: [cb.setChecked(True) for cb in cbs.values()]),
                            ("Recommended", lambda: [cb.setChecked(n in recommended) for n, cb in cbs.items()]),
                            ("Clear All", lambda: [cb.setChecked(False) for cb in cbs.values()])]:
            b = QPushButton(label)
            b.clicked.connect(slot)
            btn_layout.addWidget(b)
        layout.addLayout(btn_layout)
        
        # Checkboxes
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        sw = QWidget()
        sl = QVBoxLayout(sw)
        cbs = {}
        for name, tooltip in nn_params.items():
            cb = QCheckBox(name)
            cb.setChecked(self.nn_selected_opt_params.get(name, True))
            cb.setToolTip(tooltip)
            sl.addWidget(cb)
            cbs[name] = cb
        sl.addStretch()
        scroll.setWidget(sw)
        layout.addWidget(scroll)
        
        # OK/Cancel
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dialog.accept)
        btns.rejected.connect(dialog.reject)
        layout.addWidget(btns)
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.nn_selected_opt_params = {n: cb.isChecked() for n, cb in cbs.items()}
            n_chk = sum(self.nn_selected_opt_params.values())
            self.nn_ranges_label.setText(f"{n_chk}/{len(nn_params)} params")
    
    def _edit_nn_parameter_ranges(self):
        """Open dialog to edit NN optimization parameter ranges."""
        if not self.nn_custom_param_ranges:
            # Default ranges
            param_ranges = {
                'n_layers': (1, 5, 'int'),
                'n_units': (16, 512, 'int_log'),
                'dropout': (0.0, 0.5, 'float'),
                'activation': (['relu', 'leaky_relu', 'tanh', 'gelu'], None, 'categorical'),
                'batch_norm': ([True, False], None, 'categorical'),
                'learning_rate': (1e-5, 1e-2, 'float_log'),
                'batch_size': ([16, 32, 64, 128], None, 'categorical'),
                'weight_decay': (1e-6, 1e-3, 'float_log'),
                'optimizer': (['adam', 'adamw'], None, 'categorical'),
            }
        else:
            param_ranges = self.nn_custom_param_ranges
        
        dialog = NNParameterRangeDialog(param_ranges, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.nn_custom_param_ranges = dialog.get_custom_ranges()
            self.reset_nn_ranges_btn.setEnabled(True)
            self.nn_ranges_label.setText("Using custom ranges")
    
    def _reset_nn_parameter_ranges(self):
        """Reset parameter ranges to defaults."""
        self.nn_custom_param_ranges = {}
        self.reset_nn_ranges_btn.setEnabled(False)
        self.nn_ranges_label.setText("Using default ranges")
    
    def _optimize_hyperparameters(self):
        """Optimize neural network hyperparameters."""
        data_manager = _safe_get_parent_attr(self, 'data_tab.get_data_manager()')
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        if not self.layer_configs:
            QMessageBox.warning(self, "Warning", "Please add at least one layer!")
            return
        
        # v1.5.0: Always use internal test split; external validation is separate
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=False)
        
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        
        # Get custom or default parameter ranges
        if self.nn_custom_param_ranges:
            param_ranges = dict(self.nn_custom_param_ranges)
        else:
            # Default NN parameter ranges
            param_ranges = {
                'n_layers': (1, 5, 'int'),
                'n_units': (16, 512, 'int_log'),
                'dropout': (0.0, 0.5, 'float'),
                'activation': (['relu', 'leaky_relu', 'tanh', 'gelu'], None, 'categorical'),
                'batch_norm': ([True, False], None, 'categorical'),
                'learning_rate': (1e-5, 1e-2, 'float_log'),
                'batch_size': ([16, 32, 64, 128], None, 'categorical'),
                'weight_decay': (1e-6, 1e-3, 'float_log'),
                'optimizer': (['adam', 'adamw'], None, 'categorical'),
                'gradient_clip_val': (0.0, 5.0, 'float'),
            }
        
        # v1.4.0: Filter to only selected parameters
        if self.nn_selected_opt_params:
            param_ranges = {
                k: v for k, v in param_ranges.items()
                if self.nn_selected_opt_params.get(k, True)
            }
            if not param_ranges:
                QMessageBox.warning(self, "Warning",
                    "No parameters selected for optimization. "
                    "Please use 'Select Parameters' to choose at least one.")
                return
        
        n_trials = self.opt_trials_spin.value()
        cv_folds = self.opt_cv_folds_spin.value()
        
        # Create initial architecture
        self.nn_builder.create_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_config=self.layer_configs,
            problem_type='classification' if data_manager._is_classification else 'regression'
        )
        # v2.0: Pass the user's random_seed to build_model() so torch.manual_seed()
        # uses the correct seed for weight initialization. The TrainingConfig will
        # be created inside the optimization thread; we pass the seed directly.
        random_state = self.opt_random_seed_spin.value()
        self.nn_builder.build_model(random_seed=random_state)

        # Start optimization
        self.optimize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_text.clear()
        self.progress_text.append(
            f"Starting NN architecture optimization ({n_trials} trials)...\n"
            f"CV: {cv_folds}-fold | Epochs: {opt_epochs} (opt) / {cv_epochs} (CV) | "
            f"Patience: {opt_patience} (opt) / {cv_patience} (CV)"
        )
        
        # Read advanced optimization settings from UI (v1.2.1)
        opt_epochs = self.opt_epochs_spin.value()
        cv_epochs = self.cv_epochs_spin.value()
        opt_patience = self.opt_patience_spin.value()
        cv_patience = self.cv_patience_spin.value()
        random_state = self.opt_random_seed_spin.value()
        
        self.optimization_thread = NNOptimizationThread(
            self.nn_optimizer, self.nn_builder,
            X_train, y_train, X_test, y_test,
            input_dim, output_dim, param_ranges, n_trials, cv_folds,
            opt_epochs=opt_epochs,
            cv_epochs=cv_epochs,
            opt_patience=opt_patience,
            cv_patience=cv_patience,
            random_state=random_state
        )
        self.optimization_thread.progress.connect(self._on_optimization_progress)
        self.optimization_thread.status.connect(self._on_optimization_status)
        self.optimization_thread.finished.connect(self._on_optimization_finished)
        self.optimization_thread.error.connect(self._on_optimization_error)
        self.optimization_thread.start()
    
    def _on_optimization_progress(self, trial_num, score, params):
        """Handle optimization progress.

        v2.0 FIX: Force immediate UI repaint with processEvents() so the
        Training Progress console updates live during long optimizations.
        """
        from PyQt6.QtWidgets import QApplication
        self.progress_bar.setValue(int(100 * trial_num / self.opt_trials_spin.value()))
        self.progress_text.append(f"Trial {trial_num}: Score={score:.4f}")
        QApplication.processEvents()
    
    def _on_optimization_status(self, msg: str):
        """v1.3.0: Handle status messages from optimization thread.

        Provides visual feedback during long operations (e.g., CV fold training)
        so the user doesn't think the application has frozen.
        """
        from PyQt6.QtWidgets import QApplication
        self.progress_text.append(msg)
        # Keep progress bar visible and in indeterminate mode during status updates
        if "Compiling" in msg or "Starting" in msg:
            self.progress_bar.setRange(0, 0)  # Indeterminate (animated)
        QApplication.processEvents()

    def _on_optimization_finished(self, result):
        """Handle optimization completion."""
        self.optimize_btn.setEnabled(True)
        # v1.3.0: Reset progress bar to determinate mode before hiding
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setVisible(False)
        
        if result is None:
            QMessageBox.warning(self, "Warning", "Optimization failed!")
            return
        
        # Store optimization result and enable apply button
        self.last_nn_opt_result = result
        self.apply_best_params_btn.setEnabled(True)
        self.apply_best_params_btn.setToolTip(
            f"Best score: {result.best_score:.4f} | Click to apply to architecture"
        )
        from datetime import datetime
        timestamp = datetime.now().strftime("%H%M%S")
        opt_name = f"NN_Optimized_{timestamp}"
        # v2.0 FIX: Keep ALL NN optimizations. The previous code deleted
        # earlier optimizations, causing the export to miss optimization
        # details for earlier optimized+retrained NNs.
        self.nn_optimizer_results.append((opt_name, result))

        details = f"Optimization Complete!\n"
        details += f"Best Score: {result.best_score:.4f}\n"
        details += f"Trials: {result.n_trials}\n"
        if result.cv_metrics and result.cv_metrics.get('test_score'):
            cv_scores = result.cv_metrics['test_score']
            details += f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
        details += "\nBest Parameters:\n"
        for param, value in result.best_params.items():
            details += f"  {param}: {value}\n"
        
        self.progress_text.append(details)
        self.model_info_text.setPlainText(details)
        
        QMessageBox.information(self, "Success", f"Optimization complete!\nBest params shown above.\nNow apply them and click Train.")
    
    def _on_optimization_error(self, error_msg):
        """Handle optimization error."""
        self.optimize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Optimization failed:\n{error_msg}")
    

    def _apply_best_nn_parameters(self):
        """Apply best hyperparameters from last optimization to the UI."""
        if self.last_nn_opt_result is None:
            QMessageBox.warning(self, "Warning", "No optimization result available. Run Auto-Optimize first.")
            return

        params = self.last_nn_opt_result.best_params
        if not params:
            QMessageBox.warning(self, "Warning", "No parameters found in optimization result.")
            return

        try:
            # Clear current layers
            self.layer_configs.clear()
            self.layer_list.clear()

            # Rebuild architecture from optimized parameters
            n_layers = params.get('n_layers', 2)

            for i in range(n_layers):
                # v1.2.1 FIX: Per-layer parameter names use _{i} suffix so each
                # layer can have different values. Format: n_units_0, dropout_0, etc.
                n_units_key = f'n_units_{i}' if f'n_units_{i}' in params else 'n_units'
                dropout_key = f'dropout_{i}' if f'dropout_{i}' in params else 'dropout'
                activation_key = f'activation_{i}' if f'activation_{i}' in params else 'activation'
                bn_key = f'batch_norm_{i}' if f'batch_norm_{i}' in params else 'batch_norm'

                n_units = params.get(n_units_key, params.get('n_units', 128))
                dropout = params.get(dropout_key, params.get('dropout', 0.2))
                activation = params.get(activation_key, params.get('activation', 'relu'))
                use_bn = params.get(bn_key, params.get('batch_norm', True))

                config = {
                    'n_units': int(n_units),
                    'activation': str(activation),
                    'dropout_rate': float(dropout),
                    'use_batch_norm': bool(use_bn),
                    'use_layer_norm': False
                }
                self.layer_configs.append(config)
                self.layer_list.addItem(
                    f"{config['n_units']} units, {config['activation']}, dropout={config['dropout_rate']}"
                )

            # Apply training hyperparameters
            if 'learning_rate' in params:
                self.lr_spin.setValue(float(params['learning_rate']))
            if 'batch_size' in params:
                bs = int(params['batch_size'])
                for idx in range(self.batch_size_combo.count()):
                    if int(self.batch_size_combo.itemText(idx)) == bs:
                        self.batch_size_combo.setCurrentIndex(idx)
                        break
            if 'weight_decay' in params:
                self.weight_decay_spin.setValue(float(params['weight_decay']))
            if 'optimizer' in params:
                opt_idx = self.optimizer_combo.findText(str(params['optimizer']).lower())
                if opt_idx >= 0:
                    self.optimizer_combo.setCurrentIndex(opt_idx)

            self._update_architecture_preview()
            self._best_params_applied = True  # v1.3.0: Mark that best params were applied

            QMessageBox.information(
                self, "Applied",
                f"Best hyperparameters applied!\n"
                f"Layers: {n_layers}\n"
                f"Learning rate: {params.get('learning_rate', 'N/A')}\n"
                f"Batch size: {params.get('batch_size', 'N/A')}\n"
                f"\nNow click 'Train Neural Network' to train with these values."
            )
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply parameters:\n{str(e)}")

    def _train_network(self):
        """Train the neural network."""
        data_manager = _safe_get_parent_attr(self, 'data_tab.get_data_manager()')
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        if not self.layer_configs:
            QMessageBox.warning(self, "Warning", "Please add at least one layer!")
            return
        
        # v2.0: Configure the shared CV widget (dataset + problem type + target
        # for point-mass detection), then validate the chosen CV strategy.
        problem_type = 'classification' if data_manager._is_classification else 'regression'
        self.cv_widget.set_dataframe(data_manager.data)
        self.cv_widget.set_problem_type(problem_type)
        try:
            self.cv_widget.set_target_values(data_manager.data[data_manager.target_column].values)
        except Exception:
            self.cv_widget.set_target_values(None)

        # v1.5.0: Always use internal test split; external validation is separate
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=False)

        # v2.0: build + validate the CV config and resolve group labels.
        # prepare_data(use_validation=False) uses default test_size=0.2, seed=42.
        cv_config = self._get_cv_config()
        # v2.0: sync the CV fold shuffling with the NN Random Seed so the seed
        # controls weight init, data shuffling, AND the CV folds.
        cv_config.random_state = self.random_seed_spin.value()
        try:
            cv_groups, group_info = self._resolve_cv_groups(
                cv_config, data_manager, test_size=0.2, random_state=42)
        except GroupingError as e:
            QMessageBox.warning(self, "Grouping Error", str(e))
            return
        _gi = group_info if cv_config.strategy in ('group', 'logo') else None
        _issues = validate_cv_config(cv_config, len(X_train),
                                     problem_type=problem_type, group_info=_gi,
                                     y=y_train)
        if has_blocking_issue(_issues):
            QMessageBox.warning(self, "Cross-Validation Not Available",
                                "\n\n".join(i['message'] for i in _issues if i['severity'] == 'block'))
            return
        _warns = [i['message'] for i in _issues if i['severity'] == 'warn']
        if _warns:
            reply = QMessageBox.question(
                self, "Cross-Validation Warning",
                "\n\n".join(_warns) + "\n\nDo you want to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return
        # Stash for _on_training_finished (which runs the trainer with CV).
        self._pending_cv_config = cv_config
        self._pending_cv_groups = cv_groups

        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]

        # Create architecture
        self.nn_builder.create_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_config=self.layer_configs,
            problem_type='classification' if data_manager._is_classification else 'regression'
        )
        # v2.0: create_training_config BEFORE build_model so torch.manual_seed()
        # has access to the user's random_seed during weight initialization.
        batch_size = int(self.batch_size_combo.currentText())
        self.nn_builder.create_training_config(
            epochs=self.epochs_spin.value(),
            batch_size=batch_size,
            learning_rate=self.lr_spin.value(),
            optimizer=self.optimizer_combo.currentText(),
            scheduler=self.scheduler_combo.currentText(),
            weight_decay=self.weight_decay_spin.value(),
            early_stopping_patience=self.patience_spin.value(),
            early_stopping_min_delta=self.min_delta_spin.value(),
            gradient_clip_val=self.grad_clip_spin.value() if self.grad_clip_spin.value() > 0 else 0.0,
            random_seed=self.random_seed_spin.value()
        )
        self.nn_builder.build_model()
        
        # Store CV folds for use in trainer
        self.nn_cv_folds = self.cv_folds_spin.value()
        
        # Start training
        self.train_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_text.clear()
        
        self.training_thread = NNTrainingThread(
            self.nn_builder,
            X_train, y_train,
            X_test, y_test,
            cv_folds=self.nn_cv_folds
        )
        self.training_thread.progress.connect(self._on_training_progress)
        self.training_thread.finished.connect(self._on_training_finished)
        self.training_thread.error.connect(self._on_training_error)
        self.training_thread.start()
    
    def _on_training_progress(self, epoch, train_loss, val_loss):
        """Handle training progress."""
        max_epochs = self.epochs_spin.value()
        progress = min(100, int(100 * epoch / max_epochs))
        self.progress_bar.setValue(progress)
        
        if epoch % 10 == 0:
            self.progress_text.append(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}"
            )
    
    def _on_training_finished(self, history):
        """Handle training completion."""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Get cv_folds for training
        cv_folds = getattr(self, 'nn_cv_folds', self.cv_folds_spin.value())
        
        data_manager = _safe_get_parent_attr(self, 'data_tab.get_data_manager()')
        trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
        
        # v1.5.0: Always use internal test split; external validation is separate
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=False)
        # Get external validation set if loaded
        val_arrays = data_manager.get_validation_arrays()
        X_ext_val, y_ext_val = (val_arrays[0], val_arrays[1]) if val_arrays else (None, None)
        
        # Determine base name: normal or optimized
        if self._best_params_applied:
            base_name = "NN_Optimized"
            self._best_params_applied = False
        else:
            base_name = "Neural Network"
        
        # Train through ModelTrainer to get proper CV and results.
        # The trainer automatically makes the name unique (_1, _2, etc.).
        # v2.0: pass the validated CV config + resolved group labels so the NN
        # uses the same strategy as sklearn models.
        _cv_config = getattr(self, '_pending_cv_config', None)
        _cv_groups = getattr(self, '_pending_cv_groups', None)
        # v2.0: Pass the unfitted preprocessor so the NN CV loop can clone and
        # refit it inside each fold (zero data leakage).
        preprocessor = data_manager.get_preprocessor()
        try:
            result = trainer.train_neural_network(
                self.nn_builder,
                base_name,
                X_train, y_train,
                X_test, y_test,
                'classification' if data_manager._is_classification else 'regression',
                cv_folds=cv_folds,
                X_external_val=X_ext_val,
                y_external_val=y_ext_val,
                cv_config=_cv_config,
                cv_groups=_cv_groups,
                preprocessor=preprocessor,
            )
            
            # Display results
            self._display_results(result)

            # Store training result for results_tab access.
            # result.model_name already contains the unique name assigned by trainer.
            self.nn_training_results.append((result.model_name, result))
            
            QMessageBox.information(self, "Success", "Training complete!")
        except Exception as e:
            self._on_training_error(str(e))
    
    def _on_training_error(self, error_msg):
        """Handle training error."""
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Training failed:\n{error_msg}")
    
    def _display_results(self, result):
        """Display training results in both the table and model info text."""
        # v1.4.0 FIX: Populate the Training Results table with epoch-by-epoch data
        history = result.nn_history or {}
        train_losses = history.get('train_loss', [])
        val_losses = history.get('val_loss', [])
        train_metrics = history.get('train_metric', [])
        val_metrics = history.get('val_metric', [])
        epoch_times = history.get('epoch_time', [])
        
        n_epochs = max(len(train_losses), len(val_losses))
        self.results_table.setRowCount(n_epochs)
        
        for epoch in range(n_epochs):
            self.results_table.setItem(epoch, 0, 
                QTableWidgetItem(str(epoch + 1)))
            self.results_table.setItem(epoch, 1,
                QTableWidgetItem(f"{train_losses[epoch]:.6f}" if epoch < len(train_losses) else "N/A"))
            self.results_table.setItem(epoch, 2,
                QTableWidgetItem(f"{val_losses[epoch]:.6f}" if epoch < len(val_losses) else "N/A"))
            self.results_table.setItem(epoch, 3,
                QTableWidgetItem(f"{train_metrics[epoch]:.6f}" if epoch < len(train_metrics) else "N/A"))
            self.results_table.setItem(epoch, 4,
                QTableWidgetItem(f"{val_metrics[epoch]:.6f}" if epoch < len(val_metrics) else "N/A"))
            self.results_table.setItem(epoch, 5,
                QTableWidgetItem(f"{epoch_times[epoch]:.2f}" if epoch < len(epoch_times) else "N/A"))
        
        # Model Information text
        info = f"Training Time: {result.training_time:.2f}s\n"
        
        if result.metrics:
            info += "\nMetrics:\n"
            for metric, value in result.metrics.items():
                info += f"  {metric}: {value:.4f}\n"
        
        cv_scores = result.cv_metrics.get('test_score', [])
        if cv_scores:
            info += f"\nCV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
            info += f"CV Scores: {[f'{s:.4f}' for s in cv_scores]}\n"
        
        if result.nn_history:
            info += f"\nEpochs: {n_epochs}\n"
            if train_losses:
                info += f"Final Train Loss: {train_losses[-1]:.4f}\n"
            if val_losses:
                info += f"Final Val Loss: {val_losses[-1]:.4f}\n"
        
        self.model_info_text.setPlainText(info)
