"""Neural Network Builder tab."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QCheckBox, QTextEdit, QMessageBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QSplitter, QListWidget,
    QGridLayout, QFrame, QScrollArea, QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.nn_builder import NeuralNetworkBuilder, ActivationFunction
from ..core.optimizer import HyperparameterOptimizer


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
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, optimizer, nn_builder, X_train, y_train, X_val, y_val,
                 input_dim, output_dim, param_ranges, n_trials, cv_folds=1):
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
    
    def run(self):
        try:
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
                progress_callback=progress_callback
            )
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
        self.parent = parent
        self.nn_builder = NeuralNetworkBuilder()
        self.layer_configs = []
        
        # Results storage
        self.nn_training_results = []  # List of (name, TrainingResult) tuples
        self.nn_optimizer_results = []  # List of (name, OptimizationResult) tuples
        
        # Optimization
        self.nn_optimizer = HyperparameterOptimizer()
        self.nn_custom_param_ranges = {}
        self.nn_cv_folds = 5
        self.last_nn_opt_result = None  # Store last optimization result
        
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
        self.units_spin.setValue(128)
        layer_controls.addWidget(self.units_spin)
        
        layer_controls.addWidget(QLabel("Activation:"))
        self.activation_combo = QComboBox()
        self.activation_combo.addItems(['relu', 'leaky_relu', 'tanh', 'gelu', 'sigmoid', 'elu'])
        layer_controls.addWidget(self.activation_combo)
        
        layer_controls.addWidget(QLabel("Dropout:"))
        self.dropout_spin = QDoubleSpinBox()
        self.dropout_spin.setRange(0.0, 1.0)
        self.dropout_spin.setValue(0.2)
        self.dropout_spin.setSingleStep(0.05)
        layer_controls.addWidget(self.dropout_spin)
        
        arch_layout.addLayout(layer_controls)
        
        # Batch norm and layer norm
        norm_layout = QHBoxLayout()
        self.batch_norm_check = QCheckBox("Batch Normalization")
        self.batch_norm_check.setChecked(True)
        norm_layout.addWidget(self.batch_norm_check)
        
        self.layer_norm_check = QCheckBox("Layer Normalization")
        norm_layout.addWidget(self.layer_norm_check)
        
        arch_layout.addLayout(norm_layout)
        
        # Add layer button
        self.add_layer_btn = QPushButton("Add Layer")
        self.add_layer_btn.clicked.connect(self._add_layer)
        arch_layout.addWidget(self.add_layer_btn)
        
        # Layer list
        self.layer_list = QListWidget()
        arch_layout.addWidget(self.layer_list)
        
        # Remove layer button
        self.remove_layer_btn = QPushButton("Remove Selected Layer")
        self.remove_layer_btn.clicked.connect(self._remove_layer)
        arch_layout.addWidget(self.remove_layer_btn)
        
        left_layout.addWidget(arch_group)
        
        # Training Configuration
        train_group = QGroupBox("Training Configuration")
        train_layout = QGridLayout(train_group)
        
        train_layout.addWidget(QLabel("Epochs:"), 0, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 10000)
        self.epochs_spin.setValue(500)
        self.epochs_spin.setSingleStep(50)
        train_layout.addWidget(self.epochs_spin, 0, 1)
        
        train_layout.addWidget(QLabel("Batch Size:"), 1, 0)
        self.batch_size_combo = QComboBox()
        self.batch_size_combo.addItems(['8', '16', '32', '64', '128', '256'])
        self.batch_size_combo.setCurrentText('32')
        train_layout.addWidget(self.batch_size_combo, 1, 1)
        
        train_layout.addWidget(QLabel("Learning Rate:"), 2, 0)
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(0.001)
        self.lr_spin.setDecimals(6)
        self.lr_spin.setSingleStep(0.001)
        train_layout.addWidget(self.lr_spin, 2, 1)
        
        train_layout.addWidget(QLabel("Optimizer:"), 3, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['adam', 'adamw', 'sgd', 'rmsprop'])
        train_layout.addWidget(self.optimizer_combo, 3, 1)
        
        train_layout.addWidget(QLabel("Weight Decay:"), 4, 0)
        self.weight_decay_spin = QDoubleSpinBox()
        self.weight_decay_spin.setRange(0.0, 0.1)
        self.weight_decay_spin.setValue(0.0001)
        self.weight_decay_spin.setDecimals(6)
        train_layout.addWidget(self.weight_decay_spin, 4, 1)
        
        train_layout.addWidget(QLabel("Early Stopping Patience:"), 5, 0)
        self.patience_spin = QSpinBox()
        self.patience_spin.setRange(5, 500)
        self.patience_spin.setValue(50)
        train_layout.addWidget(self.patience_spin, 5, 1)
        
        train_layout.addWidget(QLabel("CV Folds:"), 6, 0)
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(1, 10)
        self.cv_folds_spin.setValue(5)
        train_layout.addWidget(self.cv_folds_spin, 6, 1)
        
        left_layout.addWidget(train_group)
        
        # Hyperparameter Optimization
        opt_group = QGroupBox("Hyperparameter Optimization")
        opt_layout = QGridLayout(opt_group)
        
        opt_layout.addWidget(QLabel("Optimization Trials:"), 0, 0)
        self.opt_trials_spin = QSpinBox()
        self.opt_trials_spin.setRange(10, 200)
        self.opt_trials_spin.setValue(50)
        self.opt_trials_spin.setSingleStep(10)
        opt_layout.addWidget(self.opt_trials_spin, 0, 1)
        
        opt_layout.addWidget(QLabel("CV Folds (Optimization):"), 1, 0)
        self.opt_cv_folds_spin = QSpinBox()
        self.opt_cv_folds_spin.setRange(1, 10)
        self.opt_cv_folds_spin.setValue(5)
        opt_layout.addWidget(self.opt_cv_folds_spin, 1, 1)
        
        self.edit_nn_ranges_btn = QPushButton("Edit Optimization Ranges")
        self.edit_nn_ranges_btn.clicked.connect(self._edit_nn_parameter_ranges)
        opt_layout.addWidget(self.edit_nn_ranges_btn, 2, 0, 1, 2)
        
        self.reset_nn_ranges_btn = QPushButton("Reset Ranges to Default")
        self.reset_nn_ranges_btn.clicked.connect(self._reset_nn_parameter_ranges)
        self.reset_nn_ranges_btn.setEnabled(False)
        opt_layout.addWidget(self.reset_nn_ranges_btn, 3, 0, 1, 2)
        
        self.nn_ranges_label = QLabel("Using default ranges")
        opt_layout.addWidget(self.nn_ranges_label, 4, 0, 1, 2)
        
        left_layout.addWidget(opt_group)
        
        # Optimize button
        self.optimize_btn = QPushButton("Auto-Optimize Hyperparameters")
        self.optimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 11pt;
            }
        """)
        self.optimize_btn.clicked.connect(self._optimize_hyperparameters)
        left_layout.addWidget(self.optimize_btn)
        
        # Apply Best Parameters button
        self.apply_best_params_btn = QPushButton("Apply Best Parameters to Architecture")
        self.apply_best_params_btn.setStyleSheet("""
            QPushButton {
                background-color: #4A148C;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 11pt;
            }
            QPushButton:disabled {
                background-color: #9E9E9E;
                color: #E0E0E0;
            }
        """)
        self.apply_best_params_btn.clicked.connect(self._apply_best_nn_parameters)
        self.apply_best_params_btn.setEnabled(False)
        self.apply_best_params_btn.setToolTip("Run Auto-Optimize first to enable this button")
        left_layout.addWidget(self.apply_best_params_btn)
        
        # Train button
        self.train_btn = QPushButton("Train Neural Network")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
            }
        """)
        self.train_btn.clicked.connect(self._train_network)
        left_layout.addWidget(self.train_btn)
        
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
        
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # Right panel - Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Training progress
        progress_group = QGroupBox("Training Progress")
        progress_layout = QVBoxLayout(progress_group)
        
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
        progress_layout.addWidget(self.progress_text)
        
        right_layout.addWidget(progress_group)
        
        # Architecture preview
        preview_group = QGroupBox("Architecture Preview")
        preview_layout = QVBoxLayout(preview_group)
        
        self.architecture_text = QTextEdit()
        self.architecture_text.setReadOnly(True)
        preview_layout.addWidget(self.architecture_text)
        
        right_layout.addWidget(preview_group)
        
        # Results table
        results_group = QGroupBox("Training Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Epoch", "Train Loss", "Val Loss", "Train Metric", "Val Metric", "Time (s)"
        ])
        results_layout.addWidget(self.results_table)
        
        right_layout.addWidget(results_group)
        
        # Model info
        info_group = QGroupBox("Model Information")
        info_layout = QVBoxLayout(info_group)
        
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        info_layout.addWidget(self.model_info_text)
        
        right_layout.addWidget(info_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        layout.addWidget(splitter)
    
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
        data_manager = self.parent.data_tab.get_data_manager()
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        if not self.layer_configs:
            QMessageBox.warning(self, "Warning", "Please add at least one layer!")
            return
        
        use_val = data_manager.validation_data is not None
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
        
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        
        # Get custom or default parameter ranges
        if self.nn_custom_param_ranges:
            param_ranges = self.nn_custom_param_ranges
        else:
            param_ranges = None  # Will use defaults in optimizer
        
        n_trials = self.opt_trials_spin.value()
        cv_folds = self.opt_cv_folds_spin.value()
        
        # Create initial architecture
        self.nn_builder.create_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_config=self.layer_configs,
            problem_type='classification' if data_manager._is_classification else 'regression'
        )
        self.nn_builder.build_model()
        
        # Start optimization
        self.optimize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_text.clear()
        
        self.optimization_thread = NNOptimizationThread(
            self.nn_optimizer, self.nn_builder,
            X_train, y_train, X_test, y_test,
            input_dim, output_dim, param_ranges, n_trials, cv_folds
        )
        self.optimization_thread.progress.connect(self._on_optimization_progress)
        self.optimization_thread.finished.connect(self._on_optimization_finished)
        self.optimization_thread.error.connect(self._on_optimization_error)
        self.optimization_thread.start()
    
    def _on_optimization_progress(self, trial_num, score, params):
        """Handle optimization progress."""
        self.progress_bar.setValue(int(100 * trial_num / self.opt_trials_spin.value()))
        self.progress_text.append(f"Trial {trial_num}: Score={score:.4f}")
    
    def _on_optimization_finished(self, result):
        """Handle optimization completion."""
        self.optimize_btn.setEnabled(True)
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
                # Extract per-layer parameters (Optuna names them like n_units_l0, dropout_l1, etc.)
                n_units_key = f'n_units_l{i}' if f'n_units_l{i}' in params else 'n_units'
                dropout_key = f'dropout_l{i}' if f'dropout_l{i}' in params else 'dropout'
                activation_key = f'activation_l{i}' if f'activation_l{i}' in params else 'activation'
                bn_key = f'batch_norm_l{i}' if f'batch_norm_l{i}' in params else 'batch_norm'

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
        data_manager = self.parent.data_tab.get_data_manager()
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        if not self.layer_configs:
            QMessageBox.warning(self, "Warning", "Please add at least one layer!")
            return
        
        use_val = data_manager.validation_data is not None
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
        
        input_dim = X_train.shape[1]
        output_dim = 1 if len(y_train.shape) == 1 else y_train.shape[1]
        
        # Create architecture
        self.nn_builder.create_architecture(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers_config=self.layer_configs,
            problem_type='classification' if data_manager._is_classification else 'regression'
        )
        self.nn_builder.build_model()
        
        # Create training config
        batch_size = int(self.batch_size_combo.currentText())
        self.nn_builder.create_training_config(
            epochs=self.epochs_spin.value(),
            batch_size=batch_size,
            learning_rate=self.lr_spin.value(),
            optimizer=self.optimizer_combo.currentText(),
            weight_decay=self.weight_decay_spin.value(),
            early_stopping_patience=self.patience_spin.value()
        )
        
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
        
        data_manager = self.parent.data_tab.get_data_manager()
        trainer = self.parent.training_tab.get_trainer()
        
        # Get data (use external validation set if loaded)
        use_val = data_manager.validation_data is not None
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
        
        # Train through ModelTrainer to get proper CV and results
        try:
            result = trainer.train_neural_network(
                self.nn_builder,
                "Neural Network",
                X_train, y_train,
                X_test, y_test,
                'classification' if data_manager._is_classification else 'regression',
                cv_folds=cv_folds
            )
            
            # Display results
            self._display_results(result)

            # Store training result for results_tab access
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
        """Display training results."""
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
            epochs = len(result.nn_history.get('train_loss', []))
            info += f"\nEpochs: {epochs}\n"
            if result.nn_history.get('train_loss'):
                info += f"Final Train Loss: {result.nn_history['train_loss'][-1]:.4f}\n"
            if result.nn_history.get('val_loss'):
                info += f"Final Val Loss: {result.nn_history['val_loss'][-1]:.4f}\n"
        
        self.model_info_text.setPlainText(info)
