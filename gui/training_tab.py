"""Model training tab."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QMessageBox, QSplitter, QListWidget, QListWidgetItem,
    QTabWidget, QGridLayout, QFrame, QDialog, QDialogButtonBox, QScrollArea
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.model_registry import ModelRegistry, ModelInfo
from ..core.trainer import ModelTrainer, TrainingResult


class ManualParamDialog(QDialog):
    """Dialog for manually configuring model parameters before training."""
    
    def __init__(self, model_registry: ModelRegistry, problem_type: str = 'regression', parent=None):
        super().__init__(parent)
        self.model_registry = model_registry
        self.problem_type = problem_type
        self.selected_model_name = None
        self.custom_params = {}
        self.param_widgets = {}
        
        self.setWindowTitle("Train with Custom Parameters")
        self.setMinimumWidth(600)
        self.setMinimumHeight(450)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Select a model and configure its parameters manually. "
            "This bypasses optimization - useful when you already know the best parameters."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)
        
        # Model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(250)
        self._populate_models()
        self.model_combo.currentTextChanged.connect(self._on_model_changed)
        model_layout.addWidget(self.model_combo)
        layout.addLayout(model_layout)
        
        # Parameter scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        
        self.scroll_widget = QWidget()
        self.scroll_layout = QGridLayout(self.scroll_widget)
        self.scroll_layout.setColumnStretch(2, 1)
        
        scroll.setWidget(self.scroll_widget)
        layout.addWidget(scroll)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Initialize with first model
        if self.model_combo.count() > 0:
            self._on_model_changed(self.model_combo.currentText())
    
    def _populate_models(self):
        """Populate model combo with available models."""
        try:
            models = self.model_registry.get_model_info()
            for name, info in models.items():
                pt = self.problem_type
                if pt == 'regression' and info.get('supports_regression'):
                    display = f"{info['display_name']} ({name})"
                    self.model_combo.addItem(display, name)
                elif pt == 'classification' and info.get('supports_classification'):
                    display = f"{info['display_name']} ({name})"
                    self.model_combo.addItem(display, name)
        except Exception:
            pass
    
    def _on_model_changed(self, display_text):
        """Regenerate parameter widgets when model changes."""
        # Clear existing widgets
        while self.scroll_layout.count():
            item = self.scroll_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        self.param_widgets.clear()
        
        model_name = self.model_combo.currentData()
        if not model_name:
            return
        
        try:
            model_info = self.model_registry.get_model(model_name)
        except Exception:
            return
        
        # v1.3.0: Widen dialog to prevent parameter overlap
        self.setMinimumWidth(600)
        
        # Header with wider columns
        self.scroll_layout.addWidget(QLabel("<b>Parameter</b>"), 0, 0)
        self.scroll_layout.addWidget(QLabel("<b>Value</b>"), 0, 1)
        self.scroll_layout.addWidget(QLabel("<b>Info</b>"), 0, 2)
        self.scroll_layout.setColumnStretch(1, 2)  # Value column gets more space
        
        row = 1
        
        # v1.3.0 FIX: Skip random_state/n_jobs from model params — handled in Common section.
        # Skip random_seed for CatBoost — handled by mapping random_state → random_seed.
        SKIP_PARAMS = {'random_state', 'n_jobs', 'random_seed'}
        
        # Show param_ranges (tunable params) only
        for param_name in sorted(model_info.param_ranges.keys()):
            if param_name in SKIP_PARAMS:
                continue
            
            param_range = model_info.param_ranges[param_name]
            default_val = model_info.default_params.get(param_name)
            min_val, max_val, ptype = param_range
            
            self.scroll_layout.addWidget(QLabel(f"{param_name}:"), row, 0)
            
            # v1.3.0 FIX: Special handling for gamma (SVR) — can be 'scale', 'auto', or float
            if param_name == 'gamma' and isinstance(default_val, str):
                combo = QComboBox()
                combo.setEditable(True)
                combo.addItems(['scale', 'auto'])
                combo.setCurrentText(default_val)
                combo.setMinimumWidth(180)
                combo.setToolTip(
                    "'scale' = 1/(n_features * X.var()); 'auto' = 1/n_features; "
                    "or enter a numeric value directly"
                )
                self.scroll_layout.addWidget(combo, row, 1)
                self.param_widgets[param_name] = ('gamma', combo)
                self.scroll_layout.addWidget(QLabel("(str/float)"), row, 2)
                row += 1
                continue
            
            if ptype == 'int':
                spin = QSpinBox()
                spin.setRange(int(min_val), int(max_val))
                spin.setValue(int(default_val) if default_val is not None else int(min_val))
                spin.setMinimumWidth(180)
                self._apply_tooltip(spin, param_name)
                self.scroll_layout.addWidget(spin, row, 1)
                self.param_widgets[param_name] = ('int', spin)
                
            elif ptype == 'float' or ptype == 'float_log':
                # v1.3.0 FIX: If default_val is a string (e.g., max_features='sqrt'),
                # use a combo box instead of a spin box.
                if isinstance(default_val, str):
                    combo = QComboBox()
                    combo.setEditable(True)
                    combo.setMinimumWidth(180)
                    
                    if param_name == 'max_features':
                        combo.addItems(['sqrt', 'log2', 'None'])
                        combo.setToolTip(
                            "'sqrt' = sqrt(n_features), 'log2' = log2(n_features), "
                            "'None' = all features, or enter a fraction (0.0-1.0) or integer"
                        )
                    else:
                        combo.addItem(default_val)
                        combo.setToolTip(f"Default: {default_val}. Enter a numeric value or keep as-is.")
                    
                    combo.setCurrentText(default_val)
                    self.scroll_layout.addWidget(combo, row, 1)
                    self.param_widgets[param_name] = ('str_combo', combo)
                    self.scroll_layout.addWidget(QLabel("(str/float)"), row, 2)
                    row += 1
                    continue
                
                spin = QDoubleSpinBox()
                spin.setRange(float(min_val), float(max_val))
                spin.setValue(float(default_val) if default_val is not None else float(min_val))
                spin.setDecimals(6)
                if param_name in ('reg_alpha', 'reg_lambda', 'l2_leaf_reg'):
                    spin.setDecimals(8)
                spin.setMinimumWidth(180)
                self._apply_tooltip(spin, param_name)
                self.scroll_layout.addWidget(spin, row, 1)
                # v1.3.0 FIX: Store original type to convert float→int when needed
                # (GradientBoosting min_samples_leaf expects int or float, not always float)
                original_type = type(default_val) if default_val is not None else float
                self.param_widgets[param_name] = ('float', spin, original_type)
            
            elif ptype == 'str':
                # v1.3.0: String/categorical parameters (kernel, max_features, etc.)
                combo = QComboBox()
                combo.setEditable(True)
                combo.setMinimumWidth(180)
                
                if param_name == 'kernel':
                    combo.addItems(['linear', 'poly', 'rbf', 'sigmoid'])
                    combo.setToolTip(
                        "Kernel type for SVM. 'rbf' (default) works well for most problems. "
                        "'linear' for large sparse datasets. 'poly' for polynomial boundaries."
                    )
                elif param_name == 'max_features':
                    combo.addItems(['sqrt', 'log2', 'None'])
                    combo.setToolTip(
                        "Number of features to consider at each split. 'sqrt' = sqrt(n_features), "
                        "'log2' = log2(n_features), or enter a fraction (0.0-1.0) or integer."
                    )
                else:
                    combo.addItem(str(default_val) if default_val is not None else "")
                
                if default_val is not None:
                    combo.setCurrentText(str(default_val))
                
                self.scroll_layout.addWidget(combo, row, 1)
                self.param_widgets[param_name] = ('str', combo)
                
            type_label = f"({ptype})"
            self.scroll_layout.addWidget(QLabel(type_label), row, 2)
            row += 1
        
        # v1.3.0: Common Parameters section
        separator = QLabel("<b>Common Parameters</b>")
        self.scroll_layout.addWidget(separator, row, 0, 1, 3)
        row += 1
        
        self.scroll_layout.addWidget(QLabel("random_state:"), row, 0)
        self._rs_spin = QSpinBox()
        self._rs_spin.setRange(0, 9999)
        self._rs_spin.setValue(42)
        self._rs_spin.setMinimumWidth(180)
        self._rs_spin.setToolTip(
            "Random seed for reproducible training. "
            "Change this value to test sensitivity to initialization."
        )
        self.scroll_layout.addWidget(self._rs_spin, row, 1)
        self.scroll_layout.addWidget(QLabel("(int)"), row, 2)
        row += 1
        
        self.scroll_layout.addWidget(QLabel("n_jobs:"), row, 0)
        self._nj_spin = QSpinBox()
        self._nj_spin.setRange(-1, 64)
        self._nj_spin.setValue(-1)
        self._nj_spin.setMinimumWidth(180)
        self._nj_spin.setToolTip(
            "Number of parallel jobs. -1 = use all CPU cores. 1 = single-threaded (reproducible)."
        )
        self.scroll_layout.addWidget(self._nj_spin, row, 1)
        self.scroll_layout.addWidget(QLabel("(int)"), row, 2)
        row += 1
        
        self.scroll_layout.setRowStretch(row, 1)
    
    def _apply_tooltip(self, widget, param_name: str):
        """Apply educational tooltip to a parameter widget."""
        tooltips = {
            'n_estimators': "Number of trees in the ensemble. More = better accuracy but slower training",
            'max_depth': "Maximum tree depth. Deeper trees model complex patterns but may overfit",
            'min_samples_split': "Minimum samples required to split a node. Higher = less overfitting",
            'min_samples_leaf': "Minimum samples in a leaf node. Higher = less overfitting",
            'learning_rate': "Step size for gradient descent. Higher = faster learning but may oscillate. Typical: 0.001 - 0.1",
            'subsample': "Fraction of samples used per boosting round. Lower = more regularization",
            'colsample_bytree': "Fraction of features used per tree. Lower = more regularization",
            'reg_alpha': "L1 regularization. Higher = sparser feature importance",
            'reg_lambda': "L2 regularization. Higher = more weight shrinkage. Prevents overfitting",
            'l2_leaf_reg': "L2 regularization for CatBoost leaf weights. Higher = simpler models",
            'num_leaves': "Number of leaves per tree. LightGBM-specific complexity control",
            'iterations': "Number of boosting iterations. More = better accuracy but slower",
            'C': "Regularization strength for SVM. Smaller C = stronger regularization",
            'epsilon': "Epsilon-tube for SVR. Points within this distance are not penalized",
            'gamma': "Kernel coefficient. 'scale' or 'auto' recommended. Higher = more complex boundary",
            'kernel': "Kernel type for SVM. 'rbf' works well for most problems",
            'alpha': "Regularization strength for Ridge. Higher = stronger regularization",
            'max_features': "Features considered per split. 'sqrt' or 'log2' recommended, or enter a fraction",
            'min_child_weight': "Minimum sum of instance weight in a child. Higher = less overfitting (XGBoost)",
            'dropout': "Fraction of neurons disabled during training. Prevents overfitting in NNs",
            'weight_decay': "L2 weight decay for NNs. Penalizes large weights",
            'patience': "Epochs to wait before early stopping if no improvement",
        }
        if param_name in tooltips:
            widget.setToolTip(tooltips[param_name])
    
    def _on_accept(self):
        """Collect parameters and accept."""
        model_name = self.model_combo.currentData()
        if not model_name:
            QMessageBox.warning(self, "Warning", "Please select a model")
            return
        
        self.selected_model_name = model_name
        self.custom_params = {}
        
        # v1.3.0: Detect CatBoost for random_state → random_seed mapping
        is_catboost = 'catboost' in model_name.lower()
        
        for param_name, widget_info in self.param_widgets.items():
            try:
                # Unpack widget info (may be 2 or 3 elements)
                if len(widget_info) == 3:
                    ptype, widget, original_type = widget_info
                else:
                    ptype, widget = widget_info
                    original_type = None
                
                if ptype == 'int':
                    self.custom_params[param_name] = widget.value()
                elif ptype == 'float':
                    val = widget.value()
                    # v1.3.0 FIX: Convert float→int when original default was int
                    # (e.g., GradientBoosting's min_samples_leaf default is 1, not 1.0)
                    if original_type is int and val == int(val):
                        val = int(val)
                    self.custom_params[param_name] = val
                elif ptype == 'gamma':
                    # v1.3.0 FIX: Handle SVR gamma — can be 'scale', 'auto', or numeric
                    text = widget.currentText().strip()
                    if text.lower() in ('scale', 'auto'):
                        self.custom_params[param_name] = text.lower()
                    else:
                        try:
                            self.custom_params[param_name] = float(text)
                        except ValueError:
                            self.custom_params[param_name] = 'scale'  # fallback
                elif ptype in ('str', 'str_combo'):
                    # v1.3.0: Handle string params with smart type conversion
                    text = widget.currentText().strip()
                    if text == "None":
                        self.custom_params[param_name] = None
                    else:
                        # Try numeric conversion first, fall back to string
                        try:
                            self.custom_params[param_name] = int(text)
                        except ValueError:
                            try:
                                self.custom_params[param_name] = float(text)
                            except ValueError:
                                self.custom_params[param_name] = text
                elif ptype == 'bool':
                    self.custom_params[param_name] = widget.isChecked()
                elif ptype == 'none_str':
                    text = widget.currentText()
                    if text == "None":
                        self.custom_params[param_name] = None
                    else:
                        try:
                            self.custom_params[param_name] = int(text)
                        except ValueError:
                            try:
                                self.custom_params[param_name] = float(text)
                            except ValueError:
                                self.custom_params[param_name] = text
            except Exception:
                pass
        
        # v1.3.0: Add common parameters with model-specific mapping
        rs = self._rs_spin.value()
        nj = self._nj_spin.value()
        
        if is_catboost:
            # CatBoost uses 'random_seed' not 'random_state'
            self.custom_params['random_seed'] = rs
        else:
            self.custom_params['random_state'] = rs
        self.custom_params['n_jobs'] = nj
        
        self.accept()
    
    def get_config(self):
        """Get selected model and parameters."""
        return self.selected_model_name, self.custom_params


class TrainingThread(QThread):
    """Thread for training models."""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, trainer, model_name, model, X_train, X_test, y_train, y_test, problem_type, feature_names=None):
        super().__init__()
        self.trainer = trainer
        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.problem_type = problem_type
        self.feature_names = feature_names  # FIX: Agregar feature_names
    
    def run(self):
        try:
            def progress_callback(msg, progress):
                self.progress.emit(msg, progress)
            
            result = self.trainer.train(
                self.model,
                self.model_name,
                self.X_train, self.y_train,
                self.X_test, self.y_test,
                self.problem_type,
                feature_names=self.feature_names,  # FIX: Pasar feature_names
                progress_callback=progress_callback
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrainingTab(QWidget):
    """Model training tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.model_registry = ModelRegistry()
        self.trainer = ModelTrainer()
        # v1.3.0: Initialize here (not just in _init_ui) to prevent race conditions
        # when callbacks try to access training_results before _init_ui completes.
        self.training_results = []
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Model selection and configuration
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model selection group
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        # Problem type
        problem_layout = QHBoxLayout()
        problem_layout.addWidget(QLabel("Problem Type:"))
        self.problem_combo = QComboBox()
        self.problem_combo.addItems(['regression', 'classification'])
        self.problem_combo.setToolTip(
            "Regression: predict continuous values (e.g., formation energy). "
            "Classification: predict discrete categories (e.g., metal vs insulator). "
            "Select before training - it determines which models and metrics are used."
        )
        problem_layout.addWidget(self.problem_combo)
        model_layout.addLayout(problem_layout)
        
        # Model list
        model_layout.addWidget(QLabel("Select Models:"))
        self.model_list = QListWidget()
        self.model_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        self._populate_model_list()
        model_layout.addWidget(self.model_list)
        
        # Select all/none buttons
        btn_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        select_all_btn.clicked.connect(self._select_all_models)
        btn_layout.addWidget(select_all_btn)
        
        select_none_btn = QPushButton("Select None")
        select_none_btn.clicked.connect(self._select_no_models)
        btn_layout.addWidget(select_none_btn)
        
        model_layout.addLayout(btn_layout)
        
        left_layout.addWidget(model_group)
        
        # Training configuration
        config_group = QGroupBox("Training Configuration")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("CV Folds:"), 0, 0)
        self.cv_spin = QSpinBox()
        self.cv_spin.setRange(2, 10)
        self.cv_spin.setValue(5)
        self.cv_spin.setToolTip(
            "Number of cross-validation folds. Each model is trained K times "
            "on different train/validation splits. Higher = more robust estimate, "
            "but takes K times longer."
        )
        config_layout.addWidget(self.cv_spin, 0, 1)
        
        config_layout.addWidget(QLabel("Test Size:"), 1, 0)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.1, 0.5)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.2)
        self.test_spin.setToolTip(
            "Fraction of data held out as a final test set. "
            "The model never sees this data during training or CV. "
            "Used only for the final evaluation."
        )
        config_layout.addWidget(self.test_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Random State:"), 2, 0)
        self.random_spin = QSpinBox()
        self.random_spin.setRange(0, 9999)
        self.random_spin.setValue(42)
        self.random_spin.setToolTip(
            "Random seed for splitting data into train/test sets. "
            "Use the same value to get the exact same split every time. "
            "Change it to test sensitivity to data partitioning."
        )
        config_layout.addWidget(self.random_spin, 2, 1)
        
        left_layout.addWidget(config_group)
        
        # Train button
        self.train_btn = QPushButton("Train Selected Models")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                font-weight: bold;
                padding: 15px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
        """)
        self.train_btn.clicked.connect(self.train_selected_models)
        left_layout.addWidget(self.train_btn)
        
        self.train_all_btn = QPushButton("Train All Models")
        self.train_all_btn.clicked.connect(self.train_all_models)
        left_layout.addWidget(self.train_all_btn)
        
        # v1.3.0: Train with custom parameters (bypass optimization)
        self.train_custom_btn = QPushButton("Train with Custom Parameters...")
        self.train_custom_btn.setStyleSheet("""
            QPushButton {
                background-color: #6A1B9A;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 10pt;
            }
            QPushButton:hover {
                background-color: #8E24AA;
            }
        """)
        self.train_custom_btn.setToolTip(
            "Manually configure any model's parameters and train directly. "
            "Useful when you already know the optimal parameters."
        )
        self.train_custom_btn.clicked.connect(self._train_with_custom_params)
        left_layout.addWidget(self.train_custom_btn)
        
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        # Right panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Progress
        self.progress_label = QLabel("Ready to train")
        right_layout.addWidget(self.progress_label)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        right_layout.addWidget(self.progress_bar)
        
        # Results table
        right_layout.addWidget(QLabel("<b>Training Results</b>"))
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(8)
        self.results_table.setHorizontalHeaderLabels([
            'Model', 'Status', 'RMSE/Acc', 'MAE', 'R²/F1', 'CV Score', 'Time (s)', 'Actions'
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        right_layout.addWidget(self.results_table)
        
        # Log output
        right_layout.addWidget(QLabel("<b>Training Log</b>"))
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        right_layout.addWidget(self.log_text)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        layout.addWidget(splitter)
    
    def _populate_model_list(self):
        """Populate model list."""
        models = self.model_registry.get_model_info()
        
        for name, info in models.items():
            display_name = info['display_name']
            item = QListWidgetItem(f"{display_name} ({name})")
            item.setData(Qt.ItemDataRole.UserRole, name)
            item.setToolTip(info['description'])
            self.model_list.addItem(item)
    
    def _select_all_models(self):
        """Select all models."""
        self.model_list.selectAll()
    
    def _select_no_models(self):
        """Deselect all models."""
        self.model_list.clearSelection()
    
    def train_all_models(self):
        """Train all available models."""
        self.model_list.selectAll()
        self.train_selected_models()
    
    def _train_with_custom_params(self):
        """Train a single model with manually configured parameters."""
        if self.parent is None or not hasattr(self.parent, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self.parent.data_tab.get_data_manager()
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return
        
        # Open custom parameters dialog
        problem_type = self.problem_combo.currentText()
        dialog = ManualParamDialog(self.model_registry, problem_type, self)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        model_name, custom_params = dialog.get_config()
        if not model_name:
            return
        
        # Prepare data
        try:
            self.log_text.append(f"Preparing data for custom training: {model_name}...")
            use_validation = data_manager.validation_data is not None
            X_train, X_test, y_train, y_test = data_manager.prepare_data(
                test_size=self.test_spin.value(),
                random_state=self.random_spin.value(),
                use_validation=use_validation
            )
            self.log_text.append(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare data:\n{str(e)}")
            return
        
        try:
            # Create estimator with custom parameters
            model = self.model_registry.create_estimator(
                model_name,
                problem_type=problem_type,
                params=custom_params
            )
            
            # Log the parameters being used
            self.log_text.append(f"Training {model_name} with custom parameters:")
            for k, v in sorted(custom_params.items()):
                self.log_text.append(f"  {k}: {v}")
            
            # Get feature names
            feature_names = data_manager.get_feature_names() if data_manager else None
            
            # Use manual suffix to distinguish from default training
            manual_name = f"{model_name}_Manual"
            
            # Show progress
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 1)
            self.progress_bar.setValue(0)
            self.progress_label.setText(f"Training {model_name} (custom params)...")
            
            # Create and start training thread
            self.training_thread = TrainingThread(
                self.trainer,
                manual_name,
                model,
                X_train, X_test,
                y_train, y_test,
                problem_type,
                feature_names=feature_names
            )
            self.training_thread.progress.connect(self._on_training_progress)
            self.training_thread.finished.connect(self._on_manual_training_finished)
            self.training_thread.error.connect(self._on_manual_training_error)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start custom training:\n{str(e)}")
            self.progress_bar.setVisible(False)
    
    def _on_manual_training_finished(self, result: TrainingResult):
        """Handle completion of custom parameter training."""
        self.training_results.append(result)
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Custom training complete!")
        self._update_results_table()
        
        self.log_text.append(
            f"[CUSTOM] {result.model_name} trained in {result.training_time:.2f}s. "
            f"Score: {result.metrics.get('r2', result.metrics.get('accuracy', 0)):.4f}"
        )
        
        QMessageBox.information(
            self, "Success",
            f"Custom training complete!\n"
            f"Model: {result.model_name}\n"
            f"CV Score: {np.mean(result.cv_metrics.get('test_score', [0])):.4f}"
        )
    
    def _on_manual_training_error(self, error_msg: str):
        """Handle custom training error."""
        self.progress_bar.setVisible(False)
        self.progress_label.setText("Custom training failed")
        self.log_text.append(f"[CUSTOM ERROR] {error_msg}")
        QMessageBox.critical(self, "Error", f"Custom training failed:\n{error_msg}")
    
    def train_selected_models(self):
        """Train selected models."""
        # Get selected models
        selected_items = self.model_list.selectedItems()
        
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select at least one model")
            return
        
        # Get data from parent
        if self.parent is None or not hasattr(self.parent, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return
        
        data_manager = self.parent.data_tab.get_data_manager()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return
        
        if data_manager.target_column is None:
            QMessageBox.warning(self, "Warning", "Please select a target column")
            return
        
        # Prepare data
        try:
            self.log_text.append("Preparing data...")
            use_validation = data_manager.validation_data is not None
            X_train, X_test, y_train, y_test = data_manager.prepare_data(
                test_size=self.test_spin.value(),
                random_state=self.random_spin.value(),
                use_validation=use_validation
            )
            
            self.log_text.append(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare data:\n{str(e)}")
            return
        
        # Train models
        problem_type = self.problem_combo.currentText()
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(selected_items))
        self.progress_bar.setValue(0)
        
        self.training_results = []
        self.current_training_index = 0
        self.selected_models = selected_items
        self.X_train, self.X_test = X_train, X_test
        self.y_train, self.y_test = y_train, y_test
        self.problem_type = problem_type
        
        self._train_next_model()
    
    def _train_next_model(self):
        """Train next model in queue."""
        if self.current_training_index >= len(self.selected_models):
            self.progress_bar.setVisible(False)
            self.progress_label.setText("Training complete!")
            self._update_results_table()
            return
        
        item = self.selected_models[self.current_training_index]
        model_name = item.data(Qt.ItemDataRole.UserRole)
        
        self.progress_label.setText(f"Training {model_name}...")
        self.progress_bar.setValue(self.current_training_index)
        
        try:
            # Create model
            model = self.model_registry.create_estimator(
                model_name,
                problem_type=self.problem_type
            )
            
            # Get feature names from data manager
            data_manager = self.parent.data_tab.get_data_manager()
            feature_names = data_manager.get_feature_names() if data_manager else None
            
            # Create training thread
            self.training_thread = TrainingThread(
                self.trainer,
                model_name,
                model,
                self.X_train, self.X_test,
                self.y_train, self.y_test,
                self.problem_type,
                feature_names=feature_names  # FIX: Pasar feature_names
            )
            
            self.training_thread.progress.connect(self._on_training_progress)
            self.training_thread.finished.connect(self._on_model_trained)
            self.training_thread.error.connect(self._on_training_error)
            self.training_thread.start()
            
        except Exception as e:
            self.log_text.append(f"Error training {model_name}: {str(e)}")
            self.current_training_index += 1
            self._train_next_model()
    
    def _on_training_progress(self, msg: str, progress: float):
        """Handle training progress."""
        self.log_text.append(msg)
    
    def _on_model_trained(self, result: TrainingResult):
        """Handle model trained."""
        self.training_results.append(result)
        self.log_text.append(
            f"{result.model_name} trained in {result.training_time:.2f}s. "
            f"Score: {result.metrics.get('r2', result.metrics.get('accuracy', 0)):.4f}"
        )
        
        self.current_training_index += 1
        self._train_next_model()
    
    def _on_training_error(self, error_msg: str):
        """Handle training error."""
        self.log_text.append(f"Error: {error_msg}")
        self.current_training_index += 1
        self._train_next_model()
    
    def _update_results_table(self):
        """Update results table."""
        self.results_table.setRowCount(len(self.training_results))
        
        for i, result in enumerate(self.training_results):
            self.results_table.setItem(i, 0, QTableWidgetItem(result.model_name))
            self.results_table.setItem(i, 1, QTableWidgetItem("Complete"))
            
            # Metrics
            if result.problem_type == 'regression':
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.metrics.get('rmse', 0):.4f}"))
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.metrics.get('mae', 0):.4f}"))
                self.results_table.setItem(i, 4, QTableWidgetItem(f"{result.metrics.get('r2', 0):.4f}"))
            else:
                self.results_table.setItem(i, 2, QTableWidgetItem(f"{result.metrics.get('accuracy', 0):.4f}"))
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{result.metrics.get('precision', 0):.4f}"))
                self.results_table.setItem(i, 4, QTableWidgetItem(f"{result.metrics.get('f1', 0):.4f}"))
            
            # CV score
            cv_scores = result.cv_metrics.get('test_score', [])
            if cv_scores and len(cv_scores) > 0:
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                self.results_table.setItem(i, 5, QTableWidgetItem(f"{cv_mean:.4f} ± {cv_std:.4f}"))
            else:
                # No hay datos de CV o CV fallo
                self.results_table.setItem(i, 5, QTableWidgetItem("N/A"))
            
            # Training time
            self.results_table.setItem(i, 6, QTableWidgetItem(f"{result.training_time:.2f}"))
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(5, 0, 5, 0)
            
            details_btn = QPushButton("Details")
            details_btn.setProperty('model_name', result.model_name)
            details_btn.clicked.connect(self._show_model_details)
            actions_layout.addWidget(details_btn)
            
            save_btn = QPushButton("Save")
            save_btn.setProperty('model_name', result.model_name)
            save_btn.clicked.connect(self._save_model)
            actions_layout.addWidget(save_btn)
            
            self.results_table.setCellWidget(i, 7, actions_widget)
        
        # Notify parent
        if self.parent:
            self.parent.training_results = self.training_results
    
    def _show_model_details(self):
        """Show model details."""
        btn = self.sender()
        model_name = btn.property('model_name')
        
        # Find result
        for result in self.training_results:
            if result.model_name == model_name:
                details = f"""
                <h3>Model: {model_name}</h3>
                <p><b>Problem Type:</b> {result.problem_type}</p>
                <p><b>Training Time:</b> {result.training_time:.2f} seconds</p>
                <h4>Metrics:</h4>
                <ul>
                """
                for metric, value in result.metrics.items():
                    details += f"<li><b>{metric.upper()}:</b> {value:.6f}</li>"
                
                details += "</ul>"
                
                # v1.3.0: Show per-fold CV scores
                cv_scores = result.cv_metrics.get('test_score', [])
                if cv_scores:
                    details += "<h4>Cross-Validation Scores (per fold):</h4><table border='1' cellpadding='5'>"
                    details += "<tr><th>Fold</th><th>Score</th></tr>"
                    for fold_idx, score in enumerate(cv_scores, 1):
                        details += f"<tr><td>Fold {fold_idx}</td><td>{score:.6f}</td></tr>"
                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    details += f"<tr><td><b>Mean ± Std</b></td><td><b>{mean_score:.6f} ± {std_score:.6f}</b></td></tr>"
                    details += "</table>"
                
                if result.feature_importance:
                    details += "<h4>Top 10 Important Features:</h4><ul>"
                    sorted_imp = sorted(
                        result.feature_importance.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:10]
                    for feat, imp in sorted_imp:
                        details += f"<li>{feat}: {imp:.4f}</li>"
                    details += "</ul>"
                
                QMessageBox.information(self, "Model Details", details)
                break
    
    def _save_model(self):
        """Save model with format options."""
        btn = self.sender()
        model_name = btn.property('model_name')
        
        from pathlib import Path
        from PyQt6.QtWidgets import QFileDialog, QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QDialogButtonBox, QLabel
        
        # Create dialog for format selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Save Model")
        dialog.setMinimumWidth(400)
        
        layout = QVBoxLayout(dialog)
        
        layout.addWidget(QLabel(f"Save model: <b>{model_name}</b>"))
        layout.addWidget(QLabel("Select format:"))
        
        # Format options
        joblib_radio = QRadioButton("Joblib/Pickle (legacy format)")
        joblib_radio.setChecked(False)
        layout.addWidget(joblib_radio)
        
        bundle_radio = QRadioButton("Model Bundle with manifest.json (for Predict tab)")
        bundle_radio.setChecked(True)
        bundle_radio.setStyleSheet("font-weight: bold;")
        layout.addWidget(bundle_radio)
        
        bundle_info = QLabel(
            "<small>The Model Bundle includes:<br>"
            "• manifest.json - Model metadata and feature names<br>"
            "• Complete model with preprocessing<br>"
            "• Compatible with the 'Predict' tab</small>"
        )
        layout.addWidget(bundle_info)
        
        # Buttons
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        if bundle_radio.isChecked():
            # Save as bundle
            self._save_model_bundle(model_name)
        else:
            # Save as legacy joblib
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save Model",
                f"{model_name}.joblib",
                "Model Files (*.joblib *.pkl);;All Files (*)"
            )
            
            if file_path:
                try:
                    self.trainer.save_model(model_name, file_path)
                    QMessageBox.information(self, "Success", f"Model saved to {file_path}")
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Failed to save model:\n{str(e)}")
    
    def _save_model_bundle(self, model_name: str):
        """Save model as bundle with manifest.json."""
        from PyQt6.QtWidgets import QFileDialog
        
        # Verify we have access to data_manager
        if not self.parent or not hasattr(self.parent, 'data_tab'):
            QMessageBox.critical(self, "Error", "Cannot access data manager. Please load data first.")
            return
        
        data_manager = self.parent.data_tab.get_data_manager()
        if not data_manager or not data_manager._target_column:
            QMessageBox.critical(self, "Error", "No data loaded. Please load and prepare data first.")
            return
        
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Save Model Bundle",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            try:
                from pathlib import Path
                output_path = Path(folder_path) / f"{model_name}_bundle"
                
                saved_path = self.trainer.save_model_bundle(model_name, str(output_path), data_manager)
                
                QMessageBox.information(
                    self,
                    "Success",
                    f"Model bundle saved to:\n{saved_path}\n\n"
                    f"The bundle contains:\n"
                    f"• manifest.json - Model metadata\n"
                    f"• pipeline.joblib or model.pt - The trained model\n"
                    f"• preprocessor.joblib - Preprocessing (for NN)\n\n"
                    f"Use this folder in the 'Predict' tab for standalone inference."
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export model bundle:\n{str(e)}")
    
    def get_trainer(self) -> ModelTrainer:
        """Get the model trainer."""
        return self.trainer
