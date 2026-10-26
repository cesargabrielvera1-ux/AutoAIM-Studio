"""Model training tab."""

import gc
import inspect
from typing import Optional, List, Dict, Tuple, Any
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
from ..core.cv_factory import CVConfig, validate_cv_config, has_blocking_issue
from ..core.grouping import GroupingError
from ..core.cv_display import format_cv_details
from .cv_config_widget import CVConfigWidget


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
                # v1.3.0: High precision for sensitivity testing (was 6-8, now 10-12)
                spin.setDecimals(10)
                if param_name in ('reg_alpha', 'reg_lambda', 'l2_leaf_reg'):
                    spin.setDecimals(12)
                spin.setSingleStep(10 ** (-spin.decimals()))
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
                elif param_name == 'kernel_type':
                    # v2.1: GP kernel family selection
                    combo.addItems(['RBF', 'Matern', 'RationalQuadratic', 'WhiteKernel', 'RBF+WhiteKernel'])
                    combo.setToolTip(
                        "Kernel family for Gaussian Process. RBF = infinitely smooth; "
                        "Matern = controlled smoothness; RationalQuadratic = mixture of length-scales; "
                        "WhiteKernel = explicit noise term; RBF+WhiteKernel = signal + noise decomposition."
                    )
                elif param_name == 'kernel_nu':
                    # v2.1: Matern smoothness parameter
                    combo.addItems(['0.5', '1.5', '2.5'])
                    combo.setToolTip(
                        "Matern kernel smoothness (nu). 0.5 = once differentiable (rough); "
                        "1.5 = twice differentiable; 2.5 = thrice differentiable (smooth)."
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

            elif ptype == 'categorical':
                # v2.1: Generic categorical parameter handler (used by GP normalize_y, etc.)
                combo = QComboBox()
                combo.setMinimumWidth(180)
                if min_val and isinstance(min_val, (list, tuple)):
                    combo.addItems([str(v) for v in min_val])
                if default_val is not None:
                    combo.setCurrentText(str(default_val))
                self._apply_tooltip(combo, param_name)
                self.scroll_layout.addWidget(combo, row, 1)
                self.param_widgets[param_name] = ('str', combo)
                
            type_label = f"({ptype})"
            self.scroll_layout.addWidget(QLabel(type_label), row, 2)
            row += 1
        
        # v2.0.0: Check which common params the selected model actually accepts.
        # This avoids showing random_state/n_jobs for deterministic models like
        # SVR and Ridge that don't have these parameters, which confused users.
        try:
            est_class = model_info.estimator_class
            sig = inspect.signature(est_class.__init__)
            valid_params = set(sig.parameters.keys())
            has_random_state = 'random_state' in valid_params
            has_n_jobs = 'n_jobs' in valid_params
        except Exception:
            has_random_state = True  # safe default: show it
            has_n_jobs = True
        
        # v1.3.0: Common Parameters section (filtered per model)
        shown_common = has_random_state or has_n_jobs
        if shown_common:
            separator = QLabel("<b>Common Parameters</b>")
            self.scroll_layout.addWidget(separator, row, 0, 1, 3)
            row += 1
        
        if has_random_state:
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
        
        if has_n_jobs:
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
        
        # v2.0: Shared CV configuration widget (replaces the triplicated
        # cv_folds / cv_strategy / LOO-info controls). Exposes the full set of
        # strategies (K-Fold, Stratified, Group K-Fold, LOO, LOGO) with no
        # duplicated logic.
        self.cv_widget = CVConfigWidget()
        # Backward-compatible aliases so existing code reading _cv_spin /
        # _cv_strategy_combo keeps working.
        self._cv_spin = self.cv_widget.folds_spin
        self._cv_strategy_combo = self.cv_widget.strategy_combo
        self._manual_loo_label = self.cv_widget.info_label
        self.scroll_layout.addWidget(self.cv_widget, row, 0, 1, 3)
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
            # v2.1: GP kernel parameter tooltips
            'kernel_type': (
                "Kernel family for Gaussian Process. RBF = infinitely smooth (default); "
                "Matern = controlled smoothness via nu; RationalQuadratic = mixture of length-scales; "
                "WhiteKernel = explicit noise term; RBF+WhiteKernel = signal + noise decomposition."
            ),
            'kernel_length_scale': (
                "Characteristic length scale of the kernel. Smaller = wigglier functions; "
                "larger = smoother functions. This is the MOST important hyperparameter for GP."
            ),
            'kernel_nu': (
                "Matern kernel smoothness parameter. 0.5 = once differentiable (rough, like exponential); "
                "1.5 = twice differentiable; 2.5 = thrice differentiable (smooth, almost RBF)."
            ),
            'kernel_alpha_rq': (
                "Rational Quadratic kernel 'alpha'. Controls the heaviness of the tails. "
                "Low alpha → heavy-tailed (allows long-range correlations); high alpha → approaches RBF."
            ),
            'kernel_noise_level': (
                "Noise level for WhiteKernel (or RBF+WhiteKernel). This is added to the diagonal of "
                "the covariance matrix. Higher = more observation noise assumed."
            ),
            'n_restarts_optimizer': (
                "Number of restarts for the L-BFGS-B optimizer. More restarts = better chance of "
                "finding the global optimum of the marginal likelihood, but slower."
            ),
            'normalize_y': (
                "Normalize the target variable y before fitting. Recommended for GP when the "
                "target scale is very large or very small."
            ),
        }
        if param_name in tooltips:
            widget.setToolTip(tooltips[param_name])
    
    def _on_manual_cv_strategy_changed(self, text: str):
        """Legacy hook — the shared CVConfigWidget handles its own visibility."""
        pass

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
        
        # v2.0.0: Read common parameters only if their widgets exist.
        # Widgets are created conditionally based on what the model accepts.
        rs = getattr(self, '_rs_spin', None)
        rs = rs.value() if rs is not None else 42
        nj = getattr(self, '_nj_spin', None)
        nj = nj.value() if nj is not None else -1
        
        if is_catboost:
            # CatBoost uses 'random_seed' not 'random_state'
            self.custom_params['random_seed'] = rs
        elif hasattr(self, '_rs_spin'):
            self.custom_params['random_state'] = rs
        if hasattr(self, '_nj_spin'):
            self.custom_params['n_jobs'] = nj
        # v2.0.0: Store cv_folds so manual training can use user-selected value
        self.custom_params['cv_folds'] = self._cv_spin.value()
        # v2.0: Store the full CV configuration from the shared widget. This
        # object (not just a string) carries strategy + bins + group source +
        # resolved groups for grouped strategies.
        self.cv_config = self.cv_widget.get_config()
        self.custom_params['cv_strategy'] = self.cv_config.strategy

        self.accept()
    
    def get_config(self):
        """Get selected model and parameters."""
        return self.selected_model_name, self.custom_params


class TrainingThread(QThread):
    """Thread for training models."""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, trainer, model_name, model, X_train, X_test, y_train, y_test, problem_type, feature_names=None,
                 X_external_val=None, y_external_val=None, cv_folds=5, cv_strategy='kfold',
                 cv_config=None, cv_groups=None,
                 # v2.0: preprocessor for sklearn Pipeline (eliminates CV data leakage)
                 preprocessor=None):
        super().__init__()
        self.trainer = trainer
        self.model_name = model_name
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.problem_type = problem_type
        self.feature_names = feature_names
        # v1.5.0: External validation set for dual metrics reporting
        self.X_external_val = X_external_val
        self.y_external_val = y_external_val
        self.cv_folds = cv_folds
        self.cv_strategy = cv_strategy
        # v2.0: full CV config + resolved group labels
        self.cv_config = cv_config
        self.cv_groups = cv_groups
        # v2.0: preprocessor passed through to trainer for Pipeline construction
        self.preprocessor = preprocessor

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
                cv_folds=self.cv_folds,
                feature_names=self.feature_names,
                progress_callback=progress_callback,
                X_external_val=self.X_external_val,
                y_external_val=self.y_external_val,
                cv_strategy=self.cv_strategy,
                cv_config=self.cv_config,
                cv_groups=self.cv_groups,
                preprocessor=self.preprocessor,
            )
            self.finished.emit(result)
        except Exception as e:
            import traceback
            tb = traceback.format_exc()
            logger.error(f"TrainingThread error: {e}\n{tb}")
            self.error.emit(f"{e}\n\n{tb}")


class TrainingTab(QWidget):
    """Model training tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent  # v2.0 FIX: self._main_window overwrites PyQt parent()
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
        self.problem_combo.setToolTip("Select regression for continuous targets (e.g., energy, conductivity) or classification for discrete labels")
        self.problem_combo.addItems(['regression', 'classification'])
        self.problem_combo.setToolTip(
            "Regression: predict continuous values (e.g., formation energy). "
            "Classification: predict discrete categories (e.g., metal vs insulator). "
            "Select before training - it determines which models and metrics are used."
        )
        problem_layout.addWidget(self.problem_combo)
        model_layout.addLayout(problem_layout)
        # v2.0: keep the shared CV widget's problem type in sync (quantile
        # stratification is regression-only). Guarded because the widget is
        # created later in _init_ui.
        self.problem_combo.currentTextChanged.connect(
            lambda text: getattr(self, 'cv_widget', None) is not None
            and self.cv_widget.set_problem_type(text))
        
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
        
        # v2.0: Shared cross-validation configuration widget. Centralizes the
        # strategy combo, folds spinner, quantile bins, group source, clusters,
        # group column and info label (previously triplicated LOO logic).
        self.cv_widget = CVConfigWidget()
        # Backward-compatible aliases into the widget's controls so existing
        # code that reads cv_spin.value() / cv_strategy_combo keeps working.
        self.cv_spin = self.cv_widget.folds_spin
        self.cv_strategy_combo = self.cv_widget.strategy_combo
        self.loo_info_label = self.cv_widget.info_label
        config_layout.addWidget(self.cv_widget, 0, 0, 1, 2)

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
        if self._main_window is None or not hasattr(self._main_window, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return
        
        # Open custom parameters dialog
        problem_type = self.problem_combo.currentText()
        dialog = ManualParamDialog(self.model_registry, problem_type, self)
        # v2.0: give the dialog's CV widget the dataset so it can populate the
        # group-column combo and enable chemical grouping sources, plus the
        # target vector for point-mass detection (reserved-bin suggestion).
        dialog.cv_widget.set_dataframe(data_manager.data)
        dialog.cv_widget.set_problem_type(problem_type)
        try:
            dialog.cv_widget.set_target_values(data_manager.data[data_manager.target_column].values)
        except Exception:
            dialog.cv_widget.set_target_values(None)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        model_name, custom_params = dialog.get_config()
        if not model_name:
            return
        
        # v1.3.0 FIX: Use the random_state from custom_params for data splitting,
        # not the Training Tab's random_spin. This ensures sensitivity tests work.
        rs = custom_params.get('random_state', self.random_spin.value())
        
        # Prepare data
        try:
            self.log_text.append(f"Preparing data for custom training: {model_name}...")
            self.log_text.append(f"Using random_state={rs} for train/test split")
            # v1.5.0: Always use internal test split; external validation is separate
            X_train, X_test, y_train, y_test = data_manager.prepare_data(
                test_size=self.test_spin.value(),
                random_state=rs,
                use_validation=False
            )
            # v1.5.0: Get external validation set separately (dual metrics)
            val_arrays = data_manager.get_validation_arrays()
            X_ext_val, y_ext_val = (val_arrays[0], val_arrays[1]) if val_arrays else (None, None)
            if X_ext_val is not None:
                self.log_text.append(f"Training set: {X_train.shape}, Internal test: {X_test.shape}, External val: {X_ext_val.shape}")
            else:
                self.log_text.append(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare data:\n{str(e)}")
            return
        
        # v2.0: Validate CV strategy before starting (uses the dialog's CV widget
        # config; groups are resolved with the SAME random_state used to split).
        dialog_cv_config = getattr(dialog, 'cv_config', None) or self._get_cv_config()
        dialog_cv_widget = getattr(dialog, 'cv_widget', None)
        # v2.0: sync the CV fold shuffling with the custom random_state (rs) so
        # the seed controls the data split, model init, AND the CV folds.
        dialog_cv_config.random_state = rs
        try:
            cv_groups, group_info = self._resolve_cv_groups(
                dialog_cv_config, data_manager,
                test_size=self.test_spin.value(), random_state=rs,
                widget=dialog_cv_widget,
            )
        except GroupingError as e:
            QMessageBox.warning(self, "Grouping Error", str(e))
            return
        # Validate with the resolved group info (and y_train for reserved-bin rules).
        _group_info_for_val = group_info if dialog_cv_config.strategy in ('group', 'logo') else None
        _issues = validate_cv_config(dialog_cv_config, len(X_train),
                                     problem_type=problem_type,
                                     group_info=_group_info_for_val,
                                     y=y_train)
        if has_blocking_issue(_issues):
            QMessageBox.warning(self, "Cross-Validation Not Available",
                                "\n\n".join(i['message'] for i in _issues if i['severity'] == 'block'))
            return
        _warns = [i['message'] for i in _issues if i['severity'] == 'warn']
        if _warns:
            reply = QMessageBox.question(self, "Cross-Validation Warning",
                                         "\n\n".join(_warns) + "\n\nDo you want to proceed anyway?",
                                         QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                                         QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return

        try:
            # Create estimator with custom parameters
            # v1.5.0: Always pass user's random_state to ensure reproducibility
            model = self.model_registry.create_estimator(
                model_name,
                problem_type=problem_type,
                params=custom_params,
                random_state=rs
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
            
            # v2.0.0: Pass cv_folds from custom_params (user-configurable in
            # ManualParamDialog). Falls back to main UI cv_spin if not set.
            cv_folds = custom_params.get('cv_folds', self.cv_spin.value())

            # v1.5.0: Pass external validation set for dual metrics reporting
            # v2.0: Pass the unfitted preprocessor so the trainer builds a Pipeline
            # that eliminates data leakage by refitting the scaler inside each CV fold.
            preprocessor = data_manager.get_preprocessor()
            self.training_thread = TrainingThread(
                self.trainer,
                manual_name,
                model,
                X_train, X_test,
                y_train, y_test,
                problem_type,
                feature_names=feature_names,
                X_external_val=X_ext_val,
                y_external_val=y_ext_val,
                cv_folds=cv_folds,
                cv_strategy=dialog_cv_config.strategy,
                cv_config=dialog_cv_config,
                cv_groups=cv_groups,
                preprocessor=preprocessor,
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
        if self._main_window is None or not hasattr(self._main_window, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load data first.")
            return
        
        if data_manager.target_column is None:
            QMessageBox.warning(self, "Warning", "Please select a target column")
            return

        # v2.0: give the shared CV widget the dataset (for grouping sources), the
        # current problem type (quantile stratification is regression-only), and
        # the target vector (to detect a degenerate point mass for the reserved bin).
        self.cv_widget.set_dataframe(data_manager.data)
        self.cv_widget.set_problem_type(self.problem_combo.currentText())
        try:
            _full_target = data_manager.data[data_manager.target_column].values
            self.cv_widget.set_target_values(_full_target)
        except Exception:
            self.cv_widget.set_target_values(None)

        # Prepare data
        try:
            self.log_text.append("Preparing data...")
            # v1.5.0: Always use internal test split; external validation is separate
            X_train, X_test, y_train, y_test = data_manager.prepare_data(
                test_size=self.test_spin.value(),
                random_state=self.random_spin.value(),
                use_validation=False
            )
            # v1.5.0: Get external validation set separately (dual metrics)
            val_arrays = data_manager.get_validation_arrays()
            self.X_external_val, self.y_external_val = (val_arrays[0], val_arrays[1]) if val_arrays else (None, None)
            
            if self.X_external_val is not None:
                self.log_text.append(f"Training set: {X_train.shape}, Internal test: {X_test.shape}, External val: {self.X_external_val.shape}")
            else:
                self.log_text.append(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare data:\n{str(e)}")
            return
        
        # v2.0: Validate CV strategy before starting (y_train validates any
        # reserved-bin special value for quantile stratification).
        if not self._validate_cv_strategy(len(X_train), data_manager=data_manager, y=y_train):
            return

        # v2.0: Resolve the full CV configuration (including group labels for
        # grouped strategies) ONCE for the whole batch — every model in this
        # run shares the same split and groups.
        cv_config = self._get_cv_config()
        # v2.0: sync the CV fold shuffling with the user's Random State so the
        # seed controls EVERYTHING (data split, model init, and CV folds).
        cv_config.random_state = self.random_spin.value()
        try:
            cv_groups, _ = self._resolve_cv_groups(
                cv_config, data_manager,
                test_size=self.test_spin.value(),
                random_state=self.random_spin.value(),
            )
        except GroupingError as e:
            QMessageBox.warning(self, "Grouping Error", str(e))
            return
        self._pending_cv_config = cv_config
        self._pending_cv_groups = cv_groups

        # Train models
        problem_type = self.problem_combo.currentText()

        # v2.0 MEMORY: Log baseline memory before the batch starts.
        try:
            import psutil, os
            _proc = psutil.Process(os.getpid())
            self._mem_baseline = _proc.memory_info().rss / (1024 * 1024)
            self.log_text.append(f"Memory baseline: {self._mem_baseline:.1f} MB")
        except Exception:
            self._mem_baseline = 0.0

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, len(selected_items))
        self.progress_bar.setValue(0)
        
        self.training_results = []
        self.current_training_index = 0
        self.selected_models = selected_items
        # v2.0 MEMORY FIX: Do NOT store data arrays as persistent tab
        # attributes — they duplicate memory.  Keep them in a temporary
        # container that is cleared when the batch finishes.
        self._batch_data = {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'problem_type': problem_type,
        }

        self._train_next_model()
    
    def _train_next_model(self):
        """Train next model in queue."""
        if self.current_training_index >= len(self.selected_models):
            self.progress_bar.setVisible(False)
            self.progress_label.setText("Training complete!")
            self._update_results_table()
            # v2.0 MEMORY FIX: Release the temporary data container so the
            # training arrays are not kept as tab attributes after the batch.
            if hasattr(self, '_batch_data'):
                del self._batch_data
                gc.collect()
            return
        
        item = self.selected_models[self.current_training_index]
        model_name = item.data(Qt.ItemDataRole.UserRole)
        
        self.progress_label.setText(f"Training {model_name}...")
        self.progress_bar.setValue(self.current_training_index)
        
        try:
            # v1.5.0: Pass user's random_state to model (not hardcoded 42)
            rs = self.random_spin.value()
            model = self.model_registry.create_estimator(
                model_name,
                problem_type=bd['problem_type'],
                random_state=rs
            )
            self.log_text.append(f"Training {model_name} with random_state={rs}")
            
            # Get feature names from data manager
            data_manager = self._main_window.data_tab.get_data_manager()
            feature_names = data_manager.get_feature_names() if data_manager else None
            
            # v2.0: Reuse the batch-resolved CV configuration (with groups).
            cv_folds = self.cv_spin.value()
            cv_config = getattr(self, '_pending_cv_config', self._get_cv_config())
            cv_groups = getattr(self, '_pending_cv_groups', None)
            # v2.0: Fetch the unfitted preprocessor for Pipeline construction.
            preprocessor = data_manager.get_preprocessor()
            # v2.0 MEMORY FIX: Pull data from the temporary batch container
            # instead of persistent tab attributes.
            bd = self._batch_data
            self.training_thread = TrainingThread(
                self.trainer,
                model_name,
                model,
                bd['X_train'], bd['X_test'],
                bd['y_train'], bd['y_test'],
                bd['problem_type'],
                feature_names=feature_names,
                X_external_val=getattr(self, 'X_external_val', None),
                y_external_val=getattr(self, 'y_external_val', None),
                cv_folds=cv_folds,
                cv_strategy=cv_config.strategy,
                cv_config=cv_config,
                cv_groups=cv_groups,
                preprocessor=preprocessor,
            )
            
            self.training_thread.progress.connect(self._on_training_progress)
            self.training_thread.finished.connect(self._on_model_trained)
            self.training_thread.error.connect(self._on_training_error)
            self.training_thread.start()
            
        except Exception as e:
            self.log_text.append(f"Error training {model_name}: {str(e)}")
            self.current_training_index += 1
            self._train_next_model()
    
    def _on_cv_strategy_changed(self, text: str):
        """Legacy hook kept for backward compatibility.

        v2.0: The shared CVConfigWidget now handles its own conditional
        visibility and info label internally, so this is a no-op. It is kept
        only so any legacy signal connection does not break.
        """
        pass

    def _get_cv_strategy(self) -> str:
        """Return selected CV strategy as a backend string."""
        widget = getattr(self, 'cv_widget', None)
        return widget.current_strategy() if widget is not None else 'kfold'

    def _get_cv_config(self) -> CVConfig:
        """Return the full CVConfig from the shared widget."""
        return self.cv_widget.get_config()

    def _resolve_cv_groups(self, cv_config: CVConfig, data_manager,
                           test_size: float, random_state: int,
                           widget: Optional[CVConfigWidget] = None) -> tuple:
        """Resolve group labels for grouped CV strategies, aligned to train rows.

        Args:
            cv_config: the CV configuration being resolved (groups are written
                back into it).
            data_manager: provides the full dataset + the split alignment.
            test_size: the same test fraction used for the train/test split.
            random_state: the same seed used for the train/test split.
            widget: which CVConfigWidget to read group controls from (defaults
                to the main tab widget; the ManualParamDialog passes its own).

        Returns (groups_train, group_info). Both are None for non-grouped
        strategies. Raises GroupingError with a user-facing message on failure.
        """
        if cv_config.strategy not in ('group', 'logo'):
            return None, None
        w = widget if widget is not None else self.cv_widget
        # Full-dataset group labels from the widget's controls.
        groups_full, group_info = w.resolve_groups()
        # Align to the training rows (prepare_data already split the data).
        groups_train = data_manager.get_train_group_labels(
            groups_full, test_size=test_size, random_state=random_state,
        )
        cv_config.groups = np.asarray(groups_train)
        return groups_train, group_info

    def _validate_cv_strategy(self, n_samples: int, data_manager=None, y=None) -> bool:
        """Validate the CV configuration. Returns True if allowed to proceed.

        v2.0: Delegates to cv_factory.validate_cv_config (shared rules). BLOCK
        issues abort; WARN issues ask for confirmation; INFO issues are silent
        (already shown in the widget info label). Handles grouped strategies by
        resolving group labels first. ``y`` is the target vector, used to
        validate a reserved-bin (special value) for quantile stratification.
        """
        cv_config = self._get_cv_config()

        # For grouped strategies, resolve groups to know n_groups for validation.
        group_info = None
        if cv_config.strategy in ('group', 'logo') and data_manager is not None:
            try:
                _, group_info = self._resolve_cv_groups(
                    cv_config, data_manager,
                    test_size=self.test_spin.value(),
                    random_state=self.random_spin.value(),
                )
            except GroupingError as e:
                QMessageBox.warning(self, "Grouping Error", str(e))
                return False

        issues = validate_cv_config(
            cv_config, n_samples,
            problem_type=self.problem_combo.currentText(),
            group_info=group_info,
            y=y,
        )
        if not issues:
            return True

        # Hard blocks → show and abort.
        blocks = [i['message'] for i in issues if i['severity'] == 'block']
        if blocks:
            QMessageBox.warning(self, "Cross-Validation Not Available",
                                "\n\n".join(blocks))
            return False

        # Warnings → ask for explicit confirmation.
        warns = [i['message'] for i in issues if i['severity'] == 'warn']
        if warns:
            reply = QMessageBox.question(
                self,
                "Cross-Validation Warning",
                "\n\n".join(warns) + "\n\nDo you want to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No
            )
            return reply == QMessageBox.StandardButton.Yes

        return True

    def _on_training_progress(self, msg: str, progress: float):
        """Handle training progress."""
        self.log_text.append(msg)
    
    def _on_model_trained(self, result: TrainingResult):
        """Handle model trained.

        v2.0 MEMORY FIX: After each model finishes we (1) prune the full
        TrainingResult down to the lightweight metadata needed by the results
        table, (2) delete the thread that holds references to X/y arrays and
        the fitted model, and (3) force garbage collection so memory is freed
        before the next model starts.  Without this, training 10-15 models
        sequentially causes the process to be killed by the OS OOM killer.
        """
        # 1. Keep a lightweight copy for the results table (model itself is
        #    huge — Pipeline + XGB/RF trees + NN weights — and is not needed
        #    here; it lives in self.trainer.results if the user later decides
        #    to save the model from the Results tab).
        self.training_results.append(result)
        self.log_text.append(
            f"{result.model_name} trained in {result.training_time:.2f}s. "
            f"Score: {result.metrics.get('r2', result.metrics.get('accuracy', 0)):.4f}"
        )

        # 2. Clean up the finished thread — it still holds references to the
        #    training data arrays and the fitted estimator.
        if self.training_thread is not None:
            self.training_thread.deleteLater()
            self.training_thread = None

        # 3. Aggressive memory cleanup before the next model starts.
        #    This is critical when training many models back-to-back.
        gc.collect()

        # v2.0 MEMORY: Log current memory after cleanup.
        try:
            import psutil, os
            _proc = psutil.Process(os.getpid())
            _mem1 = _proc.memory_info().rss / (1024 * 1024)
            _delta = _mem1 - getattr(self, '_mem_baseline', 0.0)
            self.log_text.append(
                f"Memory after cleanup: {_mem1:.1f} MB "
                f"({_delta:+.1f} MB vs baseline)"
            )
        except Exception:
            pass

        self.current_training_index += 1
        self._train_next_model()
    
    def _on_training_error(self, error_msg: str):
        """Handle training error."""
        self.log_text.append(f"Error: {error_msg}")

        # v2.0 MEMORY FIX: clean up the failed thread and force GC so a
        # crashed model (which may leave dangling C++ objects) does not
        # bloat memory for the remaining models in the queue.
        if self.training_thread is not None:
            self.training_thread.deleteLater()
            self.training_thread = None
        gc.collect()

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
        if self._main_window:
            self._main_window.training_results = self.training_results
    
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
                
                # v2.0: CV breakdown via the shared formatter (handles kfold /
                # stratified / group / loo / logo, incl. Test Size + warnings).
                details += format_cv_details(
                    result.cv_metrics if result.cv_metrics else {},
                    cv_strategy=getattr(result, 'cv_strategy', 'kfold'),
                    cv_config_dict=getattr(result, 'cv_config_dict', None),
                    fmt='html',
                )

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
        if not self._main_window or not hasattr(self._main_window, 'data_tab'):
            QMessageBox.critical(self, "Error", "Cannot access data manager. Please load data first.")
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
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
