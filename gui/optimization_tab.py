"""Hyperparameter Optimization tab."""

import logging
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem,
    QGroupBox, QTextEdit, QMessageBox, QProgressBar, QDialog,
    QDialogButtonBox, QScrollArea, QDoubleSpinBox, QGridLayout,
    QCheckBox, QSplitter
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.optimizer import HyperparameterOptimizer, GridSearchOptimizer, OptimizationResult
from ..core.model_registry import ModelRegistry  # v2.0: for default_params access
from ..core.cv_factory import CVConfig, validate_cv_config, has_blocking_issue
from ..core.grouping import GroupingError
from ..core.cv_display import format_cv_details
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


class ParameterRangeDialog(QDialog):
    """Dialog for editing optimization parameter ranges."""
    
    def __init__(self, param_ranges, parent=None):
        super().__init__(parent)
        self.param_ranges = param_ranges
        self.custom_ranges = {}
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("Edit Optimization Parameter Ranges")
        self.setMinimumWidth(400)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout(self)
        
        info_label = QLabel(
            "Modify the ranges for each hyperparameter. "
            "These ranges will be used by Optuna during optimization."
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
        
        row = 1
        for param_name, (min_val, max_val, param_type) in self.param_ranges.items():
            name_label = QLabel(f"{param_name}")
            scroll_layout.addWidget(name_label, row, 0)
            
            if param_type == 'categorical':
                min_spin = QLabel(f"{', '.join(map(str, min_val))}")
                max_spin = QLabel("N/A")
            elif param_type == 'float_log':
                min_spin = QDoubleSpinBox()
                min_spin.setDecimals(8)
                min_spin.setValue(min_val)
                min_spin.setRange(1e-10, 1.0)
                
                max_spin = QDoubleSpinBox()
                max_spin.setDecimals(8)
                max_spin.setValue(max_val)
                max_spin.setRange(1e-10, 1.0)
            elif param_type == 'int_log':
                min_spin = QSpinBox()
                min_spin.setValue(min_val)
                min_spin.setRange(1, 10000)
                
                max_spin = QSpinBox()
                max_spin.setValue(max_val)
                max_spin.setRange(1, 10000)
            else:
                min_spin = QDoubleSpinBox()
                min_spin.setValue(min_val)
                
                max_spin = QDoubleSpinBox()
                max_spin.setValue(max_val)
            
            scroll_layout.addWidget(min_spin, row, 1)
            scroll_layout.addWidget(max_spin, row, 2)
            
            self.custom_ranges[param_name] = {
                'min_widget': min_spin,
                'max_widget': max_spin,
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
                result[param_name] = (widgets['original_min'], widgets['original_max'], param_type)
            else:
                min_val = widgets['min_widget'].value()
                max_val = widgets['max_widget'].value()
                
                # Ensure min <= max
                if min_val > max_val:
                    min_val, max_val = max_val, min_val
                
                result[param_name] = (min_val, max_val, param_type)
        
        return result


class OptimizationParamSelectorDialog(QDialog):
    """v1.4.0: Dialog for selecting which parameters to optimize.
    
    Shows checkboxes for all parameters in param_ranges.
    Recommended set selects high-impact parameters only.
    """
    
    HIGH_IMPACT_PARAMS = {
        'random_forest': ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf'],
        'gradient_boosting': ['n_estimators', 'max_depth', 'learning_rate', 'subsample'],
        'xgboost': ['n_estimators', 'max_depth', 'learning_rate', 'subsample', 'colsample_bytree', 'min_child_weight', 'gamma'],
        'lightgbm': ['n_estimators', 'max_depth', 'learning_rate', 'num_leaves', 'subsample', 'colsample_bytree'],
        'catboost': ['iterations', 'depth', 'learning_rate'],
        'svr': ['C', 'gamma', 'kernel'],
        'ridge': ['alpha'],
    }
    
    PARAM_TOOLTIPS = {
        'n_estimators': "Number of trees/iterations. High impact on accuracy.",
        'max_depth': "Tree depth. High impact on model complexity and overfitting.",
        'learning_rate': "Step size. High impact on convergence speed and stability.",
        'subsample': "Fraction of samples per round. Medium-high regularization impact.",
        'colsample_bytree': "Fraction of features per tree. Medium regularization impact.",
        'reg_alpha': "L1 regularization. Medium impact on feature sparsity.",
        'reg_lambda': "L2 regularization. Medium impact on weight shrinkage.",
        'min_child_weight': "Min sum of instance weight in child. Medium-high overfitting control.",
        'gamma': "Min loss reduction for split. Medium impact on tree conservativeness.",
        'num_leaves': "Leaves per tree (LightGBM). High impact on complexity.",
        'iterations': "Boosting rounds (CatBoost). High impact on accuracy.",
        'depth': "Tree depth (CatBoost). High impact on complexity.",
        'l2_leaf_reg': "L2 regularization (CatBoost). Medium impact.",
        'min_samples_split': "Min samples to split. Medium impact on overfitting.",
        'min_samples_leaf': "Min samples in leaf. Medium impact on overfitting.",
        'max_features': "Features per split. Medium impact with many features.",
        'C': "Regularization strength. High impact on SVM.",
        'epsilon': "Epsilon-tube width. Low-medium impact.",
        'gamma': "Kernel coefficient. High impact on SVM.",
        'kernel': "Kernel type. High impact on SVM (but increases search space 4x).",
        'alpha': "Regularization strength. High impact on Ridge.",
    }
    
    def __init__(self, algorithm, param_ranges, selected_params, parent=None):
        super().__init__(parent)
        self.algorithm = algorithm
        self.param_ranges = param_ranges
        self.selected_params = selected_params or {}
        self.checkboxes = {}
        
        self.setWindowTitle(f"Select Parameters to Optimize — {algorithm}")
        self.setMinimumWidth(450)
        self.setMinimumHeight(400)
        
        layout = QVBoxLayout(self)
        
        info = QLabel(
            "Checked parameters will be optimized by Bayesian search. "
            "Unchecked parameters use their default values. "
            "Fewer checked parameters = faster optimization."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        
        # Quick selection buttons
        btn_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("Select All")
        select_all_btn.setToolTip("Select all models for optimization")
        select_all_btn.clicked.connect(self._select_all)
        btn_layout.addWidget(select_all_btn)
        
        select_rec_btn = QPushButton("Select Recommended")
        select_rec_btn.setToolTip("Select only high-impact parameters for faster optimization")
        select_rec_btn.clicked.connect(self._select_recommended)
        btn_layout.addWidget(select_rec_btn)
        
        clear_all_btn = QPushButton("Clear All")
        clear_all_btn.setToolTip("Deselect all models")
        clear_all_btn.clicked.connect(self._clear_all)
        btn_layout.addWidget(clear_all_btn)
        
        layout.addLayout(btn_layout)
        
        # Scrollable checkbox list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        
        for param_name in sorted(param_ranges.keys()):
            cb = QCheckBox(param_name)
            cb.setChecked(self.selected_params.get(param_name, True))
            
            # Tooltip with impact level
            tooltip = self.PARAM_TOOLTIPS.get(param_name, "")
            ptype = param_ranges[param_name][2] if len(param_ranges[param_name]) > 2 else ''
            range_info = f"({ptype})"
            cb.setToolTip(f"{tooltip} {range_info}" if tooltip else range_info)
            
            scroll_layout.addWidget(cb)
            self.checkboxes[param_name] = cb
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        # OK/Cancel
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
    
    def _select_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(True)
    
    def _clear_all(self):
        for cb in self.checkboxes.values():
            cb.setChecked(False)
    
    def _select_recommended(self):
        recommended = self.HIGH_IMPACT_PARAMS.get(self.algorithm, set(self.checkboxes.keys()))
        for name, cb in self.checkboxes.items():
            cb.setChecked(name in recommended)
    
    def get_selected_params(self):
        """Return dict of param_name -> bool for selected parameters."""
        return {name: cb.isChecked() for name, cb in self.checkboxes.items()}


class OptimizationThread(QThread):
    """Thread for running hyperparameter optimization (Bayesian or Grid Search)."""
    progress = pyqtSignal(int, float, dict)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, optimizer, model_type, algorithm, param_ranges, n_trials,
                 X_train, y_train, X_test, y_test, problem_type, cv_folds=5,
                 timeout=None, pruning=True, random_state=42,
                 selected_params=None, default_params=None,
                 search_strategy="bayesian", grid_resolution="medium",
                 cv_strategy='kfold', cv_config=None, cv_groups=None,
                 # v2.0: preprocessor for sklearn Pipeline (eliminates CV data leakage)
                 preprocessor=None):
        super().__init__()
        self.optimizer = optimizer
        self.model_type = model_type
        self.algorithm = algorithm
        self.param_ranges = param_ranges
        self.n_trials = n_trials
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.timeout = timeout
        self.pruning = pruning
        self.random_state = random_state
        self.selected_params = selected_params
        self.default_params = default_params
        self.search_strategy = search_strategy
        self.grid_resolution = grid_resolution
        self.cv_strategy = cv_strategy
        # v2.0: full CV config + resolved group labels
        self.cv_config = cv_config
        self.cv_groups = cv_groups
        # v2.0: preprocessor passed through for Pipeline construction
        self.preprocessor = preprocessor

    def run(self):
        try:
            def progress_callback(trial_num, score, params):
                self.progress.emit(trial_num, score, params)

            if self.search_strategy == "grid":
                # Use GridSearchOptimizer
                gs_optimizer = GridSearchOptimizer(resolution=self.grid_resolution)
                result = gs_optimizer.optimize_sklearn_model(
                    model_type=self.model_type,
                    algorithm=self.algorithm,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    problem_type=self.problem_type,
                    param_ranges=self.param_ranges,
                    cv_folds=self.cv_folds,
                    timeout=self.timeout if self.timeout else None,
                    progress_callback=progress_callback,
                    random_state=self.random_state,
                    selected_params=self.selected_params,
                    default_params=self.default_params,
                    cv_strategy=self.cv_strategy,
                    cv_config=self.cv_config,
                    cv_groups=self.cv_groups,
                    preprocessor=self.preprocessor,
                )
            else:
                # Use Bayesian (Optuna)
                result = self.optimizer.optimize_sklearn_model(
                    model_type=self.model_type,
                    algorithm=self.algorithm,
                    X_train=self.X_train,
                    y_train=self.y_train,
                    X_test=self.X_test,
                    y_test=self.y_test,
                    problem_type=self.problem_type,
                    param_ranges=self.param_ranges,
                    n_trials=self.n_trials,
                    cv_folds=self.cv_folds,
                    timeout=self.timeout,
                    pruning=self.pruning,
                    random_state=self.random_state,
                    progress_callback=progress_callback,
                    selected_params=self.selected_params,
                    default_params=self.default_params,
                    cv_strategy=self.cv_strategy,
                    cv_config=self.cv_config,
                    cv_groups=self.cv_groups,
                    preprocessor=self.preprocessor,
                )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class TrainBestThread(QThread):
    """Thread for training a model with best optimized parameters."""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, trainer, model, model_name, X_train, y_train, X_test, y_test,
                 problem_type, cv_folds, feature_names=None,
                 X_external_val=None, y_external_val=None,
                 cv_strategy='kfold', cv_config=None, cv_groups=None):
        super().__init__()
        self.trainer = trainer
        self.model = model
        self.model_name = model_name
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.problem_type = problem_type
        self.cv_folds = cv_folds
        self.feature_names = feature_names
        # v1.5.0: External validation set for dual metrics
        self.X_external_val = X_external_val
        self.y_external_val = y_external_val
        self.cv_strategy = cv_strategy
        # v2.0: full CV config + resolved group labels
        self.cv_config = cv_config
        self.cv_groups = cv_groups

    def run(self):
        try:
            def progress_callback(msg, progress):
                self.progress.emit(msg, progress)

            result = self.trainer.train(
                model=self.model,
                model_name=self.model_name,
                X_train=self.X_train,
                y_train=self.y_train,
                X_test=self.X_test,
                y_test=self.y_test,
                problem_type=self.problem_type,
                cv_folds=self.cv_folds,
                feature_names=self.feature_names,
                progress_callback=progress_callback,
                X_external_val=self.X_external_val,
                y_external_val=self.y_external_val,
                cv_strategy=self.cv_strategy,
                cv_config=self.cv_config,
                cv_groups=self.cv_groups,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OptimizationTab(QWidget):
    """Tab for hyperparameter optimization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent  # v2.0 FIX: self.parent overwrites PyQt parent()
        self.optimizer = HyperparameterOptimizer()
        self.model_registry = ModelRegistry()  # v2.0: for default_params access
        self.optimization_results = []
        self.custom_param_ranges = {}
        self.selected_opt_params = {}  # v1.4.0: param_name -> bool for each model type
        self.last_opt_result = None  # Store last optimization result for "Apply" button
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Model Type:"))
        self.model_type_combo = QComboBox()
        self.model_type_combo.setToolTip("Select which model type to configure hyperparameter ranges for")
        self.model_type_combo.addItems([
            "Random Forest", "Gradient Boosting", "Histogram Gradient Boosting",
            "XGBoost", "LightGBM", "CatBoost", "Support Vector Machine",
            "k-Nearest Neighbors", "Gaussian Process"
        ])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        model_layout.addWidget(self.model_type_combo)
        
        left_layout.addWidget(model_group)
        
        # Optimization Configuration
        opt_group = QGroupBox("Optimization Configuration")
        opt_layout = QGridLayout(opt_group)
        
        # v2.0: Search strategy selection
        opt_layout.addWidget(QLabel("Search Strategy:"), 0, 0)
        self.strategy_combo = QComboBox()
        self.strategy_combo.addItems(["Bayesian (Optuna)", "Grid Search"])
        self.strategy_combo.setToolTip(
            "Bayesian: smart exploration, fewer trials, recommended for most cases.\n"
            "Grid Search: exhaustive evaluation, deterministic, recommended for <=3 parameters."
        )
        self.strategy_combo.currentTextChanged.connect(self._on_strategy_changed)
        opt_layout.addWidget(self.strategy_combo, 0, 1)

        # v2.0: Grid resolution (only visible for Grid Search)
        opt_layout.addWidget(QLabel("Grid Resolution:"), 1, 0)
        self.grid_resolution_combo = QComboBox()
        self.grid_resolution_combo.addItems(["Coarse (3 pts)", "Medium (5 pts)", "Fine (7 pts)"])
        self.grid_resolution_combo.setCurrentIndex(1)
        self.grid_resolution_combo.setToolTip(
            "Number of grid points per parameter. Coarse = faster but less precise. "
            "Fine = slower but more thorough."
        )
        self.grid_resolution_combo.setEnabled(False)
        opt_layout.addWidget(self.grid_resolution_combo, 1, 1)

        # v2.0: Grid resolution info label (shown when Bayesian selected)
        self.grid_info_label = QLabel(
            "ℹ Grid Resolution is only used with Grid Search. "
            "Bayesian optimization uses adaptive sampling, not fixed grids."
        )
        self.grid_info_label.setStyleSheet("color: #ff9800; font-size: 8pt;")
        self.grid_info_label.setWordWrap(True)
        opt_layout.addWidget(self.grid_info_label, 2, 0, 1, 2)

        # v2.0: Grid estimation label
        self.grid_estimate_label = QLabel("")
        self.grid_estimate_label.setStyleSheet("color: #ff9800; font-size: 8pt;")
        self.grid_estimate_label.setWordWrap(True)
        opt_layout.addWidget(self.grid_estimate_label, 3, 0, 1, 2)

        opt_layout.addWidget(QLabel("Number of Trials:"), 4, 0)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(10, 200)
        self.trials_spin.setValue(50)
        self.trials_spin.setSingleStep(10)
        self.trials_spin.setToolTip(
            "Number of optimization trials. Each trial tests a different combination "
            "of hyperparameters. More trials = better results but much longer. "
            "50 is a good balance for most problems."
        )
        opt_layout.addWidget(self.trials_spin, 4, 1)
        
        # v2.0: Shared cross-validation configuration widget (replaces the
        # triplicated CV Folds / CV Strategy / LOO-info controls). Exposes
        # K-Fold, Stratified, Group K-Fold, LOO and LOGO with no duplicated code.
        self.cv_widget = CVConfigWidget()
        # Backward-compatible aliases into the widget's controls.
        self.cv_folds_spin = self.cv_widget.folds_spin
        self.cv_strategy_combo = self.cv_widget.strategy_combo
        self.loo_info_label = self.cv_widget.info_label
        opt_layout.addWidget(self.cv_widget, 5, 0, 1, 2)
        
        opt_layout.addWidget(QLabel("Random Seed:"), 8, 0)
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setToolTip(
            "Random seed for reproducible optimization. "
            "Same seed = same sequence of hyperparameter trials."
        )
        opt_layout.addWidget(self.random_seed_spin, 8, 1)
        
        opt_layout.addWidget(QLabel("Timeout (s, 0 = none):"), 9, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 36000)
        self.timeout_spin.setValue(0)
        self.timeout_spin.setSingleStep(60)
        self.timeout_spin.setToolTip(
            "Maximum time allowed for optimization (seconds). "
            "0 = no limit. Useful to prevent runaway optimizations on slow machines."
        )
        opt_layout.addWidget(self.timeout_spin, 9, 1)
        
        self.pruning_check = QCheckBox("Enable Pruning (stop unpromising trials)")
        self.pruning_check.setChecked(True)
        self.pruning_check.setToolTip(
            "Automatically stop trials that perform poorly early. "
            "Speeds up optimization by ~30-50% without sacrificing quality. "
            "Recommended: leave enabled."
        )
        opt_layout.addWidget(self.pruning_check, 10, 0, 1, 2)
        
        # v1.4.0: Compact 3-button toolbar (was 3 separate rows)
        btn_style = "QPushButton{padding:3px 6px;font-size:8pt;}"
        
        self.select_params_btn = QPushButton("Select Params")
        self.select_params_btn.setStyleSheet(btn_style)
        self.select_params_btn.setToolTip(
            "Choose which hyperparameters to optimize. "
            "Unselect low-impact parameters to speed up optimization on CPU."
        )
        self.select_params_btn.clicked.connect(self._select_opt_params)
        opt_layout.addWidget(self.select_params_btn, 8, 0)
        
        self.edit_ranges_btn = QPushButton("Edit Ranges")
        self.edit_ranges_btn.setStyleSheet(btn_style)
        self.edit_ranges_btn.clicked.connect(self._edit_parameter_ranges)
        opt_layout.addWidget(self.edit_ranges_btn, 8, 1)
        
        self.reset_ranges_btn = QPushButton("Reset")
        self.reset_ranges_btn.setStyleSheet(btn_style)
        self.reset_ranges_btn.clicked.connect(self._reset_parameter_ranges)
        self.reset_ranges_btn.setEnabled(False)
        opt_layout.addWidget(self.reset_ranges_btn, 8, 2)
        
        # Single compact status label
        self.ranges_label = QLabel("Default ranges, all params")
        self.ranges_label.setStyleSheet("color: #888888; font-size: 7pt;")
        opt_layout.addWidget(self.ranges_label, 8, 3)
        
        left_layout.addWidget(opt_group)
        
        # Problem type
        problem_group = QGroupBox("Problem Type")
        problem_layout = QVBoxLayout(problem_group)
        
        self.problem_type_combo = QComboBox()
        self.problem_type_combo.addItems(["Regression", "Classification"])
        problem_layout.addWidget(self.problem_type_combo)
        
        left_layout.addWidget(problem_group)
        
        # Optimize button
        self.optimize_btn = QPushButton("Start Optimization")
        self.optimize_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
            }
            QPushButton:hover {
                background-color: #1B5E20;
            }
        """)
        self.optimize_btn.clicked.connect(self._start_optimization)
        left_layout.addWidget(self.optimize_btn)
        
        # Train with Best Parameters button
        self.train_best_btn = QPushButton("Train with Best Parameters")
        self.train_best_btn.setStyleSheet("""
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
        self.train_best_btn.clicked.connect(self._train_with_best_params)
        self.train_best_btn.setEnabled(False)
        self.train_best_btn.setToolTip("Run optimization first to enable this button")
        left_layout.addWidget(self.train_best_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        left_layout.addStretch()
        splitter.addWidget(left_widget)
        
        # Right panel - Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Progress log
        log_group = QGroupBox("Optimization Progress")
        log_layout = QVBoxLayout(log_group)
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(150)
        log_layout.addWidget(self.progress_text)
        right_layout.addWidget(log_group)
        
        # Results table
        results_group = QGroupBox("Optimization Results")
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(4)
        self.results_table.setHorizontalHeaderLabels([
            "Rank", "Score", "Parameters", "Trial#"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        right_layout.addWidget(results_group)
        
        # Trial details
        details_group = QGroupBox("Best Trial Details")
        details_layout = QVBoxLayout(details_group)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        right_layout.addWidget(details_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([350, 650])
        layout.addWidget(splitter)
    
    def _on_model_type_changed(self):
        """Handle model type change."""
        # v1.4.0: Update the selected params label
        model_type = self.model_type_combo.currentText()
        algo = model_type.lower().replace(' ', '_')
        selected = self.selected_opt_params.get(algo)
        if selected:
            n_checked = sum(1 for v in selected.values() if v)
            n_total = len(selected)
            self.ranges_label.setText(f"Custom: {n_checked}/{n_total} params")
        else:
            param_ranges = self.optimizer.get_param_ranges(model_type)
            n_total = len(param_ranges) if param_ranges else 0
            self.ranges_label.setText(f"Default ranges, all {n_total} params")

        # v2.0: Random Seed only applies to stochastic models. Disable it for
        # deterministic models (SVR, k-NN) that accept no random_state, so the
        # user is never offered a control that would have no effect.
        supports_rs = model_type not in ('Support Vector Machine', 'k-Nearest Neighbors')
        self.random_seed_spin.setEnabled(supports_rs)
        if supports_rs:
            self.random_seed_spin.setToolTip(
                "Random seed for reproducible optimization. "
                "Same seed = same sequence of hyperparameter trials."
            )
        else:
            self.random_seed_spin.setToolTip(
                f"{model_type} is deterministic (no random_state), so the Random "
                "Seed has no effect. It only controls the CV split shuffling "
                "for this model."
            )
        # v2.0: Update grid estimate
        self._update_grid_estimate()

    def _on_strategy_changed(self):
        """v2.0: Handle search strategy change (Bayesian vs Grid Search)."""
        is_grid = self.strategy_combo.currentText() == "Grid Search"
        self.grid_resolution_combo.setEnabled(is_grid)
        self.trials_spin.setEnabled(not is_grid)
        self.pruning_check.setEnabled(not is_grid)
        self.timeout_spin.setEnabled(not is_grid)
        if is_grid:
            self.grid_info_label.setText("")
        else:
            self.grid_info_label.setText(
                "ℹ Grid Resolution is only used with Grid Search. "
                "Bayesian optimization uses adaptive sampling, not fixed grids."
            )
        self._update_grid_estimate()

    def _on_cv_strategy_changed(self, text: str):
        """Legacy hook — the shared CVConfigWidget handles its own visibility."""
        pass

    def _get_cv_config(self) -> CVConfig:
        """Return the full CVConfig from the shared widget."""
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

    def _update_grid_estimate(self):
        """v2.0: Estimate and display GridSearch combinations."""
        is_grid = self.strategy_combo.currentText() == "Grid Search"
        if not is_grid:
            self.grid_estimate_label.setText("")
            return
        model_type = self.model_type_combo.currentText()
        algorithm = model_type
        param_ranges = self.custom_param_ranges.get(model_type)
        if param_ranges is None:
            param_ranges = self.optimizer.get_param_ranges(algorithm)
        if not param_ranges:
            self.grid_estimate_label.setText("")
            return
        # Get selected params
        algo_key = algorithm.lower().replace(' ', '_')
        selected = self.selected_opt_params.get(algo_key)
        resolution_map = {'Coarse (3 pts)': 'coarse', 'Medium (5 pts)': 'medium', 'Fine (7 pts)': 'fine'}
        resolution = resolution_map.get(self.grid_resolution_combo.currentText(), 'medium')
        gs = GridSearchOptimizer(resolution=resolution)
        try:
            grid = gs._ranges_to_grid(param_ranges, selected)
            n = gs._estimate_combinations(grid)
            est_min = max(1, int(n * 0.5))  # rough estimate
            self.grid_estimate_label.setText(
                f"Est. combinations: ~{n} (takes ~{est_min} min at 30s/model)"
            )
        except Exception:
            self.grid_estimate_label.setText("Unable to estimate — check parameter ranges")
    
    def _select_opt_params(self):
        """v1.4.0: Open dialog to select which parameters to optimize."""
        model_type = self.model_type_combo.currentText()
        algorithm = model_type.lower().replace(' ', '_')
        
        # Get param_ranges for this model
        param_ranges = self.custom_param_ranges.get(model_type)
        if param_ranges is None:
            param_ranges = self.optimizer.get_param_ranges(model_type)
        
        if not param_ranges:
            QMessageBox.warning(self, "Warning", f"No parameter ranges for {model_type}")
            return
        
        # Open selector dialog
        dialog = OptimizationParamSelectorDialog(
            algorithm, param_ranges,
            self.selected_opt_params.get(algorithm),
            self
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.selected_opt_params[algorithm] = dialog.get_selected_params()
            n_checked = sum(1 for v in dialog.get_selected_params().values() if v)
            n_total = len(dialog.get_selected_params())
            self.ranges_label.setText(
                f"Custom: {n_checked}/{n_total} params"
            )
    
    def _edit_parameter_ranges(self):
        """Open dialog to edit parameter ranges."""
        model_type = self.model_type_combo.currentText()
        param_ranges = self.custom_param_ranges.get(model_type)
        
        if param_ranges is None:
            param_ranges = self.optimizer.get_param_ranges(self.model_type_combo.currentText())
        
        # v1.5.0: Filter to show only selected parameters for optimization
        algo_key = model_type.lower().replace(' ', '_')
        selected = self.selected_opt_params.get(algo_key)
        if selected:
            param_ranges = {
                k: v for k, v in param_ranges.items()
                if selected.get(k, True)
            }
        
        dialog = ParameterRangeDialog(param_ranges, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.custom_param_ranges[model_type] = dialog.get_custom_ranges()
            self.reset_ranges_btn.setEnabled(True)
            self.ranges_label.setText("Using custom ranges")
    
    def _reset_parameter_ranges(self):
        """Reset parameter ranges to defaults."""
        model_type = self.model_type_combo.currentText()
        if model_type in self.custom_param_ranges:
            del self.custom_param_ranges[model_type]
        self.reset_ranges_btn.setEnabled(False)
        self.ranges_label.setText("Using default ranges")
    
    def _start_optimization(self):
        """Start optimization."""
        data_manager = _safe_get_parent_attr(self, 'data_tab.get_data_manager()')
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        # v1.5.0: Always use internal test split; external validation is separate
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=False)
        
        if X_train is None or y_train is None:
            QMessageBox.warning(self, "Warning", "No training data available!")
            return
        
        # Get parameters
        model_type = self.model_type_combo.currentText()
        algorithm = model_type
        n_trials = self.trials_spin.value()
        cv_folds = self.cv_folds_spin.value()
        timeout = self.timeout_spin.value()
        pruning = self.pruning_check.isChecked()
        problem_type = self.problem_type_combo.currentText().lower()
        
        # v2.0: Search strategy
        is_grid = self.strategy_combo.currentText() == "Grid Search"
        search_strategy = "grid" if is_grid else "bayesian"
        resolution_map = {'Coarse (3 pts)': 'coarse', 'Medium (5 pts)': 'medium', 'Fine (7 pts)': 'fine'}
        grid_resolution = resolution_map.get(self.grid_resolution_combo.currentText(), 'medium')
        
        # Get parameter ranges
        param_ranges = self.custom_param_ranges.get(model_type)
        if param_ranges is None:
            param_ranges = self.optimizer.get_param_ranges(algorithm)

        # v2.0: Get default_params to preserve fixed settings (n_jobs, etc.)
        default_params = None
        try:
            model_info = self.model_registry.get_model(model_type)
            default_params = model_info.default_params.copy()
        except Exception:
            pass  # Model not in registry — optimizer will use defaults

        # v1.5.0: Get selected parameters (which ones to optimize vs use defaults)
        algo_key = algorithm.lower().replace(' ', '_')
        selected_params = self.selected_opt_params.get(algo_key)
        if selected_params:
            n_checked = sum(1 for v in selected_params.values() if v)
            if n_checked == 0:
                QMessageBox.warning(self, "Warning",
                    "No parameters selected for optimization. "
                    "Please use 'Select Parameters to Optimize' to choose at least one parameter.")
                return

        # Start optimization
        self.optimize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_text.clear()
        self.results_table.setRowCount(0)

        search_strategy = self.strategy_combo.currentText()
        if search_strategy == "Grid Search":
            # GridSearchCV has no per-iteration callback → indeterminate bar
            self.progress_bar.setRange(0, 0)
            self.progress_text.append(
                f"Starting Grid Search optimization...\n"
                f"Strategy: {self.algo_combo.currentText()} | "
                f"CV: {self.cv_folds_spin.value()}-fold | "
                f"Resolution: {self.grid_resolution_combo.currentText()}"
            )
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_text.append(
                f"Starting Bayesian optimization ({n_trials} trials)...\n"
                f"Strategy: {self.algo_combo.currentText()} | "
                f"CV: {self.cv_folds_spin.value()}-fold"
            )

        # v2.0: Configure the shared CV widget with the dataset + problem type +
        # target (for point-mass detection), build the full CV config, resolve
        # groups, and validate via the shared cv_factory rules (BLOCK aborts,
        # WARN asks for confirmation).
        self.cv_widget.set_dataframe(data_manager.data)
        self.cv_widget.set_problem_type(problem_type)
        try:
            self.cv_widget.set_target_values(data_manager.data[data_manager.target_column].values)
        except Exception:
            self.cv_widget.set_target_values(None)
        cv_config = self._get_cv_config()
        # v2.0: sync the CV fold shuffling with the user's Random Seed so the
        # seed controls the model init, the sampler, AND the CV folds.
        cv_config.random_state = self.random_seed_spin.value()
        # prepare_data(use_validation=False) uses its default test_size/seed.
        _split_test_size, _split_seed = 0.2, 42
        try:
            cv_groups, group_info = self._resolve_cv_groups(
                cv_config, data_manager,
                test_size=_split_test_size, random_state=_split_seed,
            )
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
                QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        # v2.0: Pass the unfitted preprocessor so the optimizer builds a Pipeline
        # that eliminates data leakage by refitting the scaler inside each CV fold.
        preprocessor = data_manager.get_preprocessor()
        self.optimization_thread = OptimizationThread(
            self.optimizer, model_type, algorithm, param_ranges, n_trials,
            X_train, y_train, X_test, y_test, problem_type, cv_folds,
            timeout=timeout if timeout > 0 else None,
            pruning=pruning,
            random_state=self.random_seed_spin.value(),
            selected_params=selected_params,
            default_params=default_params,
            search_strategy=search_strategy,
            grid_resolution=grid_resolution,
            cv_strategy=cv_config.strategy,
            cv_config=cv_config,
            cv_groups=cv_groups,
            preprocessor=preprocessor,
        )
        self.optimization_thread.progress.connect(self._on_optimization_progress)
        self.optimization_thread.finished.connect(self._on_optimization_finished)
        self.optimization_thread.error.connect(self._on_optimization_error)
        self.optimization_thread.start()
    
    def _on_optimization_progress(self, trial_num, score, params):
        """Handle optimization progress.

        v2.0 FIX: Force immediate UI repaint with processEvents() so the
        Training Progress console updates live during long optimizations.
        Grid Search runs with an indeterminate bar because GridSearchCV.fit()
        has no per-iteration callback.
        """
        from PyQt6.QtWidgets import QApplication

        is_grid = self.strategy_combo.currentText() == "Grid Search"
        if is_grid:
            # GridSearch emits progress only at completion
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(100)
            self.progress_text.append(
                f"Grid Search complete: Score={score:.4f}, Params={params}"
            )
        else:
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(int(100 * trial_num / self.trials_spin.value()))
            self.progress_text.append(
                f"Trial {trial_num}: Score={score:.4f}, Params={params}"
            )
        # Force immediate repaint so the user sees updates as they arrive
        QApplication.processEvents()
    
    def _on_optimization_finished(self, result):
        """Handle optimization completion."""
        self.optimize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if result is None:
            QMessageBox.warning(self, "Warning", "Optimization failed!")
            return
        
        # Store result and enable "Train with Best" button
        self.last_opt_result = result
        self.train_best_btn.setEnabled(True)
        self.train_best_btn.setToolTip(
            f"Best score: {result.best_score:.4f} | Click to train with these parameters"
        )
        
        # Store in list for results_tab (list of tuples format)
        # v2.0 FIX: Keep ALL optimizations (don't replace). The previous
        # code deleted earlier optimizations of the same algorithm, causing
        # the export to miss optimization details for models trained from
        # earlier optimizations (e.g., "XGBoost_Optimized" from 5-fold opt
        # lost its details when user later ran a 3-fold opt of XGBoost).
        model_type = self.model_type_combo.currentText()
        self.optimization_results.append((model_type, result))
        
        # Display results
        self._display_results(result)
        
        QMessageBox.information(
            self, "Success",
            f"Optimization complete!\n"
            f"Best score: {result.best_score:.4f}\n"
            f"Trials: {result.n_trials}\n"
            f"\nClick 'Train with Best Parameters' to train the model."
        )

    def _train_with_best_params(self):
        """Train a model using the best hyperparameters from last optimization."""
        if self.last_opt_result is None:
            QMessageBox.warning(self, "Warning", "No optimization result available. Run optimization first.")
            return
        
        data_manager = _safe_get_parent_attr(self, 'data_tab.get_data_manager()')
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
        if trainer is None:
            QMessageBox.warning(self, "Warning", "Trainer not available!")
            return
        
        # Get the pre-trained best model from optimization result
        best_model = self.last_opt_result.best_model
        if best_model is None:
            QMessageBox.warning(self, "Warning", "No best model found in optimization result.")
            return
        
        model_type = self.model_type_combo.currentText()
        problem_type = self.problem_type_combo.currentText().lower()
        cv_folds = self.cv_folds_spin.value()
        
        # v1.5.0: Get data with internal split + external validation separately
        X_train, X_test, y_train, y_test = data_manager.prepare_data(
            use_validation=False
        )
        # Get external validation set if loaded
        val_arrays = data_manager.get_validation_arrays()
        X_ext_val, y_ext_val = (val_arrays[0], val_arrays[1]) if val_arrays else (None, None)
        
        # Get feature names
        feature_names = data_manager.data.feature_names if hasattr(data_manager.data, 'feature_names') else None
        
        # Show progress bar and disable button
        self.train_best_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_text.append("Training with best parameters...")
        
        # v2.0: Train the best model with the SAME CV configuration that was
        # used during optimization (rebuilt from the stored cv_config_dict).
        # Group labels are re-resolved against the current (identical) split.
        cv_config = CVConfig.from_dict(getattr(self.last_opt_result, 'cv_config_dict', None))
        try:
            cv_groups, _gi = self._resolve_cv_groups(
                cv_config, data_manager, test_size=0.2, random_state=42,
            )
        except GroupingError as e:
            self.train_best_btn.setEnabled(True)
            self.progress_bar.setVisible(False)
            QMessageBox.warning(self, "Grouping Error", str(e))
            return

        # Start training in thread (non-blocking)
        self.train_best_thread = TrainBestThread(
            trainer=trainer,
            model=best_model,
            model_name=f"{model_type}_Optimized",
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            problem_type=problem_type,
            cv_folds=cv_folds,
            feature_names=feature_names,
            X_external_val=X_ext_val,
            y_external_val=y_ext_val,
            cv_strategy=cv_config.strategy,
            cv_config=cv_config,
            cv_groups=cv_groups,
        )
        self.train_best_thread.progress.connect(self._on_train_best_progress)
        self.train_best_thread.finished.connect(self._on_train_best_finished)
        self.train_best_thread.error.connect(self._on_train_best_error)
        self.train_best_thread.start()
    
    def _on_train_best_progress(self, msg, progress):
        """Update progress bar during Train with Best."""
        self.progress_bar.setValue(int(progress * 100))
        self.progress_text.append(msg)
    
    def _on_train_best_finished(self, result):
        """Handle completion of Train with Best."""
        self.progress_bar.setVisible(False)
        self.train_best_btn.setEnabled(True)
        self.progress_text.append("Training complete!")
        
        # v2.0: Build detailed message via the shared CV formatter (handles
        # kfold / stratified / group / loo / logo with a single code path).
        cv_metrics_dict = result.cv_metrics if result.cv_metrics else {}
        msg = f"Model trained with best parameters!\n\n"
        msg += f"Name: {result.model_name}\n"
        cv_text = format_cv_details(
            cv_metrics_dict,
            cv_strategy=getattr(result, 'cv_strategy', 'kfold'),
            cv_config_dict=getattr(result, 'cv_config_dict', None),
            fmt='text',
        )
        if cv_text:
            msg += "\n" + cv_text
        else:
            msg += f"CV Score: N/A\n"
        msg += f"\nCheck Results tab for full details."

        QMessageBox.information(self, "Success", msg)
    
    def _on_train_best_error(self, error_msg):
        """Handle Train with Best error."""
        self.progress_bar.setVisible(False)
        self.train_best_btn.setEnabled(True)
        self.progress_text.append("Training failed.")
        QMessageBox.critical(self, "Error", f"Failed to train with best parameters:\n{error_msg}")
    def _on_optimization_error(self, error_msg):
        """Handle optimization error."""
        self.optimize_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Optimization failed:\n{error_msg}")
    
    def _display_results(self, result):
        """Display optimization results."""
        # Display best trial details
        details = f"Best Score: {result.best_score:.4f}\n"
        details += f"Number of Trials: {result.n_trials}\n"
        details += f"Optimization Time: {result.optimization_time:.2f}s\n"
        if result.cv_metrics and result.cv_metrics.get('test_score'):
            cv_scores = result.cv_metrics['test_score']
            details += f"CV Score: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n"
        
        # v1.5.0: Merge best_params with model defaults so ALL parameters are shown.
        # Mark with (*) only the parameters the user ACTUALLY selected for optimization.
        from ..core.model_registry import ModelRegistry
        registry = ModelRegistry()
        algorithm = self.model_type_combo.currentText().lower().replace(' ', '_')
        try:
            model_info = registry.get_model(algorithm)
            full_params = dict(model_info.default_params)  # Start with all defaults
            full_params.update(result.best_params)  # Override with optimized values
            # v1.5.0: Only mark as (*) the params the user selected for optimization
            selected = self.selected_opt_params.get(algorithm, {})
            optimizable = {k for k, v in selected.items() if v} if selected else set()
            if not optimizable:
                # Fallback: if no selection was made, all params from best_params are marked
                optimizable = set(result.best_params.keys())
            # Remove internal/fixed params that aren't tunable
            for p in ('random_state', 'n_jobs', 'verbose', 'random_seed', 'tree_method'):
                full_params.pop(p, None)
                optimizable.discard(p)
        except Exception:
            full_params = result.best_params
            optimizable = set(result.best_params.keys())
        
        details += "\nBest Parameters:\n"
        for param, value in sorted(full_params.items()):
            marker = " *" if param in optimizable else "  "
            details += f"{marker}{param}: {value}\n"
        details += "\n(*) = parameter included in Bayesian optimization\n"
        self.details_text.setPlainText(details)
        
        # v1.3.0: Display top trials in table (clean 4-column format)
        if result.all_trials is not None:
            trials_df = result.all_trials
            trials_df = trials_df.sort_values('value', ascending=False)
            
            display_trials = trials_df.head(10)
            self.results_table.setRowCount(len(display_trials))
            
            for i, (_, trial) in enumerate(display_trials.iterrows()):
                self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.results_table.setItem(i, 1, QTableWidgetItem(f"{trial['value']:.4f}"))
                
                params_str = str(trial.get('params', {}))
                if len(params_str) > 80:
                    params_str = params_str[:80] + "..."
                self.results_table.setItem(i, 2, QTableWidgetItem(params_str))
                
                self.results_table.setItem(i, 3, QTableWidgetItem(str(trial.get('number', ''))))
        
        # v2.0: CV breakdown via the shared formatter (single code path for
        # kfold / stratified / group / loo / logo, incl. Test Size + warnings).
        cv_metrics_dict = result.cv_metrics if result.cv_metrics else {}
        cv_text = format_cv_details(
            cv_metrics_dict,
            cv_strategy=getattr(result, 'cv_strategy', 'kfold'),
            cv_config_dict=getattr(result, 'cv_config_dict', None),
            fmt='text',
        )
        if cv_text:
            details += "\n\n" + cv_text

        self.details_text.setPlainText(details)
    
    def get_optimization_result(self, model_name):
        """Get optimization result for a model."""
        for name, res in self.optimization_results:
            if name == model_name:
                return res
        return None
    
    def get_all_optimization_results(self):
        """Get all optimization results."""
        return self.optimization_results
