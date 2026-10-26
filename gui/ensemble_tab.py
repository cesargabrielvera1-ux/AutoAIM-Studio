"""Ensemble tab for creating and training ensemble models."""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QTextEdit, QMessageBox, QProgressBar, QCheckBox, QSplitter,
    QListWidget, QListWidgetItem, QDialog, QDoubleSpinBox, QGridLayout,
    QDialogButtonBox, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.ensemble_trainer import EnsembleTrainer, EnsembleResult
from ..core.cv_factory import CVConfig, validate_cv_config, has_blocking_issue
from ..core.grouping import GroupingError
from .cv_config_widget import CVConfigWidget


class EnsembleOptimizationThread(QThread):
    """Thread for optimizing ensemble weights."""
    progress = pyqtSignal(int, float, dict)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, ensemble_trainer, X, y, model_names, n_trials=50, cv_folds=5,
                 random_state=42, cv_config=None, cv_groups=None):
        super().__init__()
        self.ensemble_trainer = ensemble_trainer
        self.X = X
        self.y = y
        self.model_names = model_names
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
        # v2.0: full CV config + resolved group labels
        self.cv_config = cv_config
        self.cv_groups = cv_groups

    def run(self):
        try:
            def progress_callback(trial_num, score, weights):
                self.progress.emit(trial_num, score, weights)

            result = self.ensemble_trainer.optimize_weights(
                self.X, self.y,
                model_names=self.model_names,
                n_trials=self.n_trials,
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                progress_callback=progress_callback,
                cv_config=self.cv_config,
                cv_groups=self.cv_groups,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class EnsembleTrainingThread(QThread):
    """Thread for training ensemble."""
    progress = pyqtSignal(str, float)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, ensemble_trainer, ensemble_type, model_names, X_train, y_train,
                 X_test, y_test, cv_folds=5, random_state=42, weights=None,
                 X_external_val=None, y_external_val=None,
                 cv_config=None, cv_groups=None,
                 # v2.0: preprocessor for sklearn Pipeline (eliminates CV data leakage)
                 preprocessor=None):
        super().__init__()
        self.ensemble_trainer = ensemble_trainer
        self.ensemble_type = ensemble_type
        self.model_names = model_names
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.weights = weights
        # v1.5.0: External validation for dual metrics
        self.X_external_val = X_external_val
        self.y_external_val = y_external_val
        # v2.0: full CV config + resolved group labels
        self.cv_config = cv_config
        self.cv_groups = cv_groups
        # v2.0: preprocessor passed through for Pipeline construction
        self.preprocessor = preprocessor

    def run(self):
        try:
            def progress_callback(msg, progress):
                self.progress.emit(msg, progress)

            result = self.ensemble_trainer.train_ensemble(
                self.ensemble_type,
                self.model_names,
                self.X_train, self.y_train,
                self.X_test, self.y_test,
                cv_folds=self.cv_folds,
                random_state=self.random_state,
                progress_callback=progress_callback,
                weights=self.weights,
                X_external_val=self.X_external_val,
                y_external_val=self.y_external_val,
                cv_config=self.cv_config,
                cv_groups=self.cv_groups,
                preprocessor=self.preprocessor,
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class EnsembleTab(QWidget):
    """Tab for ensemble model training."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent  # v2.0 FIX: self._main_window overwrites PyQt parent()
        self.ensemble_trainer = None
        self._last_optimized_weights = None  # Store optimized weights for apply
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel - Configuration
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        
        # Available models
        models_group = QGroupBox("Available Models")
        models_layout = QVBoxLayout(models_group)
        
        self.models_list = QListWidget()
        self.models_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
        models_layout.addWidget(self.models_list)
        
        left_layout.addWidget(models_group)
        
        # Refresh models button
        self.refresh_btn = QPushButton("Refresh Models List")
        self.refresh_btn.setToolTip("Update the list with models trained in the Training and Optimization tabs")
        self.refresh_btn.clicked.connect(self._refresh_models)
        left_layout.addWidget(self.refresh_btn)
        
        # Ensemble Configuration
        config_group = QGroupBox("Ensemble Configuration")
        config_layout = QGridLayout(config_group)
        
        config_layout.addWidget(QLabel("Ensemble Type:"), 0, 0)
        self.ensemble_type_combo = QComboBox()
        self.ensemble_type_combo.addItems([
            "weighted_average", "stacking"
        ])
        self.ensemble_type_combo.setToolTip(
            "weighted_average: combines predictions using weighted sum. "
            "Faster, good when models are diverse. "
            "stacking: trains a meta-learner on model outputs. "
            "More complex but can capture interactions between models."
        )
        config_layout.addWidget(self.ensemble_type_combo, 0, 1)
        
        config_layout.addWidget(QLabel("Random Seed:"), 1, 0)
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setToolTip("Random seed for reproducible ensemble training")
        config_layout.addWidget(self.random_seed_spin, 1, 1)

        left_layout.addWidget(config_group)

        # v2.0: Cross-validation configuration (shared widget). Ensembles now
        # support the same five strategies as sklearn models and NNs.
        cv_group = QGroupBox("Cross-Validation (Training)")
        cv_group_layout = QVBoxLayout(cv_group)
        cv_group_layout.setContentsMargins(4, 4, 4, 4)
        self.cv_widget = CVConfigWidget()
        # Backward-compatible alias: code reading self.cv_folds_spin.value()
        # now reads the shared widget's folds spinner.
        self.cv_folds_spin = self.cv_widget.folds_spin
        cv_group_layout.addWidget(self.cv_widget)
        left_layout.addWidget(cv_group)
        
        # v1.3.0: Manual Weights Configuration
        manual_group = QGroupBox("Manual Weights (Optional)")
        manual_layout = QVBoxLayout(manual_group)
        
        manual_info = QLabel(
            "Set custom weights for each model. If empty, equal weights are used."
        )
        manual_info.setWordWrap(True)
        manual_layout.addWidget(manual_info)
        
        # Container for weight spinboxes (populated dynamically)
        self.manual_weights_widget = QWidget()
        self.manual_weights_layout = QGridLayout(self.manual_weights_widget)
        self.manual_weights_layout.setColumnStretch(1, 1)
        self.manual_weights_layout.addWidget(QLabel("<b>Model</b>"), 0, 0)
        self.manual_weights_layout.addWidget(QLabel("<b>Weight</b>"), 0, 1)
        manual_layout.addWidget(self.manual_weights_widget)
        
        self.refresh_weights_btn = QPushButton("Refresh from Selection")
        self.refresh_weights_btn.setToolTip("Update weight fields based on currently selected models")
        self.refresh_weights_btn.clicked.connect(self._refresh_manual_weights)
        manual_layout.addWidget(self.refresh_weights_btn)
        
        self.train_manual_btn = QPushButton("Train with Manual Weights")
        self.train_manual_btn.setToolTip(
            "Train the ensemble using weights you set above. "
            "If no weights are set, all models get equal weight (1/N)."
        )
        self.train_manual_btn.setStyleSheet("""
            QPushButton {
                background-color: #E65100;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        self.train_manual_btn.clicked.connect(self._train_with_manual_weights)
        manual_layout.addWidget(self.train_manual_btn)
        
        left_layout.addWidget(manual_group)
        
        # Ensemble Optimization
        opt_group = QGroupBox("Weight Optimization (Bayesian)")
        opt_layout = QGridLayout(opt_group)
        
        opt_layout.addWidget(QLabel("Optimization Trials:"), 0, 0)
        self.opt_trials_spin = QSpinBox()
        self.opt_trials_spin.setRange(10, 200)
        self.opt_trials_spin.setValue(50)
        self.opt_trials_spin.setSingleStep(10)
        self.opt_trials_spin.setToolTip(
            "Number of Bayesian optimization trials to find optimal weights. "
            "More trials = better weights but longer optimization."
        )
        opt_layout.addWidget(self.opt_trials_spin, 0, 1)
        
        opt_layout.addWidget(QLabel("CV Folds (Opt):"), 1, 0)
        self.opt_cv_folds_spin = QSpinBox()
        self.opt_cv_folds_spin.setRange(1, 10)
        self.opt_cv_folds_spin.setValue(5)
        self.opt_cv_folds_spin.setToolTip(
            "CV folds for evaluating weight combinations during optimization."
        )
        opt_layout.addWidget(self.opt_cv_folds_spin, 1, 1)
        
        opt_layout.addWidget(QLabel("Random Seed:"), 2, 0)
        self.opt_random_seed_spin = QSpinBox()
        self.opt_random_seed_spin.setRange(0, 9999)
        self.opt_random_seed_spin.setValue(42)
        self.opt_random_seed_spin.setToolTip("Random seed for reproducible weight optimization")
        opt_layout.addWidget(self.opt_random_seed_spin, 2, 1)
        
        left_layout.addWidget(opt_group)
        
        # v2.0: Search strategy info (Bayesian only for ensemble weights)
        strategy_label = QLabel("Strategy: Bayesian (Optuna)")
        strategy_label.setStyleSheet("color: #00bcd4; font-size: 8pt;")
        strategy_label.setToolTip(
            "Ensemble weight optimization uses Bayesian optimization only.\n"
            "Grid Search is not available because weight spaces are continuous\n"
            "with a sum-to-one constraint, which cannot be efficiently searched\n"
            "with standard grid enumeration."
        )
        left_layout.addWidget(strategy_label)

        # v2.0: Ensembles now support the full CV strategy set (see the
        # Cross-Validation section above). Show a short pointer.
        cv_label = QLabel("CV: see Cross-Validation section")
        cv_label.setStyleSheet("color: #00bcd4; font-size: 8pt;")
        cv_label.setToolTip(
            "Ensembles now support the same cross-validation strategies as\n"
            "sklearn models: K-Fold, Stratified (quantiles, with optional reserved\n"
            "bin), Group K-Fold, LOO, and LOGO. Configure them in the\n"
            "Cross-Validation section. Ensembles are most reliable with N ≥ 100."
        )
        left_layout.addWidget(cv_label)

        # Buttons
        self.optimize_weights_btn = QPushButton("Optimize Weights")
        self.optimize_weights_btn.setToolTip(
            "Use Bayesian optimization (Optuna) to find the best weights "
            "for each model that maximize the ensemble's CV score."
        )
        self.optimize_weights_btn.setStyleSheet(""""
            QPushButton {
                background-color: #1565C0;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        self.optimize_weights_btn.clicked.connect(self._optimize_weights)
        left_layout.addWidget(self.optimize_weights_btn)
        
        self.apply_weights_btn = QPushButton("Apply Optimized Weights")
        self.apply_weights_btn.setStyleSheet("""
            QPushButton {
                background-color: #6A1B9A;
                color: white;
                font-weight: bold;
                padding: 8px;
            }
        """)
        self.apply_weights_btn.clicked.connect(self._apply_optimized_weights)
        self.apply_weights_btn.setEnabled(False)
        self.apply_weights_btn.setToolTip(
            "After optimizing weights, click this to apply them before training"
        )
        left_layout.addWidget(self.apply_weights_btn)
        
        self.train_btn = QPushButton("Train Ensemble")
        self.train_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
            }
        """)
        self.train_btn.clicked.connect(self._train_ensemble)
        left_layout.addWidget(self.train_btn)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        self.progress_text = QTextEdit()
        self.progress_text.setReadOnly(True)
        self.progress_text.setMaximumHeight(120)
        left_layout.addWidget(self.progress_text)
        
        left_layout.addStretch()

        # v2.0: Make the left (control) panel scrollable so all options —
        # model list, config, manual weights, Cross-Validation, weight
        # optimization and action buttons — stay reachable on typical laptop
        # and desktop screens. Nothing is ever cut off (same pattern as the
        # Neural Network tab).
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)
        
        # Right panel - Results
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        
        # Ensemble Results
        results_group = QGroupBox("Ensemble Results")
        results_layout = QVBoxLayout(results_group)
        
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)
        self.results_table.setHorizontalHeaderLabels([
            "Model", "Weight", "Test Score", "CV Score", "Problem Type", "Selected"
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        results_layout.addWidget(self.results_table)
        
        right_layout.addWidget(results_group)
        
        # Details
        details_group = QGroupBox("Details")
        details_layout = QVBoxLayout(details_group)
        self.details_text = QTextEdit()
        self.details_text.setReadOnly(True)
        details_layout.addWidget(self.details_text)
        
        right_layout.addWidget(details_group)
        
        splitter.addWidget(right_widget)
        splitter.setSizes([400, 600])
        layout.addWidget(splitter)
    
    def _refresh_models(self):
        """Refresh list of available models."""
        self.models_list.clear()
        
        if self._main_window is None:
            return
        
        # 1. Add trained models
        if hasattr(self._main_window, 'training_tab'):
            try:
                trainer = self._main_window.training_tab.get_trainer()
                if trainer and hasattr(trainer, 'results') and trainer.results:
                    for name, result in trainer.results.items():
                        if hasattr(result, 'model') and result.model is not None:
                            item = QListWidgetItem(f"{name} (Trained)")
                            item.setData(Qt.ItemDataRole.UserRole, ('trained', name, result))
                            self.models_list.addItem(item)
            except Exception as e:
                print(f"Error loading trained models: {e}")
        
        # 2. Add optimized models
        if hasattr(self._main_window, 'optimization_tab'):
            try:
                opt_tab = self._main_window.optimization_tab
                if hasattr(opt_tab, 'optimization_results') and opt_tab.optimization_results:
                    for model_name, opt_result in opt_tab.optimization_results:
                        if hasattr(opt_result, 'best_model') and opt_result.best_model is not None:
                            item = QListWidgetItem(f"{model_name} (Optimized)")
                            item.setData(Qt.ItemDataRole.UserRole, ('optimized', model_name, opt_result))
                            self.models_list.addItem(item)
            except Exception as e:
                print(f"Error loading optimized models: {e}")
        
        # 3. Add neural network models
        if hasattr(self._main_window, 'nn_tab'):
            try:
                nn_tab = self._main_window.nn_tab
                if hasattr(nn_tab, 'nn_training_results') and nn_tab.nn_training_results:
                    for nn_name, nn_result in nn_tab.nn_training_results:
                        if hasattr(nn_result, 'model') and nn_result.model is not None:
                            item = QListWidgetItem(f"{nn_name} (Neural Network)")
                            item.setData(Qt.ItemDataRole.UserRole, ('nn', nn_name, nn_result))
                            self.models_list.addItem(item)
            except Exception as e:
                print(f"Error loading NN models: {e}")
    
    def _refresh_manual_weights(self):
        """Populate manual weight fields from currently selected models."""
        # Clear existing weight widgets (keep header row 0)
        while self.manual_weights_layout.count() > 2:
            item = self.manual_weights_layout.takeAt(2)
            if item.widget():
                item.widget().deleteLater()
        
        selected_models, _ = self._get_selected_models()
        
        if not selected_models:
            no_models = QLabel("(No models selected. Select models above and click Refresh.)")
            self.manual_weights_layout.addWidget(no_models, 1, 0, 1, 2)
            return
        
        # Equal weights by default
        default_weight = 1.0 / len(selected_models) if selected_models else 1.0
        
        for i, model_name in enumerate(selected_models):
            row = i + 1
            self.manual_weights_layout.addWidget(QLabel(f"{model_name}:"), row, 0)
            
            spin = QDoubleSpinBox()
            spin.setRange(0.0, 10.0)
            spin.setValue(round(default_weight, 4))
            spin.setDecimals(4)
            spin.setSingleStep(0.05)
            # Store model name as property for later retrieval
            spin.setProperty('model_name', model_name)
            self.manual_weights_layout.addWidget(spin, row, 1)
    
    def _get_manual_weights(self):
        """Collect manually entered weights. Returns None if no weights set."""
        weights = {}
        has_any = False
        
        for i in range(self.manual_weights_layout.count()):
            item = self.manual_weights_layout.itemAt(i)
            widget = item.widget()
            if isinstance(widget, QDoubleSpinBox):
                model_name = widget.property('model_name')
                if model_name:
                    weights[model_name] = widget.value()
                    if widget.value() != 0.0:
                        has_any = True
        
        return weights if has_any else None
    
    def _train_with_manual_weights(self):
        """Train ensemble with manually configured weights."""
        if self._main_window is None or not hasattr(self._main_window, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
        trainer = self._main_window.training_tab.get_trainer()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        selected_models, _ = self._get_selected_models()
        if len(selected_models) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 models!")
            return
        
        weights = self._get_manual_weights()
        
        try:
            # v1.5.0: Always use internal test split; external validation is separate
            X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=False)
            val_arrays = data_manager.get_validation_arrays()
            X_ext_val, y_ext_val = (val_arrays[0], val_arrays[1]) if val_arrays else (None, None)
            
            ensemble_type = self.ensemble_type_combo.currentText()
            cv_folds = self.cv_folds_spin.value()

            # v2.0: validate CV strategy + resolve groups (shared rules). The
            # CV fold shuffling is synced to the user's Random Seed.
            _pt = self._get_problem_type(data_manager)
            _cv = self._prepare_cv_config(data_manager, X_train, y_train, _pt,
                                          random_state=self.random_seed_spin.value())
            if _cv is None:
                return
            cv_config, cv_groups = _cv

            self.train_manual_btn.setEnabled(False)
            self.train_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_text.clear()

            if weights:
                self.progress_text.append("Training ensemble with manual weights:")
                for name, w in weights.items():
                    self.progress_text.append(f"  {name}: {w:.4f}")
            else:
                self.progress_text.append("Training ensemble with equal weights (no manual weights set)")

            # v2.0 FIX: Preserve existing ensemble results by reusing the
            # ensemble_trainer instance. Creating a new EnsembleTrainer on
            # every train call wiped the results dict, so only the most recent
            # ensemble appeared in the Results tab.
            if self.ensemble_trainer is None:
                self.ensemble_trainer = EnsembleTrainer(trainer)
            else:
                # Refresh the wrapped trainer reference but keep the results dict
                self.ensemble_trainer.trainer = trainer

            # v2.0: Pass preprocessor for API consistency (ensemble reuses
            # already-trained base models, so it is not used directly).
            preprocessor = data_manager.get_preprocessor()
            self.training_thread = EnsembleTrainingThread(
                self.ensemble_trainer,
                ensemble_type,
                selected_models,
                X_train, y_train,
                X_test, y_test,
                cv_folds=cv_folds,
                random_state=self.random_seed_spin.value(),
                weights=weights,
                X_external_val=X_ext_val,
                y_external_val=y_ext_val,
                cv_config=cv_config,
                cv_groups=cv_groups,
                preprocessor=preprocessor,
            )
            self.training_thread.progress.connect(self._on_training_progress)
            self.training_thread.finished.connect(self._on_training_finished)
            self.training_thread.error.connect(self._on_training_error)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start manual ensemble training:\n{str(e)}")
            self.train_manual_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
    
    def _get_problem_type(self, data_manager) -> str:
        """Return 'classification' or 'regression' based on the loaded dataset."""
        return 'classification' if getattr(data_manager, '_is_classification', False) else 'regression'

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

    def _prepare_cv_config(self, data_manager, X_train, y_train, problem_type,
                           n_folds_override=None, random_state=None):
        """Set up the shared CV widget, validate the strategy, and resolve groups.

        Returns (cv_config, cv_groups), or None if the user was blocked or
        declined a warning. Centralizes the CV validation for both the regular
        and the manual-weights ensemble training paths. ``n_folds_override`` lets
        the weight-optimization path use its own fold count while sharing the
        same strategy/groups. ``random_state`` syncs the CV fold shuffling with
        the user's chosen seed (so the seed controls folds too, not just models).
        """
        self.cv_widget.set_dataframe(data_manager.data)
        self.cv_widget.set_problem_type(problem_type)
        try:
            self.cv_widget.set_target_values(data_manager.data[data_manager.target_column].values)
        except Exception:
            self.cv_widget.set_target_values(None)
        cv_config = self.cv_widget.get_config()
        if n_folds_override is not None:
            cv_config.n_folds = n_folds_override
        if random_state is not None:
            cv_config.random_state = random_state
        # prepare_data(use_validation=False) uses default test_size=0.2, seed=42.
        try:
            cv_groups, group_info = self._resolve_cv_groups(
                cv_config, data_manager, test_size=0.2, random_state=42)
        except GroupingError as e:
            QMessageBox.warning(self, "Grouping Error", str(e))
            return None
        _gi = group_info if cv_config.strategy in ('group', 'logo') else None
        _issues = validate_cv_config(cv_config, len(X_train),
                                     problem_type=problem_type, group_info=_gi,
                                     y=y_train)
        if has_blocking_issue(_issues):
            QMessageBox.warning(self, "Cross-Validation Not Available",
                                "\n\n".join(i['message'] for i in _issues if i['severity'] == 'block'))
            return None
        _warns = [i['message'] for i in _issues if i['severity'] == 'warn']
        if _warns:
            reply = QMessageBox.question(
                self, "Cross-Validation Warning",
                "\n\n".join(_warns) + "\n\nDo you want to proceed anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No)
            if reply != QMessageBox.StandardButton.Yes:
                return None
        return cv_config, cv_groups

    def _get_selected_models(self):
        """Get list of selected model names and their source info."""
        selected_names = []
        selected_data = []
        for item in self.models_list.selectedItems():
            data = item.data(Qt.ItemDataRole.UserRole)
            if data:
                source_type, name, result = data
                selected_names.append(name)
                selected_data.append(data)
            else:
                selected_names.append(item.text())
        return selected_names, selected_data
    
    def _train_ensemble(self):
        """Train the ensemble."""
        if self._main_window is None or not hasattr(self._main_window, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
        trainer = self._main_window.training_tab.get_trainer()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        selected_models, selected_data = self._get_selected_models()
        if len(selected_models) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 models!")
            return
        
        try:
            # v1.5.0: Always use internal test split; external validation is separate
            X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=False)
            val_arrays = data_manager.get_validation_arrays()
            X_ext_val, y_ext_val = (val_arrays[0], val_arrays[1]) if val_arrays else (None, None)
            
            ensemble_type = self.ensemble_type_combo.currentText()
            cv_folds = self.cv_folds_spin.value()

            # v2.0: validate CV strategy + resolve groups (shared rules). The
            # CV fold shuffling is synced to the user's Random Seed.
            _pt = self._get_problem_type(data_manager)
            _cv = self._prepare_cv_config(data_manager, X_train, y_train, _pt,
                                          random_state=self.random_seed_spin.value())
            if _cv is None:
                return
            cv_config, cv_groups = _cv

            # v2.0 FIX: Preserve existing ensemble results by reusing the
            # ensemble_trainer instance. Creating a new EnsembleTrainer on
            # every train call wiped the results dict, so only the most recent
            # ensemble appeared in the Results tab.
            if self.ensemble_trainer is None:
                self.ensemble_trainer = EnsembleTrainer(trainer)
            else:
                # Refresh the wrapped trainer reference but keep the results dict
                self.ensemble_trainer.trainer = trainer

            self.train_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_text.clear()

            # v2.0: Pass preprocessor for API consistency.
            preprocessor = data_manager.get_preprocessor()
            self.training_thread = EnsembleTrainingThread(
                self.ensemble_trainer,
                ensemble_type,
                selected_models,
                X_train, y_train,
                X_test, y_test,
                cv_folds=cv_folds,
                random_state=self.random_seed_spin.value(),
                X_external_val=X_ext_val,
                y_external_val=y_ext_val,
                cv_config=cv_config,
                cv_groups=cv_groups,
                preprocessor=preprocessor,
            )
            self.training_thread.progress.connect(self._on_training_progress)
            self.training_thread.finished.connect(self._on_training_finished)
            self.training_thread.error.connect(self._on_training_error)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start training:\n{str(e)}")
            self.train_btn.setEnabled(True)
    
    def _optimize_weights(self):
        """Optimize ensemble weights using Bayesian optimization."""
        if self._main_window is None or not hasattr(self._main_window, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
        trainer = self._main_window.training_tab.get_trainer()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        selected_models, selected_data = self._get_selected_models()
        if len(selected_models) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 models!")
            return
        
        try:
            # v1.5.0: Use internal test split for weight optimization
            X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=False)
            
            n_trials = self.opt_trials_spin.value()
            cv_folds = self.opt_cv_folds_spin.value()

            # v2.0: weight optimization shares the training CV strategy/groups,
            # but uses its own fold count (opt_cv_folds_spin).
            _pt = self._get_problem_type(data_manager)
            _cv = self._prepare_cv_config(data_manager, X_train, y_train, _pt,
                                          n_folds_override=cv_folds,
                                          random_state=self.opt_random_seed_spin.value())
            if _cv is None:
                return
            cv_config, cv_groups = _cv

            # v2.0 FIX: Preserve existing ensemble results by reusing the
            # ensemble_trainer instance. Creating a new EnsembleTrainer on
            # every optimize call wiped the results dict, so only the most recent
            # ensemble appeared in the Results tab.
            if self.ensemble_trainer is None:
                self.ensemble_trainer = EnsembleTrainer(trainer)
            else:
                # Refresh the wrapped trainer reference but keep the results dict
                self.ensemble_trainer.trainer = trainer

            self.optimize_weights_btn.setEnabled(False)
            self.train_btn.setEnabled(True)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_text.clear()
            self.progress_text.append("Starting weight optimization...")

            self.opt_thread = EnsembleOptimizationThread(
                self.ensemble_trainer,
                X_train, y_train,
                model_names=selected_models,
                n_trials=n_trials,
                cv_folds=cv_folds,
                random_state=self.opt_random_seed_spin.value(),
                cv_config=cv_config,
                cv_groups=cv_groups,
            )
            self.opt_thread.progress.connect(self._on_optimization_progress)
            self.opt_thread.finished.connect(self._on_optimization_finished)
            self.opt_thread.error.connect(self._on_optimization_error)
            self.opt_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start optimization:\n{str(e)}")
            self.optimize_weights_btn.setEnabled(True)
    
    def _apply_optimized_weights(self):
        """Apply optimized weights to be used during ensemble training."""
        if self._last_optimized_weights is None:
            QMessageBox.warning(
                self, "Warning",
                "No optimized weights available. Run 'Optimize Weights' first."
            )
            return
        
        # Build a display of the weights
        weight_text = "Optimized weights to be applied:\n"
        for name, weight in self._last_optimized_weights.items():
            weight_text += f"  {name}: {weight:.4f}\n"
        
        self.progress_text.append(weight_text)
        QMessageBox.information(
            self, "Weights Applied",
            "Optimized weights are ready to use.\n"
            "Now click 'Train Ensemble' to train with these weights."
        )
    
    def _on_training_progress(self, msg, progress):
        """Handle training progress."""
        self.progress_bar.setValue(int(progress * 100))
        self.progress_text.append(msg)
    
    def _on_training_finished(self, result):
        """Handle training completion."""
        self.train_btn.setEnabled(True)
        self.train_manual_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # v1.3.0: Ensemble results are stored in ensemble_trainer.results ONLY.
        # Do NOT duplicate in trainer.results to avoid duplicated entries.
        self._display_results(result)
        
        score_msg = ""
        if result.metrics:
            if result.problem_type == 'regression':
                r2 = result.metrics.get('r2', 'N/A')
                score_msg = f"R2: {r2:.4f}" if isinstance(r2, (int, float)) else f"R2: {r2}"
            else:
                acc = result.metrics.get('accuracy', 'N/A')
                score_msg = f"Accuracy: {acc:.4f}" if isinstance(acc, (int, float)) else f"Accuracy: {acc}"
        
        QMessageBox.information(
            self, "Success",
            f"Ensemble training complete!\n{score_msg}"
        )
    
    def _on_training_error(self, error_msg):
        """Handle training error."""
        self.train_btn.setEnabled(True)
        self.optimize_weights_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Training failed:\n{error_msg}")
    
    def _on_optimization_progress(self, trial_num, score, weights):
        """Handle optimization progress."""
        self.progress_bar.setValue(int(100 * trial_num / self.opt_trials_spin.value()))
        self.progress_text.append(f"Trial {trial_num}: Score={score:.4f}")
    
    def _on_optimization_finished(self, result):
        """Handle optimization completion."""
        self.optimize_weights_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        if not isinstance(result, EnsembleResult):
            QMessageBox.warning(self, "Warning", f"Unexpected result type: {type(result)}")
            return
        
        # Store optimized weights for apply button
        if result.optimized_weights:
            self._last_optimized_weights = result.optimized_weights
            self.apply_weights_btn.setEnabled(True)
            self.apply_weights_btn.setToolTip(
                f"Best CV Score: {result.metrics.get('cv_r2_mean', 'N/A'):.4f}"
                if isinstance(result.metrics.get('cv_r2_mean'), (int, float))
                else "Optimized weights ready"
            )
        
        # v1.3.0: Results are already in ensemble_trainer.results via optimize_weights().
        # Do NOT duplicate in trainer.results to avoid duplicates in Results tab.
        self._display_results(result)
        
        # Build summary message
        best_score = result.metrics.get('cv_r2_mean', 'N/A')
        score_str = f"{best_score:.4f}" if isinstance(best_score, (int, float)) else str(best_score)
        
        weight_summary = "\n".join(
            f"  {name}: {weight:.4f}"
            for name, weight in result.weights.items()
        )
        
        QMessageBox.information(
            self, "Optimization Complete",
            f"Weight optimization complete!\n"
            f"Best CV Score: {score_str}\n"
            f"\nOptimized Weights:\n{weight_summary}\n\n"
            f"Click 'Apply Optimized Weights' then 'Train Ensemble' to use them."
        )
    
    def _on_optimization_error(self, error_msg):
        """Handle optimization error."""
        self.optimize_weights_btn.setEnabled(True)
        self.train_btn.setEnabled(True)
        self.progress_bar.setVisible(False)
        QMessageBox.critical(self, "Error", f"Optimization failed:\n{error_msg}")
    
    def _display_results(self, result):
        """Display ensemble results."""
        if not hasattr(result, 'model_names'):
            self.details_text.setPlainText(f"Invalid result: {type(result)}")
            return
        
        self.results_table.setRowCount(len(result.model_names))
        
        for i, model_name in enumerate(result.model_names):
            self.results_table.setItem(i, 0, QTableWidgetItem(model_name))
            
            weight = result.weights.get(model_name, 1.0)
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{weight:.4f}"))
            
            if result.problem_type == 'regression':
                test_score = result.metrics.get('r2', 'N/A')
                score_label = f"R2: {test_score:.4f}" if isinstance(test_score, (int, float)) else str(test_score)
            else:
                test_score = result.metrics.get('accuracy', 'N/A')
                score_label = f"Acc: {test_score:.4f}" if isinstance(test_score, (int, float)) else str(test_score)
            self.results_table.setItem(i, 2, QTableWidgetItem(score_label))
            
            # CV Score
            cv_scores = result.cv_scores if hasattr(result, 'cv_scores') else []
            if cv_scores:
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                self.results_table.setItem(i, 3, QTableWidgetItem(f"{cv_mean:.4f} ± {cv_std:.4f}"))
            else:
                self.results_table.setItem(i, 3, QTableWidgetItem("N/A"))
            
            self.results_table.setItem(i, 4, QTableWidgetItem(result.problem_type))
            
            checkbox = QCheckBox()
            checkbox.setChecked(True)
            self.results_table.setCellWidget(i, 5, checkbox)
        
        # Display details
        details = f"Ensemble Type: {result.ensemble_type}\n"
        details += f"Number of Models: {len(result.model_names)}\n"
        details += f"Training Time: {result.training_time:.2f}s\n"
        if result.cv_scores:
            details += f"CV Score: {np.mean(result.cv_scores):.4f} ± {np.std(result.cv_scores):.4f}\n"
        details += "\nMetrics:\n"
        for metric, value in result.metrics.items():
            details += f"  {metric}: {value:.4f}\n"
        details += "\nModel Weights:\n"
        for model_name, weight in result.weights.items():
            details += f"  {model_name}: {weight:.4f}\n"
        
        self.details_text.setPlainText(details)
    
    def get_ensemble_result(self):
        """Get the latest ensemble result."""
        if self.ensemble_trainer and self.ensemble_trainer.results:
            return list(self.ensemble_trainer.results.values())[-1]
        return None
    
    def get_ensemble_trainer(self):
        """Get the ensemble trainer instance."""
        return self.ensemble_trainer
