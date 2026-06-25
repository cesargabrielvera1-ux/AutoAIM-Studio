"""Hyperparameter Optimization tab."""

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

from ..core.optimizer import HyperparameterOptimizer, OptimizationResult


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


class OptimizationThread(QThread):
    """Thread for running hyperparameter optimization."""
    progress = pyqtSignal(int, float, dict)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, optimizer, model_type, algorithm, param_ranges, n_trials,
                 X_train, y_train, X_test, y_test, problem_type, cv_folds=5,
                 timeout=None, pruning=True, random_state=42):
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
    
    def run(self):
        try:
            def progress_callback(trial_num, score, params):
                self.progress.emit(trial_num, score, params)
            
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
                progress_callback=progress_callback
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
                 problem_type, cv_folds, feature_names=None):
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
                progress_callback=progress_callback
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class OptimizationTab(QWidget):
    """Tab for hyperparameter optimization."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.optimizer = HyperparameterOptimizer()
        self.optimization_results = []
        self.custom_param_ranges = {}
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
        self.model_type_combo.addItems([
            "Random Forest", "Gradient Boosting", "XGBoost",
            "LightGBM", "CatBoost", "Support Vector Machine",
            "k-Nearest Neighbors"
        ])
        self.model_type_combo.currentTextChanged.connect(self._on_model_type_changed)
        model_layout.addWidget(self.model_type_combo)
        
        left_layout.addWidget(model_group)
        
        # Optimization Configuration
        opt_group = QGroupBox("Optimization Configuration")
        opt_layout = QGridLayout(opt_group)
        
        opt_layout.addWidget(QLabel("Number of Trials:"), 0, 0)
        self.trials_spin = QSpinBox()
        self.trials_spin.setRange(10, 200)
        self.trials_spin.setValue(50)
        self.trials_spin.setSingleStep(10)
        self.trials_spin.setToolTip(
            "Number of optimization trials. Each trial tests a different combination "
            "of hyperparameters. More trials = better results but much longer. "
            "50 is a good balance for most problems."
        )
        opt_layout.addWidget(self.trials_spin, 0, 1)
        
        opt_layout.addWidget(QLabel("CV Folds:"), 1, 0)
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(1, 10)
        self.cv_folds_spin.setValue(5)
        self.cv_folds_spin.setToolTip(
            "Cross-validation folds used to evaluate each trial. "
            "Higher = more reliable score estimates during optimization."
        )
        opt_layout.addWidget(self.cv_folds_spin, 1, 1)
        
        opt_layout.addWidget(QLabel("Random Seed:"), 2, 0)
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setToolTip(
            "Random seed for reproducible optimization. "
            "Same seed = same sequence of hyperparameter trials."
        )
        opt_layout.addWidget(self.random_seed_spin, 2, 1)
        
        opt_layout.addWidget(QLabel("Timeout (s, 0 = none):"), 3, 0)
        self.timeout_spin = QSpinBox()
        self.timeout_spin.setRange(0, 36000)
        self.timeout_spin.setValue(0)
        self.timeout_spin.setSingleStep(60)
        self.timeout_spin.setToolTip(
            "Maximum time allowed for optimization (seconds). "
            "0 = no limit. Useful to prevent runaway optimizations on slow machines."
        )
        opt_layout.addWidget(self.timeout_spin, 3, 1)
        
        self.pruning_check = QCheckBox("Enable Pruning (stop unpromising trials)")
        self.pruning_check.setChecked(True)
        self.pruning_check.setToolTip(
            "Automatically stop trials that perform poorly early. "
            "Speeds up optimization by ~30-50% without sacrificing quality. "
            "Recommended: leave enabled."
        )
        opt_layout.addWidget(self.pruning_check, 4, 0, 1, 2)
        
        self.edit_ranges_btn = QPushButton("Edit Parameter Ranges")
        self.edit_ranges_btn.clicked.connect(self._edit_parameter_ranges)
        opt_layout.addWidget(self.edit_ranges_btn, 5, 0, 1, 2)
        
        self.reset_ranges_btn = QPushButton("Reset Ranges to Default")
        self.reset_ranges_btn.clicked.connect(self._reset_parameter_ranges)
        self.reset_ranges_btn.setEnabled(False)
        opt_layout.addWidget(self.reset_ranges_btn, 6, 0, 1, 2)
        
        self.ranges_label = QLabel("Using default ranges")
        opt_layout.addWidget(self.ranges_label, 7, 0, 1, 2)
        
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
        pass  # Parameter ranges are set per algorithm
    
    def _edit_parameter_ranges(self):
        """Open dialog to edit parameter ranges."""
        model_type = self.model_type_combo.currentText()
        param_ranges = self.custom_param_ranges.get(model_type)
        
        if param_ranges is None:
            param_ranges = self.optimizer.get_param_ranges(self.model_type_combo.currentText())
        
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
        data_manager = self.parent.data_tab.get_data_manager()
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        # Get data (use external validation set if loaded)
        use_val = data_manager.validation_data is not None
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
        
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
        
        # Get parameter ranges
        param_ranges = self.custom_param_ranges.get(model_type)
        if param_ranges is None:
            param_ranges = self.optimizer.get_param_ranges(algorithm)
        
        # Start optimization
        self.optimize_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_text.clear()
        self.results_table.setRowCount(0)
        
        self.optimization_thread = OptimizationThread(
            self.optimizer, model_type, algorithm, param_ranges, n_trials,
            X_train, y_train, X_test, y_test, problem_type, cv_folds,
            timeout=timeout if timeout > 0 else None,
            pruning=pruning,
            random_state=self.random_seed_spin.value()
        )
        self.optimization_thread.progress.connect(self._on_optimization_progress)
        self.optimization_thread.finished.connect(self._on_optimization_finished)
        self.optimization_thread.error.connect(self._on_optimization_error)
        self.optimization_thread.start()
    
    def _on_optimization_progress(self, trial_num, score, params):
        """Handle optimization progress."""
        self.progress_bar.setValue(int(100 * trial_num / self.trials_spin.value()))
        self.progress_text.append(
            f"Trial {trial_num}: Score={score:.4f}, Params={params}"
        )
    
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
        # v1.2.1 FIX: Replace previous optimization of same model_type
        # instead of appending, to prevent accumulation.
        model_type = self.model_type_combo.currentText()
        self.optimization_results = [
            (mt, r) for mt, r in self.optimization_results if mt != model_type
        ]
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
        
        data_manager = self.parent.data_tab.get_data_manager()
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        trainer = self.parent.training_tab.get_trainer()
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
        
        # Get data (use external validation set if loaded)
        use_val = data_manager.validation_data is not None
        X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
        
        # Get feature names
        feature_names = data_manager.data.feature_names if hasattr(data_manager.data, 'feature_names') else None
        
        # Show progress bar and disable button
        self.train_best_btn.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_text.append("Training with best parameters...")
        
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
            feature_names=feature_names
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
        
        # v1.3.0: Build detailed message with per-fold CV scores
        cv_scores = result.cv_metrics.get('test_score', []) if result.cv_metrics else []
        msg = f"Model trained with best parameters!\n\n"
        msg += f"Name: {result.model_name}\n"
        
        if cv_scores:
            msg += f"\nCV Scores (per fold):\n"
            for fold_idx, score in enumerate(cv_scores, 1):
                msg += f"  Fold {fold_idx}: {score:.6f}\n"
            msg += f"  Mean ± Std: {np.mean(cv_scores):.6f} ± {np.std(cv_scores):.6f}\n"
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
        
        # v1.3.0: Merge best_params with model defaults so ALL parameters are shown.
        # Optuna's best_params only includes sampled parameters; we fill in the rest
        # from the model's default_params for a complete picture.
        from ..core.model_registry import ModelRegistry
        registry = ModelRegistry()
        algorithm = self.model_type_combo.currentText().lower().replace(' ', '_')
        try:
            model_info = registry.get_model(algorithm)
            full_params = dict(model_info.default_params)  # Start with all defaults
            full_params.update(result.best_params)  # Override with optimized values
            # Remove internal/fixed params that aren't tunable
            for p in ('random_state', 'n_jobs', 'verbose', 'random_seed', 'tree_method'):
                full_params.pop(p, None)
        except Exception:
            full_params = result.best_params  # Fallback to Optuna's params only
        
        details += "\nBest Parameters:\n"
        for param, value in sorted(full_params.items()):
            marker = " *" if param in result.best_params else "  "
            details += f"{marker}{param}: {value}\n"
        details += "\n(*) = optimized by Optuna\n"
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
        
        # v1.3.0: Show CV fold scores in details (same format as Training Tab)
        cv_scores = result.cv_metrics.get('test_score', []) if result.cv_metrics else []
        if cv_scores and len(cv_scores) > 0:
            details += "\n\nCross-Validation Scores (per fold):\n"
            details += "-" * 35 + "\n"
            for fold_idx, score in enumerate(cv_scores, 1):
                details += f"  Fold {fold_idx:2d}: {score:.6f}\n"
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            details += f"  {'Mean ± Std':10s}: {mean_score:.6f} ± {std_score:.6f}\n"
            details += "-" * 35 + "\n"
        
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
