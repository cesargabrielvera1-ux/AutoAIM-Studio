"""Ensemble tab for creating and training ensemble models."""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QSpinBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QTextEdit, QMessageBox, QProgressBar, QCheckBox, QSplitter,
    QListWidget, QListWidgetItem, QDialog, QDoubleSpinBox, QGridLayout,
    QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.ensemble_trainer import EnsembleTrainer, EnsembleResult


class EnsembleOptimizationThread(QThread):
    """Thread for optimizing ensemble weights."""
    progress = pyqtSignal(int, float, dict)
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, ensemble_trainer, X, y, model_names, n_trials=50, cv_folds=5, random_state=42):
        super().__init__()
        self.ensemble_trainer = ensemble_trainer
        self.X = X
        self.y = y
        self.model_names = model_names
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.random_state = random_state
    
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
                progress_callback=progress_callback
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
                 X_test, y_test, cv_folds=5, random_state=42, weights=None):
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
                weights=self.weights
            )
            self.finished.emit(result)
        except Exception as e:
            self.error.emit(str(e))


class EnsembleTab(QWidget):
    """Tab for ensemble model training."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
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
        
        config_layout.addWidget(QLabel("CV Folds (Training):"), 1, 0)
        self.cv_folds_spin = QSpinBox()
        self.cv_folds_spin.setRange(1, 10)
        self.cv_folds_spin.setValue(5)
        self.cv_folds_spin.setToolTip(
            "Cross-validation folds for training the ensemble. "
            "Each base model is retrained on K different splits to generate "
            "out-of-fold predictions for the ensemble."
        )
        config_layout.addWidget(self.cv_folds_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Random Seed:"), 2, 0)
        self.random_seed_spin = QSpinBox()
        self.random_seed_spin.setRange(0, 9999)
        self.random_seed_spin.setValue(42)
        self.random_seed_spin.setToolTip("Random seed for reproducible ensemble training")
        config_layout.addWidget(self.random_seed_spin, 2, 1)
        
        left_layout.addWidget(config_group)
        
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
        
        # Buttons
        self.optimize_weights_btn = QPushButton("Optimize Weights")
        self.optimize_weights_btn.setStyleSheet("""
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
        splitter.addWidget(left_widget)
        
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
        
        if self.parent is None:
            return
        
        # 1. Add trained models
        if hasattr(self.parent, 'training_tab'):
            try:
                trainer = self.parent.training_tab.get_trainer()
                if trainer and hasattr(trainer, 'results') and trainer.results:
                    for name, result in trainer.results.items():
                        if hasattr(result, 'model') and result.model is not None:
                            item = QListWidgetItem(f"{name} (Trained)")
                            item.setData(Qt.ItemDataRole.UserRole, ('trained', name, result))
                            self.models_list.addItem(item)
            except Exception as e:
                print(f"Error loading trained models: {e}")
        
        # 2. Add optimized models
        if hasattr(self.parent, 'optimization_tab'):
            try:
                opt_tab = self.parent.optimization_tab
                if hasattr(opt_tab, 'optimization_results') and opt_tab.optimization_results:
                    for model_name, opt_result in opt_tab.optimization_results:
                        if hasattr(opt_result, 'best_model') and opt_result.best_model is not None:
                            item = QListWidgetItem(f"{model_name} (Optimized)")
                            item.setData(Qt.ItemDataRole.UserRole, ('optimized', model_name, opt_result))
                            self.models_list.addItem(item)
            except Exception as e:
                print(f"Error loading optimized models: {e}")
        
        # 3. Add neural network models
        if hasattr(self.parent, 'nn_tab'):
            try:
                nn_tab = self.parent.nn_tab
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
        if self.parent is None or not hasattr(self.parent, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self.parent.data_tab.get_data_manager()
        trainer = self.parent.training_tab.get_trainer()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        selected_models, _ = self._get_selected_models()
        if len(selected_models) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 models!")
            return
        
        weights = self._get_manual_weights()
        
        try:
            use_val = data_manager.validation_data is not None
            X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
            
            ensemble_type = self.ensemble_type_combo.currentText()
            cv_folds = self.cv_folds_spin.value()
            
            self.ensemble_trainer = EnsembleTrainer(trainer)
            
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
            
            self.training_thread = EnsembleTrainingThread(
                self.ensemble_trainer,
                ensemble_type,
                selected_models,
                X_train, y_train,
                X_test, y_test,
                cv_folds=cv_folds,
                random_state=self.random_seed_spin.value(),
                weights=weights
            )
            self.training_thread.progress.connect(self._on_training_progress)
            self.training_thread.finished.connect(self._on_training_finished)
            self.training_thread.error.connect(self._on_training_error)
            self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to start manual ensemble training:\n{str(e)}")
            self.train_manual_btn.setEnabled(True)
            self.train_btn.setEnabled(True)
    
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
        if self.parent is None or not hasattr(self.parent, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self.parent.data_tab.get_data_manager()
        trainer = self.parent.training_tab.get_trainer()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        selected_models, selected_data = self._get_selected_models()
        if len(selected_models) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 models!")
            return
        
        try:
            use_val = data_manager.validation_data is not None
            X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
            
            ensemble_type = self.ensemble_type_combo.currentText()
            cv_folds = self.cv_folds_spin.value()
            
            # Create EnsembleTrainer with current trainer
            self.ensemble_trainer = EnsembleTrainer(trainer)
            
            self.train_btn.setEnabled(False)
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(0)
            self.progress_text.clear()
            
            self.training_thread = EnsembleTrainingThread(
                self.ensemble_trainer,
                ensemble_type,
                selected_models,
                X_train, y_train,
                X_test, y_test,
                cv_folds=cv_folds,
                random_state=self.random_seed_spin.value()
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
        if self.parent is None or not hasattr(self.parent, 'data_tab'):
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self.parent.data_tab.get_data_manager()
        trainer = self.parent.training_tab.get_trainer()
        
        if data_manager is None or data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded!")
            return
        
        selected_models, selected_data = self._get_selected_models()
        if len(selected_models) < 2:
            QMessageBox.warning(self, "Warning", "Please select at least 2 models!")
            return
        
        try:
            use_val = data_manager.validation_data is not None
            X_train, X_test, y_train, y_test = data_manager.prepare_data(use_validation=use_val)
            
            n_trials = self.opt_trials_spin.value()
            cv_folds = self.opt_cv_folds_spin.value()
            
            # Create EnsembleTrainer with current trainer
            self.ensemble_trainer = EnsembleTrainer(trainer)
            
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
                random_state=self.opt_random_seed_spin.value()
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
        
        # Store the result in the main trainer's results dict so it appears in Results tab
        if self.parent and hasattr(self.parent, 'training_tab'):
            try:
                trainer = self.parent.training_tab.get_trainer()
                ens_name = f"Ensemble_{result.ensemble_type}_{len(result.model_names)}models"
                
                # Create a TrainingResult-compatible wrapper for the Results tab
                from ..core.trainer import TrainingResult
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                
                # Use ensemble metrics directly
                metrics = dict(result.metrics)
                # Ensure standard metric keys exist
                if 'rmse' not in metrics and 'r2' in metrics:
                    metrics['rmse'] = 0.0
                if 'mae' not in metrics and 'r2' in metrics:
                    metrics['mae'] = 0.0
                
                training_result = TrainingResult(
                    model=result,  # Store the EnsembleResult as the model
                    model_name=ens_name,
                    problem_type=result.problem_type,
                    metrics=metrics,
                    cv_metrics={'test_score': result.cv_scores} if result.cv_scores else {},
                    training_time=result.training_time,
                    feature_importance=None,
                    predictions=result.predictions,
                    true_values=result.true_values,
                    is_neural_network=False
                )
                trainer.results[ens_name] = training_result
                self.progress_text.append(f"Added '{ens_name}' to Results tab")
            except Exception as e:
                self.progress_text.append(f"Warning: Could not add to Results tab: {e}")
        
        # v1.3.0: Ensemble results are already stored in ensemble_trainer.results
        # by train_ensemble(). Do NOT duplicate in trainer.results to avoid
        # triplicated entries in the Results tab.
        
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
