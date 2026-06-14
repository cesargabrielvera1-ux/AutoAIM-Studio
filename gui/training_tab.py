"""Model training tab."""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QProgressBar, QCheckBox, QSpinBox, QDoubleSpinBox,
    QTextEdit, QMessageBox, QSplitter, QListWidget, QListWidgetItem,
    QTabWidget, QGridLayout, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.model_registry import ModelRegistry
from ..core.trainer import ModelTrainer, TrainingResult


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
        config_layout.addWidget(self.cv_spin, 0, 1)
        
        config_layout.addWidget(QLabel("Test Size:"), 1, 0)
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.1, 0.5)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.2)
        config_layout.addWidget(self.test_spin, 1, 1)
        
        config_layout.addWidget(QLabel("Random State:"), 2, 0)
        self.random_spin = QSpinBox()
        self.random_spin.setRange(0, 9999)
        self.random_spin.setValue(42)
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
