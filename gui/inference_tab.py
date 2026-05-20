"""Inference/Prediction tab for standalone model prediction."""

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QLabel, QLineEdit, QFileDialog,
    QTableWidget, QTableWidgetItem, QProgressBar,
    QMessageBox, QScrollArea, QFrame, QTextEdit,
    QCheckBox, QSpinBox, QSplitter, QHeaderView,
    QDialog, QDialogButtonBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ..core.inference_engine import InferenceEngine, LegacyInferenceEngine
from ..utils.logger import LoggerMixin


class PredictionThread(QThread):
    """Thread for running predictions."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(self, engine: InferenceEngine, df: pd.DataFrame):
        super().__init__()
        self.engine = engine
        self.df = df
    
    def run(self):
        try:
            self.progress.emit(10)
            
            # Run prediction
            result = self.engine.predict(self.df)
            
            self.progress.emit(100)
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class InferenceTab(QWidget, LoggerMixin):
    """Tab for standalone model inference/prediction."""
    
    def __init__(self, parent=None):
        """Initialize inference tab."""
        super().__init__(parent)
        
        self.parent = parent
        self.inference_engine: Optional[InferenceEngine] = None
        self.input_data: Optional[pd.DataFrame] = None
        self.prediction_result: Optional[pd.DataFrame] = None
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Create splitter for resizable sections
        splitter = QSplitter(Qt.Orientation.Vertical)
        main_layout.addWidget(splitter)
        
        # === TOP SECTION: Model Loading ===
        model_group = QGroupBox("Load Model")
        model_layout = QVBoxLayout(model_group)
        
        # Model path selection
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Model Folder:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select folder containing manifest.json and model files...")
        self.model_path_edit.setReadOnly(True)
        path_layout.addWidget(self.model_path_edit)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(self.browse_btn)
        
        model_layout.addLayout(path_layout)
        
        # Model info display
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(150)
        self.model_info_text.setPlaceholderText("Model information will appear here after loading...")
        model_layout.addWidget(self.model_info_text)
        
        # Version warning label
        self.version_warning_label = QLabel("")
        self.version_warning_label.setStyleSheet("color: orange;")
        model_layout.addWidget(self.version_warning_label)
        
        splitter.addWidget(model_group)
        
        # === MIDDLE SECTION: Data Loading ===
        data_group = QGroupBox("Data for Prediction")
        data_layout = QVBoxLayout(data_group)
        
        # Data file selection
        data_path_layout = QHBoxLayout()
        data_path_layout.addWidget(QLabel("CSV File:"))
        self.data_path_edit = QLineEdit()
        self.data_path_edit.setPlaceholderText("Select CSV file with data to predict...")
        self.data_path_edit.setReadOnly(True)
        data_path_layout.addWidget(self.data_path_edit)
        
        self.load_data_btn = QPushButton("Load CSV...")
        self.load_data_btn.clicked.connect(self._load_data)
        data_path_layout.addWidget(self.load_data_btn)
        
        data_layout.addLayout(data_path_layout)
        
        # Validation status
        self.validation_label = QLabel("No data loaded")
        self.validation_label.setStyleSheet("padding: 5px; border-radius: 3px;")
        data_layout.addWidget(self.validation_label)
        
        # Data preview table
        data_layout.addWidget(QLabel("Data Preview (first 5 rows):"))
        self.data_preview_table = QTableWidget()
        self.data_preview_table.setMaximumHeight(200)
        data_layout.addWidget(self.data_preview_table)
        
        # Feature generation option
        self.auto_features_check = QCheckBox("Auto-generate composition features from formula column if missing")
        self.auto_features_check.setChecked(True)
        data_layout.addWidget(self.auto_features_check)
        
        splitter.addWidget(data_group)
        
        # === BOTTOM SECTION: Prediction & Results ===
        bottom_widget = QWidget()
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Prediction button
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.setEnabled(False)
        self.predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 12pt;
            }
            QPushButton:disabled {
                background-color: #555555;
            }
            QPushButton:hover:enabled {
                background-color: #14a085;
            }
        """)
        self.predict_btn.clicked.connect(self._run_prediction)
        bottom_layout.addWidget(self.predict_btn)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)
        
        # Results section
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout(results_group)
        
        # Statistics
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("No predictions yet")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        
        self.export_btn = QPushButton("Export to CSV...")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        stats_layout.addWidget(self.export_btn)
        
        results_layout.addLayout(stats_layout)
        
        # Results preview table
        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(200)
        results_layout.addWidget(self.results_table)
        
        # Prediction distribution plot
        self.plot_frame = QFrame()
        self.plot_layout = QVBoxLayout(self.plot_frame)
        self.figure = plt.Figure(figsize=(6, 3))
        self.canvas = FigureCanvas(self.figure)
        self.plot_layout.addWidget(self.canvas)
        results_layout.addWidget(self.plot_frame)
        
        bottom_layout.addWidget(results_group)
        
        splitter.addWidget(bottom_widget)
        
        # Set splitter proportions
        splitter.setSizes([200, 300, 400])
    
    def _browse_model(self):
        """Browse for model folder or file."""
        # First try to select a folder (for bundle format)
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Model Folder (containing manifest.json) - Cancel to select file",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder:
            # Check if manifest.json exists
            manifest_path = Path(folder) / "manifest.json"
            if manifest_path.exists():
                self.model_path_edit.setText(folder)
                self._load_model(folder)
                return
            else:
                # Folder selected but no manifest - ask if they want to select a file
                reply = QMessageBox.question(
                    self,
                    "No manifest.json found",
                    f"No manifest.json found in {folder}\n\nDo you want to select a .joblib or .pkl file instead?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    return
        
        # Select a file (.joblib or .pkl)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Model File",
            "",
            "Model Files (*.joblib *.pkl *.pt);;Joblib Files (*.joblib);;Pickle Files (*.pkl);;PyTorch Files (*.pt);;All Files (*)"
        )
        
        if file_path:
            path = Path(file_path)
            # Check if manifest.json exists in the same folder
            manifest_path = path.parent / "manifest.json"
            if manifest_path.exists():
                # Load as bundle
                folder = str(path.parent)
                self.model_path_edit.setText(folder)
                self._load_model(folder)
            else:
                # Load as legacy format
                self._load_legacy_model(path)
    
    def _load_legacy_model(self, file_path: Path):
        """Load legacy model format (.joblib or .pkl without manifest)."""
        try:
            import joblib
            
            # Load the model
            model = joblib.load(file_path)
            
            # Create a basic inference engine for legacy models
            self.inference_engine = LegacyInferenceEngine(model, str(file_path))
            
            # Display basic info
            info_text = f"""
<b>Model:</b> {file_path.name}<br>
<b>Format:</b> Legacy (no manifest)<br>
<b>Type:</b> {type(model).__name__}<br>
<b>Note:</b> Legacy models may not have all features available.
            """
            self.model_info_text.setHtml(info_text)
            self.version_warning_label.setText("⚠️ Legacy model format - limited functionality")
            
            self.logger.info(f"Legacy model loaded: {file_path}")
            self._update_predict_button()
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", f"Failed to load legacy model:\n{str(e)}")
            self.inference_engine = None
            self.model_path_edit.clear()
    
    def _load_model(self, folder_path: str):
        """Load model from folder."""
        try:
            self.inference_engine = InferenceEngine(folder_path)
            
            # Get model info
            info = self.inference_engine.get_model_info()
            
            # Display info
            info_text = f"""
<b>Model:</b> {info.get('model_name', 'Unknown')}<br>
<b>Algorithm:</b> {info.get('algorithm', 'Unknown')}<br>
<b>Type:</b> {info.get('model_type', 'Unknown')}<br>
<b>Created:</b> {info.get('creation_date', 'Unknown')[:10]}<br>
<b>Features Required:</b> {info.get('feature_count', 0)}<br>
<b>Target Column:</b> {info.get('target_column', 'Unknown')}<br>
<b>Metrics:</b> {json.dumps(info.get('metrics', {}), indent=2)}
            """
            self.model_info_text.setHtml(info_text)
            
            # Check version compatibility
            compatible, message = self.inference_engine.check_version_compatibility()
            if not compatible:
                self.version_warning_label.setText(f"⚠️ {message}")
                reply = QMessageBox.warning(
                    self,
                    "Version Mismatch",
                    f"{message}\n\nDo you want to continue anyway?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply == QMessageBox.StandardButton.No:
                    self.inference_engine = None
                    self.model_path_edit.clear()
                    return
            else:
                self.version_warning_label.setText("")
            
            self.logger.info(f"Model loaded: {info.get('model_name')}")
            self._update_predict_button()
            
        except Exception as e:
            QMessageBox.critical(self, "Error Loading Model", str(e))
            self.inference_engine = None
            self.model_path_edit.clear()
    
    def _load_data(self):
        """Load data CSV file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Data File",
            "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.input_data = pd.read_csv(file_path, index_col=False)
                self.data_path_edit.setText(file_path)
                
                # Strip trailing "Unnamed" artifact columns from CSV
                unnamed_cols = [c for c in self.input_data.columns if str(c).startswith("Unnamed:")]
                if unnamed_cols:
                    self.input_data = self.input_data.drop(columns=unnamed_cols)
                    self.logger.warning(f"Dropped {len(unnamed_cols)} 'Unnamed' artifact columns from CSV: {unnamed_cols}")
                
                # Update preview
                self._update_data_preview()
                
                # Validate against model
                if self.inference_engine:
                    self._validate_data()
                
                self.logger.info(f"Data loaded: {len(self.input_data)} rows, {len(self.input_data.columns)} columns")
                self._update_predict_button()
                
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Data", str(e))
                self.input_data = None
    
    def _update_data_preview(self):
        """Update data preview table."""
        if self.input_data is None:
            return
        
        # Show first 5 rows
        preview = self.input_data.head()
        
        self.data_preview_table.setColumnCount(len(preview.columns))
        self.data_preview_table.setRowCount(len(preview))
        
        self.data_preview_table.setHorizontalHeaderLabels(preview.columns)
        
        for i, row in preview.iterrows():
            for j, col in enumerate(preview.columns):
                item = QTableWidgetItem(str(row[col]))
                self.data_preview_table.setItem(i, j, item)
        
        self.data_preview_table.resizeColumnsToContents()
    
    def _validate_data(self):
        """Validate input data against model requirements."""
        if self.inference_engine is None or self.input_data is None:
            return
        
        validation = self.inference_engine.validate_input(self.input_data)
        
        if validation["valid"]:
            self.validation_label.setText(f"✅ {validation['message']}")
            self.validation_label.setStyleSheet("background-color: #2d5a3d; color: white; padding: 5px; border-radius: 3px;")
        else:
            missing = validation.get("missing_columns", [])
            extra = validation.get("extra_columns", [])
            
            msg_parts = []
            if missing:
                msg_parts.append(f"Missing: {missing}")
            if extra:
                msg_parts.append(f"Extra (will be ignored): {extra}")
            
            self.validation_label.setText(f"❌ {'; '.join(msg_parts)}")
            self.validation_label.setStyleSheet("background-color: #5a2d2d; color: white; padding: 5px; border-radius: 3px;")
            
            # Check if we can auto-generate features
            if missing and self.auto_features_check.isChecked():
                can_generate, generatable = self.inference_engine.can_generate_features(self.input_data)
                if can_generate:
                    self.validation_label.setText(
                        f"⚠️ Missing {len(missing)} features, but {len(generatable)} can be auto-generated from formula. "
                        f"Click 'Run Prediction' to generate them."
                    )
                    self.validation_label.setStyleSheet("background-color: #5a5a2d; color: white; padding: 5px; border-radius: 3px;")
        
        self._update_predict_button()
    
    def _update_predict_button(self):
        """Update predict button state."""
        can_predict = (
            self.inference_engine is not None and
            self.input_data is not None
        )
        self.predict_btn.setEnabled(can_predict)
    
    def _run_prediction(self):
        """Run prediction on loaded data."""
        if self.inference_engine is None or self.input_data is None:
            return
        
        # Check if we need to generate features
        validation = self.inference_engine.validate_input(self.input_data)
        
        if not validation["valid"] and validation.get("missing_columns"):
            if self.auto_features_check.isChecked():
                can_generate, _ = self.inference_engine.can_generate_features(self.input_data)
                if can_generate:
                    reply = QMessageBox.question(
                        self,
                        "Generate Features",
                        f"{len(validation['missing_columns'])} features are missing. "
                        f"Generate them automatically from the formula column?",
                        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                    )
                    
                    if reply == QMessageBox.StandardButton.Yes:
                        try:
                            self.input_data = self.inference_engine.generate_missing_features(self.input_data)
                            self._update_data_preview()
                            self._validate_data()
                        except Exception as e:
                            QMessageBox.critical(self, "Feature Generation Error", str(e))
                            return
                    else:
                        return
                else:
                    QMessageBox.warning(
                        self,
                        "Missing Features",
                        f"Cannot generate missing features automatically.\n"
                        f"Missing: {validation['missing_columns']}"
                    )
                    return
            else:
                QMessageBox.warning(
                    self,
                    "Missing Features",
                    f"Input data is missing required features:\n"
                    f"{validation['missing_columns']}"
                )
                return
        
        # Load model if not loaded
        if not self.inference_engine.is_loaded:
            try:
                self.inference_engine.load()
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Model", str(e))
                return
        
        # Run prediction in thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.predict_btn.setEnabled(False)
        
        self.prediction_thread = PredictionThread(self.inference_engine, self.input_data)
        self.prediction_thread.progress.connect(self.progress_bar.setValue)
        self.prediction_thread.finished.connect(self._prediction_finished)
        self.prediction_thread.error.connect(self._prediction_error)
        self.prediction_thread.start()
    
    def _prediction_finished(self, result: pd.DataFrame):
        """Handle prediction completion."""
        self.prediction_result = result
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        
        # Update results display
        self._update_results_display()
        
        # Enable export
        self.export_btn.setEnabled(True)
        
        QMessageBox.information(
            self,
            "Prediction Complete",
            f"Successfully predicted {len(result)} rows.\n"
            f"Predictions saved in 'prediction' column."
        )
    
    def _prediction_error(self, error_msg: str):
        """Handle prediction error."""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Prediction Error", error_msg)
    
    def _update_results_display(self):
        """Update results display."""
        if self.prediction_result is None:
            return
        
        predictions = self.prediction_result['prediction']
        
        # Update statistics
        stats_text = f"""
<b>Prediction Statistics:</b><br>
Count: {len(predictions)} | 
Min: {predictions.min():.4f} | 
Max: {predictions.max():.4f} | 
Mean: {predictions.mean():.4f} | 
Std: {predictions.std():.4f}
        """
        self.stats_label.setText(stats_text)
        
        # Update results table (show first 10 rows with prediction)
        display_df = self.prediction_result.head(10)
        
        self.results_table.setColumnCount(len(display_df.columns))
        self.results_table.setRowCount(len(display_df))
        
        self.results_table.setHorizontalHeaderLabels(display_df.columns)
        
        for i, (idx, row) in enumerate(display_df.iterrows()):
            for j, col in enumerate(display_df.columns):
                val = row[col]
                if isinstance(val, float):
                    item = QTableWidgetItem(f"{val:.4f}")
                else:
                    item = QTableWidgetItem(str(val))
                
                # Highlight prediction column
                if col == 'prediction':
                    item.setBackground(QColor(13, 115, 119, 100))
                
                self.results_table.setItem(i, j, item)
        
        self.results_table.resizeColumnsToContents()
        
        # Update plot
        self._update_prediction_plot(predictions)
    
    def _update_prediction_plot(self, predictions: pd.Series):
        """Update prediction distribution plot."""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        ax.hist(predictions, bins=30, edgecolor='black', alpha=0.7, color='#0d7377')
        ax.set_xlabel('Prediction Value')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Predictions')
        ax.axvline(predictions.mean(), color='red', linestyle='--', label=f'Mean: {predictions.mean():.4f}')
        ax.legend()
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _export_results(self):
        """Export prediction results to CSV."""
        if self.prediction_result is None:
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Predictions",
            "predictions.csv",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        
        if file_path:
            try:
                self.prediction_result.to_csv(file_path, index=False)
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Predictions saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))
