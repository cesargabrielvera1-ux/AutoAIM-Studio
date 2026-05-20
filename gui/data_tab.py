"""Data loading and preprocessing tab."""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QCheckBox,
    QSplitter, QTextEdit, QHeaderView, QScrollArea, QFrame
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.data_manager import DataManager
from ..core.feature_engineering import FeatureEngineer


class DataLoadingThread(QThread):
    """Thread for loading data."""
    finished = pyqtSignal(object)
    error = pyqtSignal(str)
    
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.data_manager = DataManager()
    
    def run(self):
        try:
            info = self.data_manager.load_data(self.file_path)
            self.finished.emit((self.data_manager, info))
        except Exception as e:
            self.error.emit(str(e))


class DataTab(QWidget):
    """Data loading and preprocessing tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.parent = parent
        self.data_manager: DataManager = None
        self.feature_engineer = FeatureEngineer()
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Data")
        self.load_btn.setToolTip("Load CSV or Excel file")
        self.load_btn.clicked.connect(self.load_data_dialog)
        controls_layout.addWidget(self.load_btn)
        
        self.load_val_btn = QPushButton("Load Validation Data")
        self.load_val_btn.setToolTip("Load external validation dataset")
        self.load_val_btn.clicked.connect(self.load_validation_dialog)
        self.load_val_btn.setEnabled(False)
        controls_layout.addWidget(self.load_val_btn)
        
        controls_layout.addStretch()
        
        # Target selection
        controls_layout.addWidget(QLabel("Target Column:"))
        self.target_combo = QComboBox()
        self.target_combo.setMinimumWidth(200)
        self.target_combo.currentTextChanged.connect(self._on_target_changed)
        controls_layout.addWidget(self.target_combo)
        
        layout.addLayout(controls_layout)
        
        # Splitter for data preview and info
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Data preview
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        left_layout.addWidget(QLabel("<b>Data Preview</b>"))
        
        self.data_table = QTableWidget()
        self.data_table.setMaximumHeight(400)
        left_layout.addWidget(self.data_table)
        
        # Preprocessing options
        prep_group = QGroupBox("Preprocessing Options")
        prep_layout = QVBoxLayout(prep_group)
        
        self.impute_combo = QComboBox()
        self.impute_combo.addItems(['mean', 'median', 'most_frequent', 'knn'])
        prep_layout.addWidget(QLabel("Imputation Strategy:"))
        prep_layout.addWidget(self.impute_combo)
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['standard', 'minmax', 'robust', 'none'])
        prep_layout.addWidget(QLabel("Scaling Strategy:"))
        prep_layout.addWidget(self.scale_combo)
        
        self.encode_combo = QComboBox()
        self.encode_combo.addItems(['onehot', 'label', 'none'])
        prep_layout.addWidget(QLabel("Encoding Strategy:"))
        prep_layout.addWidget(self.encode_combo)
        
        self.apply_prep_btn = QPushButton("Apply Preprocessing")
        self.apply_prep_btn.clicked.connect(self._apply_preprocessing)
        self.apply_prep_btn.setEnabled(False)
        prep_layout.addWidget(self.apply_prep_btn)
        
        left_layout.addWidget(prep_group)
        
        splitter.addWidget(left_panel)
        
        # Right panel: Dataset info and column details
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Dataset info
        info_group = QGroupBox("Dataset Information")
        info_layout = QVBoxLayout(info_group)
        
        self.info_text = QTextEdit()
        self.info_text.setReadOnly(True)
        self.info_text.setMaximumHeight(150)
        info_layout.addWidget(self.info_text)
        
        right_layout.addWidget(info_group)
        
        # Column details
        col_group = QGroupBox("Column Details")
        col_layout = QVBoxLayout(col_group)
        
        self.col_table = QTableWidget()
        self.col_table.setColumnCount(6)
        self.col_table.setHorizontalHeaderLabels([
            'Column', 'Type', 'Numeric', 'Categorical', 'Composition', 'Missing'
        ])
        self.col_table.horizontalHeader().setStretchLastSection(True)
        col_layout.addWidget(self.col_table)
        
        # Feature engineering
        fe_layout = QHBoxLayout()
        
        self.comp_check = QCheckBox("Generate composition features (106 descriptors)")
        fe_layout.addWidget(self.comp_check)
        
        self.apply_fe_btn = QPushButton("Apply Feature Engineering")
        self.apply_fe_btn.clicked.connect(self._apply_feature_engineering)
        self.apply_fe_btn.setEnabled(False)
        fe_layout.addWidget(self.apply_fe_btn)
        
        col_layout.addLayout(fe_layout)
        
        right_layout.addWidget(col_group)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 600])
        
        layout.addWidget(splitter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)
        
        # Status label
        self.status_label = QLabel("No data loaded")
        layout.addWidget(self.status_label)
    
    def load_data_dialog(self):
        """Open dialog to load data file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Data File",
            "",
            "Data Files (*.csv *.xlsx *.xls *.json *.parquet);;CSV Files (*.csv);;Excel Files (*.xlsx *.xls);;All Files (*)"
        )
        
        if file_path:
            self._load_data(file_path)
    
    def _load_data(self, file_path: str):
        """Load data from file."""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText(f"Loading {file_path}...")
        
        # Create loading thread
        self.loading_thread = DataLoadingThread(file_path)
        self.loading_thread.finished.connect(self._on_data_loaded)
        self.loading_thread.error.connect(self._on_load_error)
        self.loading_thread.start()
    
    def _on_data_loaded(self, result):
        """Handle data loaded."""
        self.data_manager, info = result
        
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Loaded: {info.n_samples} samples, {info.n_features} features")
        
        # Update target combo
        self.target_combo.clear()
        self.target_combo.addItems(list(info.columns.keys()))
        
        # Update data preview
        self._update_data_preview()
        
        # Update column details
        self._update_column_details(info)
        
        # Update dataset info
        self._update_dataset_info(info)
        
        # Enable buttons
        self.load_val_btn.setEnabled(True)
        self.apply_prep_btn.setEnabled(True)
        self.apply_fe_btn.setEnabled(True)
        
        # Notify parent
        if self.parent:
            self.parent.data_tab = self
    
    def _on_load_error(self, error_msg: str):
        """Handle load error."""
        self.progress_bar.setVisible(False)
        self.status_label.setText(f"Error loading data")
        QMessageBox.critical(self, "Error", f"Failed to load data:\n{error_msg}")
    
    def load_validation_dialog(self):
        """Open dialog to load validation data."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Validation Data",
            "",
            "Data Files (*.csv *.xlsx *.xls);;All Files (*)"
        )
        
        if file_path and self.data_manager:
            try:
                self.data_manager.load_validation_data(file_path)
                self.status_label.setText(
                    f"Validation data loaded: {len(self.data_manager.validation_data)} samples"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load validation data:\n{str(e)}")
    
    def _update_data_preview(self):
        """Update data preview table."""
        if self.data_manager is None or self.data_manager.data is None:
            return
        
        df = self.data_manager.data
        
        # Show first 100 rows
        preview_df = df.head(100)
        
        self.data_table.setColumnCount(len(preview_df.columns))
        self.data_table.setRowCount(len(preview_df))
        self.data_table.setHorizontalHeaderLabels(preview_df.columns.tolist())
        
        for i, row in enumerate(preview_df.values):
            for j, value in enumerate(row):
                item = QTableWidgetItem(str(value))
                self.data_table.setItem(i, j, item)
        
        self.data_table.resizeColumnsToContents()
    
    def _update_column_details(self, info):
        """Update column details table."""
        self.col_table.setRowCount(len(info.columns))
        
        for i, (col_name, col_info) in enumerate(info.columns.items()):
            self.col_table.setItem(i, 0, QTableWidgetItem(col_name))
            self.col_table.setItem(i, 1, QTableWidgetItem(col_info.dtype))
            self.col_table.setItem(i, 2, QTableWidgetItem("Yes" if col_info.is_numeric else "No"))
            self.col_table.setItem(i, 3, QTableWidgetItem("Yes" if col_info.is_categorical else "No"))
            self.col_table.setItem(i, 4, QTableWidgetItem("Yes" if col_info.is_composition else "No"))
            self.col_table.setItem(i, 5, QTableWidgetItem(str(col_info.null_count)))
    
    def _update_dataset_info(self, info):
        """Update dataset info text."""
        text = f"""
        <b>Dataset Summary:</b><br>
        Samples: {info.n_samples:,}<br>
        Features: {info.n_features}<br>
        Memory Usage: {info.memory_usage_mb:.2f} MB<br>
        Missing Values: {'Yes' if info.has_missing_values else 'No'}<br>
        """
        self.info_text.setHtml(text)
    
    def _on_target_changed(self, target_column: str):
        """Handle target column selection."""
        if self.data_manager and target_column:
            try:
                self.data_manager.set_target_column(target_column)
                problem_type = "classification" if self.data_manager.is_classification else "regression"
                self.status_label.setText(f"Target: {target_column} ({problem_type})")
            except Exception as e:
                QMessageBox.warning(self, "Warning", str(e))
    
    def _apply_preprocessing(self):
        """Apply preprocessing pipeline."""
        if self.data_manager is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        try:
            preprocessor = self.data_manager.get_preprocessor(
                numeric_strategy=self.scale_combo.currentText(),
                categorical_strategy=self.encode_combo.currentText(),
                impute_strategy=self.impute_combo.currentText()
            )
            
            QMessageBox.information(
                self,
                "Success",
                "Preprocessing pipeline configured successfully.\n"
                "It will be applied when training models."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply preprocessing:\n{str(e)}")
    
    def _apply_feature_engineering(self):
        """Apply feature engineering."""
        if self.data_manager is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        comp_columns = self.data_manager.get_composition_columns()
        
        if not comp_columns:
            QMessageBox.information(
                self,
                "Info",
                "No composition columns detected in the dataset."
            )
            return
        
        try:
            # Guardar numero original de columnas para mostrar en el mensaje
            n_original_cols = len(self.data_manager._data.columns)
            
            for col in comp_columns:
                if self.comp_check.isChecked():
                    # FIX: Usar implementacion propia (no requiere pymatgen)
                    result = self.feature_engineer.add_composition_features(
                        self.data_manager.data,
                        composition_column=col,
                        use_magpie=True,      # Implementacion propia (106 descriptores)
                        use_matminer=False    # Deshabilitado - requiere pymatgen
                    )
                    self.data_manager._data = result
            
            # FIX: Actualizar feature columns despues de agregar nuevas features
            if self.data_manager._target_column:
                self.data_manager._feature_columns = [
                    c for c in self.data_manager._data.columns 
                    if c != self.data_manager._target_column
                ]
            
            # FIX: Resetear nombres procesados para que se recalculen
            self.data_manager._processed_feature_names = None
            
            self._update_data_preview()
            
            # Update info
            info = self.data_manager._analyze_dataset()
            self._update_dataset_info(info)
            self._update_column_details(info)
            
            # Calcular cuantas columnas nuevas se agregaron
            n_new_cols = len(self.data_manager._data.columns) - n_original_cols
            
            QMessageBox.information(
                self,
                "Success",
                f"Feature engineering applied.\n"
                f"Original columns: {n_original_cols}\n"
                f"New columns added: {n_new_cols}\n"
                f"Total features for training: {len(self.data_manager._feature_columns)}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to apply feature engineering:\n{str(e)}")
    
    def get_data_manager(self) -> DataManager:
        """Get the data manager."""
        return self.data_manager
