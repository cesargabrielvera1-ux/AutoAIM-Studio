"""Data loading and preprocessing tab."""

import os
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QFileDialog, QMessageBox, QProgressBar, QCheckBox,
    QSplitter, QTextEdit, QHeaderView, QScrollArea, QFrame,
    QDialog, QInputDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.data_manager import DataManager
from ..core.feature_engineering import FeatureEngineer
from ..core.crystal_structure import (
    CrystalStructureLoader, CrystalStructureFeaturizer,
    CrystalStructureDatasetBuilder
)
from ..core.data_health_check import DataHealthChecker, HealthReport, IssueSeverity
from ..core.error_messages import ErrorMessageTranslator, show_friendly_error
from ..core.dataset_exporter import DatasetExporter


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
        # v2.0 FIX: Same as ResultsTab — self.parent overwrites PyQt's parent() method.
        # Use _main_window to access the MainWindow after addTab() reparents us.
        self._main_window = parent
        self.data_manager: DataManager = None
        self.feature_engineer = FeatureEngineer()
        
        # Simple logger using print (no LoggerMixin dependency)
        self.logger = self._SimpleLogger()
        
        self._init_ui()
    
    class _SimpleLogger:
        """Simple logger that prints to console. No external dependencies."""
        def info(self, msg): print(f"[INFO] DataTab: {msg}")
        def warning(self, msg): print(f"[WARNING] DataTab: {msg}")
        def error(self, msg): print(f"[ERROR] DataTab: {msg}")
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Top controls
        controls_layout = QHBoxLayout()
        
        self.load_btn = QPushButton("Load Data")
        self.load_btn.setToolTip("Load a CSV or Excel file with your material data. Each row is a sample, each column is a feature or property.")
        self.load_btn.clicked.connect(self.load_data_dialog)
        controls_layout.addWidget(self.load_btn)
        
        self.load_val_btn = QPushButton("Load Validation Data")
        self.load_val_btn.setToolTip(
            "Load a separate validation dataset to test model performance on unseen data. "
            "Optional but highly recommended for reliable results."
        )
        self.load_val_btn.clicked.connect(self.load_validation_dialog)
        self.load_val_btn.setEnabled(False)
        controls_layout.addWidget(self.load_val_btn)
        
        controls_layout.addSpacing(20)
        
        # Crystal structure loading (v1.2.0)
        self.load_xtal_btn = QPushButton("Load Crystal Structures")
        self.load_xtal_btn.setToolTip(
            "Load crystal structure files (CIF, POSCAR, XYZ). AutoAIM will extract "
            "~154 structural descriptors (lattice, composition, coordination, etc.) automatically."
        )
        self.load_xtal_btn.setStyleSheet("background-color: #6A1B9A; color: white;")
        self.load_xtal_btn.clicked.connect(self.load_crystal_dialog)
        controls_layout.addWidget(self.load_xtal_btn)
        
        self.load_xtal_dir_btn = QPushButton("Load Crystal Directory")
        self.load_xtal_dir_btn.setToolTip("Load all structures from a directory")
        self.load_xtal_dir_btn.setStyleSheet("background-color: #6A1B9A; color: white;")
        self.load_xtal_dir_btn.clicked.connect(self.load_crystal_directory_dialog)
        controls_layout.addWidget(self.load_xtal_dir_btn)

        controls_layout.addSpacing(10)

        # v2.0: Molecular structure loading (.xyz files)
        self.load_mol_btn = QPushButton("Load Molecular Structures")
        self.load_mol_btn.setToolTip(
            "Load molecular structure files (.xyz). AutoAIM will extract ~149 "
            "molecular descriptors (composition, geometry, topology, periodic properties, "
            "oxidation states) automatically.\n\n"
            "Molecules are individual structures WITHOUT a unit cell. "
            "For periodic structures with a unit cell, use Load Crystal Structures instead."
        )
        self.load_mol_btn.setStyleSheet("background-color: #00695C; color: white;")
        self.load_mol_btn.clicked.connect(self.load_molecular_dialog)
        controls_layout.addWidget(self.load_mol_btn)

        self.load_mol_dir_btn = QPushButton("Load Molecular Directory")
        self.load_mol_dir_btn.setToolTip(
            "Load all .xyz files from a directory. Each file should contain "
            "one molecule with element symbols and 3D coordinates."
        )
        self.load_mol_dir_btn.setStyleSheet("background-color: #00695C; color: white;")
        self.load_mol_dir_btn.clicked.connect(self.load_molecular_directory_dialog)
        controls_layout.addWidget(self.load_mol_dir_btn)

        # Data Health Check button
        self.health_btn = QPushButton("Data Health Check")
        self.health_btn.setToolTip(
            "Analyze data quality: missing values, outliers, duplicates, correlations, "
            "and get actionable recommendations before training."
        )
        self.health_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0;
                color: white;
                font-weight: bold;
                padding: 5px 15px;
            }
            QPushButton:hover { background-color: #1976D2; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.health_btn.clicked.connect(self._run_health_check)
        self.health_btn.setEnabled(False)
        controls_layout.addWidget(self.health_btn)
        
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
        self.impute_combo.setToolTip(
            "How to fill missing values:\n"
            "• mean/median: fast, works for most datasets\n"
            "• knn: most accurate but slower (uses nearest neighbors)\n"
            "• most_frequent: best for categorical columns"
        )
        prep_layout.addWidget(QLabel("Imputation Strategy:"))
        prep_layout.addWidget(self.impute_combo)
        
        self.scale_combo = QComboBox()
        self.scale_combo.addItems(['standard', 'minmax', 'robust', 'none'])
        self.scale_combo.setToolTip(
            "How to scale numeric features:\n"
            "• standard: zero mean, unit variance (best for SVM, Neural Networks)\n"
            "• robust: resistant to outliers (best if data has extreme values)\n"
            "• minmax: scales to [0, 1] range\n"
            "• none: no scaling (Random Forest and Gradient Boosting don't need it)"
        )
        prep_layout.addWidget(QLabel("Scaling Strategy:"))
        prep_layout.addWidget(self.scale_combo)
        
        self.encode_combo = QComboBox()
        self.encode_combo.addItems(['onehot', 'label', 'none'])
        self.encode_combo.setToolTip(
            "How to convert categorical (text) columns to numbers:\n"
            "• onehot: creates binary columns (best for most models)\n"
            "• label: assigns numbers 0,1,2... (only for tree-based models)\n"
            "• none: keeps as-is (may cause errors with some models)"
        )
        prep_layout.addWidget(QLabel("Encoding Strategy:"))
        prep_layout.addWidget(self.encode_combo)
        
        self.apply_prep_btn = QPushButton("Apply Preprocessing")
        self.apply_prep_btn.setToolTip(
            "Configure the preprocessing pipeline. It will be applied automatically "
            "when you train models. You can change these settings anytime."
        )
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
        self.comp_check.setToolTip(
            "When checked, AutoAIM will compute 106 Magpie descriptors from chemical formulas "
            "(atomic number stats, electronegativity, thermal conductivity, etc.). "
            "Requires a column named 'formula', 'composition', or 'chemical_formula'."
        )
        fe_layout.addWidget(self.comp_check)
        
        self.apply_fe_btn = QPushButton("Apply Feature Engineering")
        self.apply_fe_btn.setToolTip(
            "Extract composition-based features from chemical formulas. "
            "This adds ~106 new numeric columns that can significantly improve model accuracy."
        )
        self.apply_fe_btn.clicked.connect(self._apply_feature_engineering)
        self.apply_fe_btn.setEnabled(False)
        fe_layout.addWidget(self.apply_fe_btn)
        
        # v2.0: Link Crystal Structures buttons
        self.link_crystal_btn = QPushButton("Link Crystal Structures")
        self.link_crystal_btn.setToolTip(
            "Link CIF structure files to your dataset via an ID column. "
            "Extracts crystal descriptors (density, cell parameters, space group, etc.) "
            "and merges them with your tabular data. Replaces composition features "
            "if they already exist (to avoid duplicates)."
        )
        self.link_crystal_btn.clicked.connect(self._on_link_crystal_structures)
        self.link_crystal_btn.setEnabled(False)
        fe_layout.addWidget(self.link_crystal_btn)

        self.unlink_crystal_btn = QPushButton("Unlink Crystals")
        self.unlink_crystal_btn.setToolTip(
            "Remove linked crystal structure features and restore the ID column "
            "as a regular feature. Re-enables the composition features checkbox."
        )
        self.unlink_crystal_btn.clicked.connect(self._on_unlink_crystal_structures)
        self.unlink_crystal_btn.setEnabled(False)
        self.unlink_crystal_btn.setStyleSheet("background-color: #d32f2f; color: white;")
        fe_layout.addWidget(self.unlink_crystal_btn)

        fe_layout.addSpacing(10)

        # v2.0: Link Molecular Structures (.xyz files)
        self.link_mol_btn = QPushButton("Link Molecular Structures")
        self.link_mol_btn.setToolTip(
            "Link .xyz molecular structure files to your dataset via an ID column. "
            "Extracts 149 molecular descriptors (composition, geometry 3D, topology, "
            "34 periodic properties, oxidation states) and merges them with your data.\n\n"
            "Molecules are individual structures WITHOUT a unit cell — "
            "for periodic structures use Link Crystal Structures instead."
        )
        self.link_mol_btn.clicked.connect(self._on_link_molecular_structures)
        self.link_mol_btn.setEnabled(False)
        fe_layout.addWidget(self.link_mol_btn)

        self.unlink_mol_btn = QPushButton("Unlink Molecules")
        self.unlink_mol_btn.setToolTip("Remove linked molecular structure features from dataset")
        self.unlink_mol_btn.clicked.connect(self._on_unlink_molecular_structures)
        self.unlink_mol_btn.setEnabled(False)
        self.unlink_mol_btn.setStyleSheet("background-color: #d32f2f; color: white;")
        fe_layout.addWidget(self.unlink_mol_btn)

        col_layout.addLayout(fe_layout)

        # v2.0: Export featurized dataset
        export_layout = QHBoxLayout()
        export_layout.addStretch()

        self.export_dataset_btn = QPushButton("Export Dataset")
        self.export_dataset_btn.setToolTip(
            "Export the current featurized dataset to a CSV file. "
            "Includes all features: original columns, composition descriptors, "
            "linked crystal/molecular features, and any feature engineering applied."
        )
        self.export_dataset_btn.setEnabled(False)
        self.export_dataset_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                padding: 6px 16px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.export_dataset_btn.clicked.connect(self._export_dataset)
        export_layout.addWidget(self.export_dataset_btn)

        col_layout.addLayout(export_layout)

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
        self.health_btn.setEnabled(True)
        self.link_crystal_btn.setEnabled(True)
        self.unlink_crystal_btn.setEnabled(False)  # Fresh data — nothing linked yet
        self.link_mol_btn.setEnabled(True)
        self.unlink_mol_btn.setEnabled(False)
        self.export_dataset_btn.setEnabled(True)
        
        # Notify parent
        if self._main_window:
            self._main_window.data_tab = self
    
    def _on_load_error(self, error_msg: str):
        """Handle load error with user-friendly message."""
        self.progress_bar.setVisible(False)
        self.status_label.setText("Error loading data")
        # Use friendly error translator
        try:
            exc = Exception(error_msg)
            friendly = ErrorMessageTranslator.translate(exc, "loading data")
            dialog_text = ErrorMessageTranslator.get_dialog_text(friendly)
            msg_box = QMessageBox(self)
            msg_box.setIcon(QMessageBox.Icon.Critical)
            msg_box.setWindowTitle(friendly.title)
            msg_box.setTextFormat(Qt.TextFormat.RichText)
            msg_box.setText(dialog_text)
            msg_box.exec()
        except Exception:
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
                
                # Auto-featurize validation data if training data has comp_* features
                self._sync_validation_featurization()
                
                # v1.5.0: Validate column alignment between train and val
                col_report = self.data_manager.validate_columns()
                if not col_report['valid']:
                    QMessageBox.warning(
                        self, "Validation Column Mismatch",
                        f"{col_report['message']}\n\n"
                        f"Training features: {col_report['train_features']}\n"
                        f"Validation features: {col_report['val_features']}\n\n"
                        f"If both datasets should have the same columns, verify they "
                        f"were featurized with the same method before loading."
                    )
                
                val_data = self.data_manager.validation_data
                self.status_label.setText(
                    f"Validation data loaded: {len(val_data)} samples, "
                    f"{len(val_data.columns)} columns"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load validation data:\n{str(e)}")
    
    def _sync_validation_featurization(self):
        """Apply featurizer to validation data if training data has comp_* features.
        
        This ensures validation data has the same feature columns as training data,
        preventing 'comp_* not in index' errors during prepare_data().
        """
        if self.data_manager is None:
            return
        if self.data_manager.data is None or self.data_manager.validation_data is None:
            return
        
        train_data = self.data_manager.data
        val_data = self.data_manager.validation_data
        
        # Check if training has comp_* columns that validation is missing
        train_comp_cols = [c for c in train_data.columns if c.startswith('comp_')]
        val_has_comp = any(c.startswith('comp_') for c in val_data.columns)
        
        if not train_comp_cols or val_has_comp:
            return  # No sync needed
        
        # Training has comp_* but validation doesn't - need to featurize validation
        self.logger.info(
            f"Training has {len(train_comp_cols)} comp_* columns that validation "
            f"is missing. Auto-featurizing validation data..."
        )
        
        # Find composition column in validation data
        val_comp_col = None
        for col in val_data.columns:
            if col.lower() in ('formula', 'composition', 'chemical_formula'):
                val_comp_col = col
                break
        
        if val_comp_col is None:
            self.logger.warning(
                "Cannot auto-featurize validation: no 'formula' or 'composition' "
                "column found. Validation data must have the same comp_* columns "
                "as training data, or a 'formula' column for auto-featurization."
            )
            return
        
        # Apply same featurizer to validation data
        try:
            result = self.feature_engineer.add_composition_features(
                val_data,
                composition_column=val_comp_col,
                use_magpie=True,
                use_matminer=False
            )
            self.data_manager._validation_data = result
            
            new_comp_cols = [c for c in result.columns if c.startswith('comp_')]
            self.logger.info(
                f"Auto-featurized validation: {len(new_comp_cols)} comp_* columns added"
            )
        except Exception as e:
            self.logger.warning(f"Auto-featurization of validation failed: {e}")
    
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
            # v1.2.0: Check if dataset already has crystal structure features
            # (auto-generated when loading CIF/POSCAR/XYZ files)
            cols = list(self.data_manager._data.columns)
            has_crystal_features = any(c.startswith('lattice_') for c in cols) and \
                                   any(c.startswith('volume_per_atom') for c in cols)
            
            if has_crystal_features:
                # Count crystal feature categories
                n_lattice = sum(1 for c in cols if c.startswith('lattice_'))
                n_comp = sum(1 for c in cols if c.startswith('comp_'))
                n_magpie = sum(1 for c in cols if c.startswith('magpie_'))
                n_adv = sum(1 for c in cols if c in ['mean_bond_length', 'mean_coordination_number',
                          'rdf_peak_1', 'structure_complexity_index', 'lattice_anisotropy'])
                
                QMessageBox.information(
                    self,
                    "Crystal Features Already Present",
                    f"This dataset already contains automatically-generated crystal "
                    f"structure features ({len(cols)} total).\n\n"
                    f"The following feature groups are already included:\n"
                    f"  - Lattice features: ~{n_lattice}\n"
                    f"  - Composition features: ~{n_comp}\n"
                    f"  - Magpie descriptors: ~{n_magpie}\n"
                    f"  - Advanced structural: ~{n_adv}\n\n"
                    f"No additional featurization is needed for crystal structures. "
                    f"The manual featurizer only works on CSV files with a "
                    f"'formula' or 'composition' column containing chemical formulas "
                    f"(e.g., 'Fe2O3', 'SiO2')."
                )
            else:
                QMessageBox.information(
                    self,
                    "Info",
                    "No composition columns detected in the dataset.\n\n"
                    "The featurizer requires a column named 'formula', 'composition', "
                    "or 'chemical_formula' with chemical formulas (e.g., 'Fe2O3')."
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
                    
                    # FIX v1.2.0: Also apply featurizer to validation data if loaded
                    # This ensures validation data has the same comp_* columns as training
                    if (self.data_manager._validation_data is not None and 
                            col in self.data_manager._validation_data.columns):
                        try:
                            val_result = self.feature_engineer.add_composition_features(
                                self.data_manager._validation_data,
                                composition_column=col,
                                use_magpie=True,
                                use_matminer=False
                            )
                            self.data_manager._validation_data = val_result
                            self.logger.info(
                                f"Applied featurizer to validation data: "
                                f"{len(val_result.columns)} columns"
                            )
                        except Exception as e:
                            self.logger.warning(
                                f"Could not apply featurizer to validation data: {e}"
                            )
            
            # FIX: Actualizar feature columns despues de agregar nuevas features.
            # v1.2.0: Solo columnas numericas — columnas string (structure_name,
            # file paths, etc.) NO deben ser features.
            if self.data_manager._target_column:
                numeric_cols = self.data_manager._data.select_dtypes(
                    include=[np.number]
                ).columns.tolist()
                self.data_manager._feature_columns = [
                    c for c in numeric_cols 
                    if c != self.data_manager._target_column
                ]
            
            # FIX: Resetear nombres procesados para que se recalculen
            self.data_manager._processed_feature_names = None
            
            self._update_data_preview()
            
            # Update info
            info = self.data_manager.get_column_info()
            self._update_dataset_info(info)
            self._update_column_details(info)
            
            # Calculate how many new columns were added
            n_new_cols = len(self.data_manager.data.columns) - n_original_cols
            
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

    # ------------------------------------------------------------------
    # v2.0: Link Crystal Structures
    # ------------------------------------------------------------------

    def _on_link_crystal_structures(self):
        """Open dialog to link crystal structure CIF files with the dataset.

        This is the main entry point for the Link Crystal Structures feature.
        It guides the user through selecting an ID column and CIF folder,
        then extracts crystal features and merges them into the dataset.
        """
        if self.data_manager is None or self.data_manager.data is None:
            QMessageBox.warning(self, "Warning", "No data loaded. Please load a CSV file first.")
            return

        from ..core.crystal_feature_extractor import CrystalFeatureExtractor

        df = self.data_manager.data
        extractor = CrystalFeatureExtractor()

        if not extractor.has_pymatgen:
            QMessageBox.warning(
                self, "pymatgen Required",
                "Crystal structure linking requires pymatgen.\n\n"
                "Install it with:\n"
                "  pip install pymatgen\n\n"
                "Then restart AutoAIM."
            )
            return

        # 1. Detect ID column candidates
        candidates = extractor.detect_id_candidates(df)
        if not candidates:
            QMessageBox.information(
                self, "No ID Column Found",
                "Your dataset needs a column that uniquely identifies each material "
                "and matches your CIF filenames.\n\n"
                "For example, add a 'material_id' column:\n"
                "  material_id | formula | band_gap\n"
                "  mp-1234     | TiO2    | 3.0     ←  TiO2.cif or mp-1234.cif\n"
                "  mp-5678     | Fe2O3   | 2.1     ←  Fe2O3.cif or mp-5678.cif\n\n"
                "The ID values should match the CIF filenames (without .cif extension)."
            )
            return

        # 2. Show ID column selection dialog
        candidate_items = [f"{name}  ({n_unique} unique values, {n_unique/len(df)*100:.0f}%)" for name, n_unique in candidates]
        candidate_keys = [c[0] for c in candidates]  # Pure column names

        selected_text, ok = QInputDialog.getItem(
            self, "Select ID Column",
            "Choose the column that matches your CIF filenames:\n\n"
            "(CIF files should be named {ID_value}.cif)",
            candidate_items, 0, False
        )
        if not ok or not selected_text:
            return  # User cancelled

        # Extract column name from selected text (before the "  (")
        id_column = candidate_keys[candidate_items.index(selected_text)]

        # 3. Select CIF folder
        cif_folder = QFileDialog.getExistingDirectory(
            self, "Select Folder with CIF Files", ""
        )
        if not cif_folder:
            return

        # 4. Check CIF availability
        availability = extractor.check_cif_availability(df, id_column, cif_folder)

        if availability.get('error'):
            QMessageBox.critical(self, "Error", availability['error'])
            return

        # 5. Scan for non-crystalline structures (v2.0)
        non_crystalline = extractor.scan_for_noncrystalline(df, id_column, cif_folder)
        
        # 6. Show warnings if needed
        warnings_text = []
        if availability['duplicate_ids']:
            dupes = availability['duplicate_ids'][:5]
            warnings_text.append(
                f"WARNING: {len(availability['duplicate_ids'])} IDs appear multiple times "
                f"in your dataset (e.g., {', '.join(str(d) for d in dupes)}).\n"
                f"Multiple rows with the same ID will receive identical crystal features."
            )

        if availability['missing']:
            warnings_text.append(
                f"WARNING: {len(availability['missing'])} rows have no matching CIF file.\n"
                f"Crystal features for these rows will be NaN (filled by imputation)."
            )

        # v2.0: Non-crystalline structure warning
        if non_crystalline:
            nc_count = len(non_crystalline)
            nc_examples = non_crystalline[:3]
            nc_text = "\n".join(
                f"  • {mid}: {reason[:120]}..." if len(reason) > 120 else f"  • {mid}: {reason}"
                for mid, reason in nc_examples
            )
            warnings_text.append(
                f"WARNING: {nc_count} structure(s) appear to be isolated molecules "
                f"in artificial simulation boxes, not periodic crystals:\n"
                f"{nc_text}\n\n"
                f"Crystal descriptors for these structures will be numerically valid "
                f"but physically meaningless (e.g., density reflects the box size, "
                f"not the material).\n\n"
                f"This tool is designed for periodic crystalline materials. "
                f"For isolated organic molecules, consider using molecular descriptors "
                f"(e.g., RDKit) instead."
            )

        if availability['matched'] == 0:
            QMessageBox.critical(
                self, "No CIF Files Matched",
                f"Found {availability['n_cif_files']} CIF files in the folder, "
                f"but none match the '{id_column}' values.\n\n"
                f"Example: if {id_column}='mp-1234', the CIF should be named 'mp-1234.cif'.\n\n"
                f"Please check that your CIF filenames match the ID values exactly."
            )
            return

        # Show pre-link summary with warnings
        msg = (
            f"ID Column: {id_column}\n"
            f"CIF Folder: {cif_folder}\n"
            f"Total rows: {availability['n_ids']}\n"
            f"Matched CIFs: {len(availability['matched'])}\n"
            f"Missing CIFs: {len(availability['missing'])}\n"
        )
        if warnings_text:
            msg += "\n" + "\n\n".join(warnings_text)
        msg += "\n\nProceed with linking?"

        reply = QMessageBox.question(
            self, "Link Crystal Structures", msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        # 6. Check if composition features already exist
        comp_cols = [c for c in df.columns if c.startswith('comp_')]
        replace_comp = False
        if comp_cols:
            reply = QMessageBox.question(
                self, "Composition Features Detected",
                f"Your dataset already has {len(comp_cols)} composition features (comp_*) "
                f"from a previous featurization.\n\n"
                f"Crystal structure files also contain composition descriptors. "
                f"Keeping both would create duplicate columns.\n\n"
                f"Remove existing comp_* columns and replace with CIF-derived ones?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            replace_comp = (reply == QMessageBox.StandardButton.Yes)

        # 7. Perform the linking
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.status_label.setText("Extracting crystal features from CIFs...")

        try:
            df_new, report = extractor.link_structures(
                df, id_column, cif_folder,
                replace_composition_features=replace_comp
            )

            # Update data manager
            self.data_manager._data = df_new
            self.data_manager.set_id_column(id_column)
            self.data_manager.set_crystal_extractor(extractor)

            # Update feature columns (exclude ID column)
            if self.data_manager._target_column:
                numeric_cols = df_new.select_dtypes(include=[np.number]).columns.tolist()
                id_col = self.data_manager.id_column
                self.data_manager._feature_columns = [
                    c for c in numeric_cols
                    if c != self.data_manager._target_column and c != id_col
                ]
            self.data_manager._processed_feature_names = None

            # Disable Magpie checkbox since crystal features include composition
            self.comp_check.setChecked(False)
            self.comp_check.setEnabled(False)
            self.comp_check.setToolTip(
                "Composition features are already included in the linked crystal structures. "
                "Unlink crystal structures to re-enable this option."
            )

            # Update button states
            self.link_crystal_btn.setEnabled(False)
            self.unlink_crystal_btn.setEnabled(True)

            self._update_data_preview()
            info = self.data_manager.get_column_info()
            self._update_dataset_info(info)
            self._update_column_details(info)

            # Show success report
            msg = (
                f"Crystal structures linked successfully!\n\n"
                f"{report.to_text()}\n\n"
                f"Total features for training: {len(self.data_manager._feature_columns)}"
            )
            QMessageBox.information(self, "Link Complete", msg)
            self.status_label.setText(
                f"Crystal linked: {report.n_matched}/{report.n_rows_total} CIFs matched, "
                f"{len(report.crystal_features_added)} features added"
            )

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to link crystal structures:\n{str(e)}")
            self.status_label.setText(f"Crystal linking failed: {str(e)[:80]}")
        finally:
            self.progress_bar.setVisible(False)

    def _on_unlink_crystal_structures(self):
        """Remove linked crystal structure features and restore previous state."""
        if self.data_manager is None or not self.data_manager.crystal_features_active:
            QMessageBox.information(self, "Info", "No crystal structures are currently linked.")
            return

        reply = QMessageBox.question(
            self, "Confirm Unlink",
            "This will remove all crystal structure features from the dataset "
            "and restore the ID column as a regular feature.\n\n"
            "Proceed?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            self.data_manager.unset_crystal_link()

            # Re-enable Magpie checkbox
            self.comp_check.setEnabled(True)
            self.comp_check.setToolTip(
                "When checked, AutoAIM will compute 106 Magpie descriptors from chemical formulas "
                "(atomic number stats, electronegativity, thermal conductivity, etc.). "
                "Requires a column named 'formula', 'composition', or 'chemical_formula'."
            )

            # Update button states
            self.link_crystal_btn.setEnabled(True)
            self.unlink_crystal_btn.setEnabled(False)

            self._update_data_preview()
            info = self.data_manager.get_column_info()
            self._update_dataset_info(info)
            self._update_column_details(info)

            QMessageBox.information(
                self, "Unlink Complete",
                "Crystal structure features have been removed.\n\n"
                "The ID column has been restored as a regular feature.\n"
                "You can now re-link with different settings or use composition features."
            )
            self.status_label.setText("Crystal structures unlinked")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to unlink crystal structures:\n{str(e)}")
            self.status_label.setText(f"Unlink failed: {str(e)[:80]}")

    # ------------------------------------------------------------------
    # Crystal structure loading (v1.2.0)
    # ------------------------------------------------------------------
    
    def load_crystal_dialog(self):
        """Open dialog to load crystal structure files (CIF, POSCAR, XYZ)."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Crystal Structure Files",
            "",
            "CIF Files (*.cif);;POSCAR Files (POSCAR*);;VASP Files (*.poscar *.vasp);;"
            "XYZ Files (*.xyz);;All Structure Files (*.cif *.poscar *.xyz *.vasp)"
        )
        
        if file_paths:
            self._load_crystal_structures(file_paths=file_paths)
    
    def load_crystal_directory_dialog(self):
        """Open dialog to load a directory of crystal structures."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Crystal Structures",
            ""
        )
        
        if directory:
            self._load_crystal_structures(directory=directory)
    
    def _load_crystal_structures(self, file_paths=None, directory=None):
        """Load and featurize crystal structures.
        
        Args:
            file_paths: List of structure file paths
            directory: Directory containing structure files
        """
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Loading crystal structures...")
        
        try:
            builder = CrystalStructureDatasetBuilder()
            
            if directory:
                # Ask for target CSV
                target_df = self._load_target_csv_dialog()
                
                if target_df is not None:
                    df = builder.build_from_directory(
                        directory,
                        target_values=None,
                        target_column='target'
                    )
                    # Merge with target
                    df = self._merge_targets(df, target_df)
                else:
                    df = builder.build_from_directory(
                        directory,
                        target_values=None,
                        target_column='target'
                    )
            else:
                target_df = self._load_target_csv_dialog()
                
                if target_df is not None:
                    df = builder.build_from_files(
                        file_paths,
                        target_values=None,
                        target_column='target'
                    )
                    df = self._merge_targets(df, target_df)
                else:
                    df = builder.build_from_files(
                        file_paths,
                        target_values=None,
                        target_column='target'
                    )
            
            self.progress_bar.setVisible(False)
            
            if df.empty:
                QMessageBox.warning(
                    self, "Warning",
                    "No crystal structure features extracted."
                )
                return
            
            # Store the DataFrame in data_manager
            if self.data_manager is None:
                self.data_manager = DataManager()
            
            self.data_manager._data = df
            
            # v1.2.0 FIX: Keep ONLY numeric columns as features.
            # After merging with target CSV, non-numeric columns from the CSV
            # (composition strings, file paths, etc.) would be included as features
            # and one-hot encoded, exploding feature count from ~154 to 600+.
            # We explicitly select only numeric columns, excluding structure_name
            # and any target column.
            target_col = 'target'  # default target column name
            if target_col in df.columns:
                # Identify which columns are numeric (excluding structure_name)
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                # Exclude the target column from features
                self.data_manager._feature_columns = [
                    c for c in numeric_cols if c != target_col
                ]
            else:
                # No target yet — let user select it
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                self.data_manager._feature_columns = numeric_cols
            
            self.logger.info(
                f"Crystal features: {len(self.data_manager._feature_columns)} numeric columns. "
                f"Excluded: {[c for c in df.columns if c not in self.data_manager._feature_columns and c != target_col]}"
            )
            
            # Update UI
            info = self.data_manager.get_column_info()
            
            # Update target combo
            self.target_combo.clear()
            self.target_combo.addItems(list(info.columns.keys()))
            
            # Suggest 'target' column if present
            if 'target' in info.columns:
                self.target_combo.setCurrentText('target')
                self.data_manager.set_target_column('target')
                # Recalculate feature columns after target is set
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                self.data_manager._feature_columns = [
                    c for c in numeric_cols if c != 'target'
                ]
            
            self._update_data_preview()
            self._update_column_details(info)
            self._update_dataset_info(info)
            
            # Enable buttons
            self.load_val_btn.setEnabled(True)
            self.apply_prep_btn.setEnabled(True)
            self.apply_fe_btn.setEnabled(True)
            
            if self._main_window:
                self._main_window.data_tab = self
            
            self.status_label.setText(
                f"Crystal structures: {len(df)} samples, {len(df.columns)} features"
            )
            
            # Report any failed files
            if builder.loader.failed_files:
                failed_names = [Path(fp).name for fp, _ in builder.loader.failed_files]
                QMessageBox.warning(
                    self, "Partial Load",
                    f"Loaded {len(df)} structures.\n"
                    f"Failed to load {len(failed_names)} files:\n"
                    + "\n".join(failed_names[:10])
                    + ("\n..." if len(failed_names) > 10 else "")
                )
            else:
                QMessageBox.information(
                    self, "Success",
                    f"Loaded {len(df)} crystal structures with "
                    f"{len([c for c in df.columns if c not in ['structure_name', 'target']])} features.\n\n"
                    f"Please select the target column and proceed to Training."
                )
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(
                self, "Error",
                f"Failed to load crystal structures:\n{str(e)}"
            )

    # ------------------------------------------------------------------
    # Molecular structure loading & linking (v2.0)
    # ------------------------------------------------------------------

    def load_molecular_dialog(self):
        """Open file dialog to load individual .xyz molecular structure files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Load Molecular Structures", "",
            "XYZ Files (*.xyz);;All Files (*)"
        )
        if file_paths:
            self._load_molecular_structures(file_paths=file_paths)

    def load_molecular_directory_dialog(self):
        """Open directory dialog to load all .xyz files from a folder."""
        directory = QFileDialog.getExistingDirectory(
            self, "Load Molecular Directory"
        )
        if directory:
            self._load_molecular_structures(directory=directory)

    def _load_molecular_structures(self, file_paths=None, directory=None):
        """Load and featurize molecular .xyz structures.

        Creates a MoleculeFeaturizerThread to extract ~149 molecular
        descriptors from each .xyz file.
        """
        from ..core.molecule_featurizer import MoleculeFeaturizer

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("Featurizing molecular structures...")

        try:
            featurizer = MoleculeFeaturizer()

            if directory:
                import glob
                file_paths = sorted(glob.glob(os.path.join(directory, "*.xyz")))

            if not file_paths:
                QMessageBox.warning(self, "No Files", "No .xyz files found.")
                return

            # Featurize all files
            results = []
            errors = []
            for fp in file_paths:
                try:
                    df = featurizer.featurize_file(fp)
                    name = os.path.splitext(os.path.basename(fp))[0]
                    df.insert(0, 'molecule_name', name)
                    results.append(df)
                except Exception as e:
                    err_msg = f"  • {os.path.basename(fp)}: {type(e).__name__}: {e}"
                    errors.append(err_msg)
                    self.logger.warning(f"Failed to featurize {fp}: {e}")

            if not results:
                error_detail = "\n".join(errors[:10])
                if len(errors) > 10:
                    error_detail += f"\n  ... and {len(errors) - 10} more files"
                QMessageBox.critical(
                    self, "Error",
                    f"No .xyz files could be featurized.\n\n"
                    f"Common causes:\n"
                    f"  • Files saved with BOM (Byte Order Mark) from Windows editors\n"
                    f"  • Missing atom count in first line\n"
                    f"  • Invalid coordinate format\n\n"
                    f"Errors per file:\n{error_detail}"
                )
                return
            elif errors:
                # Some files failed but not all — warn the user
                error_detail = "\n".join(errors[:5])
                if len(errors) > 5:
                    error_detail += f"\n  ... and {len(errors) - 5} more"
                self.logger.warning(f"{len(errors)}/{len(file_paths)} .xyz files failed featurization")
                # Non-blocking warning — continue since we have some results

            df = pd.concat(results, ignore_index=True)
            self._molecular_features_df = df
            self._molecular_id_column = 'molecule_name'

            # Show preview
            if hasattr(self, 'data_preview') and self.data_preview is not None:
                try:
                    self.data_preview.load_dataframe(df.head(100))
                    self.status_label.setText(
                        f"Loaded {len(df)} molecular structures with "
                        f"{len([c for c in df.columns if c not in ['molecule_name', 'mol_source_file']])} features."
                    )
                except Exception:
                    pass

            self.progress_bar.setVisible(False)
            self.status_label.setText(
                f"Loaded {len(df)} molecular structures. "
                f"Please select target column and proceed."
            )

            # Enable link button if we have data loaded
            if self.data_manager and self.data_manager.data is not None:
                self.link_mol_btn.setEnabled(True)

            QMessageBox.information(
                self, "Success",
                f"Loaded {len(df)} molecular structures with "
                f"{len([c for c in df.columns if c not in ['molecule_name', 'mol_source_file']])} features.\n\n"
                f"You can now:\n"
                f"1. Select a target column and train directly on these structures\n"
                f"2. Use 'Link Molecular Structures' to merge with your loaded CSV data"
            )

        except Exception as e:
            self.progress_bar.setVisible(False)
            QMessageBox.critical(
                self, "Error",
                f"Failed to load molecular structures:\n{str(e)}"
            )

    def _on_link_molecular_structures(self):
        """Link .xyz molecular structures to the loaded dataset via an ID column."""
        if not self.data_manager or self.data_manager.data is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return

        try:
            from ..core.molecule_feature_linker import MoleculeFeatureLinker

            # Ask user for .xyz folder
            xyz_folder = QFileDialog.getExistingDirectory(
                self, "Select Folder with .xyz Files"
            )
            if not xyz_folder:
                return

            # Get available ID columns
            df = self.data_manager.data
            id_candidates = [c for c in df.columns
                           if df[c].dtype == object or df[c].nunique() / len(df) > 0.5]

            if not id_candidates:
                QMessageBox.warning(
                    self, "No ID Columns",
                    "No suitable ID columns found. Need a column with unique identifiers "
                    "that match .xyz filenames."
                )
                return

            # Ask user to pick ID column
            item, ok = QInputDialog.getItem(
                self, "Select ID Column",
                "Choose the column that matches .xyz filenames (without extension):\n\n"
                "The values in this column should match the .xyz filenames exactly.",
                id_candidates, 0, False
            )
            if not ok or not item:
                return

            id_column = item

            # Pre-check availability
            linker = MoleculeFeatureLinker()
            avail = linker.check_xyz_availability(df, id_column, xyz_folder)

            if avail.get('n_xyz_files', 0) == 0:
                QMessageBox.warning(
                    self, "No .xyz Files",
                    f"No .xyz files found in:\n{xyz_folder}\n\n"
                    f"Please select a folder containing .xyz files."
                )
                return

            matched = len(avail.get('matched', []))
            missing = len(avail.get('missing', []))

            # Show summary and confirm
            msg = (f"Found {avail.get('n_xyz_files', 0)} .xyz files.\n\n"
                   f"Matched: {matched} rows\n"
                   f"Missing: {missing} rows (will be filled with NaN)\n\n"
                   f"Proceed with linking?")
            reply = QMessageBox.question(self, "Link Molecular Structures", msg)
            if reply != QMessageBox.StandardButton.Yes:
                return

            # Perform linking
            self.status_label.setText("Linking molecular structures...")
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 0)

            linked_df, report = linker.link_structures(df, id_column, xyz_folder)

            if report.n_matched == 0:
                QMessageBox.warning(
                    self, "No Matches",
                    "No .xyz files matched the ID column values. "
                    "Ensure filenames match exactly (without extension)."
                )
                self.progress_bar.setVisible(False)
                return

            # Update data manager with linked data
            # v2.0 FIX: Use _data (private) because 'data' is a read-only property.
            self.data_manager._data = linked_df
            self._update_data_preview()
            self._update_column_details(self.data_manager.get_column_info())

            # Store state
            self._molecular_features_df = linked_df
            self._molecular_id_column = id_column
            self._molecular_linker = linker

            self.status_label.setText(
                f"Linked {report.n_matched} molecular structures. "
                f"{len(report.molecule_features_added)} molecular features added."
            )
            self.progress_bar.setVisible(False)

            # Update button states
            self.link_mol_btn.setEnabled(False)
            self.unlink_mol_btn.setEnabled(True)

            # Update target combo
            old_target = self.target_combo.currentText()
            self.target_combo.clear()
            for col in linked_df.columns:
                if pd.api.types.is_numeric_dtype(linked_df[col]):
                    self.target_combo.addItem(col)
            idx = self.target_combo.findText(old_target)
            if idx >= 0:
                self.target_combo.setCurrentIndex(idx)

            QMessageBox.information(
                self, "Link Complete",
                f"Successfully linked {report.n_matched} molecular structures.\n\n"
                f"Features added: {len(report.molecule_features_added)}\n"
                f"Missing (NaN-filled): {report.n_missing}\n\n"
                f"You can now train models using these molecular features."
            )

        except Exception as e:
            self.progress_bar.setVisible(False)
            self.status_label.setText(f"Error: {str(e)[:100]}")
            QMessageBox.critical(self, "Error", f"Failed to link molecular structures:\n{str(e)}")

    def _on_unlink_molecular_structures(self):
        """Remove linked molecular features and restore original data."""
        if self._molecular_linker is None:
            return

        reply = QMessageBox.question(
            self, "Confirm Unlink",
            "Remove all linked molecular structure features?\n\n"
            "This will restore the dataset to its state before linking.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        try:
            # Remove molecular feature columns directly (robust fallback)
            if self.data_manager is not None and self.data_manager.data is not None:
                df = self.data_manager.data
                mol_cols = [c for c in df.columns
                           if c.startswith('mol_') or c.startswith('has_')]
                if mol_cols:
                    self.data_manager._data = df.drop(columns=mol_cols)
                    self.logger.info(f"Removed {len(mol_cols)} molecular feature columns")

                self._update_data_preview()
                self._update_column_details(self.data_manager.get_column_info())

            self._molecular_features_df = None
            self._molecular_id_column = None
            self._molecular_linker = None

            self.link_mol_btn.setEnabled(True)
            self.unlink_mol_btn.setEnabled(False)

            self.status_label.setText("Molecular structures unlinked.")
            QMessageBox.information(self, "Unlinked", "Molecular structure features removed.")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to unlink: {str(e)}")

    def _load_target_csv_dialog(self) -> Optional[pd.DataFrame]:
        """Ask user to load a CSV with target values and select target column.
        
        Only the 'structure_name' and the selected target column are returned.
        All other columns (composition, file paths, extra properties) are
        discarded to prevent them from being used as features.
        
        Returns:
            DataFrame with 'structure_name' and ONE target column, or None
        """
        reply = QMessageBox.question(
            self,
            "Load Target Values?",
            "Do you have a CSV file with target values for these structures?\n\n"
            "The CSV must have:\n"
            "- 'structure_name': name matching the structure files\n"
            "- ONE target column: the property you want to predict\n\n"
            "Other columns (composition, file paths, etc.) will be IGNORED.\n\n"
            "Click 'Yes' to select the CSV, or 'No' to skip.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return None
        
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Load Target Values CSV",
            "",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if not file_path:
            return None
        
        try:
            df = pd.read_csv(file_path, index_col=False)
            if 'structure_name' not in df.columns:
                QMessageBox.warning(
                    self, "Warning",
                    "CSV must have a 'structure_name' column matching structure file names."
                )
                return None
            
            # Let user select which column is the target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            target_candidates = [c for c in numeric_cols if c != 'structure_name']
            
            if not target_candidates:
                QMessageBox.warning(
                    self, "Warning",
                    "No numeric columns found in CSV (besides structure_name)."
                )
                return None
            
            if len(target_candidates) == 1:
                target_col = target_candidates[0]
            else:
                # Multiple numeric columns — ask user to select target
                target_col, ok = QInputDialog.getItem(
                    self,
                    "Select Target Column",
                    f"Found {len(target_candidates)} numeric columns.\n"
                    f"Which one is the property to predict?\n\n"
                    f"Other columns will be IGNORED.",
                    target_candidates,
                    0,
                    False
                )
                if not ok:
                    return None
            
            # Return ONLY structure_name + selected target column
            result = df[['structure_name', target_col]].copy()
            self.logger.info(
                f"Target CSV loaded: {len(result)} rows, target='{target_col}'. "
                f"Ignored columns: {[c for c in df.columns if c not in result.columns]}"
            )
            return result
            
        except Exception as e:
            QMessageBox.warning(self, "Warning", f"Failed to load target CSV:\n{str(e)}")
            return None
    
    @staticmethod
    def _merge_targets(features_df: pd.DataFrame, target_df: pd.DataFrame) -> pd.DataFrame:
        """Merge target values into features DataFrame by structure_name.
        
        Args:
            features_df: DataFrame with structure features and 'structure_name' column
            target_df: DataFrame with 'structure_name' and ONE target column
            
        Returns:
            Merged DataFrame
        """
        # Identify target column (single, all except structure_name)
        target_cols = [c for c in target_df.columns if c != 'structure_name']
        
        if len(target_cols) > 1:
            # This shouldn't happen with the new dialog, but handle defensively
            from ..utils.logger import get_logger
            logger = get_logger(__name__)
            logger.warning(
                f"Multiple target columns in CSV: {target_cols}. "
                f"Using only the first: {target_cols[0]}"
            )
            target_cols = [target_cols[0]]
        
        # Keep only structure_name + target column from target_df
        target_subset = target_df[['structure_name'] + target_cols]
        
        # Merge
        merged = features_df.merge(
            target_subset, on='structure_name', how='left'
        )
        
        return merged
    
    def _run_health_check(self):
        """Run the Data Health Check and display results."""
        if self.data_manager is None or self.data_manager.data is None:
            QMessageBox.warning(self, "No Data", "Please load a dataset first.")
            return
        
        self.status_label.setText("Running Data Health Check...")
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        try:
            checker = DataHealthChecker()
            target = self.data_manager.target_column
            report = checker.check(self.data_manager.data, target)
            
            self.progress_bar.setVisible(False)
            self.status_label.setText(checker.get_quick_summary(report))
            
            # Build HTML report
            severity_colors = {
                "critical": "#FF5252",
                "high": "#FF9800",
                "medium": "#FFC107",
                "low": "#4CAF50",
                "info": "#2196F3",
            }
            
            html = f"""
            <h2>Data Health Report — Score: {report.overall_score:.0f}/100</h2>
            <p><b>Samples:</b> {report.n_samples:,} &nbsp;|&nbsp; 
               <b>Features:</b> {report.n_features} &nbsp;|&nbsp;
               <b>Target:</b> {target or 'Not set'}</p>
            <hr>
            """
            
            if report.has_blocking_issues:
                html += """
                <p style="color:#FF5252; font-weight:bold;">
                ⚠ Issues detected that should be fixed before training.
                </p>
                """
            else:
                html += """
                <p style="color:#4CAF50; font-weight:bold;">
                ✓ No blocking issues. Ready for training!
                </p>
                """
            
            if not report.issues:
                html += "<p>No issues found. Your data looks great!</p>"
            else:
                # Sort by severity
                severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
                sorted_issues = sorted(report.issues, key=lambda i: severity_order.get(i.severity.value, 5))
                
                for issue in sorted_issues:
                    color = severity_colors.get(issue.severity.value, "#888")
                    html += f"""
                    <div style="margin:10px 0; padding:8px; border-left:4px solid {color};
                                background:#2a2a2a; border-radius:4px;">
                        <b style="color:{color};">{issue.title}</b> 
                        <span style="color:#888; font-size:10px;">({issue.severity.value.upper()})</span><br>
                        {issue.description}<br>
                        <span style="color:#81C784;"><b>Fix:</b> {issue.recommendation}</span>
                    </div>
                    """
            
            # Show in a dialog
            dialog = QDialog(self)
            dialog.setWindowTitle("Data Health Check")
            dialog.setMinimumWidth(700)
            dialog.setMinimumHeight(500)
            layout = QVBoxLayout(dialog)
            
            text_edit = QTextEdit()
            text_edit.setReadOnly(True)
            text_edit.setHtml(html)
            layout.addWidget(text_edit)
            
            btn_box = QPushButton("Close")
            btn_box.clicked.connect(dialog.close)
            layout.addWidget(btn_box)
            
            dialog.exec()
            
        except Exception as e:
            self.progress_bar.setVisible(False)
            show_friendly_error(self, e, "running health check", include_traceback=False)
    
    def _export_dataset(self):
        """Export the current featurized dataset to CSV.

        v2.0: Universal export that works regardless of how the dataset was
        created — pure CSV, crystal structures, molecular structures, linked
        features, or any combination. Uses DatasetExporter for testable logic.
        """
        if self.data_manager is None or self.data_manager.data is None:
            QMessageBox.warning(self, "No Data", "No dataset to export. Please load data first.")
            return

        df = self.data_manager.data
        default_name = DatasetExporter.suggest_filename(
            getattr(self.data_manager, '_file_path', None)
        )

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Featurized Dataset",
            default_name,
            "CSV Files (*.csv);;All Files (*.*)"
        )

        if not file_path:
            return

        try:
            n_rows, n_cols = DatasetExporter.export_to_csv(df, file_path)
            summary = DatasetExporter.get_export_summary(df, file_path)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Dataset exported successfully!\n\n"
                f"File: {summary['file']}\n"
                f"Rows: {summary['rows']}\n"
                f"Columns: {summary['columns']}\n\n"
                f"Feature breakdown:\n{summary['breakdown']}\n\n"
                f"You can now use this CSV for external analysis, "
                f"sharing, or as input for other tools."
            )
            self.status_label.setText(
                f"Dataset exported: {summary['rows']} rows x {summary['columns']} "
                f"columns → {Path(file_path).name}"
            )
            self.logger.info(
                f"Dataset exported: {file_path} ({summary['rows']} rows x {summary['columns']} columns)"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export dataset:\n{str(e)}")
            self.logger.error(f"Dataset export failed: {e}")

    def get_data_manager(self) -> DataManager:
        """Get the data manager."""
        return self.data_manager
