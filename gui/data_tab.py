"""Data loading and preprocessing tab."""

from typing import Optional
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
from ..core.crystal_structure import (
    CrystalStructureLoader, CrystalStructureFeaturizer,
    CrystalStructureDatasetBuilder
)


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
        self.load_btn.setToolTip("Load CSV or Excel file")
        self.load_btn.clicked.connect(self.load_data_dialog)
        controls_layout.addWidget(self.load_btn)
        
        self.load_val_btn = QPushButton("Load Validation Data")
        self.load_val_btn.setToolTip("Load external validation dataset")
        self.load_val_btn.clicked.connect(self.load_validation_dialog)
        self.load_val_btn.setEnabled(False)
        controls_layout.addWidget(self.load_val_btn)
        
        controls_layout.addSpacing(20)
        
        # Crystal structure loading (v1.2.0)
        self.load_xtal_btn = QPushButton("Load Crystal Structures")
        self.load_xtal_btn.setToolTip("Load CIF, POSCAR, or XYZ files")
        self.load_xtal_btn.setStyleSheet("background-color: #6A1B9A; color: white;")
        self.load_xtal_btn.clicked.connect(self.load_crystal_dialog)
        controls_layout.addWidget(self.load_xtal_btn)
        
        self.load_xtal_dir_btn = QPushButton("Load Crystal Directory")
        self.load_xtal_dir_btn.setToolTip("Load all structures from a directory")
        self.load_xtal_dir_btn.setStyleSheet("background-color: #6A1B9A; color: white;")
        self.load_xtal_dir_btn.clicked.connect(self.load_crystal_directory_dialog)
        controls_layout.addWidget(self.load_xtal_dir_btn)
        
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
                
                # Auto-featurize validation data if training data has comp_* features
                self._sync_validation_featurization()
                
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
            info = self.data_manager._analyze_dataset()
            
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
            
            if self.parent:
                self.parent.data_tab = self
            
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
                from PyQt6.QtWidgets import QInputDialog
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
    
    def get_data_manager(self) -> DataManager:
        """Get the data manager."""
        return self.data_manager
