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
    QDialog, QDialogButtonBox, QComboBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QColor



from ..core.inference_engine import InferenceEngine, LegacyInferenceEngine
from ..core.dataset_exporter import DatasetExporter
from ..core.crystal_structure import (
    CrystalStructureLoader, CrystalStructureFeaturizer,
    CrystalStructureDatasetBuilder
)
from ..utils.logger import LoggerMixin


class PredictionThread(QThread):
    """Thread for running predictions with optional confidence intervals."""
    progress = pyqtSignal(int)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(self, engine: InferenceEngine, df: pd.DataFrame, include_intervals: bool = False):
        super().__init__()
        self.engine = engine
        self.df = df
        self.include_intervals = include_intervals
    
    def run(self):
        try:
            self.progress.emit(10)

            # v2.0: Always use predict_with_ood() so the GUI receives the OOD
            # columns (ood_distance, ood_flag, ood_warning) alongside predictions.
            # If the bundle is legacy and has no OOD detector, the columns are
            # filled with neutral values (NaN, False, "") — no branching needed.
            if self.include_intervals:
                # TODO: predict_with_intervals could also return OOD columns;
                # for now we keep intervals and OOD as separate calls.
                result = self.engine.predict_with_intervals(self.df)
            else:
                result = self.engine.predict_with_ood(self.df)

            self.progress.emit(100)
            self.finished.emit(result)

        except Exception as e:
            self.error.emit(str(e))


class LinkedPredictionThread(QThread):
    """Thread for running predictions with linked structures.
    
    Handles all 4 cases:
    1. Crystal structures only (.cif linking)
    2. Molecule structures only (.xyz linking)
    3. Crystal + composition featurization
    4. Molecule + composition featurization
    """
    progress = pyqtSignal(str)
    finished = pyqtSignal(pd.DataFrame)
    error = pyqtSignal(str)
    
    def __init__(
        self,
        engine: InferenceEngine,
        df: pd.DataFrame,
        id_column: str,
        structure_folder: str,
        structure_type: str,
        include_intervals: bool = False,
    ):
        super().__init__()
        self.engine = engine
        self.df = df
        self.id_column = id_column
        self.structure_folder = structure_folder
        self.structure_type = structure_type
        self.include_intervals = include_intervals
    
    def run(self):
        try:
            self.progress.emit("Step 1/3: Linking structures...")
            
            # Step 1: Link structures
            if self.structure_type == "crystal":
                try:
                    from ..core.crystal_feature_extractor import CrystalFeatureExtractor
                except ImportError as e:
                    raise RuntimeError(
                        "Crystal structure support not available. "
                        "Install pymatgen: pip install pymatgen-core"
                    ) from e
                linker = CrystalFeatureExtractor()
                linked_df, report = linker.link_structures(
                    self.df, self.id_column, self.structure_folder
                )
                
                self.progress.emit(
                    f"Step 1/3: Linked {report.n_matched}/{report.n_rows_total} "
                    f"crystal structures ({report.match_rate*100:.0f}%)"
                )
                
            elif self.structure_type == "molecule":
                from ..core.molecule_feature_linker import MoleculeFeatureLinker
                linker = MoleculeFeatureLinker()
                linked_df, report = linker.link_structures(
                    self.df, self.id_column, self.structure_folder
                )
                
                self.progress.emit(
                    f"Step 1/3: Linked {report.n_matched}/{report.n_rows_total} "
                    f"molecular structures ({report.match_rate*100:.0f}%)"
                )
            else:
                raise ValueError(f"Unknown structure type: {self.structure_type}")
            
            # Step 2: Predict
            # v2.0: Composition is auto-derived from structure files during linking.
            # No separate Magpie step — both linkers include periodic properties.
            self.progress.emit("Step 2/2: Running prediction...")
            if self.include_intervals:
                result = self.engine.predict_with_intervals(linked_df)
            else:
                result = self.engine.predict(linked_df)
            
            self.progress.emit("Complete!")
            self.finished.emit(result)
            
        except Exception as e:
            self.error.emit(str(e))


class InferenceTab(QWidget, LoggerMixin):
    """Tab for standalone model inference/prediction."""
    
    def __init__(self, parent=None):
        """Initialize inference tab."""
        super().__init__(parent)
        
        # CRITICAL: Do NOT use 'self.parent = parent' — it overwrites
        # the PyQt parent() method and breaks the widget hierarchy.
        self._main_window = parent
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
        model_group.setMinimumHeight(120)
        model_layout = QVBoxLayout(model_group)
        
        # Model path selection
        path_layout = QHBoxLayout()
        path_layout.addWidget(QLabel("Model Folder:"))
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setPlaceholderText("Select folder containing manifest.json and model files...")
        self.model_path_edit.setReadOnly(True)
        path_layout.addWidget(self.model_path_edit)
        
        self.browse_btn = QPushButton("Browse...")
        self.browse_btn.setToolTip("Select a model bundle (.joblib or folder) exported from the Results tab")
        self.browse_btn.clicked.connect(self._browse_model)
        path_layout.addWidget(self.browse_btn)
        
        model_layout.addLayout(path_layout)
        
        # Model info display
        self.model_info_text = QTextEdit()
        self.model_info_text.setReadOnly(True)
        self.model_info_text.setMaximumHeight(100)
        self.model_info_text.setPlaceholderText("Model information will appear here after loading...")
        model_layout.addWidget(self.model_info_text)
        
        # Version warning label
        self.version_warning_label = QLabel("")
        self.version_warning_label.setStyleSheet("color: orange;")
        model_layout.addWidget(self.version_warning_label)
        
        splitter.addWidget(model_group)
        
        # === MIDDLE SECTION: Data Loading ===
        data_group = QGroupBox("Data for Prediction")
        data_group.setMinimumHeight(180)
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
        
        # === CRYSTAL STRUCTURE SECTION (v1.2.0) ===
        # Compact single-row layout with buttons + status
        xtal_group = QGroupBox("Crystal Structures for Prediction (v1.2)")
        xtal_layout = QVBoxLayout(xtal_group)
        xtal_layout.setSpacing(5)
        xtal_layout.setContentsMargins(8, 8, 8, 8)
        
        xtal_btn_layout = QHBoxLayout()
        
        self.load_xtal_files_btn = QPushButton("Load Structure Files...")
        self.load_xtal_files_btn.setToolTip("Select CIF, POSCAR, or XYZ files")
        self.load_xtal_files_btn.setStyleSheet("background-color: #6A1B9A; color: white;")
        self.load_xtal_files_btn.clicked.connect(self._load_xtal_for_prediction)
        xtal_btn_layout.addWidget(self.load_xtal_files_btn)
        
        self.load_xtal_dir_btn = QPushButton("Load Structure Directory...")
        self.load_xtal_dir_btn.setToolTip("Select directory with structure files")
        self.load_xtal_dir_btn.setStyleSheet("background-color: #6A1B9A; color: white;")
        self.load_xtal_dir_btn.clicked.connect(self._load_xtal_dir_for_prediction)
        xtal_btn_layout.addWidget(self.load_xtal_dir_btn)
        
        xtal_btn_layout.addStretch()
        
        self.xtal_status_label = QLabel("No crystal structures loaded")
        self.xtal_status_label.setStyleSheet("padding: 5px; border-radius: 3px; font-size: 9pt;")
        xtal_btn_layout.addWidget(self.xtal_status_label)
        
        xtal_layout.addLayout(xtal_btn_layout)
        
        splitter.addWidget(xtal_group)
        
        # === v2.0: LINKED STRUCTURE SECTION (crystal/molecule from tabular data) ===
        self.link_group = QGroupBox("Link Structures for Prediction (v2.0)")
        self.link_group.setVisible(False)  # Shown only when linked model is loaded
        link_layout = QVBoxLayout(self.link_group)
        link_layout.setSpacing(6)
        link_layout.setContentsMargins(8, 8, 8, 8)
        
        # Row 1: Structure folder + ID column + Type
        row1 = QHBoxLayout()
        
        row1.addWidget(QLabel("Folder:"))
        self.link_folder_edit = QLineEdit()
        self.link_folder_edit.setPlaceholderText("Select folder with .cif or .xyz files...")
        self.link_folder_edit.setReadOnly(True)
        row1.addWidget(self.link_folder_edit, stretch=2)
        
        self.link_browse_btn = QPushButton("Browse...")
        self.link_browse_btn.clicked.connect(self._browse_link_folder)
        row1.addWidget(self.link_browse_btn)
        
        row1.addSpacing(15)
        row1.addWidget(QLabel("ID Column:"))
        self.link_id_combo = QComboBox()
        self.link_id_combo.setMinimumWidth(120)
        self.link_id_combo.setToolTip("Column in CSV that matches structure filenames")
        row1.addWidget(self.link_id_combo, stretch=1)
        
        row1.addSpacing(15)
        row1.addWidget(QLabel("Type:"))
        self.link_type_label = QLabel("Auto-detect")
        self.link_type_label.setStyleSheet("font-weight: bold; color: #4fc3f7;")
        row1.addWidget(self.link_type_label)
        
        link_layout.addLayout(row1)
        
        # Row 2: Status + Predict with Linked Structures button
        row2 = QHBoxLayout()
        
        self.link_status_label = QLabel("Load a linked model to enable")
        self.link_status_label.setStyleSheet("padding: 5px; border-radius: 3px; font-size: 9pt;")
        row2.addWidget(self.link_status_label, stretch=1)
        
        self.link_predict_btn = QPushButton("Predict with Linked Structures")
        self.link_predict_btn.setToolTip(
            "Automatically link structure files to the CSV data, then run prediction"
        )
        self.link_predict_btn.setEnabled(False)
        self.link_predict_btn.setStyleSheet("""
            QPushButton {
                background-color: #1565C0; color: white; font-weight: bold;
                padding: 8px 16px;
            }
            QPushButton:disabled { background-color: #555555; }
            QPushButton:hover:enabled { background-color: #1976D2; }
        """)
        self.link_predict_btn.clicked.connect(self._run_linked_prediction)
        row2.addWidget(self.link_predict_btn)
        
        link_layout.addLayout(row2)
        
        splitter.addWidget(self.link_group)
        
        # === BOTTOM SECTION: Prediction & Results ===
        # This section is OUTSIDE the splitter to always remain visible
        bottom_widget = QWidget()
        bottom_widget.setMinimumHeight(120)
        bottom_layout = QVBoxLayout(bottom_widget)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        bottom_layout.setSpacing(6)
        
        # Prediction button + confidence interval checkbox
        predict_row = QHBoxLayout()
        
        self.predict_btn = QPushButton("Run Prediction")
        self.predict_btn.setToolTip("Generate predictions using the loaded model and data")
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
        predict_row.addWidget(self.predict_btn, stretch=1)
        
        # v1.5.0: Confidence interval checkbox
        self.ci_check = QCheckBox("Include Confidence Intervals")
        self.ci_check.setToolTip(
            "When checked, predictions include confidence intervals based on "
            "CV residual analysis (regression) or class probabilities (classification). "
            "Intervals are saved as additional columns in the CSV export."
        )
        self.ci_check.setChecked(True)
        predict_row.addWidget(self.ci_check)
        
        bottom_layout.addLayout(predict_row)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        bottom_layout.addWidget(self.progress_bar)
        
        # v2.0: Peer-review compliant disclaimer banner — visible BEFORE results
        self.disclaimer_banner = QLabel(
            "<b>These predictions are model estimates for screening and prioritisation, "
            "NOT experimentally validated properties.</b><br>"
            "Use them to rank and filter candidate materials, then verify independently "
            "before making design decisions."
        )
        self.disclaimer_banner.setWordWrap(True)
        self.disclaimer_banner.setStyleSheet(
            "QLabel {"
            "  background-color: #FFF3CD;"
            "  color: #856404;"
            "  border: 1px solid #F0AD4E;"
            "  border-radius: 4px;"
            "  padding: 8px;"
            "  font-size: 12px;"
            "}"
        )
        self.disclaimer_banner.setVisible(False)  # shown after model is loaded
        bottom_layout.addWidget(self.disclaimer_banner)

        # Results section
        results_group = QGroupBox("Prediction Results")
        results_layout = QVBoxLayout(results_group)
        
        # Statistics
        stats_layout = QHBoxLayout()
        self.stats_label = QLabel("No predictions yet")
        stats_layout.addWidget(self.stats_label)
        stats_layout.addStretch()
        
        self.export_featurized_btn = QPushButton("Export Featurized Dataset")
        self.export_featurized_btn.setToolTip(
            "Export the input dataset with all generated features to CSV. "
            "This is the featurized data used for prediction, before predictions are added."
        )
        self.export_featurized_btn.setEnabled(False)
        self.export_featurized_btn.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
                padding: 4px 12px;
            }
            QPushButton:hover { background-color: #388E3C; }
            QPushButton:disabled { background-color: #555; color: #888; }
        """)
        self.export_featurized_btn.clicked.connect(self._export_featurized_dataset)
        stats_layout.addWidget(self.export_featurized_btn)

        self.export_btn = QPushButton("Export Predictions")
        self.export_btn.setEnabled(False)
        self.export_btn.clicked.connect(self._export_results)
        stats_layout.addWidget(self.export_btn)
        
        results_layout.addLayout(stats_layout)
        
        # Results preview table
        self.results_table = QTableWidget()
        self.results_table.setMaximumHeight(200)
        results_layout.addWidget(self.results_table)
        

        
        bottom_layout.addWidget(results_group)
        
        splitter.addWidget(bottom_widget)
        
        # Set splitter proportions: model(150) + data(250) + xtal(80) + bottom(320)
        # Total = 800. The bottom section (Run Prediction + Results) must always be visible.
        splitter.setSizes([150, 250, 80, 320])
        splitter.setStretchFactor(0, 0)  # Model: fixed minimum
        splitter.setStretchFactor(1, 1)  # Data: can grow
        splitter.setStretchFactor(2, 0)  # Crystal: fixed minimum (compact)
        splitter.setStretchFactor(3, 0)  # Bottom: fixed (always visible)
    
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
            
            # v1.2.0: Detect models trained with the structure_name one-hot bug
            # (buggy models have 400+ features due to one-hot encoded IDs)
            feature_names = self.inference_engine.manifest.get('feature_names', [])
            n_features = len(feature_names)
            id_like_features = [f for f in feature_names 
                               if f.startswith('structure_name_') 
                               or f.startswith('formula_')
                               or f.startswith('cif_file_')
                               or f.startswith('xyz_file_')]
            has_id_bug = len(id_like_features) > 0 or n_features > 400
            
            # Display info
            info_text = f"""
<b>Model:</b> {info.get('model_name', 'Unknown')}<br>
<b>Algorithm:</b> {info.get('algorithm', 'Unknown')}<br>
<b>Type:</b> {info.get('model_type', 'Unknown')}<br>
<b>Created:</b> {info.get('creation_date', 'Unknown')[:10]}<br>
<b>Features Required:</b> {n_features}<br>
<b>Target Column:</b> {info.get('target_column', 'Unknown')}<br>
<b>Metrics:</b> {json.dumps(info.get('metrics', {}), indent=2)}
            """
            # v2.0: CV provenance — show how the model was validated during
            # training (e.g. "Group K-Fold (by chemical system)"). This is a
            # read-only traceability line; prediction itself never runs CV.
            cv_val = info.get('cv_validation')
            if cv_val:
                info_text += f"<br><b>Validated with:</b> {cv_val}"
            # v2.0: Gaussian Process native uncertainty info
            uncertainty = info.get('uncertainty')
            if uncertainty:
                info_text += f"""
<br><br><span style='color: #FFB74D;'><b>🔮 Native Uncertainty Quantification</b></span><br>
Kernel: {uncertainty.get('kernel', 'N/A')} | Alpha: {uncertainty.get('alpha', 'N/A')}<br>
{uncertainty.get('note', '')}
                """
            if has_id_bug:
                info_text += f"""
<br><br><span style='color: #ff6b6b;'><b>⚠️ This model appears to have been trained with a bug</b></span><br>
It has {n_features} features including {len(id_like_features)} one-hot encoded ID columns
(structure_name, formula, file paths). These should NOT be features.<br>
<b>Crystal structure prediction will NOT work.</b> Please retrain the model
with AutoAIM Studio v1.3.0 or later.
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
                if has_id_bug:
                    self.version_warning_label.setText(
                        f"⚠️ Model has {n_features} features (ID bug detected). "
                        f"Retrain with v1.3.0 for crystal structure prediction."
                    )
                else:
                    self.version_warning_label.setText("")
            
            # v2.0: Check if model requires linked structures
            self._check_linked_model()

            # v2.0: Show disclaimer banner once a model is loaded
            self.disclaimer_banner.setVisible(True)

            self.logger.info(f"Model loaded: {info.get('model_name')} ({n_features} features)")
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
                
                # v2.0: Update ID column combo if linked model is loaded
                if self.link_group.isVisible():
                    self._update_link_id_combo()
                    self._update_link_predict_button()
                
                self.logger.info(f"Data loaded: {len(self.input_data)} rows, {len(self.input_data.columns)} columns")
                self.export_featurized_btn.setEnabled(True)
                self._update_predict_button()
                
            except Exception as e:
                QMessageBox.critical(self, "Error Loading Data", str(e))
                self.input_data = None
                self.export_featurized_btn.setEnabled(False)
    
    def _update_data_preview(self):
        """Update data preview table."""
        if self.input_data is None:
            return
        
        # Show first 5 rows
        preview = self.input_data.head()
        
        self.data_preview_table.setColumnCount(len(preview.columns))
        self.data_preview_table.setRowCount(len(preview))
        
        self.data_preview_table.setHorizontalHeaderLabels(preview.columns)
        
        for row_idx, (_, row) in enumerate(preview.iterrows()):
            for col_idx, col in enumerate(preview.columns):
                item = QTableWidgetItem(str(row[col]))
                self.data_preview_table.setItem(row_idx, col_idx, item)
        
        self.data_preview_table.resizeColumnsToContents()
    
    def _validate_data(self):
        """Validate input data against model requirements."""
        if self.inference_engine is None or self.input_data is None:
            return
        
        validation = self.inference_engine.validate_input(self.input_data)
        
        # v2.0: If model requires linked structures and ONLY linked features are missing,
        # show a special message directing the user to use linked prediction
        if validation.get("needs_linking_action"):
            missing_linked = validation.get("missing_linked_features", [])
            structure_type = validation.get("requires_structure_linking")
            self.validation_label.setText(
                f"🔗 Linked features required: {len(missing_linked)} features missing. "
                f"Use 'Predict with Linked Structures' section below."
            )
            self.validation_label.setStyleSheet(
                "background-color: #1a3a4a; color: white; padding: 5px; border-radius: 3px;"
            )
            self._update_predict_button()
            return
        
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
        
        # v2.0: If linked model and only linked features are missing, redirect user
        if validation.get("needs_linking_action"):
            missing_linked = validation.get("missing_linked_features", [])
            reqs = self.inference_engine.get_linked_structure_requirements()
            structure_type = reqs.get("structure_type", "structure")
            
            QMessageBox.information(
                self,
                "Linked Features Required",
                f"This model was trained with linked {structure_type} features that are "
                f"missing from the input data.\n\n"
                f"Missing: {len(missing_linked)} features\n\n"
                f"Please use the 'Predict with Linked Structures' section below:\n"
                f"1. Select the folder containing your {structure_type} files\n"
                f"2. Select the ID column that matches filenames\n"
                f"3. Click 'Predict with Linked Structures'"
            )
            return
        
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
        
        # v1.3.0: Validate feature count before predicting
        # Compare only columns that exist in the manifest (ignore target/extra columns)
        manifest_features = self.inference_engine.manifest.get('feature_names', [])
        input_feature_cols = [c for c in self.input_data.columns if c in manifest_features]
        input_features = len(input_feature_cols)
        expected_features = len(manifest_features)
        if input_features != expected_features:
            missing = [c for c in manifest_features if c not in self.input_data.columns]
            reply = QMessageBox.warning(
                self, "Feature Count Mismatch",
                f"Input data has {input_features} matching features but model "
                f"'{self.inference_engine.manifest.get('model_name', 'unknown')}' "
                f"expects {expected_features} features.\n\n"
                f"Missing: {missing}\n\n"
                f"Do you want to continue anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.No:
                self.predict_btn.setEnabled(True)
                return
        
        # Run prediction in thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.predict_btn.setEnabled(False)
        
        # v1.5.0: Check if confidence intervals are requested
        include_ci = self.ci_check.isChecked()
        self.prediction_thread = PredictionThread(
            self.inference_engine, self.input_data, include_intervals=include_ci
        )
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
        
        # v2.0: Gaussian Process — mention native uncertainty in completion message
        has_uncertainty = 'prediction_std' in result.columns
        uncertainty_msg = (
            "\n\nThe model also provides a 'prediction_std' column with the "
            "native posterior standard deviation (±1σ). Lower values = higher confidence."
            if has_uncertainty else ""
        )
        
        QMessageBox.information(
            self,
            "Prediction Complete",
            f"Successfully predicted {len(result)} rows.\n"
            f"Predictions saved in 'prediction' column."
            f"{uncertainty_msg}"
        )
    
    def _prediction_error(self, error_msg: str):
        """Handle prediction error."""
        self.progress_bar.setVisible(False)
        self.predict_btn.setEnabled(True)
        
        QMessageBox.critical(self, "Prediction Error", error_msg)
    
    def _update_results_display(self):
        """Update results display.

        v2.0: Adds OOD highlighting — rows where ood_flag is True are tinted
        red, and a count of extrapolated samples is shown in the statistics.
        """
        if self.prediction_result is None:
            return

        predictions = self.prediction_result['prediction']

        # v2.0: Count OOD samples if present
        ood_flags = self.prediction_result.get('ood_flag')
        n_ood = int(ood_flags.sum()) if ood_flags is not None else 0

        # Update statistics
        stats_text = f"""
<b>Prediction Statistics:</b><br>
Count: {len(predictions)} |
Min: {predictions.min():.4f} |
Max: {predictions.max():.4f} |
Mean: {predictions.mean():.4f} |
Std: {predictions.std():.4f}
        """
        if n_ood > 0:
            stats_text += (
                f"<br><span style='color: #C62828;'>"
                f"⚠️ {n_ood} sample(s) flagged as out-of-distribution (extrapolation)."
                f"</span>"
            )
        self.stats_label.setText(stats_text)

        # Update results table (show first 10 rows with prediction)
        display_df = self.prediction_result.head(10)

        self.results_table.setColumnCount(len(display_df.columns))
        self.results_table.setRowCount(len(display_df))

        self.results_table.setHorizontalHeaderLabels(display_df.columns)

        for i, (idx, row) in enumerate(display_df.iterrows()):
            # v2.0: Determine if this row is OOD
            is_ood = bool(row.get('ood_flag', False))
            for j, col in enumerate(display_df.columns):
                val = row[col]
                if isinstance(val, float):
                    item = QTableWidgetItem(f"{val:.4f}")
                else:
                    item = QTableWidgetItem(str(val))

                # Highlight prediction column
                if col == 'prediction':
                    item.setBackground(QColor(13, 115, 119, 100))
                # v2.0: Highlight uncertainty column for Gaussian Process
                if col == 'prediction_std':
                    item.setBackground(QColor(255, 183, 77, 80))
                    item.setToolTip(
                        "Posterior standard deviation (±1σ) from the Gaussian Process. "
                        "Lower values indicate higher model confidence."
                    )
                # v2.0: Tint entire OOD row red
                if is_ood:
                    item.setBackground(QColor(255, 205, 210, 180))
                    if col == 'ood_flag':
                        item.setToolTip(
                            "This composition lies outside the model's applicability domain. "
                            "The prediction is an extrapolation and should be interpreted "
                            "with extreme caution."
                        )
                    if col == 'ood_distance':
                        item.setToolTip(
                            "Distance to the nearest training neighbour in standardised "
                            "feature space. Higher values = further from known data."
                        )

                self.results_table.setItem(i, j, item)

        self.results_table.resizeColumnsToContents()
    
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
                # v1.2.0: If index contains structure names (crystal structure
                # prediction), include it in export with proper column name.
                # Otherwise use index=False for clean tabular data.
                has_named_index = (
                    self.prediction_result.index.name == 'structure_name'
                    or self.prediction_result.index.name is None
                    and len(self.prediction_result) > 0
                    and isinstance(self.prediction_result.index[0], str)
                )
                if has_named_index:
                    self.prediction_result.to_csv(
                        file_path,
                        index=True,
                        index_label='structure_name'
                    )
                else:
                    self.prediction_result.to_csv(file_path, index=False)
                QMessageBox.information(
                    self,
                    "Export Complete",
                    f"Predictions saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _export_featurized_dataset(self):
        """Export the featurized input dataset to CSV (before prediction).

        v2.0: This exports self.input_data — the dataset with all generated
        features (auto-generated composition features, linked structure
        features, etc.) that was used as input for prediction.
        Uses DatasetExporter for testable core logic.
        """
        if self.input_data is None:
            QMessageBox.warning(self, "No Data", "No featurized dataset to export. Please load data first.")
            return

        df = self.input_data
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Featurized Dataset",
            "autoaim_featurized_dataset.csv",
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
                f"Featurized dataset exported successfully!\n\n"
                f"File: {summary['file']}\n"
                f"Rows: {summary['rows']}\n"
                f"Columns: {summary['columns']}\n\n"
                f"Feature breakdown:\n{summary['breakdown']}\n\n"
                f"This dataset contains all generated features used for prediction. "
                f"Use it for external analysis or as input for other tools."
            )
            self.logger.info(
                f"Featurized dataset exported: {file_path} "
                f"({summary['rows']} rows x {summary['columns']} columns)"
            )

        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export featurized dataset:\n{str(e)}")

    # ------------------------------------------------------------------
    # v2.0: Linked Structure Support (crystal/molecule from tabular data)
    # ------------------------------------------------------------------
    
    def _check_linked_model(self):
        """Check if loaded model requires linked structures and update UI."""
        if self.inference_engine is None:
            self.link_group.setVisible(False)
            return
        
        is_linked = self.inference_engine.is_structure_linked_model()
        self.link_group.setVisible(is_linked)
        
        if is_linked:
            reqs = self.inference_engine.get_linked_structure_requirements()
            structure_type = reqs.get("structure_type", "unknown")
            missing_count = reqs.get("missing_feature_count", 0)
            prefixes = reqs.get("linked_prefixes", [])
            
            # Update type label
            type_display = {
                "crystal": "Crystal (CIF)",
                "molecule": "Molecule (.xyz)",
            }.get(structure_type, structure_type)
            self.link_type_label.setText(type_display)
            
            # Style based on type
            if structure_type == "crystal":
                self.link_type_label.setStyleSheet("font-weight: bold; color: #CE93D8;")
            elif structure_type == "molecule":
                self.link_type_label.setStyleSheet("font-weight: bold; color: #80CBC4;")
            
            # Update status
            prefix_desc = ", ".join(p.rstrip("_") for p in prefixes if p not in ("composition",))
            self.link_status_label.setText(
                f"Model requires {missing_count} linked features ({prefix_desc}). "
                f"Select structure folder and ID column, then click 'Predict with Linked Structures'."
            )
            self.link_status_label.setStyleSheet(
                "padding: 5px; border-radius: 3px; font-size: 9pt; background-color: #1a3a4a;"
            )
            
            # Update ID column combo if data is loaded
            self._update_link_id_combo()
            self._update_link_predict_button()
            
            self.logger.info(
                f"Linked model detected: {structure_type}, "
                f"{missing_count} linked features required"
            )
    
    def _browse_link_folder(self):
        """Browse for structure folder (CIF or .xyz)."""
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select Structure Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        if folder:
            self.link_folder_edit.setText(folder)
            # Count structure files
            path = Path(folder)
            if self.link_type_label.text().startswith("Crystal"):
                n_files = len(list(path.glob("*.cif")))
                ext = ".cif"
            elif self.link_type_label.text().startswith("Molecule"):
                n_files = len(list(path.glob("*.xyz")))
                ext = ".xyz"
            else:
                n_cif = len(list(path.glob("*.cif")))
                n_xyz = len(list(path.glob("*.xyz")))
                if n_cif > 0 and n_xyz == 0:
                    n_files = n_cif
                    ext = ".cif"
                    self.link_type_label.setText("Crystal (CIF)")
                elif n_xyz > 0 and n_cif == 0:
                    n_files = n_xyz
                    ext = ".xyz"
                    self.link_type_label.setText("Molecule (.xyz)")
                else:
                    n_files = n_cif + n_xyz
                    ext = ".cif/.xyz"
            
            self.link_status_label.setText(
                f"Found {n_files} {ext} files in folder. Ready to link."
            )
            self.link_status_label.setStyleSheet(
                "padding: 5px; border-radius: 3px; font-size: 9pt; background-color: #1a472a;"
            )
            self._update_link_predict_button()
    
    def _update_link_id_combo(self):
        """Update ID column dropdown with columns from loaded data."""
        self.link_id_combo.clear()
        if self.input_data is not None and not self.input_data.empty:
            for col in self.input_data.columns:
                self.link_id_combo.addItem(col)
    
    def _update_link_predict_button(self):
        """Enable/disable linked prediction button."""
        can_link = (
            self.inference_engine is not None
            and self.input_data is not None
            and self.link_folder_edit.text()
            and self.link_id_combo.currentText()
        )
        self.link_predict_btn.setEnabled(can_link)
    
    def _run_linked_prediction(self):
        """Run prediction with linked structures.
        
        This is the main method that handles all 4 cases:
        1. Crystal structures only (.cif linking)
        2. Molecule structures only (.xyz linking)
        3. Crystal + composition featurization
        4. Molecule + composition featurization
        """
        if self.inference_engine is None or self.input_data is None:
            return
        
        structure_folder = self.link_folder_edit.text()
        id_column = self.link_id_combo.currentText()
        
        if not structure_folder or not id_column:
            QMessageBox.warning(
                self, "Missing Information",
                "Please select a structure folder and an ID column."
            )
            return
        
        if id_column not in self.input_data.columns:
            QMessageBox.warning(
                self, "Invalid ID Column",
                f"Column '{id_column}' not found in the data."
            )
            return
        
        # Determine structure type from manifest
        reqs = self.inference_engine.get_linked_structure_requirements()
        structure_type = reqs.get("structure_type")
        
        if structure_type is None:
            # Fallback: try to detect from folder contents
            path = Path(structure_folder)
            n_cif = len(list(path.glob("*.cif")))
            n_xyz = len(list(path.glob("*.xyz")))
            if n_cif > n_xyz:
                structure_type = "crystal"
            elif n_xyz > n_cif:
                structure_type = "molecule"
            else:
                QMessageBox.critical(
                    self, "Cannot Detect Structure Type",
                    "Cannot determine if structures are crystal (CIF) or molecule (.xyz). "
                    "Please ensure the model was saved with linked features."
                )
                return
        
        # v2.0: Composition is auto-derived from structure files — no separate step.
        # Both CrystalFeatureExtractor and MoleculeFeatureLinker include periodic
        # properties (electronegativity, atomic mass) directly from the structure.
        
        # Run linking + prediction in a thread
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate
        self.predict_btn.setEnabled(False)
        self.link_predict_btn.setEnabled(False)
        
        self.linked_pred_thread = LinkedPredictionThread(
            engine=self.inference_engine,
            df=self.input_data.copy(),
            id_column=id_column,
            structure_folder=structure_folder,
            structure_type=structure_type,
            include_intervals=self.ci_check.isChecked(),
        )
        self.linked_pred_thread.progress.connect(self._on_linked_progress)
        self.linked_pred_thread.finished.connect(self._on_linked_finished)
        self.linked_pred_thread.error.connect(self._on_linked_error)
        self.linked_pred_thread.start()
    
    def _on_linked_progress(self, message: str):
        """Update status during linked prediction."""
        self.link_status_label.setText(message)
    
    def _on_linked_finished(self, result: pd.DataFrame):
        """Handle linked prediction completion."""
        self.prediction_result = result
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.predict_btn.setEnabled(True)
        self.link_predict_btn.setEnabled(True)
        
        self._update_results_display()
        self.export_btn.setEnabled(True)
        
        self.link_status_label.setText(
            f"Prediction complete: {len(result)} rows. "
            f"Predictions saved in 'prediction' column."
        )
        self.link_status_label.setStyleSheet(
            "padding: 5px; border-radius: 3px; font-size: 9pt; background-color: #1a472a;"
        )
        
        QMessageBox.information(
            self, "Linked Prediction Complete",
            f"Successfully linked structures and predicted {len(result)} rows.\n"
            f"Predictions saved in 'prediction' column."
        )
    
    def _on_linked_error(self, error_msg: str):
        """Handle linked prediction error."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setRange(0, 100)
        self.predict_btn.setEnabled(True)
        self.link_predict_btn.setEnabled(True)
        
        self.link_status_label.setText(f"Error: {error_msg[:100]}")
        self.link_status_label.setStyleSheet(
            "padding: 5px; border-radius: 3px; font-size: 9pt; background-color: #5a2d2d;"
        )
        
        QMessageBox.critical(self, "Linked Prediction Error", error_msg)
    
    # ------------------------------------------------------------------
    # Crystal structure prediction (v1.2.0)
    # ------------------------------------------------------------------
    
    def _load_xtal_for_prediction(self):
        """Load crystal structure files for prediction."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Load Crystal Structures for Prediction",
            "",
            "CIF Files (*.cif);;POSCAR Files (POSCAR*);;VASP Files (*.poscar);;"
            "XYZ Files (*.xyz);;All Structure Files (*.cif *.poscar *.xyz)"
        )
        if file_paths:
            self._featurize_and_predict_structures(file_paths=file_paths)
    
    def _load_xtal_dir_for_prediction(self):
        """Load directory of crystal structures for prediction."""
        directory = QFileDialog.getExistingDirectory(
            self,
            "Select Directory with Crystal Structures",
            ""
        )
        if directory:
            self._featurize_and_predict_structures(directory=directory)
    
    def _featurize_and_predict_structures(self, file_paths=None, directory=None):
        """Featurize crystal structures and prepare for prediction.
        
        Args:
            file_paths: List of structure file paths
            directory: Directory containing structure files
        """
        # v1.2.0: Check if loaded model has the ID one-hot bug
        if self.inference_engine is not None:
            manifest = self.inference_engine.manifest or {}
            feature_names = manifest.get('feature_names', [])
            id_features = [f for f in feature_names 
                          if f.startswith('structure_name_') 
                          or f.startswith('formula_')
                          or f.startswith('cif_file_')
                          or f.startswith('xyz_file_')]
            if len(id_features) > 0 or len(feature_names) > 400:
                QMessageBox.critical(
                    self,
                    "Incompatible Model — ID Bug Detected",
                    f"This model was trained with {len(feature_names)} features, including "
                    f"{len(id_features)} one-hot encoded ID columns (structure names, formulas, "
                    f"file paths). These are NOT valid features and will not generalize to "
                    f"new structures.\n\n"
                    f"SOLUTION: Retrain the model using AutoAIM Studio v1.3.0 or later.\n"
                    f"The new training pipeline correctly keeps only ~154 numeric "
                    f"crystallographic descriptors (structure_name is kept as index, "
                    f"not as a one-hot encoded feature).\n\n"
                    f"Steps:\n"
                    f"1. Go to the Data tab\n"
                    f"2. Load your crystal structures (CIF/POSCAR/XYZ)\n"
                    f"3. Set target column\n"
                    f"4. Train a new model\n"
                    f"5. Save the new Model Bundle\n"
                    f"6. Use the new bundle for prediction"
                )
                return
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        self.xtal_status_label.setText("Featurizing structures...")
        
        try:
            # Load structures
            loader = CrystalStructureLoader()
            if directory:
                n = loader.load_directory(directory)
            else:
                n = loader.load_files(file_paths)
            
            if n == 0:
                QMessageBox.warning(self, "Warning", "No structures could be loaded.")
                self.progress_bar.setVisible(False)
                return
            
            # Featurize
            featurizer = CrystalStructureFeaturizer(
                use_magpie=True,
                use_advanced_structural=True
            )
            features_df = featurizer.featurize_structures(loader.structures)
            
            if features_df.empty:
                QMessageBox.warning(self, "Warning", "No features extracted.")
                self.progress_bar.setVisible(False)
                return
            
            # Set structure_name as index (NOT a feature column — same as training).
            # This preserves structure identifiers through the prediction pipeline.
            if 'structure_name' in features_df.columns:
                features_df = features_df.set_index('structure_name')
            
            # Keep ONLY numeric columns — drop any string columns (formulas,
            # file paths, etc.) that should NOT be model features.
            numeric_cols = features_df.select_dtypes(include=[np.number]).columns
            features_df = features_df[numeric_cols].astype(float)
            
            # Store as input data for prediction (index = structure_name preserved)
            self.input_data = features_df
            
            # Update UI
            self.xtal_status_label.setText(
                f"{n} structures loaded | {len(features_df.columns)} features"
            )
            self.xtal_status_label.setStyleSheet(
                "padding: 5px; border-radius: 3px; background-color: #1a472a;"
            )
            
            # Update data preview
            self._update_data_preview()
            
            # Enable prediction if model is loaded
            if self.inference_engine is not None:
                self.predict_btn.setEnabled(True)
                self.validation_label.setText(
                    f"Crystal structures ready: {len(features_df)} samples, "
                    f"{len(features_df.columns)} features"
                )
                self.validation_label.setStyleSheet(
                    "padding: 5px; border-radius: 3px; background-color: #1a472a;"
                )
            
            self.progress_bar.setVisible(False)
            
            # Report
            if loader.failed_files:
                failed = [Path(fp).name for fp, _ in loader.failed_files]
                QMessageBox.information(
                    self, "Partial Load",
                    f"Loaded {n} structures with {len(features_df.columns)} features.\n"
                    f"Failed: {', '.join(failed[:5])}"
                    + ("..." if len(failed) > 5 else "")
                )
            else:
                QMessageBox.information(
                    self, "Success",
                    f"Loaded and featurized {n} structures.\n"
                    f"Features: {len(features_df.columns)} (25 crystal + 106 Magpie + 15 advanced)\n\n"
                    f"Click 'Run Prediction' when model is loaded."
                )
        
        except Exception as e:
            self.progress_bar.setVisible(False)
            self.xtal_status_label.setText(f"Error: {str(e)[:80]}")
            QMessageBox.critical(
                self, "Error",
                f"Failed to featurize structures:\n{str(e)}"
            )
