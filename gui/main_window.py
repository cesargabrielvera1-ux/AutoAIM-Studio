"""Main window for AutoAIM Studio - Auto Artificial Intelligence for Materials."""

import sys
from pathlib import Path
from typing import Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QMenuBar, QToolBar, QStatusBar,
    QFileDialog, QMessageBox, QProgressDialog, QLabel,
    QPushButton, QComboBox, QLineEdit, QTextEdit,
    QSplitter, QFrame, QApplication
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt6.QtGui import QAction, QIcon, QFont, QPalette, QColor, QPixmap

from ..utils.config import get_config
from ..utils.logger import setup_logger
from ..utils.hardware_detector import get_hardware_detector

from .data_tab import DataTab
from .training_tab import TrainingTab
from .nn_builder_tab import NNBuilderTab
from .optimization_tab import OptimizationTab
from .results_tab import ResultsTab
from .explainability_tab import ExplainabilityTab
from .inference_tab import InferenceTab
from .ensemble_tab import EnsembleTab


class MainWindow(QMainWindow):
    """Main application window."""
    
    def __init__(self):
        """Initialize main window."""
        super().__init__()
        
        # Setup logger
        self.logger = setup_logger("gui")
        
        # Get configuration
        self.config = get_config()
        
        # Get hardware info
        self.hardware = get_hardware_detector()
        self.logger.info(self.hardware.format_info())
        
        # Initialize UI
        version = "1.3.0"
        window_title = f"AutoAIM Studio: Auto Artificial Intelligence for Materials Studio v{version}"
        self.setWindowTitle(window_title)
        self.setMinimumSize(1400, 900)
        
        # Set application icon (logo with fallback)
        self._set_application_icon()
        
        # Apply theme
        self._apply_theme()
        
        # Create central widget and layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create menu bar
        self._create_menu_bar()
        
        # Toolbar removed in v1.1.0 - functionality available in tabs
        # self._create_toolbar()
        
        # Create tab widget
        self._create_tabs()
        
        # Create status bar
        self._create_status_bar()
        
        # Show welcome message
        self._show_welcome()
        
        self.logger.info("Main window initialized")
    
    def _apply_theme(self):
        """Apply application theme."""
        theme = self.config.current.theme
        
        if theme == 'dark':
            self._apply_dark_theme()
        else:
            self._apply_light_theme()
    
    def _apply_dark_theme(self):
        """Apply dark theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #3d3d3d;
                background-color: #2b2b2b;
            }
            QTabBar::tab {
                background-color: #3d3d3d;
                color: #ffffff;
                padding: 10px 20px;
                border: 1px solid #555555;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                border-bottom: 2px solid #14a085;
            }
            QTabBar::tab:hover {
                background-color: #4d4d4d;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5c5f;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #3d3d3d;
                border: 1px solid #555555;
                padding: 5px;
                border-radius: 4px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 1px solid #0d7377;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #3d3d3d;
                color: white;
                selection-background-color: #0d7377;
            }
            QGroupBox {
                border: 1px solid #555555;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTableWidget {
                background-color: #3d3d3d;
                border: 1px solid #555555;
                gridline-color: #555555;
            }
            QTableWidget::item:selected {
                background-color: #0d7377;
            }
            QHeaderView::section {
                background-color: #4d4d4d;
                color: white;
                padding: 5px;
                border: 1px solid #555555;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
            }
            QMenuBar {
                background-color: #3d3d3d;
                color: white;
            }
            QMenuBar::item:selected {
                background-color: #0d7377;
            }
            QMenu {
                background-color: #3d3d3d;
                color: white;
                border: 1px solid #555555;
            }
            QMenu::item:selected {
                background-color: #0d7377;
            }
            QToolBar {
                background-color: #3d3d3d;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #3d3d3d;
                color: white;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #555555;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #777777;
            }
        """)
    
    def _apply_light_theme(self):
        """Apply light theme."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
                color: #333333;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #cccccc;
                background-color: #ffffff;
            }
            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                padding: 10px 20px;
                border: 1px solid #cccccc;
                border-bottom: none;
            }
            QTabBar::tab:selected {
                background-color: #0d7377;
                color: white;
                border-bottom: 2px solid #14a085;
            }
            QTabBar::tab:hover {
                background-color: #d0d0d0;
            }
            QPushButton {
                background-color: #0d7377;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #14a085;
            }
            QPushButton:pressed {
                background-color: #0a5c5f;
            }
            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }
            QLineEdit, QTextEdit, QComboBox {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                padding: 5px;
                border-radius: 4px;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border: 1px solid #0d7377;
            }
            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QTableWidget {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                gridline-color: #cccccc;
            }
            QTableWidget::item:selected {
                background-color: #0d7377;
                color: white;
            }
            QHeaderView::section {
                background-color: #e0e0e0;
                color: #333333;
                padding: 5px;
                border: 1px solid #cccccc;
            }
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #0d7377;
            }
            QMenuBar {
                background-color: #e0e0e0;
                color: #333333;
            }
            QMenuBar::item:selected {
                background-color: #0d7377;
                color: white;
            }
            QMenu {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
            }
            QMenu::item:selected {
                background-color: #0d7377;
                color: white;
            }
            QToolBar {
                background-color: #e0e0e0;
                border: none;
                spacing: 5px;
            }
            QStatusBar {
                background-color: #e0e0e0;
                color: #333333;
            }
        """)
    
    def _set_application_icon(self):
        """Set application icon from logo.png with fallback to generic icon.
        
        Tries to load logo.png from multiple locations:
        - PyInstaller bundle root (sys._MEIPASS)
        - Project root (development mode)
        - Current working directory
        Falls back to a generic system application icon if logo.png is not found.
        """
        # v1.3.0: Support both development and PyInstaller frozen mode
        logo_paths = []
        
        # 1. PyInstaller bundle (frozen executable)
        if hasattr(sys, '_MEIPASS'):
            logo_paths.append(Path(sys._MEIPASS) / "logo.png")
        
        # 2. Project root (development mode)
        logo_paths.append(Path(__file__).parent.parent.parent / "logo.png")
        logo_paths.append(Path(__file__).parent.parent.parent / "assets" / "logo.png")
        logo_paths.append(Path(__file__).parent.parent.parent / "images" / "logo.png")
        
        # 3. Current working directory
        logo_paths.append(Path.cwd() / "logo.png")
        
        for logo_path in logo_paths:
            if logo_path.exists():
                try:
                    icon = QIcon(str(logo_path))
                    self.setWindowIcon(icon)
                    # Also set for QApplication if available
                    app = QApplication.instance()
                    if app:
                        app.setWindowIcon(icon)
                    self.logger.info(f"Loaded logo from {logo_path}")
                    return
                except Exception as e:
                    self.logger.warning(f"Failed to load logo from {logo_path}: {e}")
                    continue
        
        # Fallback: use generic system icon (silent, no error)
        self.logger.info("No logo.png found, using default system icon")
    
    def _create_menu_bar(self):
        """Create menu bar."""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu('File')
        
        load_data_action = QAction('Load Data...', self)
        load_data_action.setShortcut('Ctrl+O')
        load_data_action.triggered.connect(self._load_data)
        file_menu.addAction(load_data_action)
        
        save_model_action = QAction('Save Model...', self)
        save_model_action.setShortcut('Ctrl+S')
        save_model_action.triggered.connect(self._save_model)
        file_menu.addAction(save_model_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction('Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # Tools menu
        tools_menu = menubar.addMenu('Tools')
        
        settings_action = QAction('Settings...', self)
        settings_action.triggered.connect(self._show_settings)
        tools_menu.addAction(settings_action)
        
        hardware_action = QAction('Hardware Info...', self)
        hardware_action.triggered.connect(self._show_hardware_info)
        tools_menu.addAction(hardware_action)
        
        # Help menu
        help_menu = menubar.addMenu('Help')
        
        about_action = QAction('About', self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)
    
    def _create_toolbar(self):
        """Create toolbar."""
        toolbar = QToolBar()
        self.addToolBar(toolbar)
        
        # Quick actions
        load_btn = QPushButton("Load Data")
        load_btn.clicked.connect(self._load_data)
        toolbar.addWidget(load_btn)
        
        toolbar.addSeparator()
        
        train_btn = QPushButton("Train Models")
        train_btn.clicked.connect(self._quick_train)
        toolbar.addWidget(train_btn)
        
        optimize_btn = QPushButton("Optimize")
        optimize_btn.clicked.connect(self._quick_optimize)
        toolbar.addWidget(optimize_btn)
        
        toolbar.addSeparator()
        
        # Theme selector
        toolbar.addWidget(QLabel("Theme:"))
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(['Dark', 'Light'])
        self.theme_combo.setCurrentText(self.config.current.theme.capitalize())
        self.theme_combo.currentTextChanged.connect(self._change_theme)
        toolbar.addWidget(self.theme_combo)
    
    def _create_tabs(self):
        """Create tab widget with all tabs."""
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        # v1.3.0: Tabs in logical workflow order
        # 1. Data loading and preprocessing
        self.data_tab = DataTab(self)
        self.tabs.addTab(self.data_tab, "Data")
        
        # 2. Individual model training
        self.training_tab = TrainingTab(self)
        self.tabs.addTab(self.training_tab, "Training")
        
        # 3. Hyperparameter optimization
        self.optimization_tab = OptimizationTab(self)
        self.tabs.addTab(self.optimization_tab, "Optimization")
        
        # 4. Ensemble methods (needs trained models from Training)
        self.ensemble_tab = EnsembleTab(self)
        self.tabs.addTab(self.ensemble_tab, "Ensemble")
        
        # 5. Neural Network Builder
        self.nn_tab = NNBuilderTab(self)
        self.tabs.addTab(self.nn_tab, "Neural Network")
        
        # 6. Results overview (all trained/optimized models)
        self.results_tab = ResultsTab(self)
        self.tabs.addTab(self.results_tab, "Results")
        
        # 7. Model explainability (needs trained models)
        self.explainability_tab = ExplainabilityTab(self)
        self.tabs.addTab(self.explainability_tab, "Explainability")
        
        # 8. Prediction/Inference (final step)
        self.inference_tab = InferenceTab(self)
        self.tabs.addTab(self.inference_tab, "Predict")
    
    def _create_status_bar(self):
        """Create status bar."""
        self.statusbar = QStatusBar()
        self.setStatusBar(self.statusbar)
        
        # Hardware info
        hardware_text = f"CPU: {self.hardware.info.cpu_count} cores"
        if self.hardware.info.has_cuda:
            hardware_text += f" | GPU: {self.hardware.info.gpu_count}x CUDA"
        self.statusbar.showMessage(f"Ready | {hardware_text}")
    
    def _show_welcome(self):
        """Show welcome message."""
        welcome_text = f"""
        <h2>Welcome to {self.config.current.app_name}</h2>
        <p>Get started by loading your dataset:</p>
        <ol>
            <li>Go to the <b>Data</b> tab</li>
            <li>Click <b>Load Data</b> to import your CSV file</li>
            <li>Select your target variable</li>
            <li>Proceed to the <b>Training</b> tab to train models</li>
        </ol>
        <p>For help, check the Help menu.</p>
        """
        # Could show in a dialog or info panel
    
    def _load_data(self):
        """Open dialog to load data."""
        self.tabs.setCurrentWidget(self.data_tab)
        self.data_tab.load_data_dialog()
    
    def _save_model(self):
        """Save trained model."""
        self.tabs.setCurrentWidget(self.results_tab)
        self.results_tab.save_model_dialog()
    
    def _quick_train(self):
        """Quick train all models."""
        self.tabs.setCurrentWidget(self.training_tab)
        self.training_tab.train_all_models()
    
    def _quick_optimize(self):
        """Quick optimize best model."""
        self.tabs.setCurrentWidget(self.optimization_tab)
        self.optimization_tab.optimize_best_model()
    
    def _change_theme(self, theme: str):
        """Change application theme."""
        self.config.update(theme=theme.lower())
        self._apply_theme()
    
    def _show_settings(self):
        """Show settings dialog."""
        QMessageBox.information(self, "Settings", "Settings dialog not yet implemented.")
    
    def _show_hardware_info(self):
        """Show hardware information dialog."""
        info = self.hardware.format_info()
        
        dialog = QMessageBox(self)
        dialog.setWindowTitle("Hardware Information")
        dialog.setText("<pre>" + info + "</pre>")
        dialog.setStandardButtons(QMessageBox.StandardButton.Ok)
        dialog.exec()
    
    def _show_about(self):
        """Show about dialog."""
        version = "1.3.0"
        about_text = f"""
        <h2>AutoAIM Studio</h2>
        <p><b>Version:</b> {version}</p>
        <p><b>Codename:</b> Crystal</p>
        <p>An AutoML platform for materials science.</p>
        <p><b>v1.2.1 Features:</b></p>
        <ul>
            <li>Automated machine learning for tabular materials data</li>
            <li>Crystal structure support (CIF, POSCAR, XYZ) with pymatgen featurization</li>
            <li>Neural network builder with PyTorch</li>
            <li>Bayesian hyperparameter optimization</li>
            <li>Model explainability with SHAP</li>
            <li>Domain applicability analysis</li>
            <li>Ensemble modeling with weight optimization</li>
        </ul>
        <p><b>Built with:</b> PyQt6, scikit-learn, PyTorch, Optuna, pymatgen</p>
        <p><b>Note:</b> Crystal structure features use only pymatgen (no matminer).</p>
        """
        
        QMessageBox.about(self, "About AutoAIM Studio", about_text)
    
    def closeEvent(self, event):
        """Handle window close event."""
        reply = QMessageBox.question(
            self,
            'Confirm Exit',
            'Are you sure you want to exit?\nAny unsaved work will be lost.',
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            self.config.save()
            event.accept()
        else:
            event.ignore()


def run_application():
    """Run the GUI application."""
    app = QApplication(sys.argv)
    
    # Set application info
    app.setApplicationName("AutoAIM Studio")
    app.setApplicationVersion("1.3.0")
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())
