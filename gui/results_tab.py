"""Results and model export tab."""

import logging
import traceback
import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QFileDialog, QMessageBox, QTextEdit, QSplitter,
    QTabWidget, QCheckBox, QLineEdit
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor

from ..core.trainer import ModelTrainer
from ..core.ensemble_trainer import EnsembleResult
from ..core.session_exporter import SessionExporter
from ..core.cv_display import format_cv_details, cv_strategy_label

logger = logging.getLogger(__name__)


def _safe_get_parent_attr(widget, attr_path, default=None):
    """Safely get a nested attribute from the widget's main window.
    
    v2.0 FIX: After addTab(), widget.parent() returns the QTabWidget — NOT
    the MainWindow. We first try the _main_window reference stored during
    __init__, then fall back to walking the entire parent chain until we find
    an object that has the first attribute in attr_path.
    
    Args:
        widget: The widget that has a .parent() method
        attr_path: Dot-separated path like 'training_tab.get_trainer'
        default: Value to return if path cannot be resolved
        
    Returns:
        The resolved attribute or default
    """
    if widget is None:
        return default
    
    parts = attr_path.split('.')
    first_attr = parts[0]
    
    # v2.0: Strategy 1 — use the _main_window reference stored in __init__
    # before addTab() reparented the widget.
    obj = getattr(widget, '_main_window', None)
    if obj is not None and hasattr(obj, first_attr):
        # Fast path: _main_window exists and has the attribute we need
        pass  # obj is already correct, proceed to traverse the rest
    else:
        # Strategy 2 — walk the full parent chain looking for an object
        # that has the first attribute. This covers addTab() reparenting
        # (QTabWidget -> QStackedWidget -> ... -> MainWindow).
        try:
            current = widget.parent()
        except (TypeError, AttributeError):
            return default
        
        obj = None
        visited = set()  # Guard against circular references
        while current is not None:
            obj_id = id(current)
            if obj_id in visited:
                break
            visited.add(obj_id)
            if hasattr(current, first_attr):
                obj = current
                break
            try:
                current = current.parent()
            except (TypeError, AttributeError):
                break
        
        if obj is None:
            return default
    
    # Traverse the rest of the attribute path
    try:
        for part in parts:
            if part.endswith('()'):
                obj = getattr(obj, part[:-2])()
            else:
                if not hasattr(obj, part):
                    return default
                obj = getattr(obj, part)
        return obj
    except Exception:
        return default


class ResultsTab(QWidget):
    """Results and model export tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        # v2.0 FIX: Store main_window reference before addTab() reparents us.
        # After addTab(), self.parent() returns the QTabWidget (not MainWindow),
        # so we need this stored reference to access sibling tabs.
        self._main_window = parent
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Vertical)
        
        # Top panel: Model comparison
        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.setContentsMargins(0, 0, 0, 0)
        
        top_layout.addWidget(QLabel("<b>Model Comparison</b>"))
        
        self.comparison_table = QTableWidget()
        # v1.5.0: Added external validation metric columns
        self.comparison_table.setColumnCount(12)
        self.comparison_table.setHorizontalHeaderLabels([
            'Model', 'RMSE/Acc', 'MAE', 'R²/F1', 'CV Score',
            'Training Time', 'Rank', 'Select', 'Actions',
            'Val. RMSE/Acc', 'Val. MAE', 'Val. R²/F1'
        ])
        top_layout.addWidget(self.comparison_table)
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh Results")
        self.refresh_btn.setToolTip("Update the table with all trained models from this session")
        self.refresh_btn.clicked.connect(self._refresh_results)
        top_layout.addWidget(self.refresh_btn)
        
        splitter.addWidget(top_panel)
        
        # Bottom panel: Export options
        bottom_panel = QWidget()
        bottom_layout = QVBoxLayout(bottom_panel)
        bottom_layout.setContentsMargins(0, 0, 0, 0)
        
        # Export tabs
        export_tabs = QTabWidget()
        
        # Model export
        model_export = QWidget()
        model_layout = QVBoxLayout(model_export)
        
        model_layout.addWidget(QLabel("Select Model to Export:"))
        self.export_model_combo = QComboBox()
        model_layout.addWidget(self.export_model_combo)
        
        export_btn_layout = QHBoxLayout()
        
        self.export_joblib_btn = QPushButton("Export as Joblib")
        self.export_joblib_btn.setToolTip("Export the raw sklearn model for use in Python scripts")
        self.export_joblib_btn.clicked.connect(self._export_joblib_legacy)
        export_btn_layout.addWidget(self.export_joblib_btn)
        
        self.export_bundle_btn = QPushButton("Export as Model Bundle (Predict)")
        self.export_bundle_btn.setToolTip(
            "Export a complete bundle with model + preprocessing + manifest. "
            "Use this bundle in the Predict tab to make predictions on new data."
        )
        self.export_bundle_btn.clicked.connect(self._export_model_bundle)
        self.export_bundle_btn.setStyleSheet("background-color: #0d7377; color: white; font-weight: bold;")
        export_btn_layout.addWidget(self.export_bundle_btn)
        
        model_layout.addLayout(export_btn_layout)
        
        # Pipeline export
        self.export_pipeline_check = QCheckBox("Include preprocessing pipeline")
        self.export_pipeline_check.setToolTip(
            "When checked, the exported model bundle includes the preprocessing "
            "pipeline so predictions automatically apply the same scaling/imputation."
        )
        self.export_pipeline_check.setChecked(True)
        model_layout.addWidget(self.export_pipeline_check)
        
        model_layout.addStretch()
        
        export_tabs.addTab(model_export, "Model Export")
        
        # Report generation
        report_export = QWidget()
        report_layout = QVBoxLayout(report_export)
        
        report_layout.addWidget(QLabel("Generate Report:"))
        
        # Session Report Export (v1.5.0)
        report_layout.addWidget(QLabel("<b>Export Complete Session</b>"))
        session_info = QLabel(
            "<small>Exports ALL results from this session to a single file: "
            "trained models, optimized parameters, neural network training, "
            "ensembles, and metrics — with a final ranking table.</small>"
        )
        session_info.setWordWrap(True)
        report_layout.addWidget(session_info)

        session_btn_layout = QHBoxLayout()

        self.export_session_txt_btn = QPushButton("Export as .txt")
        self.export_session_txt_btn.setToolTip("Human-readable text report with all results and ranking")
        self.export_session_txt_btn.clicked.connect(lambda: self._export_session_report("txt"))
        session_btn_layout.addWidget(self.export_session_txt_btn)

        self.export_session_md_btn = QPushButton("Export as .md")
        self.export_session_md_btn.setToolTip("Markdown report for GitHub, papers, or documentation")
        self.export_session_md_btn.clicked.connect(lambda: self._export_session_report("md"))
        session_btn_layout.addWidget(self.export_session_md_btn)

        self.export_session_csv_btn = QPushButton("Export as .csv")
        self.export_session_csv_btn.setToolTip("Spreadsheet-friendly table for Excel")
        self.export_session_csv_btn.clicked.connect(lambda: self._export_session_report("csv"))
        session_btn_layout.addWidget(self.export_session_csv_btn)

        self.export_session_json_btn = QPushButton("Export as .json")
        self.export_session_json_btn.setToolTip("Machine-readable JSON with complete data")
        self.export_session_json_btn.clicked.connect(lambda: self._export_session_report("json"))
        session_btn_layout.addWidget(self.export_session_json_btn)

        report_layout.addLayout(session_btn_layout)

        report_layout.addSpacing(20)
        report_layout.addWidget(QLabel("<b>Generate Report</b>"))

        self.report_btn = QPushButton("Generate PDF Report")
        self.report_btn.clicked.connect(self._generate_report)
        report_layout.addWidget(self.report_btn)

        self.html_report_btn = QPushButton("Generate HTML Report")
        self.html_report_btn.clicked.connect(self._generate_html_report)
        report_layout.addWidget(self.html_report_btn)

        report_layout.addStretch()
        
        export_tabs.addTab(report_export, "Reports")
        
        # Learning Curves
        learning_export = QWidget()
        learning_layout = QVBoxLayout(learning_export)
        
        learning_layout.addWidget(QLabel("Generate Learning Curves (Training vs Validation Performance):"))
        
        learning_btn_layout = QHBoxLayout()
        
        self.learning_curve_btn = QPushButton("Generate Learning Curves")
        self.learning_curve_btn.clicked.connect(self._generate_learning_curves)
        learning_btn_layout.addWidget(self.learning_curve_btn)
        
        self.export_learning_btn = QPushButton("Export to CSV")
        self.export_learning_btn.clicked.connect(self._export_learning_curves)
        self.export_learning_btn.setEnabled(False)
        learning_btn_layout.addWidget(self.export_learning_btn)
        
        learning_layout.addLayout(learning_btn_layout)
        
        self.learning_curve_table = QTableWidget()
        self.learning_curve_table.setColumnCount(4)
        self.learning_curve_table.setHorizontalHeaderLabels([
            'Training Size', 'Train Score', 'Validation Score', 'Gap'
        ])
        learning_layout.addWidget(self.learning_curve_table)
        
        learning_info = QLabel(
            "<small>Learning curves show how model performance improves with more training data. "
            "A large gap between train and validation scores indicates overfitting.</small>"
        )
        learning_layout.addWidget(learning_info)
        
        learning_layout.addStretch()
        
        export_tabs.addTab(learning_export, "Learning Curves")
        
        bottom_layout.addWidget(export_tabs)
        
        splitter.addWidget(bottom_panel)
        splitter.setSizes([400, 300])
        
        layout.addWidget(splitter)
        
        # Initial refresh
        self._refresh_results()
    
    def _refresh_results(self):
        """Refresh results from training, optimization, and neural networks."""
        if self._main_window is None:
            return
        
        all_results = []
        
        try:
            # Get trainer results
            trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
            if trainer and trainer.results:
                for name, result in trainer.results.items():
                    if result is not None:
                        all_results.append((name, result, "Trained"))
            
            # v1.2.0 FIX: Do NOT show optimization_results here.
            # When the user clicks "Train with Best Parameters", a proper
            # TrainingResult is created and registered in trainer.results.
            # Showing the intermediate OptimizationResult alongside creates
            # a confusing duplicate entry with the same metrics.
            # The optimization details remain accessible in the Optimization tab.
            
            # Get neural network results (training only — not optimization)
            # v1.2.1 FIX: Filter out duplicates already present in trainer.results.
            # train_neural_network() registers in BOTH trainer.results AND
            # nn_training_results, so the same model would appear twice.
            trainer_names = {name for name, _, _ in all_results}
            nn_tab = _safe_get_parent_attr(self, 'nn_tab')
            if nn_tab and hasattr(nn_tab, 'nn_training_results') and nn_tab.nn_training_results:
                    for nn_name, nn_result in nn_tab.nn_training_results:
                        if nn_result is not None and nn_name not in trainer_names:
                            all_results.append((
                                nn_name,
                                nn_result,
                                "Neural Network"
                            ))
                
                # v1.2.0 FIX: Do NOT show nn_optimizer_results.
                # Same reason: "Apply Best Parameters + Train" creates a
                # proper TrainingResult in nn_training_results. Showing
                # the intermediate NN optimization result creates a duplicate.
            
            # Get ensemble results
            ens_tab = _safe_get_parent_attr(self, 'ensemble_tab')
            if ens_tab and hasattr(ens_tab, 'ensemble_trainer') and ens_tab.ensemble_trainer:
                    try:
                        ens_trainer = ens_tab.ensemble_trainer
                        if hasattr(ens_trainer, 'results') and ens_trainer.results:
                            for ens_name, ens_result in ens_trainer.results.items():
                                if ens_result is not None:
                                    all_results.append((
                                        ens_name,
                                        ens_result,
                                        "Ensemble"
                                    ))
                    except Exception as e:
                        print(f"Warning: Could not load ensemble results: {e}")
            
            if all_results:
                # Update comparison table with all results
                self._update_comparison_table_all(all_results)
                
                # Update export combo
                self.export_model_combo.clear()
                for name, _, _ in all_results:
                    self.export_model_combo.addItem(name)
                    
        except Exception as e:
            import traceback
            error_msg = f"Failed to refresh results: {str(e)}"
            logger.error(error_msg, exc_info=True)
            QMessageBox.warning(
                self, "Refresh Error",
                f"Could not refresh results table.\n\nError: {str(e)}\n\n"
                f"Check the console for full traceback."
            )
    
    def _update_comparison_table_all(self, all_results):
        """Update comparison table with all model types."""
        # Sort by available metric (handle different result types)
        def get_sort_key(item):
            result = item[1]
            if hasattr(result, 'metrics') and result.metrics is not None:
                return result.metrics.get('r2', result.metrics.get('accuracy', 0))
            elif hasattr(result, 'best_score') and result.best_score is not None:
                return result.best_score if result.best_score > 0 else -result.best_score
            return 0
        
        sorted_results = sorted(all_results, key=get_sort_key, reverse=True)
        
        self.comparison_table.setRowCount(len(sorted_results))
        
        for i, (name, result, model_type) in enumerate(sorted_results):
            # Model name with type indicator
            display_name = f"{name}\n[{model_type}]"
            item = QTableWidgetItem(display_name)
            if model_type == "Optimized":
                item.setBackground(QColor(13, 115, 119, 50))  # Teal tint
            elif model_type == "Neural Network":
                item.setBackground(QColor(115, 13, 117, 50))  # Purple tint
            elif model_type == "NN Optimized":
                item.setBackground(QColor(80, 20, 120, 50))  # Dark purple
            elif model_type == "Ensemble":
                item.setBackground(QColor(20, 80, 120, 50))  # Blue tint
            self.comparison_table.setItem(i, 0, item)
            
            # Metrics - handle different result types safely
            problem_type = getattr(result, 'problem_type', 'regression')
            metrics = getattr(result, 'metrics', None) or {}
            best_score = getattr(result, 'best_score', None)
            
            if metrics:
                if problem_type == 'regression':
                    self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{metrics.get('rmse', 0):.4f}"))
                    self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{metrics.get('mae', 0):.4f}"))
                    self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{metrics.get('r2', 0):.4f}"))
                else:
                    self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{metrics.get('accuracy', 0):.4f}"))
                    self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{metrics.get('precision', 0):.4f}"))
                    self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{metrics.get('f1', 0):.4f}"))
            elif best_score is not None:
                # OptimizationResult type without metrics - use best_score
                is_regression = best_score < 0
                if is_regression:
                    rmse = (-best_score) ** 0.5
                    self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{rmse:.4f}"))
                    self.comparison_table.setItem(i, 2, QTableWidgetItem("N/A"))
                    self.comparison_table.setItem(i, 3, QTableWidgetItem("N/A"))
                else:
                    self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{best_score:.4f}"))
                    self.comparison_table.setItem(i, 2, QTableWidgetItem("N/A"))
                    self.comparison_table.setItem(i, 3, QTableWidgetItem("N/A"))
            else:
                self.comparison_table.setItem(i, 1, QTableWidgetItem("N/A"))
                self.comparison_table.setItem(i, 2, QTableWidgetItem("N/A"))
                self.comparison_table.setItem(i, 3, QTableWidgetItem("N/A"))
            
            # CV score - safely handle cv_metrics and cv_scores
            cv_label = "N/A"
            cv_metrics = getattr(result, 'cv_metrics', None)
            if cv_metrics and isinstance(cv_metrics, dict):
                test_scores = cv_metrics.get('test_score')
                if test_scores and len(test_scores) > 0:
                    cv_mean = np.mean(test_scores)
                    cv_std = np.std(test_scores)
                    cv_label = f"{cv_mean:.4f} ± {cv_std:.4f}"
            
            # Also check cv_scores (for EnsembleResult)
            cv_scores = getattr(result, 'cv_scores', None)
            if cv_label == "N/A" and cv_scores and len(cv_scores) > 0:
                cv_mean = np.mean(cv_scores)
                cv_std = np.std(cv_scores)
                cv_label = f"{cv_mean:.4f} ± {cv_std:.4f}"
            
            self.comparison_table.setItem(i, 4, QTableWidgetItem(cv_label))
            
            # v1.5.0: External validation metrics (dual reporting)
            # Column indices: 9=Val.RMSE/Acc, 10=Val.MAE, 11=Val.R2/F1
            ext_metrics = getattr(result, 'external_metrics', None)
            has_ext = getattr(result, 'has_external_validation', False)
            if has_ext and ext_metrics:
                if problem_type == 'regression':
                    self.comparison_table.setItem(i, 9, QTableWidgetItem(f"{ext_metrics.get('rmse', 0):.4f}"))
                    self.comparison_table.setItem(i, 10, QTableWidgetItem(f"{ext_metrics.get('mae', 0):.4f}"))
                    self.comparison_table.setItem(i, 11, QTableWidgetItem(f"{ext_metrics.get('r2', 0):.4f}"))
                else:
                    self.comparison_table.setItem(i, 9, QTableWidgetItem(f"{ext_metrics.get('accuracy', 0):.4f}"))
                    self.comparison_table.setItem(i, 10, QTableWidgetItem(f"{ext_metrics.get('precision', 0):.4f}"))
                    self.comparison_table.setItem(i, 11, QTableWidgetItem(f"{ext_metrics.get('f1', 0):.4f}"))
            else:
                self.comparison_table.setItem(i, 9, QTableWidgetItem("—"))
                self.comparison_table.setItem(i, 10, QTableWidgetItem("—"))
                self.comparison_table.setItem(i, 11, QTableWidgetItem("—"))
            
            # Training time
            time_val = getattr(result, 'training_time', None) or getattr(result, 'optimization_time', None)
            if time_val is not None:
                self.comparison_table.setItem(i, 5, QTableWidgetItem(f"{time_val:.2f}"))
            else:
                self.comparison_table.setItem(i, 5, QTableWidgetItem("N/A"))
            
            # Rank
            self.comparison_table.setItem(i, 6, QTableWidgetItem(str(i + 1)))
            
            # Select checkbox
            select_widget = QWidget()
            select_layout = QHBoxLayout(select_widget)
            select_layout.setContentsMargins(5, 0, 5, 0)
            select_check = QCheckBox()
            select_check.setProperty('model_name', name)
            select_layout.addWidget(select_check)
            select_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.comparison_table.setCellWidget(i, 7, select_widget)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(2)
            
            details_btn = QPushButton("Details")
            details_btn.setProperty('model_name', name)
            details_btn.setMaximumWidth(55)
            details_btn.setStyleSheet("padding: 2px 4px; font-size: 8pt;")
            details_btn.clicked.connect(self._show_model_details)
            actions_layout.addWidget(details_btn)
            
            delete_btn = QPushButton("Delete")
            delete_btn.setProperty('model_name', name)
            delete_btn.setMaximumWidth(55)
            delete_btn.setStyleSheet("padding: 2px 4px; font-size: 8pt; background-color: #d32f2f; color: white;")
            delete_btn.clicked.connect(self._delete_model_result)
            actions_layout.addWidget(delete_btn)
            
            self.comparison_table.setCellWidget(i, 8, actions_widget)
    
    def _update_comparison_table(self, trainer: ModelTrainer):
        """Update comparison table."""
        results = trainer.results
        
        # Sort by R2 or accuracy
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1].metrics.get('r2', x[1].metrics.get('accuracy', 0)),
            reverse=True
        )
        
        self.comparison_table.setRowCount(len(sorted_results))
        
        for i, (name, result) in enumerate(sorted_results):
            self.comparison_table.setItem(i, 0, QTableWidgetItem(name))
            
            # Metrics
            if result.problem_type == 'regression':
                self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{result.metrics.get('rmse', 0):.4f}"))
                self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{result.metrics.get('mae', 0):.4f}"))
                self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{result.metrics.get('r2', 0):.4f}"))
            else:
                self.comparison_table.setItem(i, 1, QTableWidgetItem(f"{result.metrics.get('accuracy', 0):.4f}"))
                self.comparison_table.setItem(i, 2, QTableWidgetItem(f"{result.metrics.get('precision', 0):.4f}"))
                self.comparison_table.setItem(i, 3, QTableWidgetItem(f"{result.metrics.get('f1', 0):.4f}"))
            
            # CV score
            if result.cv_metrics.get('test_score'):
                cv_mean = np.mean(result.cv_metrics['test_score'])
                cv_std = np.std(result.cv_metrics['test_score'])
                self.comparison_table.setItem(i, 4, QTableWidgetItem(f"{cv_mean:.4f} ± {cv_std:.4f}"))
            
            # Training time
            self.comparison_table.setItem(i, 5, QTableWidgetItem(f"{result.training_time:.2f}"))
            
            # Rank
            self.comparison_table.setItem(i, 6, QTableWidgetItem(str(i + 1)))
            
            # Select checkbox
            select_widget = QWidget()
            select_layout = QHBoxLayout(select_widget)
            select_layout.setContentsMargins(5, 0, 5, 0)
            select_check = QCheckBox()
            select_check.setProperty('model_name', name)
            select_layout.addWidget(select_check)
            select_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.comparison_table.setCellWidget(i, 7, select_widget)
            
            # Actions
            actions_widget = QWidget()
            actions_layout = QHBoxLayout(actions_widget)
            actions_layout.setContentsMargins(2, 0, 2, 0)
            actions_layout.setSpacing(2)
            
            details_btn = QPushButton("Details")
            details_btn.setProperty('model_name', name)
            details_btn.setMaximumWidth(55)
            details_btn.setStyleSheet("padding: 2px 4px; font-size: 8pt;")
            details_btn.clicked.connect(self._show_model_details)
            actions_layout.addWidget(details_btn)
            
            delete_btn = QPushButton("Delete")
            delete_btn.setProperty('model_name', name)
            delete_btn.setMaximumWidth(55)
            delete_btn.setStyleSheet("padding: 2px 4px; font-size: 8pt; background-color: #d32f2f; color: white;")
            delete_btn.clicked.connect(self._delete_model_result)
            actions_layout.addWidget(delete_btn)
            
            self.comparison_table.setCellWidget(i, 8, actions_widget)
    
    def _delete_model_result(self):
        """Delete a model result from all storage locations."""
        btn = self.sender()
        model_name = btn.property('model_name')
        
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete model '{model_name}' from results?\n\n"
            f"This will remove it from the results table but will NOT "
            f"delete any saved model files.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        deleted = False
        locations_checked = []
        
        # 1. Try trainer.results (primary storage for trained/optimized/manual models)
        trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
        if trainer:
            locations_checked.append(f"trainer.results keys: {list(trainer.results.keys())}")
            if model_name in trainer.results:
                del trainer.results[model_name]
                deleted = True
                logger.info(f"Deleted '{model_name}' from trainer.results")
        
        # 2. Try training_tab.training_results (local list, includes manual models)
        training_tab = _safe_get_parent_attr(self, 'training_tab')
        if training_tab and hasattr(training_tab, 'training_results'):
            orig = len(training_tab.training_results)
            training_tab.training_results = [
                r for r in training_tab.training_results 
                if getattr(r, 'model_name', None) != model_name
            ]
            if len(training_tab.training_results) < orig:
                deleted = True
                logger.info(f"Deleted '{model_name}' from training_tab.training_results")
        
        # 3. Try NN training results
        nn_tab = _safe_get_parent_attr(self, 'nn_tab')
        if nn_tab and hasattr(nn_tab, 'nn_training_results'):
            orig = len(nn_tab.nn_training_results)
            nn_tab.nn_training_results = [
                (n, r) for n, r in nn_tab.nn_training_results if n != model_name
            ]
            if len(nn_tab.nn_training_results) < orig:
                deleted = True
                logger.info(f"Deleted '{model_name}' from nn_training_results")
        
        # 4. Try ensemble results
        ens_tab = _safe_get_parent_attr(self, 'ensemble_tab')
        if ens_tab and hasattr(ens_tab, 'ensemble_trainer') and ens_tab.ensemble_trainer:
            if model_name in ens_tab.ensemble_trainer.results:
                del ens_tab.ensemble_trainer.results[model_name]
                deleted = True
                logger.info(f"Deleted '{model_name}' from ensemble_trainer.results")
        
        # 5. Try optimization_tab.optimization_results (v1.5.0 fix)
        # Optimization results are stored as (model_type, result) tuples
        opt_tab = _safe_get_parent_attr(self, 'optimization_tab')
        if opt_tab and hasattr(opt_tab, 'optimization_results'):
            orig = len(opt_tab.optimization_results)
            opt_tab.optimization_results = [
                (mt, r) for mt, r in opt_tab.optimization_results
                if not (hasattr(r, 'model_name') and r.model_name == model_name)
                and mt != model_name
            ]
            if len(opt_tab.optimization_results) < orig:
                deleted = True
                logger.info(f"Deleted '{model_name}' from optimization_tab.optimization_results")
        
        if deleted:
            self._refresh_results()
        else:
            logger.warning(f"Could not find '{model_name}' to delete. Checked: {locations_checked}")
            QMessageBox.warning(
                self, "Warning", 
                f"Could not find '{model_name}' to delete.\n\n"
                f"The model may have already been deleted, or it may be "
                f"stored in a different location. Try refreshing the results."
            )
    
    def _show_model_details(self):
        """Show model details for trained, optimized, and neural network models."""
        btn = self.sender()
        model_name = btn.property('model_name')
        
        if not self._main_window:
            QMessageBox.warning(self, "Warning", "Cannot access parent window")
            return
        
        result = None
        source_type = ""
        
        # 1. Check trained models
        if hasattr(self._main_window, 'training_tab'):
            trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
            if model_name in trainer.results:
                result = trainer.results[model_name]
                source_type = "Trained Model"
        
        # 2. Check optimized models
        if result is None and hasattr(self._main_window, 'optimization_tab'):
            opt_tab = _safe_get_parent_attr(self, 'optimization_tab')
            if hasattr(opt_tab, 'optimization_results') and opt_tab.optimization_results:
                for opt_name, opt_result in opt_tab.optimization_results:
                    if f"{opt_name} (Optimized)" == model_name:
                        result = opt_result
                        source_type = "Optimized Model"
                        break
        
        # 3. Check neural networks
        if result is None and hasattr(self._main_window, 'nn_tab'):
            nn_tab = _safe_get_parent_attr(self, 'nn_tab')
            if hasattr(nn_tab, 'nn_training_results') and nn_tab.nn_training_results:
                for nn_name, nn_result in nn_tab.nn_training_results:
                    if nn_name == model_name:
                        result = nn_result
                        source_type = "Neural Network"
                        break
            
            # 3b. Check NN optimized models
            if result is None and hasattr(nn_tab, 'nn_optimizer_results') and nn_tab.nn_optimizer_results:
                for opt_name, opt_result in nn_tab.nn_optimizer_results:
                    if f"{opt_name} (NN Optimized)" == model_name:
                        result = opt_result
                        source_type = "NN Optimized"
                        break
        
        # 4. Check ensemble models
        if result is None and hasattr(self._main_window, 'ensemble_tab'):
            ens_tab = _safe_get_parent_attr(self, 'ensemble_tab')
            if hasattr(ens_tab, 'get_ensemble_trainer'):
                ens_trainer = ens_tab.get_ensemble_trainer()
                if ens_trainer and hasattr(ens_trainer, 'results') and ens_trainer.results:
                    if model_name in ens_trainer.results:
                        result = ens_trainer.results[model_name]
                        source_type = "Ensemble"
        
        if result is None:
            QMessageBox.warning(self, "Warning", f"Model '{model_name}' not found in any source")
            return
        
        # Build details dialog
        details = f"""
        <h3>Model: {model_name}</h3>
        <p><b>Source:</b> {source_type}</p>
        """
        
        # Problem type
        if hasattr(result, 'problem_type'):
            details += f"<p><b>Problem Type:</b> {result.problem_type}</p>"

        # v2.0: Search strategy (for optimized models)
        search_strategy = getattr(result, 'search_strategy', None)
        if search_strategy:
            strat_label = "Bayesian (Optuna)" if search_strategy == "bayesian" else "Grid Search"
            details += f"<p><b>Search Strategy:</b> {strat_label}</p>"

        # v2.0: CV strategy and folds (full label via the shared formatter)
        cv_strategy = getattr(result, 'cv_strategy', 'kfold')
        cv_config_dict = getattr(result, 'cv_config_dict', None)
        details += f"<p><b>CV Strategy:</b> {cv_strategy_label(cv_strategy, cv_config_dict)}</p>"

        # CV folds (only meaningful for fold-based strategies; for LOO/LOGO the
        # number of folds equals N samples / n_groups automatically).
        cv_folds = None
        if hasattr(result, 'training_config') and result.training_config:
            cv_folds = result.training_config.get('cv_folds')
        elif hasattr(result, 'optimization_config') and result.optimization_config:
            cv_folds = result.optimization_config.get('cv_folds')
        if cv_folds is not None and cv_strategy not in ('loo', 'logo'):
            details += f"<p><b>CV Folds:</b> {cv_folds}</p>"

        # Training/Optimization time
        if hasattr(result, 'training_time'):
            details += f"<p><b>Training Time:</b> {result.training_time:.2f} seconds</p>"
        elif hasattr(result, 'optimization_time'):
            details += f"<p><b>Optimization Time:</b> {result.optimization_time:.2f} seconds</p>"
        
        # Number of trials for optimized models
        if hasattr(result, 'n_trials'):
            details += f"<p><b>Number of Trials:</b> {result.n_trials}</p>"
        
        # Best params for optimized models
        if hasattr(result, 'best_params') and result.best_params:
            details += "<h4>Best Parameters:</h4><ul>"
            for param, value in result.best_params.items():
                details += f"<li><b>{param}:</b> {value}</li>"
            details += "</ul>"
        
        # Metrics
        if hasattr(result, 'metrics') and result.metrics:
            details += "<h4>Metrics:</h4><ul>"
            for metric, value in result.metrics.items():
                if isinstance(value, (int, float)):
                    details += f"<li><b>{metric.upper()}:</b> {value:.6f}</li>"
                else:
                    details += f"<li><b>{metric.upper()}:</b> {value}</li>"
            details += "</ul>"
        
        # Best score for optimized models
        if hasattr(result, 'best_score'):
            details += f"<p><b>Best Score:</b> {result.best_score:.6f}</p>"
        
        # Feature importance (only for models that support it)
        if hasattr(result, 'feature_importance') and result.feature_importance:
            details += "<h4>Top 10 Important Features:</h4><ul>"
            sorted_imp = sorted(
                result.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
            for feat, imp in sorted_imp:
                details += f"<li>{feat}: {imp:.4f}</li>"
            details += "</ul>"
        
        # v2.0: Cross-Validation breakdown — delegated to the shared formatter
        # (cv_display.format_cv_details), which renders every strategy:
        # kfold / stratified / group (with Test Size) / loo / logo.
        cv_metrics_dict = None
        if hasattr(result, 'cv_metrics') and result.cv_metrics:
            cv_metrics_dict = result.cv_metrics
        elif hasattr(result, 'cv_scores') and result.cv_scores:
            # Legacy NN results stored a bare list under cv_scores.
            cv_metrics_dict = {'test_score': result.cv_scores}

        details += format_cv_details(
            cv_metrics_dict,
            cv_strategy=getattr(result, 'cv_strategy', 'kfold'),
            cv_config_dict=getattr(result, 'cv_config_dict', None),
            fmt='html',
        )

        # v2.0: Gaussian Process native metrics — show in details popup
        # v2.1: Extended kernel provenance for scientific reproducibility.
        if hasattr(result, 'gp_metadata') and result.gp_metadata:
            details += "<h4>Gaussian Process Parameters:</h4><ul>"
            gp = result.gp_metadata
            if 'kernel_family' in gp:
                details += f"<li><b>Kernel Family:</b> {gp['kernel_family']}</li>"
            if 'kernel_str' in gp:
                details += f"<li><b>Kernel (fitted):</b> {gp['kernel_str']}</li>"
            if gp.get('kernel_length_scale') is not None:
                details += f"<li><b>Kernel Length Scale:</b> {gp['kernel_length_scale']:.4f}</li>"
            if gp.get('kernel_nu') is not None:
                details += f"<li><b>Kernel Nu (Matern):</b> {gp['kernel_nu']}</li>"
            if gp.get('kernel_alpha') is not None:
                details += f"<li><b>Kernel Alpha (RQ):</b> {gp['kernel_alpha']:.4f}</li>"
            if gp.get('kernel_noise_level') is not None:
                details += f"<li><b>Kernel Noise Level:</b> {gp['kernel_noise_level']:.6f}</li>"
            if 'alpha' in gp:
                details += f"<li><b>Alpha (noise):</b> {gp['alpha']}</li>"
            if 'log_marginal_likelihood_value' in gp and gp['log_marginal_likelihood_value'] is not None:
                details += f"<li><b>Log Marginal Likelihood:</b> {gp['log_marginal_likelihood_value']:.4f}</li>"
            if 'n_restarts_optimizer' in gp:
                details += f"<li><b>Optimizer Restarts:</b> {gp['n_restarts_optimizer']}</li>"
            if 'normalize_y' in gp:
                details += f"<li><b>Normalize Y:</b> {gp['normalize_y']}</li>"
            details += "</ul>"

        # Neural network specific details
        if hasattr(result, 'is_neural_network') and result.is_neural_network:
            details += "<h4>Neural Network Details:</h4>"
            if hasattr(result, 'nn_history') and result.nn_history:
                history = result.nn_history
                if 'train_loss' in history and history['train_loss']:
                    details += f"<p><b>Final Train Loss:</b> {history['train_loss'][-1]:.6f}</p>"
                if 'val_loss' in history and history['val_loss']:
                    details += f"<p><b>Final Val Loss:</b> {history['val_loss'][-1]:.6f}</p>"
                    details += f"<p><b>Best Val Loss:</b> {min(history['val_loss']):.6f}</p>"
        
        QMessageBox.information(self, "Model Details", details)
    
    def save_model_dialog(self):
        """Open dialog to save model with format options."""
        model_name = self.export_model_combo.currentText()
        
        if not model_name:
            QMessageBox.warning(self, "Warning", "No model selected")
            return
        
        # Create dialog for format selection
        from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QRadioButton, QDialogButtonBox
        
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
        
        bundle_radio = QRadioButton("Model Bundle (for standalone inference)")
        bundle_radio.setChecked(True)
        layout.addWidget(bundle_radio)
        
        bundle_info = QLabel(
            "<small>The Model Bundle includes:<br>"
            "• Complete model with preprocessing<br>"
            "• manifest.json with metadata<br>"
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
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            if bundle_radio.isChecked():
                self._export_model_bundle(model_name)
            else:
                self._export_joblib_legacy()
    
    def _export_joblib_legacy(self):
        """Export model as joblib (legacy format)."""
        model_name = self.export_model_combo.currentText()
        
        if not model_name:
            QMessageBox.warning(self, "Warning", "No model selected. Please train a model first.")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Model",
            f"{model_name}.joblib",
            "Joblib Files (*.joblib);;Pickle Files (*.pkl);;All Files (*)"
        )
        
        if file_path:
            try:
                if self._main_window and hasattr(self._main_window, 'training_tab'):
                    trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
                    trainer.save_model(model_name, file_path)
                    QMessageBox.information(self, "Success", f"Model exported to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to export model:\n{str(e)}")
    
    def _export_model_bundle(self):
        """Export model as unified bundle for standalone inference."""
        from PyQt6.QtWidgets import QFileDialog
        from pathlib import Path
        from ..core.model_saver import save_training_result, ModelSaver
        
        model_name = self.export_model_combo.currentText()
        
        if not model_name:
            QMessageBox.warning(self, "Warning", "No model selected. Please train a model first.")
            return
        
        # Verify we have access to data_manager
        if not self._main_window or not hasattr(self._main_window, 'data_tab'):
            QMessageBox.critical(self, "Error", "Cannot access data manager. Please load data first.")
            return
        
        data_manager = _safe_get_parent_attr(self, 'data_tab').get_data_manager()
        if not data_manager or not data_manager._target_column:
            QMessageBox.critical(self, "Error", "No data loaded. Please load and prepare data first.")
            return
        
        # Find the model in all sources
        model_result = None
        actual_model_name = model_name
        
        # 1. Check trained models
        if self._main_window and hasattr(self._main_window, 'training_tab'):
            trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
            if model_name in trainer.results:
                model_result = trainer.results[model_name]
        
        # 2. Check optimized models (remove " (Optimized)" suffix)
        if model_result is None and " (Optimized)" in model_name:
            opt_tab = _safe_get_parent_attr(self, 'optimization_tab')
            if hasattr(opt_tab, 'optimization_results') and opt_tab.optimization_results:
                base_name = model_name.replace(" (Optimized)", "")
                for opt_name, opt_result in opt_tab.optimization_results:
                    if opt_name == base_name:
                        model_result = opt_result
                        actual_model_name = base_name
                        break
        
        # 3. Check neural networks
        if model_result is None:
            nn_tab = _safe_get_parent_attr(self, 'nn_tab')
            if hasattr(nn_tab, 'nn_training_results') and nn_tab.nn_training_results:
                for nn_name, nn_result in nn_tab.nn_training_results:
                    if nn_name == model_name:
                        model_result = nn_result
                        break
            
            # 3b. Check NN optimized models
            if model_result is None and " (NN Optimized)" in model_name:
                if hasattr(nn_tab, 'nn_optimizer_results') and nn_tab.nn_optimizer_results:
                    base_name = model_name.replace(" (NN Optimized)", "")
                    for opt_name, opt_result in nn_tab.nn_optimizer_results:
                        if opt_name == base_name:
                            model_result = opt_result
                            actual_model_name = base_name
                            break
        
        # 4. Check ensemble models
        if model_result is None and hasattr(self._main_window, 'ensemble_tab'):
            ens_tab = _safe_get_parent_attr(self, 'ensemble_tab')
            if hasattr(ens_tab, 'get_ensemble_trainer'):
                ens_trainer = ens_tab.get_ensemble_trainer()
                if ens_trainer and hasattr(ens_trainer, 'results') and ens_trainer.results:
                    if model_name in ens_trainer.results:
                        model_result = ens_trainer.results[model_name]
        
        if model_result is None:
            QMessageBox.warning(self, "Warning", f"Model '{model_name}' not found in any source.")
            return
        
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder to Save Model Bundle",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if folder_path:
            try:
                output_path = Path(folder_path) / f"{actual_model_name}_bundle"
                
                # Use save_training_result for all model types
                saved_path = save_training_result(
                    result=model_result,
                    output_path=str(output_path),
                    data_manager=data_manager,
                    model_name=actual_model_name
                )
                
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
    
    def _export_onnx(self):
        """Export model as ONNX."""
        QMessageBox.information(self, "Info", "ONNX export not yet implemented")
    
    def _generate_script(self):
        """Generate prediction script."""
        model_name = self.export_model_combo.currentText()
        
        script = f'''"""
Prediction script generated by Materials AutoML Studio
"""

import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load('{model_name}.joblib')

# Load preprocessor (if exported)
# preprocessor = joblib.load('preprocessor.joblib')

def predict(data):
    """
    Make predictions on new data.
    
    Args:
        data: pandas DataFrame or numpy array with same features as training
        
    Returns:
        numpy array of predictions
    """
    # Preprocess if needed
    # data_processed = preprocessor.transform(data)
    
    # Predict
    predictions = model.predict(data)
    
    return predictions

if __name__ == "__main__":
    # Example usage
    # Load your data
    # df = pd.read_csv('your_data.csv')
    
    # Make predictions
    # predictions = predict(df)
    
    # Save predictions
    # np.save('predictions.npy', predictions)
    pass
'''
        
        self.script_preview.setPlainText(script)
        
        # Save to file
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Prediction Script",
            "predict.py",
            "Python Files (*.py);;All Files (*)"
        )
        
        if file_path:
            with open(file_path, 'w') as f:
                f.write(script)
            QMessageBox.information(self, "Success", f"Script saved to {file_path}")
    
    def _generate_report(self):
        """Generate PDF report."""
        QMessageBox.information(self, "Info", "PDF report generation not yet implemented")
    
    def _generate_html_report(self):
        """Generate HTML report."""
        if self._main_window and hasattr(self._main_window, 'training_tab'):
            trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
            
            # Generate simple HTML report
            html = """
            <html>
            <head>
                <title>Materials AutoML Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #0d7377; }
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    th { background-color: #0d7377; color: white; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                </style>
            </head>
            <body>
                <h1>Materials AutoML Studio - Training Report</h1>
                <h2>Model Comparison</h2>
                <table>
                    <tr>
                        <th>Model</th>
                        <th>RMSE/Accuracy</th>
                        <th>MAE</th>
                        <th>R²/F1</th>
                        <th>Training Time (s)</th>
                    </tr>
            """
            
            for name, result in trainer.results.items():
                if result.problem_type == 'regression':
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{result.metrics.get('rmse', 0):.4f}</td>
                        <td>{result.metrics.get('mae', 0):.4f}</td>
                        <td>{result.metrics.get('r2', 0):.4f}</td>
                        <td>{result.training_time:.2f}</td>
                    </tr>
                    """
                else:
                    html += f"""
                    <tr>
                        <td>{name}</td>
                        <td>{result.metrics.get('accuracy', 0):.4f}</td>
                        <td>{result.metrics.get('precision', 0):.4f}</td>
                        <td>{result.metrics.get('f1', 0):.4f}</td>
                        <td>{result.training_time:.2f}</td>
                    </tr>
                    """
            
            html += """
                </table>
            </body>
            </html>
            """
            
            # Save to file
            file_path, _ = QFileDialog.getSaveFileName(
                self,
                "Save HTML Report",
                "report.html",
                "HTML Files (*.html);;All Files (*)"
            )
            
            if file_path:
                with open(file_path, 'w') as f:
                    f.write(html)
                QMessageBox.information(self, "Success", f"Report saved to {file_path}")
    
    def _generate_learning_curves(self):
        """Generate learning curves for selected model."""
        from sklearn.model_selection import learning_curve
        
        model_name = self.export_model_combo.currentText()
        display_name = model_name  # Store original name for success message
        
        if not model_name:
            QMessageBox.warning(self, "Warning", "No model selected")
            return
        
        # Get the model and data
        if not self._main_window or not hasattr(self._main_window, 'training_tab'):
            QMessageBox.critical(self, "Error", "Cannot access training data")
            return
        
        try:
            trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
            data_manager = _safe_get_parent_attr(self, 'data_tab').get_data_manager()
            
            # Get the model - check all possible sources
            # IMPORTANT: Check special types (NN, Ensemble) FIRST before trainer.results
            # because these are also stored in trainer.results but are not sklearn-compatible
            model = None
            source_type = ""
            
            # 1. Check neural networks FIRST - Use custom implementation
            # NN names: "Neural Network" (trained), "NeuralNetwork_143022" (original format)
            if "Neural Network" in model_name or "NeuralNetwork" in model_name or "NN_" in model_name:
                self._generate_nn_learning_curves(model_name, data_manager)
                return
            
            # 2. Check ensemble models - Use custom implementation
            elif "Ensemble" in model_name:
                self._generate_ensemble_learning_curves(model_name, data_manager)
                return
            
            # 3. Check optimized models
            elif "(Optimized)" in model_name:
                opt_tab = _safe_get_parent_attr(self, 'optimization_tab')
                if hasattr(opt_tab, 'optimization_results') and opt_tab.optimization_results:
                    for opt_name, opt_result in opt_tab.optimization_results:
                        if f"{opt_name} (Optimized)" == model_name:
                            model = opt_result.best_model
                            source_type = "optimized"
                            break
                if model is None:
                    QMessageBox.warning(self, "Warning", f"Optimized model '{model_name}' not found.")
                    return
            
            # 4. Check trained models (regular sklearn models)
            elif model_name in trainer.results:
                result = trainer.results[model_name]
                # v2.0: Load from disk cache if the model was offloaded to free RAM.
                if result.model is None and getattr(result, 'is_cached', False) and result.model_cache_path:
                    from ..core.model_cache import get_model_cache
                    try:
                        cache = get_model_cache()
                        model = cache.load(result.model_cache_path, is_neural_network=result.is_neural_network)
                    except Exception as e:
                        QMessageBox.warning(self, "Warning", f"Model '{model_name}' is cached but could not be loaded:\n{str(e)}")
                        return
                else:
                    model = result.model
                source_type = "trained"

            else:
                QMessageBox.warning(self, "Warning", f"Model '{model_name}' not found in any source.")
                return

            # Verify model is valid
            if model is None:
                QMessageBox.warning(self, "Warning", f"Model '{model_name}' was found but the model object is None.")
                return
            
            # Get data
            X_train, X_test, y_train, y_test = data_manager.prepare_data()
            
            # Determine scoring metric
            if data_manager._is_classification:
                scoring = 'accuracy'
            else:
                scoring = 'r2'
            
            # Generate learning curves
            # v2.0: n_jobs=1 because the model already has internal parallelism
            # (n_jobs for RF/XGB/LGBM, thread_count for CatBoost). Using n_jobs>1
            # here would cause nested parallelism / oversubscription.
            train_sizes, train_scores, val_scores = learning_curve(
                model, X_train, y_train,
                cv=5,
                scoring=scoring,
                train_sizes=np.linspace(0.1, 1.0, 10),
                n_jobs=1,
                random_state=42
            )
            
            # Store results for export
            self._learning_curve_data = []
            
            # Update table
            self.learning_curve_table.setRowCount(len(train_sizes))
            
            for i, (size, train_score, val_score) in enumerate(zip(train_sizes, train_scores, val_scores)):
                train_mean = np.mean(train_score)
                val_mean = np.mean(val_score)
                gap = train_mean - val_mean
                
                self.learning_curve_table.setItem(i, 0, QTableWidgetItem(str(int(size))))
                self.learning_curve_table.setItem(i, 1, QTableWidgetItem(f"{train_mean:.4f}"))
                self.learning_curve_table.setItem(i, 2, QTableWidgetItem(f"{val_mean:.4f}"))
                self.learning_curve_table.setItem(i, 3, QTableWidgetItem(f"{gap:.4f}"))
                
                self._learning_curve_data.append({
                    'training_size': int(size),
                    'train_score': train_mean,
                    'validation_score': val_mean,
                    'gap': gap
                })
            
            self.learning_curve_table.resizeColumnsToContents()
            self.export_learning_btn.setEnabled(True)
            
            # v1.5.0: Also store in TrainingResult for session export
            if source_type == "trained" and model_name in trainer.results:
                trainer.compute_learning_curve(
                    model_name=model_name,
                    X=X_train,
                    y=y_train,
                    cv_folds=5,
                    train_sizes=10,
                    random_state=42,
                )
            
            QMessageBox.information(
                self,
                "Success",
                f"Learning curves generated successfully!\n\n"
                f"Model: {display_name}\n"
                f"Source: {source_type.capitalize()} model\n"
                f"Training sizes: {len(train_sizes)} points\n"
                f"Scoring metric: {scoring}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate learning curves:\n{str(e)}")
    
    def _generate_nn_learning_curves(self, model_name: str, data_manager):
        """Generate learning curves for neural networks using training history.
        
        For neural networks, we use the training history (train/val loss vs epochs)
        as a proxy for learning curves. This shows how the model learns over time.
        
        Args:
            model_name: Name of the neural network model
            data_manager: DataManager instance
        """
        try:
            # Find the neural network result
            nn_tab = _safe_get_parent_attr(self, 'nn_tab')
            nn_result = None
            source = ""
            
            # 1. Search trained NNs
            if hasattr(nn_tab, 'nn_training_results') and nn_tab.nn_training_results:
                for nn_name, result in nn_tab.nn_training_results:
                    if nn_name == model_name:
                        nn_result = result
                        source = "training"
                        break
            
            # 2. Search optimized NNs (name has " (NN Optimized)" suffix in combo)
            if nn_result is None and hasattr(nn_tab, 'nn_optimizer_results') and nn_tab.nn_optimizer_results:
                # Try exact match first
                for opt_name, result in nn_tab.nn_optimizer_results:
                    if opt_name == model_name or f"{opt_name} (NN Optimized)" == model_name:
                        nn_result = result
                        source = "optimization"
                        break
                # Try stripping suffix
                if nn_result is None and " (NN Optimized)" in model_name:
                    base_name = model_name.replace(" (NN Optimized)", "")
                    for opt_name, result in nn_tab.nn_optimizer_results:
                        if opt_name == base_name:
                            nn_result = result
                            source = "optimization"
                            break
            
            if nn_result is None:
                QMessageBox.warning(self, "Warning", f"Neural network '{model_name}' not found")
                return
            
            # Optimized NNs don't have per-epoch training history (they have trial scores)
            if source == "optimization":
                QMessageBox.information(
                    self, "Learning Curves Not Available",
                    f"Model '{model_name}' was created via hyperparameter optimization.\n\n"
                    f"Learning curves (train/validation loss per epoch) are only available "
                    f"for models trained directly through the Training tab, not for "
                    f"optimized architectures.\n\n"
                    f"To see learning curves, train the NN with the optimized parameters "
                    f"using the 'Apply Best Parameters' button and then train."
                )
                return
            
            # Get training history (only available for trained NNs)
            history = nn_result.nn_history if hasattr(nn_result, 'nn_history') else None
            
            if history is None or 'train_loss' not in history or 'val_loss' not in history:
                QMessageBox.warning(
                    self, 
                    "Warning", 
                    f"No training history available for '{model_name}'.\n"
                    f"Learning curves require train and validation loss history."
                )
                return
            
            train_losses = history.get('train_loss', [])
            val_losses = history.get('val_loss', [])
            
            if not train_losses or not val_losses:
                QMessageBox.warning(self, "Warning", "Training history is empty")
                return
            
            # Convert losses to scores (lower loss = higher score)
            # Use exponential decay to convert loss to a score-like metric
            max_loss = max(max(train_losses), max(val_losses)) * 1.1
            
            train_scores = [1.0 - (loss / max_loss) for loss in train_losses]
            val_scores = [1.0 - (loss / max_loss) for loss in val_losses]
            
            # Store results for export
            self._learning_curve_data = []
            
            # Update table - show epochs instead of training sizes
            self.learning_curve_table.setRowCount(len(train_losses))
            self.learning_curve_table.setHorizontalHeaderLabels([
                'Epoch', 'Train Loss', 'Val Loss', 'Gap'
            ])
            
            for i, (train_loss, val_loss, train_score, val_score) in enumerate(
                zip(train_losses, val_losses, train_scores, val_scores)
            ):
                gap = train_score - val_score
                
                self.learning_curve_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
                self.learning_curve_table.setItem(i, 1, QTableWidgetItem(f"{train_loss:.6f}"))
                self.learning_curve_table.setItem(i, 2, QTableWidgetItem(f"{val_loss:.6f}"))
                self.learning_curve_table.setItem(i, 3, QTableWidgetItem(f"{gap:.4f}"))
                
                self._learning_curve_data.append({
                    'epoch': i + 1,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'train_score': train_score,
                    'val_score': val_score,
                    'gap': gap
                })
            
            self.learning_curve_table.resizeColumnsToContents()
            self.export_learning_btn.setEnabled(True)
            
            # Calculate some statistics
            final_train_loss = train_losses[-1]
            final_val_loss = val_losses[-1]
            best_val_loss = min(val_losses)
            best_epoch = val_losses.index(best_val_loss) + 1
            
            QMessageBox.information(
                self,
                "Success",
                f"Learning curves generated for Neural Network!\n\n"
                f"Model: {model_name}\n"
                f"Total Epochs: {len(train_losses)}\n"
                f"Final Train Loss: {final_train_loss:.6f}\n"
                f"Final Val Loss: {final_val_loss:.6f}\n"
                f"Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})\n\n"
                f"Note: For neural networks, learning curves show loss vs epochs.\n"
                f"A large gap indicates overfitting."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate NN learning curves:\n{str(e)}")
    
    def _generate_ensemble_learning_curves(self, model_name: str, data_manager):
        """Generate learning curves for ensemble models.
        
        For ensembles, we show the cross-validation fold scores and
        individual model contributions as a proxy for learning curves.
        """
        try:
            # Find the ensemble result
            trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
            ens_result = None
            
            if model_name in trainer.results:
                result = trainer.results[model_name]
                # The model field contains the EnsembleResult
                if hasattr(result, 'model') and isinstance(result.model, EnsembleResult):
                    ens_result = result.model
                # Or the result itself might be the ensemble
                elif hasattr(result, 'model_names'):
                    ens_result = result
            
            if ens_result is None and hasattr(self._main_window, 'ensemble_tab'):
                ens_tab = _safe_get_parent_attr(self, 'ensemble_tab')
                if hasattr(ens_tab, 'get_ensemble_trainer') and ens_tab.get_ensemble_trainer():
                    ens_trainer = ens_tab.get_ensemble_trainer()
                    if hasattr(ens_trainer, 'results'):
                        for ens_name, er in ens_trainer.results.items():
                            if ens_name == model_name or model_name.endswith(ens_name.split('_')[-1] if '_' in ens_name else ens_name):
                                ens_result = er
                                break
            
            if ens_result is None:
                QMessageBox.warning(self, "Warning", f"Ensemble '{model_name}' not found")
                return
            
            # Store results for export
            self._learning_curve_data = []
            
            # Update table - show model contributions and CV fold scores
            rows = []
            
            # Row 1: Model weights
            for model_name_entry, weight in ens_result.weights.items():
                rows.append({
                    'item': model_name_entry,
                    'train': f"Weight: {weight:.4f}",
                    'val': "N/A",
                    'gap': ""
                })
            
            # Add separator row
            rows.append({
                'item': "---",
                'train': "---",
                'val': "---",
                'gap': "---"
            })
            
            # Add CV fold scores
            if hasattr(ens_result, 'cv_scores') and ens_result.cv_scores:
                for i, score in enumerate(ens_result.cv_scores):
                    rows.append({
                        'item': f"CV Fold {i+1}",
                        'train': f"{score:.4f}",
                        'val': "N/A",
                        'gap': ""
                    })
                # Add summary row
                cv_mean = np.mean(ens_result.cv_scores)
                cv_std = np.std(ens_result.cv_scores)
                rows.append({
                    'item': "CV Mean ± Std",
                    'train': f"{cv_mean:.4f} ± {cv_std:.4f}",
                    'val': "N/A",
                    'gap': ""
                })
            
            # Add ensemble metrics
            if hasattr(ens_result, 'metrics') and ens_result.metrics:
                rows.append({
                    'item': "---",
                    'train': "---",
                    'val': "---",
                    'gap': "---"
                })
                for metric_name, metric_value in ens_result.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        rows.append({
                            'item': metric_name.upper(),
                            'train': f"{metric_value:.4f}",
                            'val': "N/A",
                            'gap': ""
                        })
            
            # Update table
            self.learning_curve_table.setRowCount(len(rows))
            self.learning_curve_table.setHorizontalHeaderLabels([
                'Component', 'Value', 'Validation', 'Note'
            ])
            
            for i, row in enumerate(rows):
                self.learning_curve_table.setItem(i, 0, QTableWidgetItem(row['item']))
                self.learning_curve_table.setItem(i, 1, QTableWidgetItem(row['train']))
                self.learning_curve_table.setItem(i, 2, QTableWidgetItem(row['val']))
                self.learning_curve_table.setItem(i, 3, QTableWidgetItem(row['gap']))
            
            self.learning_curve_table.resizeColumnsToContents()
            self.export_learning_btn.setEnabled(True)
            
            # Build data for export
            self._learning_curve_data = [
                {
                    'component': r['item'],
                    'value': r['train'],
                }
                for r in rows if r['item'] != "---"
            ]
            
            cv_info = ""
            if hasattr(ens_result, 'cv_scores') and ens_result.cv_scores:
                cv_mean = np.mean(ens_result.cv_scores)
                cv_info = f"CV Score: {cv_mean:.4f}"
            
            QMessageBox.information(
                self,
                "Success",
                f"Learning curves generated for Ensemble!\n\n"
                f"Model: {model_name}\n"
                f"Type: {ens_result.ensemble_type}\n"
                f"Models: {len(ens_result.model_names)}\n"
                f"{cv_info}\n\n"
                f"Note: For ensembles, the table shows model weights "
                f"and CV fold scores instead of traditional learning curves."
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to generate ensemble learning curves:\n{str(e)}")
    
    def _export_learning_curves(self):
        """Export learning curves to CSV."""
        if not hasattr(self, '_learning_curve_data') or not self._learning_curve_data:
            QMessageBox.warning(self, "Warning", "No learning curve data to export")
            return
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Learning Curves",
            "learning_curves.csv",
            "CSV Files (*.csv);;All Files (*)"
        )
        
        if file_path:
            try:
                df = pd.DataFrame(self._learning_curve_data)
                df.to_csv(file_path, index=False)
                QMessageBox.information(self, "Success", f"Learning curves saved to {file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save learning curves:\n{str(e)}")

    def _export_session_report(self, fmt: str):
        """Export the complete session report to a file (v1.5.0)."""
        exporter = SessionExporter()

        # ------------------------------------------------------------------
        # Gather session info (dataset, preprocessing, features)
        # ------------------------------------------------------------------
        session_info = {}
        preprocessing_config = {}
        feature_names = []
        try:
            data_tab = _safe_get_parent_attr(self, 'data_tab')
            if data_tab and data_tab.data_manager:
                dm = data_tab.data_manager
                if dm.data is not None:
                    session_info['n_samples'] = len(dm.data)
                    session_info['n_features'] = len(dm.data.columns)
                session_info['target_column'] = dm.target_column or 'Not set'
                session_info['problem_type'] = 'classification' if dm.is_classification else 'regression'
                feature_names = dm.get_feature_names() if hasattr(dm, 'get_feature_names') else []

                # Preprocessing config
                if hasattr(data_tab, 'impute_combo'):
                    preprocessing_config['imputation'] = data_tab.impute_combo.currentText()
                if hasattr(data_tab, 'scale_combo'):
                    preprocessing_config['scaling'] = data_tab.scale_combo.currentText()
                if hasattr(data_tab, 'encode_combo'):
                    preprocessing_config['encoding'] = data_tab.encode_combo.currentText()
        except Exception:
            pass

        exporter.session_info = session_info
        exporter.set_preprocessing_config(preprocessing_config)
        exporter.set_feature_names(feature_names)

        # Hardware info
        try:
            from app.utils.hardware_detector import get_hardware_detector
            hw = get_hardware_detector()
            exporter.set_hardware_info(hw.format_info())
        except Exception as e:
            logger.debug(f"Hardware info collection failed: {e}")

        # ------------------------------------------------------------------
        # Collect crystal link info (v2.0)
        # ------------------------------------------------------------------
        try:
            parent = self._main_window
            data_tab = getattr(parent, 'data_tab', None) if parent else None
            dm = data_tab.get_data_manager() if data_tab else None
            if dm and dm.crystal_features_active:
                extractor = getattr(dm, '_crystal_extractor', None)
                if extractor and extractor.link_report:
                    report = extractor.link_report
                    exporter.add_crystal_link_info(
                        id_column=report.id_column,
                        cif_folder=report.cif_folder,
                        n_matched=report.n_matched,
                        n_total=report.n_rows_total,
                        crystal_features=report.crystal_features_added,
                        composition_replaced=report.composition_features_replaced,
                    )
        except Exception as e:
            logger.debug(f"Crystal link info collection failed: {e}")

        # ------------------------------------------------------------------
        # Collect training results (sklearn models + NNs from trainer.results)
        # v2.0: Separate NN results (is_neural_network=True) from sklearn
        # results. NNs go to add_nn_training_result() with full architecture
        # and history. Sklearn models go to add_training_results().
        # This avoids duplicating NNs that also appear in nn_tab.nn_training_results.
        # ------------------------------------------------------------------
        # v2.0 FIX: Per-result error handling so one bad result doesn't
        # destroy the entire export. The previous code had a single try/except
        # around the whole loop — a TypeError (e.g., passing history_stats to
        # add_nn_training_result which doesn't accept it) would silently drop
        # ALL training results.
        try:
            trainer = _safe_get_parent_attr(self, 'training_tab.get_trainer()')
        except Exception as e:
            logger.warning(f"Could not get trainer: {e}")
            trainer = None

        if trainer and hasattr(trainer, 'results') and trainer.results:
            sklearn_results = []
            for result in trainer.results.values():
                try:
                    if getattr(result, 'is_neural_network', False):
                        # v2.0: Read ALL NN data from immutable TrainingResult fields.
                        # NO references to live nn_builder — completely robust.
                        history = getattr(result, 'nn_history', {}) or {}
                        if not history:
                            continue

                        cfg = dict(getattr(result, 'training_config', {}) or {})
                        cfg['random_seed'] = getattr(result, 'random_state', 42)

                        # Architecture from immutable fields (captured at training time)
                        arch_desc = getattr(result, 'nn_architecture_desc', None) or 'Unknown'
                        arch_detail = getattr(result, 'nn_architecture_detail', None) or {}

                        # NOTE: add_nn_training_result() computes history_stats
                        # internally — do NOT pass it as a parameter.
                        # v2.0: Pass cv_metrics for per-fold breakdown in export.
                        # v2.0: Pass optimization_time_sec for NN that was optimized.
                        cv_metrics = getattr(result, 'cv_metrics', None)
                        exporter.add_nn_training_result(
                            model_name=result.model_name,
                            architecture_desc=arch_desc,
                            architecture_detail=arch_detail,
                            history=history,
                            training_config=cfg,
                            metrics=getattr(result, 'metrics', None),
                            training_time_sec=getattr(result, 'training_time', 0.0),
                            cv_metrics=cv_metrics,
                            optimization_time_sec=0.0,  # Will be populated below if NN was optimized
                        )
                    else:
                        # Sklearn model
                        sklearn_results.append(result)
                except Exception as e:
                    # Per-result error: log the specific model that failed
                    # but continue processing the rest.
                    model_name = getattr(result, 'model_name', 'Unknown')
                    logger.error(
                        f"Export collection FAILED for model '{model_name}': {e}",
                        exc_info=True
                    )
            # Add sklearn models (if any were successfully collected)
            if sklearn_results:
                try:
                    exporter.add_training_results(sklearn_results)
                except Exception as e:
                    logger.error(f"add_training_results failed: {e}", exc_info=True)

        # v2.0: Post-process NN entries to add optimization_time from
        # nn_optimizer_results. When a NN is optimized and then retrained,
        # the optimization time is stored in nn_tab.nn_optimizer_results
        # but NOT in the TrainingResult. We match by name and update.
        try:
            nn_tab = _safe_get_parent_attr(self, 'nn_tab')
            if nn_tab and hasattr(nn_tab, 'nn_optimizer_results') and nn_tab.nn_optimizer_results:
                # Build lookup: algorithm name → optimization time
                opt_times = {}
                for opt_name, opt_result in nn_tab.nn_optimizer_results:
                    if hasattr(opt_result, 'optimization_time'):
                        opt_times[opt_name] = opt_result.optimization_time

                # Update NN entries in exporter
                if opt_times:
                    for entry in exporter._entries:
                        if entry.get('section') == 'Neural Network':
                            model_name = entry.get('model_name', '')
                            # Try exact match first
                            if model_name in opt_times:
                                entry['optimization_time_sec'] = round(float(opt_times[model_name]), 3)
                            else:
                                # Try base name match (e.g., "Neural Network_1" → "Neural Network")
                                for opt_name, opt_time in opt_times.items():
                                    if model_name.startswith(opt_name):
                                        entry['optimization_time_sec'] = round(float(opt_time), 3)
                                        break
        except Exception as e:
            logger.debug(f"NN optimization time post-processing failed: {e}")

        # ------------------------------------------------------------------
        # Collect optimization results (with ALL trials + full config)
        # ------------------------------------------------------------------
        try:
            opt_tab = _safe_get_parent_attr(self, 'optimization_tab')
            if opt_tab and hasattr(opt_tab, 'optimization_results') and opt_tab.optimization_results:
                for name, result in opt_tab.optimization_results:
                    trials_df = None
                    if hasattr(result, 'all_trials') and result.all_trials is not None:
                        trials_df = result.all_trials
                    elif hasattr(opt_tab, 'all_trial_results') and opt_tab.all_trial_results:
                        trials_df = opt_tab.all_trial_results

                    # v2.0 FIX: Read the ACTUAL optimization configuration from the
                    # OptimizationResult (stored at optimization time) instead of
                    # reading from the current GUI values. This ensures each
                    # optimization shows its REAL config (cv_folds, random_state,
                    # etc.) even when the user ran multiple optimizations with
                    # different settings in the same session.
                    opt_cfg = dict(getattr(result, 'optimization_config', {}) or {})
                    # Fallback to algorithm name if not stored
                    opt_cfg.setdefault('model_type', name)
                    # Supplement with GUI values only for fields not in stored config
                    if 'n_trials' not in opt_cfg and hasattr(opt_tab, 'trials_spin'):
                        opt_cfg['n_trials'] = opt_tab.trials_spin.value()
                    if 'cv_folds' not in opt_cfg and hasattr(opt_tab, 'cv_folds_spin'):
                        opt_cfg['cv_folds'] = opt_tab.cv_folds_spin.value()
                    if 'random_state' not in opt_cfg and hasattr(opt_tab, 'random_seed_spin'):
                        opt_cfg['random_state'] = opt_tab.random_seed_spin.value()
                    if 'timeout_sec' not in opt_cfg and hasattr(opt_tab, 'timeout_spin'):
                        opt_cfg['timeout_sec'] = opt_tab.timeout_spin.value()

                    # v2.0 FIX: getattr(..., lambda: None)() fails when the attr
                    # exists but is a QSpinBox (not callable). Use direct access.
                    timeout_val = None
                    if hasattr(opt_tab, 'timeout_spin') and opt_tab.timeout_spin is not None:
                        try:
                            timeout_val = opt_tab.timeout_spin.value()
                        except Exception:
                            pass

                    exporter.add_optimization_result(
                        result, algorithm=name,
                        n_trials=getattr(result, 'n_trials', 0),
                        timeout=timeout_val,
                        all_trials_df=trials_df,
                        optimization_config=opt_cfg,
                    )
        except Exception as e:
            logger.warning(f"Optimization results collection failed: {e}")

        # NOTE: NN results are collected in the trainer.results loop above
        # (v2.0: separated by is_neural_network flag). Do NOT duplicate here.

        # ------------------------------------------------------------------
        # v2.0: Collect NN optimization results
        # ------------------------------------------------------------------
        # NN optimizations are stored in nn_builder_tab.nn_optimizer_results,
        # NOT in optimization_tab.optimization_results. They must be collected
        # separately so they can be linked to their trained NN counterparts
        # in _link_optimization_entries().
        # The algorithm key 'Neural Network' normalizes to 'neural_network' in
        # _opt_entries, which matches the base 'nn' from model names like
        # 'NN_Optimized' (via the nn → neural_network mapping in
        # _link_optimization_entries).
        try:
            nn_tab = _safe_get_parent_attr(self, 'nn_tab')
            if nn_tab and hasattr(nn_tab, 'nn_optimizer_results') and nn_tab.nn_optimizer_results:
                for opt_name, result in nn_tab.nn_optimizer_results:
                    opt_cfg = dict(getattr(result, 'optimization_config', {}) or {})
                    opt_cfg.setdefault('model_type', 'Neural Network')
                    exporter.add_optimization_result(
                        result, algorithm='Neural Network',
                        n_trials=getattr(result, 'n_trials', 0),
                        timeout=None,
                        all_trials_df=getattr(result, 'all_trials', None),
                        optimization_config=opt_cfg,
                    )
        except Exception as e:
            logger.warning(f"NN optimization results collection failed: {e}")

        # ------------------------------------------------------------------
        # Collect ensemble results (with weight optimization config)
        # ------------------------------------------------------------------
        try:
            ens_tab = _safe_get_parent_attr(self, 'ensemble_tab')
            if ens_tab and hasattr(ens_tab, 'get_ensemble_trainer'):
                ens_trainer = ens_tab.get_ensemble_trainer()
                if ens_trainer and hasattr(ens_trainer, 'results') and ens_trainer.results:
                    for name, result in ens_trainer.results.items():
                        # v1.5.0: Extract weight optimization configuration
                        wopt_cfg = {}
                        if hasattr(ens_tab, 'opt_trials_spin'):
                            wopt_cfg['n_trials'] = ens_tab.opt_trials_spin.value()
                        if hasattr(ens_tab, 'opt_cv_folds_spin'):
                            wopt_cfg['cv_folds'] = ens_tab.opt_cv_folds_spin.value()
                        if hasattr(ens_tab, 'opt_random_seed_spin'):
                            wopt_cfg['random_state'] = ens_tab.opt_random_seed_spin.value()
                        if hasattr(ens_tab, 'ensemble_type_combo'):
                            wopt_cfg['ensemble_type'] = ens_tab.ensemble_type_combo.currentText()
                        if hasattr(ens_tab, 'selected_models_list') and ens_tab.selected_models_list:
                            wopt_cfg['base_models'] = [
                                ens_tab.selected_models_list.item(i).text()
                                for i in range(ens_tab.selected_models_list.count())
                            ]

                        exporter.add_ensemble_result(
                            result,
                            weight_optimization_config=wopt_cfg,
                        )
        except Exception as e:
            logger.warning(f"Ensemble results collection failed: {e}")

        # ------------------------------------------------------------------
        # Export
        # ------------------------------------------------------------------
        if not exporter._entries:
            QMessageBox.warning(
                self, "No Results",
                "No results available to export.\n\n"
                "Train at least one model before exporting the session report."
            )
            return

        ext = fmt if fmt != "md" else "md"
        default_name = f"autoaim_session_report.{ext}"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            f"Export Session Report (.{fmt})",
            default_name,
            f"{fmt.upper()} Files (*.{ext});;All Files (*)"
        )

        if not file_path:
            return

        try:
            written = exporter.export(file_path, fmt=fmt)
            QMessageBox.information(
                self, "Success",
                f"Session report exported to:\n{written}\n\n"
                f"Entries: {len(exporter._entries)} result(s)\n"
                f"Features: {len(feature_names)}\n"
                f"Preprocessing: {preprocessing_config}"
            )
        except Exception as e:
            QMessageBox.critical(
                self, "Error",
                f"Failed to export session report:\n{str(e)}"
            )
