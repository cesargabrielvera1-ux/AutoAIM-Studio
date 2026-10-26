"""Model explainability tab with SHAP and PDP."""

import numpy as np
import pandas as pd
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QMessageBox, QSplitter, QTextEdit, QListWidget, QListWidgetItem,
    QSpinBox, QTabWidget, QCheckBox, QProgressBar,
    QFileDialog
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from ..core.explainer import ModelExplainer
from ..core.domain_applicability import DomainApplicabilityAnalyzer


class ExplainabilityThread(QThread):
    """Thread for running explainability analysis."""
    progress = pyqtSignal(str, int)  # message, percent
    log = pyqtSignal(str)  # log message
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)
    
    def __init__(self, explainer, domain_analyzer, model, is_nn, X_train, X_test, y_train, y_test, 
                 feature_names, perm_check, shap_check, pdp_check, domain_check, n_features):
        super().__init__()
        self.explainer = explainer
        self.domain_analyzer = domain_analyzer
        self.model = model
        self.is_nn = is_nn
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.perm_check = perm_check
        self.shap_check = shap_check
        self.pdp_check = pdp_check
        self.domain_check = domain_check
        self.n_features = n_features
    
    def run(self):
        try:
            results = {
                'importance_df': None,
                'shap_result': None,
                'pdp_result': None,
                'domain_summary': None,
                'is_nn': self.is_nn
            }
            
            # Feature importance summary
            if self.perm_check or self.shap_check:
                self.progress.emit("Computing feature importance...", 10)
                self.log.emit("Starting feature importance calculation...")
                
                try:
                    importance_df = self.explainer.get_feature_importance_summary(
                        self.model, self.X_test, self.y_test, self.feature_names
                    )
                    results['importance_df'] = importance_df
                    
                    available_cols = [c for c in importance_df.columns if c != 'feature']
                    self.log.emit(f"Feature importance computed. Columns: {available_cols}")
                    self.log.emit(f"DataFrame shape: {importance_df.shape}")
                    
                    # v2.0: Log actual values for diagnostics — helps identify
                    # when a model type produces zero/NaN values.
                    for col in available_cols:
                        if col.endswith('_norm') or col == 'average_rank':
                            continue  # Skip derived columns
                        vals = importance_df[col]
                        total = vals.abs().sum()
                        all_zero = (total == 0)
                        all_nan = vals.isna().all()
                        self.log.emit(
                            f"  {col}: sum={total:.6f}, "
                            f"all_zero={all_zero}, all_nan={all_nan}"
                        )
                    
                    if not available_cols:
                        self.log.emit("WARNING: No importance columns were computed!")
                    
                except Exception as e:
                    self.log.emit(f"ERROR in feature importance: {str(e)}")
                    import traceback
                    self.log.emit(traceback.format_exc())
            
            # SHAP values
            if self.shap_check:
                self.progress.emit("Computing SHAP values...", 40)
                self.log.emit("Starting SHAP calculation...")
                
                try:
                    shap_result = self.explainer.compute_shap_values(
                        self.model, self.X_test[:100],
                        feature_names=self.feature_names
                    )
                    
                    if shap_result:
                        results['shap_result'] = shap_result
                        global_imp = shap_result.get('global_importance', {})
                        self.log.emit(f"SHAP computed. Top features: {list(global_imp.keys())[:5]}")
                    else:
                        self.log.emit("WARNING: SHAP returned empty result")
                        
                except Exception as e:
                    self.log.emit(f"ERROR in SHAP: {str(e)}")
                    import traceback
                    self.log.emit(traceback.format_exc())
            
            # Partial dependence (NOT available for Neural Networks)
            if self.pdp_check:
                if self.is_nn:
                    self.progress.emit("Skipping PDP for Neural Network...", 70)
                    self.log.emit("PDP skipped: Not available for Neural Networks (sklearn limitation)")
                    results['pdp_result'] = None
                    results['pdp_message'] = "Partial Dependence Plots are not available for Neural Networks. This analysis requires sklearn's partial_dependence function, which only supports sklearn-compatible models."
                else:
                    self.progress.emit("Computing partial dependence...", 70)
                    self.log.emit("Starting partial dependence calculation...")
                    
                    try:
                        # Get importance df if available
                        if results['importance_df'] is not None:
                            top_features = results['importance_df'].head(min(self.n_features, 6))['feature'].tolist()
                        else:
                            top_features = self.feature_names[:min(self.n_features, 6)]
                        
                        pdp_result = self.explainer.compute_partial_dependence(
                            self.model, self.X_test,
                            features=top_features,
                            feature_names=self.feature_names
                        )
                        
                        if pdp_result:
                            results['pdp_result'] = pdp_result
                            self.log.emit(f"PDP computed for: {list(pdp_result.keys())}")
                        else:
                            self.log.emit("WARNING: PDP returned empty result")
                            
                    except Exception as e:
                        self.log.emit(f"ERROR in PDP: {str(e)}")
                        import traceback
                        self.log.emit(traceback.format_exc())
            
            # Domain Applicability (available for all models)
            if self.domain_check:
                self.progress.emit("Computing domain applicability...", 85)
                self.log.emit("Starting domain applicability calculation...")
                
                try:
                    self.domain_analyzer.fit(self.X_train, self.y_train)
                    domain_summary = self.domain_analyzer.get_applicability_domain_summary()
                    results['domain_summary'] = domain_summary
                    self.log.emit(f"Domain applicability computed. Training samples: {domain_summary.get('n_training_samples', 0)}")
                except Exception as e:
                    self.log.emit(f"ERROR in domain applicability: {str(e)}")
                    import traceback
                    self.log.emit(traceback.format_exc())
            
            self.progress.emit("Complete!", 100)
            self.log.emit("Analysis complete!")
            self.finished.emit(results)
            
        except Exception as e:
            self.log.emit(f"FATAL ERROR: {str(e)}")
            import traceback
            self.log.emit(traceback.format_exc())
            self.error.emit(str(e))


class ExplainabilityTab(QWidget):
    """Model explainability tab."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self._main_window = parent  # v2.0 FIX: self._main_window overwrites PyQt parent()
        self.explainer = ModelExplainer()
        self.domain_analyzer = DomainApplicabilityAnalyzer()
        self.analysis_thread = None
        
        self._init_ui()
    
    def _init_ui(self):
        """Initialize UI."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        
        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel: Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        # Model selection
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout(model_group)
        
        model_layout.addWidget(QLabel("Select Model:"))
        self.model_combo = QComboBox()
        model_layout.addWidget(self.model_combo)
        
        self.refresh_btn = QPushButton("Refresh Model List")
        self.refresh_btn.setToolTip("Update the list with trained models available for analysis")
        self.refresh_btn.clicked.connect(self._refresh_models)
        model_layout.addWidget(self.refresh_btn)
        
        left_layout.addWidget(model_group)
        
        # Analysis options
        analysis_group = QGroupBox("Analysis Options")
        analysis_layout = QVBoxLayout(analysis_group)
        
        self.shap_check = QCheckBox("Compute SHAP Values")
        self.shap_check.setToolTip(
            "SHAP explains how each feature pushes the prediction higher or lower "
            "from the baseline. Most accurate but slow for tree models."
        )
        self.shap_check.setChecked(True)
        analysis_layout.addWidget(self.shap_check)
        
        self.perm_check = QCheckBox("Compute Permutation Importance")
        self.perm_check.setToolTip(
            "Measures feature importance by shuffling each feature and observing "
            "the drop in model performance. Model-agnostic and reliable."
        )
        self.perm_check.setChecked(True)
        analysis_layout.addWidget(self.perm_check)
        
        self.pdp_check = QCheckBox("Compute Partial Dependence Plots")
        self.pdp_check.setToolTip(
            "Shows how changing one feature affects the model's prediction "
            "while keeping all other features fixed. Useful for understanding trends."
        )
        self.pdp_check.setChecked(True)
        analysis_layout.addWidget(self.pdp_check)
        
        self.domain_check = QCheckBox("Analyze Domain Applicability")
        self.domain_check.setToolTip(
            "Estimates how confident the model is when predicting on new data. "
            "Flags predictions outside the training domain (low reliability)."
        )
        self.domain_check.setChecked(True)
        analysis_layout.addWidget(self.domain_check)
        
        # Number of features to display
        analysis_layout.addWidget(QLabel("Number of top features to display:"))
        self.n_features_spin = QSpinBox()
        self.n_features_spin.setRange(1, 100)
        self.n_features_spin.setValue(10)
        self.n_features_spin.setSingleStep(1)
        self.n_features_spin.setToolTip("Number of most important features to display in plots and tables")
        analysis_layout.addWidget(self.n_features_spin)
        
        left_layout.addWidget(analysis_group)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)
        
        # Run analysis button
        self.analyze_btn = QPushButton("Run Explainability Analysis")
        self.analyze_btn.setStyleSheet("""
            QPushButton {
                background-color: #0d7377;
                color: white;
                font-weight: bold;
                padding: 15px;
                font-size: 12pt;
            }
        """)
        self.analyze_btn.clicked.connect(self._run_analysis)
        left_layout.addWidget(self.analyze_btn)
        
        # Export results button
        self.export_btn = QPushButton("Export Results as Text")
        self.export_btn.setToolTip("Save all explainability results to a text file")
        self.export_btn.setStyleSheet("""
            QPushButton {
                background-color: #455a64;
                color: white;
                font-weight: bold;
                padding: 10px;
                font-size: 11pt;
            }
        """)
        self.export_btn.clicked.connect(self._export_results)
        left_layout.addWidget(self.export_btn)
        
        # Log area
        log_group = QGroupBox("Analysis Log")
        log_layout = QVBoxLayout(log_group)
        
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)
        
        left_layout.addWidget(log_group)
        
        left_layout.addStretch()
        
        splitter.addWidget(left_panel)
        
        # Right panel: Results
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        
        # Results tabs
        results_tabs = QTabWidget()
        
        # Feature importance
        importance_widget = QWidget()
        importance_layout = QVBoxLayout(importance_widget)
        
        self.importance_table = QTableWidget()
        self.importance_table.setColumnCount(5)
        self.importance_table.setHorizontalHeaderLabels([
            'Feature', 'Model Importance', 'Permutation', 'SHAP', 'Average Rank'
        ])
        importance_layout.addWidget(self.importance_table)
        
        results_tabs.addTab(importance_widget, "Feature Importance")
        
        # SHAP values
        shap_widget = QWidget()
        shap_layout = QVBoxLayout(shap_widget)
        
        self.shap_text = QTextEdit()
        self.shap_text.setReadOnly(True)
        shap_layout.addWidget(self.shap_text)
        
        results_tabs.addTab(shap_widget, "SHAP Values")
        
        # PDP
        pdp_widget = QWidget()
        pdp_layout = QVBoxLayout(pdp_widget)
        
        self.pdp_text = QTextEdit()
        self.pdp_text.setReadOnly(True)
        pdp_layout.addWidget(self.pdp_text)
        
        results_tabs.addTab(pdp_widget, "Partial Dependence")
        
        # Domain applicability
        domain_widget = QWidget()
        domain_layout = QVBoxLayout(domain_widget)
        
        self.domain_text = QTextEdit()
        self.domain_text.setReadOnly(True)
        domain_layout.addWidget(self.domain_text)
        
        results_tabs.addTab(domain_widget, "Domain Applicability")
        
        # Prediction explanation
        explain_widget = QWidget()
        explain_layout = QVBoxLayout(explain_widget)
        
        explain_layout.addWidget(QLabel("Select Instance to Explain:"))
        self.instance_spin = QSpinBox()
        self.instance_spin.setValue(0)
        explain_layout.addWidget(self.instance_spin)
        
        self.explain_btn = QPushButton("Explain Prediction")
        self.explain_btn.clicked.connect(self._explain_prediction)
        explain_layout.addWidget(self.explain_btn)
        
        self.explanation_text = QTextEdit()
        self.explanation_text.setReadOnly(True)
        explain_layout.addWidget(self.explanation_text)
        
        results_tabs.addTab(explain_widget, "Prediction Explanation")
        
        right_layout.addWidget(results_tabs)
        
        splitter.addWidget(right_panel)
        splitter.setSizes([400, 800])
        
        layout.addWidget(splitter)
        
        # Initial refresh
        self._refresh_models()
    
    def _log(self, message):
        """Add message to log."""
        self.log_text.append(message)
        # Also print to console
        print(f"[Explainability] {message}")
    
    def _refresh_models(self):
        """Refresh model list from all sources."""
        self.model_combo.clear()
        
        if self._main_window is None:
            return
        
        models_added = []
        
        # 1. Add trained models (sklearn only — NNs are added separately from nn_tab)
        # v2.0 FIX: Skip is_neural_network=True results here to prevent duplication.
        # NNs are stored in BOTH trainer.results (by train_neural_network) AND
        # nn_tab.nn_training_results (by nn_builder_tab). Adding from both sources
        # causes each NN to appear twice in the combo.
        if hasattr(self._main_window, 'training_tab'):
            try:
                trainer = self._main_window.training_tab.get_trainer()
                if trainer and hasattr(trainer, 'results') and trainer.results:
                    for name, result in trainer.results.items():
                        # Skip NNs — they are added from nn_tab in section 3
                        if getattr(result, 'is_neural_network', False):
                            continue
                        display_name = f"{name} (Trained)"
                        self.model_combo.addItem(display_name, ('trained', name, result))
                        models_added.append(name)
            except Exception as e:
                print(f"Error loading trained models: {e}")
        
        # 2. Add optimized models
        if hasattr(self._main_window, 'optimization_tab'):
            try:
                opt_tab = self._main_window.optimization_tab
                if hasattr(opt_tab, 'optimization_results') and opt_tab.optimization_results:
                    for model_name, opt_result in opt_tab.optimization_results:
                        display_name = f"{model_name} (Optimized)"
                        self.model_combo.addItem(display_name, ('optimized', model_name, opt_result))
                        models_added.append(model_name)
            except Exception as e:
                print(f"Error loading optimized models: {e}")
        
        # 3. Add neural networks (trained)
        if hasattr(self._main_window, 'nn_tab'):
            try:
                nn_tab = self._main_window.nn_tab
                if hasattr(nn_tab, 'nn_training_results') and nn_tab.nn_training_results:
                    for nn_name, nn_result in nn_tab.nn_training_results:
                        display_name = f"{nn_name} (Neural Network)"
                        self.model_combo.addItem(display_name, ('neural_network', nn_name, nn_result))
                        models_added.append(nn_name)
            except Exception as e:
                print(f"Error loading neural networks: {e}")
        
        # 4. Add neural networks (optimized)
        if hasattr(self._main_window, 'nn_tab'):
            try:
                nn_tab = self._main_window.nn_tab
                if hasattr(nn_tab, 'nn_optimizer_results') and nn_tab.nn_optimizer_results:
                    for opt_name, opt_result in nn_tab.nn_optimizer_results:
                        display_name = f"{opt_name} (NN Optimized)"
                        self.model_combo.addItem(display_name, ('nn_optimized', opt_name, opt_result))
                        models_added.append(opt_name)
            except Exception as e:
                print(f"Error loading optimized neural networks: {e}")
    
    def _get_selected_model(self):
        """Get the currently selected model."""
        index = self.model_combo.currentIndex()
        if index < 0:
            return None, None, None
        return self.model_combo.itemData(index)
    
    def _on_analysis_progress(self, message, percent):
        """Handle analysis progress."""
        self.progress_bar.setValue(percent)
        self._log(f"[{percent}%] {message}")
    
    def _on_analysis_log(self, message):
        """Handle log message."""
        self._log(message)
    
    def _on_analysis_finished(self, results):
        """Handle analysis finished."""
        # v2.0 GUARD: Prevent duplicate processing if thread emits multiple times
        if hasattr(self, '_analysis_completed') and self._analysis_completed:
            return
        self._analysis_completed = True
        
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        # v2.0: Clean up thread reference to allow re-runs
        self.analysis_thread = None
        
        importance_df = results.get('importance_df')
        shap_result = results.get('shap_result')
        pdp_result = results.get('pdp_result')
        is_nn = results.get('is_nn', False)
        
        # Update Feature Importance table
        if importance_df is not None and not importance_df.empty:
            # v2.0: Verify the DataFrame has actual values before displaying
            value_cols = [c for c in importance_df.columns 
                         if c not in ('feature', 'average_rank') 
                         and not c.endswith('_norm')]
            has_any_values = any(
                importance_df[c].abs().sum() > 0 
                for c in value_cols if c in importance_df.columns
            )
            if has_any_values:
                self._update_importance_table(importance_df)
                self._log(f"Feature Importance table updated with {len(importance_df)} rows")
            else:
                self._log("WARNING: Feature importance computed but all values are zero!")
                self._log(f"  Columns checked: {value_cols}")
                self.importance_table.setRowCount(0)
        else:
            self._log("WARNING: Feature importance DataFrame is empty!")
            self.importance_table.setRowCount(0)
        
        # Update SHAP tab
        if shap_result:
            n_features = self.n_features_spin.value()
            shap_text = "<h3>SHAP Values Summary</h3>"
            shap_text += f"<p>Global feature importance (top {n_features} features by mean absolute SHAP value):</p><ul>"
            
            global_imp = shap_result.get('global_importance', {})
            for feat, imp in sorted(global_imp.items(), key=lambda x: x[1], reverse=True)[:n_features]:
                shap_text += f"<li>{feat}: {imp:.4f}</li>"
            
            shap_text += "</ul>"
            self.shap_text.setHtml(shap_text)
            self._log(f"SHAP tab updated with {len(global_imp)} features")
        else:
            self.shap_text.setHtml("<p>SHAP values not available for this model.</p>")
            self._log("SHAP result is None")
        
        # Update PDP tab
        if pdp_result:
            pdp_text = "<h3>Partial Dependence Plots</h3>"
            pdp_text += f"<p>Top {len(pdp_result)} analyzed features:</p><ul>"
            
            for feat in pdp_result.keys():
                pdp_text += f"<li>{feat}</li>"
            
            pdp_text += "</ul>"
            self.pdp_text.setHtml(pdp_text)
            self._log(f"PDP tab updated with {len(pdp_result)} features")
        elif is_nn:
            # Show NN-specific message
            pdp_message = results.get('pdp_message', 
                "Partial Dependence Plots are not available for Neural Networks.<br><br>"
                "This analysis requires sklearn's partial_dependence function, "
                "which only supports sklearn-compatible models.")
            self.pdp_text.setHtml(f"<p style='color: orange;'>{pdp_message}</p>")
            self._log("PDP not available for Neural Networks")
        else:
            self.pdp_text.setHtml("<p>Partial dependence plots not available.</p>")
            self._log("PDP result is None")
        
        # Update Domain Applicability tab
        domain_summary = results.get('domain_summary')
        if domain_summary:
            domain_text = f"""
            <h3>Domain Applicability Analysis</h3>
            <p><b>Training Samples:</b> {domain_summary.get('n_training_samples', 0)}</p>
            <p><b>Original Features:</b> {domain_summary.get('n_features_original', 0)}</p>
            <p><b>PCA Components:</b> {domain_summary.get('n_features_pca', 0)}</p>
            <p><b>Explained Variance:</b> {domain_summary.get('explained_variance_ratio', 0):.2%}</p>
            <h4>Thresholds:</h4>
            <ul>
                <li>Leverage: {domain_summary.get('thresholds', {}).get('leverage', 0):.4f}</li>
                <li>Mahalanobis: {domain_summary.get('thresholds', {}).get('mahalanobis', 0):.4f}</li>
                <li>k-NN Distance: {domain_summary.get('thresholds', {}).get('knn_distance', 0):.4f}</li>
            </ul>
            """
            self.domain_text.setHtml(domain_text)
            self._log("Domain Applicability tab updated")
        else:
            self.domain_text.setHtml("<p>Domain applicability analysis not available.</p>")
            self._log("Domain summary is None")
        
        QMessageBox.information(self, "Success", "Explainability analysis complete!")
    
    def _on_analysis_error(self, error_msg):
        """Handle analysis error."""
        # v2.0 GUARD: Prevent duplicate error dialogs
        if hasattr(self, '_analysis_completed') and self._analysis_completed:
            return
        self._analysis_completed = True
        
        self.progress_bar.setVisible(False)
        self.analyze_btn.setEnabled(True)
        # v2.0: Clean up thread reference
        self.analysis_thread = None
        self._log(f"ERROR: {error_msg}")
        QMessageBox.critical(self, "Error", f"Analysis failed:\n{error_msg}")
    
    def _run_analysis(self):
        """Run explainability analysis."""
        # v2.0 GUARD: Prevent multiple simultaneous analyses
        if self.analysis_thread is not None and self.analysis_thread.isRunning():
            QMessageBox.warning(
                self, "Analysis Running",
                "An explainability analysis is already in progress.\n"
                "Please wait for it to complete."
            )
            return
        
        # v2.0: Reset completion flag for new run
        self._analysis_completed = False
        
        # Get selected model
        model_data = self._get_selected_model()
        if model_data is None:
            QMessageBox.warning(self, "Warning", "Please select a model")
            return
        
        model_type, model_name, result = model_data
        
        # Get data
        if self._main_window is None:
            QMessageBox.warning(self, "Warning", "No data available")
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
        if data_manager is None:
            QMessageBox.warning(self, "Warning", "No data loaded")
            return
        
        try:
            # Clear log
            self.log_text.clear()
            self._log(f"Starting analysis for: {model_name} ({model_type})")
            
            # Get data
            X_train, X_test, y_train, y_test = data_manager.prepare_data()
            feature_names = data_manager.get_feature_names()
            
            # FIX: Verificar que feature_names coincida con X_test
            if len(feature_names) != X_test.shape[1]:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            self._log(f"Data shape: X_test={X_test.shape}, features={len(feature_names)}")
            
            # Get the actual model object
            if model_type == 'trained':
                model = result.model
                is_nn = getattr(result, 'is_neural_network', False)
                # Detect ensemble models (stored as trained but model is EnsembleResult)
                if hasattr(model, 'ensemble_type') and hasattr(model, 'model_names'):
                    QMessageBox.information(
                        self, "Explainability Not Available",
                        f"Model '{model_name}' is an ensemble model.\n\n"
                        f"Explainability analysis (SHAP, feature importance, PDP) "
                        f"is not supported for ensemble models because they combine "
                        f"multiple base models.\n\n"
                        f"To analyze feature importance, please select one of the "
                        f"individual base models instead."
                    )
                    return
            elif model_type == 'optimized':
                model = result.best_model
                is_nn = False
            elif model_type == 'neural_network':
                model = result.model
                is_nn = True
            elif model_type == 'nn_optimized':
                # NN optimized models store best_model=None (only hyperparameters)
                # User must apply params and train to get an explainable model
                QMessageBox.information(
                    self, "Explainability Not Available",
                    f"Model '{model_name}' contains optimized hyperparameters only.\n\n"
                    f"The neural network architecture was optimized, but no trained "
                    f"model is available for explainability analysis.\n\n"
                    f"To get explainability results:\n"
                    f"1. Go to the Neural Network tab\n"
                    f"2. Click 'Apply Best Parameters'\n"
                    f"3. Train the network\n"
                    f"4. Return here and select the trained model"
                )
                return
            else:
                QMessageBox.warning(self, "Warning", f"Unknown model type: {model_type}")
                return
            
            if model is None:
                QMessageBox.warning(self, "Warning", "Selected model is None")
                return
            
            self._log(f"Is Neural Network: {is_nn}")
            
            # For NN, use the DynamicNN model directly (it now has predict method)
            if is_nn:
                analysis_model = model.model  # DynamicNN
                self._log(f"NN Model type: {type(analysis_model)}")
                self._log(f"NN has predict: {hasattr(analysis_model, 'predict')}")
            else:
                analysis_model = model
                self._log(f"Model type: {type(analysis_model)}")
            
            # v2.0: Check if model supports built-in feature importance.
            # CRITICAL FIX: When model is a sklearn Pipeline, we must drill
            # into the last step (the actual estimator) to check for
            # feature_importances_. The Pipeline wrapper itself does NOT have
            # this attribute, which was causing HistGradientBoosting and all
            # tree-based models to incorrectly show "no built-in FI".
            _fi_check_model = analysis_model
            if hasattr(analysis_model, 'named_steps') and analysis_model.named_steps:
                _fi_check_model = list(analysis_model.named_steps.values())[-1]
            has_builtin_fi = hasattr(_fi_check_model, 'feature_importances_')
            self._log(f"Model has feature_importances_: {has_builtin_fi} (checked on {_fi_check_model.__class__.__name__})")
            if not has_builtin_fi and not is_nn:
                self._log(
                    "WARNING: This model does not provide built-in feature importance. "
                    "Only permutation importance and SHAP will be available."
                )
            
            # Check if model has predict method
            if not hasattr(analysis_model, 'predict'):
                QMessageBox.warning(self, "Warning", f"Selected model ({type(analysis_model).__name__}) does not have predict method")
                return
            
            # Test predict method
            try:
                test_pred = analysis_model.predict(X_test[:1])
                self._log(f"Predict test successful: shape={test_pred.shape}")
            except Exception as e:
                self._log(f"Predict test failed: {e}")
                QMessageBox.warning(self, "Warning", f"Model predict test failed: {e}")
                return
            
            # Start analysis thread
            self.progress_bar.setVisible(True)
            self.progress_bar.setRange(0, 100)
            self.analyze_btn.setEnabled(False)
            
            self.analysis_thread = ExplainabilityThread(
                self.explainer,
                self.domain_analyzer,
                analysis_model,
                is_nn,
                X_train, X_test,
                y_train, y_test,
                feature_names,
                self.perm_check.isChecked(),
                self.shap_check.isChecked(),
                self.pdp_check.isChecked(),
                self.domain_check.isChecked(),
                self.n_features_spin.value()
            )
            self.analysis_thread.progress.connect(self._on_analysis_progress)
            self.analysis_thread.log.connect(self._on_analysis_log)
            self.analysis_thread.finished.connect(self._on_analysis_finished)
            self.analysis_thread.error.connect(self._on_analysis_error)
            self.analysis_thread.start()
            
        except Exception as e:
            self._log(f"ERROR starting analysis: {str(e)}")
            import traceback
            self._log(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to start analysis:\n{str(e)}")
    
    def _update_importance_table(self, df):
        """Update importance table."""
        self.importance_table.clearContents()
        self.importance_table.setRowCount(len(df))
        
        for row_idx, (_, row) in enumerate(df.iterrows()):
            self.importance_table.setItem(row_idx, 0, QTableWidgetItem(str(row['feature'])))
            
            # Model importance (N/A for NN)
            if 'model_importance' in row and pd.notna(row['model_importance']):
                self.importance_table.setItem(row_idx, 1, QTableWidgetItem(f"{row['model_importance']:.4f}"))
            else:
                self.importance_table.setItem(row_idx, 1, QTableWidgetItem("N/A"))
            
            # Permutation importance
            if 'permutation_importance' in row and pd.notna(row['permutation_importance']):
                self.importance_table.setItem(row_idx, 2, QTableWidgetItem(f"{row['permutation_importance']:.4f}"))
            else:
                self.importance_table.setItem(row_idx, 2, QTableWidgetItem("N/A"))
            
            # SHAP importance
            if 'shap_importance' in row and pd.notna(row['shap_importance']):
                self.importance_table.setItem(row_idx, 3, QTableWidgetItem(f"{row['shap_importance']:.4f}"))
            else:
                self.importance_table.setItem(row_idx, 3, QTableWidgetItem("N/A"))
            
            # Average rank
            if 'average_rank' in row and pd.notna(row['average_rank']):
                self.importance_table.setItem(row_idx, 4, QTableWidgetItem(f"{row['average_rank']:.4f}"))
            else:
                self.importance_table.setItem(row_idx, 4, QTableWidgetItem("N/A"))
    
    def _explain_prediction(self):
        """Explain a single prediction."""
        # v2.0 GUARD: Prevent multiple simultaneous explanations
        if hasattr(self, '_explain_running') and self._explain_running:
            return
        self._explain_running = True
        
        try:
            self._do_explain_prediction()
        finally:
            self._explain_running = False
    
    def _do_explain_prediction(self):
        """Internal: explain a single prediction."""
        # Get selected model
        model_data = self._get_selected_model()
        if model_data is None:
            QMessageBox.warning(self, "Warning", "Please select a model")
            return
        
        model_type, model_name, result = model_data
        
        if self._main_window is None:
            return
        
        data_manager = self._main_window.data_tab.get_data_manager()
        if data_manager is None:
            return
        
        try:
            X_train, X_test, y_train, y_test = data_manager.prepare_data()
            feature_names = data_manager.get_feature_names()
            
            # FIX: Verificar que feature_names coincida con X_test
            if len(feature_names) != X_test.shape[1]:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
            
            # Get the actual model object
            if model_type == 'trained':
                model = result.model
                is_nn = getattr(result, 'is_neural_network', False)
            elif model_type == 'optimized':
                model = result.best_model
                is_nn = False
            elif model_type == 'neural_network':
                model = result.model
                is_nn = True
            else:
                return
            
            if model is None:
                return
            
            instance_idx = self.instance_spin.value()

            if instance_idx >= len(X_test):
                QMessageBox.warning(self, "Warning", "Instance index out of range")
                return

            # v2.0: X_test is now a DataFrame (raw). Use .iloc for positional
            # indexing so it works regardless of the DataFrame's index labels.
            if hasattr(X_test, 'iloc'):
                X_instance = X_test.iloc[instance_idx].values.reshape(1, -1)
            else:
                X_instance = X_test[instance_idx]

            # For NN, use the actual PyTorch model
            analysis_model = model.model if is_nn else model
            
            explanation = self.explainer.explain_prediction(
                analysis_model,
                X_instance,
                feature_names=feature_names
            )
            
            text = f"""
            <h3>Prediction Explanation</h3>
            <p><b>Predicted Value:</b> {explanation.get('prediction', 'N/A')}</p>
            """
            
            if 'shap_contributions' in explanation:
                n_features = self.n_features_spin.value()
                text += f"<h4>SHAP Contributions (top {n_features}):</h4><ul>"
                
                contributions = explanation['shap_contributions']
                for feat, contrib in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:n_features]:
                    direction = "+" if contrib > 0 else ""
                    text += f"<li>{feat}: {direction}{contrib:.4f}</li>"
                
                text += "</ul>"
                
                if explanation.get('top_positive'):
                    text += f"<h4>Top Positive Contributors (top {min(n_features, 5)}):</h4><ul>"
                    for feat, val in explanation['top_positive'][:min(n_features, 5)]:
                        text += f"<li>{feat}: +{val:.4f}</li>"
                    text += "</ul>"
                
                if explanation.get('top_negative'):
                    text += f"<h4>Top Negative Contributors (top {min(n_features, 5)}):</h4><ul>"
                    for feat, val in explanation['top_negative'][:min(n_features, 5)]:
                        text += f"<li>{feat}: {val:.4f}</li>"
                    text += "</ul>"
            else:
                text += "<p>Could not generate SHAP explanation for this prediction.</p>"
            
            self.explanation_text.setHtml(text)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to explain prediction:\n{str(e)}")
    
    def _export_results(self):
        """Export all explainability results to a text file."""
        # Get selected model info for the filename
        model_data = self._get_selected_model()
        model_type, model_name, _ = model_data if model_data else (None, "unknown", None)
        
        # Suggest filename with model name and timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_name = f"explainability_{model_name or 'results'}_{timestamp}.txt"
        
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Export Explainability Results",
            default_name,
            "Text Files (*.txt);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            lines = []
            separator = "=" * 80
            
            # Header
            lines.append(separator)
            lines.append("AUTOAIM STUDIO v2.0.0 - EXPLAINABILITY REPORT")
            lines.append(separator)
            lines.append("")
            lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            lines.append(f"Model: {model_name or 'N/A'} ({model_type or 'N/A'})")
            lines.append("")
            
            # 1. Feature Importance
            lines.append(separator)
            lines.append("SECTION 1: FEATURE IMPORTANCE")
            lines.append(separator)
            lines.append("")
            
            row_count = self.importance_table.rowCount()
            col_count = self.importance_table.columnCount()
            
            if row_count > 0:
                # Header row
                headers = []
                for c in range(col_count):
                    header_item = self.importance_table.horizontalHeaderItem(c)
                    headers.append(header_item.text() if header_item else f"Col{c}")
                lines.append("  ".join(f"{h:>20}" for h in headers))
                lines.append("-" * 100)
                
                # Data rows
                for r in range(row_count):
                    row_values = []
                    for c in range(col_count):
                        item = self.importance_table.item(r, c)
                        row_values.append(item.text() if item else "N/A")
                    lines.append("  ".join(f"{v:>20}" for v in row_values))
            else:
                lines.append("No feature importance data available.")
            
            lines.append("")
            
            # 2. SHAP Values
            lines.append(separator)
            lines.append("SECTION 2: SHAP VALUES")
            lines.append(separator)
            lines.append("")
            
            shap_plain = self.shap_text.toPlainText().strip()
            if shap_plain:
                lines.append(shap_plain)
            else:
                lines.append("No SHAP values available.")
            
            lines.append("")
            
            # 3. Partial Dependence
            lines.append(separator)
            lines.append("SECTION 3: PARTIAL DEPENDENCE PLOTS")
            lines.append(separator)
            lines.append("")
            
            pdp_plain = self.pdp_text.toPlainText().strip()
            if pdp_plain:
                lines.append(pdp_plain)
            else:
                lines.append("No partial dependence data available.")
            
            lines.append("")
            
            # 4. Domain Applicability
            lines.append(separator)
            lines.append("SECTION 4: DOMAIN APPLICABILITY")
            lines.append(separator)
            lines.append("")
            
            domain_plain = self.domain_text.toPlainText().strip()
            if domain_plain:
                lines.append(domain_plain)
            else:
                lines.append("No domain applicability data available.")
            
            lines.append("")
            
            # 5. Prediction Explanation
            lines.append(separator)
            lines.append("SECTION 5: PREDICTION EXPLANATION")
            lines.append(separator)
            lines.append("")
            
            explain_plain = self.explanation_text.toPlainText().strip()
            if explain_plain:
                lines.append(explain_plain)
            else:
                lines.append("No prediction explanation available. Use 'Explain Prediction' to generate one.")
            
            lines.append("")
            
            # 6. Analysis Log
            lines.append(separator)
            lines.append("SECTION 6: ANALYSIS LOG")
            lines.append(separator)
            lines.append("")
            
            log_plain = self.log_text.toPlainText().strip()
            if log_plain:
                lines.append(log_plain)
            else:
                lines.append("No analysis log entries.")
            
            lines.append("")
            lines.append(separator)
            lines.append("END OF REPORT")
            lines.append(separator)
            
            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(lines))
            
            self._log(f"Results exported to: {file_path}")
            QMessageBox.information(
                self, "Export Complete",
                f"Explainability results successfully exported to:\n{file_path}"
            )
            
        except Exception as e:
            QMessageBox.critical(
                self, "Export Error",
                f"Failed to export results:\n{str(e)}"
            )
            self._log(f"ERROR exporting results: {str(e)}")
