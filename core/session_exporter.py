"""
Session Exporter for AutoAIM Studio.

Exports a comprehensive, scientifically complete session report containing
ALL training results, optimization details, NN training history, ensemble
configurations, and metrics — with full reproducibility information.

Formats: .txt (human-readable), .md (Markdown), .csv (spreadsheet), .json
"""

import json
import csv
import io
import inspect
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


class SessionExporter:
    """Exports the complete AutoAIM session with full scientific detail."""

    FORMAT_TXT = "txt"
    FORMAT_MD = "md"
    FORMAT_CSV = "csv"
    FORMAT_JSON = "json"

    def __init__(self, session_info: Optional[Dict[str, Any]] = None):
        self.session_info = session_info or {}
        self._entries: List[Dict[str, Any]] = []
        self._hardware_info: Optional[str] = None
        self._preprocessing_config: Optional[Dict[str, Any]] = None
        self._feature_names: List[str] = []
        # v2.0: Store optimization entries by normalized name. Uses a LIST
        # to handle multiple optimizations of the same algorithm (e.g., XGBoost
        # optimized with 5 folds, then again with 3 folds, then with 9 folds).
        # Each entry is matched sequentially to trained models with the same base name.
        self._opt_entries: Dict[str, List[Dict[str, Any]]] = {}

    # ------------------------------------------------------------------
    # Session-level metadata
    # ------------------------------------------------------------------

    def set_hardware_info(self, info: str):
        """Set hardware information string."""
        self._hardware_info = info

    def set_preprocessing_config(self, config: Dict[str, Any]):
        """Set preprocessing configuration (imputation, scaling, encoding)."""
        self._preprocessing_config = config

    def set_feature_names(self, feature_names: List[str]):
        """Set feature names used for training."""
        self._feature_names = list(feature_names)

    # ------------------------------------------------------------------
    # Collect results — exhaustive extraction
    # ------------------------------------------------------------------

    def add_training_results(self, results: List[Any]):
        """Add TrainingResult objects with exhaustive field extraction.

        v2.0: Automatically links each trained model with its corresponding
        optimization entry (stored in _opt_entries) so optimization details
        (trials, config, best_params) appear inline in the export.
        """
        for r in results:
            # Extract random_state from the model itself via get_params()
            model = getattr(r, 'model', None)
            random_state = _extract_random_state(model)
            model_type = type(model).__name__ if model is not None else "Unknown"
            model_module = type(model).__module__ if model is not None else ""

            # Full CV metrics (train_score + test_score per fold)
            cv = _safe_dict(getattr(r, 'cv_metrics', {}))
            cv_test_scores = cv.get('test_score', [])
            cv_train_scores = cv.get('train_score', [])

            entry = {
                "section": "Training",
                "model_name": getattr(r, 'model_name', 'Unknown'),
                "model_type": model_type,
                "model_module": model_module,
                "problem_type": getattr(r, 'problem_type', 'regression'),
                "random_state": random_state,
                "metrics": _safe_dict(getattr(r, 'metrics', {})),
                "cv_test_scores": [round(v, 6) for v in cv_test_scores],
                "cv_train_scores": [round(v, 6) for v in cv_train_scores],
                "cv_test_mean": round(float(np.mean(cv_test_scores)), 6) if cv_test_scores else None,
                "cv_test_std": round(float(np.std(cv_test_scores)), 6) if cv_test_scores else None,
                "cv_train_mean": round(float(np.mean(cv_train_scores)), 6) if cv_train_scores else None,
                "cv_train_std": round(float(np.std(cv_train_scores)), 6) if cv_train_scores else None,
                "cv_test_metric": cv.get('test_score_metric', 'r2'),
                "cv_r2_global": cv.get('r2_global'),
                "cv_n_folds": len(cv_test_scores),
                "cv_strategy": getattr(r, 'cv_strategy', 'kfold'),
                # v2.0: full CV config + grouped-CV diagnostics
                "cv_config": getattr(r, 'cv_config_dict', None),
                "cv_fold_test_sizes": cv.get('fold_test_sizes'),
                "cv_group_info": cv.get('group_info'),
                "cv_diagnostics": cv.get('cv_diagnostics'),
                "training_time_sec": round(float(getattr(r, 'training_time', 0.0)), 3),
                "feature_importance": _safe_dict(
                    getattr(r, 'feature_importance', {}) or {}, top_n=15
                ),
                "is_neural_network": getattr(r, 'is_neural_network', False),
                "has_predictions": getattr(r, 'predictions', None) is not None,
                "source": "training_tab",
                # v1.5.0: Include all model parameters for full reproducibility
                "model_params": _extract_model_params(model),
                # v1.5.0: External validation metrics (dual reporting)
                "has_external_validation": getattr(r, 'has_external_validation', False),
                "external_metrics": _safe_dict(getattr(r, 'external_metrics', None) or {}),
                # v2.0: GP native uncertainty (prediction standard deviation)
                "prediction_std": (
                    getattr(r, 'prediction_std', None).tolist()
                    if getattr(r, 'prediction_std', None) is not None else None
                ),
                # v2.0: GP metadata (kernel, alpha, log-likelihood, etc.)
                "gp_metadata": getattr(r, 'gp_metadata', None),
            }

            # Include complete NN history if applicable
            if entry["is_neural_network"]:
                nn_hist = getattr(r, 'nn_history', None)
                if nn_hist:
                    entry["nn_history"] = {
                        k: [round(v, 6) for v in vals]
                        for k, vals in nn_hist.items()
                    }
                    # Compute best epoch from val_loss
                    if 'val_loss' in nn_hist and nn_hist['val_loss']:
                        entry["nn_best_epoch"] = int(np.argmin(nn_hist['val_loss'])) + 1
                        entry["nn_best_val_loss"] = round(float(min(nn_hist['val_loss'])), 6)

            self._entries.append(entry)

    def add_crystal_link_info(self, id_column: Optional[str] = None,
                               cif_folder: Optional[str] = None,
                               n_matched: int = 0,
                               n_total: int = 0,
                               crystal_features: Optional[List[str]] = None,
                               composition_replaced: bool = False):
        """v2.0: Add crystal structure linking metadata to the export.
        
        Records information about linked CIF structures for reproducibility.
        """
        entry = {
            "section": "Crystal Link",
            "id_column": id_column,
            "cif_folder": cif_folder,
            "n_matched": n_matched,
            "n_total": n_total,
            "match_rate": n_matched / n_total if n_total > 0 else 0.0,
            "crystal_features": crystal_features or [],
            "composition_features_replaced": composition_replaced,
        }
        self._entries.append(entry)

    def add_optimization_result(self, result: Any, algorithm: str = "",
                                 n_trials: int = 0, timeout: Optional[int] = None,
                                 all_trials_df: Optional[Any] = None,
                                 optimization_config: Optional[Dict[str, Any]] = None):
        """Add OptimizationResult with ALL trial data and full configuration.

        v1.5.0: Added optimization_config dict to capture the complete
        optimization settings (cv_folds, random_state, pruning, selected
        parameters, parameter ranges, etc.) for full reproducibility.
        """
        # Extract random_state from best_model
        best_model = getattr(result, 'best_model', None)
        random_state = _extract_random_state(best_model)

        # Get ALL individual trial scores if available
        all_trials = []
        trials_df = getattr(result, 'all_trials', None)
        if trials_df is None:
            trials_df = all_trials_df
        if trials_df is not None:
            try:
                df_dict = trials_df.to_dict('records') if hasattr(trials_df, 'to_dict') else []
                for record in df_dict:
                    trial_entry = {}
                    for k, v in record.items():
                        if isinstance(v, (np.floating, np.integer)):
                            trial_entry[k] = float(v) if isinstance(v, np.floating) else int(v)
                        elif isinstance(v, (int, float, str, bool)) or v is None:
                            trial_entry[k] = v
                    if trial_entry:
                        all_trials.append(trial_entry)
            except Exception:
                pass

        # CV metrics per trial if available
        cv = _safe_dict(getattr(result, 'cv_metrics', {}))

        # v1.5.0: Merge optimization config with defaults
        opt_cfg = dict(optimization_config) if optimization_config else {}

        entry = {
            "section": "Optimization",
            "model_name": algorithm or getattr(result, 'algorithm', 'Optimized'),
            "algorithm": algorithm or getattr(result, 'algorithm', ''),
            "best_params": _safe_dict(getattr(result, 'best_params', {})),
            "best_score": getattr(result, 'best_score', None),
            "metrics": _safe_dict(getattr(result, 'metrics', {})),
            "n_trials": getattr(result, 'n_trials', n_trials),
            "timeout_sec": timeout,
            "optimization_time_sec": round(float(getattr(result, 'optimization_time', 0.0)), 3),
            "random_state": random_state,
            "study_name": getattr(result, 'study_name', ''),
            "search_strategy": getattr(result, 'search_strategy', 'bayesian'),
            "cv_strategy": getattr(result, 'cv_strategy', 'kfold'),
            "problem_type": getattr(result, 'problem_type', 'regression'),
            "cv_test_scores": [round(v, 6) for v in cv.get('test_score', [])],
            "cv_train_scores": [round(v, 6) for v in cv.get('train_score', [])],
            "cv_test_mean": round(float(np.mean(cv.get('test_score', []))), 6) if cv.get('test_score') else None,
            "cv_test_std": round(float(np.std(cv.get('test_score', []))), 6) if cv.get('test_score') else None,
            "cv_train_mean": round(float(np.mean(cv.get('train_score', []))), 6) if cv.get('train_score') else None,
            "cv_train_std": round(float(np.std(cv.get('train_score', []))), 6) if cv.get('train_score') else None,
            "cv_test_metric": cv.get('test_score_metric', 'r2'),
            "cv_r2_global": cv.get('r2_global'),
            # v2.0: full CV config + grouped-CV diagnostics
            "cv_config": getattr(result, 'cv_config_dict', None),
            "cv_fold_test_sizes": cv.get('fold_test_sizes'),
            "cv_group_info": cv.get('group_info'),
            "cv_diagnostics": cv.get('cv_diagnostics'),
            "all_trials": all_trials[:200],  # cap at 200 to keep file size reasonable
            "n_trials_recorded": len(all_trials),
            "source": "optimization_tab",
            "model_params": _extract_model_params(best_model),
            # v1.5.0: Full optimization configuration for reproducibility
            "optimization_config": opt_cfg,
        }
        self._entries.append(entry)
        # v2.0: Store by normalized name in a list to support multiple
        # optimizations of the same algorithm (e.g., XGBoost with different
        # CV folds). Entries are matched sequentially to trained models.
        norm_algo = self._normalize_name(algorithm or getattr(result, 'algorithm', ''))
        if norm_algo:
            if norm_algo not in self._opt_entries:
                self._opt_entries[norm_algo] = []
            self._opt_entries[norm_algo].append(entry)

    def add_nn_training_result(self, model_name: str, architecture_desc: str,
                                architecture_detail: Optional[Dict] = None,
                                history: Optional[Dict[str, List[float]]] = None,
                                training_config: Optional[Dict[str, Any]] = None,
                                best_epoch: Optional[int] = None,
                                optimization_info: Optional[Dict] = None,
                                model_params: Optional[Dict] = None,
                                nn_optimization_config: Optional[Dict[str, Any]] = None,
                                metrics: Optional[Dict[str, float]] = None,
                                training_time_sec: float = 0.0,
                                cv_metrics: Optional[Dict[str, List[float]]] = None,
                                optimization_time_sec: float = 0.0):
        """Add Neural Network training results with FULL epoch history."""
        cfg = training_config or {}
        hist = history or {}

        # Full epoch history (all epochs, not truncated)
        full_history = {}
        for k, vals in hist.items():
            if isinstance(vals, (list, tuple, np.ndarray)):
                full_history[k] = [round(float(v), 6) for v in vals]

        # Compute statistics from history
        stats = {}
        if 'val_loss' in full_history and full_history['val_loss']:
            stats['best_epoch'] = int(np.argmin(full_history['val_loss'])) + 1
            stats['best_val_loss'] = round(float(min(full_history['val_loss'])), 6)
            stats['final_val_loss'] = round(float(full_history['val_loss'][-1]), 6)
        if 'train_loss' in full_history and full_history['train_loss']:
            stats['final_train_loss'] = round(float(full_history['train_loss'][-1]), 6)
            stats['best_train_loss'] = round(float(min(full_history['train_loss'])), 6)
        if 'val_loss' in full_history and 'train_loss' in full_history:
            n = min(len(full_history['val_loss']), len(full_history['train_loss']))
            if n > 0:
                overfit = [full_history['train_loss'][i] - full_history['val_loss'][i]
                           for i in range(n)]
                stats['mean_gap_train_val'] = round(float(np.mean(overfit)), 6)

        entry = {
            "section": "Neural Network",
            "model_name": model_name,
            "architecture_desc": architecture_desc,
            "architecture_detail": architecture_detail or {},
            "epochs_trained": len(full_history.get('train_loss', [])),
            "history": full_history,
            "history_stats": stats,
            "training_config": cfg,
            "best_epoch": best_epoch or stats.get('best_epoch'),
            "optimization_info": optimization_info or {},
            "model_params": model_params or {},
            "source": "nn_tab",
            # v1.5.0: NN optimization configuration (trials, cv folds, opt epochs, etc.)
            "nn_optimization_config": dict(nn_optimization_config) if nn_optimization_config else {},
            # v2.0: Metrics and timing for ranking/comparison
            "metrics": _safe_dict(metrics) if metrics else {},
            "training_time_sec": round(float(training_time_sec), 3),
            # v2.0: CV fold scores (for per-fold breakdown in export)
            "cv_test_scores": [round(v, 6) for v in (cv_metrics or {}).get('test_score', [])],
            "cv_train_scores": [round(v, 6) for v in (cv_metrics or {}).get('train_score', [])],
            "cv_test_mean": round(float(np.mean((cv_metrics or {}).get('test_score', []))), 6) if cv_metrics and cv_metrics.get('test_score') else None,
            "cv_test_std": round(float(np.std((cv_metrics or {}).get('test_score', []))), 6) if cv_metrics and cv_metrics.get('test_score') else None,
            "cv_train_mean": round(float(np.mean((cv_metrics or {}).get('train_score', []))), 6) if cv_metrics and cv_metrics.get('train_score') else None,
            "cv_train_std": round(float(np.std((cv_metrics or {}).get('train_score', []))), 6) if cv_metrics and cv_metrics.get('train_score') else None,
            "cv_test_metric": (cv_metrics or {}).get('test_score_metric', 'r2'),
            "cv_r2_global": (cv_metrics or {}).get('r2_global'),
            # v2.0: grouped-CV diagnostics (NNs are K-Fold-only, but keep the
            # fields for schema consistency)
            "cv_fold_test_sizes": (cv_metrics or {}).get('fold_test_sizes'),
            "cv_group_info": (cv_metrics or {}).get('group_info'),
            "cv_diagnostics": (cv_metrics or {}).get('cv_diagnostics'),
            # v2.0: Optimization time (for NN that was optimized then retrained)
            "optimization_time_sec": round(float(optimization_time_sec), 3),
        }
        self._entries.append(entry)

    def add_ensemble_result(self, result: Any,
                            weight_optimization_config: Optional[Dict[str, Any]] = None):
        """Add EnsembleResult with full detail.

        v1.5.0: Added weight_optimization_config to capture the complete
        weight optimization settings for reproducibility.
        """
        entry = {
            "section": "Ensemble",
            "model_name": getattr(result, 'model_name',
                                  f"Ensemble_{getattr(result, 'ensemble_type', 'unknown')}"),
            "ensemble_type": getattr(result, 'ensemble_type', 'unknown'),
            "base_models": list(getattr(result, 'model_names', [])),
            "weights": _safe_dict(getattr(result, 'weights', {})),
            "optimized_weights": _safe_dict(getattr(result, 'optimized_weights', {})),
            "metrics": _safe_dict(getattr(result, 'metrics', {})),
            "cv_scores": [round(v, 6) for v in _safe_list(getattr(result, 'cv_scores', []))],
            "training_time_sec": round(float(getattr(result, 'training_time', 0.0)), 3),
            "problem_type": getattr(result, 'problem_type', 'regression'),
            "optimization_trials": getattr(result, 'optimization_trials', 0),
            "has_predictions": getattr(result, 'predictions', None) is not None,
            "source": "ensemble_tab",
            # v1.5.0: Weight optimization configuration
            "weight_optimization_config": dict(weight_optimization_config) if weight_optimization_config else {},
        }
        self._entries.append(entry)

    # ------------------------------------------------------------------
    # Optimization linking — attach opt details to trained models (v2.0)
    # ------------------------------------------------------------------

    def _link_optimization_entries(self):
        """Link each Training/Neural Network entry with its Optimization entry.

        v2.0: Uses _opt_entries (populated during add_optimization_result calls)
        instead of scanning _entries. This works regardless of collection order
        in _export_session_report (Training is collected BEFORE Optimization).

        When a model is trained with optimized parameters, there are two entries:
        1. A "Training" entry (e.g., "XGBoost_Optimized") with metrics/params
        2. An "Optimization" entry (e.g., "XGBoost") with trials/config/best_params

        v2.0 FIX: Supports models with counter suffixes like "XGBoost_Optimized_1"
        (from _get_unique_model_name when the user optimizes+trains the same
        algorithm multiple times in one session).

        v2.0 FIX: Also links Neural Network entries (e.g., "NN_Optimized") with
        their NN optimization results. NN optimizations are stored under key
        "neural_network" in _opt_entries.
        """
        # v2.0: Link each Training/NN entry with its optimization from _opt_entries.
        # Uses sequential matching: the 1st "XGBoost_Optimized" model gets the
        # 1st XGBoost optimization entry, the 2nd model gets the 2nd entry, etc.
        # This correctly handles multiple optimizations of the same algorithm with
        # different CV folds.
        opt_indices: Dict[str, int] = {}  # Track next available index per algorithm
        for e in self._entries:
            # v2.0: Process BOTH Training entries (sklearn) and Neural Network
            # entries (NNs). NNs optimized via nn_builder_tab go to section
            # 'Neural Network', not 'Training', so they were never linked before.
            if e['section'] not in ('Training', 'Neural Network'):
                continue
            name = e.get('model_name', '')
            norm = self._normalize_name(name)
            # v2.0 FIX: Use '_optimized' in norm instead of endswith.
            # Names like 'xgboost_optimized_1' (from _get_unique_model_name)
            # contain '_optimized' but don't end with it. We extract the base
            # as everything before the first '_optimized' occurrence.
            if '_optimized' in norm:
                base = norm.split('_optimized')[0]
                # v2.0 FIX: NN optimizations are stored under "neural_network"
                # (from algorithm='Neural Network' in add_optimization_result).
                # The model name normalizes to "nn_optimized" so base is "nn".
                # Map "nn" → "neural_network" for lookup.
                lookup_base = base
                if base == 'nn':
                    lookup_base = 'neural_network'
                if lookup_base in self._opt_entries and self._opt_entries[lookup_base]:
                    idx = opt_indices.get(lookup_base, 0)
                    if idx < len(self._opt_entries[lookup_base]):
                        e['_linked_optimization'] = self._opt_entries[lookup_base][idx]
                        opt_indices[lookup_base] = idx + 1

    @staticmethod
    def _normalize_name(name: str) -> str:
        """Normalize a model name for matching."""
        return name.lower().replace(' ', '_').replace('-', '_').strip()

    def _get_linked_algorithms(self) -> set:
        """Return a set of normalized algorithm names that are linked to a
        Training or Neural Network entry (via _linked_optimization). Used to
        suppress redundant standalone Optimization sections and ranking entries.

        v2.0: Also processes 'Neural Network' entries for optimized NNs.
        """
        linked = set()
        for e in self._entries:
            if e['section'] in ('Training', 'Neural Network') and '_linked_optimization' in e:
                algo = e['_linked_optimization'].get('algorithm', '')
                if algo:
                    linked.add(self._normalize_name(algo))
        return linked

    # ------------------------------------------------------------------
    # Orphan filtering — critical for deleted results (v1.5.0)
    # ------------------------------------------------------------------

    def filter_orphaned_entries(self):
        """Remove optimization entries that don't have a corresponding trained model.

        When a user deletes a result in the UI, the trained model is removed from
        trainer.results (and related stores), but optimization_tab.optimization_results
        may still hold the optimization metadata. This method filters out those
        orphaned optimization records so the export reflects exactly what the user sees.
        """
        # Build a set of model names from primary entries (training, nn, ensemble)
        primary_names = set()
        for e in self._entries:
            if e['section'] in ('Training', 'Neural Network', 'Ensemble'):
                primary_names.add(e['model_name'])
                # Also add common variants (e.g., "XGBoost" matches "XGBoost_Optimized")
                # by adding each prefix
                parts = e['model_name'].split('_')
                for i in range(1, len(parts) + 1):
                    primary_names.add('_'.join(parts[:i]))

        # Keep only optimization entries whose algorithm matches a primary model
        # v1.5.0 FIX: Normalize names (lowercase + underscore) before comparing
        # to handle "Random Forest" vs "Random_Forest" mismatches.
        def _normalize(name):
            return name.lower().replace(' ', '_').replace('-', '_').strip()
        
        normalized_primary = set(_normalize(pn) for pn in primary_names if pn)
        
        filtered = []
        removed_count = 0
        for e in self._entries:
            if e['section'] == 'Optimization':
                # v1.5.0: NEVER remove an optimization that has valid data
                # (best_params, best_score, trials). This ensures optimization
                # details are always preserved even if name matching fails.
                has_valid_data = bool(
                    e.get('best_params') or 
                    e.get('best_score') is not None or
                    e.get('all_trials')
                )
                
                algo = e.get('algorithm', '') or e.get('model_name', '')
                norm_algo = _normalize(algo)
                # Check normalized match + original fallback
                name_matches = (
                    norm_algo in normalized_primary or
                    any(norm_algo == _normalize(pn) or algo in pn or pn in algo 
                        for pn in primary_names if pn)
                )
                
                if name_matches or has_valid_data:
                    filtered.append(e)
                else:
                    removed_count += 1
            else:
                filtered.append(e)

        self._entries = filtered
        return removed_count

    # ------------------------------------------------------------------
    # Export to file
    # ------------------------------------------------------------------

    def export(self, file_path: str, fmt: Optional[str] = None) -> str:
        # v2.0: Link optimization entries with their trained model counterparts
        self._link_optimization_entries()
        # Filter orphaned optimization entries before exporting
        removed = self.filter_orphaned_entries()
        if removed > 0:
            pass

        path = Path(file_path)
        if fmt is None:
            fmt = path.suffix.lstrip('.').lower()
            if fmt not in (self.FORMAT_TXT, self.FORMAT_MD, self.FORMAT_CSV, self.FORMAT_JSON):
                fmt = self.FORMAT_TXT
        content = self._render(fmt)
        path.write_text(content, encoding='utf-8')
        return str(path)

    def _render(self, fmt: str) -> str:
        if fmt == self.FORMAT_JSON:
            return self._render_json()
        if fmt == self.FORMAT_CSV:
            return self._render_csv()
        if fmt == self.FORMAT_MD:
            return self._render_markdown()
        return self._render_txt()

    # ------------------------------------------------------------------
    # TXT renderer — exhaustive scientific report
    # ------------------------------------------------------------------

    def _render_txt(self) -> str:
        lines = []
        sep = "=" * 76
        subsep = "-" * 76
        minorsep = "." * 40

        def L(text=""): lines.append(text)

        L(sep)
        L("  AutoAIM Studio v2.0.0 — COMPLETE Session Report")
        L(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        L(sep)
        L()

        # ==================== REPRODUCIBILITY SECTION ====================
        L("REPRODUCIBILITY INFORMATION")
        L(subsep)
        L(f"  AutoAIM Version : 2.0.0")
        if self._hardware_info:
            L(f"  Hardware        : {self._hardware_info}")
        if self._preprocessing_config:
            L("  Preprocessing   :")
            for k, v in self._preprocessing_config.items():
                L(f"    {k}: {v}")
        if self._feature_names:
            L(f"  Feature Names   : {', '.join(self._feature_names[:20])}")
            if len(self._feature_names) > 20:
                L(f"                    ... and {len(self._feature_names) - 20} more")
        L()

        # ==================== DATASET ====================
        if self.session_info:
            L("DATASET")
            L(subsep)
            for k, v in self.session_info.items():
                label = k.replace('_', ' ').title()
                L(f"  {label:<25} {v}")
            L()

        # ==================== SESSION SUMMARY ====================
        # v2.0: Count only entries that will be displayed (skip linked opts)
        linked_algos = self._get_linked_algorithms()
        sections = {}
        visible_count = 0
        for e in self._entries:
            if e['section'] == 'Optimization':
                algo_norm = self._normalize_name(e.get('algorithm', e.get('model_name', '')))
                if algo_norm in linked_algos:
                    continue  # Will be suppressed — don't count
            sections[e['section']] = sections.get(e['section'], 0) + 1
            visible_count += 1
        if sections:
            L("SESSION SUMMARY")
            L(subsep)
            for section, count in sorted(sections.items()):
                L(f"  {section:<20} {count} result(s)")
            L(f"  {'TOTAL':<20} {visible_count} result(s)")
            L()

        # ==================== DETAILED ENTRIES ====================
        # v2.0: Filter out linked Optimization entries so they don't produce
        # phantom headers (the continue inside the loop skips content but the
        # header/subsep are already printed).
        linked_algos = self._get_linked_algorithms()
        visible_entries = []
        for e in self._entries:
            if e['section'] == 'Optimization':
                algo_norm = self._normalize_name(e.get('algorithm', e.get('model_name', '')))
                if algo_norm in linked_algos:
                    continue
            visible_entries.append(e)

        for idx, entry in enumerate(visible_entries, 1):
            L(f"[{idx}] {entry['section'].upper()}: {entry['model_name']}")
            L(subsep)

            # ---- TRAINING ----
            if entry['section'] == 'Training':
                L(f"  Model Type       : {entry.get('model_type', 'N/A')} ({entry.get('model_module', '')})")
                L(f"  Problem Type     : {entry['problem_type']}")
                L(f"  Random State     : {entry.get('random_state', 'N/A')}")
                L(f"  Training Time    : {entry['training_time_sec']:.3f}s")
                L(f"  Has Predictions  : {entry['has_predictions']}")
                L()

                # Model parameters (full reproducibility)
                mp = entry.get('model_params', {})
                if mp:
                    L("  Model Parameters:")
                    for k, v in sorted(mp.items()):
                        L(f"    {k:<25} {v}")
                    L()

                # v2.0: GP metadata (kernel, alpha, log-likelihood, etc.)
                gp_meta = entry.get('gp_metadata')
                if gp_meta:
                    L("  Gaussian Process Metadata:")
                    for k, v in sorted(gp_meta.items()):
                        L(f"    {k:<25} {v}")
                    L()

                # v2.0: GP prediction standard deviation summary
                pred_std = entry.get('prediction_std')
                if pred_std is not None:
                    try:
                        arr = np.array(pred_std)
                        L("  Prediction Uncertainty (GP std):")
                        L(f"    Mean  : {float(np.mean(arr)):.6f}")
                        L(f"    Std   : {float(np.std(arr)):.6f}")
                        L(f"    Min   : {float(np.min(arr)):.6f}")
                        L(f"    Max   : {float(np.max(arr)):.6f}")
                        L()
                    except Exception:
                        pass

                # Metrics (internal test split)
                _render_metrics_txt(L, entry.get('metrics', {}), indent=2)
                L()

                # v1.5.0: External validation metrics
                ext_metrics = entry.get('external_metrics', {})
                has_ext = entry.get('has_external_validation', False)
                if has_ext and ext_metrics:
                    L("  [EXTERNAL VALIDATION METRICS — unbiased evaluation]")
                    _render_metrics_txt(L, ext_metrics, indent=2)
                    L()

                # v1.5.0: Confidence interval params
                ci_params = entry.get('confidence_interval_params', {})
                if ci_params and ci_params.get('type'):
                    L("  Confidence Interval Parameters:")
                    for k, v in ci_params.items():
                        L(f"    {k:<25} {v}")
                    L()

                # v1.5.0: Learning curve data
                lc_data = entry.get('learning_curve_data', {})
                if lc_data:
                    L(f"  Learning Curve ({len(lc_data.get('train_sizes_abs', []))} points):")
                    L(f"    {'Size':>8} | {'Train Score':>12} | {'Val Score':>12} | {'Gap':>10}")
                    L(f"    {'-'*8}-+-{'-'*12}-+-{'-'*12}-+-{'-'*10}")
                    for i, size in enumerate(lc_data.get('train_sizes_abs', [])):
                        train_s = lc_data.get('train_scores_mean', [])[i] if i < len(lc_data.get('train_scores_mean', [])) else 'N/A'
                        val_s = lc_data.get('test_scores_mean', [])[i] if i < len(lc_data.get('test_scores_mean', [])) else 'N/A'
                        if isinstance(train_s, float) and isinstance(val_s, float):
                            gap = train_s - val_s
                            L(f"    {size:8d} | {train_s:12.4f} | {val_s:12.4f} | {gap:10.4f}")
                    L()

                # CV scores (per fold) — v2.0: shared renderer for all strategies
                if entry.get('cv_test_scores'):
                    _render_cv_block_txt(L, entry, indent=2)
                    L()

                # v2.0: Linked optimization details (inline with trained model)
                linked_opt = entry.get('_linked_optimization')
                if linked_opt:
                    _render_optimization_inline_txt(L, linked_opt, indent=2)

                # Feature importance
                fi = entry.get('feature_importance', {})
                if fi:
                    L("  Feature Importance (top 15):")
                    for feat, imp in fi.items():
                        L(f"    {feat:<35} {imp:10.6f}")
                    L()

                # NN history if applicable
                if entry.get('is_neural_network') and entry.get('nn_history'):
                    L("  Neural Network Training History:")
                    nnh = entry['nn_history']
                    for k, vals in nnh.items():
                        if vals:
                            L(f"    {k}: {len(vals)} epochs, best={min(vals):.6f}, final={vals[-1]:.6f}")
                    if entry.get('nn_best_epoch'):
                        L(f"    Best epoch: {entry['nn_best_epoch']} (val_loss={entry.get('nn_best_val_loss', 'N/A')})")
                    L()

            # ---- CRYSTAL LINK (v2.0) ----
            elif entry['section'] == 'Crystal Link':
                L(f"  ID Column        : {entry.get('id_column', 'N/A')}")
                L(f"  CIF Folder       : {entry.get('cif_folder', 'N/A')}")
                L(f"  Matched          : {entry['n_matched']}/{entry['n_total']} "
                  f"({entry.get('match_rate', 0)*100:.1f}%)")
                if entry.get('composition_features_replaced'):
                    L(f"  Composition      : Replaced (comp_* from CIF)")
                cf = entry.get('crystal_features', [])
                if cf:
                    L(f"  Crystal Features : {len(cf)} columns")
                    for feat_name in cf:
                        L(f"    - {feat_name}")
                L()

            # ---- OPTIMIZATION ----
            # v2.0: Standalone Optimization entries (not linked to any Training
            # model) are shown here. Linked ones are filtered out above.
            elif entry['section'] == 'Optimization':
                L(f"  Algorithm        : {entry['model_name']}")
                L(f"  Problem Type     : {entry.get('problem_type', 'N/A')}")
                search_strat = entry.get('search_strategy', 'bayesian')
                L(f"  Search Strategy  : {'Bayesian (Optuna)' if search_strat == 'bayesian' else 'Grid Search'}")
                cv_strat = entry.get('cv_strategy', 'kfold')
                L(f"  CV Strategy      : {'Leave-One-Out' if cv_strat == 'loo' else 'K-Fold'}")
                L(f"  Trials           : {entry['n_trials']}")
                L(f"  Trials Recorded  : {entry['n_trials_recorded']}")
                if entry.get('timeout_sec'):
                    L(f"  Timeout          : {entry['timeout_sec']}s")
                L(f"  Best Score       : {_fmt(entry['best_score'])}")
                L(f"  Opt. Time        : {entry['optimization_time_sec']:.3f}s")
                L(f"  Study Name       : {entry.get('study_name', 'N/A')}")

                # v1.5.0: Full optimization configuration
                opt_cfg = entry.get('optimization_config', {})
                if opt_cfg:
                    L()
                    L("  Optimization Configuration:")
                    for k, v in sorted(opt_cfg.items()):
                        if isinstance(v, dict):
                            L(f"    {k:<25} (dict with {len(v)} entries)")
                            for sub_k, sub_v in sorted(v.items())[:10]:
                                L(f"      {sub_k:<23} {sub_v}")
                            if len(v) > 10:
                                L(f"      ... and {len(v) - 10} more")
                        else:
                            L(f"    {k:<25} {v}")
                L()

                # Best parameters
                bp = entry.get('best_params', {})
                if bp:
                    L("  Best Parameters:")
                    for k, v in sorted(bp.items()):
                        L(f"    {k:<25} {v}")
                    L()

                # Model params
                mp = entry.get('model_params', {})
                if mp:
                    L("  Best Model Parameters:")
                    for k, v in sorted(mp.items()):
                        L(f"    {k:<25} {v}")
                    L()

                _render_metrics_txt(L, entry.get('metrics', {}), indent=2)
                L()

                # v1.5.0: External validation metrics
                ext_metrics = entry.get('external_metrics', {})
                has_ext = entry.get('has_external_validation', False)
                if has_ext and ext_metrics:
                    L("  [EXTERNAL VALIDATION METRICS — unbiased evaluation]")
                    _render_metrics_txt(L, ext_metrics, indent=2)
                    L()

                # CV scores
                test_scores = entry.get('cv_test_scores', [])
                if test_scores:
                    L(f"  CV Test Scores ({len(test_scores)} folds): " + ", ".join(f"{s:.4f}" for s in test_scores))
                    L()

                # All trials summary
                trials = entry.get('all_trials', [])
                if trials:
                    L(f"  Trial History (top 20 by score):")
                    sorted_trials = sorted(trials, key=lambda t: t.get('value', t.get('score', -float('inf'))), reverse=True)[:20]
                    L(f"    {'Trial':>6} | {'Score':>12} | Parameters")
                    L(f"    {'-'*6}-+-{'-'*12}-+-{'-'*40}")
                    for t in sorted_trials:
                        tid = t.get('number', t.get('trial_id', '?'))
                        score = t.get('value', t.get('score', 'N/A'))
                        # v2.0: Trials may have flat columns (params_n_estimators)
                        # or nested dict (params). Handle both formats.
                        params_dict = t.get('params', {})
                        if not params_dict:
                            # Extract from flat columns like params_n_estimators
                            params_dict = {}
                            for k, v in t.items():
                                if k.startswith('params_'):
                                    param_name = k[7:]  # Remove 'params_' prefix
                                    params_dict[param_name] = v
                        params_str = str(params_dict)[:50] if params_dict else '()'
                        L(f"    {tid:6} | {_fmt(score):>12} | {params_str}")
                    L()

            # ---- NEURAL NETWORK ----
            elif entry['section'] == 'Neural Network':
                L(f"  Architecture     : {entry.get('architecture_desc', 'N/A')}")
                detail = entry.get('architecture_detail', {})
                if detail:
                    L("  Layer Details:")
                    for k, v in detail.items():
                        L(f"    {k}: {v}")
                L(f"  Epochs Trained   : {entry['epochs_trained']}")
                if entry.get('best_epoch'):
                    L(f"  Best Epoch       : {entry['best_epoch']}")
                # v2.0: Training time (was missing from NN renderer)
                tt = entry.get('training_time_sec', 0)
                if tt:
                    L(f"  Training Time    : {tt:.3f}s")
                # v2.0: Optimization time (for NN that was optimized then retrained)
                opt_time = entry.get('optimization_time_sec', 0)
                if opt_time:
                    L(f"  Optimization Time: {opt_time:.3f}s")
                L()

                # v2.0: CV fold scores (per-fold breakdown) — shared renderer
                if entry.get('cv_test_scores'):
                    _render_cv_block_txt(L, entry, indent=2)
                    L()

                # v2.0: Linked optimization details (inline with trained NN)
                # This shows [OPTIMIZATION DETAILS] for optimized NNs (e.g.,
                # NN_Optimized) when they were trained from an NN architecture
                # optimization run in nn_builder_tab.
                linked_opt = entry.get('_linked_optimization')
                if linked_opt:
                    _render_optimization_inline_txt(L, linked_opt, indent=2)

                # v2.0: Training config with ALL reproducibility fields
                cfg = entry.get('training_config', {})
                if cfg:
                    L("  Training Configuration:")
                    # Reproducibility-critical fields first
                    repro_fields = ['random_seed', 'cv_folds', 'cv_strategy', 'problem_type',
                                    'epochs', 'batch_size', 'learning_rate',
                                    'optimizer', 'scheduler', 'weight_decay',
                                    'early_stopping_patience', 'early_stopping_min_delta',
                                    'gradient_clip_val']
                    shown = set()
                    for field in repro_fields:
                        if field in cfg:
                            L(f"    {field:<25} {cfg[field]}")
                            shown.add(field)
                    # Any remaining fields
                    for k, v in sorted(cfg.items()):
                        if k not in shown:
                            L(f"    {k:<25} {v}")
                    L()

                # Optimization info
                opt = entry.get('optimization_info', {})
                if opt:
                    L("  Optimization:")
                    for k, v in opt.items():
                        L(f"    {k:<25} {v}")
                    L()

                # v1.5.0: NN optimization configuration (trials, cv folds, etc.)
                nn_opt_cfg = entry.get('nn_optimization_config', {})
                if nn_opt_cfg:
                    L("  NN Optimization Configuration:")
                    for k, v in sorted(nn_opt_cfg.items()):
                        if isinstance(v, dict):
                            L(f"    {k:<25} (dict with {len(v)} entries)")
                            for sub_k, sub_v in sorted(v.items())[:10]:
                                L(f"      {sub_k:<23} {sub_v}")
                            if len(v) > 10:
                                L(f"      ... and {len(v) - 10} more")
                        else:
                            L(f"    {k:<25} {v}")
                    L()

                # History stats
                stats = entry.get('history_stats', {})
                if stats:
                    L("  Training Statistics:")
                    for k, v in stats.items():
                        L(f"    {k:<25} {v}")
                    L()

                # v1.5.0: External validation metrics
                ext_metrics = entry.get('external_metrics', {})
                has_ext = entry.get('has_external_validation', False)
                if has_ext and ext_metrics:
                    L("  [EXTERNAL VALIDATION METRICS — unbiased evaluation]")
                    _render_metrics_txt(L, ext_metrics, indent=2)
                    L()

                # Full epoch history table
                hist = entry.get('history', {})
                if hist:
                    keys = list(hist.keys())
                    n_epochs = len(hist.get(keys[0], [])) if keys else 0
                    L(f"  Full Epoch History ({n_epochs} epochs):")
                    # Header
                    epoch_width = max(5, len(str(n_epochs)))
                    header_parts = [f"{'Epoch':>{epoch_width}}"]
                    for k in keys:
                        header_parts.append(f"{k:>14}")
                    L("  " + " | ".join(header_parts))
                    L("  " + "-" * (epoch_width + 3 + 17 * len(keys)))
                    # Data — show first 5, last 15, with ellipsis
                    display_epochs = []
                    if n_epochs <= 25:
                        display_epochs = list(range(n_epochs))
                    else:
                        display_epochs = list(range(5)) + [-1] + list(range(n_epochs - 15, n_epochs))

                    for ei in display_epochs:
                        if ei == -1:
                            L("  " + "...".center(epoch_width + 3 + 17 * len(keys)))
                            continue
                        row_parts = [f"{ei+1:>{epoch_width}}"]
                        for k in keys:
                            vals = hist.get(k, [])
                            val = vals[ei] if ei < len(vals) else "N/A"
                            row_parts.append(f"{val:14.6f}" if isinstance(val, float) else f"{str(val):>14}")
                        L("  " + " | ".join(row_parts))
                    L()

            # ---- ENSEMBLE ----
            elif entry['section'] == 'Ensemble':
                L(f"  Ensemble Type    : {entry['ensemble_type']}")
                L(f"  Problem Type     : {entry.get('problem_type', 'N/A')}")
                L(f"  Training Time    : {entry['training_time_sec']:.3f}s")
                L(f"  Opt. Trials      : {entry.get('optimization_trials', 0)}")
                L(f"  Base Models      : {', '.join(entry['base_models'])}")
                L()

                # v1.5.0: Weight optimization configuration
                wopt_cfg = entry.get('weight_optimization_config', {})
                if wopt_cfg:
                    L("  Weight Optimization Configuration:")
                    for k, v in sorted(wopt_cfg.items()):
                        L(f"    {k:<25} {v}")
                    L()

                # Weights
                w = entry.get('weights', {})
                ow = entry.get('optimized_weights', {})
                L("  Weights:")
                for m in entry['base_models']:
                    manual = w.get(m, 'N/A')
                    optimized = ow.get(m, 'N/A')
                    if ow:
                        L(f"    {m:<20} manual={manual}, optimized={optimized}")
                    else:
                        L(f"    {m:<20} {manual}")
                L()

                _render_metrics_txt(L, entry.get('metrics', {}), indent=2)
                L()

                cv = entry.get('cv_scores', [])
                if cv:
                    # These are trial scores (one per trial), not fold scores
                    L(f"  CV Scores ({len(cv)} trials): " + ", ".join(f"{s:.4f}" for s in cv))
                    L()

            L()

        # ==================== RANKING ====================
        if self._entries:
            L(sep)
            L("FINAL RANKING (sorted by primary metric)")
            L(subsep)
            ranking = self._build_ranking()
            if ranking:
                L(f"  {'Pos':>4} | {'Model':<40} | {'Section':<14} | {'R2/Acc':>8} | {'RMSE':>8} | {'Time(s)':>8}")
                L("  " + "-" * 95)
                for i, row in enumerate(ranking, 1):
                    marker = " *** BEST ***" if i == 1 else ""
                    L(f"  {i:4d} | {row['name']:<40} | {row['section']:<14} | "
                      f"{row['primary_metric']:>8.4f} | {row['rmse']:>8.4f} | {row['time']:>8.1f}{marker}")
            L()

        # ==================== REPRODUCIBILITY CHECKLIST ====================
        L(sep)
        L("REPRODUCIBILITY CHECKLIST")
        L(subsep)
        L("  To reproduce these results exactly:")
        if self.session_info.get('data_file'):
            L(f"  [ ] Dataset: {self.session_info['data_file']}")
        if self._preprocessing_config:
            L(f"  [ ] Preprocessing: {self._preprocessing_config}")
        if self._feature_names:
            L(f"  [ ] Features: {len(self._feature_names)} columns")
        for e in self._entries:
            if e['section'] == 'Training' and e.get('random_state') is not None:
                L(f"  [ ] {e['model_name']}: random_state={e['random_state']}")
            # v2.0: Suppress linked Optimization entries from checklist
            if e['section'] == 'Optimization' and e.get('random_state') is not None:
                algo_norm = self._normalize_name(e.get('algorithm', e.get('model_name', '')))
                if algo_norm not in linked_algos:
                    L(f"  [ ] {e['model_name']}: random_state={e['random_state']}, trials={e['n_trials']}")
            if e['section'] == 'Neural Network':
                cfg = e.get('training_config', {})
                rs = cfg.get('random_seed', cfg.get('random_state', 'N/A'))
                L(f"  [ ] {e['model_name']}: random_seed={rs}, epochs={cfg.get('epochs', 'N/A')}")
        L()

        L(sep)
        L("End of Report — AutoAIM Studio v2.0.0")
        L(sep)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Markdown renderer
    # ------------------------------------------------------------------

    def _render_markdown(self) -> str:
        lines = []
        lines.append("# AutoAIM Studio v2.0.0 — Session Report")
        lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")

        # Reproducibility
        lines.append("## Reproducibility Information")
        if self._hardware_info:
            lines.append(f"- **Hardware:** {self._hardware_info}")
        if self._preprocessing_config:
            lines.append("- **Preprocessing:**")
            for k, v in self._preprocessing_config.items():
                lines.append(f"  - `{k}`: `{v}`")
        lines.append("")

        # Dataset
        if self.session_info:
            lines.append("## Dataset")
            lines.append("| Property | Value |")
            lines.append("|----------|-------|")
            for k, v in self.session_info.items():
                lines.append(f"| {k.replace('_', ' ').title()} | {v} |")
            lines.append("")

        # v2.0: Filter out linked Optimization entries (same as TXT renderer)
        linked_algos = self._get_linked_algorithms()
        visible_entries = []
        for e in self._entries:
            if e['section'] == 'Optimization':
                algo_norm = self._normalize_name(e.get('algorithm', e.get('model_name', '')))
                if algo_norm in linked_algos:
                    continue
            visible_entries.append(e)

        # Entries
        for idx, entry in enumerate(visible_entries, 1):
            lines.append(f"## [{idx}] {entry['section']}: {entry['model_name']}")
            lines.append("")

            if entry['section'] == 'Training':
                lines.append(f"- **Model Type:** `{entry.get('model_type', 'N/A')}`")
                lines.append(f"- **Problem Type:** {entry['problem_type']}")
                lines.append(f"- **Random State:** {entry.get('random_state', 'N/A')}")
                cv_strat = entry.get('cv_strategy', 'kfold')
                lines.append(f"- **CV Strategy:** {'Leave-One-Out' if cv_strat == 'loo' else 'K-Fold'}")
                lines.append(f"- **Training Time:** {entry['training_time_sec']:.3f}s")
                mp = entry.get('model_params', {})
                if mp:
                    lines.append("- **Model Parameters:**")
                    for k, v in sorted(mp.items()):
                        lines.append(f"  - `{k}`: `{v}`")
                lines.append("")
                lines.append(_format_metrics_md(entry.get('metrics', {})))

                # v2.0: Linked optimization details
                linked_opt = entry.get('_linked_optimization')
                if linked_opt:
                    lines.append(_render_optimization_inline_md(linked_opt))

                cv_t = entry.get('cv_test_scores', [])
                if cv_t:
                    test_metric = entry.get('cv_test_metric', 'r2')
                    metric_name = test_metric.upper() if test_metric != 'r2' else 'Score'
                    # v2.0: full strategy label + grouping info + diagnostics
                    lines.append(f"\n**CV ({_cv_block_label(entry)}):** {', '.join(f'{s:.4f}' for s in cv_t)}")
                    group_info = entry.get('cv_group_info') or {}
                    if group_info.get('n_groups'):
                        lines.append(f"**Groups:** {group_info['n_groups']} "
                                     f"(sizes min={group_info.get('min_size')}, max={group_info.get('max_size')})")
                    for d in (entry.get('cv_diagnostics') or []):
                        lines.append(f"*⚠ {d}*")
                    r2_global = entry.get('cv_r2_global')
                    if r2_global is not None:
                        lines.append(f"**R² (global):** {r2_global:.4f}")
                    train_mean = entry.get('cv_train_mean')
                    test_mean = entry.get('cv_test_mean')
                    train_std = entry.get('cv_train_std')
                    test_std = entry.get('cv_test_std')
                    if train_mean is not None:
                        lines.append(f"**Train Mean:** {train_mean:.4f} ± {train_std:.4f}")
                    lines.append(f"**Test {metric_name} Mean:** {test_mean:.4f} ± {test_std:.4f}")
                fi = entry.get('feature_importance', {})
                if fi:
                    lines.append("\n### Top Features")
                    for feat, imp in fi.items():
                        lines.append(f"- `{feat}`: `{imp:.6f}`")

            elif entry['section'] == 'Optimization':
                # v2.0: Standalone Optimization (not linked). Linked ones filtered above.
                lines.append(f"- **Algorithm:** {entry['model_name']}")
                search_strat = entry.get('search_strategy', 'bayesian')
                lines.append(f"- **Search Strategy:** {'Bayesian (Optuna)' if search_strat == 'bayesian' else 'Grid Search'}")
                cv_strat = entry.get('cv_strategy', 'kfold')
                lines.append(f"- **CV Strategy:** {'Leave-One-Out' if cv_strat == 'loo' else 'K-Fold'}")
                lines.append(f"- **Trials:** {entry['n_trials']} (recorded: {entry['n_trials_recorded']})")
                lines.append(f"- **Random State:** {entry.get('random_state', 'N/A')}")
                lines.append(f"- **Best Score:** {_fmt(entry['best_score'])}")
                bp = entry.get('best_params', {})
                if bp:
                    lines.append("\n### Best Parameters")
                    for k, v in sorted(bp.items()):
                        lines.append(f"- `{k}`: `{v}`")
                lines.append("")
                lines.append(_format_metrics_md(entry.get('metrics', {})))
                trials = entry.get('all_trials', [])
                if trials:
                    lines.append(f"\n### Top 10 Trials")
                    sorted_t = sorted(trials, key=lambda t: t.get('value', -float('inf')), reverse=True)[:10]
                    lines.append("| Trial | Score | Key Params |")
                    lines.append("|-------|-------|------------|")
                    for t in sorted_t:
                        tid = t.get('number', '?')
                        score = _fmt(t.get('value', 'N/A'))
                        params = str(t.get('params', {}))[:40]
                        lines.append(f"| {tid} | {score} | {params} |")

            elif entry['section'] == 'Neural Network':
                lines.append(f"- **Architecture:** `{entry.get('architecture_desc', 'N/A')}`")
                detail = entry.get('architecture_detail', {})
                if detail:
                    lines.append("\n#### Layer Details")
                    for k, v in detail.items():
                        lines.append(f"- `{k}`: `{v}`")
                lines.append(f"- **Epochs Trained:** {entry['epochs_trained']}")
                if entry.get('best_epoch'):
                    lines.append(f"- **Best Epoch:** `{entry['best_epoch']}`")
                stats = entry.get('history_stats', {})
                if stats:
                    for k, v in stats.items():
                        lines.append(f"- **{k.replace('_', ' ').title()}:** `{v}`")
                # v2.0: Full training config with reproducibility fields
                cfg = entry.get('training_config', {})
                if cfg:
                    lines.append("\n#### Training Configuration")
                    repro_fields = ['random_seed', 'cv_folds', 'cv_strategy', 'problem_type',
                                    'epochs', 'batch_size', 'learning_rate',
                                    'optimizer', 'scheduler', 'weight_decay',
                                    'early_stopping_patience', 'early_stopping_min_delta',
                                    'gradient_clip_val']
                    shown = set()
                    for field in repro_fields:
                        if field in cfg:
                            lines.append(f"- `{field}`: `{cfg[field]}`")
                            shown.add(field)
                    for k, v in sorted(cfg.items()):
                        if k not in shown:
                            lines.append(f"- `{k}`: `{v}`")

                # v2.0: Linked optimization details (inline with trained NN)
                linked_opt = entry.get('_linked_optimization')
                if linked_opt:
                    lines.append(_render_optimization_inline_md(linked_opt))

                # NN optimization config
                nn_opt = entry.get('nn_optimization_config', {})
                if nn_opt:
                    lines.append("\n#### NN Optimization Configuration")
                    for k, v in sorted(nn_opt.items()):
                        if isinstance(v, dict):
                            lines.append(f"- `{k}`:")
                            for sk, sv in sorted(v.items()):
                                lines.append(f"  - `{sk}`: `{sv}`")
                        else:
                            lines.append(f"- `{k}`: `{v}`")

            elif entry['section'] == 'Ensemble':
                lines.append(f"- **Type:** {entry['ensemble_type']}")
                lines.append(f"- **Base Models:** {', '.join(entry['base_models'])}")
                lines.append("\n### Weights")
                for m, w in entry['weights'].items():
                    lines.append(f"- {m}: `{w:.4f}`")
                lines.append("")
                lines.append(_format_metrics_md(entry.get('metrics', {})))

            lines.append("")

        # Ranking
        ranking = self._build_ranking()
        if ranking:
            lines.append("## Final Ranking")
            lines.append("| Pos | Model | Section | R2/Acc | RMSE | Time(s) |")
            lines.append("|-----|-------|---------|--------|------|---------|")
            for i, row in enumerate(ranking, 1):
                marker = " **BEST**" if i == 1 else ""
                lines.append(
                    f"| {i} | {row['name']} | {row['section']} | "
                    f"{row['primary_metric']:.4f} | {row['rmse']:.4f} | {row['time']:.1f} |{marker}"
                )
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # CSV renderer
    # ------------------------------------------------------------------

    def _render_csv(self) -> str:
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["AutoAIM Studio v2.0.0 Session Report"])
        writer.writerow(["Generated", datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow([])

        if self.session_info:
            writer.writerow(["Dataset Info"])
            for k, v in self.session_info.items():
                writer.writerow([k, v])
            writer.writerow([])

        if self._preprocessing_config:
            writer.writerow(["Preprocessing"])
            for k, v in self._preprocessing_config.items():
                writer.writerow([k, v])
            writer.writerow([])

        writer.writerow(["#", "Section", "Model", "Problem Type", "Random State",
                         "Primary Metric", "RMSE", "MAE", "CV Test Mean", "CV Test Std",
                         "CV Folds", "Training Time (s)"])
        for idx, entry in enumerate(self._entries, 1):
            m = entry.get('metrics', {})
            writer.writerow([
                idx,
                entry['section'],
                entry['model_name'],
                entry.get('problem_type', ''),
                entry.get('random_state', ''),
                m.get('r2', m.get('accuracy', '')),
                m.get('rmse', ''),
                m.get('mae', ''),
                entry.get('cv_test_mean', ''),
                entry.get('cv_test_std', ''),
                entry.get('cv_n_folds', entry.get('cv_test_scores', []).__len__() if hasattr(entry.get('cv_test_scores', []), '__len__') else ''),
                f"{entry.get('training_time_sec', 0):.3f}",
            ])

        return output.getvalue()

    # ------------------------------------------------------------------
    # JSON renderer — complete data preservation
    # ------------------------------------------------------------------

    def _render_json(self) -> str:
        data = {
            "autoaim_version": "2.0.0",
            "generated_at": datetime.now().isoformat(),
            "session_info": self.session_info,
            "hardware_info": self._hardware_info,
            "preprocessing_config": self._preprocessing_config,
            "feature_names": self._feature_names,
            "entries": self._entries,
            "ranking": self._build_ranking(),
            "reproducibility_notes": {
                "all_random_states": _collect_all_random_states(self._entries),
                "total_entries": len(self._entries),
            },
        }
        return json.dumps(data, indent=2, default=_json_default)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_ranking(self) -> List[Dict[str, Any]]:
        # v2.0: Exclude standalone Optimization entries that are already
        # shown inline within their linked Training entry.
        linked_algos = self._get_linked_algorithms()
        rows = []
        for e in self._entries:
            if e['section'] == 'Optimization':
                algo_norm = self._normalize_name(e.get('algorithm', e.get('model_name', '')))
                if algo_norm in linked_algos:
                    continue
            m = e.get('metrics', {})
            # v2.0: For NNs, metrics may be empty — use val_loss-based proxy
            # (lower val_loss = better). If no standard metric, skip from ranking.
            primary = m.get('r2', m.get('accuracy', None))
            if primary is None and e['section'] == 'Neural Network':
                # Use negative val_loss as proxy (so higher = better for sorting)
                stats = e.get('history_stats', {})
                best_val = stats.get('best_val_loss')
                if best_val is not None and best_val > 0:
                    primary = -best_val  # Negative so lower loss = higher rank
            if primary is None:
                primary = 0
            rows.append({
                'name': e['model_name'],
                'section': e['section'],
                'primary_metric': float(primary),
                'rmse': float(m.get('rmse', 0) or 0),
                'time': float(e.get('training_time_sec', 0) or 0),
            })
        rows.sort(key=lambda x: x['primary_metric'], reverse=True)
        return rows


# ------------------------------------------------------------------
# Private helpers
# ------------------------------------------------------------------

def _safe_dict(d, top_n: Optional[int] = None) -> Dict:
    if d is None:
        return {}
    try:
        d = dict(d)
    except Exception:
        return {}
    if top_n:
        d = dict(sorted(d.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:top_n])
    return {k: v for k, v in d.items()}


def _safe_list(lst) -> List:
    if lst is None:
        return []
    try:
        return list(lst)
    except Exception:
        return []


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.6f}"
    return str(val)


def _extract_random_state(model) -> Any:
    """Extract random_state (or random_seed for CatBoost) from a model.

    v2.0: Falls back to 42 when the model doesn't expose random_state
    in get_params() (e.g., some XGBoost versions). Never returns None.
    """
    if model is None:
        return 42
    try:
        params = model.get_params()
        if 'random_seed' in params and params['random_seed'] is not None:
            return params['random_seed']
        if 'random_state' in params and params['random_state'] is not None:
            return params['random_state']
    except Exception:
        pass
    return 42  # v2.0: Never return None — 42 is the guaranteed default


def _extract_model_params(model) -> Dict[str, Any]:
    """Extract relevant parameters from a model for reproducibility.

    v2.0 FIX: When the model is a sklearn Pipeline, this filters out the
    internal preprocessor parameters (e.g. 'preprocessor__numeric__imputer',
    'preprocessor__numeric__scaler', 'steps', 'verbose', 'memory') that
    clutter the report and provide no value to the user. Only the actual
    estimator parameters (prefixed with 'model__' when inside a Pipeline)
    and Pipeline metadata are retained.
    """
    if model is None:
        return {}
    try:
        params = model.get_params()
        # v2.0: Filter out None values, internal sklearn objects, and
        # preprocessor pipeline noise that appears when Pipeline is used.
        def _is_noise(key: str, val) -> bool:
            if val is None or callable(val) or hasattr(val, 'fit'):
                return True
            # Pipeline structure noise (doesn't help user reproduce)
            if key in ('memory', 'steps', 'verbose'):
                return True
            # ColumnTransformer internals (imputer, scaler, encoder params)
            if key.startswith('preprocessor__'):
                return True
            # Model-saver metadata (not a hyperparameter)
            if key in ('force_alpha', 'n_iter_'):
                return True
            return False

        clean = {k: v for k, v in sorted(params.items()) if not _is_noise(k, v)}

        # v2.0: When inside a Pipeline, the model params are prefixed with
        # 'model__'. Strip that prefix for readability so the report shows
        # 'n_estimators: 100' instead of 'model__n_estimators: 100'.
        if hasattr(model, 'named_steps') and model.named_steps:
            stripped = {}
            for k, v in clean.items():
                if k.startswith('model__'):
                    stripped[k[7:]] = v  # strip 'model__' prefix
                else:
                    stripped[k] = v
            clean = stripped

        return clean
    except Exception:
        return {}


def _collect_all_random_states(entries: List[Dict]) -> Dict[str, Any]:
    """Collect all random states for the reproducibility checklist."""
    result = {}
    for e in entries:
        name = f"{e['section']}/{e['model_name']}"
        rs = e.get('random_state')
        if rs is not None:
            result[name] = rs
        # Also check training_config for NN
        cfg = e.get('training_config', {})
        if 'random_seed' in cfg:
            result[name] = cfg['random_seed']
    return result


def _render_metrics_txt(append_fn, metrics: Dict, indent: int = 0):
    prefix = "  " * (indent // 2)
    if not metrics:
        append_fn(f"{prefix}No metrics available.")
        return
    append_fn(f"{prefix}Metrics:")
    for k, v in metrics.items():
        if isinstance(v, float):
            append_fn(f"{prefix}  {k:<20} {v:.6f}")
        else:
            append_fn(f"{prefix}  {k:<20} {v}")


def _cv_block_label(entry: Dict) -> str:
    """Human-readable CV strategy label for the export header."""
    strat = entry.get('cv_strategy', 'kfold')
    cfg = entry.get('cv_config') or {}
    base = {
        'kfold': 'K-Fold',
        'stratified': 'Stratified K-Fold',
        'group': 'Group K-Fold',
        'loo': 'Leave-One-Out',
        'logo': 'Leave-One-Group-Out',
    }.get(strat, 'K-Fold')
    if strat == 'stratified' and cfg.get('n_bins'):
        base += f" ({cfg['n_bins']} quantile bins)"
    elif strat in ('group', 'logo'):
        src = cfg.get('group_source')
        src_label = {
            'chemical_system': 'chemical system',
            'formula': 'formula',
            'cluster': 'clustering',
            'column': f"column '{cfg.get('group_column')}'" if cfg.get('group_column') else 'column',
        }.get(src)
        if src_label:
            base += f" (by {src_label})"
    return base


def _render_cv_block_txt(append_fn, entry: Dict, indent: int = 2):
    """Render the cross-validation block (per-fold or per-unit) for TXT export.

    v2.0: Single renderer shared by the Training and NN sections. Handles all
    strategies (kfold / stratified / group / loo / logo), adds a Test Size
    column for uneven (grouped) folds, and surfaces any CV diagnostics
    (e.g. reduced quantile bins, OOF fallbacks) as warning lines.
    """
    test_scores = entry.get('cv_test_scores', [])
    if not test_scores:
        return
    train_scores = entry.get('cv_train_scores', [])
    test_metric = entry.get('cv_test_metric', 'r2')
    is_oof = test_metric == 'mae'
    strat = entry.get('cv_strategy', 'kfold')
    fold_sizes = entry.get('cv_fold_test_sizes') or []
    diagnostics = entry.get('cv_diagnostics') or []
    group_info = entry.get('cv_group_info') or {}
    r2_global = entry.get('cv_r2_global')

    pad = "  " * indent
    append_fn(f"{pad}Cross-Validation ({_cv_block_label(entry)}):")
    for d in diagnostics:
        append_fn(f"{pad}  ⚠ {d}")
    if group_info.get('n_groups'):
        append_fn(f"{pad}  Groups: {group_info['n_groups']} "
                  f"(sizes min={group_info.get('min_size')}, "
                  f"max={group_info.get('max_size')})")

    if is_oof:
        unit = "Group" if strat == 'logo' else "Sample"
        if r2_global is not None:
            append_fn(f"{pad}  R² (global): {r2_global:.6f}")
        append_fn(f"{pad}  Per-{unit.lower()} MAE:")
        for i, s in enumerate(test_scores, 1):
            append_fn(f"{pad}    {unit} {i:3d}: {s:.6f}")
        append_fn(f"{pad}    Mean MAE: {np.mean(test_scores):.6f}  "
                  f"Std MAE: {np.std(test_scores):.6f}")
        return

    metric_label = test_metric.upper() if test_metric != 'r2' else 'Score'
    show_size = bool(fold_sizes) and len(set(fold_sizes)) > 1
    if show_size:
        append_fn(f"{pad}  {'Fold':>6} | {'Train Score':>12} | {'Test ' + metric_label:>12} | {'Test Size':>9}")
        append_fn(f"{pad}  {'-'*6}-+-{'-'*12}-+-{'-'*12}-+-{'-'*9}")
    else:
        append_fn(f"{pad}  {'Fold':>6} | {'Train Score':>12} | {'Test ' + metric_label:>12}")
        append_fn(f"{pad}  {'-'*6}-+-{'-'*12}-+-{'-'*12}")
    for i, test_s in enumerate(test_scores, 1):
        train_s = train_scores[i-1] if i - 1 < len(train_scores) else None
        train_str = f"{train_s:12.6f}" if train_s is not None else "      N/A"
        row = f"{pad}  {i:6d} | {train_str} | {test_s:12.6f}"
        if show_size:
            size = fold_sizes[i-1] if i - 1 < len(fold_sizes) else None
            row += f" | {size:9d}" if size is not None else " |         —"
        append_fn(row)
    train_mean = entry.get('cv_train_mean')
    test_mean = entry.get('cv_test_mean')
    train_std = entry.get('cv_train_std')
    test_std = entry.get('cv_test_std')
    train_mean_str = f"{train_mean:12.6f}" if train_mean is not None else "         N/A"
    test_mean_str = f"{test_mean:12.6f}" if test_mean is not None else "         N/A"
    mean_row = f"{pad}  {'Mean':>6} | {train_mean_str} | {test_mean_str}"
    train_std_str = f"{train_std:12.6f}" if train_std is not None else "         N/A"
    test_std_str = f"{test_std:12.6f}" if test_std is not None else "         N/A"
    std_row = f"{pad}  {'Std':>6} | {train_std_str} | {test_std_str}"
    if show_size:
        mean_row += " |          "
        std_row += " |          "
    append_fn(mean_row)
    append_fn(std_row)


def _format_metrics_md(metrics: Dict) -> str:
    if not metrics:
        return "**No metrics available.**"
    lines = ["### Metrics"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"- **{k}:** `{v:.6f}`")
        else:
            lines.append(f"- **{k}:** `{v}`")
    return "\n".join(lines)


def _render_optimization_inline_txt(append_fn, opt_entry: Dict, indent: int = 2):
    """Render optimization details inline within a Training entry (TXT).

    v2.0: Shows only unique information NOT already present in the Training
    entry (model params, metrics, CV). Trials, config, and top trials are
    preserved since they are optimization-specific.
    """
    prefix = "  " * (indent // 2)
    sub_prefix = prefix + "  "

    append_fn(f"{prefix}[OPTIMIZATION DETAILS]")
    append_fn(f"{sub_prefix}Algorithm        : {opt_entry.get('algorithm', 'N/A')}")
    append_fn(f"{sub_prefix}Trials           : {opt_entry.get('n_trials', 'N/A')}")
    append_fn(f"{sub_prefix}Trials Recorded  : {opt_entry.get('n_trials_recorded', 'N/A')}")
    append_fn(f"{sub_prefix}Best Score       : {_fmt(opt_entry.get('best_score'))}")
    append_fn(f"{sub_prefix}Opt. Time        : {opt_entry.get('optimization_time_sec', 0):.3f}s")
    # v2.0: CV strategy used during optimization (full label) + grouping info
    append_fn(f"{sub_prefix}CV Strategy      : {_cv_block_label(opt_entry)}")
    _gi = opt_entry.get('cv_group_info') or {}
    if _gi.get('n_groups'):
        append_fn(f"{sub_prefix}CV Groups        : {_gi['n_groups']} "
                  f"(sizes min={_gi.get('min_size')}, max={_gi.get('max_size')})")
    for _d in (opt_entry.get('cv_diagnostics') or []):
        append_fn(f"{sub_prefix}  ⚠ {_d}")

    # Optimization configuration (unique to optimization)
    opt_cfg = opt_entry.get('optimization_config', {})
    if opt_cfg:
        append_fn(f"{sub_prefix}Configuration:")
        for k, v in sorted(opt_cfg.items()):
            if isinstance(v, dict):
                append_fn(f"{sub_prefix}  {k:<23} (dict with {len(v)} entries)")
                for sub_k, sub_v in sorted(v.items())[:10]:
                    append_fn(f"{sub_prefix}    {sub_k:<21} {sub_v}")
                if len(v) > 10:
                    append_fn(f"{sub_prefix}    ... and {len(v) - 10} more")
            else:
                append_fn(f"{sub_prefix}  {k:<23} {v}")

    # CV scores from optimization (may differ from training CV)
    test_scores = opt_entry.get('cv_test_scores', [])
    if test_scores:
        # NOTE: These are trial scores (one per trial), NOT fold scores.
        # Each value is the mean CV score across all folds for that trial.
        append_fn(f"{sub_prefix}CV Scores ({len(test_scores)} trials): " + ", ".join(f"{s:.4f}" for s in test_scores))

    # Trial history (unique to optimization)
    trials = opt_entry.get('all_trials', [])
    if trials:
        append_fn(f"{sub_prefix}Top 10 Trials by Score:")
        sorted_trials = sorted(
            trials, key=lambda t: t.get('value', t.get('score', -float('inf'))), reverse=True
        )[:10]
        append_fn(f"{sub_prefix}  {'Trial':>6} | {'Score':>12} | Parameters")
        for t in sorted_trials:
            tid = t.get('number', t.get('trial_id', '?'))
            score = _fmt(t.get('value', t.get('score', 'N/A')))
            params = str(t.get('params', {}))[:55]
            append_fn(f"{sub_prefix}  {tid:6} | {score:>12} | {params}")
    append_fn("")


def _render_optimization_inline_md(opt_entry: Dict) -> str:
    """Render optimization details inline within a Training entry (Markdown).

    v2.0: Returns a markdown string with optimization details.
    """
    lines = ["", "### Optimization Details"]
    lines.append(f"- **Algorithm:** {opt_entry.get('algorithm', 'N/A')}")
    lines.append(f"- **Trials:** {opt_entry.get('n_trials', 'N/A')}")
    lines.append(f"- **Best Score:** {_fmt(opt_entry.get('best_score'))}")
    lines.append(f"- **Opt. Time:** {opt_entry.get('optimization_time_sec', 0):.3f}s")
    # v2.0: CV strategy used during optimization (full label) + grouping info
    lines.append(f"- **CV Strategy:** {_cv_block_label(opt_entry)}")
    _gi = opt_entry.get('cv_group_info') or {}
    if _gi.get('n_groups'):
        lines.append(f"- **CV Groups:** {_gi['n_groups']} "
                     f"(sizes min={_gi.get('min_size')}, max={_gi.get('max_size')})")
    for _d in (opt_entry.get('cv_diagnostics') or []):
        lines.append(f"- *⚠ {_d}*")

    opt_cfg = opt_entry.get('optimization_config', {})
    if opt_cfg:
        lines.append("- **Configuration:**")
        for k, v in sorted(opt_cfg.items()):
            if isinstance(v, dict):
                lines.append(f"  - `{k}`: dict with {len(v)} entries")
            else:
                lines.append(f"  - `{k}`: `{v}`")

    bp = opt_entry.get('best_params', {})
    if bp:
        lines.append("- **Best Parameters:**")
        for k, v in sorted(bp.items()):
            lines.append(f"  - `{k}`: `{v}`")

    trials = opt_entry.get('all_trials', [])
    if trials:
        lines.append(f"- **Top 5 Trials:**")
        sorted_t = sorted(
            trials, key=lambda t: t.get('value', -float('inf')), reverse=True
        )[:5]
        lines.append("| Trial | Score | Params |")
        lines.append("|-------|-------|--------|")
        for t in sorted_t:
            tid = t.get('number', '?')
            score = _fmt(t.get('value', 'N/A'))
            params = str(t.get('params', {}))[:35]
            lines.append(f"| {tid} | {score} | {params} |")

    return "\n".join(lines)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)) and (np.isnan(obj) or np.isinf(obj)):
        return None
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")
