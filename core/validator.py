"""Materials-aware validation strategies."""

import warnings
from typing import Iterator, List, Optional, Tuple, Union, Dict, Any
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold, StratifiedKFold, LeaveOneOut, LeavePOut,
    GroupKFold, StratifiedGroupKFold
)
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import KMeans

from ..utils.logger import LoggerMixin


class MaterialsAwareValidator(LoggerMixin):
    """Validation strategies for materials data."""
    
    def __init__(self):
        """Initialize validator."""
        self._strategies = {
            'kfold': 'Standard K-Fold CV',
            'stratified': 'Stratified K-Fold (for small datasets)',
            'loco': 'Leave-One-Cluster-Out (materials-aware)',
            'group': 'Group K-Fold (by chemical system)',
            'loo': 'Leave-One-Out (for very small datasets)',
        }
    
    @property
    def available_strategies(self) -> Dict[str, str]:
        """Get available validation strategies."""
        return self._strategies.copy()
    
    def get_cv_splitter(
        self,
        strategy: str,
        n_splits: int = 5,
        X: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        groups: Optional[np.ndarray] = None,
        composition_column: Optional[pd.Series] = None,
        random_state: int = 42
    ):
        """Get cross-validation splitter.
        
        Args:
            strategy: Validation strategy name
            n_splits: Number of splits
            X: Feature matrix
            y: Target vector
            groups: Group labels
            composition_column: Column with chemical compositions
            random_state: Random seed
            
        Returns:
            CV splitter object
        """
        from .cv_factory import CVConfig, build_cv_plan

        # Legacy validations (messages preserved for backward compatibility).
        if strategy == 'stratified' and y is None:
            raise ValueError("Stratified CV requires target variable")
        if strategy == 'loco' and composition_column is None and groups is None:
            raise ValueError("LOCO requires composition column or groups")
        if strategy == 'group' and groups is None and composition_column is None:
            raise ValueError("Group CV requires groups or composition column")
        if strategy not in self._strategies:
            raise ValueError(f"Unknown strategy: {strategy}")

        # v2.0: Delegate splitter construction to the shared cv_factory. This
        # fixes three latent defects of the previous implementation:
        #   * 'stratified' now ACTUALLY stratifies a regression target by
        #     quantile (previously it computed bins but discarded them,
        #     returning a plain StratifiedKFold unable to stratify a
        #     continuous y).
        #   * 'loco' now uses LeaveOneGroupOut instead of silently clamping
        #     n_splits to n_groups.
        #   * group labels come from grouping.groups_from_chemical_system,
        #     which reuses CompositionParser (no local regex).

        # Resolve group labels for grouped strategies.
        resolved_groups = groups
        if strategy in ('group', 'loco') and resolved_groups is None:
            resolved_groups = self._create_chemical_system_groups(composition_column)

        factory_strategy = {
            'kfold': 'kfold',
            'stratified': 'stratified',
            'group': 'group',
            'loco': 'logo',   # leave-one-cluster-out == LeaveOneGroupOut
            'loo': 'loo',
        }[strategy]

        # Representative X length (factory only needs len()).
        if X is not None:
            n = len(X)
        elif y is not None:
            n = len(y)
        elif resolved_groups is not None:
            n = len(resolved_groups)
        else:
            n = 0
        X_arr = np.asarray(X) if X is not None else np.zeros((n, 1))
        y_arr = np.asarray(y) if y is not None else np.zeros(n)

        config = CVConfig(
            strategy=factory_strategy,
            n_folds=n_splits,
            group_source='chemical_system' if resolved_groups is not None else None,
            groups=(np.asarray(resolved_groups) if resolved_groups is not None else None),
            random_state=random_state,
        )
        plan = build_cv_plan(config, X_arr, y_arr, problem_type='regression')
        return plan.splitter
    
    def _create_chemical_system_groups(self, compositions: pd.Series) -> np.ndarray:
        """Create groups based on chemical systems.
        
        Args:
            compositions: Series with chemical formulas
            
        Returns:
            Array of group labels
        """
        from .grouping import groups_from_chemical_system
        labels, _info = groups_from_chemical_system(compositions)
        # Convert string labels to stable integer codes (legacy return type).
        unique = list(dict.fromkeys(labels.tolist()))  # preserve first-seen order
        mapping = {g: i for i, g in enumerate(unique)}
        return np.array([mapping[g] for g in labels])
    
    def _parse_elements(self, formula: str) -> List[str]:
        """Parse elements from chemical formula.
        
        Args:
            formula: Chemical formula
            
            
Returns:
            List of element symbols
        """
        import re
        from .feature_engineering import CompositionParser

        if not isinstance(formula, str) or not formula.strip():
            return []

        # v2.0: validate against CompositionParser (single parsing source),
        # then preserve the legacy first-seen ordering of element symbols.
        frac = CompositionParser.parse_formula(formula)
        order = re.findall(r'([A-Z][a-z]?)', formula)

        seen = set()
        unique_elements = []
        for e in order:
            if e in frac and e not in seen:
                seen.add(e)
                unique_elements.append(e)

        return unique_elements
    
    def recommend_strategy(
        self,
        n_samples: int,
        n_classes: Optional[int] = None,
        has_compositions: bool = False
    ) -> str:
        """Recommend validation strategy based on dataset characteristics.
        
        Args:
            n_samples: Number of samples
            n_classes: Number of classes (for classification)
            has_compositions: Whether dataset has composition information
            
        Returns:
            Recommended strategy name
        """
        if n_samples < 50:
            return 'loo'
        elif n_samples < 1000:
            if has_compositions:
                return 'loco'
            elif n_classes is not None and n_classes > 1:
                return 'stratified'
            else:
                return 'kfold'
        else:
            if has_compositions:
                return 'loco'
            else:
                return 'kfold'
    
    def cross_validate_with_uncertainty(
        self,
        model: BaseEstimator,
        X: np.ndarray,
        y: np.ndarray,
        strategy: str = 'kfold',
        n_splits: int = 5,
        groups: Optional[np.ndarray] = None,
        composition_column: Optional[pd.Series] = None,
        return_predictions: bool = False,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Cross-validate with uncertainty estimation.
        
        Args:
            model: Model to validate
            X: Features
            y: Targets
            strategy: Validation strategy
            n_splits: Number of splits
            groups: Group labels
            composition_column: Composition column
            return_predictions: Whether to return out-of-fold predictions
            random_state: Random seed
            
        Returns:
            Dictionary with CV results
        """
        # Get CV splitter
        cv = self.get_cv_splitter(
            strategy, n_splits, X, y, groups, composition_column, random_state
        )
        
        # Store results
        fold_scores = []
        oof_predictions = np.zeros(len(y))
        fold_models = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
            self.logger.info(f"Training fold {fold_idx + 1}/{n_splits}...")

            # v2.0: X may be a DataFrame (from prepare_data). Use .iloc for
            # positional indexing so integer train/val indices work correctly.
            if hasattr(X, 'iloc'):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            else:
                X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Clone and train model
            fold_model = clone(model)
            fold_model.fit(X_train, y_train)
            
            # Predict
            y_pred = fold_model.predict(X_val)
            
            # Store predictions
            oof_predictions[val_idx] = y_pred
            
            # Compute score
            from sklearn.metrics import r2_score, mean_squared_error
            
            if len(np.unique(y)) <= 10:  # Classification
                from sklearn.metrics import accuracy_score
                score = accuracy_score(y_val, y_pred)
            else:  # Regression
                score = r2_score(y_val, y_pred)
            
            fold_scores.append(score)
            fold_models.append(fold_model)
        
        # Overall metrics
        from sklearn.metrics import r2_score, mean_squared_error
        
        overall_r2 = r2_score(y, oof_predictions)
        overall_rmse = np.sqrt(mean_squared_error(y, oof_predictions))
        
        results = {
            'fold_scores': fold_scores,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'overall_r2': overall_r2,
            'overall_rmse': overall_rmse,
            'fold_models': fold_models,
        }
        
        if return_predictions:
            results['oof_predictions'] = oof_predictions
        
        return results
    
    def cluster_based_split(
        self,
        X: np.ndarray,
        n_clusters: int = 5,
        random_state: int = 42
    ) -> np.ndarray:
        """Create train/test split based on feature space clustering.
        
        Args:
            X: Features
            n_clusters: Number of clusters
            random_state: Random seed
            
        Returns:
            Cluster labels
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        clusters = kmeans.fit_predict(X)
        
        return clusters
    
    def temporal_split(
        self,
        timestamps: pd.Series,
        train_fraction: float = 0.8
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create temporal train/test split.
        
        Args:
            timestamps: Series with timestamps
            train_fraction: Fraction for training
            
        Returns:
            Tuple of (train_indices, test_indices)
        """
        sorted_indices = timestamps.argsort()
        n_train = int(len(sorted_indices) * train_fraction)
        
        train_idx = sorted_indices[:n_train]
        test_idx = sorted_indices[n_train:]
        
        return train_idx.values, test_idx.values


class UncertaintyEstimator(LoggerMixin):
    """Estimate prediction uncertainty."""
    
    def __init__(self):
        """Initialize uncertainty estimator."""
        pass
    
    def estimate_from_cv(
        self,
        cv_results: Dict[str, Any],
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Estimate uncertainty from cross-validation results.
        
        Args:
            cv_results: Results from cross_validate_with_uncertainty
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with uncertainty estimates
        """
        oof_predictions = cv_results.get('oof_predictions')
        
        if oof_predictions is None:
            raise ValueError("CV results must include out-of-fold predictions")
        
        # Compute prediction variance across folds
        fold_models = cv_results.get('fold_models', [])
        
        if not fold_models:
            return {
                'mean': oof_predictions,
                'std': np.zeros_like(oof_predictions),
                'ci_lower': oof_predictions,
                'ci_upper': oof_predictions,
            }
        
        # For ensemble uncertainty, we'd need predictions from all models
        # This is a simplified version
        
        # Use residual-based uncertainty
        residuals = np.abs(cv_results.get('y_true', oof_predictions) - oof_predictions)
        mean_residual = np.mean(residuals)
        
        uncertainty = np.full_like(oof_predictions, mean_residual)
        
        # Compute confidence intervals
        from scipy import stats
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        ci_lower = oof_predictions - z_score * uncertainty
        ci_upper = oof_predictions + z_score * uncertainty
        
        return {
            'mean': oof_predictions,
            'std': uncertainty,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'confidence_level': confidence_level,
        }
    
    def bootstrap_uncertainty(
        self,
        model: BaseEstimator,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        n_bootstrap: int = 100,
        confidence_level: float = 0.95
    ) -> Dict[str, np.ndarray]:
        """Estimate uncertainty using bootstrap.
        
        Args:
            model: Base model
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level
            
        Returns:
            Dictionary with uncertainty estimates
        """
        predictions = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
            X_boot = X_train[indices]
            y_boot = y_train[indices]
            
            # Train model
            boot_model = clone(model)
            boot_model.fit(X_boot, y_boot)
            
            # Predict
            y_pred = boot_model.predict(X_test)
            predictions.append(y_pred)
        
        predictions = np.array(predictions)
        
        # Compute statistics
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        
        # Confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(predictions, alpha/2 * 100, axis=0)
        ci_upper = np.percentile(predictions, (1 - alpha/2) * 100, axis=0)
        
        return {
            'mean': mean_pred,
            'std': std_pred,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'all_predictions': predictions,
        }
