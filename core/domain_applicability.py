"""Domain applicability analysis for reliable predictions."""

import warnings
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, mahalanobis
from scipy.stats import f
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from ..utils.logger import LoggerMixin


@dataclass
class ApplicabilityMetrics:
    """Metrics for domain applicability."""
    leverage: float
    cook_distance: float
    mahalanobis_distance: float
    kNN_distance: float
    isolation_forest_score: float
    overall_score: float
    is_reliable: bool
    confidence_level: str


class DomainApplicabilityAnalyzer(LoggerMixin):
    """Analyze domain applicability for predictions."""
    
    def __init__(self):
        """Initialize analyzer."""
        self._X_train = None
        self._scaler = StandardScaler()
        self._pca = None
        self._nn_model = None
        self._iso_forest = None
        self._cov_matrix_inv = None
        self._mean_vector = None
        self._thresholds = {}
    
    def fit(self, X_train: np.ndarray, y_train: Optional[np.ndarray] = None) -> 'DomainApplicabilityAnalyzer':
        """Fit the analyzer on training data.
        
        Args:
            X_train: Training features
            y_train: Training targets (optional)
            
        Returns:
            Self
        """
        self._X_train = X_train.copy()
        
        # Scale data
        X_scaled = self._scaler.fit_transform(X_train)
        
        # Fit PCA for dimensionality reduction
        n_components = min(10, X_train.shape[1], X_train.shape[0] - 1)
        self._pca = PCA(n_components=n_components)
        X_pca = self._pca.fit_transform(X_scaled)
        
        # Fit k-NN model
        self._nn_model = NearestNeighbors(n_neighbors=min(5, len(X_train) - 1))
        self._nn_model.fit(X_pca)
        
        # Fit Isolation Forest
        self._iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1
        )
        self._iso_forest.fit(X_pca)
        
        # Compute covariance matrix for Mahalanobis distance
        self._mean_vector = np.mean(X_pca, axis=0)
        try:
            cov_matrix = np.cov(X_pca, rowvar=False)
            self._cov_matrix_inv = np.linalg.inv(cov_matrix)
        except np.linalg.LinAlgError:
            # Use pseudo-inverse if singular
            self._cov_matrix_inv = np.linalg.pinv(cov_matrix)
        
        # Compute thresholds on training data
        self._compute_thresholds(X_pca, y_train)
        
        self.logger.info("Domain applicability analyzer fitted")
        
        return self
    
    def _compute_thresholds(
        self,
        X_pca: np.ndarray,
        y_train: Optional[np.ndarray]
    ) -> None:
        """Compute thresholds for applicability metrics.
        
        Args:
            X_pca: PCA-transformed training data
            y_train: Training targets
        """
        # Leverage threshold (using hat matrix diagonal)
        n_samples, n_features = X_pca.shape
        
        # Hat matrix diagonal (leverage)
        try:
            hat_matrix = X_pca @ np.linalg.inv(X_pca.T @ X_pca) @ X_pca.T
            leverage_values = np.diag(hat_matrix)
        except np.linalg.LinAlgError:
            leverage_values = np.zeros(n_samples)
        
        # Threshold: 3 * average leverage
        self._thresholds['leverage'] = 3 * (n_features / n_samples)
        
        # k-NN distance threshold (95th percentile)
        distances, _ = self._nn_model.kneighbors(X_pca)
        avg_distances = distances.mean(axis=1)
        self._thresholds['knn_distance'] = np.percentile(avg_distances, 95)
        
        # Mahalanobis distance threshold (chi-square, p=0.975)
        from scipy.stats import chi2
        self._thresholds['mahalanobis'] = np.sqrt(chi2.ppf(0.975, df=X_pca.shape[1]))
        
        # Isolation Forest threshold
        iso_scores = self._iso_forest.decision_function(X_pca)
        self._thresholds['isolation_forest'] = np.percentile(iso_scores, 5)
    
    def analyze(
        self,
        X: np.ndarray,
        return_detailed: bool = False
    ) -> Union[np.ndarray, List[ApplicabilityMetrics]]:
        """Analyze domain applicability for new data.
        
        Args:
            X: Features to analyze
            return_detailed: Whether to return detailed metrics
            
        Returns:
            Applicability scores or detailed metrics
        """
        if self._X_train is None:
            raise ValueError("Analyzer must be fitted before analyzing")
        
        # Transform data
        X_scaled = self._scaler.transform(X)
        X_pca = self._pca.transform(X_scaled)
        
        metrics_list = []
        
        for i in range(len(X_pca)):
            x = X_pca[i:i+1]
            
            # Compute leverage
            leverage = self._compute_leverage(x, X_pca)
            
            # Compute Mahalanobis distance
            mahal_dist = self._compute_mahalanobis(x[0])
            
            # Compute k-NN distance
            knn_dist = self._compute_knn_distance(x)
            
            # Compute Isolation Forest score
            iso_score = self._iso_forest.decision_function(x)[0]
            
            # Compute overall score (normalized and combined)
            leverage_score = 1 - min(leverage / self._thresholds['leverage'], 1)
            mahal_score = 1 - min(mahal_dist / self._thresholds['mahalanobis'], 1)
            knn_score = 1 - min(knn_dist / self._thresholds['knn_distance'], 1)
            iso_score_norm = (iso_score - self._thresholds['isolation_forest']) / \
                           abs(self._thresholds['isolation_forest'])
            iso_score_norm = max(0, min(iso_score_norm, 1))
            
            # Overall score (weighted average)
            overall_score = 0.25 * leverage_score + 0.25 * mahal_score + \
                          0.25 * knn_score + 0.25 * iso_score_norm
            
            # Determine reliability
            is_reliable = (
                leverage < self._thresholds['leverage'] and
                mahal_dist < self._thresholds['mahalanobis'] and
                knn_dist < self._thresholds['knn_distance'] and
                iso_score > self._thresholds['isolation_forest']
            )
            
            # Confidence level
            if overall_score >= 0.8:
                confidence = 'High'
            elif overall_score >= 0.5:
                confidence = 'Medium'
            else:
                confidence = 'Low'
            
            metrics = ApplicabilityMetrics(
                leverage=leverage,
                cook_distance=0.0,  # Would need y values
                mahalanobis_distance=mahal_dist,
                kNN_distance=knn_dist,
                isolation_forest_score=iso_score,
                overall_score=overall_score,
                is_reliable=is_reliable,
                confidence_level=confidence
            )
            
            metrics_list.append(metrics)
        
        if return_detailed:
            return metrics_list
        else:
            return np.array([m.overall_score for m in metrics_list])
    
    def _compute_leverage(self, x: np.ndarray, X_ref: np.ndarray) -> float:
        """Compute leverage (hat matrix diagonal).
        
        Args:
            x: Single instance
            X_ref: Reference data
            
        Returns:
            Leverage value
        """
        try:
            # Simplified leverage computation
            x_centered = x - X_ref.mean(axis=0)
            leverage = np.sum(x_centered ** 2) / np.sum((X_ref - X_ref.mean(axis=0)) ** 2)
            return float(leverage)
        except Exception:
            return 0.0
    
    def _compute_mahalanobis(self, x: np.ndarray) -> float:
        """Compute Mahalanobis distance.
        
        Args:
            x: Single instance
            
        Returns:
            Mahalanobis distance
        """
        try:
            diff = x - self._mean_vector
            dist = np.sqrt(diff @ self._cov_matrix_inv @ diff)
            return float(dist)
        except Exception:
            return float('inf')
    
    def _compute_knn_distance(self, x: np.ndarray) -> float:
        """Compute average k-NN distance.
        
        Args:
            x: Single instance
            
        Returns:
            Average k-NN distance
        """
        try:
            distances, _ = self._nn_model.kneighbors(x)
            return float(distances.mean())
        except Exception:
            return float('inf')
    
    def get_applicability_domain_summary(self) -> Dict[str, Any]:
        """Get summary of applicability domain.
        
        Returns:
            Dictionary with summary information
        """
        if self._X_train is None:
            return {}
        
        return {
            'n_training_samples': len(self._X_train),
            'n_features_original': self._X_train.shape[1],
            'n_features_pca': self._pca.n_components_,
            'explained_variance_ratio': self._pca.explained_variance_ratio_.sum(),
            'thresholds': self._thresholds,
        }
    
    def plot_applicability_distribution(
        self,
        X_new: Optional[np.ndarray] = None,
        figsize: Tuple[int, int] = (12, 4)
    ):
        """Plot distribution of applicability metrics.
        
        Args:
            X_new: New data to compare (optional)
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        try:
            import matplotlib.pyplot as plt
            
            # Analyze training data
            train_metrics = self.analyze(self._X_train, return_detailed=True)
            
            # Analyze new data if provided
            new_metrics = None
            if X_new is not None:
                new_metrics = self.analyze(X_new, return_detailed=True)
            
            fig, axes = plt.subplots(1, 3, figsize=figsize)
            
            # Plot 1: Overall score distribution
            train_scores = [m.overall_score for m in train_metrics]
            axes[0].hist(train_scores, bins=30, alpha=0.7, label='Training')
            if new_metrics:
                new_scores = [m.overall_score for m in new_metrics]
                axes[0].hist(new_scores, bins=30, alpha=0.7, label='New')
            axes[0].axvline(0.5, color='r', linestyle='--', label='Threshold')
            axes[0].set_xlabel('Overall Applicability Score')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Applicability Score Distribution')
            axes[0].legend()
            
            # Plot 2: Mahalanobis distance
            train_mahal = [m.mahalanobis_distance for m in train_metrics]
            axes[1].hist(train_mahal, bins=30, alpha=0.7, label='Training')
            if new_metrics:
                new_mahal = [m.mahalanobis_distance for m in new_metrics]
                axes[1].hist(new_mahal, bins=30, alpha=0.7, label='New')
            axes[1].axvline(self._thresholds['mahalanobis'], color='r', linestyle='--')
            axes[1].set_xlabel('Mahalanobis Distance')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Mahalanobis Distance Distribution')
            axes[1].legend()
            
            # Plot 3: k-NN distance
            train_knn = [m.kNN_distance for m in train_metrics]
            axes[2].hist(train_knn, bins=30, alpha=0.7, label='Training')
            if new_metrics:
                new_knn = [m.kNN_distance for m in new_metrics]
                axes[2].hist(new_knn, bins=30, alpha=0.7, label='New')
            axes[2].axvline(self._thresholds['knn_distance'], color='r', linestyle='--')
            axes[2].set_xlabel('k-NN Distance')
            axes[2].set_ylabel('Frequency')
            axes[2].set_title('k-NN Distance Distribution')
            axes[2].legend()
            
            plt.tight_layout()
            return fig
            
        except Exception as e:
            self.logger.warning(f"Could not create plot: {e}")
            return None
    
    def filter_reliable_predictions(
        self,
        X: np.ndarray,
        predictions: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter predictions based on applicability.
        
        Args:
            X: Features
            predictions: Predictions
            threshold: Applicability threshold
            
        Returns:
            Tuple of (reliable_X, reliable_predictions, reliability_scores)
        """
        scores = self.analyze(X, return_detailed=False)
        
        reliable_mask = scores >= threshold
        
        return X[reliable_mask], predictions[reliable_mask], scores[reliable_mask]
