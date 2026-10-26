"""Out-of-Distribution (OOD) detector based on nearest-neighbour distance.

v2.0.0
------
This module implements the Applicability-Domain (AD) flag requested during
peer review: an *automatic* warning that fires when a query composition lies
outside the region covered by the training data in descriptor space.

Method
------
1. Standardise the training descriptors (zero-mean, unit-variance).
2. Fit a k-NN model (k = 1) on the standardised training set.
3. Compute the 1-NN distance for every training sample to itself.
4. Define the threshold as the 95-th percentile of those distances.
   The percentile is stored in the model bundle so the choice is
   transparent and reproducible.

For a new sample the detector:
* standardises it with the *same* scaler,
* queries the 1-NN distance,
* returns the distance and a boolean flag (distance > threshold).

Why this is scientifically defensible
-------------------------------------
* The distance is computed in the **raw feature space** (after
  standardisation), not in a PCA-transformed or kernel space.  This keeps
  the metric interpretable and matches the reviewer's request.
* The threshold is derived **entirely from the training set**; no
  hyper-parameter is tuned on external data.
* All parameters (percentile, n_neighbors, n_train_samples) are serialised
  in the model manifest so a third party can reproduce the exact boundary.

References
----------
* Section 4.4 of the AutoAIM manuscript (Yb and rare-element discussion).
* Standard cheminformatics AD practice (Netzeva et al., QSAR Comb. Sci.
  2005; Sheridan, J. Chem. Inf. Model. 2012).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import joblib
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class OodDetector:
    """Detect out-of-distribution samples via 1-NN distance in standardised space.

    Parameters
    ----------
    n_neighbors : int, default 1
        Number of neighbours used for the distance query.  k = 1 is the
        most conservative choice and matches the reviewer's wording
        "distance to the nearest neighbour".
    threshold_percentile : float, default 95.0
        Percentile of the training-set 1-NN distances used as the cutoff.
        95 % is a conventional choice in AD literature.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        threshold_percentile: float = 95.0,
    ) -> None:
        self.n_neighbors = n_neighbors
        self.threshold_percentile = threshold_percentile

        # Fitted attributes
        self.scaler_: Optional[StandardScaler] = None
        self.knn_: Optional[NearestNeighbors] = None
        self.threshold_: Optional[float] = None
        self.n_train_samples_: Optional[int] = None
        self.n_features_: Optional[int] = None

    # ------------------------------------------------------------------ #
    #  Fitting
    # ------------------------------------------------------------------ #
    def fit(self, X_train: Union[np.ndarray, Any]) -> "OodDetector":
        """Fit the detector on the training descriptors.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The feature matrix used to train the ML model.  pandas
            DataFrames are accepted and converted internally.
        """
        # Defensive conversion — same rationale as in nn_builder.py v2.0
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X_train must be 2-D, got shape {X.shape}")

        self.n_train_samples_, self.n_features_ = X.shape

        # 1. Standardise (zero-mean, unit-variance)
        self.scaler_ = StandardScaler()
        X_std = self.scaler_.fit_transform(X)

        # 2. Fit k-NN on the standardised training data
        self.knn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors + 1,  # +1 because the point itself is included
            metric="euclidean",
            algorithm="auto",
        )
        self.knn_.fit(X_std)

        # 3. Compute 1-NN distances *within* the training set.
        #    The nearest neighbour of a point is itself (distance 0), so we
        #    take the *second* nearest neighbour (index 1) as the true 1-NN.
        distances, _ = self.knn_.kneighbors(X_std)
        nn_distances = distances[:, 1]  # exclude self-distance

        # 4. Threshold = requested percentile of those distances
        self.threshold_ = float(
            np.percentile(nn_distances, self.threshold_percentile)
        )

        # Re-fit with k=1 only for faster queries at inference time
        self.knn_ = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric="euclidean",
            algorithm="auto",
        )
        self.knn_.fit(X_std)

        return self

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #
    def predict(
        self, X_new: Union[np.ndarray, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (distance, flag) for each new sample.

        Parameters
        ----------
        X_new : array-like of shape (n_samples, n_features)
            Query descriptors.  pandas DataFrames are accepted.

        Returns
        -------
        distances : ndarray of shape (n_samples,)
            Euclidean distance to the nearest training neighbour in
            standardised feature space.
        flags : ndarray of shape (n_samples,), dtype bool
            True  -> sample is outside the applicability domain
                     (distance > threshold).
            False -> sample lies inside the AD.
        """
        if self.scaler_ is None or self.knn_ is None:
            raise RuntimeError("OodDetector has not been fitted yet.")

        X = np.asarray(X_new, dtype=np.float64)
        if X.ndim != 2:
            raise ValueError(f"X_new must be 2-D, got shape {X.shape}")
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Feature dimension mismatch: expected {self.n_features_}, "
                f"got {X.shape[1]}"
            )

        X_std = self.scaler_.transform(X)
        distances, _ = self.knn_.kneighbors(X_std)
        # squeeze safely: (n,1) -> (n,) but keep (n,) shape when n==1
        distances = np.asarray(distances).reshape(-1).astype(float)
        flags = distances > self.threshold_

        return distances, flags

    # ------------------------------------------------------------------ #
    #  Metadata
    # ------------------------------------------------------------------ #
    def get_metadata(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dictionary describing the detector."""
        return {
            "ood_enabled": True,
            "ood_method": "knn_distance_standardized",
            "ood_n_neighbors": self.n_neighbors,
            "ood_threshold": self.threshold_,
            "ood_threshold_percentile": self.threshold_percentile,
            "ood_n_train_samples": self.n_train_samples_,
            "ood_n_features": self.n_features_,
        }

    # ------------------------------------------------------------------ #
    #  Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: Union[str, Path]) -> None:
        """Serialise the fitted detector to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Union[str, Path]) -> "OodDetector":
        """Load a previously saved detector."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"OOD detector not found: {path}")
        obj = joblib.load(path)
        if not isinstance(obj, OodDetector):
            raise TypeError(f"Loaded object is not an OodDetector: {type(obj)}")
        return obj


# ---------------------------------------------------------------------- #
#  Utility helpers used by the inference engine / GUI
# ---------------------------------------------------------------------- #
def format_ood_warning(
    distance: float,
    threshold: float,
    percentile: float = 95.0,
) -> str:
    """Return a human-readable warning string for an OOD sample."""
    ratio = distance / threshold if threshold > 0 else float("inf")
    return (
        f"Extrapolation detected (distance = {distance:.3f}, "
        f"threshold = {threshold:.3f}, ratio = {ratio:.2f}x). "
        f"This composition lies outside the applicability domain defined by "
        f"the {percentile:.0f}th percentile of training-set nearest-neighbour "
        f"distances. The prediction should be treated as an extrapolation and "
        f"interpreted with extreme caution."
    )
