"""Tests for the Out-of-Distribution (OOD) detector.

v2.0.0
------
The OOD detector is a peer-review requirement: it must flag samples that lie
outside the training domain in standardised feature space.  These tests verify:

1. Fitting produces a sensible threshold (percentile of training 1-NN distances).
2. In-domain samples receive ood_flag=False.
3. Far-away samples receive ood_flag=True.
4. Persistence (save/load) round-trips correctly.
5. Metadata contains all fields a reviewer would expect.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from app.core.ood_detector import OodDetector, format_ood_warning


# --------------------------------------------------------------------------- #
#  Fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def simple_train():
    """A small 2-D training cloud centred at the origin."""
    rng = np.random.RandomState(42)
    return rng.randn(50, 2)


@pytest.fixture
def fitted_detector(simple_train):
    """Pre-fitted detector for reuse across tests."""
    det = OodDetector(n_neighbors=1, threshold_percentile=95.0)
    det.fit(simple_train)
    return det


# --------------------------------------------------------------------------- #
#  Construction & fitting
# --------------------------------------------------------------------------- #
def test_detector_construction():
    det = OodDetector(n_neighbors=1, threshold_percentile=95.0)
    assert det.n_neighbors == 1
    assert det.threshold_percentile == 95.0
    assert det.scaler_ is None
    assert det.knn_ is None


def test_fit_sets_attributes(fitted_detector, simple_train):
    det = fitted_detector
    assert det.scaler_ is not None
    assert det.knn_ is not None
    assert det.threshold_ is not None
    assert det.n_train_samples_ == len(simple_train)
    assert det.n_features_ == simple_train.shape[1]
    # Threshold must be positive (distances are non-negative)
    assert det.threshold_ > 0


def test_fit_invalid_dimensions():
    det = OodDetector()
    with pytest.raises(ValueError, match="2-D"):
        det.fit(np.array([1.0, 2.0, 3.0]))  # 1-D input


# --------------------------------------------------------------------------- #
#  Prediction
# --------------------------------------------------------------------------- #
def test_predict_in_domain(fitted_detector, simple_train):
    """Training samples themselves (excluding exact duplicates) should be
    considered in-domain because the threshold is the 95th percentile."""
    det = fitted_detector
    # Slightly jittered copies of training points
    jittered = simple_train + np.random.RandomState(7).normal(0, 0.01, simple_train.shape)
    distances, flags = det.predict(jittered)
    assert distances.shape == (len(simple_train),)
    assert flags.shape == (len(simple_train),)
    assert flags.dtype == bool
    # Most should be in-domain (threshold is 95th percentile)
    assert flags.mean() < 0.2


def test_predict_out_of_domain(fitted_detector):
    """Points far from the training cloud must be flagged."""
    det = fitted_detector
    far_away = np.array([[10.0, 10.0], [-10.0, -10.0]])
    distances, flags = det.predict(far_away)
    assert flags.all(), "Far-away points should all be OOD"
    assert (distances > det.threshold_).all()


def test_predict_feature_mismatch(fitted_detector):
    """Predicting with wrong number of features must raise."""
    det = fitted_detector
    wrong_dim = np.array([[1.0, 2.0, 3.0]])  # 3 features instead of 2
    with pytest.raises(ValueError, match="Feature dimension mismatch"):
        det.predict(wrong_dim)


def test_predict_before_fit():
    det = OodDetector()
    with pytest.raises(RuntimeError, match="not been fitted"):
        det.predict(np.array([[1.0, 2.0]]))


# --------------------------------------------------------------------------- #
#  Persistence
# --------------------------------------------------------------------------- #
def test_save_load_roundtrip(fitted_detector):
    det = fitted_detector
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "ood.joblib"
        det.save(str(path))
        loaded = OodDetector.load(str(path))

    assert loaded.n_neighbors == det.n_neighbors
    assert loaded.threshold_percentile == det.threshold_percentile
    assert loaded.threshold_ == pytest.approx(det.threshold_)
    assert loaded.n_train_samples_ == det.n_train_samples_
    assert loaded.n_features_ == det.n_features_
    # Functional check
    distances, flags = loaded.predict(np.array([[10.0, 10.0]]))
    assert bool(flags[0]) is True


def test_load_missing_file():
    with pytest.raises(FileNotFoundError):
        OodDetector.load("/nonexistent/path/ood.joblib")


# --------------------------------------------------------------------------- #
#  Metadata
# --------------------------------------------------------------------------- #
def test_metadata_keys(fitted_detector):
    meta = fitted_detector.get_metadata()
    required_keys = {
        "ood_enabled",
        "ood_method",
        "ood_n_neighbors",
        "ood_threshold",
        "ood_threshold_percentile",
        "ood_n_train_samples",
        "ood_n_features",
    }
    assert required_keys.issubset(meta.keys())
    assert meta["ood_enabled"] is True
    assert meta["ood_method"] == "knn_distance_standardized"
    assert meta["ood_n_neighbors"] == 1
    assert meta["ood_threshold_percentile"] == 95.0


# --------------------------------------------------------------------------- #
#  Warning formatting
# --------------------------------------------------------------------------- #
def test_format_ood_warning():
    msg = format_ood_warning(distance=2.5, threshold=1.0, percentile=95.0)
    assert "Extrapolation detected" in msg
    assert "2.500" in msg
    assert "1.000" in msg
    assert "2.50x" in msg
    assert "applicability domain" in msg
    assert "95th percentile" in msg
