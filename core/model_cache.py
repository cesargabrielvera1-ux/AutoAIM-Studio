"""Disk-based model cache for memory-constrained environments.

v2.0.0
------
When a user trains dozens of models back-to-back, keeping every fitted
estimator (Pipeline, XGB trees, PyTorch weights, OOD detectors, etc.) in
RAM eventually exhausts memory and the OS kills the process.

ModelCache solves this by:

1. Serialising each trained model to disk immediately after training.
2. Replacing the in-memory model with ``None`` so Python can reclaim the
   memory via garbage collection.
3. Loading the model back from disk **only when needed** (export, inference,
   ensemble construction, session export).

The cache directory lives at ``~/.autoaim/model_cache/`` and is cleaned
automatically on startup (files older than 7 days are removed).  Each cached
model is a single ``joblib`` file (sklearn / XGB / etc.) or ``torch`` state
(dict + architecture) for neural networks.

Scientific reproducibility is preserved because:
* The cache path is stored in ``TrainingResult.model_cache_path``.
* The cache file name includes the model name and a timestamp.
* If the cache file is missing, the system falls back to the in-memory model
  (if still present) or logs a clear error.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union

import joblib
import numpy as np


class ModelCache:
    """Singleton-style disk cache for fitted ML models.

    Parameters
    ----------
    cache_dir : str or Path, optional
        Directory where cached models are stored.  Defaults to
        ``~/.autoaim/model_cache/``.
    max_age_days : int, default 7
        Maximum age of cache files before they are considered stale and
        removed during startup cleanup.
    """

    _instance: Optional["ModelCache"] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        cache_dir: Optional[Union[str, Path]] = None,
        max_age_days: int = 7,
    ) -> None:
        if self._initialized:
            return
        self._initialized = True

        if cache_dir is None:
            cache_dir = Path.home() / ".autoaim" / "model_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_age_days = max_age_days
        self._index_path = self.cache_dir / "cache_index.json"
        self._index: Dict[str, Dict[str, Any]] = {}

        # Cleanup stale entries on startup
        self._cleanup_stale()
        self._load_index()

    # ------------------------------------------------------------------ #
    #  Public API
    # ------------------------------------------------------------------ #
    def save(
        self,
        model: Any,
        model_name: str,
        is_neural_network: bool = False,
        nn_architecture: Optional[Dict] = None,
    ) -> Path:
        """Serialize a fitted model to disk and return the cache path.

        Parameters
        ----------
        model : fitted estimator or torch.nn.Module
            The trained model to cache.
        model_name : str
            Human-readable name (used in the filename).
        is_neural_network : bool, default False
            If True, uses ``torch.save()`` instead of ``joblib``.
        nn_architecture : dict, optional
            Architecture description required to reconstruct a PyTorch model.
            Only used when ``is_neural_network=True``.

        Returns
        -------
        path : Path
            Absolute path to the cached file.
        """
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        safe_name = self._sanitize_name(model_name)
        filename = f"{safe_name}_{ts}"

        if is_neural_network:
            path = self.cache_dir / f"{filename}.pt"
            self._save_pytorch(model, path, nn_architecture)
        else:
            path = self.cache_dir / f"{filename}.joblib"
            joblib.dump(model, path, compress=3)

        # Update index
        self._index[str(path)] = {
            "model_name": model_name,
            "created": datetime.now().isoformat(),
            "is_neural_network": is_neural_network,
            "size_mb": path.stat().st_size / (1024 * 1024),
        }
        self._save_index()

        return path

    def load(self, path: Union[str, Path], is_neural_network: bool = False) -> Any:
        """Load a model from its cache path.

        Parameters
        ----------
        path : str or Path
            The path returned by ``save()``.
        is_neural_network : bool, default False
            Must match the value passed to ``save()``.

        Returns
        -------
        model : fitted estimator or torch.nn.Module
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Cached model not found: {path}")

        if is_neural_network:
            return self._load_pytorch(path)
        return joblib.load(path)

    def remove(self, path: Union[str, Path]) -> None:
        """Remove a cached model file and its index entry."""
        path = Path(path)
        if path.exists():
            path.unlink()
        self._index.pop(str(path), None)
        self._save_index()

    def clear_all(self) -> None:
        """Remove **all** cached models.  Use with caution."""
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index = {}
        self._save_index()

    def get_stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        total_size = sum(
            p.stat().st_size for p in self.cache_dir.iterdir() if p.is_file()
        )
        return {
            "cache_dir": str(self.cache_dir),
            "entries": len(self._index),
            "total_size_mb": total_size / (1024 * 1024),
            "max_age_days": self.max_age_days,
        }

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #
    def _sanitize_name(self, name: str) -> str:
        """Make a string safe for use in a filename."""
        return "".join(c if c.isalnum() or c in "-_" else "_" for c in name)

    def _save_pytorch(self, model, path: Path, architecture: Optional[Dict]) -> None:
        """Save a PyTorch model (state_dict + architecture)."""
        import torch
        payload = {
            "state_dict": model.state_dict(),
            "architecture": architecture,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(payload, path)

    def _load_pytorch(self, path: Path) -> Any:
        """Load a PyTorch model from cache.  Returns a dict with state_dict."""
        import torch
        return torch.load(path, map_location="cpu", weights_only=False)

    def _cleanup_stale(self) -> None:
        """Remove cache files older than ``max_age_days``."""
        if not self.cache_dir.exists():
            return
        cutoff = datetime.now() - timedelta(days=self.max_age_days)
        removed = 0
        for fpath in self.cache_dir.iterdir():
            if fpath.is_file():
                mtime = datetime.fromtimestamp(fpath.stat().st_mtime)
                if mtime < cutoff:
                    fpath.unlink()
                    removed += 1
        if removed:
            warnings.warn(
                f"ModelCache: removed {removed} stale cache file(s) older than "
                f"{self.max_age_days} days."
            )

    def _load_index(self) -> None:
        """Load the in-memory index from disk."""
        if self._index_path.exists():
            with open(self._index_path, "r") as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        """Persist the index to disk."""
        with open(self._index_path, "w") as f:
            json.dump(self._index, f, indent=2, default=str)


# Convenience singleton accessor
_model_cache_instance: Optional[ModelCache] = None


def get_model_cache() -> ModelCache:
    """Return the global ModelCache singleton."""
    global _model_cache_instance
    if _model_cache_instance is None:
        _model_cache_instance = ModelCache()
    return _model_cache_instance
