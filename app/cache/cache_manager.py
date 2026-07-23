"""Owns the on-disk cache directory tree for tracks, presets, and logs."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from app.models.app_config import AppConfig, get_app_config


class CacheManager:
    """Creates and resolves paths within the app's local cache directory tree.

    Layout:
        cache/<track_id>/stems
        cache/<track_id>/renders
        cache/presets
        cache/logs
        cache/models
    """

    STEMS_DIRNAME = "stems"
    RENDERS_DIRNAME = "renders"
    PRESETS_DIRNAME = "presets"
    LOGS_DIRNAME = "logs"
    MODELS_DIRNAME = "models"

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or get_app_config()
        self._root = self._config.cache_root
        self._presets_dir = self._root / self.PRESETS_DIRNAME
        self._logs_dir = self._root / self.LOGS_DIRNAME
        self._models_dir = self._root / self.MODELS_DIRNAME
        self._ensure_dir(self._root)
        self._ensure_dir(self._presets_dir)
        self._ensure_dir(self._logs_dir)
        self._ensure_dir(self._models_dir)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def presets_dir(self) -> Path:
        return self._presets_dir

    @property
    def logs_dir(self) -> Path:
        return self._logs_dir

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def track_dir(self, track_id: str) -> Path:
        """Return (creating if needed) the root cache folder for a given track."""
        path = self._root / track_id
        self._ensure_dir(path)
        return path

    def stems_dir(self, track_id: str) -> Path:
        """Return (creating if needed) the stems folder for a given track."""
        path = self.track_dir(track_id) / self.STEMS_DIRNAME
        self._ensure_dir(path)
        return path

    def renders_dir(self, track_id: str) -> Path:
        """Return (creating if needed) the renders folder for a given track."""
        path = self.track_dir(track_id) / self.RENDERS_DIRNAME
        self._ensure_dir(path)
        return path

    @staticmethod
    def compute_track_id(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute a stable sha256 hash of a file's content, used as its track_id."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
