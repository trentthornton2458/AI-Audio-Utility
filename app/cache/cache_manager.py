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
    BIN_DIRNAME = "bin"

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self._config = config or get_app_config()
        self._root = self._config.cache_root
        self._presets_dir = self._root / self.PRESETS_DIRNAME
        self._logs_dir = self._root / self.LOGS_DIRNAME
        self._models_dir = self._root / self.MODELS_DIRNAME
        self._bin_dir = self._root / self.BIN_DIRNAME
        self._ensure_dir(self._root)
        self._ensure_dir(self._presets_dir)
        self._ensure_dir(self._logs_dir)
        self._ensure_dir(self._models_dir)
        self._ensure_dir(self._bin_dir)

    @property
    def bin_dir(self) -> Path:
        return self._bin_dir

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

    def clear_track_cache(self, track_id: str) -> None:
        """Delete all cached files (stems, renders, etc.) for a given track."""
        path = self._root / track_id
        if path.exists():
            import shutil
            shutil.rmtree(path)

    @staticmethod
    def compute_track_id(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute a stable sha256 hash of a file's content, used as its track_id."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def verify_stem_wav(self, path: Path) -> bool:
        """Verify the integrity of a cached stem WAV file.

        Checks:
        1. Path exists and is a file.
        2. File size is greater than 0.
        3. File is readable as a WAV using soundfile.
        """
        if not path.is_file():
            return False
        if path.stat().st_size == 0:
            return False
        try:
            import soundfile as sf
            sf.info(str(path))
            return True
        except Exception:
            return False

    def get_total_cache_size(self) -> int:
        """Return total size of the cache root in bytes."""
        total = 0
        if not self._root.exists():
            return 0
        for p in self._root.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except Exception:
                    pass
        return total

    def prune_cache(self, quota_bytes: int = 5 * 1024 * 1024 * 1024) -> None:
        """Prune old track render folders and logs when total cache usage exceeds quota_bytes.

        Candidate files to prune:
        - Files under <track_id>/renders for any track.
        - Files under the logs directory.

        Sorted by modification time (oldest first). We keep deleting them until the total size
        of the cache is <= quota_bytes or no candidates remain.
        """
        if self.get_total_cache_size() <= quota_bytes:
            return

        candidates = []

        # 1. Collect log files
        if self._logs_dir.is_dir():
            for p in self._logs_dir.rglob("*"):
                if p.is_file():
                    try:
                        candidates.append((p, p.stat().st_mtime))
                    except Exception:
                        pass

        # 2. Collect render files
        protected_dirs = {self.PRESETS_DIRNAME, self.LOGS_DIRNAME, self.MODELS_DIRNAME, self.BIN_DIRNAME}
        if self._root.is_dir():
            for track_dir in self._root.iterdir():
                if track_dir.is_dir() and track_dir.name not in protected_dirs:
                    renders_path = track_dir / self.RENDERS_DIRNAME
                    if renders_path.is_dir():
                        for p in renders_path.rglob("*"):
                            if p.is_file():
                                try:
                                    candidates.append((p, p.stat().st_mtime))
                                except Exception:
                                    pass

        # Sort candidate files by mtime (oldest first)
        candidates.sort(key=lambda item: item[1])

        # Delete candidates one by one until total size <= quota_bytes
        for file_path, _ in candidates:
            if self.get_total_cache_size() <= quota_bytes:
                break
            try:
                file_path.unlink()
                # If parent directory is empty and is a renders folder or under it, clean it up
                parent = file_path.parent
                while parent != self._root and parent != self._logs_dir:
                    if parent.is_dir() and not any(parent.iterdir()):
                        parent.rmdir()
                        parent = parent.parent
                    else:
                        break
            except Exception:
                pass

    @staticmethod
    def _ensure_dir(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)

