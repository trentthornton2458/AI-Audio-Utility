"""Downloads and checksum-verifies the neural model weights required by the pipeline.

Runs during the guided first-run setup wizard so BS-RoFormer (stem separation) and
resemble-enhance (vocal/instrumental neural cleanup) weights are fetched with visible
progress up front, rather than lazily on first use.
"""

from __future__ import annotations

import hashlib
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

from app.cache import get_logger
from app.cache.cache_manager import CacheManager

logger = get_logger(__name__)

DOWNLOAD_CHUNK_SIZE = 1024 * 1024  # 1 MiB
REQUEST_TIMEOUT_SECONDS = 30
PART_SUFFIX = ".part"

# Must match app.core.separation.MODEL_FILENAME. Duplicated (rather than imported) so this
# module stays free of torch/audio-separator's heavy import chain during first-run setup.
BS_ROFORMER_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


class ModelDownloadError(Exception):
    """Raised when a required model file cannot be downloaded or fails checksum verification.

    Carries model_name/reason/retryable so the setup UI can show a specific message
    and offer a retry action instead of a generic failure dialog.
    """

    def __init__(self, model_name: str, reason: str, *, retryable: bool = True) -> None:
        self.model_name = model_name
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Failed to download model '{model_name}': {reason}")


@dataclass(frozen=True)
class ModelSpec:
    """Describes one downloadable model weight file."""

    name: str
    filename: str
    url: str
    sha256: str


# Models are natively auto-downloaded by audio-separator and resemble-enhance on first use,
# so we no longer manually pre-fetch them here. This avoids duplicating logic and managing
# their Hugging Face URLs/checksums.
REQUIRED_MODEL_SPECS: tuple[ModelSpec, ...] = ()


class ModelDownloader:
    """Downloads required neural model weights into cache_root/models with progress + checksum verification."""

    def __init__(self, cache_manager: Optional[CacheManager] = None) -> None:
        self._cache_manager = cache_manager or CacheManager()
        self._models_dir = self._cache_manager.models_dir

    @property
    def models_dir(self) -> Path:
        return self._models_dir

    def download_required_models(self, progress_callback: Callable[[str, float], None]) -> None:
        """Download (or reuse) every required model, reporting progress via progress_callback.

        progress_callback is invoked with (model_name, fraction_complete), fraction_complete
        in [0.0, 1.0], as each file streams down. Raises ModelDownloadError on network failure
        or checksum mismatch, with model_name/reason/retryable attached for the UI's retry flow.
        """
        for spec in REQUIRED_MODEL_SPECS:
            self._download_one(spec, progress_callback)

    def _download_one(self, spec: ModelSpec, progress_callback: Callable[[str, float], None]) -> Path:
        destination = self._models_dir / spec.filename

        if destination.is_file() and self._matches_checksum(destination, spec.sha256):
            logger.info("Model %s already present and verified at %s; skipping download", spec.name, destination)
            progress_callback(spec.name, 1.0)
            return destination

        logger.info("Downloading model %s from %s", spec.name, spec.url)
        progress_callback(spec.name, 0.0)

        tmp_path = destination.with_name(destination.name + PART_SUFFIX)
        try:
            self._stream_download(spec, tmp_path, progress_callback)
        except urllib.error.URLError as exc:
            self._cleanup(tmp_path)
            raise ModelDownloadError(
                spec.name, f"network error while downloading: {exc}", retryable=True
            ) from exc
        except OSError as exc:
            self._cleanup(tmp_path)
            raise ModelDownloadError(
                spec.name, f"local I/O error while downloading: {exc}", retryable=True
            ) from exc

        if not self._matches_checksum(tmp_path, spec.sha256):
            self._cleanup(tmp_path)
            raise ModelDownloadError(
                spec.name,
                f"checksum mismatch after download (expected sha256={spec.sha256}); "
                "the file may be corrupt or the source may have changed",
                retryable=True,
            )

        shutil.move(str(tmp_path), str(destination))
        logger.info("Downloaded and verified model %s -> %s", spec.name, destination)
        progress_callback(spec.name, 1.0)
        return destination

    def _stream_download(
        self, spec: ModelSpec, tmp_path: Path, progress_callback: Callable[[str, float], None]
    ) -> None:
        request = urllib.request.Request(spec.url, headers={"User-Agent": "MusicMasteryEnhancer/1.0"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            total_bytes = self._content_length(response)
            downloaded_bytes = 0

            with open(tmp_path, "wb") as out_file:
                while True:
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    progress_callback(spec.name, self._fraction(downloaded_bytes, total_bytes))

    @staticmethod
    def _content_length(response) -> Optional[int]:
        length = response.headers.get("Content-Length")
        try:
            return int(length) if length is not None else None
        except ValueError:
            return None

    @staticmethod
    def _fraction(downloaded_bytes: int, total_bytes: Optional[int]) -> float:
        if not total_bytes:
            return 0.0
        return min(downloaded_bytes / total_bytes, 1.0)

    @staticmethod
    def _matches_checksum(path: Path, expected_sha256: str) -> bool:
        hasher = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(DOWNLOAD_CHUNK_SIZE), b""):
                hasher.update(chunk)
        return hasher.hexdigest() == expected_sha256

    @staticmethod
    def _cleanup(path: Path) -> None:
        if path.is_file():
            path.unlink(missing_ok=True)
