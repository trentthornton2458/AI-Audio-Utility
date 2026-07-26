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


# --- Rubberband CLI binary (native executable used by pyrubberband for pitch/time-stretching) ---
#
# This is a small native CLI tool, not a neural checkpoint, so it's provisioned separately from
# REQUIRED_MODEL_SPECS above and installed under CacheManager.bin_dir rather than models_dir. It's
# GPL-licensed, so it's fetched on demand here instead of bundled in the installer.
RUBBERBAND_BIN_FILENAME = "rubberband.exe"
RUBBERBAND_WINDOWS_URL = (
    "https://breakfastquay.com/files/releases/rubberband-3.3.0-gpl-executable-windows/rubberband.exe"
)
# TODO: pin to the real sha256 of the chosen release asset before shipping the Humanizer feature.
RUBBERBAND_WINDOWS_SHA256 = "0" * 64


class RubberbandDownloadError(Exception):
    """Raised when the rubberband CLI binary cannot be downloaded or fails checksum verification."""

    def __init__(self, reason: str, *, retryable: bool = True) -> None:
        self.reason = reason
        self.retryable = retryable
        super().__init__(f"Failed to download rubberband binary: {reason}")


class RubberbandBinaryNotFoundError(Exception):
    """Raised by get_rubberband_binary_path when the rubberband CLI hasn't been downloaded yet."""


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


def download_rubberband_binary(
    cache_manager: Optional[CacheManager] = None,
    progress_callback: Optional[Callable[[str, float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Download (or reuse) the rubberband CLI binary into cache_manager.bin_dir.

    Mirrors ModelDownloader's progress + checksum-verify flow, plus is_cancelled support so it
    can be driven from a cancellable background worker like other setup-wizard download steps.

    progress_callback, if given, is invoked with ("Rubberband CLI", fraction_complete) as bytes
    stream in, fraction_complete in [0.0, 1.0]. Raises RubberbandDownloadError on network
    failure, cancellation, or checksum mismatch.
    """
    cache_mgr = cache_manager or CacheManager()
    destination = cache_mgr.bin_dir / RUBBERBAND_BIN_FILENAME
    name = "Rubberband CLI"

    def _report(fraction: float) -> None:
        if progress_callback:
            progress_callback(name, fraction)

    if destination.is_file() and ModelDownloader._matches_checksum(destination, RUBBERBAND_WINDOWS_SHA256):
        logger.info("Rubberband binary already present and verified at %s; skipping download", destination)
        _report(1.0)
        return destination

    if is_cancelled and is_cancelled():
        raise RubberbandDownloadError("cancelled before download started", retryable=True)

    logger.info("Downloading rubberband binary from %s", RUBBERBAND_WINDOWS_URL)
    _report(0.0)

    tmp_path = destination.with_name(destination.name + PART_SUFFIX)
    request = urllib.request.Request(RUBBERBAND_WINDOWS_URL, headers={"User-Agent": "MusicMasteryEnhancer/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            total_bytes = ModelDownloader._content_length(response)
            downloaded_bytes = 0
            with open(tmp_path, "wb") as out_file:
                while True:
                    if is_cancelled and is_cancelled():
                        raise InterruptedError("Rubberband binary download cancelled")
                    chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded_bytes += len(chunk)
                    _report(ModelDownloader._fraction(downloaded_bytes, total_bytes))
    except InterruptedError as exc:
        ModelDownloader._cleanup(tmp_path)
        raise RubberbandDownloadError("cancelled during download", retryable=True) from exc
    except urllib.error.URLError as exc:
        ModelDownloader._cleanup(tmp_path)
        raise RubberbandDownloadError(f"network error while downloading: {exc}", retryable=True) from exc
    except OSError as exc:
        ModelDownloader._cleanup(tmp_path)
        raise RubberbandDownloadError(f"local I/O error while downloading: {exc}", retryable=True) from exc

    if not ModelDownloader._matches_checksum(tmp_path, RUBBERBAND_WINDOWS_SHA256):
        ModelDownloader._cleanup(tmp_path)
        raise RubberbandDownloadError(
            f"checksum mismatch after download (expected sha256={RUBBERBAND_WINDOWS_SHA256}); "
            "the file may be corrupt or the source may have changed",
            retryable=True,
        )

    shutil.move(str(tmp_path), str(destination))
    logger.info("Downloaded and verified rubberband binary -> %s", destination)
    _report(1.0)
    return destination


def get_rubberband_binary_path(cache_manager: Optional[CacheManager] = None) -> Path:
    """Locate the installed rubberband CLI binary for pyrubberband to invoke.

    Raises RubberbandBinaryNotFoundError if it hasn't been downloaded yet (e.g. the user skipped
    that step in the setup wizard), so callers get a clear, actionable error instead of
    pyrubberband failing deep inside a subprocess call.
    """
    cache_mgr = cache_manager or CacheManager()
    path = cache_mgr.bin_dir / RUBBERBAND_BIN_FILENAME
    if not path.is_file():
        raise RubberbandBinaryNotFoundError(
            f"rubberband binary not found at {path}; run the setup wizard's rubberband download "
            "step (or call download_rubberband_binary()) before using pitch-drift features"
        )
    return path
