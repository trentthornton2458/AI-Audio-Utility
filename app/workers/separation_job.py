"""Background QThread that runs ONLY ingestion + stem separation for the "Extract Stems"
action, without the (CPU-expensive) neural denoise/enhance, mix and master stages that the
full RenderJob performs.

Separation is the one stage users want to run on its own — the resemble-enhance neural passes
in the full pipeline can take many minutes per stem on a CPU-only machine, so wiring the
"Extract Stems" button to the full render made it look like separation itself had hung.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import ingestion, separation

logger = get_logger(__name__)


class SeparationJob(QThread):
    """Runs ingestion + BS-RoFormer stem separation for one track on a background thread.

    Emits stageChanged(str)/progressChanged(float) for status reporting, then either
    separationFinished(vocal_path, instrumental_path) on success, failed(str) on error,
    or cancelled() if cancellation was requested before separation started. Separation
    itself exposes no mid-call cancel hook, so a cancel requested after it begins takes
    effect only once the current separation returns.
    """

    stageChanged = Signal(str)
    progressChanged = Signal(float)
    separationFinished = Signal(Path, Path)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        input_path: Path,
        cache_manager: Optional[CacheManager] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._input_path = input_path
        self._cache_manager = cache_manager or CacheManager()

    def cancel(self) -> None:
        """Request cancellation; honored at the boundary before separation starts."""
        self.requestInterruption()

    def run(self) -> None:
        try:
            self.stageChanged.emit("Normalizing")
            self.progressChanged.emit(0.0)
            normalized_path = ingestion.load_and_normalize_track(self._input_path, self._cache_manager)

            if self.isInterruptionRequested():
                logger.info("Separation job cancelled for %s", self._input_path)
                self.cancelled.emit()
                return

            self.stageChanged.emit("Separating")
            self.progressChanged.emit(0.2)
            vocal_path, instrumental_path = separation.separate_stems(normalized_path, self._cache_manager)
        except Exception as exc:  # noqa: BLE001 - surfaced to the UI via failed()
            logger.exception("Separation job failed for %s", self._input_path)
            self.failed.emit(str(exc))
        else:
            self.progressChanged.emit(1.0)
            logger.info(
                "Separation job finished for %s -> %s, %s",
                self._input_path,
                vocal_path,
                instrumental_path,
            )
            self.separationFinished.emit(vocal_path, instrumental_path)
