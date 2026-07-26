"""Background QThread running the pre-DSP Gemini stem-analysis QA checkpoint on the vocal and
instrumental stems concurrently, right after app.workers.separation_job.SeparationJob finishes.

Vocal and instrumental analysis are independent Gemini requests, so they run on their own
worker threads (via concurrent.futures.ThreadPoolExecutor) rather than one after the other --
each is a several-second network round trip, and running them serially would double that wait.
A failure on one stem does not block the other: analysisFinished always fires with whatever
succeeded, plus any per-stem error messages, so a transient network/API failure degrades to
"keep today's slider values" instead of blocking the render pipeline.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, QThread, Signal

from app.cache import get_logger
from app.core import gemini_qa

logger = get_logger(__name__)


class StemAnalysisJob(QThread):
    """Runs Gemini stem analysis on the vocal and instrumental stems for one track.

    Emits analysisFinished(vocal_updates, instrumental_updates, errors) on completion, where
    vocal_updates/instrumental_updates are {} for any stem whose analysis failed, and errors is
    a list of human-readable failure messages (empty if both succeeded).
    """

    analysisFinished = Signal(dict, dict, list)

    def __init__(
        self,
        vocal_path: Path,
        instrumental_path: Path,
        api_key: str,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._vocal_path = vocal_path
        self._instrumental_path = instrumental_path
        self._api_key = api_key

    def run(self) -> None:
        errors: list[str] = []

        with ThreadPoolExecutor(max_workers=2) as executor:
            vocal_future = executor.submit(gemini_qa.analyze_vocal_stem, self._vocal_path, self._api_key)
            instrumental_future = executor.submit(
                gemini_qa.analyze_instrumental_stem, self._instrumental_path, self._api_key
            )

            vocal_updates = self._resolve(vocal_future, "vocal", errors)
            instrumental_updates = self._resolve(instrumental_future, "instrumental", errors)

        logger.info(
            "Stem analysis finished (vocal_ok=%s, instrumental_ok=%s, errors=%s)",
            bool(vocal_updates),
            bool(instrumental_updates),
            errors,
        )
        self.analysisFinished.emit(vocal_updates, instrumental_updates, errors)

    @staticmethod
    def _resolve(future, stem_label: str, errors: list[str]) -> dict:
        try:
            return future.result()
        except gemini_qa.GeminiAnalysisError as exc:
            logger.warning("Gemini analysis failed for %s stem: %s", stem_label, exc)
            errors.append(f"{stem_label.capitalize()} stem analysis failed: {exc}")
            return {}
