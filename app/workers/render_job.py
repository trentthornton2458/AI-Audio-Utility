"""Background QThread orchestrating the full render pipeline: ingestion -> separation ->
vocal/instrumental neural+DSP chains -> remix/master/export, with per-stage progress and
cancellation support."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QThread, Signal

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import ingestion, instrumental_chain, remix_master, separation, vocal_chain
from app.core.instrumental_chain import InstrumentalEqParams
from app.models.preset import Preset

logger = get_logger(__name__)

VOCAL_DSP_FILENAME = "vocal_dsp.wav"
INSTRUMENTAL_DSP_FILENAME = "instrumental_dsp.wav"

# (stage name, fraction of total progress it accounts for); must sum to 1.0.
_STAGE_WEIGHTS: list[tuple[str, float]] = [
    ("Normalizing", 0.05),
    ("Separating", 0.15),
    ("Denoising Vocal", 0.30),
    ("Denoising Instrumental", 0.30),
    ("Mixing", 0.05),
    ("Mastering", 0.15),
]


class _JobCancelled(Exception):
    """Internal signal that a checkpoint detected a cancellation request."""


class RenderJob(QThread):
    """Runs one full render of a track against a Preset on a background thread.

    Neural passes (resemble-enhance) are cached per-stem by settings hash (see
    app.core.neural_common), so re-running a job with a Preset whose neural settings are
    unchanged from a previous render automatically skips straight to the DSP stage for that
    stem — no special-casing is needed here beyond simply calling the same run_neural_pass
    functions each time.
    """

    stageChanged = Signal(str)
    progressChanged = Signal(float)
    # Shadows QThread's own no-arg `finished` signal by design (spec calls for finished(Path));
    # QThread's automatic zero-arg completion emission still fires harmlessly alongside it.
    finished = Signal(Path)
    failed = Signal(str)
    cancelled = Signal()

    def __init__(
        self,
        input_path: Path,
        preset: Preset,
        cache_manager: Optional[CacheManager] = None,
        output_path: Optional[Path] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._input_path = input_path
        self._preset = preset
        self._cache_manager = cache_manager or CacheManager()
        self._output_path = output_path
        self._stage_offsets: dict[str, float] = {}

        offset = 0.0
        for name, weight in _STAGE_WEIGHTS:
            self._stage_offsets[name] = offset
            offset += weight

    def cancel(self) -> None:
        """Request cancellation. Checked at stage boundaries and between sub-steps within a
        stage; the neural/separation libraries themselves expose no mid-call cancel hook, so a
        checkpoint already in flight (e.g. mid resemble-enhance pass) will still run to completion
        before the next checkpoint stops the job."""
        self.requestInterruption()

    def run(self) -> None:
        try:
            output_path = self._render()
        except _JobCancelled:
            logger.info("Render job cancelled for %s", self._input_path)
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Render job failed for %s", self._input_path)
            self.failed.emit(str(exc))
        else:
            logger.info("Render job finished for %s -> %s", self._input_path, output_path)
            self.finished.emit(output_path)

    def _render(self) -> Path:
        preset = self._preset
        cache_manager = self._cache_manager

        self._enter_stage("Normalizing")
        normalized_path = ingestion.load_and_normalize_track(self._input_path, cache_manager)
        track_id = normalized_path.parent.name

        self._enter_stage("Separating")
        vocal_stem_path, instrumental_stem_path = separation.separate_stems(normalized_path, cache_manager)

        self._enter_stage("Denoising Vocal")
        vocal_audio, vocal_samplerate = self._process_vocal(vocal_stem_path, preset, cache_manager, track_id)

        self._enter_stage("Denoising Instrumental")
        instrumental_audio, instrumental_samplerate = self._process_instrumental(
            instrumental_stem_path, preset, cache_manager, track_id
        )

        if vocal_samplerate != instrumental_samplerate:
            raise RuntimeError(
                f"Sample rate mismatch between processed vocal ({vocal_samplerate}Hz) "
                f"and instrumental ({instrumental_samplerate}Hz)"
            )
        sample_rate = vocal_samplerate

        self._enter_stage("Mixing")
        mixed = remix_master.mix_stems(
            vocal_audio, instrumental_audio, preset.vocal_gain_db, preset.instrumental_gain_db
        )
        self._checkpoint()

        self._enter_stage("Mastering")
        mastered = remix_master.master(mixed, sample_rate, preset.lufs_target)
        self._checkpoint()

        output_path = self._resolve_output_path(track_id, cache_manager)
        return remix_master.export_wav(mastered, sample_rate, output_path)

    def _process_vocal(
        self,
        vocal_stem_path: Path,
        preset: Preset,
        cache_manager: CacheManager,
        track_id: str,
    ) -> tuple[np.ndarray, int]:
        neural_vocal_path = vocal_chain.run_neural_pass(
            vocal_stem_path,
            preset.vocal_denoise_enabled,
            preset.vocal_denoise_intensity,
            preset.vocal_enhance_enabled,
            preset.vocal_enhance_intensity,
            cache_manager,
        )
        self._sub_progress("Denoising Vocal", 0.6)
        self._checkpoint()

        dsp_vocal_path = cache_manager.stems_dir(track_id) / VOCAL_DSP_FILENAME
        vocal_chain.apply_dsp_chain(neural_vocal_path, preset.notch_depth_db, dsp_vocal_path)
        self._sub_progress("Denoising Vocal", 0.9)
        self._checkpoint()

        vocal_audio = vocal_chain.blend_vocal(neural_vocal_path, dsp_vocal_path, preset.vocal_clean_intensity)
        samplerate = sf.info(str(dsp_vocal_path)).samplerate
        return vocal_audio, samplerate

    def _process_instrumental(
        self,
        instrumental_stem_path: Path,
        preset: Preset,
        cache_manager: CacheManager,
        track_id: str,
    ) -> tuple[np.ndarray, int]:
        neural_instrumental_path = instrumental_chain.run_neural_pass(
            instrumental_stem_path,
            preset.instrumental_denoise_enabled,
            preset.instrumental_denoise_intensity,
            preset.instrumental_enhance_enabled,
            preset.instrumental_enhance_intensity,
            cache_manager,
        )
        self._sub_progress("Denoising Instrumental", 0.6)
        self._checkpoint()

        eq_params = InstrumentalEqParams(
            mud_cut_hz=preset.instrumental_mud_cut_hz,
            dehiss_shelf_hz=preset.instrumental_dehiss_shelf_hz,
            dehiss_gain_db=preset.instrumental_dehiss_gain_db,
        )
        dsp_instrumental_path = cache_manager.stems_dir(track_id) / INSTRUMENTAL_DSP_FILENAME
        instrumental_chain.apply_dsp_chain(neural_instrumental_path, eq_params, dsp_instrumental_path)
        self._sub_progress("Denoising Instrumental", 0.9)
        self._checkpoint()

        instrumental_audio, samplerate = sf.read(str(dsp_instrumental_path), always_2d=True, dtype="float64")
        return instrumental_audio, samplerate

    def _resolve_output_path(self, track_id: str, cache_manager: CacheManager) -> Path:
        if self._output_path is not None:
            return self._output_path
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        return cache_manager.renders_dir(track_id) / f"render_{timestamp}.wav"

    def _enter_stage(self, name: str) -> None:
        self._checkpoint()
        logger.info("Render stage: %s (%s)", name, self._input_path)
        self.stageChanged.emit(name)
        self.progressChanged.emit(self._stage_offsets[name])

    def _sub_progress(self, stage_name: str, fraction_complete: float) -> None:
        weight = dict(_STAGE_WEIGHTS)[stage_name]
        self.progressChanged.emit(self._stage_offsets[stage_name] + weight * fraction_complete)

    def _checkpoint(self) -> None:
        if self.isInterruptionRequested():
            raise _JobCancelled()
