"""Background QThread orchestrating the full render pipeline: ingestion -> separation ->
vocal/instrumental neural+DSP chains -> remix/master/export, with per-stage progress and
cancellation support."""

from __future__ import annotations

import contextlib
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QThread, Signal

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import ingestion, instrumental_chain, neural_common, remix_master, separation, vocal_chain
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

        self._max_progress = 0.0
        self._progress_lock = threading.Lock()
        self._active_files: set[Path] = set()

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
            self._cleanup_partial_files()
            self.cancelled.emit()
        except Exception as exc:
            logger.exception("Render job failed for %s", self._input_path)
            self._cleanup_partial_files()
            self.failed.emit(str(exc))
        else:
            logger.info("Render job finished for %s -> %s", self._input_path, output_path)
            self.finished.emit(output_path)

    def _cleanup_partial_files(self) -> None:
        logger.info("Cleaning up partial files...")
        for path in list(self._active_files):
            try:
                if path.is_file():
                    path.unlink()
                    logger.info("Removed partial file: %s", path)
            except Exception as exc:
                logger.warning("Failed to remove partial file %s: %s", path, exc)

    def _emit_progress(self, val: float) -> None:
        with self._progress_lock:
            if val > self._max_progress:
                self._max_progress = val
                self.progressChanged.emit(val)

    @contextlib.contextmanager
    def _smooth_progress(self, stage_name: str, start_fraction: float, end_fraction: float, estimated_duration: float):
        stop_event = threading.Event()

        def estimator():
            start_time = time.time()
            weight = dict(_STAGE_WEIGHTS)[stage_name]
            stage_offset = self._stage_offsets[stage_name]
            start_progress = stage_offset + weight * start_fraction
            end_progress = stage_offset + weight * end_fraction

            while not stop_event.is_set() and not self.isInterruptionRequested():
                elapsed = time.time() - start_time
                if estimated_duration > 0:
                    t = elapsed / estimated_duration
                    fraction = min(0.95, 1.0 - np.exp(-1.5 * t))
                else:
                    fraction = 0.0

                current_progress = start_progress + (end_progress - start_progress) * fraction
                self._emit_progress(current_progress)
                time.sleep(0.1)

        thread = threading.Thread(target=estimator, daemon=True)
        thread.start()
        success = False
        try:
            yield
            success = True
        finally:
            stop_event.set()
            thread.join()
            if success:
                self._sub_progress(stage_name, end_fraction)

    def _neural_progress_callback_vocal(self, fraction: float) -> None:
        self._sub_progress("Denoising Vocal", 0.6 * fraction)

    def _neural_progress_callback_instrumental(self, fraction: float) -> None:
        self._sub_progress("Denoising Instrumental", 0.6 * fraction)

    def _render(self) -> Path:
        import torch
        preset = self._preset
        cache_manager = self._cache_manager

        self._enter_stage("Normalizing")
        track_id = cache_manager.compute_track_id(self._input_path)
        normalized_path = cache_manager.track_dir(track_id) / ingestion.NORMALIZED_FILENAME

        self._active_files.add(normalized_path)
        normalized_path = ingestion.load_and_normalize_track(self._input_path, cache_manager)
        self._active_files.discard(normalized_path)

        track_id = normalized_path.parent.name

        self._enter_stage("Separating")
        stems_dir = cache_manager.stems_dir(track_id)
        vocal_path = stems_dir / separation.VOCAL_FILENAME
        instrumental_path = stems_dir / separation.INSTRUMENTAL_FILENAME

        self._active_files.add(vocal_path)
        self._active_files.add(instrumental_path)

        duration = 6.0 if torch.cuda.is_available() else 30.0
        with self._smooth_progress("Separating", 0.0, 1.0, duration):
            vocal_stem_path, instrumental_stem_path = separation.separate_stems(normalized_path, cache_manager)

        self._active_files.discard(vocal_path)
        self._active_files.discard(instrumental_path)

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
        json_path = output_path.with_suffix(".json")

        self._active_files.add(output_path)
        self._active_files.add(json_path)

        exported_path = remix_master.export_wav(mastered, sample_rate, output_path)
        self._write_metadata(exported_path, track_id)

        self._active_files.clear()
        return exported_path

    def _write_metadata(self, render_path: Path, track_id: str) -> None:
        try:
            metadata = {
                "timestamp": datetime.now().isoformat(),
                "track_id": track_id,
                "render_file": render_path.name,
                "preset": asdict(self._preset),
            }
            json_path = render_path.with_suffix(".json")
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2)
            logger.info("Wrote render metadata to %s", json_path)
        except Exception as exc:
            logger.warning("Failed to write render metadata for %s: %s", render_path, exc)

    def _process_vocal(
        self,
        vocal_stem_path: Path,
        preset: Preset,
        cache_manager: CacheManager,
        track_id: str,
    ) -> tuple[np.ndarray, int]:
        import torch
        settings_hash = neural_common._hash_settings(
            preset.vocal_denoise_enabled,
            preset.vocal_denoise_intensity,
            preset.vocal_enhance_enabled,
            preset.vocal_enhance_intensity,
        )
        neural_vocal_path = cache_manager.stems_dir(track_id) / f"{vocal_chain.NEURAL_FILENAME_PREFIX}{settings_hash}.wav"

        self._active_files.add(neural_vocal_path)

        duration = 3.0 if torch.cuda.is_available() else 25.0
        with self._smooth_progress("Denoising Vocal", 0.0, 0.6, duration):
            try:
                neural_vocal_path = vocal_chain.run_neural_pass(
                    vocal_stem_path,
                    preset.vocal_denoise_enabled,
                    preset.vocal_denoise_intensity,
                    preset.vocal_enhance_enabled,
                    preset.vocal_enhance_intensity,
                    cache_manager,
                    progress_callback=self._neural_progress_callback_vocal,
                    is_cancelled=self.isInterruptionRequested,
                )
            except InterruptedError:
                raise _JobCancelled()

        self._active_files.discard(neural_vocal_path)
        self._checkpoint()

        dsp_vocal_path = cache_manager.stems_dir(track_id) / VOCAL_DSP_FILENAME
        self._active_files.add(dsp_vocal_path)
        vocal_chain.apply_dsp_chain(neural_vocal_path, preset.notch_depth_db, dsp_vocal_path)
        self._active_files.discard(dsp_vocal_path)

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
        import torch
        settings_hash = neural_common._hash_settings(
            preset.instrumental_denoise_enabled,
            preset.instrumental_denoise_intensity,
            preset.instrumental_enhance_enabled,
            preset.instrumental_enhance_intensity,
        )
        neural_instrumental_path = cache_manager.stems_dir(track_id) / f"{instrumental_chain.NEURAL_FILENAME_PREFIX}{settings_hash}.wav"

        self._active_files.add(neural_instrumental_path)

        duration = 3.0 if torch.cuda.is_available() else 25.0
        with self._smooth_progress("Denoising Instrumental", 0.0, 0.6, duration):
            try:
                neural_instrumental_path = instrumental_chain.run_neural_pass(
                    instrumental_stem_path,
                    preset.instrumental_denoise_enabled,
                    preset.instrumental_denoise_intensity,
                    preset.instrumental_enhance_enabled,
                    preset.instrumental_enhance_intensity,
                    cache_manager,
                    progress_callback=self._neural_progress_callback_instrumental,
                    is_cancelled=self.isInterruptionRequested,
                )
            except InterruptedError:
                raise _JobCancelled()

        self._active_files.discard(neural_instrumental_path)
        self._checkpoint()

        eq_params = InstrumentalEqParams(
            mud_cut_hz=preset.instrumental_mud_cut_hz,
            dehiss_shelf_hz=preset.instrumental_dehiss_shelf_hz,
            dehiss_gain_db=preset.instrumental_dehiss_gain_db,
        )
        dsp_instrumental_path = cache_manager.stems_dir(track_id) / INSTRUMENTAL_DSP_FILENAME
        self._active_files.add(dsp_instrumental_path)
        instrumental_chain.apply_dsp_chain(neural_instrumental_path, eq_params, dsp_instrumental_path)
        self._active_files.discard(dsp_instrumental_path)

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
        self._emit_progress(self._stage_offsets[name])

    def _sub_progress(self, stage_name: str, fraction_complete: float) -> None:
        weight = dict(_STAGE_WEIGHTS)[stage_name]
        self._emit_progress(self._stage_offsets[stage_name] + weight * fraction_complete)

    def _checkpoint(self) -> None:
        if self.isInterruptionRequested():
            raise _JobCancelled()
