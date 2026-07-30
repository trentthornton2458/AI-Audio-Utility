"""Vocal stem pipeline: neural denoise pass (pre-DSP), adjustable Pedalboard DSP chain
(HPF/LPF/notch/de-esser), neural enhance pass (post-DSP, the last AI stage), a
QA-gated capped residual blend of the enhance output back into the DSP signal, and a
Humanizer pass (pitch drift + automatic breath blend-back) on that QA-gated output."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from pedalboard import (Compressor, HighpassFilter, LowpassFilter, PeakFilter,
                        Pedalboard)

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import humanizer, neural_common, qa_gate

logger = get_logger(__name__)

NEURAL_FILENAME_PREFIX = "vocal_neural_"
NEURAL_STEM_LABEL = "vocal"

DSP_SUBTYPE = "PCM_24"

HPF_CUTOFF_HZ = 80.0
LPF_CUTOFF_HZ = 14500.0

NOTCH_CENTER_HZ = 4000.0
NOTCH_Q = 1.0
NOTCH_DEPTH_MIN_DB = 3.0
NOTCH_DEPTH_MAX_DB = 6.0

DEESSER_BAND_LOW_HZ = 5000.0
DEESSER_BAND_HIGH_HZ = 8000.0
DEESSER_THRESHOLD_MIN_DB = -24.0
DEESSER_THRESHOLD_MAX_DB = 0.0
DEESSER_RATIO = 4.0
DEESSER_ATTACK_MS = 2.0
DEESSER_RELEASE_MS = 40.0

ENHANCE_INTENSITY_MAX = qa_gate.MAX_ENHANCE_GAIN


def run_denoise_pass(
    vocal_stem_path: Path,
    denoise_enabled: bool,
    denoise_intensity: float,
    cache_manager: CacheManager,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run resemble-enhance's denoise stage on an isolated vocal stem (pre-DSP).

    Delegates to app.core.neural_common.run_denoise_pass, which caches the result at
    cache/<track_id>/stems/vocal_neural_denoise_<settings_hash>.wav.
    """
    return neural_common.run_denoise_pass(
        vocal_stem_path,
        denoise_enabled,
        denoise_intensity,
        cache_manager,
        NEURAL_FILENAME_PREFIX,
        NEURAL_STEM_LABEL,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )


def run_enhance_pass(
    dsp_vocal_path: Path,
    enhance_enabled: bool,
    cache_manager: CacheManager,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run resemble-enhance's enhance stage (full wet, unblended) on the DSP-processed vocal
    stem -- the last AI stage in the pipeline. Returns dsp_vocal_path unchanged if disabled.

    Delegates to app.core.neural_common.run_enhance_pass, which caches the result at
    cache/<track_id>/stems/vocal_neural_enhance_<settings_hash>.wav.
    """
    return neural_common.run_enhance_pass(
        dsp_vocal_path,
        enhance_enabled,
        cache_manager,
        NEURAL_FILENAME_PREFIX,
        NEURAL_STEM_LABEL,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )


def apply_dsp_chain(
    denoised_vocal_path: Path,
    notch_depth_db: float,
    deesser_threshold_db: float,
    out_path: Path,
) -> Path:
    """Run the adjustable Pedalboard DSP chain on a (denoise-passed) vocal stem and write
    24-bit WAV.

    Chain: 80Hz highpass -> 14.5kHz lowpass -> adjustable 4kHz peak notch (harshness cut,
    notch_depth_db clamped to [3, 6] and applied as negative peak-filter gain) -> de-esser
    (deesser_threshold_db clamped to [-24, 0]).
    """
    notch_depth_db = _clamp(notch_depth_db, NOTCH_DEPTH_MIN_DB, NOTCH_DEPTH_MAX_DB)
    deesser_threshold_db = _clamp(
        deesser_threshold_db, DEESSER_THRESHOLD_MIN_DB, DEESSER_THRESHOLD_MAX_DB
    )

    audio, samplerate = sf.read(
        str(denoised_vocal_path), always_2d=True, dtype="float32"
    )
    channels_first = audio.T

    tone_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=HPF_CUTOFF_HZ),
            LowpassFilter(cutoff_frequency_hz=LPF_CUTOFF_HZ),
            PeakFilter(
                cutoff_frequency_hz=NOTCH_CENTER_HZ, gain_db=-notch_depth_db, q=NOTCH_Q
            ),
        ]
    )
    toned = tone_board(channels_first, samplerate)
    deessed = _apply_deesser(toned, samplerate, deesser_threshold_db)

    sf.write(str(out_path), deessed.T, samplerate, subtype=DSP_SUBTYPE)
    logger.info(
        "Wrote DSP vocal chain (notch=-%.1fdB, deesser=%.1fdB) for %s -> %s",
        notch_depth_db,
        deesser_threshold_db,
        denoised_vocal_path,
        out_path,
    )
    return out_path


def _apply_deesser(
    channels_first_audio: np.ndarray, samplerate: int, threshold_db: float
) -> np.ndarray:
    """De-ess the 5-8kHz sibilant band via split-band compression.

    Pedalboard's Compressor has no external sidechain input, so the band-limited copy of the
    signal produced by band_extract acts as both the detector and the audio that gets
    gain-reduced when it exceeds the threshold; the untouched rest of the spectrum (residual)
    is then added back in to reconstitute the full-band signal.
    """
    band_extract = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=DEESSER_BAND_LOW_HZ),
            LowpassFilter(cutoff_frequency_hz=DEESSER_BAND_HIGH_HZ),
        ]
    )
    sibilant_band = band_extract(channels_first_audio, samplerate)

    compressor = Pedalboard(
        [
            Compressor(
                threshold_db=threshold_db,
                ratio=DEESSER_RATIO,
                attack_ms=DEESSER_ATTACK_MS,
                release_ms=DEESSER_RELEASE_MS,
            )
        ]
    )
    compressed_band = compressor(sibilant_band, samplerate)

    residual = channels_first_audio - sibilant_band
    return residual + compressed_band


def blend_vocal(
    dsp_vocal_path: Path,
    enhanced_vocal_path: Path,
    enhance_intensity: float,
    gemini_api_key: Optional[str] = None,
) -> qa_gate.QAGateResult:
    """Blend the post-DSP enhance output back into the DSP signal via a QA-gated, hard-capped,
    energy-normalized parallel residual blend (never a linear crossfade -- that would rescale
    the dry signal and cause comb filtering/phase cancellation).

    enhance_intensity is clamped to [0.0, ENHANCE_INTENSITY_MAX] before being used as the
    gate's per-window gain cap. See app.core.qa_gate.apply_qa_gated_blend for the deterministic
    spectral QA gate (silence/clipping/hallucination fail-safes, per-window auto-attenuation)
    and optional Gemini diagnostic escalation.
    """
    enhance_intensity = _clamp(enhance_intensity, 0.0, ENHANCE_INTENSITY_MAX)
    return qa_gate.apply_qa_gated_blend(
        dsp_vocal_path,
        enhanced_vocal_path,
        max_gain=enhance_intensity,
        stem_label="vocal",
        gemini_api_key=gemini_api_key,
    )


def run_humanizer_pass(
    vocal_audio_path: Path,
    humanizer_intensity: float,
    residual_stem_path: Path,
    cache_manager: CacheManager,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run the Humanizer stage on the QA-gated blended vocal output -- the last vocal-only
    stage, run strictly after app.core.qa_gate's capped blend and strictly before remix/mastering.

    Applies app.core.humanizer.apply_pitch_drift (LFO micro-pitch drift, depth/rate scaled by
    humanizer_intensity, clamped to [0.0, 1.0]) followed by apply_breath_blend, which mixes the
    cached stem-separation residual at residual_stem_path back in at its fixed automatic amount
    (not user-controlled, per Counsel's spec).

    Cached at cache/<track_id>/stems/vocal_neural_humanize_<settings_hash>.wav, where
    settings_hash is derived from (humanizer_intensity, content-hash of vocal_audio_path) --
    like run_enhance_pass, this stage's cache key must track its input's *content* rather than
    just the intensity, since any upstream change (denoise/DSP/enhance/QA blend) needs to
    invalidate it. track_id is inferred from vocal_audio_path's location under the track's
    stems folder.
    """
    if is_cancelled and is_cancelled():
        raise InterruptedError("Humanizer pass cancelled")

    if progress_callback:
        progress_callback(0.0)

    humanizer_intensity = _clamp(humanizer_intensity, 0.0, 1.0)

    track_id = vocal_audio_path.parent.parent.name
    content_hash = CacheManager.compute_track_id(vocal_audio_path)
    settings_hash = _hash_humanizer_settings(humanizer_intensity, content_hash)
    output_path = (
        cache_manager.stems_dir(track_id)
        / f"{NEURAL_FILENAME_PREFIX}humanize_{settings_hash}.wav"
    )

    if cache_manager.verify_stem_wav(output_path):
        logger.info(
            "Using cached humanizer pass for %s stem of track %s: %s",
            NEURAL_STEM_LABEL,
            track_id,
            output_path,
        )
        if progress_callback:
            progress_callback(1.0)
        return output_path

    audio, samplerate = sf.read(str(vocal_audio_path), always_2d=True, dtype="float64")

    if is_cancelled and is_cancelled():
        raise InterruptedError("Humanizer pass cancelled")

    drifted = humanizer.apply_pitch_drift(audio, samplerate, humanizer_intensity)
    if progress_callback:
        progress_callback(0.5)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Humanizer pass cancelled")

    residual_audio, residual_samplerate = sf.read(
        str(residual_stem_path), always_2d=True, dtype="float64"
    )
    if residual_samplerate != samplerate:
        raise ValueError(
            f"Sample rate mismatch between vocal signal ({samplerate}Hz) "
            f"and residual stem ({residual_samplerate}Hz)"
        )
    humanized = humanizer.apply_breath_blend(drifted, residual_audio, samplerate)

    sf.write(str(output_path), humanized, samplerate, subtype=DSP_SUBTYPE)
    logger.info(
        "Wrote humanizer pass (intensity=%.2f) for %s stem of track %s -> %s",
        humanizer_intensity,
        NEURAL_STEM_LABEL,
        track_id,
        output_path,
    )
    if progress_callback:
        progress_callback(1.0)
    return output_path


def _hash_humanizer_settings(humanizer_intensity: float, content_hash: str) -> str:
    """Derive a short, stable hash identifying this humanizer-pass settings combination.

    Includes a content hash of the QA-blended vocal input (not just the intensity) since this
    stage runs after the QA gate: any upstream denoise/DSP/enhance/QA-blend change must
    invalidate this cache entry.
    """
    payload = f"{humanizer_intensity:.6f}|{content_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
