"""Vocal stem pipeline: neural denoise pass (pre-DSP), adjustable Pedalboard DSP chain
(HPF/LPF/notch/de-esser), neural enhance pass (post-DSP, the last AI stage), and a
QA-gated capped residual blend of the enhance output back into the DSP signal."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, LowpassFilter, Pedalboard, PeakFilter

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import neural_common, qa_gate

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
    deesser_threshold_db = _clamp(deesser_threshold_db, DEESSER_THRESHOLD_MIN_DB, DEESSER_THRESHOLD_MAX_DB)

    audio, samplerate = sf.read(str(denoised_vocal_path), always_2d=True, dtype="float32")
    channels_first = audio.T

    tone_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=HPF_CUTOFF_HZ),
            LowpassFilter(cutoff_frequency_hz=LPF_CUTOFF_HZ),
            PeakFilter(cutoff_frequency_hz=NOTCH_CENTER_HZ, gain_db=-notch_depth_db, q=NOTCH_Q),
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


def _apply_deesser(channels_first_audio: np.ndarray, samplerate: int, threshold_db: float) -> np.ndarray:
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


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
