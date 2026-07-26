"""Instrumental stem pipeline: neural denoise pass (pre-DSP), gentler adjustable Pedalboard DSP
chain (low-end mud cut + mild de-hiss high-shelf), neural enhance pass (post-DSP, the last AI
stage), and a QA-gated capped residual blend of the enhance output back into the DSP signal."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import soundfile as sf
from pedalboard import HighpassFilter, HighShelfFilter, Pedalboard

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import neural_common, qa_gate

logger = get_logger(__name__)

NEURAL_FILENAME_PREFIX = "instrumental_neural_"
NEURAL_STEM_LABEL = "instrumental"

DSP_SUBTYPE = "PCM_24"

MUD_CUT_HZ_DEFAULT = 40.0
MUD_CUT_HZ_MIN = 20.0
MUD_CUT_HZ_MAX = 120.0

DEHISS_SHELF_HZ_DEFAULT = 10000.0
DEHISS_SHELF_HZ_MIN = 6000.0
DEHISS_SHELF_HZ_MAX = 16000.0

DEHISS_GAIN_DB_DEFAULT = -3.0
DEHISS_GAIN_DB_MIN = -6.0
DEHISS_GAIN_DB_MAX = 0.0

ENHANCE_INTENSITY_MAX = qa_gate.MAX_ENHANCE_GAIN


@dataclass
class InstrumentalEqParams:
    """Individually adjustable knobs for the gentle instrumental DSP chain.

    mud_cut_hz: highpass cutoff trimming low-end mud, clamped to [MUD_CUT_HZ_MIN, MUD_CUT_HZ_MAX].
    dehiss_shelf_hz: high-shelf corner frequency for the mild de-hiss cut, clamped to
        [DEHISS_SHELF_HZ_MIN, DEHISS_SHELF_HZ_MAX].
    dehiss_gain_db: high-shelf gain (negative cuts hiss), clamped to
        [DEHISS_GAIN_DB_MIN, DEHISS_GAIN_DB_MAX].
    """

    mud_cut_hz: float = MUD_CUT_HZ_DEFAULT
    dehiss_shelf_hz: float = DEHISS_SHELF_HZ_DEFAULT
    dehiss_gain_db: float = DEHISS_GAIN_DB_DEFAULT


def run_denoise_pass(
    instrumental_stem_path: Path,
    denoise_enabled: bool,
    denoise_intensity: float,
    cache_manager: CacheManager,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run resemble-enhance's denoise stage on an isolated instrumental stem (pre-DSP).

    Delegates to app.core.neural_common.run_denoise_pass, which caches the result at
    cache/<track_id>/stems/instrumental_neural_denoise_<settings_hash>.wav.
    """
    return neural_common.run_denoise_pass(
        instrumental_stem_path,
        denoise_enabled,
        denoise_intensity,
        cache_manager,
        NEURAL_FILENAME_PREFIX,
        NEURAL_STEM_LABEL,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )


def run_enhance_pass(
    dsp_instrumental_path: Path,
    enhance_enabled: bool,
    cache_manager: CacheManager,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run resemble-enhance's enhance stage (full wet, unblended) on the DSP-processed
    instrumental stem -- the last AI stage in the pipeline. Returns dsp_instrumental_path
    unchanged if disabled.

    Delegates to app.core.neural_common.run_enhance_pass, which caches the result at
    cache/<track_id>/stems/instrumental_neural_enhance_<settings_hash>.wav.
    """
    return neural_common.run_enhance_pass(
        dsp_instrumental_path,
        enhance_enabled,
        cache_manager,
        NEURAL_FILENAME_PREFIX,
        NEURAL_STEM_LABEL,
        progress_callback=progress_callback,
        is_cancelled=is_cancelled,
    )


def apply_dsp_chain(denoised_instrumental_path: Path, eq_params: InstrumentalEqParams, out_path: Path) -> Path:
    """Run the gentle adjustable Pedalboard DSP chain on a (denoise-passed) instrumental stem.

    Chain: highpass at eq_params.mud_cut_hz (low-end mud cut) -> high-shelf at
    eq_params.dehiss_shelf_hz with eq_params.dehiss_gain_db gain (mild de-hiss). Unlike the
    vocal chain's fixed HPF/LPF, both cutoffs here are caller-adjustable (each clamped to its
    configured range) rather than hardcoded, since the instrumental chain is meant to run gentler
    and be tuned per track.
    """
    mud_cut_hz = _clamp(eq_params.mud_cut_hz, MUD_CUT_HZ_MIN, MUD_CUT_HZ_MAX)
    dehiss_shelf_hz = _clamp(eq_params.dehiss_shelf_hz, DEHISS_SHELF_HZ_MIN, DEHISS_SHELF_HZ_MAX)
    dehiss_gain_db = _clamp(eq_params.dehiss_gain_db, DEHISS_GAIN_DB_MIN, DEHISS_GAIN_DB_MAX)

    audio, samplerate = sf.read(str(denoised_instrumental_path), always_2d=True, dtype="float32")
    channels_first = audio.T

    board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=mud_cut_hz),
            HighShelfFilter(cutoff_frequency_hz=dehiss_shelf_hz, gain_db=dehiss_gain_db),
        ]
    )
    processed = board(channels_first, samplerate)

    sf.write(str(out_path), processed.T, samplerate, subtype=DSP_SUBTYPE)
    logger.info(
        "Wrote DSP instrumental chain (mud_cut=%.1fHz, dehiss=%.1fHz@%.1fdB) for %s -> %s",
        mud_cut_hz,
        dehiss_shelf_hz,
        dehiss_gain_db,
        denoised_instrumental_path,
        out_path,
    )
    return out_path


def blend_instrumental(
    dsp_instrumental_path: Path,
    enhanced_instrumental_path: Path,
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
        dsp_instrumental_path,
        enhanced_instrumental_path,
        max_gain=enhance_intensity,
        stem_label="instrumental",
        gemini_api_key=gemini_api_key,
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
