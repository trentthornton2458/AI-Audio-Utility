"""Neural denoise/enhance pass on the isolated instrumental stem via resemble-enhance, cached by
settings hash, plus a gentler adjustable Pedalboard DSP chain (low-end mud cut + mild de-hiss
high-shelf) for the instrumental stem."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import HighpassFilter, HighShelfFilter, Pedalboard

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import neural_common

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


def run_neural_pass(
    instrumental_stem_path: Path,
    denoise_enabled: bool,
    denoise_intensity: float,
    enhance_enabled: bool,
    enhance_intensity: float,
    cache_manager: CacheManager,
) -> Path:
    """Run resemble-enhance's denoise and/or enhance stages on an isolated instrumental stem.

    Delegates to app.core.neural_common.run_neural_pass, which caches the result at
    cache/<track_id>/stems/instrumental_neural_<settings_hash>.wav; see that module for the
    caching, per-channel processing, and dry/wet blending behavior.
    """
    return neural_common.run_neural_pass(
        instrumental_stem_path,
        denoise_enabled,
        denoise_intensity,
        enhance_enabled,
        enhance_intensity,
        cache_manager,
        NEURAL_FILENAME_PREFIX,
        NEURAL_STEM_LABEL,
    )


def apply_dsp_chain(neural_instrumental_path: Path, eq_params: InstrumentalEqParams, out_path: Path) -> Path:
    """Run the gentle adjustable Pedalboard DSP chain on a (neural-passed) instrumental stem.

    Chain: highpass at eq_params.mud_cut_hz (low-end mud cut) -> high-shelf at
    eq_params.dehiss_shelf_hz with eq_params.dehiss_gain_db gain (mild de-hiss). Unlike the
    vocal chain's fixed HPF/LPF, both cutoffs here are caller-adjustable (each clamped to its
    configured range) rather than hardcoded, since the instrumental chain is meant to run gentler
    and be tuned per track.
    """
    audio, samplerate = sf.read(str(neural_instrumental_path), always_2d=True, dtype="float32")
    channels_first = audio.T

    # Sanitize input: replace any NaN or Inf with 0.0 to prevent DSP instability
    channels_first = np.nan_to_num(channels_first, nan=0.0, posinf=0.0, neginf=0.0)

    # To ensure smooth frequency response transitions and prevent digital filter instability,
    # the cutoff frequencies must not exceed a safe margin below the Nyquist frequency.
    # We define a safe limit at 49% of the sample rate.
    safe_cutoff_limit = 0.49 * samplerate

    # Clamping range dynamically configured so safe limit takes precedence if Nyquist is extremely low
    max_mud_cut = min(MUD_CUT_HZ_MAX, safe_cutoff_limit)
    min_mud_cut = min(MUD_CUT_HZ_MIN, max_mud_cut)
    mud_cut_hz = _clamp(eq_params.mud_cut_hz, min_mud_cut, max_mud_cut)

    max_dehiss = min(DEHISS_SHELF_HZ_MAX, safe_cutoff_limit)
    min_dehiss = min(DEHISS_SHELF_HZ_MIN, max_dehiss)
    dehiss_shelf_hz = _clamp(eq_params.dehiss_shelf_hz, min_dehiss, max_dehiss)

    # Ensure mud_cut_hz doesn't exceed dehiss_shelf_hz for logical filter ordering
    if mud_cut_hz > dehiss_shelf_hz:
        mud_cut_hz = dehiss_shelf_hz

    dehiss_gain_db = _clamp(eq_params.dehiss_gain_db, DEHISS_GAIN_DB_MIN, DEHISS_GAIN_DB_MAX)

    board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=mud_cut_hz),
            HighShelfFilter(cutoff_frequency_hz=dehiss_shelf_hz, gain_db=dehiss_gain_db),
        ]
    )
    processed = board(channels_first, samplerate)

    # Sanitize output: replace any NaN or Inf with 0.0 to prevent file corruption
    processed = np.nan_to_num(processed, nan=0.0, posinf=0.0, neginf=0.0)

    sf.write(str(out_path), processed.T, samplerate, subtype=DSP_SUBTYPE)
    logger.info(
        "Wrote DSP instrumental chain (mud_cut=%.1fHz, dehiss=%.1fHz@%.1fdB) for %s -> %s",
        mud_cut_hz,
        dehiss_shelf_hz,
        dehiss_gain_db,
        neural_instrumental_path,
        out_path,
    )
    return out_path


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
