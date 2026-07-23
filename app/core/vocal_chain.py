"""Neural denoise/enhance pass on the isolated vocal stem via resemble-enhance, cached by settings hash,
plus the adjustable Pedalboard DSP chain (HPF/LPF/notch/de-esser) and neural/DSP vocal blend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from pedalboard import Compressor, HighpassFilter, LowpassFilter, Pedalboard, PeakFilter

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import neural_common

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
DEESSER_THRESHOLD_DB = -24.0
DEESSER_RATIO = 4.0
DEESSER_ATTACK_MS = 2.0
DEESSER_RELEASE_MS = 40.0


def run_neural_pass(
    vocal_stem_path: Path,
    denoise_enabled: bool,
    denoise_intensity: float,
    enhance_enabled: bool,
    enhance_intensity: float,
    cache_manager: CacheManager,
) -> Path:
    """Run resemble-enhance's denoise and/or enhance stages on an isolated vocal stem.

    Delegates to app.core.neural_common.run_neural_pass, which caches the result at
    cache/<track_id>/stems/vocal_neural_<settings_hash>.wav; see that module for the
    caching, per-channel processing, and dry/wet blending behavior.
    """
    return neural_common.run_neural_pass(
        vocal_stem_path,
        denoise_enabled,
        denoise_intensity,
        enhance_enabled,
        enhance_intensity,
        cache_manager,
        NEURAL_FILENAME_PREFIX,
        NEURAL_STEM_LABEL,
    )


def apply_dsp_chain(neural_vocal_path: Path, notch_depth_db: float, out_path: Path) -> Path:
    """Run the adjustable Pedalboard DSP chain on a (neural-passed) vocal stem and write 24-bit WAV.

    Chain: 80Hz highpass -> 14.5kHz lowpass -> adjustable 4kHz peak notch (harshness cut,
    notch_depth_db clamped to [3, 6] and applied as negative peak-filter gain) -> de-esser.
    """
    notch_depth_db = _clamp(notch_depth_db, NOTCH_DEPTH_MIN_DB, NOTCH_DEPTH_MAX_DB)

    audio, samplerate = sf.read(str(neural_vocal_path), always_2d=True, dtype="float32")
    channels_first = audio.T

    tone_board = Pedalboard(
        [
            HighpassFilter(cutoff_frequency_hz=HPF_CUTOFF_HZ),
            LowpassFilter(cutoff_frequency_hz=LPF_CUTOFF_HZ),
            PeakFilter(cutoff_frequency_hz=NOTCH_CENTER_HZ, gain_db=-notch_depth_db, q=NOTCH_Q),
        ]
    )
    toned = tone_board(channels_first, samplerate)
    deessed = _apply_deesser(toned, samplerate)

    sf.write(str(out_path), deessed.T, samplerate, subtype=DSP_SUBTYPE)
    logger.info(
        "Wrote DSP vocal chain (notch=-%.1fdB) for %s -> %s", notch_depth_db, neural_vocal_path, out_path
    )
    return out_path


def _apply_deesser(channels_first_audio: np.ndarray, samplerate: int) -> np.ndarray:
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
                threshold_db=DEESSER_THRESHOLD_DB,
                ratio=DEESSER_RATIO,
                attack_ms=DEESSER_ATTACK_MS,
                release_ms=DEESSER_RELEASE_MS,
            )
        ]
    )
    compressed_band = compressor(sibilant_band, samplerate)

    residual = channels_first_audio - sibilant_band
    return residual + compressed_band


def blend_vocal(neural_vocal_path: Path, dsp_vocal_path: Path, clean_intensity: float) -> np.ndarray:
    """Linearly crossfade the neural-only vocal (0.0) against the fully DSP-processed vocal (1.0).

    clean_intensity is clamped to [0.0, 1.0]. Both inputs are read as (frames, channels) float64
    and trimmed to their shorter common length before blending.
    """
    clean_intensity = _clamp(clean_intensity, 0.0, 1.0)

    neural_audio, neural_samplerate = sf.read(str(neural_vocal_path), always_2d=True, dtype="float64")
    dsp_audio, dsp_samplerate = sf.read(str(dsp_vocal_path), always_2d=True, dtype="float64")

    if neural_samplerate != dsp_samplerate:
        raise ValueError(
            f"Sample rate mismatch between neural vocal ({neural_samplerate}Hz) "
            f"and DSP vocal ({dsp_samplerate}Hz)"
        )

    length = min(len(neural_audio), len(dsp_audio))
    return (1.0 - clean_intensity) * neural_audio[:length] + clean_intensity * dsp_audio[:length]


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))
