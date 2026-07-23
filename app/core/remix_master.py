"""Blend the processed vocal and instrumental stems with per-stem gain, then LUFS-normalize and
true-peak-limit the mix into a mastered track, exporting it as 24-bit PCM WAV."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf
from pedalboard import Limiter, Pedalboard

from app.cache import get_logger

logger = get_logger(__name__)

EXPORT_SUBTYPE = "PCM_24"

LIMITER_CEILING_DBTP = -1.0
LIMITER_RELEASE_MS = 100.0


def mix_stems(
    vocal_audio: np.ndarray,
    instrumental_audio: np.ndarray,
    vocal_gain_db: float,
    instrumental_gain_db: float,
) -> np.ndarray:
    """Apply per-stem gain and sum the vocal and instrumental stems into a single mix.

    Both inputs are (frames, channels) arrays. If their frame counts differ, the shorter one
    is zero-padded to match before summing.
    """
    vocal_gained = vocal_audio * _db_to_linear(vocal_gain_db)
    instrumental_gained = instrumental_audio * _db_to_linear(instrumental_gain_db)

    vocal_gained, instrumental_gained = _pad_to_match(vocal_gained, instrumental_gained)

    mixed = vocal_gained + instrumental_gained
    logger.info(
        "Mixed stems (vocal_gain=%.2fdB, instrumental_gain=%.2fdB) -> %d frames",
        vocal_gain_db,
        instrumental_gain_db,
        mixed.shape[0],
    )
    return mixed


def master(mixed_audio: np.ndarray, sample_rate: int, lufs_target: float = -14.0) -> np.ndarray:
    """LUFS-normalize the mix to lufs_target, then run a true-peak limiter to prevent clipping.

    Integrated loudness is measured and normalized via pyloudnorm, then a Pedalboard Limiter
    with a ceiling around LIMITER_CEILING_DBTP catches any remaining peaks the gain change
    introduced. mixed_audio is a (frames, channels) array, as returned by mix_stems.
    """
    meter = pyln.Meter(sample_rate)
    input_loudness = meter.integrated_loudness(mixed_audio)
    logger.info("Pre-master loudness: %.2f LUFS (target %.2f LUFS)", input_loudness, lufs_target)

    normalized = pyln.normalize.loudness(mixed_audio, input_loudness, lufs_target)

    limiter_board = Pedalboard(
        [Limiter(threshold_db=LIMITER_CEILING_DBTP, release_ms=LIMITER_RELEASE_MS)]
    )
    channels_first = normalized.T.astype(np.float32)
    limited = limiter_board(channels_first, sample_rate).T.astype(np.float64)

    output_loudness = meter.integrated_loudness(limited)
    logger.info(
        "Post-master loudness: %.2f LUFS -> %.2f LUFS (limiter ceiling %.1f dBTP)",
        input_loudness,
        output_loudness,
        LIMITER_CEILING_DBTP,
    )
    return limited


def export_wav(audio: np.ndarray, sample_rate: int, out_path: Path) -> Path:
    """Write audio to out_path as 24-bit PCM WAV."""
    sf.write(str(out_path), audio, sample_rate, subtype=EXPORT_SUBTYPE)
    logger.info("Exported mastered WAV -> %s (%dHz, %s)", out_path, sample_rate, EXPORT_SUBTYPE)
    return out_path


def _pad_to_match(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Zero-pad the shorter of two (frames, channels) arrays so both share the same frame count."""
    length = max(len(a), len(b))
    if len(a) < length:
        a = np.pad(a, ((0, length - len(a)), (0, 0)))
    if len(b) < length:
        b = np.pad(b, ((0, length - len(b)), (0, 0)))
    return a, b


def _db_to_linear(db: float) -> float:
    return 10.0 ** (db / 20.0)
