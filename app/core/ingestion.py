"""Loads a Suno track and normalizes it to 44.1kHz/24-bit PCM WAV for the pipeline."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
from imageio_ffmpeg import get_ffmpeg_exe

from app.cache import get_logger
from app.cache.cache_manager import CacheManager

logger = get_logger(__name__)

TARGET_SAMPLE_RATE = 44100
TARGET_SUBTYPE = "PCM_24"
NORMALIZED_FILENAME = "normalized.wav"
NATIVE_DECODE_SUFFIXES = {".wav"}


class UnsupportedAudioFormatError(Exception):
    """Raised when an input file cannot be decoded by soundfile or the bundled ffmpeg."""


def load_and_normalize_track(input_path: Path, cache_manager: CacheManager) -> Path:
    """Decode a .wav/.mp3 (or other ffmpeg-readable) file and normalize it to 44.1kHz/24-bit PCM WAV.

    The normalized file is written to cache/<track_id>/normalized.wav, where
    track_id is derived from the input file's content hash, and its path is returned.
    """
    if not input_path.is_file():
        raise UnsupportedAudioFormatError(f"Input audio file not found: {input_path}")

    logger.info("Ingesting track: %s", input_path)
    audio, samplerate = _decode_audio(input_path)

    track_id = cache_manager.compute_track_id(input_path)
    output_path = cache_manager.track_dir(track_id) / NORMALIZED_FILENAME

    sf.write(str(output_path), audio, samplerate, subtype=TARGET_SUBTYPE)
    logger.info(
        "Normalized %s -> %s (%dHz, %d channel(s), %s)",
        input_path,
        output_path,
        samplerate,
        audio.shape[1],
        TARGET_SUBTYPE,
    )
    return output_path


def _decode_audio(input_path: Path) -> tuple[np.ndarray, int]:
    """Decode input_path into a (frames, channels) float64 array at TARGET_SAMPLE_RATE.

    Tries soundfile first for natively-supported formats already at the target
    sample rate; anything else (mp3, unsupported formats, decode failures, or a
    sample rate mismatch) falls back to the bundled ffmpeg binary, which decodes
    and resamples in a single pass.
    """
    suffix = input_path.suffix.lower()

    if suffix in NATIVE_DECODE_SUFFIXES:
        try:
            data, samplerate = sf.read(str(input_path), always_2d=True, dtype="float64")
        except Exception as exc:
            logger.warning("soundfile failed to decode %s (%s); falling back to ffmpeg", input_path, exc)
        else:
            if samplerate == TARGET_SAMPLE_RATE:
                return data, samplerate
            logger.info(
                "%s is %dHz, resampling to %dHz via ffmpeg", input_path, samplerate, TARGET_SAMPLE_RATE
            )

    return _decode_with_ffmpeg(input_path)


def _decode_with_ffmpeg(input_path: Path) -> tuple[np.ndarray, int]:
    """Decode and resample input_path to TARGET_SAMPLE_RATE using the bundled ffmpeg binary."""
    ffmpeg_exe = get_ffmpeg_exe()
    command = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(TARGET_SAMPLE_RATE),
        "-f",
        "wav",
        "-acodec",
        "pcm_s32le",
        "pipe:1",
    ]

    try:
        result = subprocess.run(command, capture_output=True, check=False)
    except OSError as exc:
        raise UnsupportedAudioFormatError(f"Failed to launch bundled ffmpeg for {input_path}: {exc}") from exc

    if result.returncode != 0:
        stderr_tail = result.stderr.decode("utf-8", errors="replace").strip().splitlines()[-1:] if result.stderr else []
        raise UnsupportedAudioFormatError(
            f"ffmpeg could not decode {input_path}: {stderr_tail[0] if stderr_tail else 'unknown error'}"
        )

    try:
        data, samplerate = sf.read(io.BytesIO(result.stdout), always_2d=True, dtype="float64")
    except Exception as exc:
        raise UnsupportedAudioFormatError(f"Could not parse ffmpeg output for {input_path}: {exc}") from exc

    return data, samplerate
