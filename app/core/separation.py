"""Splits a normalized track into vocal and instrumental stems via audio-separator (BS-RoFormer)."""

from __future__ import annotations

from pathlib import Path

import os
import shutil
import imageio_ffmpeg
import soundfile as sf
import torch
from audio_separator.separator import Separator

from app.cache import get_logger
from app.cache.cache_manager import CacheManager

logger = get_logger(__name__)

MODEL_FILENAME = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
VOCAL_STEM_NAME = "vocal"
INSTRUMENTAL_STEM_NAME = "instrumental"
VOCAL_FILENAME = f"{VOCAL_STEM_NAME}.wav"
INSTRUMENTAL_FILENAME = f"{INSTRUMENTAL_STEM_NAME}.wav"


def ensure_ffmpeg_in_path(bin_dir: Path) -> Path:
    """Ensure ffmpeg is available for audio-separator and pydub."""
    try:
        ffmpeg_src = Path(imageio_ffmpeg.get_ffmpeg_exe())
        target_exe = bin_dir / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")

        if not target_exe.is_file():
            shutil.copy(ffmpeg_src, target_exe)
            logger.info("Copied ffmpeg binary to %s", target_exe)

        # 1. Explicitly configure pydub to use this exact binary
        try:
            import pydub
            pydub.AudioSegment.converter = str(target_exe)
            logger.info("Configured pydub converter to %s", target_exe)
        except ImportError:
            pass

        # 2. Monkey-patch audio-separator's ffmpeg check to avoid PATH issues
        try:
            from audio_separator.separator.separator import Separator
            Separator.check_ffmpeg_installed = lambda self: None
            logger.info("Monkey-patched audio-separator ffmpeg check")
        except ImportError:
            pass

        return target_exe
    except Exception as exc:
        logger.warning("Failed to setup ffmpeg: %s", exc)
        return Path("ffmpeg")


def separate_stems(normalized_wav_path: Path, cache_manager: CacheManager) -> tuple[Path, Path]:
    """Split a normalized WAV into (vocal_stem_path, instrumental_stem_path) using BS-RoFormer.

    Stems are cached at cache/<track_id>/stems/vocal.wav and instrumental.wav, where
    track_id is the name of normalized_wav_path's parent directory (the track's cache
    folder, as written by app.core.ingestion). If both cached files already exist,
    separation is skipped and their paths are returned directly.
    """
    if not normalized_wav_path.is_file():
        raise ValueError(f"Input file does not exist: {normalized_wav_path}")

    try:
        info = sf.info(str(normalized_wav_path))
        if info.frames <= 0:
            raise ValueError(f"Input file contains no audio frames: {normalized_wav_path}")
    except Exception as exc:
        if isinstance(exc, ValueError):
            raise
        raise ValueError(f"Input file is corrupt or invalid WAV: {exc}") from exc

    track_id = normalized_wav_path.parent.name
    stems_dir = cache_manager.stems_dir(track_id)
    vocal_path = stems_dir / VOCAL_FILENAME
    instrumental_path = stems_dir / INSTRUMENTAL_FILENAME

    if vocal_path.is_file() and instrumental_path.is_file():
        logger.info("Using cached stems for track %s: %s, %s", track_id, vocal_path, instrumental_path)
        return vocal_path, instrumental_path

    ensure_ffmpeg_in_path(cache_manager.bin_dir)

    if torch.cuda.is_available():
        logger.info("CUDA available; running stem separation on GPU for track %s", track_id)
    else:
        logger.warning(
            "CUDA not available; falling back to CPU for stem separation on track %s (this will be slow)",
            track_id,
        )

    logger.info("Separating stems for track %s from %s", track_id, normalized_wav_path)

    try:
        try:
            separator = Separator(output_dir=str(stems_dir), model_file_dir=str(cache_manager.models_dir))
            separator.load_model(model_filename=MODEL_FILENAME)
            # Let audio-separator write its default-named stems, then rename them to our canonical
            # vocal.wav/instrumental.wav below. This avoids relying on custom_output_names, whose
            # chunked-processing code path matches stem-name keys case-sensitively while the model
            # produces capitalized names ("Vocals"/"Instrumental") — a mismatch that silently
            # misnames outputs.
            produced = separator.separate(str(normalized_wav_path))
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
            if not is_oom:
                raise

            logger.warning(
                "CUDA out of memory during stem separation for track %s; falling back to CPU separation",
                track_id,
            )
            torch.cuda.empty_cache()

            # Temporarily force CPU mode by disabling CUDA and MPS
            original_cuda_is_available = torch.cuda.is_available
            torch.cuda.is_available = lambda: False
            original_mps_is_available = None
            if hasattr(torch.backends, "mps"):
                original_mps_is_available = torch.backends.mps.is_available
                torch.backends.mps.is_available = lambda: False

            try:
                logger.info("Retrying separation on CPU for track %s", track_id)
                cpu_separator = Separator(output_dir=str(stems_dir), model_file_dir=str(cache_manager.models_dir))
                cpu_separator.load_model(model_filename=MODEL_FILENAME)
                produced = cpu_separator.separate(str(normalized_wav_path))
            finally:
                torch.cuda.is_available = original_cuda_is_available
                if original_mps_is_available is not None:
                    torch.backends.mps.is_available = original_mps_is_available
    finally:
        if torch.cuda.is_available():
            logger.info("Explicitly freeing PyTorch CUDA memory cache after separation.")
            torch.cuda.empty_cache()

    produced_paths = [_resolve_output_path(name, stems_dir) for name in produced]

    _rename_stem(produced_paths, "vocal", vocal_path)
    _rename_stem(produced_paths, "instrumental", instrumental_path)

    if not vocal_path.is_file() or not instrumental_path.is_file():
        raise RuntimeError(
            f"Stem separation did not produce expected output files in {stems_dir} "
            f"(separator returned {produced})"
        )

    logger.info("Separated stems for track %s -> %s, %s", track_id, vocal_path, instrumental_path)
    return vocal_path, instrumental_path


def _resolve_output_path(name: str, stems_dir: Path) -> Path:
    """Resolve a name returned by Separator.separate() (a bare filename or a path) to a Path."""
    candidate = Path(name)
    return candidate if candidate.is_absolute() else stems_dir / candidate.name


def _rename_stem(produced_paths: list[Path], token: str, target: Path) -> None:
    """Move the produced stem whose filename identifies it as `token` (e.g. 'vocal') to `target`.

    Matching is case-insensitive and, for 'vocal', excludes 'instrumental' so the substring
    'instrumental' is never mistaken for a vocal stem. No-ops if nothing matches, leaving the
    caller's existence check to raise a clear error.
    """
    for path in produced_paths:
        lower = path.name.lower()
        if token == "vocal" and "instrumental" in lower:
            continue
        if token in lower and path.is_file():
            if path != target:
                path.replace(target)
            return
