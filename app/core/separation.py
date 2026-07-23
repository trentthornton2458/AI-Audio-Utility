"""Splits a normalized track into vocal and instrumental stems via audio-separator (BS-RoFormer)."""

from __future__ import annotations

from pathlib import Path

import os
import shutil
import imageio_ffmpeg
import torch
import soundfile as sf
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


def _cleanup_stems_dir(stems_dir: Path) -> None:
    """Delete any existing files in stems_dir to clean up partial or corrupt remains."""
    if stems_dir.exists():
        for item in stems_dir.iterdir():
            if item.is_file():
                try:
                    item.unlink()
                except Exception as exc:
                    logger.warning("Failed to delete %s during cleanup: %s", item, exc)


def _run_separation(
    normalized_wav_path: Path,
    cache_manager: CacheManager,
    stems_dir: Path,
    vocal_path: Path,
    instrumental_path: Path,
    force_cpu: bool = False,
) -> None:
    """Internal helper to load separator, separate stems, and rename output paths."""
    original_is_available = torch.cuda.is_available
    if force_cpu:
        torch.cuda.is_available = lambda: False

    try:
        separator = Separator(output_dir=str(stems_dir), model_file_dir=str(cache_manager.models_dir))
        separator.load_model(model_filename=MODEL_FILENAME)
        produced = separator.separate(str(normalized_wav_path))
        produced_paths = [_resolve_output_path(name, stems_dir) for name in produced]

        _rename_stem(produced_paths, "vocal", vocal_path)
        _rename_stem(produced_paths, "instrumental", instrumental_path)

        if not vocal_path.is_file() or not instrumental_path.is_file():
            raise RuntimeError(
                f"Stem separation did not produce expected output files in {stems_dir} "
                f"(separator returned {produced})"
            )
    finally:
        if force_cpu:
            torch.cuda.is_available = original_is_available


def separate_stems(normalized_wav_path: Path, cache_manager: CacheManager) -> tuple[Path, Path]:
    """Split a normalized WAV into (vocal_stem_path, instrumental_stem_path) using BS-RoFormer.

    Stems are cached at cache/<track_id>/stems/vocal.wav and instrumental.wav, where
    track_id is the name of normalized_wav_path's parent directory (the track's cache
    folder, as written by app.core.ingestion). If both cached files already exist,
    separation is skipped and their paths are returned directly.
    """
    track_id = normalized_wav_path.parent.name
    stems_dir = cache_manager.stems_dir(track_id)
    vocal_path = stems_dir / VOCAL_FILENAME
    instrumental_path = stems_dir / INSTRUMENTAL_FILENAME

    if vocal_path.is_file() and instrumental_path.is_file():
        logger.info("Using cached stems for track %s: %s, %s", track_id, vocal_path, instrumental_path)
        return vocal_path, instrumental_path

    # Robust handling for corrupt or invalid input WAV files
    if not normalized_wav_path.is_file():
        raise FileNotFoundError(f"Input WAV file does not exist: {normalized_wav_path}")

    try:
        info = sf.info(str(normalized_wav_path))
        if info.frames == 0 or info.duration <= 0 or info.channels == 0:
            raise ValueError("WAV file contains no audio frames, duration, or channels.")
    except Exception as exc:
        raise ValueError(f"Invalid or corrupt input WAV file: {exc}") from exc

    ensure_ffmpeg_in_path(cache_manager.bin_dir)

    # Initial cleanup to remove any stale or partial results
    _cleanup_stems_dir(stems_dir)

    use_gpu = torch.cuda.is_available()
    run_on_gpu = use_gpu

    try:
        try:
            if run_on_gpu:
                logger.info("CUDA available; running stem separation on GPU for track %s", track_id)
            else:
                logger.warning(
                    "CUDA not available; running stem separation on CPU for track %s (this will be slow)",
                    track_id,
                )

            _run_separation(
                normalized_wav_path=normalized_wav_path,
                cache_manager=cache_manager,
                stems_dir=stems_dir,
                vocal_path=vocal_path,
                instrumental_path=instrumental_path,
                force_cpu=not run_on_gpu,
            )

        except Exception as exc:
            is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower() or "cuda_error_out_of_memory" in str(exc).lower()
            if run_on_gpu and is_oom:
                logger.warning(
                    "GPU OOM encountered during stem separation for track %s. Clearing cache and falling back to CPU separation.",
                    track_id,
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Clean up any partial files before retry
                _cleanup_stems_dir(stems_dir)

                # Fallback retry on CPU
                run_on_gpu = False
                logger.warning("Retrying stem separation on CPU for track %s (this will be slow)", track_id)
                _run_separation(
                    normalized_wav_path=normalized_wav_path,
                    cache_manager=cache_manager,
                    stems_dir=stems_dir,
                    vocal_path=vocal_path,
                    instrumental_path=instrumental_path,
                    force_cpu=True,
                )
            else:
                raise
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("Explicitly cleared PyTorch CUDA memory cache after stem separation finishes.")

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
