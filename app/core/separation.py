"""Splits a normalized track into vocal and instrumental stems via audio-separator (BS-RoFormer)."""

from __future__ import annotations

from pathlib import Path

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

    if torch.cuda.is_available():
        logger.info("CUDA available; running stem separation on GPU for track %s", track_id)
    else:
        logger.warning(
            "CUDA not available; falling back to CPU for stem separation on track %s (this will be slow)",
            track_id,
        )

    logger.info("Separating stems for track %s from %s", track_id, normalized_wav_path)
    separator = Separator(output_dir=str(stems_dir))
    separator.load_model(model_filename=MODEL_FILENAME)
    separator.separate(
        str(normalized_wav_path),
        custom_output_names={"vocals": VOCAL_STEM_NAME, "instrumental": INSTRUMENTAL_STEM_NAME},
    )

    if not vocal_path.is_file() or not instrumental_path.is_file():
        raise RuntimeError(f"Stem separation did not produce expected output files in {stems_dir}")

    logger.info("Separated stems for track %s -> %s, %s", track_id, vocal_path, instrumental_path)
    return vocal_path, instrumental_path
