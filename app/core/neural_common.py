"""Shared resemble-enhance neural denoise/enhance pass logic used by both the vocal and
instrumental chains (caching by settings hash, per-channel processing, dry/wet blending)."""

from __future__ import annotations

import hashlib
import sys
import types
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch

from app.cache import get_logger
from app.cache.cache_manager import CacheManager


def _lazy_import_resemble_enhance():
    """Import resemble_enhance's denoise/enhance functions, shimming out deepspeed first.

    resemble_enhance's train module does `from deepspeed import DeepSpeedConfig` at import
    time, but deepspeed cannot be built on Windows and is only used for training, not
    inference.  We inject a lightweight stub module so the import chain succeeds.
    """
    if "deepspeed" not in sys.modules:
        ds_stub = types.ModuleType("deepspeed")
        ds_stub.DeepSpeedConfig = type("DeepSpeedConfig", (), {})  # type: ignore[attr-defined]
        sys.modules["deepspeed"] = ds_stub

    from resemble_enhance.enhancer.inference import denoise, enhance  # noqa: F811

    return denoise, enhance

logger = get_logger(__name__)

NEURAL_SUBTYPE = "PCM_24"


def run_neural_pass(
    stem_path: Path,
    denoise_enabled: bool,
    denoise_intensity: float,
    enhance_enabled: bool,
    enhance_intensity: float,
    cache_manager: CacheManager,
    filename_prefix: str,
    stem_label: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run resemble-enhance's denoise and/or enhance stages on an isolated stem.

    Each enabled stage runs at full strength and is then dry/wet blended back against
    its own input using the matching intensity (0.0 = stage has no effect, 1.0 = fully
    applied), so denoise and enhance intensities behave consistently.

    The result is cached at cache/<track_id>/stems/<filename_prefix><settings_hash>.wav,
    where settings_hash is derived from the four settings and track_id is inferred
    from stem_path's location under the track's stems folder. If a matching cached file
    already exists, its path is returned immediately without re-running the model.

    stem_label is used only for logging (e.g. "vocal" or "instrumental").
    """
    if is_cancelled and is_cancelled():
        raise InterruptedError("Neural pass cancelled")

    if progress_callback:
        progress_callback(0.0)

    track_id = stem_path.parent.parent.name
    settings_hash = _hash_settings(denoise_enabled, denoise_intensity, enhance_enabled, enhance_intensity)
    output_path = cache_manager.stems_dir(track_id) / f"{filename_prefix}{settings_hash}.wav"

    if cache_manager.verify_stem_wav(output_path):
        logger.info("Using cached neural %s pass for track %s: %s", stem_label, track_id, output_path)
        if progress_callback:
            progress_callback(1.0)
        return output_path

    denoise_intensity = _clamp01(denoise_intensity)
    enhance_intensity = _clamp01(enhance_intensity)

    audio, samplerate = sf.read(str(stem_path), always_2d=True, dtype="float64")

    if not denoise_enabled and not enhance_enabled:
        logger.info(
            "Neural denoise/enhance both disabled for %s stem of track %s; passing stem through unmodified",
            stem_label,
            track_id,
        )
        sf.write(str(output_path), audio, samplerate, subtype=NEURAL_SUBTYPE)
        if progress_callback:
            progress_callback(1.0)
        return output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("CUDA available; running neural %s pass on GPU for track %s", stem_label, track_id)
    else:
        logger.warning(
            "CUDA not available; falling back to CPU for neural %s pass on track %s (this will be slow)",
            stem_label,
            track_id,
        )

    logger.info(
        "Running neural %s pass for track %s (denoise=%s@%.2f, enhance=%s@%.2f)",
        stem_label,
        track_id,
        denoise_enabled,
        denoise_intensity,
        enhance_enabled,
        enhance_intensity,
    )

    num_channels = audio.shape[1]
    processed_channels: list[np.ndarray] = []
    output_samplerate = samplerate
    for channel_index in range(num_channels):
        if is_cancelled and is_cancelled():
            raise InterruptedError("Neural pass cancelled")
        channel_audio, output_samplerate = _process_channel(
            audio[:, channel_index],
            samplerate,
            device,
            denoise_enabled,
            denoise_intensity,
            enhance_enabled,
            enhance_intensity,
        )
        processed_channels.append(channel_audio)
        if progress_callback:
            progress_callback(0.9 * (channel_index + 1) / num_channels)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Neural pass cancelled")

    min_length = min(len(channel_audio) for channel_audio in processed_channels)
    processed_audio = np.stack([channel_audio[:min_length] for channel_audio in processed_channels], axis=1)

    sf.write(str(output_path), processed_audio, output_samplerate, subtype=NEURAL_SUBTYPE)
    logger.info("Wrote neural %s pass for track %s -> %s", stem_label, track_id, output_path)
    if progress_callback:
        progress_callback(1.0)
    return output_path


def _process_channel(
    channel: np.ndarray,
    samplerate: int,
    device: torch.device,
    denoise_enabled: bool,
    denoise_intensity: float,
    enhance_enabled: bool,
    enhance_intensity: float,
) -> tuple[np.ndarray, int]:
    """Run the enabled neural stages on a single audio channel, returning (samples, samplerate)."""
    denoise, enhance = _lazy_import_resemble_enhance()

    current = torch.from_numpy(channel).float()
    current_sr = samplerate

    if denoise_enabled:
        wet, current_sr = denoise(current, current_sr, device)
        current = _blend(current, wet, denoise_intensity)

    if enhance_enabled:
        wet, current_sr = enhance(current, current_sr, device)
        current = _blend(current, wet, enhance_intensity)

    return current.cpu().numpy().astype(np.float64), current_sr


def _blend(dry: torch.Tensor, wet: torch.Tensor, intensity: float) -> torch.Tensor:
    """Dry/wet crossfade wet against dry at the given intensity, trimming to the shorter length."""
    length = min(dry.shape[-1], wet.shape[-1])
    return intensity * wet[..., :length] + (1.0 - intensity) * dry[..., :length]


def _hash_settings(
    denoise_enabled: bool,
    denoise_intensity: float,
    enhance_enabled: bool,
    enhance_intensity: float,
) -> str:
    """Derive a short, stable hash identifying this combination of neural pass settings."""
    payload = f"{denoise_enabled}|{denoise_intensity:.6f}|{enhance_enabled}|{enhance_intensity:.6f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
