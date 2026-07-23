"""Neural denoise/enhance pass on the isolated vocal stem via resemble-enhance, cached by settings hash."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from resemble_enhance.enhancer.inference import denoise, enhance

from app.cache import get_logger
from app.cache.cache_manager import CacheManager

logger = get_logger(__name__)

NEURAL_FILENAME_PREFIX = "vocal_neural_"
NEURAL_FILENAME_SUFFIX = ".wav"
NEURAL_SUBTYPE = "PCM_24"


def run_neural_pass(
    vocal_stem_path: Path,
    denoise_enabled: bool,
    denoise_intensity: float,
    enhance_enabled: bool,
    enhance_intensity: float,
    cache_manager: CacheManager,
) -> Path:
    """Run resemble-enhance's denoise and/or enhance stages on an isolated vocal stem.

    Each enabled stage runs at full strength and is then dry/wet blended back against
    its own input using the matching intensity (0.0 = stage has no effect, 1.0 = fully
    applied), so denoise and enhance intensities behave consistently.

    The result is cached at cache/<track_id>/stems/vocal_neural_<settings_hash>.wav,
    where settings_hash is derived from the four settings and track_id is inferred
    from vocal_stem_path's location under the track's stems folder. If a matching
    cached file already exists, its path is returned immediately without re-running
    the model.
    """
    track_id = vocal_stem_path.parent.parent.name
    settings_hash = _hash_settings(denoise_enabled, denoise_intensity, enhance_enabled, enhance_intensity)
    output_path = (
        cache_manager.stems_dir(track_id) / f"{NEURAL_FILENAME_PREFIX}{settings_hash}{NEURAL_FILENAME_SUFFIX}"
    )

    if output_path.is_file():
        logger.info("Using cached neural vocal pass for track %s: %s", track_id, output_path)
        return output_path

    denoise_intensity = _clamp01(denoise_intensity)
    enhance_intensity = _clamp01(enhance_intensity)

    audio, samplerate = sf.read(str(vocal_stem_path), always_2d=True, dtype="float64")

    if not denoise_enabled and not enhance_enabled:
        logger.info("Neural denoise/enhance both disabled for track %s; passing stem through unmodified", track_id)
        sf.write(str(output_path), audio, samplerate, subtype=NEURAL_SUBTYPE)
        return output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info("CUDA available; running neural vocal pass on GPU for track %s", track_id)
    else:
        logger.warning(
            "CUDA not available; falling back to CPU for neural vocal pass on track %s (this will be slow)",
            track_id,
        )

    logger.info(
        "Running neural vocal pass for track %s (denoise=%s@%.2f, enhance=%s@%.2f)",
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

    min_length = min(len(channel_audio) for channel_audio in processed_channels)
    processed_audio = np.stack([channel_audio[:min_length] for channel_audio in processed_channels], axis=1)

    sf.write(str(output_path), processed_audio, output_samplerate, subtype=NEURAL_SUBTYPE)
    logger.info("Wrote neural vocal pass for track %s -> %s", track_id, output_path)
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
