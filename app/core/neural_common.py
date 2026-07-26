"""Shared resemble-enhance neural denoise/enhance pass logic used by both the vocal and
instrumental chains (caching by settings hash, per-channel processing, dry/wet blending).

Denoise and enhance are independently-cacheable stages, run at different points in the
pipeline: denoise runs on the raw isolated stem (pre-DSP), while enhance runs on the
DSP-processed stem (post-DSP, the last AI stage before app.core.qa_gate's capped blend) --
see app/core/vocal_chain.py and app/core/instrumental_chain.py for how they're sequenced.
"""

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
from app.core.platform_compat import resemble_enhance_compat_shims


def _lazy_import_resemble_enhance():
    """Import resemble_enhance's denoise/enhance functions, shimming out deepspeed first.

    resemble_enhance's inference import chain (enhancer/train.py, utils/distributed.py,
    utils/engine.py) references `deepspeed.DeepSpeedConfig`, `deepspeed.accelerator.get_accelerator`,
    `deepspeed.runtime.engine.DeepSpeedEngine`, and `deepspeed.runtime.utils.clip_grad_norm_` at
    *module import time* (inside training-only functions, but the `from ... import ...` statements
    themselves execute on import regardless). deepspeed cannot be built on Windows (its setup.py
    needs libaio, a Linux-only library, and symlink creation that Windows blocks without Developer
    Mode) and none of these training code paths are reachable from the denoise/enhance inference
    functions we actually call, so we inject lightweight stub modules for all four instead of
    installing real deepspeed.
    """
    if "deepspeed" not in sys.modules:
        ds_stub = types.ModuleType("deepspeed")
        ds_stub.DeepSpeedConfig = type("DeepSpeedConfig", (), {})  # type: ignore[attr-defined]

        accelerator_stub = types.ModuleType("deepspeed.accelerator")
        accelerator_stub.get_accelerator = lambda: None  # type: ignore[attr-defined]

        runtime_stub = types.ModuleType("deepspeed.runtime")

        runtime_engine_stub = types.ModuleType("deepspeed.runtime.engine")
        runtime_engine_stub.DeepSpeedEngine = type("DeepSpeedEngine", (), {})  # type: ignore[attr-defined]

        runtime_utils_stub = types.ModuleType("deepspeed.runtime.utils")
        runtime_utils_stub.clip_grad_norm_ = lambda *args, **kwargs: None  # type: ignore[attr-defined]

        ds_stub.accelerator = accelerator_stub  # type: ignore[attr-defined]
        ds_stub.runtime = runtime_stub  # type: ignore[attr-defined]
        runtime_stub.engine = runtime_engine_stub  # type: ignore[attr-defined]
        runtime_stub.utils = runtime_utils_stub  # type: ignore[attr-defined]

        sys.modules["deepspeed"] = ds_stub
        sys.modules["deepspeed.accelerator"] = accelerator_stub
        sys.modules["deepspeed.runtime"] = runtime_stub
        sys.modules["deepspeed.runtime.engine"] = runtime_engine_stub
        sys.modules["deepspeed.runtime.utils"] = runtime_utils_stub

    from resemble_enhance.enhancer.inference import denoise, enhance  # noqa: F811

    return denoise, enhance


logger = get_logger(__name__)

NEURAL_SUBTYPE = "PCM_24"


def run_denoise_pass(
    stem_path: Path,
    denoise_enabled: bool,
    denoise_intensity: float,
    cache_manager: CacheManager,
    filename_prefix: str,
    stem_label: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run resemble-enhance's denoise() stage only on an isolated (pre-DSP) stem.

    Dry/wet blended against the raw stem via the module-level linear-crossfade _blend() at
    denoise_intensity (0.0 = no effect, 1.0 = fully applied).

    Cached at cache/<track_id>/stems/<filename_prefix>denoise_<settings_hash>.wav, where
    settings_hash is derived from (denoise_enabled, denoise_intensity) only -- independent of
    any DSP/enhance settings, so DSP-only or enhance-only re-renders skip this stage entirely.
    track_id is inferred from stem_path's location under the track's stems folder.

    stem_label is used only for logging (e.g. "vocal" or "instrumental").
    """
    if is_cancelled and is_cancelled():
        raise InterruptedError("Denoise pass cancelled")

    if progress_callback:
        progress_callback(0.0)

    track_id = stem_path.parent.parent.name
    settings_hash = _hash_denoise_settings(denoise_enabled, denoise_intensity)
    output_path = (
        cache_manager.stems_dir(track_id)
        / f"{filename_prefix}denoise_{settings_hash}.wav"
    )

    if cache_manager.verify_stem_wav(output_path):
        logger.info(
            "Using cached denoise pass for %s stem of track %s: %s",
            stem_label,
            track_id,
            output_path,
        )
        if progress_callback:
            progress_callback(1.0)
        return output_path

    denoise_intensity = _clamp01(denoise_intensity)
    audio, samplerate = sf.read(str(stem_path), always_2d=True, dtype="float64")

    if not denoise_enabled:
        logger.info(
            "Neural denoise disabled for %s stem of track %s; passing stem through unmodified",
            stem_label,
            track_id,
        )
        sf.write(str(output_path), audio, samplerate, subtype=NEURAL_SUBTYPE)
        if progress_callback:
            progress_callback(1.0)
        return output_path

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(
            "CUDA available; running denoise pass on GPU for %s stem of track %s",
            stem_label,
            track_id,
        )
    else:
        logger.warning(
            "CUDA not available; falling back to CPU for denoise pass on %s stem of track %s (this will be slow)",
            stem_label,
            track_id,
        )

    logger.info(
        "Running denoise pass for %s stem of track %s (denoise=%s@%.2f)",
        stem_label,
        track_id,
        denoise_enabled,
        denoise_intensity,
    )

    num_channels = audio.shape[1]
    processed_channels: list[np.ndarray] = []
    output_samplerate = samplerate
    for channel_index in range(num_channels):
        if is_cancelled and is_cancelled():
            raise InterruptedError("Denoise pass cancelled")
        channel_audio, output_samplerate = _process_channel_denoise(
            audio[:, channel_index], samplerate, device, denoise_intensity
        )
        processed_channels.append(channel_audio)
        if progress_callback:
            progress_callback(0.9 * (channel_index + 1) / num_channels)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Denoise pass cancelled")

    min_length = min(len(channel_audio) for channel_audio in processed_channels)
    processed_audio = np.stack(
        [channel_audio[:min_length] for channel_audio in processed_channels], axis=1
    )

    sf.write(
        str(output_path), processed_audio, output_samplerate, subtype=NEURAL_SUBTYPE
    )
    logger.info(
        "Wrote denoise pass for %s stem of track %s -> %s",
        stem_label,
        track_id,
        output_path,
    )
    if progress_callback:
        progress_callback(1.0)
    return output_path


def run_enhance_pass(
    dsp_stem_path: Path,
    enhance_enabled: bool,
    cache_manager: CacheManager,
    filename_prefix: str,
    stem_label: str,
    progress_callback: Optional[Callable[[float], None]] = None,
    is_cancelled: Optional[Callable[[], bool]] = None,
) -> Path:
    """Run resemble-enhance's enhance() stage only, at full wet strength, on an already
    DSP-processed stem -- the last AI stage in the pipeline, per app.core.qa_gate.

    Unlike run_denoise_pass, this performs no dry/wet blending itself: it returns the raw wet
    (fully enhanced) signal unblended. The caller (vocal_chain.blend_vocal /
    instrumental_chain.blend_instrumental, via app.core.qa_gate.apply_qa_gated_blend) is
    responsible for combining it with dsp_stem_path via a capped, energy-normalized residual
    blend rather than a plain intensity crossfade.

    If enhance_enabled is False, this is a no-op: dsp_stem_path is returned unchanged (no file
    written), since there is nothing to blend against dsp output in that case.

    Otherwise, cached at cache/<track_id>/stems/<filename_prefix>enhance_<settings_hash>.wav,
    where settings_hash is derived from (enhance_enabled, content-hash of dsp_stem_path) --
    since enhance now runs post-DSP, its cache key must track the DSP output's *content* (via
    CacheManager.compute_track_id, a generic file-content hasher) rather than just a toggle, so
    that changing any upstream DSP parameter (notch depth, de-esser, EQ, etc.) correctly
    invalidates this cache entry.
    """
    if not enhance_enabled:
        logger.info(
            "Neural enhance disabled for %s stem; skipping (using DSP output as-is)",
            stem_label,
        )
        return dsp_stem_path

    if is_cancelled and is_cancelled():
        raise InterruptedError("Enhance pass cancelled")

    if progress_callback:
        progress_callback(0.0)

    track_id = dsp_stem_path.parent.parent.name
    dsp_content_hash = CacheManager.compute_track_id(dsp_stem_path)
    settings_hash = _hash_enhance_settings(enhance_enabled, dsp_content_hash)
    output_path = (
        cache_manager.stems_dir(track_id)
        / f"{filename_prefix}enhance_{settings_hash}.wav"
    )

    if cache_manager.verify_stem_wav(output_path):
        logger.info(
            "Using cached enhance pass for %s stem of track %s: %s",
            stem_label,
            track_id,
            output_path,
        )
        if progress_callback:
            progress_callback(1.0)
        return output_path

    audio, samplerate = sf.read(str(dsp_stem_path), always_2d=True, dtype="float64")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        logger.info(
            "CUDA available; running enhance pass on GPU for %s stem of track %s",
            stem_label,
            track_id,
        )
    else:
        logger.warning(
            "CUDA not available; falling back to CPU for enhance pass on %s stem of track %s (this will be slow)",
            stem_label,
            track_id,
        )

    logger.info(
        "Running enhance pass for %s stem of track %s (full wet strength)",
        stem_label,
        track_id,
    )

    num_channels = audio.shape[1]
    processed_channels: list[np.ndarray] = []
    output_samplerate = samplerate
    for channel_index in range(num_channels):
        if is_cancelled and is_cancelled():
            raise InterruptedError("Enhance pass cancelled")
        channel_audio, output_samplerate = _process_channel_enhance(
            audio[:, channel_index], samplerate, device
        )
        processed_channels.append(channel_audio)
        if progress_callback:
            progress_callback(0.9 * (channel_index + 1) / num_channels)

    if is_cancelled and is_cancelled():
        raise InterruptedError("Enhance pass cancelled")

    min_length = min(len(channel_audio) for channel_audio in processed_channels)
    processed_audio = np.stack(
        [channel_audio[:min_length] for channel_audio in processed_channels], axis=1
    )

    sf.write(
        str(output_path), processed_audio, output_samplerate, subtype=NEURAL_SUBTYPE
    )
    logger.info(
        "Wrote enhance pass for %s stem of track %s -> %s",
        stem_label,
        track_id,
        output_path,
    )
    if progress_callback:
        progress_callback(1.0)
    return output_path


def _process_channel_denoise(
    channel: np.ndarray,
    samplerate: int,
    device: torch.device,
    denoise_intensity: float,
) -> tuple[np.ndarray, int]:
    """Run resemble-enhance's denoise() on a single audio channel and dry/wet blend it."""
    denoise, _enhance = _lazy_import_resemble_enhance()

    current = torch.from_numpy(channel).float()

    # See app/core/platform_compat.py: resemble-enhance's downloaded hparams.yaml embeds a
    # PosixPath that fails to parse on Windows, and its LCFM sampler's scipy.optimize.fsolve
    # usage breaks under numpy>=2.
    with resemble_enhance_compat_shims():
        wet, wet_sr = denoise(current, samplerate, device)
    _assert_blendable_sr(samplerate, wet_sr, "denoise")
    blended = _blend(current, wet, denoise_intensity)

    return blended.cpu().numpy().astype(np.float64), wet_sr


def _process_channel_enhance(
    channel: np.ndarray,
    samplerate: int,
    device: torch.device,
) -> tuple[np.ndarray, int]:
    """Run resemble-enhance's enhance() on a single audio channel at full wet strength (no
    blending -- the caller handles capped/residual blending against the DSP-processed input).
    """
    _denoise, enhance = _lazy_import_resemble_enhance()

    current = torch.from_numpy(channel).float()

    with resemble_enhance_compat_shims():
        wet, wet_sr = enhance(current, samplerate, device)

    return wet.cpu().numpy().astype(np.float64), wet_sr


def _assert_blendable_sr(dry_sr: int, wet_sr: int, stage: str) -> None:
    """Guard the dry/wet blend, which aligns the two signals by sample index only.

    _blend crossfades the pre-stage (dry) signal against the model's output (wet) purely by
    truncating to the shorter length, so a sample-rate change between them would silently mix
    two different timelines into pitch/timing garbage with no error. resemble-enhance operates
    at a fixed rate today, so this never trips in practice, but nothing else asserts it.
    """
    if dry_sr != wet_sr:
        raise ValueError(
            f"Neural {stage} stage changed sample rate ({dry_sr}Hz -> {wet_sr}Hz); "
            f"dry/wet blend requires matching rates"
        )


def _blend(dry: torch.Tensor, wet: torch.Tensor, intensity: float) -> torch.Tensor:
    """Dry/wet crossfade wet against dry at the given intensity, trimming to the shorter length."""
    length = min(dry.shape[-1], wet.shape[-1])
    return intensity * wet[..., :length] + (1.0 - intensity) * dry[..., :length]


def _hash_denoise_settings(denoise_enabled: bool, denoise_intensity: float) -> str:
    """Derive a short, stable hash identifying this denoise-pass settings combination."""
    payload = f"{denoise_enabled}|{denoise_intensity:.6f}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _hash_enhance_settings(enhance_enabled: bool, dsp_content_hash: str) -> str:
    """Derive a short, stable hash identifying this enhance-pass settings combination.

    Includes a content hash of the DSP-stage output (not just the enhance toggle) since
    enhance now runs post-DSP: any upstream DSP parameter change must invalidate this cache.
    """
    payload = f"{enhance_enabled}|{dsp_content_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
