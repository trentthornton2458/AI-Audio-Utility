"""Neural denoise/enhance pass on the isolated vocal stem via resemble-enhance, cached by settings hash,
plus the adjustable Pedalboard DSP chain (HPF/LPF/notch/de-esser) and neural/DSP vocal blend."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from pedalboard import Compressor, HighpassFilter, LowpassFilter, Pedalboard, PeakFilter
from resemble_enhance.enhancer.inference import denoise, enhance

from app.cache import get_logger
from app.cache.cache_manager import CacheManager

logger = get_logger(__name__)

NEURAL_FILENAME_PREFIX = "vocal_neural_"
NEURAL_FILENAME_SUFFIX = ".wav"
NEURAL_SUBTYPE = "PCM_24"

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
