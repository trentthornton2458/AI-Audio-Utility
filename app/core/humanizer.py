"""Humanizer stage: pyrubberband-driven micro-pitch drift that breaks Suno's robotic
micro-pitch locking by modulating pitch continuously over time with a low-frequency
oscillator, rather than applying one static pitch shift."""

from __future__ import annotations

import math
import os

import numpy as np
import pyrubberband as pyrb

from app.cache import get_logger
from app.setup import model_downloader

logger = get_logger(__name__)

DRIFT_DEPTH_MIN_CENTS = 3.0
DRIFT_DEPTH_MAX_CENTS = 5.0
DRIFT_RATE_MIN_HZ = 4.0
DRIFT_RATE_MAX_HZ = 7.0

# Windowed/overlap-add processing: each window is pitch-shifted by the LFO's instantaneous
# value at its center, then cross-faded into neighboring windows via a periodic Hann taper.
# 50% hop (window/2) satisfies the constant-overlap-add identity for periodic Hann, so no
# extra normalization pass is needed to reconstruct unity gain.
WINDOW_SIZE_MS = 50.0
HOP_RATIO = 0.5

CENTS_PER_SEMITONE = 100.0

_rubberband_configured = False


def apply_pitch_drift(audio: np.ndarray, sample_rate: int, intensity: float) -> np.ndarray:
    """Apply LFO-driven micro-pitch drift to `audio`.

    intensity in [0.0, 1.0] maps linearly to a drift depth of +/-[DRIFT_DEPTH_MIN_CENTS,
    DRIFT_DEPTH_MAX_CENTS] cents, modulated by a sine LFO whose rate maps linearly to
    [DRIFT_RATE_MIN_HZ, DRIFT_RATE_MAX_HZ] Hz. intensity=0.0 returns the input unchanged.

    audio is frames-first: shape (num_samples,) for mono or (num_samples, num_channels) for
    multichannel, matching soundfile's always_2d convention used elsewhere in this codebase.
    All channels share a single LFO curve (no inter-channel phase drift).

    Implemented as windowed (Hann, 50% overlap) pitch shifting via pyrubberband rather than a
    single static pitch shift, so the drift varies continuously over time per Counsel's spec.
    """
    intensity = _clamp01(intensity)
    if intensity <= 0.0:
        return np.array(audio, copy=True)

    depth_cents = DRIFT_DEPTH_MIN_CENTS + intensity * (DRIFT_DEPTH_MAX_CENTS - DRIFT_DEPTH_MIN_CENTS)
    rate_hz = DRIFT_RATE_MIN_HZ + intensity * (DRIFT_RATE_MAX_HZ - DRIFT_RATE_MIN_HZ)

    _configure_rubberband_binary()

    working = np.asarray(audio, dtype=np.float64)
    mono_input = working.ndim == 1
    if mono_input:
        working = working[:, np.newaxis]

    num_samples = working.shape[0]
    if num_samples == 0:
        return np.array(audio, copy=True)

    window_size = max(int(round(sample_rate * WINDOW_SIZE_MS / 1000.0)), 2)
    window_size += window_size % 2  # force even, so hop = window_size/2 is exact
    hop_size = int(window_size * HOP_RATIO)
    window = _periodic_hann(window_size)

    # Pad by a full window on each side so every real sample sits in the fully-overlapped
    # (COLA-valid) interior of the reconstruction, then trim the padding back off at the end.
    pad = window_size
    padded = np.pad(working, ((pad, pad), (0, 0)), mode="constant")
    output = np.zeros_like(padded)

    start = 0
    while start + window_size <= padded.shape[0]:
        end = start + window_size
        segment = padded[start:end]
        center_time_s = (start + window_size / 2.0 - pad) / sample_rate
        n_steps = _lfo_cents(center_time_s, rate_hz, depth_cents) / CENTS_PER_SEMITONE
        shifted = _match_length(_pitch_shift_window(segment, sample_rate, n_steps), window_size)
        output[start:end] += shifted * window[:, np.newaxis]
        start += hop_size

    result = output[pad : pad + num_samples]
    if mono_input:
        result = result[:, 0]
    return result.astype(audio.dtype)


def _configure_rubberband_binary() -> None:
    """Point pyrubberband at the setup-wizard-downloaded rubberband CLI binary.

    pyrubberband 0.4.0 hardcodes its subprocess command name to the literal string
    "rubberband" and does not read a RUBBERBAND_UTIL environment variable, so the only way to
    make it resolve our downloaded (non-PATH) binary is to put that binary's directory on PATH
    (Windows resolves the extension-less "rubberband" via PATHEXT). RUBBERBAND_UTIL is set
    anyway for forward-compatibility with pyrubberband versions/forks that do read it.
    """
    global _rubberband_configured
    if _rubberband_configured:
        return

    binary_path = model_downloader.get_rubberband_binary_path()
    os.environ["RUBBERBAND_UTIL"] = str(binary_path)

    bin_dir = str(binary_path.parent)
    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_entries:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")

    _rubberband_configured = True
    logger.info("Configured rubberband binary for pyrubberband: %s", binary_path)


def _pitch_shift_window(window: np.ndarray, sample_rate: int, n_steps: float) -> np.ndarray:
    """Thin wrapper around pyrubberband.pitch_shift, isolated so tests can substitute a
    synthetic (non-GPL-binary-dependent) implementation."""
    if n_steps == 0.0:
        return window
    return pyrb.pitch_shift(window, sample_rate, n_steps)


def _periodic_hann(window_size: int) -> np.ndarray:
    """Periodic (DFT-even) Hann window: satisfies constant-overlap-add at 50% hop, unlike
    numpy's symmetric np.hanning which tapers to zero at both endpoints."""
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(window_size) / window_size)


def _lfo_cents(t_seconds: float, rate_hz: float, depth_cents: float) -> float:
    return depth_cents * math.sin(2.0 * math.pi * rate_hz * t_seconds)


def _match_length(segment: np.ndarray, target_length: int) -> np.ndarray:
    current_length = segment.shape[0]
    if current_length == target_length:
        return segment
    if current_length > target_length:
        return segment[:target_length]
    pad_width = target_length - current_length
    return np.pad(segment, ((0, pad_width), (0, 0)), mode="constant")


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))
