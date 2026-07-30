"""Deterministic QA gate governing how much of the post-DSP AI-enhance signal gets blended
back into each stem.

Computes spectral flatness / centroid / crest-factor / >=8kHz sibilance-energy-ratio deltas
between the DSP-only signal and the (full-strength) enhanced signal over sliding windows,
derives a per-window blend gain capped at max_gain, applies hard fail-safes (silence,
clipping, a sharp spectral-flatness collapse used as a hallucinated-pure-tone proxy) that
force a window's gain to 0 regardless of the soft thresholds, and optionally escalates
flagged windows to a non-fatal Gemini diagnostic for extra attenuation.

The two signals are combined via an energy-normalized parallel residual blend, not a linear
crossfade: the enhanced signal is first level-matched to the DSP signal's local loudness (so a
neural pass that merely changes overall level isn't misread as a large timbral change), then
`out = dsp + gain(t) * (enhanced_level_matched - dsp)`. Unlike a crossfade, this never rescales
the dry (DSP) component itself, which is what preserves phase and avoids comb filtering.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import torch
import torchaudio

from app.cache import get_logger
from app.core import gemini_qa
from app.models.app_config import QAGateThresholds, get_app_config

logger = get_logger(__name__)

# Mirrors PRESET_SCHEMA's vocal/instrumental_enhance_intensity "maximum" (app/models/preset.py)
# and QAGateThresholds.max_enhance_gain (app/models/app_config.py) -- kept in sync manually,
# same convention as sanitize_preset_dict's `defaults` dict already being manually synced with
# PRESET_SCHEMA.
MAX_ENHANCE_GAIN = 0.35

_EPS = 1e-12

# --- Warn-only signal-quality metrics (measure_pitch_variance / measure_high_frequency_energy /
# measure_crest_factor below) -------------------------------------------------------------------
# Independent of the deterministic enhance-blend gate above: these never reduce a blend gain or
# block export, they only attach a QAMetricResult.warning flag for later UI surfacing.

# torchaudio.functional.detect_pitch_frequency (NCF-based, already a project dependency -- see
# pyproject.toml's torchaudio entry) has its own lag-quantization noise floor of a few cents even
# on a perfectly flat pure tone, so the warning threshold sits above that floor rather than at 0.
PITCH_FRAME_SECONDS = 0.01
PITCH_FREQ_LOW_HZ = 80.0
PITCH_FREQ_HIGH_HZ = 800.0
PITCH_VOICED_RMS_FLOOR = 1e-3
PITCH_VARIANCE_WARN_CENTS_MAX = 8.0

HF_ENERGY_FRAME_SECONDS = 0.05
HF_ENERGY_CUTOFF_HZ = 8000.0
HF_ENERGY_WARN_RATIO_MAX = 0.01

# Mirrors the common "DR value" loudness-war heuristic: sustained crest factor below ~6dB
# indicates the signal has been brickwall-limited rather than left with natural peak headroom.
CREST_FACTOR_WARN_DB_MIN = 6.0


@dataclass
class QAWindowFlag:
    """A single sliding window whose AI-enhance blend gain was automatically reduced below
    the requested max_gain, and why."""

    stem_label: str
    start_seconds: float
    end_seconds: float
    reason: str
    deterministic_gain: float
    final_gain: float
    gemini_multiplier: Optional[float] = None
    gemini_verdict: Optional[str] = None


@dataclass
class QAGateResult:
    """Result of app.core.qa_gate.apply_qa_gated_blend: the blended audio plus any QA flags
    raised along the way (for render-history/UI surfacing)."""

    audio: np.ndarray
    samplerate: int
    qa_flags: list[QAWindowFlag] = field(default_factory=list)


@dataclass
class QAMetricResult:
    """Result of one warn-only QA.measure_* signal-quality check: the raw metric value plus
    whether it breached its warning threshold, and why. Unlike QAGateResult/QAWindowFlag above,
    these never reduce a blend gain or block export -- they exist purely for UI surfacing.
    """

    value: float
    warning: bool
    reason: str = ""


def apply_qa_gated_blend(
    dsp_path: Path,
    enhanced_path: Path,
    max_gain: float,
    stem_label: str,
    thresholds: Optional[QAGateThresholds] = None,
    gemini_api_key: Optional[str] = None,
) -> QAGateResult:
    """Blend enhanced_path back into dsp_path at a per-window gain capped at max_gain.

    Returns QAGateResult(audio=dsp_audio, samplerate=sr, qa_flags=[]) immediately (no read of
    enhanced_path beyond the equality check) if max_gain <= 0 or enhanced_path == dsp_path
    (the enhance stage was disabled/a no-op).
    """
    dsp_audio, dsp_samplerate = sf.read(str(dsp_path), always_2d=True, dtype="float64")

    if max_gain <= 0.0 or enhanced_path == dsp_path:
        return QAGateResult(audio=dsp_audio, samplerate=dsp_samplerate, qa_flags=[])

    enhanced_audio, enhanced_samplerate = sf.read(
        str(enhanced_path), always_2d=True, dtype="float64"
    )
    if dsp_samplerate != enhanced_samplerate:
        raise ValueError(
            f"Sample rate mismatch between DSP signal ({dsp_samplerate}Hz) "
            f"and enhanced signal ({enhanced_samplerate}Hz)"
        )
    samplerate = dsp_samplerate

    length = min(len(dsp_audio), len(enhanced_audio))
    dsp_audio = dsp_audio[:length]
    enhanced_audio = enhanced_audio[:length]

    thresholds = thresholds or get_app_config().qa_gate_thresholds
    max_gain = min(max_gain, thresholds.max_enhance_gain)

    dsp_mono = dsp_audio.mean(axis=1)
    enhanced_mono = enhanced_audio.mean(axis=1)

    window_frames = max(1, min(int(thresholds.window_seconds * samplerate), length))
    hop_frames = max(1, int(thresholds.hop_seconds * samplerate))

    starts = list(range(0, max(1, length - window_frames + 1), hop_frames))
    if not starts:
        starts = [0]
    if starts[-1] + window_frames < length:
        starts.append(length - window_frames)

    window_centers: list[float] = []
    window_gains: list[float] = []
    qa_flags: list[QAWindowFlag] = []

    for start in starts:
        end = min(start + window_frames, length)
        dsp_window = dsp_mono[start:end]
        enhanced_window = enhanced_mono[start:end]

        metrics = _window_metrics(dsp_window, enhanced_window, samplerate)
        gain, reason = _evaluate_window(metrics, thresholds, max_gain)
        window_centers.append((start + end) / 2.0)

        if gain < max_gain:
            flag = QAWindowFlag(
                stem_label=stem_label,
                start_seconds=start / samplerate,
                end_seconds=end / samplerate,
                reason=reason or "unknown",
                deterministic_gain=gain,
                final_gain=gain,
            )
            if thresholds.gemini_diagnostics_enabled and gemini_api_key:
                try:
                    multiplier, verdict = gemini_qa.diagnose_qa_window(
                        dsp_window,
                        enhanced_window,
                        samplerate,
                        gemini_api_key,
                        stem_label,
                        flag.reason,
                    )
                except gemini_qa.GeminiAnalysisError as exc:
                    logger.warning(
                        "Gemini QA-window diagnostic failed for %s stem window %.1f-%.1fs "
                        "(non-fatal, falling back to deterministic gain): %s",
                        stem_label,
                        flag.start_seconds,
                        flag.end_seconds,
                        exc,
                    )
                else:
                    flag.gemini_multiplier = multiplier
                    flag.gemini_verdict = verdict
                    flag.final_gain = gain * max(0.0, min(1.0, multiplier))
                    gain = flag.final_gain
            qa_flags.append(flag)
            window_gains.append(flag.final_gain)
        else:
            window_gains.append(gain)

    total_samples = length
    window_centers_arr = np.array(window_centers, dtype=np.float64)
    gain_envelope = _build_envelope(
        window_centers_arr, np.array(window_gains, dtype=np.float64), total_samples
    )

    dsp_rms_windows = np.array(
        [_rms(dsp_mono[s : min(s + window_frames, length)]) for s in starts],
        dtype=np.float64,
    )
    enhanced_rms_windows = np.array(
        [_rms(enhanced_mono[s : min(s + window_frames, length)]) for s in starts],
        dtype=np.float64,
    )
    dsp_rms_envelope = _build_envelope(
        window_centers_arr, dsp_rms_windows, total_samples
    )
    enhanced_rms_envelope = _build_envelope(
        window_centers_arr, enhanced_rms_windows, total_samples
    )

    level_match_scale = dsp_rms_envelope / np.maximum(
        enhanced_rms_envelope, thresholds.silence_rms_floor
    )
    level_match_scale = np.clip(level_match_scale, 0.1, 10.0)

    enhanced_level_matched = enhanced_audio * level_match_scale[:, None]
    residual = enhanced_level_matched - dsp_audio
    out = dsp_audio + gain_envelope[:, None] * residual

    return QAGateResult(audio=out, samplerate=samplerate, qa_flags=qa_flags)


def measure_pitch_variance(audio: np.ndarray, sample_rate: int) -> QAMetricResult:
    """Standard deviation (in cents) of estimated pitch across voiced frames, via torchaudio's
    NCF-based detect_pitch_frequency (already a project dependency, so no new pitch-tracking
    library is needed). Frames are treated as voiced when their pitch estimate falls in-range
    and their local RMS clears PITCH_VOICED_RMS_FLOOR. Warns when the variance is at/below
    PITCH_VARIANCE_WARN_CENTS_MAX, i.e. the pitch is hard-quantized/flat rather than carrying a
    human performance's natural micro-drift.
    """
    mono = _to_mono(audio)
    if mono.size == 0:
        return QAMetricResult(value=0.0, warning=True, reason="no_audio")

    waveform = torch.from_numpy(mono.astype(np.float32)).unsqueeze(0)
    pitch = (
        torchaudio.functional.detect_pitch_frequency(
            waveform,
            sample_rate,
            frame_time=PITCH_FRAME_SECONDS,
            freq_low=int(PITCH_FREQ_LOW_HZ),
            freq_high=int(PITCH_FREQ_HIGH_HZ),
        )
        .squeeze(0)
        .numpy()
    )

    num_frames = pitch.shape[0]
    if num_frames == 0:
        return QAMetricResult(
            value=0.0, warning=True, reason="flat_or_hard_quantized_pitch"
        )

    # detect_pitch_frequency doesn't expose exact frame boundaries, so approximate them by
    # dividing the signal evenly across the returned frame count -- fine for a coarse
    # voiced/unvoiced amplitude gate.
    segment_len = max(1, len(mono) // num_frames)
    voiced = np.array(
        [
            pitch[i] > 0
            and _rms(mono[i * segment_len : (i + 1) * segment_len])
            >= PITCH_VOICED_RMS_FLOOR
            for i in range(num_frames)
        ]
    )
    voiced_pitch = pitch[voiced]
    if voiced_pitch.size < 2:
        return QAMetricResult(
            value=0.0, warning=True, reason="flat_or_hard_quantized_pitch"
        )

    cents = 1200.0 * np.log2(voiced_pitch / np.mean(voiced_pitch))
    variance_cents = float(np.std(cents))
    warning = variance_cents <= PITCH_VARIANCE_WARN_CENTS_MAX
    return QAMetricResult(
        value=variance_cents,
        warning=warning,
        reason="flat_or_hard_quantized_pitch" if warning else "",
    )


def measure_high_frequency_energy(
    audio: np.ndarray, sample_rate: int, silence_threshold_db: float
) -> QAMetricResult:
    """Fraction of spectral energy at/above HF_ENERGY_CUTOFF_HZ within the audio's detected
    non-vocal/breath sections (frames whose RMS level falls below silence_threshold_db dBFS).
    Warns when that fraction is at/below HF_ENERGY_WARN_RATIO_MAX, i.e. breath/room detail has
    been stripped from the quiet passages rather than merely denoised.
    """
    mono = _to_mono(audio)
    frame_len = max(1, int(HF_ENERGY_FRAME_SECONDS * sample_rate))
    breath_frames = [
        mono[start : start + frame_len]
        for start in range(0, len(mono), frame_len)
        if 20.0 * np.log10(_rms(mono[start : start + frame_len]) + _EPS)
        < silence_threshold_db
    ]
    if not breath_frames:
        return QAMetricResult(value=0.0, warning=False, reason="")

    breath_audio = np.concatenate(breath_frames)
    hf_ratio = _sibilance_energy_ratio(
        breath_audio, sample_rate, cutoff_hz=HF_ENERGY_CUTOFF_HZ
    )
    warning = hf_ratio <= HF_ENERGY_WARN_RATIO_MAX
    return QAMetricResult(
        value=hf_ratio,
        warning=warning,
        reason="stripped_breath_detail" if warning else "",
    )


def measure_crest_factor(audio: np.ndarray) -> QAMetricResult:
    """Peak-to-RMS ratio in dB (see _crest_factor). Warns when the value is at/below
    CREST_FACTOR_WARN_DB_MIN, indicating over-compression/brickwalling.
    """
    mono = _to_mono(audio)
    crest_db = _crest_factor(mono)
    warning = crest_db <= CREST_FACTOR_WARN_DB_MIN
    return QAMetricResult(
        value=crest_db, warning=warning, reason="over_compressed" if warning else ""
    )


def _evaluate_window(
    metrics: dict, thresholds: QAGateThresholds, max_gain: float
) -> tuple[float, Optional[str]]:
    """Derive (gain, reason) for one window from its metrics dict, starting at max_gain and
    applying soft proportional reductions per breached delta, then hard fail-safe overrides.
    """
    gain = max_gain
    worst_reason: Optional[str] = None
    worst_reduction = 1.0

    deltas = (
        (
            "spectral_flatness_delta",
            abs(metrics["flatness_enh"] - metrics["flatness_dsp"]),
            thresholds.spectral_flatness_delta_max,
        ),
        (
            "spectral_centroid_delta",
            abs(metrics["centroid_enh"] - metrics["centroid_dsp"]),
            thresholds.spectral_centroid_delta_hz_max,
        ),
        (
            "crest_factor_delta",
            abs(metrics["crest_enh"] - metrics["crest_dsp"]),
            thresholds.crest_factor_delta_db_max,
        ),
        (
            "sibilance_ratio_delta",
            abs(metrics["sibilance_enh"] - metrics["sibilance_dsp"]),
            thresholds.sibilance_ratio_delta_max,
        ),
    )
    for name, delta, limit in deltas:
        if delta > limit and delta > 0:
            reduction = max(0.0, min(1.0, limit / delta))
            gain *= reduction
            if reduction < worst_reduction:
                worst_reduction = reduction
                worst_reason = name

    # Hard fail-safes override everything above.
    if (
        metrics["rms_dsp"] < thresholds.silence_rms_floor
        or metrics["rms_enh"] < thresholds.silence_rms_floor
    ):
        return 0.0, "silence"
    if metrics["peak_enh"] >= thresholds.clipping_threshold:
        return 0.0, "clipping"
    if (
        metrics["flatness_dsp"] - metrics["flatness_enh"]
        > thresholds.hallucination_flatness_drop_max
    ):
        return 0.0, "hallucination_proxy"

    return gain, worst_reason


def _window_metrics(
    dsp_window: np.ndarray, enhanced_window: np.ndarray, samplerate: int
) -> dict:
    return {
        "flatness_dsp": _spectral_flatness(dsp_window, samplerate),
        "flatness_enh": _spectral_flatness(enhanced_window, samplerate),
        "centroid_dsp": _spectral_centroid(dsp_window, samplerate),
        "centroid_enh": _spectral_centroid(enhanced_window, samplerate),
        "crest_dsp": _crest_factor(dsp_window),
        "crest_enh": _crest_factor(enhanced_window),
        "sibilance_dsp": _sibilance_energy_ratio(dsp_window, samplerate),
        "sibilance_enh": _sibilance_energy_ratio(enhanced_window, samplerate),
        "rms_dsp": _rms(dsp_window),
        "rms_enh": _rms(enhanced_window),
        "peak_enh": (
            float(np.max(np.abs(enhanced_window))) if enhanced_window.size else 0.0
        ),
    }


def _spectral_flatness(x: np.ndarray, samplerate: int) -> float:
    """Ratio of the geometric mean to the arithmetic mean of the power spectrum: near 0 for a
    tonal/peaky signal (e.g. a pure tone), near 1 for a noise-like/flat signal."""
    if x.size == 0:
        return 0.0
    windowed = x * np.hanning(len(x)) if len(x) > 1 else x
    power = np.abs(np.fft.rfft(windowed)) ** 2
    if power.size == 0:
        return 0.0
    log_power = np.log(power + _EPS)
    geometric_mean = np.exp(np.mean(log_power))
    arithmetic_mean = np.mean(power) + _EPS
    return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))


def _spectral_centroid(x: np.ndarray, samplerate: int) -> float:
    """Energy-weighted mean frequency (Hz) of the signal's magnitude spectrum."""
    if x.size == 0:
        return 0.0
    magnitude = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / samplerate)
    total = np.sum(magnitude)
    if total <= _EPS:
        return 0.0
    return float(np.sum(freqs * magnitude) / total)


def _crest_factor(x: np.ndarray) -> float:
    """Peak-to-RMS ratio in dB."""
    if x.size == 0:
        return 0.0
    peak = np.max(np.abs(x))
    rms = _rms(x)
    if rms <= _EPS:
        return 0.0
    return float(20.0 * np.log10(peak / rms + _EPS))


def _sibilance_energy_ratio(
    x: np.ndarray, samplerate: int, cutoff_hz: float = 8000.0
) -> float:
    """Fraction of spectral energy at or above cutoff_hz."""
    if x.size == 0:
        return 0.0
    magnitude = np.abs(np.fft.rfft(x))
    freqs = np.fft.rfftfreq(len(x), d=1.0 / samplerate)
    total_energy = np.sum(magnitude**2)
    if total_energy <= _EPS:
        return 0.0
    hf_energy = np.sum(magnitude[freqs >= cutoff_hz] ** 2)
    return float(hf_energy / total_energy)


def _to_mono(audio: np.ndarray) -> np.ndarray:
    """Collapse a (samples, channels) array to mono by averaging channels; passes 1-D arrays
    through unchanged."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim > 1:
        return audio.mean(axis=1)
    return audio


def _rms(x: np.ndarray) -> float:
    if x.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))


def _build_envelope(
    window_center_samples: np.ndarray, window_values: np.ndarray, total_samples: int
) -> np.ndarray:
    """Smooth per-sample envelope built via linear interpolation between window-center values,
    edge-extended outside the window-center range -- avoids zipper/click artifacts from a
    stepped per-window gain."""
    sample_indices = np.arange(total_samples, dtype=np.float64)
    if window_center_samples.size == 1:
        return np.full(total_samples, window_values[0], dtype=np.float64)
    return np.interp(sample_indices, window_center_samples, window_values)
