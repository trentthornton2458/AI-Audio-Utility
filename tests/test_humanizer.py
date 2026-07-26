"""Tests for the pitch-drift Humanizer stage (app/core/humanizer.py)."""

from __future__ import annotations

import numpy as np
import pytest

from app.core import humanizer


def _fake_pitch_shift_window(window: np.ndarray, sample_rate: int, n_steps: float) -> np.ndarray:
    """Deterministic stand-in for pyrubberband's rubberband-CLI pitch shift.

    Reads the input waveform at `ratio = 2**(n_steps/12)` speed via linear interpolation --
    a crude but genuine pitch shift (same output length, source read faster/slower) -- so
    tests exercise apply_pitch_drift's windowing/LFO/overlap-add logic against real
    pitch-shifted audio without depending on the actual (GPL, separately downloaded at setup
    time) rubberband binary being present.
    """
    if n_steps == 0.0:
        return window
    ratio = 2.0 ** (n_steps / 12.0)
    original_length = window.shape[0]
    original_index = np.arange(original_length)
    read_positions = original_index * ratio

    def _resample_1d(channel: np.ndarray) -> np.ndarray:
        return np.interp(read_positions, original_index, channel, left=0.0, right=0.0)

    if window.ndim == 1:
        return _resample_1d(window)
    return np.stack([_resample_1d(window[:, c]) for c in range(window.shape[1])], axis=1)


@pytest.fixture(autouse=True)
def _stub_rubberband(monkeypatch: pytest.MonkeyPatch) -> None:
    """Skip real binary lookup/invocation in every test; only apply_pitch_drift's own
    windowing/LFO/overlap-add orchestration logic is under test here."""
    monkeypatch.setattr(humanizer, "_configure_rubberband_binary", lambda: None)
    monkeypatch.setattr(humanizer, "_pitch_shift_window", _fake_pitch_shift_window)


def _sine(freq: float, duration_s: float, sample_rate: int, channels: int = 1) -> np.ndarray:
    t = np.arange(int(duration_s * sample_rate)) / sample_rate
    tone = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float64)
    if channels == 1:
        return tone
    return np.repeat(tone[:, np.newaxis], channels, axis=1)


def _estimate_frequency_track(audio: np.ndarray, sample_rate: int, block_ms: float = 20.0) -> np.ndarray:
    """Estimate instantaneous frequency per analysis block via sub-sample-interpolated
    zero-crossing rate (fine enough to resolve single-digit-cent deviations at ~440Hz)."""
    block_size = int(sample_rate * block_ms / 1000.0)
    estimates = []
    for start in range(0, len(audio) - block_size, block_size):
        block = audio[start : start + block_size]
        signs = np.sign(block)
        signs[signs == 0] = 1
        sign_changes = np.nonzero(np.diff(signs) != 0)[0]
        if len(sign_changes) < 4:
            continue
        crossing_times = []
        for i in sign_changes:
            y0, y1 = block[i], block[i + 1]
            frac = -y0 / (y1 - y0) if y1 != y0 else 0.0
            crossing_times.append((i + frac) / sample_rate)
        half_periods = np.diff(crossing_times)
        half_periods = half_periods[half_periods > 0]
        if len(half_periods) == 0:
            continue
        estimates.append(1.0 / (2.0 * np.mean(half_periods)))
    return np.array(estimates)


def test_intensity_zero_returns_audio_unchanged():
    audio = _sine(440.0, 0.5, 44100)
    result = humanizer.apply_pitch_drift(audio, 44100, 0.0)

    assert np.array_equal(result, audio)
    assert result is not audio


def test_negative_intensity_clamps_to_unchanged():
    audio = _sine(440.0, 0.3, 44100)
    result = humanizer.apply_pitch_drift(audio, 44100, -0.5)

    assert np.array_equal(result, audio)


def test_intensity_positive_produces_time_varying_pitch_deviation_in_expected_range():
    sample_rate = 44100
    freq = 440.0
    audio = _sine(freq, 1.0, sample_rate)

    result = humanizer.apply_pitch_drift(audio, sample_rate, 1.0)

    assert result.shape == audio.shape
    assert not np.array_equal(result, audio)

    freq_track = _estimate_frequency_track(result, sample_rate)
    assert len(freq_track) > 8, "expected enough analysis blocks to observe drift over time"

    # Time-varying: frequency estimate must actually fluctuate block-to-block, not sit flat.
    assert freq_track.std() > 0.15

    # Bounded within the expected cent-drift envelope at max intensity (+/-5 cents), with a
    # generous tolerance for zero-crossing estimation error and the fake resample-based shift.
    max_deviation_hz = freq * (2 ** (humanizer.DRIFT_DEPTH_MAX_CENTS / 1200.0) - 1)
    tolerance_hz = 2.0
    assert np.all(np.abs(freq_track - freq) <= max_deviation_hz + tolerance_hz)


def test_low_intensity_produces_smaller_deviation_than_high_intensity():
    sample_rate = 44100
    freq = 440.0
    audio = _sine(freq, 1.0, sample_rate)

    low_result = humanizer.apply_pitch_drift(audio, sample_rate, 0.1)
    high_result = humanizer.apply_pitch_drift(audio, sample_rate, 1.0)

    low_track = _estimate_frequency_track(low_result, sample_rate)
    high_track = _estimate_frequency_track(high_result, sample_rate)

    low_max_deviation = np.max(np.abs(low_track - freq))
    high_max_deviation = np.max(np.abs(high_track - freq))

    assert low_max_deviation < high_max_deviation


@pytest.mark.parametrize("sample_rate", [44100, 48000])
@pytest.mark.parametrize("channels", [1, 2])
def test_handles_mono_stereo_and_typical_sample_rates(sample_rate: int, channels: int):
    audio = _sine(220.0, 0.3, sample_rate, channels=channels)

    result = humanizer.apply_pitch_drift(audio, sample_rate, 0.5)

    assert result.shape == audio.shape
    assert result.dtype == audio.dtype
    assert np.isfinite(result).all()


def test_stereo_channels_share_a_single_lfo_curve():
    """Both channels of a stereo signal should drift together (no inter-channel phase
    divergence), matching Counsel's spec of one LFO driving the whole stem."""
    sample_rate = 44100
    audio = _sine(440.0, 0.5, sample_rate, channels=2)

    result = humanizer.apply_pitch_drift(audio, sample_rate, 1.0)

    assert np.allclose(result[:, 0], result[:, 1])


def _hf_energy(audio: np.ndarray, sample_rate: int, cutoff_hz: float = 4000.0) -> float:
    """Sum of squared magnitude spectrum above cutoff_hz, used to measure high-frequency
    (breath/friction) energy content of a signal or signal segment."""
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(len(audio), d=1.0 / sample_rate)
    mask = freqs >= cutoff_hz
    return float(np.sum(np.abs(spectrum[mask]) ** 2))


def _make_vocal_with_breath_gap(sample_rate: int, duration_s: float) -> tuple[np.ndarray, slice]:
    """A low-frequency "sung" tone with a silent gap in the middle, simulating a breath pause
    where BS-RoFormer would have stripped out breath/friction detail entirely."""
    num_samples = int(duration_s * sample_rate)
    t = np.arange(num_samples) / sample_rate
    tone = 0.4 * np.sin(2 * np.pi * 220.0 * t)
    gap = slice(num_samples // 2 - int(0.05 * sample_rate), num_samples // 2 + int(0.05 * sample_rate))
    tone[gap] = 0.0
    return tone, gap


def _make_broadband_residual(sample_rate: int, duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Broadband noise standing in for the real leftover breath/hiss residual: present across
    the whole clip (including the vocal's silent/breath gap), unlike the vocal tone itself."""
    num_samples = int(duration_s * sample_rate)
    return rng.normal(scale=0.5, size=num_samples)


def test_breath_blend_adds_high_frequency_energy_in_silent_breath_region():
    sample_rate = 44100
    duration_s = 1.0
    rng = np.random.default_rng(42)

    vocal, gap = _make_vocal_with_breath_gap(sample_rate, duration_s)
    residual = _make_broadband_residual(sample_rate, duration_s, rng)

    blended = humanizer.apply_breath_blend(vocal, residual, sample_rate)

    unblended_gap_hf = _hf_energy(vocal[gap], sample_rate)
    blended_gap_hf = _hf_energy(blended[gap], sample_rate)

    assert blended_gap_hf > unblended_gap_hf * 10


def test_breath_blend_does_not_clip():
    sample_rate = 44100
    duration_s = 0.5

    # A vocal with realistic headroom (pre-final-limiting, as it is at this point in the
    # pipeline -- see app.core.remix_master for the true-peak limiter applied later) plus a
    # residual at a level typical of real stem-separation leftover (quiet breath/hiss, not
    # full-scale noise): the fixed -18dB gain should keep the blend well clear of clipping.
    num_samples = int(duration_s * sample_rate)
    t = np.arange(num_samples) / sample_rate
    vocal = 0.8 * np.sin(2 * np.pi * 440.0 * t)

    for seed in range(20):
        residual = np.random.default_rng(seed).normal(scale=0.1, size=num_samples)
        blended = humanizer.apply_breath_blend(vocal, residual, sample_rate)

        assert np.isfinite(blended).all()
        assert np.max(np.abs(blended)) <= 1.0


def test_breath_blend_intensity_is_fixed_not_configurable():
    """apply_breath_blend takes no intensity/gain parameter -- the mix amount is controlled
    solely by the module-level BREATH_BLEND_GAIN_DB constant, per Counsel's spec that this
    stage is automatic and not user-facing."""
    import inspect

    params = list(inspect.signature(humanizer.apply_breath_blend).parameters)
    assert params == ["processed_vocal", "residual_signal", "sample_rate"]


def test_breath_blend_handles_stereo_and_length_mismatch():
    sample_rate = 44100
    vocal = _sine(220.0, 0.3, sample_rate, channels=2)
    residual = np.random.default_rng(1).normal(scale=0.3, size=(vocal.shape[0] - 100, 2))

    result = humanizer.apply_breath_blend(vocal, residual, sample_rate)

    assert result.shape == (vocal.shape[0] - 100, 2)
    assert np.isfinite(result).all()
