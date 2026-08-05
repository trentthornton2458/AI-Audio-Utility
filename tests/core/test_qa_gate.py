"""Tests for app.core.qa_gate: the deterministic spectral QA gate governing how much of the
post-DSP AI-enhance signal gets blended back into a stem (capped energy-normalized residual
blend, per-window fail-safes, and optional non-fatal Gemini diagnostic escalation)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.core import gemini_qa, qa_gate
from app.models.app_config import QAGateThresholds


def _write_wav(path: Path, samples: np.ndarray, sr: int) -> np.ndarray:
    sf.write(str(path), samples, sr, subtype="PCM_24")
    return samples


def _sine(freq: float, sr: int, seconds: float, amplitude: float = 0.5) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def test_zero_gain_returns_dsp_signal_unchanged(tmp_path: Path):
    sr = 8000
    dsp_audio = _write_wav(tmp_path / "dsp.wav", _sine(440, sr, 1.0), sr)
    _write_wav(tmp_path / "enhanced.wav", _sine(460, sr, 1.0, amplitude=0.9), sr)

    result = qa_gate.apply_qa_gated_blend(
        tmp_path / "dsp.wav",
        tmp_path / "enhanced.wav",
        max_gain=0.0,
        stem_label="vocal",
    )

    np.testing.assert_allclose(result.audio[:, 0], dsp_audio, atol=1e-4)
    assert result.qa_flags == []


def test_identical_dsp_and_enhanced_yields_no_change(tmp_path: Path):
    sr = 8000
    dsp_path = tmp_path / "dsp.wav"
    dsp_audio = _write_wav(dsp_path, _sine(440, sr, 1.0), sr)

    # enhanced_path == dsp_path is exactly what app.core.vocal_chain.run_enhance_pass returns
    # when the enhance stage is disabled -- must be a strict no-op regardless of gain.
    result = qa_gate.apply_qa_gated_blend(
        dsp_path, dsp_path, max_gain=0.35, stem_label="vocal"
    )

    np.testing.assert_allclose(result.audio[:, 0], dsp_audio, atol=1e-6)
    assert result.qa_flags == []


def test_full_cap_matches_energy_normalized_residual_formula(tmp_path: Path):
    sr = 8000
    dsp_audio = _sine(440, sr, 1.0, amplitude=0.5)
    # A different-shaped perturbation plus an amplitude mismatch (scaling doesn't move
    # flatness/centroid/crest-factor/sibilance-ratio, all of which are scale-invariant, so this
    # stays under every soft threshold) so the level-matching normalization actually engages,
    # making the residual-blend formula provably diverge from a naive crossfade.
    enhanced_audio = (dsp_audio + 0.05 * _sine(220, sr, 1.0, amplitude=1.0)) * 1.4

    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(dsp_path, dsp_audio, sr)
    _write_wav(enhanced_path, enhanced_audio, sr)

    max_gain = 0.35
    result = qa_gate.apply_qa_gated_blend(
        dsp_path, enhanced_path, max_gain=max_gain, stem_label="vocal"
    )

    # No threshold breaches expected for this mild, low-frequency-only perturbation, so the
    # gate should apply the full requested gain uniformly (one window covers the whole file).
    assert result.qa_flags == []

    dsp_rms = qa_gate._rms(dsp_audio)
    enhanced_rms = qa_gate._rms(enhanced_audio)
    scale = np.clip(dsp_rms / max(enhanced_rms, 1e-4), 0.1, 10.0)
    expected = dsp_audio + max_gain * (enhanced_audio * scale - dsp_audio)

    np.testing.assert_allclose(result.audio[:, 0], expected, atol=1e-3)
    # And it must differ from a naive full crossfade (intensity*wet + (1-intensity)*dry).
    naive_crossfade = (1 - max_gain) * dsp_audio + max_gain * enhanced_audio
    assert not np.allclose(result.audio[:, 0], naive_crossfade, atol=1e-3)


def _multi_window_thresholds(
    window_seconds: float = 0.1, hop_seconds: float = 0.1
) -> QAGateThresholds:
    return QAGateThresholds(window_seconds=window_seconds, hop_seconds=hop_seconds)


def test_silence_window_forces_zero_gain(tmp_path: Path):
    sr = 8000
    duration = 1.0
    dsp_audio = _sine(440, sr, duration)
    enhanced_audio = dsp_audio.copy()

    # Zero out one 0.1s window (samples 4000:4800) in both signals.
    dsp_audio[4000:4800] = 0.0
    enhanced_audio[4000:4800] = 0.0

    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(dsp_path, dsp_audio, sr)
    _write_wav(enhanced_path, enhanced_audio, sr)

    result = qa_gate.apply_qa_gated_blend(
        dsp_path,
        enhanced_path,
        max_gain=0.35,
        stem_label="vocal",
        thresholds=_multi_window_thresholds(),
    )

    silence_flags = [f for f in result.qa_flags if f.reason == "silence"]
    assert len(silence_flags) >= 1
    flag = silence_flags[0]
    assert flag.final_gain == 0.0
    assert 4000 / sr <= flag.start_seconds < flag.end_seconds <= 4800 / sr + 0.11


def test_clipping_window_forces_zero_gain(tmp_path: Path):
    sr = 8000
    duration = 1.0
    dsp_audio = _sine(440, sr, duration)
    enhanced_audio = dsp_audio.copy()

    # Push one window's enhanced samples to (near) full-scale to trigger the clipping fail-safe.
    enhanced_audio[4000:4800] = 1.0

    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(dsp_path, dsp_audio, sr)
    _write_wav(enhanced_path, enhanced_audio, sr)

    result = qa_gate.apply_qa_gated_blend(
        dsp_path,
        enhanced_path,
        max_gain=0.35,
        stem_label="vocal",
        thresholds=_multi_window_thresholds(),
    )

    clipping_flags = [f for f in result.qa_flags if f.reason == "clipping"]
    assert len(clipping_flags) >= 1
    assert clipping_flags[0].final_gain == 0.0


def test_spectral_flatness_collapse_flagged_as_hallucination_proxy(tmp_path: Path):
    sr = 8000
    duration = 1.0
    rng = np.random.default_rng(0)

    # Background: identical white noise in both signals everywhere (no artifact).
    background = rng.uniform(-0.3, 0.3, size=int(sr * duration))
    dsp_audio = background.copy()
    enhanced_audio = background.copy()

    # One window: DSP stays noise-like (high flatness), enhanced becomes a pure tone (low
    # flatness) at a similar RMS level -- the hallucinated-tone proxy.
    window = slice(4000, 4800)
    t = np.arange(800) / sr
    enhanced_audio[window] = 0.2 * np.sin(2 * np.pi * 300 * t)

    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(dsp_path, dsp_audio, sr)
    _write_wav(enhanced_path, enhanced_audio, sr)

    result = qa_gate.apply_qa_gated_blend(
        dsp_path,
        enhanced_path,
        max_gain=0.35,
        stem_label="vocal",
        thresholds=_multi_window_thresholds(),
    )

    hallucination_flags = [
        f for f in result.qa_flags if f.reason == "hallucination_proxy"
    ]
    assert len(hallucination_flags) >= 1
    assert hallucination_flags[0].final_gain == 0.0


def test_gemini_diagnostic_non_fatal_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sr = 8000
    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(dsp_path, _sine(440, sr, 1.0), sr)
    _write_wav(enhanced_path, _sine(460, sr, 1.0), sr)

    monkeypatch.setattr(
        qa_gate,
        "_evaluate_window",
        lambda metrics, thresholds, max_gain: (0.2, "spectral_centroid_delta"),
    )

    def _raise(*args, **kwargs):
        raise gemini_qa.GeminiAnalysisError("network unavailable")

    monkeypatch.setattr(qa_gate.gemini_qa, "diagnose_qa_window", _raise)

    result = qa_gate.apply_qa_gated_blend(
        dsp_path,
        enhanced_path,
        max_gain=0.35,
        stem_label="vocal",
        gemini_api_key="fake-key",
    )

    assert len(result.qa_flags) == 1
    flag = result.qa_flags[0]
    assert flag.deterministic_gain == 0.2
    assert flag.final_gain == 0.2
    assert flag.gemini_multiplier is None
    assert flag.gemini_verdict is None


def test_gemini_diagnostic_applied_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sr = 8000
    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(dsp_path, _sine(440, sr, 1.0), sr)
    _write_wav(enhanced_path, _sine(460, sr, 1.0), sr)

    monkeypatch.setattr(
        qa_gate,
        "_evaluate_window",
        lambda metrics, thresholds, max_gain: (0.2, "spectral_centroid_delta"),
    )
    monkeypatch.setattr(
        qa_gate.gemini_qa, "diagnose_qa_window", lambda *a, **k: (0.5, "mild_artifact")
    )

    result = qa_gate.apply_qa_gated_blend(
        dsp_path,
        enhanced_path,
        max_gain=0.35,
        stem_label="vocal",
        gemini_api_key="fake-key",
    )

    assert len(result.qa_flags) == 1
    flag = result.qa_flags[0]
    assert flag.deterministic_gain == 0.2
    assert flag.final_gain == pytest.approx(0.1)
    assert flag.gemini_multiplier == 0.5
    assert flag.gemini_verdict == "mild_artifact"


def test_gemini_not_called_without_api_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    sr = 8000
    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_wav(dsp_path, _sine(440, sr, 1.0), sr)
    _write_wav(enhanced_path, _sine(460, sr, 1.0), sr)

    monkeypatch.setattr(
        qa_gate,
        "_evaluate_window",
        lambda metrics, thresholds, max_gain: (0.2, "spectral_centroid_delta"),
    )

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("Gemini should not be called without an api_key")

    monkeypatch.setattr(qa_gate.gemini_qa, "diagnose_qa_window", _fail_if_called)

    result = qa_gate.apply_qa_gated_blend(
        dsp_path, enhanced_path, max_gain=0.35, stem_label="vocal", gemini_api_key=None
    )

    assert result.qa_flags[0].gemini_multiplier is None


def test_spectral_flatness_of_pure_tone_is_low():
    sr = 8000
    tone = _sine(440, sr, 0.5, amplitude=0.8)
    assert qa_gate._spectral_flatness(tone, sr) < 0.2


def test_spectral_flatness_of_white_noise_is_high():
    sr = 8000
    noise = np.random.default_rng(1).uniform(-0.5, 0.5, size=int(sr * 0.5))
    assert qa_gate._spectral_flatness(noise, sr) > 0.5


def test_crest_factor_of_sine_matches_expected_value():
    sr = 8000
    tone = _sine(440, sr, 1.0, amplitude=1.0)
    expected_db = 20.0 * np.log10(np.sqrt(2.0))
    assert qa_gate._crest_factor(tone) == pytest.approx(expected_db, abs=0.2)


def test_sibilance_energy_ratio_detects_high_frequency_content():
    sr = 44100
    low_tone = _sine(200, sr, 0.5, amplitude=0.8)
    high_tone = _sine(12000, sr, 0.5, amplitude=0.8)

    low_ratio = qa_gate._sibilance_energy_ratio(low_tone, sr)
    high_ratio = qa_gate._sibilance_energy_ratio(high_tone, sr)

    assert low_ratio < 0.05
    assert high_ratio > 0.8


# --- measure_pitch_variance ---------------------------------------------------------------


def test_measure_pitch_variance_flags_pure_tone_as_flat_pitch():
    sr = 22050
    tone = _sine(220, sr, 1.0, amplitude=0.5)

    result = qa_gate.measure_pitch_variance(tone, sr)

    assert result.warning is True
    assert result.reason == "flat_or_hard_quantized_pitch"
    assert result.value < qa_gate.PITCH_VARIANCE_WARN_CENTS_MAX


def test_measure_pitch_variance_does_not_flag_natural_vibrato():
    sr = 22050
    duration = 1.0
    n = int(sr * duration)
    t = np.arange(n) / sr

    # +-50 cent, 5Hz vibrato -- well beyond the pitch tracker's own few-cent quantization noise
    # floor (see PITCH_VARIANCE_WARN_CENTS_MAX's comment in qa_gate.py).
    cents_mod = 50.0 * np.sin(2 * np.pi * 5 * t)
    freq_inst = 220.0 * (2.0 ** (cents_mod / 1200.0))
    phase = 2 * np.pi * np.cumsum(freq_inst) / sr
    vibrato = (0.5 * np.sin(phase)).astype(np.float64)

    result = qa_gate.measure_pitch_variance(vibrato, sr)

    assert result.warning is False
    assert result.reason == ""
    assert result.value > qa_gate.PITCH_VARIANCE_WARN_CENTS_MAX


# --- measure_high_frequency_energy -------------------------------------------------------


def test_measure_high_frequency_energy_flags_silence_as_stripped_breath():
    sr = 44100
    silence = np.zeros(int(sr * 0.5), dtype=np.float64)

    result = qa_gate.measure_high_frequency_energy(
        silence, sr, silence_threshold_db=-30.0
    )

    assert result.warning is True
    assert result.reason == "stripped_breath_detail"
    assert result.value == pytest.approx(0.0)


def test_measure_high_frequency_energy_does_not_flag_hissy_breath_section():
    sr = 44100
    rng = np.random.default_rng(2)
    loud_vocal = _sine(220, sr, 0.5, amplitude=0.8)
    # Quiet (well under the threshold), but HF-rich white noise -- a realistic breath/room-tone
    # section that should NOT be flagged as stripped.
    quiet_breath = (rng.uniform(-1.0, 1.0, size=int(sr * 0.5)) * 0.01).astype(
        np.float64
    )
    audio = np.concatenate([loud_vocal, quiet_breath])

    result = qa_gate.measure_high_frequency_energy(
        audio, sr, silence_threshold_db=-30.0
    )

    assert result.warning is False
    assert result.reason == ""
    assert result.value > qa_gate.HF_ENERGY_WARN_RATIO_MAX


# --- measure_crest_factor -----------------------------------------------------------------


def test_measure_crest_factor_flags_brickwalled_waveform():
    sr = 8000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine = np.sin(2 * np.pi * 220 * t)
    brickwalled = np.clip(
        sine * 20.0, -1.0, 1.0
    )  # hard-clipped into a near-square wave

    result = qa_gate.measure_crest_factor(brickwalled)

    assert result.warning is True
    assert result.reason == "over_compressed"
    assert result.value < qa_gate.CREST_FACTOR_WARN_DB_MIN


def test_measure_crest_factor_does_not_flag_normal_dynamic_range():
    sr = 8000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    tone = 0.3 * np.sin(2 * np.pi * 220 * t)
    tone[::50] = 0.95  # occasional peaks well above the sustained RMS level

    result = qa_gate.measure_crest_factor(tone)

    assert result.warning is False
    assert result.reason == ""
    assert result.value > qa_gate.CREST_FACTOR_WARN_DB_MIN
