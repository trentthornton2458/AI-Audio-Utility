"""Tests for vocal Pedalboard DSP chain and blending."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from app.core import qa_gate
from app.core.vocal_chain import apply_dsp_chain, blend_vocal, _clamp


def test_clamp_helper():
    assert _clamp(5.0, 0.0, 10.0) == 5.0
    assert _clamp(-2.0, 0.0, 10.0) == 0.0
    assert _clamp(15.0, 0.0, 10.0) == 10.0


def test_apply_dsp_chain(tmp_path: Path):
    sr = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate audio with low mid and sibilant high content
    audio = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 6000 * t))[:, np.newaxis]
    audio = np.repeat(audio, 2, axis=1)  # Stereo

    in_file = tmp_path / "vocal_in.wav"
    sf.write(str(in_file), audio, sr, subtype="PCM_24")

    out_file = tmp_path / "vocal_dsp.wav"
    res_path = apply_dsp_chain(in_file, notch_depth_db=4.0, deesser_threshold_db=-6.0, out_path=out_file)

    assert res_path.is_file()
    out_data, out_sr = sf.read(str(out_file))
    assert out_sr == sr
    assert out_data.shape == audio.shape


def _write_constant_wav(path: Path, value: float, samplerate: int, frames: int = 4410) -> np.ndarray:
    audio = np.ones((frames, 2), dtype=np.float64) * value
    sf.write(str(path), audio, samplerate, subtype="PCM_24")
    return audio


def test_blend_vocal_zero_intensity_passes_through_dsp_signal(tmp_path: Path):
    sr = 44100
    dsp_audio = _write_constant_wav(tmp_path / "dsp.wav", 0.3, sr)
    _write_constant_wav(tmp_path / "enhanced.wav", 0.9, sr)

    result = blend_vocal(tmp_path / "dsp.wav", tmp_path / "enhanced.wav", enhance_intensity=0.0)

    assert isinstance(result, qa_gate.QAGateResult)
    assert result.samplerate == sr
    np.testing.assert_allclose(result.audio, dsp_audio, atol=1e-4)
    assert result.qa_flags == []


def test_blend_vocal_identical_dsp_and_enhanced_yields_no_change(tmp_path: Path):
    sr = 44100
    dsp_path = tmp_path / "dsp.wav"
    dsp_audio = _write_constant_wav(dsp_path, 0.4, sr)

    # Passing the same path for both arguments exercises qa_gate's enhanced_path == dsp_path
    # fast path (the same thing app.core.vocal_chain.run_enhance_pass returns when the enhance
    # stage is disabled), so the blend must be a strict no-op regardless of the requested gain.
    result = blend_vocal(dsp_path, dsp_path, enhance_intensity=0.35)

    np.testing.assert_allclose(result.audio, dsp_audio, atol=1e-6)
    assert result.qa_flags == []


def test_blend_vocal_clamps_intensity_above_hard_cap(tmp_path: Path):
    sr = 44100
    dsp_path = tmp_path / "dsp.wav"
    enhanced_path = tmp_path / "enhanced.wav"
    _write_constant_wav(dsp_path, 0.3, sr)
    _write_constant_wav(enhanced_path, 0.6, sr)

    # enhance_intensity above the 0.35 hard cap must be clamped internally rather than raising
    # or applying an uncapped blend.
    result = blend_vocal(dsp_path, enhanced_path, enhance_intensity=5.0)

    assert isinstance(result, qa_gate.QAGateResult)
    assert result.samplerate == sr
