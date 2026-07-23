"""Tests for vocal Pedalboard DSP chain and blending."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

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
    res_path = apply_dsp_chain(in_file, notch_depth_db=4.0, out_path=out_file)

    assert res_path.is_file()
    out_data, out_sr = sf.read(str(out_file))
    assert out_sr == sr
    assert out_data.shape == audio.shape


def test_blend_vocal(tmp_path: Path):
    sr = 44100
    audio_a = np.ones((100, 2), dtype=np.float64) * 0.2
    audio_b = np.ones((100, 2), dtype=np.float64) * 0.8

    path_a = tmp_path / "a.wav"
    path_b = tmp_path / "b.wav"
    sf.write(str(path_a), audio_a, sr, subtype="PCM_24")
    sf.write(str(path_b), audio_b, sr, subtype="PCM_24")

    # blend 0.0 -> pure audio_a
    blend_0 = blend_vocal(path_a, path_b, clean_intensity=0.0)
    np.testing.assert_allclose(blend_0, audio_a, atol=1e-4)

    # blend 1.0 -> pure audio_b
    blend_1 = blend_vocal(path_a, path_b, clean_intensity=1.0)
    np.testing.assert_allclose(blend_1, audio_b, atol=1e-4)

    # blend 0.5 -> 50% midpoint
    blend_mid = blend_vocal(path_a, path_b, clean_intensity=0.5)
    np.testing.assert_allclose(blend_mid, 0.5 * audio_a + 0.5 * audio_b, atol=1e-4)
