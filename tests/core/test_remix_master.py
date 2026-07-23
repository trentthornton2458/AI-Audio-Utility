"""Tests for stem mixing, LUFS mastering, and WAV export."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from app.core.remix_master import export_wav, master, mix_stems


def test_mix_stems_with_gain():
    vocal = np.ones((100, 2), dtype=np.float64) * 0.5
    instrumental = np.ones((100, 2), dtype=np.float64) * 0.5

    # 0dB gains -> simple sum (1.0)
    mix_0db = mix_stems(vocal, instrumental, vocal_gain_db=0.0, instrumental_gain_db=0.0)
    np.testing.assert_allclose(mix_0db, 1.0, atol=1e-5)

    # -6dB gain (~0.5 linear) for vocal, 0dB for instrumental -> 0.25 + 0.5 = 0.75
    mix_gained = mix_stems(vocal, instrumental, vocal_gain_db=-6.0206, instrumental_gain_db=0.0)
    np.testing.assert_allclose(mix_gained, 0.75, atol=1e-3)


def test_mix_stems_padding_length_mismatch():
    vocal = np.ones((100, 2), dtype=np.float64) * 0.5
    instrumental = np.ones((150, 2), dtype=np.float64) * 0.5

    mixed = mix_stems(vocal, instrumental, vocal_gain_db=0.0, instrumental_gain_db=0.0)
    assert mixed.shape == (150, 2)


def test_master_and_export(tmp_path: Path):
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.4 * np.sin(2 * np.pi * 440 * t)[:, np.newaxis]
    signal = np.repeat(signal, 2, axis=1)

    mastered = master(signal, sample_rate=sr, lufs_target=-14.0)
    assert mastered.shape == signal.shape
    assert not np.isnan(mastered).any()

    out_file = tmp_path / "mastered.wav"
    exported = export_wav(mastered, sr, out_file)
    assert exported.is_file()

    read_data, read_sr = sf.read(str(exported))
    assert read_sr == sr
    assert read_data.shape == signal.shape
