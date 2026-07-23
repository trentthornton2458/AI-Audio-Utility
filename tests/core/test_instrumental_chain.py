"""Tests for instrumental DSP chain."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from app.core.instrumental_chain import InstrumentalEqParams, apply_dsp_chain


def test_apply_instrumental_dsp_chain(tmp_path: Path):
    sr = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 100 * t)[:, np.newaxis]
    audio = np.repeat(audio, 2, axis=1)  # Stereo

    in_file = tmp_path / "inst_in.wav"
    sf.write(str(in_file), audio, sr, subtype="PCM_24")

    out_file = tmp_path / "inst_dsp.wav"
    params = InstrumentalEqParams(mud_cut_hz=50.0, dehiss_shelf_hz=8000.0, dehiss_gain_db=-3.0)

    res_path = apply_dsp_chain(in_file, params, out_file)
    assert res_path.is_file()

    data, out_sr = sf.read(str(res_path))
    assert out_sr == sr
    assert data.shape == audio.shape
