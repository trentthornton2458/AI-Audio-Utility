"""Full pipeline integration test mocking out heavy neural/model calls but executing all DSP and file operations."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

for mod_name in [
    "resemble_enhance",
    "resemble_enhance.enhancer",
    "resemble_enhance.enhancer.inference",
    "audio_separator",
    "audio_separator.separator",
]:
    if mod_name not in sys.modules:
        sys.modules[mod_name] = MagicMock()

import numpy as np
import pytest
import soundfile as sf

from app.cache.cache_manager import CacheManager
from app.core.ingestion import load_and_normalize_track
from app.core.instrumental_chain import InstrumentalEqParams, apply_dsp_chain as apply_inst_dsp
from app.core.remix_master import export_wav, master, mix_stems
from app.core.vocal_chain import apply_dsp_chain as apply_vocal_dsp, blend_vocal
from app.models.app_config import AppConfig
from app.models.preset import Preset


def test_full_pipeline_end_to_end(tmp_path: Path):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))

    # 1. Create a dummy input audio file
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)[:, np.newaxis]
    audio = np.repeat(audio, 2, axis=1)

    input_file = tmp_path / "test_input.wav"
    sf.write(str(input_file), audio, sr, subtype="PCM_16")

    # 2. Ingestion
    norm_path = load_and_normalize_track(input_file, cache_mgr)
    assert norm_path.is_file()

    track_id = norm_path.parent.name
    stems_dir = cache_mgr.stems_dir(track_id)
    vocal_stem = stems_dir / "vocal.wav"
    inst_stem = stems_dir / "instrumental.wav"

    # Simulate stem separation output
    sf.write(str(vocal_stem), audio * 0.4, sr, subtype="PCM_24")
    sf.write(str(inst_stem), audio * 0.6, sr, subtype="PCM_24")

    # 3. Neural passes (simulated by returning stem paths)
    n_vocal = vocal_stem
    n_inst = inst_stem

    # 4. Vocal DSP & Blend
    preset = Preset(notch_depth_db=4.5, vocal_clean_intensity=0.8)
    dsp_vocal = stems_dir / "vocal_dsp.wav"
    apply_vocal_dsp(n_vocal, preset.notch_depth_db, dsp_vocal)

    vocal_blended = blend_vocal(n_vocal, dsp_vocal, preset.vocal_clean_intensity)

    # 5. Instrumental DSP
    eq_params = InstrumentalEqParams(
        mud_cut_hz=preset.instrumental_mud_cut_hz,
        dehiss_shelf_hz=preset.instrumental_dehiss_shelf_hz,
        dehiss_gain_db=preset.instrumental_dehiss_gain_db,
    )
    dsp_inst = stems_dir / "instrumental_dsp.wav"
    apply_inst_dsp(n_inst, eq_params, dsp_inst)

    inst_audio, read_sr = sf.read(str(dsp_inst), always_2d=True, dtype="float64")

    # 6. Remix & Master
    mixed = mix_stems(vocal_blended, inst_audio, preset.vocal_gain_db, preset.instrumental_gain_db)
    mastered = master(mixed, sr, preset.lufs_target)

    export_path = cache_mgr.renders_dir(track_id) / "final_master.wav"
    exported = export_wav(mastered, sr, export_path)

    assert exported.is_file()
    final_data, final_sr = sf.read(str(exported))
    assert final_sr == sr
    assert final_data.shape == audio.shape
