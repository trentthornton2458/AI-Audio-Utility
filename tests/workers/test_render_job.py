"""Tests for app.workers.render_job (RenderJob)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure missing heavy ML dependencies do not break UI test import collection
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
from app.cache.cache_manager import CacheManager
from app.models.app_config import AppConfig
from app.models.preset import Preset
from app.workers.render_job import RenderJob


def test_render_job_writes_metadata(tmp_path: Path):
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    preset = Preset(vocal_clean_intensity=0.9, notch_depth_db=5.0)

    job = RenderJob(input_path=input_path, preset=preset, cache_manager=cache_mgr)

    dummy_audio = np.zeros((44100, 2), dtype=np.float64)

    with patch("app.core.ingestion.load_and_normalize_track", return_value=tmp_path / "cache" / "track123" / "input.wav"), \
         patch("app.core.separation.separate_stems", return_value=(tmp_path / "vocal.wav", tmp_path / "inst.wav")), \
         patch("app.core.vocal_chain.run_neural_pass", return_value=tmp_path / "n_vocal.wav"), \
         patch("app.core.vocal_chain.apply_dsp_chain"), \
         patch("app.core.vocal_chain.blend_vocal", return_value=dummy_audio), \
         patch("app.core.instrumental_chain.run_neural_pass", return_value=tmp_path / "n_inst.wav"), \
         patch("app.core.instrumental_chain.apply_dsp_chain"), \
         patch("soundfile.info", return_value=MagicMock(samplerate=44100)), \
         patch("soundfile.read", return_value=(dummy_audio, 44100)), \
         patch("app.core.remix_master.mix_stems", return_value=dummy_audio), \
         patch("app.core.remix_master.master", return_value=dummy_audio), \
         patch("app.core.remix_master.export_wav", side_effect=lambda audio, sr, p: p):

        output_path = job._render()

        assert output_path.exists() or output_path.parent.exists()
        meta_path = output_path.with_suffix(".json")
        assert meta_path.exists()

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert data["preset"]["vocal_clean_intensity"] == 0.9
        assert data["preset"]["notch_depth_db"] == 5.0
