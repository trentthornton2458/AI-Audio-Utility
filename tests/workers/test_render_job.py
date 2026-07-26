"""Tests for app.workers.render_job (RenderJob), including execution, cancellation, and error signaling."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

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
from app.core import qa_gate
from app.models.app_config import AppConfig
from app.models.preset import Preset
from app.workers.render_job import RenderJob


def _patched_pipeline(tmp_path: Path, dummy_audio: np.ndarray):
    """Common set of patches standing in for every pipeline stage RenderJob._render calls,
    updated for the denoise -> DSP -> enhance -> QA-gated-blend order."""
    qa_result = qa_gate.QAGateResult(audio=dummy_audio, samplerate=44100, qa_flags=[])
    return [
        patch(
            "app.core.ingestion.load_and_normalize_track",
            return_value=tmp_path / "cache" / "track123" / "input.wav",
        ),
        patch("app.core.separation.separate_stems", return_value=(tmp_path / "vocal.wav", tmp_path / "inst.wav")),
        patch("app.core.vocal_chain.run_denoise_pass", return_value=tmp_path / "n_vocal.wav"),
        patch("app.core.vocal_chain.apply_dsp_chain"),
        patch("app.core.vocal_chain.run_enhance_pass", return_value=tmp_path / "e_vocal.wav"),
        patch("app.core.vocal_chain.blend_vocal", return_value=qa_result),
        patch("app.core.instrumental_chain.run_denoise_pass", return_value=tmp_path / "n_inst.wav"),
        patch("app.core.instrumental_chain.apply_dsp_chain"),
        patch("app.core.instrumental_chain.run_enhance_pass", return_value=tmp_path / "e_inst.wav"),
        patch("app.core.instrumental_chain.blend_instrumental", return_value=qa_result),
        patch("app.models.gemini_settings.get_gemini_api_key", return_value=None),
        patch("app.core.remix_master.mix_stems", return_value=dummy_audio),
        patch("app.core.remix_master.master", return_value=dummy_audio),
        patch("app.core.remix_master.export_wav", side_effect=lambda audio, sr, p: p),
    ]


def test_render_job_writes_metadata(tmp_path: Path):
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    preset = Preset(notch_depth_db=5.0)
    job = RenderJob(input_path=input_path, preset=preset, cache_manager=cache_mgr)
    dummy_audio = np.zeros((44100, 2), dtype=np.float64)

    patchers = _patched_pipeline(tmp_path, dummy_audio)
    with patchers[0], patchers[1], patchers[2], patchers[3], patchers[4], patchers[5], \
         patchers[6], patchers[7], patchers[8], patchers[9], patchers[10], patchers[11], \
         patchers[12], patchers[13]:

        output_path = job._render()

        assert output_path.exists() or output_path.parent.exists()
        meta_path = output_path.with_suffix(".json")
        assert meta_path.exists()

        data = json.loads(meta_path.read_text(encoding="utf-8"))
        assert "timestamp" in data
        assert "vocal_clean_intensity" not in data["preset"]
        assert data["preset"]["notch_depth_db"] == 5.0
        assert "qa_flags" in data
        assert data["qa_flags"] == []


def test_render_job_cancellation(tmp_path: Path):
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    preset = Preset()
    job = RenderJob(input_path=input_path, preset=preset, cache_manager=cache_mgr)

    cancelled_emitted = []
    job.cancelled.connect(lambda: cancelled_emitted.append(True))

    with patch.object(job, "isInterruptionRequested", return_value=True):
        job.run()

    assert len(cancelled_emitted) == 1


def test_render_job_failure_signal(tmp_path: Path):
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    preset = Preset()
    job = RenderJob(input_path=input_path, preset=preset, cache_manager=cache_mgr)

    failed_messages = []
    job.failed.connect(lambda msg: failed_messages.append(msg))

    with patch("app.core.ingestion.load_and_normalize_track", side_effect=ValueError("Corrupt WAV format")):
        job.run()

    assert len(failed_messages) == 1
    assert "Corrupt WAV format" in failed_messages[0]


def test_render_job_progress_emission(tmp_path: Path):
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    preset = Preset()
    job = RenderJob(input_path=input_path, preset=preset, cache_manager=cache_mgr)
    dummy_audio = np.zeros((44100, 2), dtype=np.float64)
    qa_result = qa_gate.QAGateResult(audio=dummy_audio, samplerate=44100, qa_flags=[])

    progress_vals = []
    job.progressChanged.connect(progress_vals.append)

    with patch("app.core.ingestion.load_and_normalize_track", return_value=tmp_path / "cache" / "track123" / "input.wav"), \
         patch("app.core.separation.separate_stems", return_value=(tmp_path / "vocal.wav", tmp_path / "inst.wav")), \
         patch("app.core.vocal_chain.run_denoise_pass", return_value=tmp_path / "n_vocal.wav") as mock_v_denoise, \
         patch("app.core.vocal_chain.apply_dsp_chain"), \
         patch("app.core.vocal_chain.run_enhance_pass", return_value=tmp_path / "e_vocal.wav") as mock_v_enhance, \
         patch("app.core.vocal_chain.blend_vocal", return_value=qa_result), \
         patch("app.core.instrumental_chain.run_denoise_pass", return_value=tmp_path / "n_inst.wav") as mock_i_denoise, \
         patch("app.core.instrumental_chain.apply_dsp_chain"), \
         patch("app.core.instrumental_chain.run_enhance_pass", return_value=tmp_path / "e_inst.wav") as mock_i_enhance, \
         patch("app.core.instrumental_chain.blend_instrumental", return_value=qa_result), \
         patch("app.models.gemini_settings.get_gemini_api_key", return_value=None), \
         patch("app.core.remix_master.mix_stems", return_value=dummy_audio), \
         patch("app.core.remix_master.master", return_value=dummy_audio), \
         patch("app.core.remix_master.export_wav", side_effect=lambda audio, sr, p: p):

        output_path = job._render()

        assert len(progress_vals) > 0
        for i in range(len(progress_vals) - 1):
            assert progress_vals[i] <= progress_vals[i + 1]

        for mock_pass in (mock_v_denoise, mock_v_enhance, mock_i_denoise, mock_i_enhance):
            mock_pass.assert_called_once()
            assert "progress_callback" in mock_pass.call_args[1]
            assert "is_cancelled" in mock_pass.call_args[1]


def test_render_job_cancellation_removes_partial_files(tmp_path: Path):
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    preset = Preset()
    job = RenderJob(input_path=input_path, preset=preset, cache_manager=cache_mgr)

    file1 = tmp_path / "partial_stem.wav"
    file2 = tmp_path / "partial_output.wav"

    file1.touch()
    file2.touch()

    assert file1.exists()
    assert file2.exists()

    job._active_files.add(file1)
    job._active_files.add(file2)

    cancelled_emitted = []
    job.cancelled.connect(lambda: cancelled_emitted.append(True))

    from app.workers.render_job import _JobCancelled
    with patch.object(job, "_render", side_effect=_JobCancelled):
        job.run()

    assert len(cancelled_emitted) == 1
    assert not file1.exists()
    assert not file2.exists()


def test_render_job_failure_removes_partial_files(tmp_path: Path):
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    preset = Preset()
    job = RenderJob(input_path=input_path, preset=preset, cache_manager=cache_mgr)

    file1 = tmp_path / "partial_stem.wav"
    file1.touch()
    assert file1.exists()

    job._active_files.add(file1)

    failed_messages = []
    job.failed.connect(failed_messages.append)

    with patch.object(job, "_render", side_effect=ValueError("Unexpected DSP error")):
        job.run()

    assert len(failed_messages) == 1
    assert "Unexpected DSP error" in failed_messages[0]
    assert not file1.exists()
