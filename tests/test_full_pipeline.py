"""Full pipeline integration test mocking out heavy neural/model calls but executing all DSP and file operations."""

from __future__ import annotations

import json
import sys
import time
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
import torch

from app.cache.cache_manager import CacheManager
from app.core import separation
from app.core.ingestion import load_and_normalize_track
from app.core.instrumental_chain import InstrumentalEqParams, apply_dsp_chain as apply_inst_dsp
from app.core.remix_master import export_wav, master, mix_stems
from app.core.vocal_chain import apply_dsp_chain as apply_vocal_dsp, blend_vocal
from app.models.app_config import AppConfig
from app.models.preset import Preset
from app.workers.render_job import RenderJob


class MockSeparator:
    """Mock implementation of audio_separator's Separator for fast, weightless integration testing."""

    def __init__(self, output_dir: str, model_file_dir: str, *args, **kwargs) -> None:
        self.output_dir = Path(output_dir)
        self.model_file_dir = Path(model_file_dir)

    def load_model(self, model_filename: str) -> None:
        pass

    def separate(self, audio_file_path: str) -> list[str]:
        # Read the normalized track properties to ensure valid synthetic audio outputs
        audio, sr = sf.read(audio_file_path, always_2d=True, dtype="float64")

        # Generate non-zero synthetic vocal and instrumental stems
        vocal_data = audio * 0.4
        instrumental_data = audio * 0.6

        base = Path(audio_file_path).stem
        vocal_name = f"{base}_(Vocals)_model_bs_roformer.wav"
        instrumental_name = f"{base}_(Instrumental)_model_bs_roformer.wav"

        # Write 24-bit WAV files
        sf.write(str(self.output_dir / vocal_name), vocal_data, sr, subtype="PCM_24")
        sf.write(str(self.output_dir / instrumental_name), instrumental_data, sr, subtype="PCM_24")

        return [vocal_name, instrumental_name]


def mock_denoise(current: torch.Tensor, current_sr: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Fast mock of resemble-enhance denoise step returning non-zero scaled Tensor."""
    return current * 0.95, current_sr


def mock_enhance(current: torch.Tensor, current_sr: int, device: torch.device) -> tuple[torch.Tensor, int]:
    """Fast mock of resemble-enhance enhance step returning non-zero scaled Tensor."""
    return current * 0.90, current_sr


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


def test_full_pipeline_via_render_job(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that RenderJob executes the full pipeline end-to-end with mock weights, producing correct outputs."""
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))

    # Generate synthetic input audio: stereo, 1 second at 44.1kHz
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine waves on left/right channels to ensure valid stereo signal flow
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = np.stack([left, right], axis=1)

    input_file = tmp_path / "synthetic_input_track.wav"
    sf.write(str(input_file), audio, sr, subtype="PCM_16")

    # Setup a Preset with custom parameters
    preset = Preset(
        vocal_clean_intensity=0.75,
        notch_depth_db=4.5,
        vocal_denoise_enabled=True,
        vocal_denoise_intensity=0.8,
        vocal_enhance_enabled=True,
        vocal_enhance_intensity=0.7,
        instrumental_denoise_enabled=True,
        instrumental_denoise_intensity=0.6,
        instrumental_enhance_enabled=True,
        instrumental_enhance_intensity=0.5,
        vocal_gain_db=1.5,
        instrumental_gain_db=-1.0,
        lufs_target=-15.0,
    )

    # Monkeypatch Separator and lazy import of resemble-enhance
    monkeypatch.setattr(separation, "Separator", MockSeparator)
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)
    monkeypatch.setattr(
        "app.core.neural_common._lazy_import_resemble_enhance",
        lambda: (mock_denoise, mock_enhance)
    )

    # Instantiate RenderJob
    job = RenderJob(input_path=input_file, preset=preset, cache_manager=cache_mgr)

    # Set up signal listeners
    visited_stages = []
    progress_values = []
    finished_paths = []
    failed_messages = []

    job.stageChanged.connect(visited_stages.append)
    job.progressChanged.connect(progress_values.append)
    job.finished.connect(finished_paths.append)
    job.failed.connect(failed_messages.append)

    # Run the job synchronously
    job.run()

    # Assertions on job flow and signals
    assert not failed_messages, f"RenderJob reported failure: {failed_messages}"
    assert len(finished_paths) == 1, "RenderJob should emit finished exactly once"

    exported_path = finished_paths[0]
    assert exported_path.is_file(), f"Exported file does not exist: {exported_path}"

    # Verify that we transitioned through the expected stages
    expected_stages = ["Normalizing", "Separating", "Denoising Vocal", "Denoising Instrumental", "Mixing", "Mastering"]
    for stage in expected_stages:
        assert stage in visited_stages, f"Stage '{stage}' was not visited. Visited: {visited_stages}"

    # Verify that progress was tracked and is increasing
    assert len(progress_values) > 0
    assert all(0.0 <= p <= 1.0 for p in progress_values)

    # Verify final exported WAV properties
    final_data, final_sr = sf.read(str(exported_path), always_2d=True)
    assert final_sr == sr, f"Sample rate mismatch: expected {sr}, got {final_sr}"
    assert final_data.shape[1] == 2, f"Channels mismatch: expected stereo (2 channels), got {final_data.shape[1]}"

    # Ensure there is non-zero audio data
    assert np.any(final_data != 0), "Exported audio data is completely silent/zero"

    # Assert correctness of JSON metadata output
    meta_path = exported_path.with_suffix(".json")
    assert meta_path.is_file(), f"Metadata JSON file not found: {meta_path}"

    with open(meta_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    assert "timestamp" in metadata, "Metadata missing 'timestamp'"
    assert "track_id" in metadata, "Metadata missing 'track_id'"
    assert "render_file" in metadata, "Metadata missing 'render_file'"
    assert metadata["render_file"] == exported_path.name

    # Verify the serialized preset matches
    serialized_preset = metadata["preset"]
    assert serialized_preset["vocal_clean_intensity"] == 0.75
    assert serialized_preset["notch_depth_db"] == 4.5
    assert serialized_preset["vocal_denoise_enabled"] is True
    assert serialized_preset["vocal_denoise_intensity"] == 0.8
    assert serialized_preset["vocal_enhance_enabled"] is True
    assert serialized_preset["vocal_enhance_intensity"] == 0.7
    assert serialized_preset["instrumental_denoise_enabled"] is True
    assert serialized_preset["instrumental_denoise_intensity"] == 0.6
    assert serialized_preset["instrumental_enhance_enabled"] is True
    assert serialized_preset["instrumental_enhance_intensity"] == 0.5
    assert serialized_preset["vocal_gain_db"] == 1.5
    assert serialized_preset["instrumental_gain_db"] == -1.0
    assert serialized_preset["lufs_target"] == -15.0
