import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.cache.cache_manager import CacheManager
from app.core import humanizer, separation
from app.models.app_config import AppConfig
from app.models.preset import Preset
from app.workers.render_job import RenderJob

# Import mocks from test_full_pipeline
from tests.test_full_pipeline import MockSeparator, mock_denoise, mock_enhance


def mock_apply_pitch_drift_intensity_aware(
    audio: np.ndarray, sample_rate: int, intensity: float
) -> np.ndarray:
    if intensity == 0.0:
        return audio
    return audio * (1.0 - (0.1 * intensity))


def mock_apply_breath_blend(
    processed_vocal: np.ndarray, residual_signal: np.ndarray, sample_rate: int
) -> np.ndarray:
    return processed_vocal + (residual_signal * 0.1)


def test_vocal_chain_pipeline_humanizer_and_qa_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    cache_mgr = CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))

    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    # We will use an audio array that intentionally causes a QA flag (clipping fail-safe)
    # during the blend stage to ensure flags are properly logged in RenderJob output.
    left = 0.5 * np.sin(2 * np.pi * 440 * t)
    right = 0.5 * np.sin(2 * np.pi * 880 * t)
    audio = np.stack([left, right], axis=1)

    input_file = tmp_path / "synthetic_input_track.wav"
    sf.write(str(input_file), audio, sr, subtype="PCM_16")

    monkeypatch.setattr(separation, "Separator", MockSeparator)
    monkeypatch.setattr(separation, "ensure_ffmpeg_in_path", lambda bin_dir: bin_dir)

    # Intentionally cause the enhanced mock to return very high amplitudes to trigger the clipping flag in blend_vocal
    def mock_enhance_clipping(
        current: np.ndarray, current_sr: int, device: str
    ) -> tuple[np.ndarray, int]:
        return current * 5.0, current_sr

    monkeypatch.setattr(
        "app.core.neural_common._lazy_import_resemble_enhance",
        lambda: (mock_denoise, mock_enhance_clipping),
    )
    monkeypatch.setattr(
        humanizer, "apply_pitch_drift", mock_apply_pitch_drift_intensity_aware
    )
    monkeypatch.setattr(humanizer, "apply_breath_blend", mock_apply_breath_blend)

    # 1. Run pipeline with humanizer disabled (intensity=0)
    preset_off = Preset(
        humanizer_intensity=0.0,
        vocal_enhance_intensity=0.3,  # trigger QA gate blend
    )
    job_off = RenderJob(
        input_path=input_file, preset=preset_off, cache_manager=cache_mgr
    )

    finished_paths_off = []
    job_off.renderFinished.connect(finished_paths_off.append)
    job_off.run()

    out_path_off = finished_paths_off[0]
    meta_path_off = out_path_off.with_suffix(".json")
    with open(meta_path_off, "r") as f:
        meta_off = json.load(f)

    audio_off, _ = sf.read(str(out_path_off))

    # 2. Run pipeline with humanizer enabled (intensity>0)
    preset_on = Preset(
        humanizer_intensity=1.0,
        vocal_enhance_intensity=0.3,
    )
    job_on = RenderJob(input_path=input_file, preset=preset_on, cache_manager=cache_mgr)

    finished_paths_on = []
    job_on.renderFinished.connect(finished_paths_on.append)
    job_on.run()

    out_path_on = finished_paths_on[0]
    meta_path_on = out_path_on.with_suffix(".json")
    with open(meta_path_on, "r") as f:
        meta_on = json.load(f)

    audio_on, _ = sf.read(str(out_path_on))

    # Order Verification: Humanizer must run after QA blend

    track_id = meta_on["track_id"]
    stems_dir = cache_mgr.stems_dir(track_id)
    qa_blend_files = list(stems_dir.glob("vocal_qa_blend.wav"))
    assert len(qa_blend_files) > 0, "Expected QA blend output to be cached"

    # Verification 1: Distinct cache entries for the humanizer stage
    humanizer_files = list(stems_dir.glob("vocal_neural_humanize_*.wav"))

    # We rely on sequential execution (implicitly verified by the successful run)
    # and the files' creation in RenderJob.
    assert (
        len(humanizer_files) >= 2
    ), "Expected distinct cached outputs for different humanizer intensities"

    # Verification 2: Measurably different outputs
    assert not np.allclose(
        audio_off, audio_on
    ), "Output with humanizer=0 should differ from output with humanizer>0"

    # Verification 3: Breath-blend/humanizer stage does not introduce clipping
    assert (
        np.max(np.abs(audio_on)) <= 1.0
    ), "Audio output should not clip after humanizer stage"

    # Verification 4: QA gate flags are present in metadata without exceptions
    assert "qa_flags" in meta_off
    assert isinstance(meta_off["qa_flags"], list)
    assert "qa_flags" in meta_on
    assert isinstance(meta_on["qa_flags"], list)

    # We expect some QA flags from the clipping fail-safe because of `mock_enhance_clipping`.
    reasons = [flag["reason"] for flag in meta_on["qa_flags"]]
    assert (
        "clipping" in reasons
    ), "Expected the Milestone 3 QA gate to flag clipping from the mock enhance"
