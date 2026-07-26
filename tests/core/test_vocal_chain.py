"""Tests for vocal Pedalboard DSP chain, blending, and the Humanizer pass."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.cache.cache_manager import CacheManager
from app.core import qa_gate, vocal_chain
from app.core.vocal_chain import (
    _clamp,
    apply_dsp_chain,
    blend_vocal,
    run_humanizer_pass,
)
from app.models.app_config import AppConfig


def test_clamp_helper():
    assert _clamp(5.0, 0.0, 10.0) == 5.0
    assert _clamp(-2.0, 0.0, 10.0) == 0.0
    assert _clamp(15.0, 0.0, 10.0) == 10.0


def test_apply_dsp_chain(tmp_path: Path):
    sr = 44100
    duration = 0.2
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Generate audio with low mid and sibilant high content
    audio = (np.sin(2 * np.pi * 440 * t) + 0.5 * np.sin(2 * np.pi * 6000 * t))[
        :, np.newaxis
    ]
    audio = np.repeat(audio, 2, axis=1)  # Stereo

    in_file = tmp_path / "vocal_in.wav"
    sf.write(str(in_file), audio, sr, subtype="PCM_24")

    out_file = tmp_path / "vocal_dsp.wav"
    res_path = apply_dsp_chain(
        in_file, notch_depth_db=4.0, deesser_threshold_db=-6.0, out_path=out_file
    )

    assert res_path.is_file()
    out_data, out_sr = sf.read(str(out_file))
    assert out_sr == sr
    assert out_data.shape == audio.shape


def _write_constant_wav(
    path: Path, value: float, samplerate: int, frames: int = 4410
) -> np.ndarray:
    audio = np.ones((frames, 2), dtype=np.float64) * value
    sf.write(str(path), audio, samplerate, subtype="PCM_24")
    return audio


def test_blend_vocal_zero_intensity_passes_through_dsp_signal(tmp_path: Path):
    sr = 44100
    dsp_audio = _write_constant_wav(tmp_path / "dsp.wav", 0.3, sr)
    _write_constant_wav(tmp_path / "enhanced.wav", 0.9, sr)

    result = blend_vocal(
        tmp_path / "dsp.wav", tmp_path / "enhanced.wav", enhance_intensity=0.0
    )

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


@pytest.fixture
def cache_mgr(tmp_path: Path) -> CacheManager:
    return CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))


def _setup_humanizer_inputs(
    cache_mgr: CacheManager,
    track_id: str,
    vocal_value: float = 0.2,
    residual_value: float = 0.01,
    samplerate: int = 44100,
    frames: int = 4410,
) -> tuple[Path, Path]:
    stems_dir = cache_mgr.stems_dir(track_id)
    vocal_path = stems_dir / "vocal_qa_blend.wav"
    residual_path = stems_dir / "residual.wav"
    _write_constant_wav(vocal_path, vocal_value, samplerate, frames)
    _write_constant_wav(residual_path, residual_value, samplerate, frames)
    return vocal_path, residual_path


def _stub_humanizer_functions(monkeypatch: pytest.MonkeyPatch, call_log: dict):
    """Replace app.core.humanizer.apply_pitch_drift/apply_breath_blend (as referenced by
    vocal_chain.humanizer) with deterministic stand-ins that record how many times/with what
    args they were invoked, so caching behavior can be verified without depending on the real
    pyrubberband/rubberband-binary pitch-shift implementation."""
    call_log.setdefault("drift_intensities", [])
    call_log.setdefault("blend_calls", 0)

    def fake_pitch_drift(
        audio: np.ndarray, sample_rate: int, intensity: float
    ) -> np.ndarray:
        call_log["drift_intensities"].append(intensity)
        return audio + 0.1

    def fake_breath_blend(
        processed_vocal: np.ndarray, residual_signal: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        call_log["blend_calls"] += 1
        return processed_vocal + 0.05

    monkeypatch.setattr(vocal_chain.humanizer, "apply_pitch_drift", fake_pitch_drift)
    monkeypatch.setattr(vocal_chain.humanizer, "apply_breath_blend", fake_breath_blend)


def test_run_humanizer_pass_applies_pitch_drift_then_breath_blend(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(
        cache_mgr, "track_humanize", vocal_value=0.2
    )
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    out_path = run_humanizer_pass(vocal_path, 0.5, residual_path, cache_mgr)

    assert out_path.is_file()
    assert out_path.parent == cache_mgr.stems_dir("track_humanize")
    assert out_path.name.startswith("vocal_neural_humanize_")
    assert call_log["drift_intensities"] == [0.5]
    assert call_log["blend_calls"] == 1

    result_audio, result_sr = sf.read(str(out_path), always_2d=True, dtype="float64")
    assert result_sr == 44100
    # pitch-drift stub adds 0.1, breath-blend stub then adds a further 0.05, applied in that order.
    np.testing.assert_allclose(result_audio, 0.2 + 0.1 + 0.05, atol=1e-3)


def test_run_humanizer_pass_second_call_uses_cache(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(cache_mgr, "track_cache_hit")
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    first_path = run_humanizer_pass(vocal_path, 0.25, residual_path, cache_mgr)
    second_path = run_humanizer_pass(vocal_path, 0.25, residual_path, cache_mgr)

    assert first_path == second_path
    assert call_log["drift_intensities"] == [0.25]
    assert call_log["blend_calls"] == 1


def test_run_humanizer_pass_cache_key_changes_with_intensity(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(
        cache_mgr, "track_intensity_change"
    )
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    path_low = run_humanizer_pass(vocal_path, 0.2, residual_path, cache_mgr)
    path_high = run_humanizer_pass(vocal_path, 0.6, residual_path, cache_mgr)

    assert path_low != path_high
    assert call_log["drift_intensities"] == [0.2, 0.6]


def test_run_humanizer_pass_cache_key_changes_with_input_content(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(
        cache_mgr, "track_content_change", vocal_value=0.2
    )
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    path_before = run_humanizer_pass(vocal_path, 0.3, residual_path, cache_mgr)

    # Overwrite the (same-path) blended vocal input with different content, simulating an
    # upstream denoise/DSP/enhance/QA-blend change -- this stage's cache key is derived from the
    # input file's content hash, so it must invalidate rather than reuse the stale cached output.
    _write_constant_wav(vocal_path, 0.9, 44100, 4410)
    path_after = run_humanizer_pass(vocal_path, 0.3, residual_path, cache_mgr)

    assert path_before != path_after
    assert call_log["blend_calls"] == 2


def test_run_humanizer_pass_clamps_intensity_out_of_range(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(cache_mgr, "track_clamp")
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    over_path = run_humanizer_pass(vocal_path, 5.0, residual_path, cache_mgr)
    at_max_path = run_humanizer_pass(vocal_path, 1.0, residual_path, cache_mgr)

    assert over_path == at_max_path
    assert call_log["drift_intensities"] == [1.0]


def test_run_humanizer_pass_reports_progress(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(cache_mgr, "track_progress")
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    progress_values: list[float] = []
    run_humanizer_pass(
        vocal_path,
        0.4,
        residual_path,
        cache_mgr,
        progress_callback=progress_values.append,
    )

    assert progress_values[0] == 0.0
    assert progress_values[-1] == 1.0
    assert progress_values == sorted(progress_values)


def test_run_humanizer_pass_cache_hit_skips_humanizer_functions_and_still_reports_progress(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(
        cache_mgr, "track_cache_progress"
    )
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    run_humanizer_pass(vocal_path, 0.4, residual_path, cache_mgr)
    assert call_log["blend_calls"] == 1

    progress_values: list[float] = []
    run_humanizer_pass(
        vocal_path,
        0.4,
        residual_path,
        cache_mgr,
        progress_callback=progress_values.append,
    )

    assert call_log["blend_calls"] == 1
    assert progress_values == [0.0, 1.0]


def test_run_humanizer_pass_respects_is_cancelled(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    vocal_path, residual_path = _setup_humanizer_inputs(cache_mgr, "track_cancel")
    call_log: dict = {}
    _stub_humanizer_functions(monkeypatch, call_log)

    with pytest.raises(InterruptedError):
        run_humanizer_pass(
            vocal_path, 0.4, residual_path, cache_mgr, is_cancelled=lambda: True
        )

    assert call_log["blend_calls"] == 0
