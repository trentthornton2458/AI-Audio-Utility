"""Tests for app.core.neural_common: the independently-cacheable denoise/enhance neural pass
stages shared by app.core.vocal_chain and app.core.instrumental_chain."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from app.cache.cache_manager import CacheManager
from app.core import neural_common
from app.models.app_config import AppConfig


@pytest.fixture
def cache_mgr(tmp_path: Path) -> CacheManager:
    return CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))


def _write_stem(
    cache_mgr: CacheManager,
    track_id: str,
    filename: str,
    seconds: float = 0.1,
    sr: int = 8000,
) -> Path:
    stems_dir = cache_mgr.stems_dir(track_id)
    path = stems_dir / filename
    audio = (
        0.3
        * np.sin(
            2 * np.pi * 440 * np.linspace(0, seconds, int(sr * seconds), endpoint=False)
        )
    ).astype(np.float64)
    sf.write(str(path), audio, sr, subtype="PCM_24")
    return path


def _counting_denoise_enhance():
    calls = {"denoise": 0, "enhance": 0}

    def denoise(current: torch.Tensor, current_sr: int, device: torch.device):
        calls["denoise"] += 1
        return current * 0.9, current_sr

    def enhance(current: torch.Tensor, current_sr: int, device: torch.device):
        calls["enhance"] += 1
        return current * 0.8, current_sr

    return calls, denoise, enhance


def test_hash_denoise_settings_is_stable_and_input_sensitive():
    a = neural_common._hash_denoise_settings(True, 0.5)
    b = neural_common._hash_denoise_settings(True, 0.5)
    assert a == b

    assert neural_common._hash_denoise_settings(True, 0.6) != a
    assert neural_common._hash_denoise_settings(False, 0.5) != a


def test_hash_enhance_settings_changes_when_dsp_content_changes():
    a = neural_common._hash_enhance_settings(True, "content-hash-a")
    b = neural_common._hash_enhance_settings(True, "content-hash-a")
    assert a == b
    assert neural_common._hash_enhance_settings(True, "content-hash-b") != a
    assert neural_common._hash_enhance_settings(False, "content-hash-a") != a


def test_run_denoise_pass_caches_and_skips_reprocessing(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    stem_path = _write_stem(cache_mgr, "track1", "vocal.wav")
    calls, denoise, enhance = _counting_denoise_enhance()
    monkeypatch.setattr(
        neural_common, "_lazy_import_resemble_enhance", lambda: (denoise, enhance)
    )

    first = neural_common.run_denoise_pass(
        stem_path, True, 0.5, cache_mgr, "vocal_neural_", "vocal"
    )
    assert calls["denoise"] == 1
    assert first.is_file()

    second = neural_common.run_denoise_pass(
        stem_path, True, 0.5, cache_mgr, "vocal_neural_", "vocal"
    )
    assert second == first
    assert calls["denoise"] == 1  # cache hit -- model not invoked again


def test_run_denoise_pass_disabled_passes_through_unmodified(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    stem_path = _write_stem(cache_mgr, "track1", "vocal.wav")
    calls, denoise, enhance = _counting_denoise_enhance()
    monkeypatch.setattr(
        neural_common, "_lazy_import_resemble_enhance", lambda: (denoise, enhance)
    )

    output_path = neural_common.run_denoise_pass(
        stem_path, False, 0.5, cache_mgr, "vocal_neural_", "vocal"
    )

    assert calls["denoise"] == 0
    original_audio, _ = sf.read(str(stem_path))
    written_audio, _ = sf.read(str(output_path))
    np.testing.assert_allclose(written_audio, original_audio, atol=1e-4)


def test_run_enhance_pass_passthrough_when_disabled(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch
):
    dsp_path = _write_stem(cache_mgr, "track1", "vocal_dsp.wav")
    calls, denoise, enhance = _counting_denoise_enhance()
    monkeypatch.setattr(
        neural_common, "_lazy_import_resemble_enhance", lambda: (denoise, enhance)
    )

    result = neural_common.run_enhance_pass(
        dsp_path, False, cache_mgr, "vocal_neural_", "vocal"
    )

    assert result == dsp_path
    assert calls["enhance"] == 0
    # No new "*enhance*" cache file should have been written.
    assert list(cache_mgr.stems_dir("track1").glob("*enhance*")) == []


def test_run_enhance_pass_cache_invalidated_by_changed_dsp_input(
    cache_mgr: CacheManager, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    dsp_path = _write_stem(cache_mgr, "track1", "vocal_dsp.wav")
    calls, denoise, enhance = _counting_denoise_enhance()
    monkeypatch.setattr(
        neural_common, "_lazy_import_resemble_enhance", lambda: (denoise, enhance)
    )

    first = neural_common.run_enhance_pass(
        dsp_path, True, cache_mgr, "vocal_neural_", "vocal"
    )
    assert calls["enhance"] == 1

    # Same settings, but the DSP output's content changed (e.g. a different notch/de-esser
    # parameter re-ran the DSP chain) -- the enhance cache key must track content, not just the
    # enhance_enabled toggle, so this must NOT be a cache hit.
    sf.write(str(dsp_path), _sine_array(0.05), 8000, subtype="PCM_24")
    second = neural_common.run_enhance_pass(
        dsp_path, True, cache_mgr, "vocal_neural_", "vocal"
    )

    assert calls["enhance"] == 2
    assert second != first


def _sine_array(seconds: float, sr: int = 8000) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    return (0.7 * np.sin(2 * np.pi * 880 * t)).astype(np.float64)
