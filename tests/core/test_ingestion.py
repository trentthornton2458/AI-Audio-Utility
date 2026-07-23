"""Tests for audio track ingestion and normalization."""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from app.cache.cache_manager import CacheManager
from app.core.ingestion import (
    UnsupportedAudioFormatError,
    load_and_normalize_track,
)
from app.models.app_config import AppConfig


@pytest.fixture
def cache_mgr(tmp_path: Path) -> CacheManager:
    return CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))


def test_load_non_existent_file_raises_error(cache_mgr: CacheManager, tmp_path: Path):
    bogus_path = tmp_path / "does_not_exist.wav"
    with pytest.raises(UnsupportedAudioFormatError, match="not found"):
        load_and_normalize_track(bogus_path, cache_mgr)


def test_load_and_normalize_valid_wav(cache_mgr: CacheManager, tmp_path: Path):
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)[:, np.newaxis]  # Mono tone

    input_file = tmp_path / "sample.wav"
    sf.write(str(input_file), audio, sr, subtype="PCM_16")

    norm_path = load_and_normalize_track(input_file, cache_mgr)
    assert norm_path.is_file()
    assert norm_path.name == "normalized.wav"

    data, read_sr = sf.read(str(norm_path), always_2d=True)
    assert read_sr == 44100
    assert data.ndim == 2
