"""Tests for Preset persistence, listing, loading, and deletion."""

from pathlib import Path
import pytest

from app.cache.cache_manager import CacheManager
from app.core.presets import (
    PresetNotFoundError,
    delete_preset,
    list_presets,
    load_preset,
    save_preset,
)
from app.models.app_config import AppConfig
from app.models.preset import Preset


@pytest.fixture
def cache_mgr(tmp_path: Path) -> CacheManager:
    return CacheManager(config=AppConfig(cache_root=tmp_path / "cache"))


def test_preset_lifecycle(cache_mgr: CacheManager):
    p = Preset(
        vocal_denoise_enabled=True,
        vocal_denoise_intensity=0.8,
        notch_depth_db=4.5,
        vocal_gain_db=1.0,
        instrumental_gain_db=-1.5,
    )

    save_preset("my_preset", p, cache_mgr)

    names = list_presets(cache_mgr)
    assert "my_preset" in names

    loaded = load_preset("my_preset", cache_mgr)
    assert loaded.vocal_denoise_enabled is True
    assert loaded.vocal_denoise_intensity == 0.8
    assert loaded.notch_depth_db == 4.5

    delete_preset("my_preset", cache_mgr)
    assert "my_preset" not in list_presets(cache_mgr)


def test_load_nonexistent_preset_raises_error(cache_mgr: CacheManager):
    with pytest.raises(PresetNotFoundError):
        load_preset("non_existent", cache_mgr)


def test_delete_nonexistent_preset_raises_error(cache_mgr: CacheManager):
    with pytest.raises(PresetNotFoundError):
        delete_preset("non_existent", cache_mgr)
