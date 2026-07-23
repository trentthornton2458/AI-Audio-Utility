"""Tests for app.cache.cache_manager (CacheManager)."""

from __future__ import annotations

from pathlib import Path
from app.cache.cache_manager import CacheManager
from app.models.app_config import AppConfig


def test_cache_manager_clear_track_cache(tmp_path: Path) -> None:
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    track_id = "test_track_abc123"
    stems_dir = cache_mgr.stems_dir(track_id)
    renders_dir = cache_mgr.renders_dir(track_id)

    dummy_stem = stems_dir / "vocal.wav"
    dummy_stem.touch()
    dummy_render = renders_dir / "render_01.wav"
    dummy_render.touch()

    assert cache_mgr.track_dir(track_id).exists()
    assert dummy_stem.exists()
    assert dummy_render.exists()

    cache_mgr.clear_track_cache(track_id)

    assert not (tmp_path / "cache" / track_id).exists()
