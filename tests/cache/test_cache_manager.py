"""Tests for app.cache.cache_manager (CacheManager)."""

from __future__ import annotations

import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf

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


def test_verify_stem_wav(tmp_path: Path) -> None:
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    # 1. Missing file
    missing_file = tmp_path / "nonexistent.wav"
    assert cache_mgr.verify_stem_wav(missing_file) is False

    # 2. Empty file
    empty_file = tmp_path / "empty.wav"
    empty_file.touch()
    assert cache_mgr.verify_stem_wav(empty_file) is False

    # 3. Corrupted file (invalid WAV data)
    corrupt_file = tmp_path / "corrupt.wav"
    corrupt_file.write_bytes(b"invalid WAV header and data content")
    assert cache_mgr.verify_stem_wav(corrupt_file) is False

    # 4. Valid WAV file
    valid_file = tmp_path / "valid.wav"
    dummy_audio = np.zeros((44100, 2))
    sf.write(str(valid_file), dummy_audio, 44100, subtype="PCM_24")
    assert cache_mgr.verify_stem_wav(valid_file) is True


def test_cache_manager_pruning(tmp_path: Path) -> None:
    config = AppConfig(cache_root=tmp_path / "cache")
    cache_mgr = CacheManager(config=config)

    # Create folders and files
    track_id = "track_one"
    stems_dir = cache_mgr.stems_dir(track_id)
    renders_dir = cache_mgr.renders_dir(track_id)
    logs_dir = cache_mgr.logs_dir
    presets_dir = cache_mgr.presets_dir

    # 1. Stem file (should be protected) - size 100 bytes
    stem_file = stems_dir / "vocal.wav"
    stem_file.write_bytes(b"a" * 100)

    # 2. Preset file (should be protected) - size 100 bytes
    preset_file = presets_dir / "my_preset.json"
    preset_file.write_bytes(b"b" * 100)

    # 3. Oldest render file - size 200 bytes
    old_render = renders_dir / "old_render.wav"
    old_render.write_bytes(b"c" * 200)

    # 4. Medium log file - size 300 bytes
    med_log = logs_dir / "med_log.txt"
    med_log.write_bytes(b"d" * 300)

    # 5. Newest render file - size 400 bytes
    new_render = renders_dir / "new_render.wav"
    new_render.write_bytes(b"e" * 400)

    # Total cache size initially = 100 (stem) + 100 (preset) + 200 (old render) + 300 (med log) + 400 (new render) = 1100 bytes
    assert cache_mgr.get_total_cache_size() == 1100

    # Set precise modification times to control the pruning order
    now = time.time()
    os.utime(old_render, (now - 100, now - 100))
    os.utime(med_log, (now - 50, now - 50))
    os.utime(new_render, (now, now))

    # Prune with quota of 1000 bytes (exceeded by 100 bytes).
    # Since old_render is 200 bytes, deleting it brings us to 900 bytes (<= 1000).
    cache_mgr.prune_cache(quota_bytes=1000)
    assert not old_render.exists()
    assert med_log.exists()
    assert new_render.exists()
    assert stem_file.exists()
    assert preset_file.exists()
    assert cache_mgr.get_total_cache_size() == 900

    # Prune with quota of 500 bytes.
    # Current size is 900 bytes.
    # Deleting med_log (300 bytes) brings size to 600.
    # Deleting new_render (400 bytes) brings size to 200 (<= 500).
    cache_mgr.prune_cache(quota_bytes=500)
    assert not med_log.exists()
    assert not new_render.exists()
    assert stem_file.exists()
    assert preset_file.exists()
    assert cache_mgr.get_total_cache_size() == 200

    # Prune with extremely small quota (50 bytes). Protected files should still NOT be deleted!
    cache_mgr.prune_cache(quota_bytes=50)
    assert stem_file.exists()
    assert preset_file.exists()
    assert cache_mgr.get_total_cache_size() == 200
