"""Tests for app.ui.render_history_panel (RenderHistoryPanel)."""

from __future__ import annotations

import json
import sys
from datetime import datetime
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

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QMessageBox

from app.cache.cache_manager import CacheManager
from app.models.app_config import AppConfig
from app.ui.render_history_panel import RenderHistoryPanel


@pytest.fixture
def temp_cache_manager(tmp_path: Path) -> CacheManager:
    config = AppConfig(cache_root=tmp_path / "cache")
    return CacheManager(config=config)


def test_render_history_panel_init(qtbot, temp_cache_manager: CacheManager):
    panel = RenderHistoryPanel(cache_manager=temp_cache_manager)
    qtbot.addWidget(panel)

    assert panel.get_track_id() is None
    assert panel._clear_cache_button.isEnabled() is False
    assert panel._status_label.text() == "No track loaded."
    assert panel._list_widget.count() == 0


def test_render_history_panel_refresh_with_renders(qtbot, temp_cache_manager: CacheManager):
    panel = RenderHistoryPanel(cache_manager=temp_cache_manager)
    qtbot.addWidget(panel)

    track_id = "track_xyz_789"
    renders_dir = temp_cache_manager.renders_dir(track_id)

    render_wav = renders_dir / "render_20260722_120000.wav"
    render_wav.touch()

    meta_json = renders_dir / "render_20260722_120000.json"
    metadata = {
        "timestamp": datetime(2026, 7, 22, 12, 0, 0).isoformat(),
        "track_id": track_id,
        "render_file": render_wav.name,
        "preset": {
            "vocal_clean_intensity": 0.8,
            "notch_depth_db": 3.5,
            "lufs_target": -14.0,
            "vocal_gain_db": 1.5,
            "instrumental_gain_db": -0.5,
        },
    }
    meta_json.write_text(json.dumps(metadata), encoding="utf-8")

    panel.set_track_id(track_id)

    assert panel.get_track_id() == track_id
    assert panel._list_widget.count() == 1
    assert panel._clear_cache_button.isEnabled() is True

    item = panel._list_widget.item(0)
    assert "render_20260722_120000.wav" in item.text()
    assert "2026-07-22 12:00:00" in item.text()
    assert "Vocal Clean: 80%" in item.text()
    assert item.data(Qt.ItemDataRole.UserRole) == render_wav


def test_render_history_panel_selection_signal(qtbot, temp_cache_manager: CacheManager):
    panel = RenderHistoryPanel(cache_manager=temp_cache_manager)
    qtbot.addWidget(panel)

    track_id = "track_select_123"
    renders_dir = temp_cache_manager.renders_dir(track_id)
    render_wav = renders_dir / "render_01.wav"
    render_wav.touch()

    panel.set_track_id(track_id)

    with qtbot.waitSignal(panel.renderSelected, timeout=1000) as blocker:
        panel._list_widget.setCurrentRow(0)

    assert blocker.args[0] == render_wav


def test_render_history_panel_clear_cache_confirmed(qtbot, temp_cache_manager: CacheManager):
    panel = RenderHistoryPanel(cache_manager=temp_cache_manager)
    qtbot.addWidget(panel)

    track_id = "track_clear_456"
    renders_dir = temp_cache_manager.renders_dir(track_id)
    (renders_dir / "render_01.wav").touch()

    panel.set_track_id(track_id)
    assert panel._list_widget.count() == 1

    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.Yes):
        panel._clear_cache_button.click()

    assert not (renders_dir / "render_01.wav").exists()
    assert panel._list_widget.count() == 0


def test_render_history_panel_clear_cache_rejected(qtbot, temp_cache_manager: CacheManager):
    panel = RenderHistoryPanel(cache_manager=temp_cache_manager)
    qtbot.addWidget(panel)

    track_id = "track_keep_789"
    renders_dir = temp_cache_manager.renders_dir(track_id)
    (renders_dir / "render_01.wav").touch()

    panel.set_track_id(track_id)
    assert panel._list_widget.count() == 1

    with patch("PySide6.QtWidgets.QMessageBox.question", return_value=QMessageBox.StandardButton.No):
        panel._clear_cache_button.click()

    assert renders_dir.exists()
    assert panel._list_widget.count() == 1
