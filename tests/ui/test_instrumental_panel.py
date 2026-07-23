"""Tests for PySide6 InstrumentalPanel UI component (app.ui.instrumental_panel)."""

from __future__ import annotations

import sys
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

from app.cache.cache_manager import CacheManager
from app.core import presets
from app.models.preset import Preset
from app.models.settings import Settings
from app.ui.instrumental_panel import InstrumentalPanel


def test_instrumental_panel_structure_and_defaults(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    mock_cache.presets_dir = Path("/tmp/presets")
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = InstrumentalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    settings = panel.get_settings()
    assert isinstance(settings, Settings)
    assert settings.instrumental_denoise_enabled is True
    assert settings.instrumental_denoise_intensity == 0.5
    assert settings.instrumental_enhance_enabled is True
    assert settings.instrumental_enhance_intensity == 0.5
    assert settings.instrumental_mud_cut_hz == 40.0
    assert settings.instrumental_dehiss_shelf_hz == 10000.0
    assert settings.instrumental_dehiss_gain_db == -3.0
    assert settings.instrumental_gain_db == 0.0


def test_instrumental_panel_set_and_get_settings(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = InstrumentalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    custom_preset = Preset(
        instrumental_denoise_enabled=True,
        instrumental_denoise_intensity=0.3,
        instrumental_enhance_enabled=False,
        instrumental_enhance_intensity=0.7,
        instrumental_mud_cut_hz=60.0,
        instrumental_dehiss_shelf_hz=12000.0,
        instrumental_dehiss_gain_db=-5.0,
        instrumental_gain_db=-2.5,
    )

    panel.set_settings(custom_preset)

    updated = panel.get_settings()
    assert updated.instrumental_denoise_enabled is True
    assert abs(updated.instrumental_denoise_intensity - 0.3) < 1e-2
    assert updated.instrumental_enhance_enabled is False
    assert abs(updated.instrumental_enhance_intensity - 0.7) < 1e-2
    assert updated.instrumental_mud_cut_hz == 60.0
    assert updated.instrumental_dehiss_shelf_hz == 12000.0
    assert abs(updated.instrumental_dehiss_gain_db - (-5.0)) < 1e-2
    assert updated.instrumental_gain_db == -2.5


def test_instrumental_panel_overall_toggle_disables_cleanup(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = InstrumentalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    assert panel._cleanup_container.isEnabled() is True
    assert panel._overall_cleanup_cb.isChecked() is True

    # Uncheck overall cleanup toggle
    panel._overall_cleanup_cb.setChecked(False)

    assert panel._cleanup_container.isEnabled() is False
    disabled_settings = panel.get_settings()
    assert disabled_settings.instrumental_denoise_enabled is False
    assert disabled_settings.instrumental_enhance_enabled is False
    assert disabled_settings.instrumental_dehiss_gain_db == 0.0


def test_instrumental_panel_slider_changes_update_labels_and_emit_signal(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = InstrumentalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    emitted_settings = []
    panel.settingsChanged.connect(lambda s: emitted_settings.append(s))

    panel._mud_cut_slider.setValue(80)
    assert panel._mud_cut_val_label.text() == "80.0 Hz"

    panel._dehiss_shelf_slider.setValue(14000)
    assert panel._dehiss_shelf_val_label.text() == "14000 Hz"

    panel._dehiss_gain_slider.setValue(-45)
    assert panel._dehiss_gain_val_label.text() == "-4.5 dB"

    panel._gain_spinner.setValue(3.5)

    assert len(emitted_settings) > 0
    latest = emitted_settings[-1]
    assert latest.instrumental_mud_cut_hz == 80.0
    assert latest.instrumental_dehiss_shelf_hz == 14000.0
    assert abs(latest.instrumental_dehiss_gain_db - (-4.5)) < 1e-2
    assert latest.instrumental_gain_db == 3.5


def test_instrumental_panel_apply_button_emits_render_requested(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = InstrumentalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    panel._gain_spinner.setValue(1.5)

    with qtbot.waitSignal(panel.renderRequested, timeout=1000) as blocker:
        qtbot.mouseClick(panel._apply_button, Qt.MouseButton.LeftButton)

    emitted_settings = blocker.args[0]
    assert isinstance(emitted_settings, Settings)
    assert emitted_settings.instrumental_gain_db == 1.5


def test_instrumental_panel_preset_loading_and_saving(qtbot):
    cache_mgr = CacheManager()
    presets_dir = cache_mgr.presets_dir

    custom_preset = Preset(
        instrumental_denoise_enabled=False,
        instrumental_denoise_intensity=0.1,
        instrumental_gain_db=5.0,
        instrumental_mud_cut_hz=50.0,
    )
    presets.save_preset("instr_test_preset", custom_preset, cache_mgr)

    try:
        panel = InstrumentalPanel(cache_manager=cache_mgr)
        qtbot.addWidget(panel)

        combo_idx = panel._preset_combo.findData("instr_test_preset")
        assert combo_idx != -1

        panel._preset_combo.setCurrentIndex(combo_idx)

        loaded_settings = panel.get_settings()
        assert loaded_settings.instrumental_denoise_enabled is False
        assert abs(loaded_settings.instrumental_denoise_intensity - 0.1) < 1e-2
        assert loaded_settings.instrumental_gain_db == 5.0
        assert loaded_settings.instrumental_mud_cut_hz == 50.0

        with patch("PySide6.QtWidgets.QInputDialog.getText", return_value=("saved_from_instr_panel", True)):
            panel.on_save_preset_clicked()

        saved_preset = presets.load_preset("saved_from_instr_panel", cache_mgr)
        assert saved_preset.instrumental_denoise_enabled is False
        assert abs(saved_preset.instrumental_denoise_intensity - 0.1) < 1e-2

    finally:
        for p_name in ["instr_test_preset", "saved_from_instr_panel"]:
            p_file = presets_dir / f"{p_name}.json"
            if p_file.is_file():
                p_file.unlink()
