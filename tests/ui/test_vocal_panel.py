"""Tests for PySide6 VocalPanel UI component (app.ui.vocal_panel)."""

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
from app.ui.vocal_panel import VocalPanel


def test_vocal_panel_structure_and_defaults(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    mock_cache.presets_dir = Path("/tmp/presets")
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    settings = panel.get_settings()
    assert isinstance(settings, Settings)
    assert settings.vocal_denoise_enabled is True
    assert settings.vocal_denoise_intensity == 0.5
    assert settings.vocal_enhance_enabled is True
    assert settings.vocal_enhance_intensity == 0.2
    assert settings.vocal_gain_db == 0.0
    assert settings.notch_depth_db == 4.5


def test_vocal_panel_set_and_get_settings(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    new_preset = Preset(
        vocal_denoise_enabled=False,
        vocal_denoise_intensity=0.2,
        vocal_enhance_enabled=True,
        vocal_enhance_intensity=0.3,
        vocal_gain_db=-3.5,
        notch_depth_db=5.2,
    )

    panel.set_settings(new_preset)

    updated = panel.get_settings()
    assert updated.vocal_denoise_enabled is False
    assert abs(updated.vocal_denoise_intensity - 0.2) < 1e-2
    assert updated.vocal_enhance_enabled is True
    assert abs(updated.vocal_enhance_intensity - 0.3) < 1e-2
    assert updated.vocal_gain_db == -3.5
    assert abs(updated.notch_depth_db - 5.2) < 1e-2


def test_vocal_panel_toggles_enable_disable_sliders(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    assert panel._denoise_slider.isEnabled() is True
    panel._denoise_cb.setChecked(False)
    assert panel._denoise_slider.isEnabled() is False

    assert panel._enhance_slider.isEnabled() is True
    panel._enhance_cb.setChecked(False)
    assert panel._enhance_slider.isEnabled() is False


def test_vocal_panel_slider_changes_update_labels(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    panel._denoise_slider.setValue(75)
    assert panel._denoise_val_label.text() == "75%"

    panel._enhance_slider.setValue(30)
    assert panel._enhance_val_label.text() == "30%"

    panel._notch_slider.setValue(55)
    assert panel._notch_val_label.text() == "-5.5 dB"


def test_vocal_panel_apply_button_emits_render_requested(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    panel._gain_spinner.setValue(2.5)

    with qtbot.waitSignal(panel.renderRequested, timeout=1000) as blocker:
        qtbot.mouseClick(panel._apply_button, Qt.MouseButton.LeftButton)

    emitted_settings = blocker.args[0]
    assert isinstance(emitted_settings, Settings)
    assert emitted_settings.vocal_gain_db == 2.5


def test_vocal_panel_sliders_do_not_auto_render(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)

    signal_received = False

    def on_render(s):
        nonlocal signal_received
        signal_received = True

    panel.renderRequested.connect(on_render)

    panel._denoise_slider.setValue(90)
    panel._enhance_slider.setValue(30)
    panel._notch_slider.setValue(60)

    assert signal_received is False


def test_vocal_panel_preset_loading_and_saving(qtbot, tmp_path):
    cache_mgr = CacheManager()
    presets_dir = cache_mgr.presets_dir

    custom_preset = Preset(
        vocal_denoise_enabled=False,
        vocal_denoise_intensity=0.1,
        vocal_gain_db=4.0,
        notch_depth_db=3.5,
    )
    presets.save_preset("vocal_test_preset", custom_preset, cache_mgr)

    try:
        panel = VocalPanel(cache_manager=cache_mgr)
        qtbot.addWidget(panel)

        # Verify preset is in dropdown
        combo_idx = panel._preset_combo.findData("vocal_test_preset")
        assert combo_idx != -1

        # Select preset in dropdown
        panel._preset_combo.setCurrentIndex(combo_idx)

        loaded_settings = panel.get_settings()
        assert loaded_settings.vocal_denoise_enabled is False
        assert abs(loaded_settings.vocal_denoise_intensity - 0.1) < 1e-2
        assert loaded_settings.vocal_gain_db == 4.0
        assert abs(loaded_settings.notch_depth_db - 3.5) < 1e-2

        # Test Save As... dialog
        with patch(
            "PySide6.QtWidgets.QInputDialog.getText",
            return_value=("saved_from_panel", True),
        ):
            panel.on_save_preset_clicked()

        saved_preset = presets.load_preset("saved_from_panel", cache_mgr)
        assert saved_preset.vocal_denoise_enabled is False
        assert abs(saved_preset.vocal_denoise_intensity - 0.1) < 1e-2

    finally:
        # Clean up created preset files
        for p_name in ["vocal_test_preset", "saved_from_panel"]:
            p_file = presets_dir / f"{p_name}.json"
            if p_file.is_file():
                p_file.unlink()


def test_vocal_panel_qa_badge_hidden_by_default(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)
        panel.show()

    assert panel._qa_badge.isHidden() is True
    assert panel._qa_details_panel.isHidden() is True


def test_vocal_panel_qa_badge_shows_on_warning_and_toggles_details(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)
        panel.show()

    from app.core.qa_gate import QAMetricResult

    qa_results = {
        "pitch_variance": QAMetricResult(
            value=4.5, warning=True, reason="flat_or_hard_quantized_pitch"
        ),
        "hf_energy": QAMetricResult(value=0.05, warning=False, reason=""),
        "crest_factor": QAMetricResult(
            value=5.0, warning=True, reason="over_compressed"
        ),
    }

    panel.set_qa_metric_results(qa_results)

    assert panel._qa_badge.isHidden() is False
    assert "2 signal quality metrics flagged" in panel._qa_badge.text()
    tooltip = panel._qa_badge.toolTip()
    assert "Pitch Variance" in tooltip
    assert "Crest Factor" in tooltip

    # Export / Apply button must stay enabled and non-blocked
    assert panel._apply_button.isEnabled() is True

    # Details panel is hidden until badge is clicked
    assert panel._qa_details_panel.isHidden() is True

    # Click badge to toggle expandable panel
    qtbot.mouseClick(panel._qa_badge, Qt.MouseButton.LeftButton)
    assert panel._qa_details_panel.isHidden() is False
    details_text = panel._qa_details_label.text()
    assert "Pitch Variance" in details_text
    assert "Crest Factor" in details_text

    # Click badge again to collapse
    qtbot.mouseClick(panel._qa_badge, Qt.MouseButton.LeftButton)
    assert panel._qa_details_panel.isHidden() is True


def test_vocal_panel_qa_badge_hidden_when_no_warnings(qtbot):
    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)
        panel.show()

    from app.core.qa_gate import QAMetricResult

    qa_results = {
        "pitch_variance": QAMetricResult(value=15.0, warning=False, reason=""),
        "hf_energy": QAMetricResult(value=0.05, warning=False, reason=""),
        "crest_factor": QAMetricResult(value=10.0, warning=False, reason=""),
    }

    panel.set_qa_metric_results(qa_results)

    assert panel._qa_badge.isHidden() is True
    assert panel._qa_details_panel.isHidden() is True


def test_vocal_panel_update_qa_from_file(qtbot, tmp_path):
    import numpy as np
    import soundfile as sf

    mock_cache = MagicMock(spec=CacheManager)
    with patch("app.core.presets.list_presets", return_value=[]):
        panel = VocalPanel(cache_manager=mock_cache)
        qtbot.addWidget(panel)
        panel.show()

    # Pure tone -> low pitch variance -> warning expected
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    pure_tone = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float64)
    audio_path = tmp_path / "test_render.wav"
    sf.write(str(audio_path), pure_tone, sr, subtype="PCM_24")

    res = panel.update_qa_from_file(audio_path)
    assert "pitch_variance" in res
    assert res["pitch_variance"].warning is True
    assert panel._qa_badge.isHidden() is False
