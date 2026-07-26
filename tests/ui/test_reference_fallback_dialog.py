"""Tests for app.ui.reference_fallback_dialog (ReferenceFallbackDialog & check_reference_assets_fallback)."""

from __future__ import annotations

from pathlib import Path
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDialog

from app.core import reference_assets
from app.ui.reference_fallback_dialog import ReferenceFallbackDialog, check_reference_assets_fallback


@pytest.fixture(autouse=True)
def reset_reference_settings():
    """Reset QSettings for reference assets before and after each test."""
    reference_assets.set_custom_reference_override_path(None)
    reference_assets.reset_missing_reference_modal_seen()
    yield
    reference_assets.set_custom_reference_override_path(None)
    reference_assets.reset_missing_reference_modal_seen()


def _touch_all_stems(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for key in reference_assets.REFERENCE_STEM_KEYS:
        (directory / f"{key}.wav").write_bytes(b"")


def test_reference_fallback_dialog_init(qtbot):
    dialog = ReferenceFallbackDialog()
    qtbot.add_widget(dialog)

    assert "Missing Reference Audio Assets" in dialog.windowTitle()
    assert dialog._save_button is not None
    assert dialog._skip_button is not None
    assert dialog._path_edit is not None


def test_reference_fallback_dialog_status_indicator_empty(qtbot, tmp_path):
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()

    dialog = ReferenceFallbackDialog()
    qtbot.add_widget(dialog)

    dialog._path_edit.setText(str(empty_dir))
    assert "No matching stem files found" in dialog._status_summary_label.text()


def test_reference_fallback_dialog_status_indicator_all_stems(qtbot, tmp_path):
    stems_dir = tmp_path / "stems"
    _touch_all_stems(stems_dir)

    dialog = ReferenceFallbackDialog()
    qtbot.add_widget(dialog)

    dialog._path_edit.setText(str(stems_dir))
    assert "All 4 stem files present" in dialog._status_summary_label.text()


def test_reference_fallback_dialog_save_persists_path(qtbot, tmp_path, monkeypatch):
    stems_dir = tmp_path / "stems"
    _touch_all_stems(stems_dir)

    dialog = ReferenceFallbackDialog()
    qtbot.add_widget(dialog)
    dialog._path_edit.setText(str(stems_dir))

    qtbot.mouseClick(dialog._save_button, Qt.MouseButton.LeftButton)

    assert reference_assets.get_custom_reference_override_path() == stems_dir
    assert reference_assets.has_seen_missing_reference_modal() is True


def test_reference_fallback_dialog_skip_marks_seen(qtbot):
    dialog = ReferenceFallbackDialog()
    qtbot.add_widget(dialog)

    qtbot.mouseClick(dialog._skip_button, Qt.MouseButton.LeftButton)

    assert reference_assets.get_custom_reference_override_path() is None
    assert reference_assets.has_seen_missing_reference_modal() is True


def test_check_reference_assets_fallback_skips_when_seen(monkeypatch):
    reference_assets.set_seen_missing_reference_modal(True)
    modal_opened = False

    def dummy_exec(self):
        nonlocal modal_opened
        modal_opened = True
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ReferenceFallbackDialog, "exec", dummy_exec)

    result = check_reference_assets_fallback()
    assert modal_opened is False
    assert result is None


def test_check_reference_assets_fallback_skips_when_not_missing(tmp_path, monkeypatch):
    factory_dir = tmp_path / "factory"
    _touch_all_stems(factory_dir)
    monkeypatch.setattr(reference_assets, "FACTORY_REFERENCES_DIR", factory_dir)

    modal_opened = False

    def dummy_exec(self):
        nonlocal modal_opened
        modal_opened = True
        return QDialog.DialogCode.Rejected

    monkeypatch.setattr(ReferenceFallbackDialog, "exec", dummy_exec)

    result = check_reference_assets_fallback()
    assert modal_opened is False
