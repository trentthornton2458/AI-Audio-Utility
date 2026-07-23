"""Tests for PySide6 MainWindow and UI components (app.ui.main_window)."""

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
from PySide6.QtCore import QMimeData, QPointF, Qt, QUrl
from PySide6.QtGui import QDragEnterEvent, QDropEvent

from app.core.ingestion import UnsupportedAudioFormatError
from app.models.preset import Preset
from app.ui.main_window import (
    FileLoadPanel,
    MainWindow,
    StemSeparationPanel,
)
from app.ui.vocal_panel import VocalPanel
from app.ui.instrumental_panel import InstrumentalPanel


def test_main_window_structure_and_title(qtbot):
    mock_cache = MagicMock()
    window = MainWindow(cache_manager=mock_cache)
    qtbot.addWidget(window)

    assert window.windowTitle() == "Music Mastery Enhancer"
    assert window._tab_widget.count() == 4
    assert window._tab_widget.tabText(0) == "Stem Separation"
    assert window._tab_widget.tabText(1) == "Vocal Controls"
    assert window._tab_widget.tabText(2) == "Instrumental Controls"
    assert window._tab_widget.tabText(3) == "A/B Compare"
    assert isinstance(window._stem_separation_panel, StemSeparationPanel)
    assert isinstance(window._vocal_panel, VocalPanel)
    assert isinstance(window._instrumental_panel, InstrumentalPanel)
    assert window._status_label.text() == "Ready. Load a .wav or .mp3 track to begin."


def test_file_load_panel_drag_enter_accepts_supported_audio(qtbot, tmp_path):
    panel = FileLoadPanel()
    qtbot.addWidget(panel)

    wav_file = tmp_path / "test.wav"
    wav_file.touch()

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(wav_file))])

    event = QDragEnterEvent(
        panel.rect().center(),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    with patch.object(event, "acceptProposedAction") as mock_accept:
        panel.dragEnterEvent(event)
        mock_accept.assert_called_once()


def test_file_load_panel_drag_enter_ignores_unsupported_file(qtbot, tmp_path):
    panel = FileLoadPanel()
    qtbot.addWidget(panel)

    txt_file = tmp_path / "test.txt"
    txt_file.touch()

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(txt_file))])

    event = QDragEnterEvent(
        panel.rect().center(),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    with patch.object(event, "ignore") as mock_ignore:
        panel.dragEnterEvent(event)
        mock_ignore.assert_called_once()


def test_file_load_panel_drop_emits_file_selected(qtbot, tmp_path):
    panel = FileLoadPanel()
    qtbot.addWidget(panel)

    mp3_file = tmp_path / "song.mp3"
    mp3_file.touch()

    mime_data = QMimeData()
    mime_data.setUrls([QUrl.fromLocalFile(str(mp3_file))])

    event = QDropEvent(
        QPointF(panel.rect().center()),
        Qt.DropAction.CopyAction,
        mime_data,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )

    with qtbot.waitSignal(panel.fileSelected, timeout=1000) as blocker:
        panel.dropEvent(event)

    assert blocker.args[0] == mp3_file


def test_main_window_ingestion_success(qtbot, tmp_path):
    mock_cache = MagicMock()
    window = MainWindow(cache_manager=mock_cache)
    qtbot.addWidget(window)

    input_path = tmp_path / "input.wav"
    input_path.touch()

    norm_path = tmp_path / "cache" / "normalized.wav"

    with patch("app.ui.main_window.load_and_normalize_track", return_value=norm_path):
        window.on_file_selected(input_path)

        assert window._current_input_path == input_path
        assert window._normalized_path == norm_path
        assert "Track ingested and normalized" in window._status_label.text()
        assert window._render_button.isEnabled()


def test_main_window_ingestion_failure(qtbot, tmp_path):
    mock_cache = MagicMock()
    window = MainWindow(cache_manager=mock_cache)
    qtbot.addWidget(window)

    invalid_file = tmp_path / "invalid.wav"
    invalid_file.touch()

    with patch(
        "app.ui.main_window.load_and_normalize_track",
        side_effect=UnsupportedAudioFormatError("Unsupported sample format"),
    ), patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msgbox:
        window.on_file_selected(invalid_file)

        mock_msgbox.assert_called_once()
        assert "Error:" in window._status_label.text()
        assert window._current_input_path is None


def test_main_window_render_job_signals_wiring(qtbot, tmp_path):
    mock_cache = MagicMock()
    window = MainWindow(cache_manager=mock_cache)
    qtbot.addWidget(window)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    window._current_input_path = input_path

    mock_job = MagicMock()
    mock_job.isRunning.return_value = False

    with patch("app.ui.main_window.RenderJob", return_value=mock_job) as mock_job_cls:
        window.start_render_job(Preset())

        mock_job_cls.assert_called_once()

        # Test stage changed
        window.on_render_stage_changed("Separating")
        assert "Separating" in window._status_label.text()

        # Test progress changed
        window.on_render_progress_changed(0.45)
        assert window._progress_bar.value() == 45

        # Test finished
        output_file = tmp_path / "render.wav"
        with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
            window.on_render_finished(output_file)
            mock_info.assert_called_once()
            assert "Render completed" in window._status_label.text()
            assert not window._progress_bar.isVisible()


def test_main_window_render_job_failure_and_cancel_handling(qtbot, tmp_path):
    mock_cache = MagicMock()
    window = MainWindow(cache_manager=mock_cache)
    qtbot.addWidget(window)

    window._current_input_path = tmp_path / "input.wav"

    mock_job = MagicMock()
    mock_job.isRunning.return_value = True

    window._active_render_job = mock_job

    # Test failed slot
    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_err_msg:
        window.on_render_failed("CUDA Out of Memory")
        mock_err_msg.assert_called_once()
        assert "Render failed" in window._status_label.text()
        assert window._active_render_job is None

    # Test cancelled slot
    window._active_render_job = mock_job
    window.on_render_cancelled()
    assert "Render cancelled" in window._status_label.text()
    assert window._active_render_job is None

    # Test cancel button click
    window._active_render_job = mock_job
    window.on_cancel_render_clicked()
    mock_job.cancel.assert_called_once()


def test_extract_stems_runs_separation_only_not_full_render(qtbot, tmp_path):
    """The 'Extract Stems' button must launch a SeparationJob, never the full RenderJob."""
    mock_cache = MagicMock()
    window = MainWindow(cache_manager=mock_cache)
    qtbot.addWidget(window)

    input_path = tmp_path / "input.wav"
    input_path.touch()
    window._current_input_path = input_path
    window._normalized_path = tmp_path / "cache" / "track" / "normalized.wav"

    mock_job = MagicMock()
    mock_job.isRunning.return_value = False

    with patch("app.ui.main_window.SeparationJob", return_value=mock_job) as mock_sep_cls, patch(
        "app.ui.main_window.RenderJob"
    ) as mock_render_cls:
        window.on_extract_stems_requested()

        mock_sep_cls.assert_called_once()
        mock_job.start.assert_called_once()
        mock_render_cls.assert_not_called()  # crucially, NOT the full pipeline
        assert window._active_separation_job is mock_job

    # Finished slot updates status and clears the active job
    vocal = tmp_path / "vocal.wav"
    instrumental = tmp_path / "instrumental.wav"
    with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
        window.on_separation_finished(vocal, instrumental)
        mock_info.assert_called_once()
        assert window._active_separation_job is None
        assert not window._progress_bar.isVisible()


def test_extract_stems_without_track_warns(qtbot):
    mock_cache = MagicMock()
    window = MainWindow(cache_manager=mock_cache)
    qtbot.addWidget(window)

    with patch("app.ui.main_window.SeparationJob") as mock_sep_cls, patch(
        "PySide6.QtWidgets.QMessageBox.warning"
    ) as mock_warn:
        window.on_extract_stems_requested()
        mock_warn.assert_called_once()
        mock_sep_cls.assert_not_called()
