"""Tests for app.ui.waveform_player (WaveformPlayerWidget and compute_waveform_data)."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

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

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QMouseEvent

from app.ui.waveform_player import (
    WaveformCanvas,
    WaveformData,
    WaveformPlayerWidget,
    compute_waveform_data,
    format_time_ms,
)


def test_format_time_ms():
    assert format_time_ms(0) == "00:00"
    assert format_time_ms(1000) == "00:01"
    assert format_time_ms(65000) == "01:05"
    assert format_time_ms(3661000) == "61:01"


def test_compute_waveform_data_from_array():
    sr = 44100
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sine_wave = np.sin(2 * np.pi * 440 * t)

    data = compute_waveform_data(sine_wave, num_bins=400, sample_rate=sr)

    assert isinstance(data, WaveformData)
    assert len(data.peaks) == 400
    assert len(data.rms) == 400
    assert data.total_samples == sr
    assert abs(data.duration_seconds - 1.0) < 1e-3
    assert np.max(data.peaks) == pytest.approx(1.0, abs=1e-3)
    assert np.min(data.peaks) >= 0.0
    assert np.min(data.rms) >= 0.0


def test_compute_waveform_data_from_file(tmp_path):
    wav_path = tmp_path / "test_tone.wav"
    sr = 44100
    data_in = np.sin(np.linspace(0, 2 * np.pi * 100, sr)).astype(np.float32)
    sf.write(str(wav_path), data_in, sr)

    data = compute_waveform_data(wav_path, num_bins=200)

    assert isinstance(data, WaveformData)
    assert len(data.peaks) == 200
    assert data.total_samples == sr
    assert abs(data.duration_seconds - 1.0) < 1e-3
    assert np.max(data.peaks) == pytest.approx(1.0, abs=1e-3)


def test_compute_waveform_data_edge_cases(tmp_path):
    # Non-existent file
    missing_data = compute_waveform_data(tmp_path / "nonexistent.wav", num_bins=100)
    assert missing_data.total_samples == 0
    assert np.all(missing_data.peaks == 0)

    # Empty array
    empty_data = compute_waveform_data(np.array([]), num_bins=100)
    assert empty_data.total_samples == 0
    assert np.all(empty_data.peaks == 0)

    # Silent array
    silent_data = compute_waveform_data(np.zeros(1000), num_bins=50)
    assert silent_data.total_samples == 1000
    assert np.all(silent_data.peaks == 0)

    # Unsupported type
    with pytest.raises(TypeError):
        compute_waveform_data(12345)  # type: ignore


def test_waveform_player_widget_init(qtbot):
    player = WaveformPlayerWidget(title="Original Track")
    qtbot.addWidget(player)

    assert player._title_label.text() == "<b>Original Track</b>"
    assert player._file_label.text() == "No track loaded"
    assert player.get_duration() == 0
    assert player.get_position() == 0
    assert player.get_file_path() is None
    assert not player._play_button.isEnabled()


def test_waveform_player_widget_load_and_clear(qtbot, tmp_path):
    player = WaveformPlayerWidget(title="Test Player")
    qtbot.addWidget(player)

    wav_path = tmp_path / "sample.wav"
    sf.write(str(wav_path), np.zeros((44100, 2)), 44100)

    player.load_file(wav_path)

    assert player.get_file_path() == wav_path
    assert player._file_label.text() == "sample.wav"
    assert player._play_button.isEnabled()
    assert player._stop_button.isEnabled()
    assert player._canvas._waveform_data is not None

    player.clear()

    assert player.get_file_path() is None
    assert player._file_label.text() == "No track loaded"
    assert not player._play_button.isEnabled()
    assert player._canvas._waveform_data is None


def test_waveform_player_widget_controls_and_seeking(qtbot, tmp_path):
    player = WaveformPlayerWidget(title="Test Controls")
    qtbot.addWidget(player)

    wav_path = tmp_path / "sample.wav"
    sf.write(str(wav_path), np.sin(np.linspace(0, 10, 44100)), 44100)
    player.load_file(wav_path)

    # Test seek
    player.seek(500)
    assert player.get_position() == 500

    # Test volume
    player.set_volume(0.5)
    assert player._volume_slider.value() == 50

    player.set_muted(True)
    assert player._audio_output.isMuted() is True


def test_waveform_canvas_interaction(qtbot):
    canvas = WaveformCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 100)

    sample_array = np.sin(np.linspace(0, 10, 44100))
    wf_data = compute_waveform_data(sample_array, num_bins=200)
    canvas.set_waveform_data(wf_data)
    canvas.set_duration(2000)
    canvas.set_playhead_position(500)

    # Force repaint
    canvas.repaint()

    # Simulate click seek
    with qtbot.waitSignal(canvas.seekRequested, timeout=1000) as blocker:
        center_pt = QPointF(canvas.rect().center())
        event = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            center_pt,
            center_pt,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        canvas.mousePressEvent(event)

    assert blocker.args[0] == pytest.approx(1000, abs=100)


def test_waveform_canvas_keyboard_seeking(qtbot):
    from PySide6.QtGui import QKeyEvent
    canvas = WaveformCanvas()
    qtbot.addWidget(canvas)
    canvas.resize(200, 100)

    sample_array = np.sin(np.linspace(0, 10, 44100))
    wf_data = compute_waveform_data(sample_array, num_bins=200)
    canvas.set_waveform_data(wf_data)
    canvas.set_duration(30000)  # 30 seconds
    canvas.set_playhead_position(10000)  # At 10 seconds

    # Check key navigation (Right arrow key -> +5s -> 15s)
    with qtbot.waitSignal(canvas.seekRequested, timeout=1000) as blocker:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Right, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(event)
    assert blocker.args[0] == 15000

    # Check Left arrow key (Left arrow key -> -5s -> 5s)
    with qtbot.waitSignal(canvas.seekRequested, timeout=1000) as blocker:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_Left, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(event)
    assert blocker.args[0] == 5000

    # Check PageDown key (+15s -> 25s)
    with qtbot.waitSignal(canvas.seekRequested, timeout=1000) as blocker:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_PageDown, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(event)
    assert blocker.args[0] == 25000

    # Check PageUp key (-15s -> 0s)
    canvas.set_playhead_position(10000)
    with qtbot.waitSignal(canvas.seekRequested, timeout=1000) as blocker:
        event = QKeyEvent(QKeyEvent.Type.KeyPress, Qt.Key.Key_PageUp, Qt.KeyboardModifier.NoModifier)
        canvas.keyPressEvent(event)
    assert blocker.args[0] == 0


def test_waveform_player_widget_accessibility(qtbot):
    player = WaveformPlayerWidget(title="A11y Player")
    qtbot.addWidget(player)

    assert player._canvas.accessibleName() == "A11y Player Waveform Display"
    assert "use keyboard to seek" in player._canvas.accessibleDescription()
    assert player._play_button.accessibleName() == "Play"
    assert player._stop_button.accessibleName() == "Stop"
    assert player._volume_slider.accessibleName() == "Volume"
    assert player._canvas.focusPolicy() == Qt.FocusPolicy.StrongFocus
