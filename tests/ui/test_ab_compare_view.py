"""Tests for app.ui.ab_compare_view (ABCompareView)."""

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

from app.ui.ab_compare_view import ABCompareView
from app.ui.waveform_player import WaveformPlayerWidget


def test_ab_compare_view_init(qtbot):
    view = ABCompareView()
    qtbot.addWidget(view)

    assert isinstance(view._original_player, WaveformPlayerWidget)
    assert isinstance(view._cleaned_player, WaveformPlayerWidget)
    assert view._original_player._title_label.text() == "<b>Original</b>"
    assert view._cleaned_player._title_label.text() == "<b>Cleaned</b>"
    assert view._sync_seek_cb.isChecked() is True


def test_ab_compare_view_load_and_clear(qtbot, tmp_path):
    view = ABCompareView()
    qtbot.addWidget(view)

    orig_wav = tmp_path / "original.wav"
    clean_wav = tmp_path / "cleaned.wav"

    sf.write(str(orig_wav), np.sin(np.linspace(0, 5, 44100)), 44100)
    sf.write(str(clean_wav), np.sin(np.linspace(0, 5, 44100)), 44100)

    view.load_original(orig_wav)
    view.load_cleaned(clean_wav)

    assert view._original_path == orig_wav
    assert view._cleaned_path == clean_wav
    assert view._original_player.get_file_path() == orig_wav
    assert view._cleaned_player.get_file_path() == clean_wav

    view.clear()

    assert view._original_path is None
    assert view._cleaned_path is None
    assert view._original_player.get_file_path() is None
    assert view._cleaned_player.get_file_path() is None


def test_ab_compare_view_sync_seek(qtbot, tmp_path):
    view = ABCompareView()
    qtbot.addWidget(view)

    orig_wav = tmp_path / "original.wav"
    clean_wav = tmp_path / "cleaned.wav"
    sf.write(str(orig_wav), np.sin(np.linspace(0, 5, 44100)), 44100)
    sf.write(str(clean_wav), np.sin(np.linspace(0, 5, 44100)), 44100)

    view.load_original(orig_wav)
    view.load_cleaned(clean_wav)

    # Trigger seek on original player
    view._original_player.seek(1200)

    assert view._original_player.get_position() == 1200
    assert view._cleaned_player.get_position() == 1200

    # Trigger seek on cleaned player
    view._cleaned_player.seek(800)

    assert view._original_player.get_position() == 800
    assert view._cleaned_player.get_position() == 800


def test_ab_compare_view_solo_toggle(qtbot):
    view = ABCompareView()
    qtbot.addWidget(view)

    view._original_player._fade_duration_ms = 0
    view._cleaned_player._fade_duration_ms = 0

    # Both mode (default)
    assert view._original_player._audio_output.isMuted() is False
    assert view._cleaned_player._audio_output.isMuted() is False

    # Original mode (A)
    view._radio_original.setChecked(True)
    assert view._original_player._audio_output.isMuted() is False
    assert view._cleaned_player._audio_output.isMuted() is True

    # Cleaned mode (B)
    view._radio_cleaned.setChecked(True)
    assert view._original_player._audio_output.isMuted() is True
    assert view._cleaned_player._audio_output.isMuted() is False


def test_ab_compare_view_playback_calls(qtbot, tmp_path):
    view = ABCompareView()
    qtbot.addWidget(view)

    orig_wav = tmp_path / "original.wav"
    clean_wav = tmp_path / "cleaned.wav"
    sf.write(str(orig_wav), np.zeros(44100), 44100)
    sf.write(str(clean_wav), np.zeros(44100), 44100)

    view.load_original(orig_wav)
    view.load_cleaned(clean_wav)

    view.play_both()
    view.pause_both()
    view.stop_both()


def test_ab_compare_view_smooth_fading(qtbot, tmp_path):
    view = ABCompareView()
    qtbot.addWidget(view)

    orig_wav = tmp_path / "original.wav"
    sf.write(str(orig_wav), np.zeros(44100), 44100)
    view.load_original(orig_wav)

    # We set fade duration to a non-zero value to test the timer fade
    view._original_player._fade_duration_ms = 10
    view._original_player._fade_interval_ms = 2

    view._original_player.set_muted(True)
    assert view._original_player._muted_state is True

    # Process events to allow timer ticks to run
    qtbot.wait_until(lambda: view._original_player._audio_output.volume() == 0.0, timeout=1000)
    assert view._original_player._audio_output.isMuted() is True


def test_ab_compare_view_drift_synchronization(qtbot, tmp_path):
    view = ABCompareView()
    qtbot.addWidget(view)

    orig_wav = tmp_path / "original.wav"
    clean_wav = tmp_path / "cleaned.wav"
    sf.write(str(orig_wav), np.zeros(44100), 44100)
    sf.write(str(clean_wav), np.zeros(44100), 44100)

    view.load_original(orig_wav)
    view.load_cleaned(clean_wav)

    # Disable fading to make assertions instantaneous
    view._original_player._fade_duration_ms = 0
    view._cleaned_player._fade_duration_ms = 0

    # Ensure "Sync Playhead" is checked
    assert view._sync_seek_cb.isChecked() is True

    # Check drift logic triggering: original position changed significantly
    # Let's mock or directly call on_original_position_changed
    view._original_player._position_ms = 500
    view._cleaned_player._position_ms = 100

    view.on_original_position_changed(500)
    assert view._cleaned_player.get_position() == 500


def test_ab_compare_view_playback_state_synchronization(qtbot, tmp_path):
    from PySide6.QtMultimedia import QMediaPlayer
    view = ABCompareView()
    qtbot.addWidget(view)

    orig_wav = tmp_path / "original.wav"
    clean_wav = tmp_path / "cleaned.wav"
    sf.write(str(orig_wav), np.zeros(44100), 44100)
    sf.write(str(clean_wav), np.zeros(44100), 44100)

    view.load_original(orig_wav)
    view.load_cleaned(clean_wav)

    # Disable fading to make assertions instantaneous
    view._original_player._fade_duration_ms = 0
    view._cleaned_player._fade_duration_ms = 0

    # Mock media source validity and mock play/pause directly, or use direct slot trigger
    # QMediaPlayer with offline file may not transition to PlayingState without an audio output backend
    # We can mock the _media_player.playbackState or the actual behavior since we want to test state sync logic
    view._original_player._media_player.play = MagicMock()
    view._cleaned_player._media_player.play = MagicMock()
    view._original_player._media_player.pause = MagicMock()
    view._cleaned_player._media_player.pause = MagicMock()

    # Trigger state change on original player and check if cleaned player mirrors it
    view.on_original_state_changed(QMediaPlayer.PlaybackState.PlayingState)
    view._cleaned_player._media_player.play.assert_called_once()

    view.on_original_state_changed(QMediaPlayer.PlaybackState.PausedState)
    view._cleaned_player._media_player.pause.assert_called_once()
