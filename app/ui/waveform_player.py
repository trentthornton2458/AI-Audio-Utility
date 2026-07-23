"""PySide6 Waveform Player Widget for Music Mastery Enhancer.

Renders audio waveforms (peak & RMS envelopes computed via numpy) using QPainter,
supports click/drag-to-seek, and controls audio playback via QMediaPlayer/QAudioOutput.
Audio loading and waveform envelope computation are separated into pure functions
for testability without a Qt display.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
from PySide6.QtCore import QPoint, QRect, QSize, Qt, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QFont, QMouseEvent, QPaintEvent, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.cache import get_logger

logger = get_logger(__name__)


@dataclass
class WaveformData:
    """Dataclass holding computed peak and RMS envelope arrays and metadata."""

    peaks: np.ndarray  # 1D float array, values in [0.0, 1.0]
    rms: np.ndarray    # 1D float array, values in [0.0, 1.0]
    duration_seconds: float
    sample_rate: int
    total_samples: int


def compute_waveform_data(
    audio_source: Union[Path, str, np.ndarray],
    num_bins: int = 800,
    sample_rate: int = 44100,
) -> WaveformData:
    """Compute peak and RMS waveform envelope data from an audio file or numpy array.

    This function contains pure numpy/soundfile computation and has no Qt dependency,
    making it fully testable in headless environments.
    """
    if isinstance(audio_source, (Path, str)):
        path = Path(audio_source)
        if not path.is_file():
            logger.warning("Audio file not found for waveform computation: %s", path)
            return WaveformData(
                peaks=np.zeros(num_bins, dtype=np.float32),
                rms=np.zeros(num_bins, dtype=np.float32),
                duration_seconds=0.0,
                sample_rate=sample_rate,
                total_samples=0,
            )
        try:
            data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        except Exception as exc:
            logger.error("Failed to read audio file for waveform: %s (%s)", path, exc)
            return WaveformData(
                peaks=np.zeros(num_bins, dtype=np.float32),
                rms=np.zeros(num_bins, dtype=np.float32),
                duration_seconds=0.0,
                sample_rate=sample_rate,
                total_samples=0,
            )
    elif isinstance(audio_source, np.ndarray):
        sr = sample_rate
        if audio_source.ndim == 1:
            data = audio_source[:, np.newaxis].astype(np.float32)
        else:
            data = audio_source.astype(np.float32)
    else:
        raise TypeError(f"Unsupported audio_source type: {type(audio_source)}")

    total_samples = len(data)
    if total_samples == 0:
        return WaveformData(
            peaks=np.zeros(num_bins, dtype=np.float32),
            rms=np.zeros(num_bins, dtype=np.float32),
            duration_seconds=0.0,
            sample_rate=sr,
            total_samples=0,
        )

    duration_seconds = total_samples / sr
    mono_peaks = np.max(np.abs(data), axis=1)

    peaks = np.zeros(num_bins, dtype=np.float32)
    rms = np.zeros(num_bins, dtype=np.float32)

    chunks = np.array_split(mono_peaks, num_bins)
    for i, chunk in enumerate(chunks):
        if len(chunk) > 0:
            peaks[i] = float(np.max(chunk))
            rms[i] = float(np.sqrt(np.mean(chunk**2)))

    max_peak = float(np.max(peaks)) if len(peaks) > 0 else 0.0
    if max_peak > 1e-6:
        peaks = peaks / max_peak
        rms = rms / max_peak
        peaks = np.clip(peaks, 0.0, 1.0)
        rms = np.clip(rms, 0.0, 1.0)

    return WaveformData(
        peaks=peaks,
        rms=rms,
        duration_seconds=duration_seconds,
        sample_rate=sr,
        total_samples=total_samples,
    )


def format_time_ms(ms: int) -> str:
    """Format milliseconds into MM:SS format."""
    total_seconds = max(0, int(round(ms / 1000.0)))
    minutes = total_seconds // 60
    seconds = total_seconds % 60
    return f"{minutes:02d}:{seconds:02d}"


class WaveformCanvas(QWidget):
    """Internal canvas widget rendering waveform peak/RMS envelopes and moving playhead."""

    seekRequested = Signal(int)  # Emits target position in ms

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setMinimumHeight(100)

        self._waveform_data: Optional[WaveformData] = None
        self._playhead_position_ms: int = 0
        self._duration_ms: int = 0
        self._hover_x: Optional[int] = None

    def set_waveform_data(self, data: Optional[WaveformData]) -> None:
        self._waveform_data = data
        self.update()

    def set_playhead_position(self, position_ms: int) -> None:
        self._playhead_position_ms = max(0, position_ms)
        self.update()

    def set_duration(self, duration_ms: int) -> None:
        self._duration_ms = max(0, duration_ms)
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)

        width = self.width()
        height = self.height()

        # Canvas Background
        painter.fillRect(0, 0, width, height, QColor("#161722"))

        if self._waveform_data is None or self._waveform_data.total_samples == 0:
            painter.setPen(QPen(QColor("#5a5d72")))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No Waveform Loaded")
            return

        center_y = height // 2
        # Center Baseline
        painter.setPen(QPen(QColor("#2d2f3d"), 1, Qt.PenStyle.DashLine))
        painter.drawLine(0, center_y, width, center_y)

        peaks = self._waveform_data.peaks
        rms = self._waveform_data.rms
        num_bins = len(peaks)

        duration_ms = self._duration_ms or int(self._waveform_data.duration_seconds * 1000)
        playhead_ratio = (
            min(1.0, max(0.0, self._playhead_position_ms / duration_ms)) if duration_ms > 0 else 0.0
        )
        playhead_x = int(playhead_ratio * width)

        max_amplitude_h = (height // 2) - 6

        for x in range(width):
            bin_idx = int((x / max(1, width)) * num_bins)
            bin_idx = min(num_bins - 1, max(0, bin_idx))

            peak_val = float(peaks[bin_idx])
            rms_val = float(rms[bin_idx])

            peak_h = max(1, int(peak_val * max_amplitude_h))
            rms_h = max(0, int(rms_val * max_amplitude_h))

            is_played = x <= playhead_x

            # Peak Envelope Line
            peak_color = QColor("#7d6dfa") if is_played else QColor("#3d3f4d")
            painter.setPen(QPen(peak_color, 1))
            painter.drawLine(x, center_y - peak_h, x, center_y + peak_h)

            # RMS Envelope Line
            if rms_h > 0:
                rms_color = QColor("#55efc4") if is_played else QColor("#22443a")
                painter.setPen(QPen(rms_color, 1))
                painter.drawLine(x, center_y - rms_h, x, center_y + rms_h)

        # Draw Playhead Line & Handle
        if duration_ms > 0:
            playhead_pen = QPen(QColor("#ff7675"), 2)
            painter.setPen(playhead_pen)
            painter.drawLine(playhead_x, 0, playhead_x, height)

            painter.setBrush(QColor("#ff7675"))
            painter.drawEllipse(QPoint(playhead_x, 4), 4, 4)

        # Draw Hover Position Line & Time Tooltip
        if self._hover_x is not None and duration_ms > 0:
            hover_pen = QPen(QColor("#a0a5b5"), 1, Qt.PenStyle.DotLine)
            painter.setPen(hover_pen)
            painter.drawLine(self._hover_x, 0, self._hover_x, height)

            hover_ratio = max(0.0, min(1.0, self._hover_x / max(1, width)))
            hover_ms = int(hover_ratio * duration_ms)
            time_str = format_time_ms(hover_ms)

            text_rect = QRect(
                max(0, min(width - 55, self._hover_x - 27)),
                2,
                54,
                18,
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#2b2d3e"))
            painter.drawRoundedRect(text_rect, 3, 3)

            painter.setPen(QColor("#ffffff"))
            painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignCenter, time_str)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._emit_seek_from_pos(event.position().x())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        x = int(event.position().x())
        self._hover_x = max(0, min(self.width(), x))
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._emit_seek_from_pos(event.position().x())
        self.update()

    def leaveEvent(self, event) -> None:
        self._hover_x = None
        self.update()

    def _emit_seek_from_pos(self, mouse_x: float) -> None:
        duration_ms = self._duration_ms
        if self._waveform_data and duration_ms == 0:
            duration_ms = int(self._waveform_data.duration_seconds * 1000)

        if duration_ms > 0 and self.width() > 0:
            ratio = max(0.0, min(1.0, mouse_x / self.width()))
            seek_ms = int(ratio * duration_ms)
            self.seekRequested.emit(seek_ms)


class WaveformPlayerWidget(QWidget):
    """Audio waveform player widget supporting scrubbing, click-to-seek, and QMediaPlayer controls."""

    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playbackStateChanged = Signal(QMediaPlayer.PlaybackState)
    seekRequested = Signal(int)

    def __init__(
        self,
        title: str = "Audio Player",
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._title = title
        self._current_file_path: Optional[Path] = None
        self._duration_ms: int = 0
        self._position_ms: int = 0

        self._media_player = QMediaPlayer(self)
        self._audio_output = QAudioOutput(self)
        self._media_player.setAudioOutput(self._audio_output)

        self._target_volume: float = 1.0
        self._muted_state: bool = False
        self._fade_duration_ms: int = 50
        self._fade_interval_ms: int = 10
        self._fade_timer = QTimer(self)
        self._fade_timer.timeout.connect(self._on_fade_tick)

        self.setObjectName("WaveformPlayerWidget")
        self._init_ui()
        self._wire_signals()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Card Frame
        self._card = QFrame()
        self._card.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; }"
        )
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(8)

        # Header bar
        header_layout = QHBoxLayout()
        self._title_label = QLabel(f"<b>{self._title}</b>")
        self._title_label.setStyleSheet("color: #7d6dfa; font-size: 13px;")

        self._file_label = QLabel("No track loaded")
        self._file_label.setStyleSheet("color: #8a8d9b; font-size: 11px;")

        self._time_label = QLabel("00:00 / 00:00")
        self._time_label.setStyleSheet("color: #55efc4; font-weight: bold; font-size: 12px;")

        header_layout.addWidget(self._title_label)
        header_layout.addWidget(self._file_label)
        header_layout.addStretch()
        header_layout.addWidget(self._time_label)
        card_layout.addLayout(header_layout)

        # Canvas
        self._canvas = WaveformCanvas()
        self._canvas.seekRequested.connect(self.on_canvas_seek_requested)
        card_layout.addWidget(self._canvas)

        # Controls bar
        ctrl_layout = QHBoxLayout()
        ctrl_layout.setSpacing(8)

        self._play_button = QPushButton("Play")
        self._play_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._play_button.setEnabled(False)
        self._play_button.setStyleSheet(
            "QPushButton { background-color: #6c5ce7; color: white; font-weight: bold; border-radius: 4px; padding: 6px 14px; border: none; }"
            "QPushButton:hover { background-color: #7d6dfa; }"
            "QPushButton:disabled { background-color: #3d3f4d; color: #8a8d9b; }"
        )
        self._play_button.clicked.connect(self.on_play_clicked)
        ctrl_layout.addWidget(self._play_button)

        self._stop_button = QPushButton("Stop")
        self._stop_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._stop_button.setEnabled(False)
        self._stop_button.setStyleSheet(
            "QPushButton { background-color: #3b3e54; color: white; font-weight: bold; border-radius: 4px; padding: 6px 12px; border: none; }"
            "QPushButton:hover { background-color: #4b4e69; }"
            "QPushButton:disabled { background-color: #3d3f4d; color: #8a8d9b; }"
        )
        self._stop_button.clicked.connect(self.on_stop_clicked)
        ctrl_layout.addWidget(self._stop_button)

        ctrl_layout.addStretch()

        vol_icon = QLabel("Vol:")
        vol_icon.setStyleSheet("color: #a0a5b5; font-size: 11px;")
        ctrl_layout.addWidget(vol_icon)

        self._volume_slider = QSlider(Qt.Orientation.Horizontal)
        self._volume_slider.setRange(0, 100)
        self._volume_slider.setValue(100)
        self._volume_slider.setFixedWidth(80)
        self._volume_slider.setStyleSheet(
            "QSlider::groove:horizontal { border: 1px solid #2d2f3d; height: 4px; background: #1a1b24; border-radius: 2px; }"
            "QSlider::sub-page:horizontal { background: #55efc4; border-radius: 2px; }"
            "QSlider::handle:horizontal { background: #ffffff; border: 1px solid #55efc4; width: 12px; margin-top: -4px; margin-bottom: -4px; border-radius: 6px; }"
        )
        self._volume_slider.valueChanged.connect(self.on_volume_changed)
        ctrl_layout.addWidget(self._volume_slider)

        card_layout.addLayout(ctrl_layout)
        main_layout.addWidget(self._card)

    def _wire_signals(self) -> None:
        self._media_player.positionChanged.connect(self.on_media_position_changed)
        self._media_player.durationChanged.connect(self.on_media_duration_changed)
        self._media_player.playbackStateChanged.connect(self.on_media_playback_state_changed)

    # --- Public API Methods ---

    def load_file(self, file_path: Union[Path, str]) -> None:
        """Load an audio file, compute waveform envelope, and configure player."""
        path = Path(file_path)
        logger.info("Loading audio file into WaveformPlayerWidget (%s): %s", self._title, path)

        waveform_data = compute_waveform_data(path)
        self._current_file_path = path
        self._canvas.set_waveform_data(waveform_data)

        self._media_player.setSource(QUrl.fromLocalFile(str(path)))
        self._file_label.setText(path.name)
        self._play_button.setEnabled(True)
        self._stop_button.setEnabled(True)

    def clear(self) -> None:
        """Reset player state and clear waveform display."""
        self._media_player.stop()
        self._media_player.setSource(QUrl())
        self._current_file_path = None
        self._duration_ms = 0
        self._position_ms = 0
        self._canvas.set_waveform_data(None)
        self._canvas.set_playhead_position(0)
        self._canvas.set_duration(0)
        self._file_label.setText("No track loaded")
        self._play_button.setEnabled(False)
        self._stop_button.setEnabled(False)
        self._update_time_label()

        self._fade_timer.stop()
        self._muted_state = False
        self._target_volume = 1.0
        self._audio_output.setVolume(1.0)
        self._audio_output.setMuted(False)

    def play(self) -> None:
        if self._media_player.source().isValid():
            self._media_player.play()

    def pause(self) -> None:
        self._media_player.pause()

    def stop(self) -> None:
        self._media_player.stop()

    def seek(self, position_ms: int) -> None:
        """Seek player position and update canvas playhead."""
        position_ms = max(0, min(self._duration_ms or position_ms, position_ms))
        self._media_player.setPosition(position_ms)
        self._position_ms = position_ms
        self._canvas.set_playhead_position(position_ms)
        self._update_time_label()
        self.seekRequested.emit(position_ms)

    def set_title(self, title: str) -> None:
        self._title = title
        self._title_label.setText(f"<b>{title}</b>")

    def set_volume(self, volume: float) -> None:
        """Set playback volume from 0.0 to 1.0."""
        self._target_volume = max(0.0, min(1.0, volume))
        vol_int = int(round(self._target_volume * 100))

        self._volume_slider.blockSignals(True)
        self._volume_slider.setValue(vol_int)
        self._volume_slider.blockSignals(False)

        if not self._muted_state:
            if self._fade_duration_ms <= 0:
                self._audio_output.setVolume(self._target_volume)
            else:
                self._fade_timer.start(self._fade_interval_ms)

    def set_muted(self, muted: bool) -> None:
        if self._muted_state == muted:
            return

        self._muted_state = muted
        if muted:
            if self._fade_duration_ms <= 0:
                self._audio_output.setVolume(0.0)
                self._audio_output.setMuted(True)
            else:
                self._fade_timer.start(self._fade_interval_ms)
        else:
            self._audio_output.setMuted(False)
            if self._fade_duration_ms <= 0:
                self._audio_output.setVolume(self._target_volume)
            else:
                self._fade_timer.start(self._fade_interval_ms)

    def is_playing(self) -> bool:
        return self._media_player.playbackState() == QMediaPlayer.PlaybackState.PlayingState

    def get_duration(self) -> int:
        return self._duration_ms

    def get_position(self) -> int:
        return self._position_ms

    def get_file_path(self) -> Optional[Path]:
        return self._current_file_path

    # --- Internal Slots & Event Handlers ---

    @Slot()
    def on_play_clicked(self) -> None:
        if self.is_playing():
            self.pause()
        else:
            self.play()

    @Slot()
    def on_stop_clicked(self) -> None:
        self.stop()

    @Slot(int)
    def on_volume_changed(self, value: int) -> None:
        vol_float = value / 100.0
        self.set_volume(vol_float)

    @Slot()
    def _on_fade_tick(self) -> None:
        dest_volume = 0.0 if self._muted_state else self._target_volume
        current_volume = self._audio_output.volume()

        step = self._fade_interval_ms / max(1, self._fade_duration_ms)

        if current_volume < dest_volume:
            new_vol = min(dest_volume, current_volume + step)
            self._audio_output.setVolume(new_vol)
        elif current_volume > dest_volume:
            new_vol = max(dest_volume, current_volume - step)
            self._audio_output.setVolume(new_vol)

        # Check if we reached the destination
        if abs(self._audio_output.volume() - dest_volume) < 1e-5:
            self._audio_output.setVolume(dest_volume)
            self._fade_timer.stop()
            if self._muted_state:
                self._audio_output.setMuted(True)

    @Slot(int)
    def on_canvas_seek_requested(self, position_ms: int) -> None:
        self.seek(position_ms)

    @Slot(int)
    def on_media_position_changed(self, position_ms: int) -> None:
        self._position_ms = position_ms
        self._canvas.set_playhead_position(position_ms)
        self._update_time_label()
        self.positionChanged.emit(position_ms)

    @Slot(int)
    def on_media_duration_changed(self, duration_ms: int) -> None:
        self._duration_ms = duration_ms
        self._canvas.set_duration(duration_ms)
        self._update_time_label()
        self.durationChanged.emit(duration_ms)

    @Slot(QMediaPlayer.PlaybackState)
    def on_media_playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        if state == QMediaPlayer.PlaybackState.PlayingState:
            self._play_button.setText("Pause")
            self._play_button.setStyleSheet(
                "QPushButton { background-color: #ff7675; color: white; font-weight: bold; border-radius: 4px; padding: 6px 14px; border: none; }"
                "QPushButton:hover { background-color: #ff6b6b; }"
            )
        else:
            self._play_button.setText("Play")
            self._play_button.setStyleSheet(
                "QPushButton { background-color: #6c5ce7; color: white; font-weight: bold; border-radius: 4px; padding: 6px 14px; border: none; }"
                "QPushButton:hover { background-color: #7d6dfa; }"
            )
        self.playbackStateChanged.emit(state)

    def _update_time_label(self) -> None:
        pos_str = format_time_ms(self._position_ms)
        dur_str = format_time_ms(self._duration_ms)
        self._time_label.setText(f"{pos_str} / {dur_str}")
