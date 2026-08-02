"""PySide6 Spectrogram View for Music Mastery Enhancer.

Renders a static Original-vs-Cleaned frequency-domain view of a render, computed post-render
on the exported WAV files (no live/real-time constraint applies to this offline batch
renderer). Follows the same pure-function-computes-data / QWidget-just-paints-it split as
app.ui.waveform_player: compute_spectrogram_data() is Qt-independent numpy/soundfile/matplotlib
computation, testable in headless environments, while SpectrogramCanvas only paints an
already-rendered QPixmap.

Uses matplotlib's Agg backend (no scipy/librosa/pyqtgraph available in this project) to
rasterize a spectrogram image once per load/resize, rather than embedding a live
FigureCanvasQTAgg -- there is nothing to update in real time here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import soundfile as sf
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPaintEvent, QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from app.cache import get_logger

logger = get_logger(__name__)

DEFAULT_N_FFT = 2048
DEFAULT_HOP_LENGTH = 512
DEFAULT_MAX_FREQ_HZ = 16000.0
_DB_FLOOR = -100.0


@dataclass
class SpectrogramData:
    """Dataclass holding a computed dB-scale magnitude spectrogram and its axes."""

    magnitudes_db: np.ndarray  # 2D float array, shape (n_freq_bins, n_time_bins)
    freqs_hz: np.ndarray  # 1D float array, length n_freq_bins
    times_seconds: np.ndarray  # 1D float array, length n_time_bins
    sample_rate: int


def _empty_spectrogram_data(sample_rate: int) -> SpectrogramData:
    return SpectrogramData(
        magnitudes_db=np.zeros((1, 1), dtype=np.float32),
        freqs_hz=np.zeros(1, dtype=np.float32),
        times_seconds=np.zeros(1, dtype=np.float32),
        sample_rate=sample_rate,
    )


def compute_spectrogram_data(
    audio_source: Union[Path, str, np.ndarray],
    sample_rate: int = 44100,
    n_fft: int = DEFAULT_N_FFT,
    hop_length: int = DEFAULT_HOP_LENGTH,
    max_freq_hz: float = DEFAULT_MAX_FREQ_HZ,
) -> SpectrogramData:
    """Compute a dB-scale magnitude spectrogram from an audio file or numpy array.

    Pure numpy/soundfile computation, no Qt dependency, so it's fully testable headlessly.
    Mirrors app.ui.waveform_player.compute_waveform_data's missing-file/empty-audio fallback
    behavior (returns a degenerate 1x1 SpectrogramData rather than raising).
    """
    if isinstance(audio_source, (Path, str)):
        path = Path(audio_source)
        if not path.is_file():
            logger.warning("Audio file not found for spectrogram computation: %s", path)
            return _empty_spectrogram_data(sample_rate)
        try:
            data, sr = sf.read(str(path), always_2d=True, dtype="float32")
        except Exception as exc:
            logger.error(
                "Failed to read audio file for spectrogram: %s (%s)", path, exc
            )
            return _empty_spectrogram_data(sample_rate)
    elif isinstance(audio_source, np.ndarray):
        sr = sample_rate
        data = audio_source[:, np.newaxis] if audio_source.ndim == 1 else audio_source
        data = data.astype(np.float32)
    else:
        raise TypeError(f"Unsupported audio_source type: {type(audio_source)}")

    mono = data.mean(axis=1)
    total_samples = len(mono)
    if total_samples < n_fft:
        return _empty_spectrogram_data(sr)

    window = np.hanning(n_fft)
    starts = list(range(0, total_samples - n_fft + 1, hop_length))
    if not starts:
        starts = [0]

    magnitudes = np.empty((n_fft // 2 + 1, len(starts)), dtype=np.float32)
    for i, start in enumerate(starts):
        frame = mono[start : start + n_fft] * window
        spectrum = np.abs(np.fft.rfft(frame))
        magnitudes[:, i] = spectrum

    freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    freq_mask = freqs_hz <= max_freq_hz
    magnitudes = magnitudes[freq_mask]
    freqs_hz = freqs_hz[freq_mask]
    times_seconds = np.array(starts, dtype=np.float64) / sr

    magnitudes_db = 20.0 * np.log10(magnitudes + 1e-9)
    magnitudes_db = np.clip(magnitudes_db, _DB_FLOOR, None)

    return SpectrogramData(
        magnitudes_db=magnitudes_db,
        freqs_hz=freqs_hz,
        times_seconds=times_seconds,
        sample_rate=sr,
    )


def render_spectrogram_pixmap(
    data: SpectrogramData, width_px: int, height_px: int, cmap: str = "magma"
) -> QPixmap:
    """Rasterize a SpectrogramData into a QPixmap via a headless (Agg) matplotlib Figure.

    No live FigureCanvasQTAgg embedding -- this is a one-shot render, re-invoked lazily by
    SpectrogramCanvas only when its data or size changes.
    """
    width_px = max(1, width_px)
    height_px = max(1, height_px)
    dpi = 100.0
    figure = Figure(figsize=(width_px / dpi, height_px / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(figure)
    axis = figure.add_axes((0.0, 0.0, 1.0, 1.0))
    axis.set_axis_off()

    extent = (
        float(data.times_seconds[0]) if data.times_seconds.size else 0.0,
        float(data.times_seconds[-1]) if data.times_seconds.size else 1.0,
        float(data.freqs_hz[0]) if data.freqs_hz.size else 0.0,
        float(data.freqs_hz[-1]) if data.freqs_hz.size else 1.0,
    )
    axis.imshow(
        data.magnitudes_db,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        extent=extent,
        interpolation="nearest",
    )

    canvas.draw()
    buffer = np.asarray(canvas.buffer_rgba())
    height, width, _ = buffer.shape
    image = QImage(buffer.tobytes(), width, height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(image.copy())


class SpectrogramCanvas(QWidget):
    """Internal canvas widget painting a lazily-rendered spectrogram QPixmap."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(100)
        self._data: Optional[SpectrogramData] = None
        self._cached_pixmap: Optional[QPixmap] = None
        self._cached_size: tuple[int, int] = (0, 0)

    def set_spectrogram_data(self, data: Optional[SpectrogramData]) -> None:
        self._data = data
        self._cached_pixmap = None
        self.update()

    def paintEvent(self, event: QPaintEvent) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#161722"))

        if self._data is None or self._data.magnitudes_db.size <= 1:
            painter.setPen(QColor("#5a5d72"))
            painter.setFont(QFont("Segoe UI", 11))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No Spectrogram Available"
            )
            return

        size = (self.width(), self.height())
        if self._cached_pixmap is None or self._cached_size != size:
            self._cached_pixmap = render_spectrogram_pixmap(
                self._data, size[0], size[1]
            )
            self._cached_size = size

        painter.drawPixmap(0, 0, self._cached_pixmap)

    def resizeEvent(self, event) -> None:
        self._cached_pixmap = None
        super().resizeEvent(event)


class SpectrogramCompareWidget(QWidget):
    """Hosts 'Original'/'Cleaned' spectrogram canvases side by side, mirroring the
    WaveformPlayerWidget pairing in app.ui.ab_compare_view.ABCompareView."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setObjectName("SpectrogramCompareWidget")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self._original_frame, self._original_canvas = self._make_column("Original")
        self._cleaned_frame, self._cleaned_canvas = self._make_column("Cleaned")

        layout.addWidget(self._original_frame)
        layout.addWidget(self._cleaned_frame)

    def _make_column(self, title: str) -> tuple[QFrame, SpectrogramCanvas]:
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; }"
        )
        column_layout = QVBoxLayout(frame)
        column_layout.setContentsMargins(12, 12, 12, 12)
        column_layout.setSpacing(8)

        title_label = QLabel(f"<b>{title} Spectrogram</b>")
        title_label.setStyleSheet("color: #7d6dfa; font-size: 13px;")
        column_layout.addWidget(title_label)

        canvas = SpectrogramCanvas()
        column_layout.addWidget(canvas)

        return frame, canvas

    def load_original(self, file_path: Union[Path, str]) -> None:
        self._original_canvas.set_spectrogram_data(
            compute_spectrogram_data(Path(file_path))
        )

    def load_cleaned(self, file_path: Union[Path, str]) -> None:
        self._cleaned_canvas.set_spectrogram_data(
            compute_spectrogram_data(Path(file_path))
        )

    def clear(self) -> None:
        self._original_canvas.set_spectrogram_data(None)
        self._cleaned_canvas.set_spectrogram_data(None)
