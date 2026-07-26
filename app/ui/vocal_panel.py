"""PySide6 Vocal Control Panel for Music Mastery Enhancer.

Provides granular control over vocal stem cleaning and processing:
- Neural Denoise toggle + intensity slider (0-100%, pre-DSP)
- Neural Enhance toggle + intensity slider (0-35%, hard-capped -- the last AI stage in the
  pipeline, blended back in via app.core.qa_gate's QA-gated capped residual blend, never a
  plain crossfade)
- Harshness Cut / 4kHz Notch Depth slider (-3dB to -6dB range)
- De-Esser Depth slider (-24dB to 0dB)
- Vocal Gain (dB) spinbox (-24.0 to +24.0 dB)
- Preset dropdown & 'Save As...' dialog wired to app.core.presets
- Manual Apply/Render button emitting renderRequested(Settings)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core import presets, qa_gate
from app.core.qa_gate import QAMetricResult
from app.models.preset import Preset
from app.models.settings import Settings

logger = get_logger(__name__)

NOTCH_SLIDER_MIN = 30  # Corresponds to 3.0 dB depth
NOTCH_SLIDER_MAX = 60  # Corresponds to 6.0 dB depth
NOTCH_SLIDER_DEFAULT = 45  # Corresponds to 4.5 dB depth


def make_slider_stylesheet(accent_color: str = "#6c5ce7") -> str:
    """Return standard dark theme stylesheet for QSlider with custom accent color."""
    return (
        "QSlider::groove:horizontal { border: 1px solid #2d2f3d; height: 6px; background: #1a1b24; border-radius: 3px; }"
        f"QSlider::sub-page:horizontal {{ background: {accent_color}; border-radius: 3px; }}"
        "QSlider::handle:horizontal { background: #ffffff; border: 2px solid "
        f"{accent_color}; width: 16px; margin-top: -6px; margin-bottom: -6px; border-radius: 8px; }}"
        "QSlider::handle:horizontal:hover { background: #e1e2e6; cursor: pointer; }"
    )


class IntensitySlider(QWidget):
    """Reusable control widget combining a toggle QCheckBox, percentage QLabel, and QSlider."""

    toggled = Signal(bool)
    valueChanged = Signal(int)

    def __init__(
        self,
        title: str,
        initial_value: int = 50,
        checked: bool = True,
        accent_color: str = "#6c5ce7",
        max_value: int = 100,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._accent_color = accent_color
        self._max_value = max_value
        self._init_ui(title, initial_value, checked)

    def _init_ui(self, title: str, initial_value: int, checked: bool) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)

        self.checkbox = QCheckBox(title)
        self.checkbox.setChecked(checked)
        self.checkbox.setStyleSheet("QCheckBox { color: #ffffff; font-weight: bold; }")

        self.value_label = QLabel(f"{initial_value}%")
        self.value_label.setStyleSheet("color: #55efc4; font-weight: bold;")

        header_layout.addWidget(self.checkbox)
        header_layout.addStretch()
        header_layout.addWidget(self.value_label)
        layout.addLayout(header_layout)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, self._max_value)
        self.slider.setValue(initial_value)
        self.slider.setEnabled(checked)
        self.slider.setStyleSheet(make_slider_stylesheet(self._accent_color))
        layout.addWidget(self.slider)

        self.checkbox.toggled.connect(self._on_toggled)
        self.slider.valueChanged.connect(self._on_value_changed)

    @Slot(bool)
    def _on_toggled(self, checked: bool) -> None:
        self.slider.setEnabled(checked)
        self.toggled.emit(checked)

    @Slot(int)
    def _on_value_changed(self, value: int) -> None:
        self.value_label.setText(f"{value}%")
        self.valueChanged.emit(value)

    def is_checked(self) -> bool:
        return self.checkbox.isChecked()

    def set_checked(self, checked: bool) -> None:
        self.checkbox.setChecked(checked)
        self.slider.setEnabled(checked)

    def intensity(self) -> float:
        return self.slider.value() / 100.0

    def set_intensity(self, val: float) -> None:
        int_val = int(round(val * 100))
        self.slider.setValue(int_val)
        self.value_label.setText(f"{int_val}%")


class VocalPanel(QWidget):
    """Control panel QWidget for configuring vocal stem cleaning, DSP parameters, and gain."""

    renderRequested = Signal(Settings)
    autoTuneRequested = Signal()

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cache_manager = cache_manager or CacheManager()
        self._current_settings = Settings()
        self._block_preset_signals = False

        self.setObjectName("VocalPanel")
        self._init_ui()
        self.load_presets_list()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Header / Title
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title = QLabel("<h2>Vocal Processing & Cleaning</h2>")
        title.setStyleSheet("color: #ffffff; margin-bottom: 0px;")
        desc = QLabel(
            "Tame Suno vocal artifacts, hiss, and high-frequency metallic resonances using "
            "Resemble-Enhance neural reconstruction and Pedalboard DSP filters."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a0a5b5; font-size: 12px;")
        header_layout.addWidget(title)
        header_layout.addWidget(desc)
        main_layout.addLayout(header_layout)

        # Preset Dropdown & Save As bar
        preset_box = QFrame()
        preset_box.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; padding: 8px 12px; }"
        )
        preset_layout = QHBoxLayout(preset_box)
        preset_layout.setContentsMargins(8, 6, 8, 6)
        preset_layout.setSpacing(12)

        preset_label = QLabel("<b>Preset:</b>")
        preset_label.setStyleSheet("color: #e1e2e6;")
        preset_layout.addWidget(preset_label)

        self._preset_combo = QComboBox()
        self._preset_combo.setMinimumWidth(200)
        self._preset_combo.setStyleSheet(
            "QComboBox { background-color: #2b2d3e; color: #ffffff; border: 1px solid #3d3f52; border-radius: 4px; padding: 4px 8px; }"
            "QComboBox::drop-down { border: none; }"
            "QComboBox QAbstractItemView { background-color: #232533; color: #ffffff; selection-background-color: #6c5ce7; }"
        )
        self._preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        preset_layout.addWidget(self._preset_combo)

        self._save_preset_button = QPushButton("Save As...")
        self._save_preset_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_preset_button.setStyleSheet(
            "QPushButton { background-color: #3b3e54; color: #ffffff; border: none; border-radius: 4px; padding: 6px 14px; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background-color: #4b4e69; }"
        )
        self._save_preset_button.clicked.connect(self.on_save_preset_clicked)
        preset_layout.addWidget(self._save_preset_button)
        preset_layout.addStretch()

        self._auto_tune_button = QPushButton("✨ Auto-Tune with Gemini")
        self._auto_tune_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._auto_tune_button.setStyleSheet(
            "QPushButton { background-color: #00cec9; color: #ffffff; border: none; border-radius: 4px; padding: 6px 14px; font-size: 12px; font-weight: bold; }"
            "QPushButton:hover { background-color: #81ecec; }"
        )
        self._auto_tune_button.clicked.connect(self.autoTuneRequested.emit)
        preset_layout.addWidget(self._auto_tune_button)

        main_layout.addWidget(preset_box)

        # Group 1: Neural Stage (Resemble-Enhance)
        neural_group = QGroupBox("Neural Cleanup (Resemble-Enhance)")
        neural_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #7d6dfa; border: 1px solid #2d2f3d; border-radius: 8px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
        )
        neural_layout = QVBoxLayout(neural_group)
        neural_layout.setSpacing(14)

        # Denoise Row
        self._denoise_widget = IntensitySlider(
            "Enable Denoise",
            initial_value=int(
                round(self._current_settings.vocal_denoise_intensity * 100)
            ),
            checked=self._current_settings.vocal_denoise_enabled,
        )
        self._denoise_cb = self._denoise_widget.checkbox
        self._denoise_slider = self._denoise_widget.slider
        self._denoise_val_label = self._denoise_widget.value_label
        self._denoise_cb.toggled.connect(self.on_denoise_toggled)
        self._denoise_slider.valueChanged.connect(self.on_denoise_intensity_changed)
        neural_layout.addWidget(self._denoise_widget)

        # Enhance Row (hard-capped at 35% -- see app.core.qa_gate.MAX_ENHANCE_GAIN)
        self._enhance_widget = IntensitySlider(
            "Enable Harmonic Enhancement",
            initial_value=int(
                round(self._current_settings.vocal_enhance_intensity * 100)
            ),
            checked=self._current_settings.vocal_enhance_enabled,
            max_value=35,
        )
        self._enhance_cb = self._enhance_widget.checkbox
        self._enhance_slider = self._enhance_widget.slider
        self._enhance_val_label = self._enhance_widget.value_label
        self._enhance_cb.toggled.connect(self.on_enhance_toggled)
        self._enhance_slider.valueChanged.connect(self.on_enhance_intensity_changed)
        neural_layout.addWidget(self._enhance_widget)

        main_layout.addWidget(neural_group)

        # Group 2: Vocal Clean & DSP Controls
        dsp_group = QGroupBox("DSP Polish & Blend Controls")
        dsp_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #7d6dfa; border: 1px solid #2d2f3d; border-radius: 8px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
        )
        dsp_layout = QVBoxLayout(dsp_group)
        dsp_layout.setSpacing(14)

        # Harshness Cut / 4kHz Notch Depth Slider (-3dB to -6dB)
        notch_row = QVBoxLayout()
        notch_header = QHBoxLayout()
        notch_title = QLabel("<b>Harshness Cut (4kHz Notch Depth)</b>")
        notch_title.setStyleSheet("color: #ffffff;")
        notch_desc = QLabel(
            "<span style='color: #8a8d9b; font-size: 11px;'>(Reduces pinched Suno frequencies)</span>"
        )

        # Map notch depth to display string, e.g. "-4.5 dB"
        notch_val = self._current_settings.notch_depth_db
        self._notch_val_label = QLabel(f"-{notch_val:.1f} dB")
        self._notch_val_label.setStyleSheet("color: #ff7675; font-weight: bold;")

        notch_header.addWidget(notch_title)
        notch_header.addWidget(notch_desc)
        notch_header.addStretch()
        notch_header.addWidget(self._notch_val_label)
        notch_row.addLayout(notch_header)

        self._notch_slider = QSlider(Qt.Orientation.Horizontal)
        self._notch_slider.setRange(NOTCH_SLIDER_MIN, NOTCH_SLIDER_MAX)
        self._notch_slider.setValue(int(round(notch_val * 10)))
        self._notch_slider.setStyleSheet(self._slider_style(accent_color="#ff7675"))
        self._notch_slider.valueChanged.connect(self.on_notch_depth_changed)
        notch_row.addWidget(self._notch_slider)
        dsp_layout.addLayout(notch_row)

        # De-Esser Depth Slider (-24dB to 0dB)
        deesser_row = QVBoxLayout()
        deesser_header = QHBoxLayout()
        deesser_title = QLabel("<b>De-Esser (Sibilance Reduction)</b>")
        deesser_title.setStyleSheet("color: #ffffff;")
        deesser_desc = QLabel(
            "<span style='color: #8a8d9b; font-size: 11px;'>(Reduces harsh 's' sounds)</span>"
        )

        deesser_val = self._current_settings.vocal_deesser_depth_db
        self._deesser_val_label = QLabel(f"{deesser_val:.1f} dB")
        self._deesser_val_label.setStyleSheet("color: #74b9ff; font-weight: bold;")

        deesser_header.addWidget(deesser_title)
        deesser_header.addWidget(deesser_desc)
        deesser_header.addStretch()
        deesser_header.addWidget(self._deesser_val_label)
        deesser_row.addLayout(deesser_header)

        self._deesser_slider = QSlider(Qt.Orientation.Horizontal)
        self._deesser_slider.setRange(-240, 0)
        self._deesser_slider.setValue(int(round(deesser_val * 10)))
        self._deesser_slider.setStyleSheet(self._slider_style(accent_color="#74b9ff"))
        self._deesser_slider.valueChanged.connect(self.on_deesser_depth_changed)
        deesser_row.addWidget(self._deesser_slider)
        dsp_layout.addLayout(deesser_row)

        # Vocal Gain Spinner (-24 to +24 dB)
        gain_row = QHBoxLayout()
        gain_title = QLabel("<b>Vocal Gain (dB):</b>")
        gain_title.setStyleSheet("color: #ffffff;")

        self._gain_spinner = QDoubleSpinBox()
        self._gain_spinner.setRange(-24.0, 24.0)
        self._gain_spinner.setSingleStep(0.5)
        self._gain_spinner.setValue(self._current_settings.vocal_gain_db)
        self._gain_spinner.setSuffix(" dB")
        self._gain_spinner.setStyleSheet(
            "QDoubleSpinBox { background-color: #2b2d3e; color: #ffffff; border: 1px solid #3d3f52; border-radius: 4px; padding: 4px 8px; font-weight: bold; width: 100px; }"
        )
        self._gain_spinner.valueChanged.connect(self.on_gain_changed)

        gain_row.addWidget(gain_title)
        gain_row.addWidget(self._gain_spinner)
        gain_row.addStretch()
        dsp_layout.addLayout(gain_row)

        main_layout.addWidget(dsp_group)

        # QA Caution Badge & Expandable Details Panel (warn, never block)
        qa_layout = QVBoxLayout()
        qa_layout.setContentsMargins(0, 0, 0, 0)
        qa_layout.setSpacing(6)

        self._qa_badge = QPushButton("⚠️ QA Caution")
        self._qa_badge.setCursor(Qt.CursorShape.PointingHandCursor)
        self._qa_badge.setStyleSheet(
            "QPushButton { background-color: #3a2e19; color: #fdcb6e; border: 1px solid #e1b12c; border-radius: 6px; padding: 8px 14px; font-weight: bold; font-size: 12px; text-align: left; }"
            "QPushButton:hover { background-color: #4d3d20; border-color: #f1c40f; }"
        )
        self._qa_badge.setVisible(False)
        self._qa_badge.clicked.connect(self._toggle_qa_details)
        qa_layout.addWidget(self._qa_badge)

        self._qa_details_panel = QFrame()
        self._qa_details_panel.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #e1b12c; border-radius: 8px; padding: 10px; }"
        )
        qa_details_inner = QVBoxLayout(self._qa_details_panel)
        qa_details_inner.setContentsMargins(8, 8, 8, 8)
        qa_details_inner.setSpacing(6)

        self._qa_details_label = QLabel()
        self._qa_details_label.setWordWrap(True)
        self._qa_details_label.setStyleSheet("color: #e1e2e6; font-size: 12px;")
        qa_details_inner.addWidget(self._qa_details_label)

        self._qa_details_panel.setVisible(False)
        qa_layout.addWidget(self._qa_details_panel)

        main_layout.addLayout(qa_layout)

        # Apply / Render Action Button
        action_layout = QHBoxLayout()
        action_layout.setContentsMargins(0, 8, 0, 0)

        self._apply_button = QPushButton("Apply Settings & Render")
        self._apply_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._apply_button.setStyleSheet(
            "QPushButton { background-color: #00b894; color: #ffffff; font-weight: bold; font-size: 14px; padding: 10px 20px; border: none; border-radius: 6px; }"
            "QPushButton:hover { background-color: #00cec9; }"
            "QPushButton:pressed { background-color: #009788; }"
            "QPushButton:disabled { background-color: #3d3f52; color: #8a8d9b; }"
        )
        self._apply_button.clicked.connect(self.on_apply_clicked)
        action_layout.addWidget(self._apply_button)

        main_layout.addLayout(action_layout)
        main_layout.addStretch()

    def _slider_style(self, accent_color: str = "#6c5ce7") -> str:
        return make_slider_stylesheet(accent_color)

    # --- Properties & State Accessors ---

    def get_settings(self) -> Settings:
        """Assemble and return the current panel values into a Settings object."""
        self._current_settings.vocal_denoise_enabled = self._denoise_cb.isChecked()
        self._current_settings.vocal_denoise_intensity = (
            self._denoise_slider.value() / 100.0
        )
        self._current_settings.vocal_enhance_enabled = self._enhance_cb.isChecked()
        self._current_settings.vocal_enhance_intensity = (
            self._enhance_slider.value() / 100.0
        )
        self._current_settings.notch_depth_db = float(self._notch_slider.value()) / 10.0
        self._current_settings.vocal_deesser_depth_db = (
            float(self._deesser_slider.value()) / 10.0
        )
        self._current_settings.vocal_gain_db = self._gain_spinner.value()
        return Settings.from_preset(self._current_settings.to_preset())

    def set_settings(self, settings: Settings | Preset) -> None:
        """Apply all control values from a Settings or Preset object to the panel."""
        preset = settings if isinstance(settings, Preset) else settings.to_preset()
        self._current_settings = Settings.from_preset(preset)

        # Guard against on_notch_depth_changed/on_deesser_depth_changed resetting the preset
        # combo back to "Default Preset" (and recursively reloading it) as a side effect of the
        # setValue() calls below programmatically changing those sliders.
        self._block_preset_signals = True
        try:
            # Update UI controls without triggering unwanted state mutations
            self._denoise_cb.setChecked(preset.vocal_denoise_enabled)
            self._denoise_slider.setValue(
                int(round(preset.vocal_denoise_intensity * 100))
            )
            self._denoise_slider.setEnabled(preset.vocal_denoise_enabled)
            self._denoise_val_label.setText(
                f"{int(round(preset.vocal_denoise_intensity * 100))}%"
            )

            self._enhance_cb.setChecked(preset.vocal_enhance_enabled)
            self._enhance_slider.setValue(
                int(round(preset.vocal_enhance_intensity * 100))
            )
            self._enhance_slider.setEnabled(preset.vocal_enhance_enabled)
            self._enhance_val_label.setText(
                f"{int(round(preset.vocal_enhance_intensity * 100))}%"
            )

            notch_val = preset.notch_depth_db
            slider_notch = int(round(notch_val * 10))
            slider_notch = max(NOTCH_SLIDER_MIN, min(NOTCH_SLIDER_MAX, slider_notch))
            self._notch_slider.setValue(slider_notch)
            self._notch_val_label.setText(f"-{notch_val:.1f} dB")

            deesser_val = preset.vocal_deesser_depth_db
            self._deesser_slider.setValue(int(round(deesser_val * 10)))
            self._deesser_val_label.setText(f"{deesser_val:.1f} dB")

            self._gain_spinner.setValue(preset.vocal_gain_db)
        finally:
            self._block_preset_signals = False

    # --- Preset Management ---

    def load_presets_list(self) -> None:
        """Refresh the preset dropdown list from app.core.presets."""
        self._block_preset_signals = True
        try:
            self._preset_combo.clear()
            self._preset_combo.addItem("Default Preset", None)

            preset_names = presets.list_presets(self._cache_manager)
            for name in preset_names:
                self._preset_combo.addItem(name, name)
        finally:
            self._block_preset_signals = False

    @Slot(int)
    def on_preset_changed(self, index: int) -> None:
        if self._block_preset_signals or index < 0:
            return

        name = self._preset_combo.currentData()
        if name is None:
            # Revert to default Preset
            default_preset = Preset()
            self.set_settings(default_preset)
            logger.info("Loaded Default Preset into VocalPanel")
            return

        try:
            loaded = presets.load_preset(name, self._cache_manager)
        except Exception as exc:
            logger.error("Failed to load preset %r: %s", name, exc)
            QMessageBox.warning(
                self, "Preset Load Error", f"Failed to load preset '{name}': {exc}"
            )
        else:
            self.set_settings(loaded)
            logger.info("Loaded preset %r into VocalPanel", name)

    @Slot()
    def on_save_preset_clicked(self) -> None:
        name, ok = QInputDialog.getText(
            self, "Save Preset As", "Enter a name for the new preset:"
        )
        if not ok or not name.strip():
            return

        name = name.strip()
        current_preset = self.get_settings().to_preset()

        try:
            presets.save_preset(name, current_preset, self._cache_manager)
        except Exception as exc:
            logger.error("Failed to save preset %r: %s", name, exc)
            QMessageBox.critical(
                self, "Save Preset Error", f"Failed to save preset '{name}': {exc}"
            )
        else:
            logger.info("Saved preset %r from VocalPanel", name)
            self.load_presets_list()

            # Select saved preset in combo box
            combo_index = self._preset_combo.findData(name)
            if combo_index != -1:
                self._preset_combo.setCurrentIndex(combo_index)

    # --- Control Event Slots ---

    @Slot(bool)
    def on_denoise_toggled(self, checked: bool) -> None:
        self._denoise_slider.setEnabled(checked)

    @Slot(int)
    def on_denoise_intensity_changed(self, value: int) -> None:
        self._denoise_val_label.setText(f"{value}%")

    @Slot(bool)
    def on_enhance_toggled(self, checked: bool) -> None:
        self._enhance_slider.setEnabled(checked)

    @Slot(int)
    def on_enhance_intensity_changed(self, value: int) -> None:
        self._enhance_val_label.setText(f"{value}%")

    @Slot(int)
    def on_notch_depth_changed(self, value_x10: int) -> None:
        real_val = float(value_x10) / 10.0
        self._notch_val_label.setText(f"-{real_val:.1f} dB")
        if not self._block_preset_signals:
            self._preset_combo.setCurrentIndex(0)

    @Slot(int)
    def on_deesser_depth_changed(self, value_x10: int) -> None:
        real_val = float(value_x10) / 10.0
        self._deesser_val_label.setText(f"{real_val:.1f} dB")
        if not self._block_preset_signals:
            self._preset_combo.setCurrentIndex(0)

    @Slot(float)
    def on_gain_changed(self, value: float) -> None:
        pass

    @Slot()
    def on_apply_clicked(self) -> None:
        self.clear_qa_warning()
        settings = self.get_settings()
        logger.info("Apply / Render clicked on VocalPanel with settings: %s", settings)
        self.renderRequested.emit(settings)

    # --- QA Metric Surfacing & Caution Badge ---

    @Slot()
    def _toggle_qa_details(self) -> None:
        """Toggle visibility of the expandable QA metric details panel."""
        self._qa_details_panel.setVisible(not self._qa_details_panel.isVisible())

    def update_qa_from_file(
        self, file_path: Path | str, silence_threshold_db: float = -30.0
    ) -> dict[str, QAMetricResult]:
        """Run app.core.qa_gate warn-only metrics on a rendered audio file and surface caution badge if flagged."""
        try:
            audio, samplerate = sf.read(str(file_path), always_2d=True, dtype="float64")
            return self.update_qa_metrics(
                audio, samplerate, silence_threshold_db=silence_threshold_db
            )
        except Exception as exc:
            logger.warning("Failed to evaluate QA metrics for %s: %s", file_path, exc)
            self.clear_qa_warning()
            return {}

    def update_qa_metrics(
        self,
        audio: np.ndarray,
        samplerate: int,
        silence_threshold_db: float = -30.0,
    ) -> dict[str, QAMetricResult]:
        """Evaluate the three warn-only QA metrics on audio array and update caution badge state."""
        pitch_res = qa_gate.measure_pitch_variance(audio, samplerate)
        hf_res = qa_gate.measure_high_frequency_energy(
            audio, samplerate, silence_threshold_db=silence_threshold_db
        )
        crest_res = qa_gate.measure_crest_factor(audio)

        results = {
            "pitch_variance": pitch_res,
            "hf_energy": hf_res,
            "crest_factor": crest_res,
        }
        self.set_qa_metric_results(results)
        return results

    def set_qa_metric_results(self, results: dict[str, QAMetricResult]) -> None:
        """Set QA metric results directly and update the caution badge state."""
        self._last_qa_results = results
        warning_items = [(name, res) for name, res in results.items() if res.warning]

        if not warning_items:
            self.clear_qa_warning()
            return

        count = len(warning_items)
        plural = "s" if count > 1 else ""
        self._qa_badge.setText(
            f"⚠️ QA Caution: {count} signal quality metric{plural} flagged (click for details)"
        )

        tooltip_lines = ["⚠️ Audio Quality Caution (Warn Only):"]
        details_lines = [
            "<b>⚠️ Audio Quality Caution (Informational Only):</b><br>"
            "<span style='color: #a0a5b5;'>The following signal quality metric(s) triggered warning flags after render:</span><br>"
        ]

        for name, res in results.items():
            if name == "pitch_variance":
                label = "Pitch Variance"
                val_str = f"{res.value:.1f} cents"
                thresh_str = f"≤ {qa_gate.PITCH_VARIANCE_WARN_CENTS_MAX:.1f} cents"
                desc = "Pitch is flat or hard-quantized rather than having natural micro-drift."
            elif name == "hf_energy":
                label = "HF / Breath Energy"
                val_str = f"{res.value * 100:.2f}% ({res.value:.4f})"
                thresh_str = f"≤ {qa_gate.HF_ENERGY_WARN_RATIO_MAX * 100:.1f}% ({qa_gate.HF_ENERGY_WARN_RATIO_MAX:.3f})"
                desc = "High-frequency breath detail stripped in quiet passages."
            elif name == "crest_factor":
                label = "Crest Factor"
                val_str = f"{res.value:.1f} dB"
                thresh_str = f"≤ {qa_gate.CREST_FACTOR_WARN_DB_MIN:.1f} dB"
                desc = "Signal peak-to-RMS is low (over-compressed or brickwalled)."
            else:
                label = name
                val_str = str(res.value)
                thresh_str = ""
                desc = res.reason

            status = "⚠️ WARN" if res.warning else "✓ OK"
            status_color = "#ff7675" if res.warning else "#55efc4"

            tooltip_lines.append(
                f"• [{status}] {label}: {val_str} (warn limit: {thresh_str})"
            )
            if res.warning:
                tooltip_lines.append(f"  Reason: {desc}")

            details_lines.append(
                f"<div style='margin-top: 4px;'>"
                f"<b style='color: {status_color};'>[{status}] {label}:</b> {val_str} "
                f"<span style='color: #8a8d9b;'>(threshold: {thresh_str})</span><br>"
                f"<span style='color: #d0d3e0; font-size: 11px;'>{desc}</span>"
                f"</div>"
            )

        self._qa_badge.setToolTip("\n".join(tooltip_lines))
        self._qa_details_label.setText("".join(details_lines))
        self._qa_badge.setVisible(True)

    def clear_qa_warning(self) -> None:
        """Hide caution badge and details panel."""
        self._last_qa_results = {}
        self._qa_badge.setVisible(False)
        self._qa_details_panel.setVisible(False)
