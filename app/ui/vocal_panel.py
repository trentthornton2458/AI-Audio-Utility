"""PySide6 Vocal Control Panel for Music Mastery Enhancer.

Provides granular control over vocal stem cleaning and processing:
- Neural Denoising & Enhancement toggles + intensity sliders (0-100%)
- Vocal Clean Intensity slider (crossfading neural-only vs neural+DSP per vocal_chain.blend_vocal)
- Harshness Cut / 4kHz Notch Depth slider (-3dB to -6dB range)
- Vocal Gain (dB) spinbox (-24.0 to +24.0 dB)
- Preset dropdown & 'Save As...' dialog wired to app.core.presets
- Manual Apply/Render button emitting renderRequested(Settings)
"""

from __future__ import annotations

from typing import Optional

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
from app.core import presets
from app.models.preset import Preset
from app.models.settings import Settings

logger = get_logger(__name__)

NOTCH_SLIDER_MIN = 30  # Corresponds to 3.0 dB depth
NOTCH_SLIDER_MAX = 60  # Corresponds to 6.0 dB depth
NOTCH_SLIDER_DEFAULT = 45  # Corresponds to 4.5 dB depth


class VocalPanel(QWidget):
    """Control panel QWidget for configuring vocal stem cleaning, DSP parameters, and gain."""

    renderRequested = Signal(Settings)

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
        denoise_row = QVBoxLayout()
        denoise_header = QHBoxLayout()
        self._denoise_cb = QCheckBox("Enable Denoise")
        self._denoise_cb.setChecked(self._current_settings.vocal_denoise_enabled)
        self._denoise_cb.setStyleSheet("QCheckBox { color: #ffffff; font-weight: bold; }")
        self._denoise_cb.toggled.connect(self.on_denoise_toggled)

        self._denoise_val_label = QLabel(f"{int(self._current_settings.vocal_denoise_intensity * 100)}%")
        self._denoise_val_label.setStyleSheet("color: #55efc4; font-weight: bold;")
        denoise_header.addWidget(self._denoise_cb)
        denoise_header.addStretch()
        denoise_header.addWidget(self._denoise_val_label)
        denoise_row.addLayout(denoise_header)

        self._denoise_slider = QSlider(Qt.Orientation.Horizontal)
        self._denoise_slider.setRange(0, 100)
        self._denoise_slider.setValue(int(self._current_settings.vocal_denoise_intensity * 100))
        self._denoise_slider.setStyleSheet(self._slider_style())
        self._denoise_slider.valueChanged.connect(self.on_denoise_intensity_changed)
        denoise_row.addWidget(self._denoise_slider)
        neural_layout.addLayout(denoise_row)

        # Enhance Row
        enhance_row = QVBoxLayout()
        enhance_header = QHBoxLayout()
        self._enhance_cb = QCheckBox("Enable Harmonic Enhancement")
        self._enhance_cb.setChecked(self._current_settings.vocal_enhance_enabled)
        self._enhance_cb.setStyleSheet("QCheckBox { color: #ffffff; font-weight: bold; }")
        self._enhance_cb.toggled.connect(self.on_enhance_toggled)

        self._enhance_val_label = QLabel(f"{int(self._current_settings.vocal_enhance_intensity * 100)}%")
        self._enhance_val_label.setStyleSheet("color: #55efc4; font-weight: bold;")
        enhance_header.addWidget(self._enhance_cb)
        enhance_header.addStretch()
        enhance_header.addWidget(self._enhance_val_label)
        enhance_row.addLayout(enhance_header)

        self._enhance_slider = QSlider(Qt.Orientation.Horizontal)
        self._enhance_slider.setRange(0, 100)
        self._enhance_slider.setValue(int(self._current_settings.vocal_enhance_intensity * 100))
        self._enhance_slider.setStyleSheet(self._slider_style())
        self._enhance_slider.valueChanged.connect(self.on_enhance_intensity_changed)
        enhance_row.addWidget(self._enhance_slider)
        neural_layout.addLayout(enhance_row)

        main_layout.addWidget(neural_group)

        # Group 2: Vocal Clean & DSP Controls
        dsp_group = QGroupBox("DSP Polish & Blend Controls")
        dsp_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #7d6dfa; border: 1px solid #2d2f3d; border-radius: 8px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
        )
        dsp_layout = QVBoxLayout(dsp_group)
        dsp_layout.setSpacing(14)

        # Vocal Clean Intensity Slider (blend neural vs neural+DSP)
        clean_row = QVBoxLayout()
        clean_header = QHBoxLayout()
        clean_title = QLabel("<b>Vocal Clean Intensity</b>")
        clean_title.setStyleSheet("color: #ffffff;")
        clean_desc = QLabel(
            "<span style='color: #8a8d9b; font-size: 11px;'>(0% = Neural only | 100% = Full Neural + DSP Chain)</span>"
        )
        self._clean_val_label = QLabel(f"{int(self._current_settings.vocal_clean_intensity * 100)}%")
        self._clean_val_label.setStyleSheet("color: #55efc4; font-weight: bold;")

        clean_header.addWidget(clean_title)
        clean_header.addWidget(clean_desc)
        clean_header.addStretch()
        clean_header.addWidget(self._clean_val_label)
        clean_row.addLayout(clean_header)

        self._clean_slider = QSlider(Qt.Orientation.Horizontal)
        self._clean_slider.setRange(0, 100)
        self._clean_slider.setValue(int(self._current_settings.vocal_clean_intensity * 100))
        self._clean_slider.setStyleSheet(self._slider_style())
        self._clean_slider.valueChanged.connect(self.on_clean_intensity_changed)
        clean_row.addWidget(self._clean_slider)
        dsp_layout.addLayout(clean_row)

        # Harshness Cut / 4kHz Notch Depth Slider (-3dB to -6dB)
        notch_row = QVBoxLayout()
        notch_header = QHBoxLayout()
        notch_title = QLabel("<b>Harshness Cut (4kHz Notch Depth)</b>")
        notch_title.setStyleSheet("color: #ffffff;")
        notch_desc = QLabel("<span style='color: #8a8d9b; font-size: 11px;'>(Reduces pinched Suno frequencies)</span>")

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
        return (
            "QSlider::groove:horizontal { border: 1px solid #2d2f3d; height: 6px; background: #1a1b24; border-radius: 3px; }"
            f"QSlider::sub-page:horizontal {{ background: {accent_color}; border-radius: 3px; }}"
            "QSlider::handle:horizontal { background: #ffffff; border: 2px solid "
            f"{accent_color}; width: 16px; margin-top: -6px; margin-bottom: -6px; border-radius: 8px; }}"
            "QSlider::handle:horizontal:hover { background: #e1e2e6; cursor: pointer; }"
        )

    # --- Properties & State Accessors ---

    def get_settings(self) -> Settings:
        """Assemble and return the current panel values into a Settings object."""
        self._current_settings.vocal_denoise_enabled = self._denoise_cb.isChecked()
        self._current_settings.vocal_denoise_intensity = self._denoise_slider.value() / 100.0
        self._current_settings.vocal_enhance_enabled = self._enhance_cb.isChecked()
        self._current_settings.vocal_enhance_intensity = self._enhance_slider.value() / 100.0
        self._current_settings.vocal_clean_intensity = self._clean_slider.value() / 100.0
        self._current_settings.notch_depth_db = self._notch_slider.value() / 10.0
        self._current_settings.vocal_gain_db = self._gain_spinner.value()
        return Settings.from_preset(self._current_settings.to_preset())

    def set_settings(self, settings: Settings | Preset) -> None:
        """Apply all control values from a Settings or Preset object to the panel."""
        preset = settings if isinstance(settings, Preset) else settings.to_preset()
        self._current_settings = Settings.from_preset(preset)

        # Update UI controls without triggering unwanted state mutations
        self._denoise_cb.setChecked(preset.vocal_denoise_enabled)
        self._denoise_slider.setValue(int(round(preset.vocal_denoise_intensity * 100)))
        self._denoise_slider.setEnabled(preset.vocal_denoise_enabled)
        self._denoise_val_label.setText(f"{int(round(preset.vocal_denoise_intensity * 100))}%")

        self._enhance_cb.setChecked(preset.vocal_enhance_enabled)
        self._enhance_slider.setValue(int(round(preset.vocal_enhance_intensity * 100)))
        self._enhance_slider.setEnabled(preset.vocal_enhance_enabled)
        self._enhance_val_label.setText(f"{int(round(preset.vocal_enhance_intensity * 100))}%")

        self._clean_slider.setValue(int(round(preset.vocal_clean_intensity * 100)))
        self._clean_val_label.setText(f"{int(round(preset.vocal_clean_intensity * 100))}%")

        notch_val = preset.notch_depth_db
        slider_notch = int(round(notch_val * 10))
        slider_notch = max(NOTCH_SLIDER_MIN, min(NOTCH_SLIDER_MAX, slider_notch))
        self._notch_slider.setValue(slider_notch)
        self._notch_val_label.setText(f"-{notch_val:.1f} dB")

        self._gain_spinner.setValue(preset.vocal_gain_db)

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
            QMessageBox.warning(self, "Preset Load Error", f"Failed to load preset '{name}': {exc}")
        else:
            self.set_settings(loaded)
            logger.info("Loaded preset %r into VocalPanel", name)

    @Slot()
    def on_save_preset_clicked(self) -> None:
        name, ok = QInputDialog.getText(self, "Save Preset As", "Enter a name for the new preset:")
        if not ok or not name.strip():
            return

        name = name.strip()
        current_preset = self.get_settings().to_preset()

        try:
            presets.save_preset(name, current_preset, self._cache_manager)
        except Exception as exc:
            logger.error("Failed to save preset %r: %s", name, exc)
            QMessageBox.critical(self, "Save Preset Error", f"Failed to save preset '{name}': {exc}")
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
    def on_clean_intensity_changed(self, value: int) -> None:
        self._clean_val_label.setText(f"{value}%")

    @Slot(int)
    def on_notch_depth_changed(self, value: int) -> None:
        depth_db = value / 10.0
        self._notch_val_label.setText(f"-{depth_db:.1f} dB")

    @Slot(float)
    def on_gain_changed(self, value: float) -> None:
        pass

    @Slot()
    def on_apply_clicked(self) -> None:
        settings = self.get_settings()
        logger.info("Apply / Render clicked on VocalPanel with settings: %s", settings)
        self.renderRequested.emit(settings)
