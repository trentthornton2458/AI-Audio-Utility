"""PySide6 Instrumental Control Panel for Music Mastery Enhancer.

Provides granular control over instrumental stem cleaning and processing:
- Overall enable/disable toggle for the optional instrumental cleanup chain
- Neural Denoising & Enhancement toggles + intensity sliders (0-100%)
- Instrumental EQ controls (Low-End Mud Cut cutoff, De-Hiss Shelf frequency, De-Hiss Shelf gain)
- Instrumental Gain (dB) spinbox (-24.0 to +24.0 dB)
- Preset dropdown & 'Save As...' dialog wired to app.core.presets
- Manual Apply/Render button emitting renderRequested(Settings) and settingsChanged(Settings)
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
from app.core import instrumental_chain, presets
from app.core.instrumental_chain import (
    DEHISS_GAIN_DB_DEFAULT,
    DEHISS_GAIN_DB_MAX,
    DEHISS_GAIN_DB_MIN,
    DEHISS_SHELF_HZ_DEFAULT,
    DEHISS_SHELF_HZ_MAX,
    DEHISS_SHELF_HZ_MIN,
    MUD_CUT_HZ_DEFAULT,
    MUD_CUT_HZ_MAX,
    MUD_CUT_HZ_MIN,
)
from app.models.preset import Preset
from app.models.settings import Settings
from app.ui.vocal_panel import IntensitySlider, make_slider_stylesheet

logger = get_logger(__name__)


class InstrumentalPanel(QWidget):
    """Control panel QWidget for configuring instrumental stem cleaning, DSP EQ, and gain."""

    settingsChanged = Signal(Settings)
    renderRequested = Signal(Settings)

    def __init__(
        self,
        cache_manager: Optional[CacheManager] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._cache_manager = cache_manager or CacheManager()
        self._current_settings = Settings()
        self._block_signals = False

        self.setObjectName("InstrumentalPanel")
        self._init_ui()
        self.load_presets_list()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Header / Title
        header_layout = QVBoxLayout()
        header_layout.setSpacing(4)
        title = QLabel("<h2>Instrumental Processing & Cleaning</h2>")
        title.setStyleSheet("color: #ffffff; margin-bottom: 0px;")
        desc = QLabel(
            "Clean Suno instrumental stems by removing background hiss and low-end rumble/mud "
            "using Resemble-Enhance neural cleaning and gentler Pedalboard EQ filters."
        )
        desc.setWordWrap(True)
        desc.setStyleSheet("color: #a0a5b5; font-size: 12px;")
        header_layout.addWidget(title)
        header_layout.addWidget(desc)
        main_layout.addLayout(header_layout)

        # Overall Enable Checkbox Box
        overall_box = QFrame()
        overall_box.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; padding: 10px 12px; }"
        )
        overall_layout = QHBoxLayout(overall_box)
        overall_layout.setContentsMargins(8, 6, 8, 6)

        self._overall_cleanup_cb = QCheckBox("Enable Instrumental Cleanup Chain")
        self._overall_cleanup_cb.setChecked(self._current_settings.instrumental_denoise_enabled)
        self._overall_cleanup_cb.setStyleSheet("QCheckBox { color: #55efc4; font-weight: bold; font-size: 13px; }")
        self._overall_cleanup_cb.toggled.connect(self.on_overall_cleanup_toggled)
        overall_layout.addWidget(self._overall_cleanup_cb)
        overall_layout.addStretch()

        main_layout.addWidget(overall_box)

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

        # Cleanup Container (contains Neural & EQ stages)
        self._cleanup_container = QWidget()
        cleanup_layout = QVBoxLayout(self._cleanup_container)
        cleanup_layout.setContentsMargins(0, 0, 0, 0)
        cleanup_layout.setSpacing(16)

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
            initial_value=int(round(self._current_settings.instrumental_denoise_intensity * 100)),
            checked=self._current_settings.instrumental_denoise_enabled,
        )
        self._denoise_cb = self._denoise_widget.checkbox
        self._denoise_slider = self._denoise_widget.slider
        self._denoise_val_label = self._denoise_widget.value_label
        self._denoise_widget.toggled.connect(self._emit_settings_changed)
        self._denoise_widget.valueChanged.connect(self._emit_settings_changed)
        neural_layout.addWidget(self._denoise_widget)

        # Enhance Row
        self._enhance_widget = IntensitySlider(
            "Enable Harmonic Enhancement",
            initial_value=int(round(self._current_settings.instrumental_enhance_intensity * 100)),
            checked=self._current_settings.instrumental_enhance_enabled,
        )
        self._enhance_cb = self._enhance_widget.checkbox
        self._enhance_slider = self._enhance_widget.slider
        self._enhance_val_label = self._enhance_widget.value_label
        self._enhance_widget.toggled.connect(self._emit_settings_changed)
        self._enhance_widget.valueChanged.connect(self._emit_settings_changed)
        neural_layout.addWidget(self._enhance_widget)

        cleanup_layout.addWidget(neural_group)

        # Group 2: EQ & Filter Controls
        eq_group = QGroupBox("Instrumental EQ Controls")
        eq_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #7d6dfa; border: 1px solid #2d2f3d; border-radius: 8px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
        )
        eq_layout = QVBoxLayout(eq_group)
        eq_layout.setSpacing(14)

        # Low-End Mud Cut Row
        mud_row = QVBoxLayout()
        mud_header = QHBoxLayout()
        mud_title = QLabel("<b>Low-End Mud Cut (Hz)</b>")
        mud_title.setStyleSheet("color: #ffffff;")
        mud_desc = QLabel("<span style='color: #8a8d9b; font-size: 11px;'>(Highpass cutoff trimming low-end rumble)</span>")

        mud_val = self._current_settings.instrumental_mud_cut_hz
        self._mud_cut_val_label = QLabel(f"{mud_val:.1f} Hz")
        self._mud_cut_val_label.setStyleSheet("color: #55efc4; font-weight: bold;")

        mud_header.addWidget(mud_title)
        mud_header.addWidget(mud_desc)
        mud_header.addStretch()
        mud_header.addWidget(self._mud_cut_val_label)
        mud_row.addLayout(mud_header)

        self._mud_cut_slider = QSlider(Qt.Orientation.Horizontal)
        self._mud_cut_slider.setRange(int(MUD_CUT_HZ_MIN), int(MUD_CUT_HZ_MAX))
        self._mud_cut_slider.setValue(int(mud_val))
        self._mud_cut_slider.setStyleSheet(make_slider_stylesheet())
        self._mud_cut_slider.valueChanged.connect(self.on_mud_cut_changed)
        mud_row.addWidget(self._mud_cut_slider)
        eq_layout.addLayout(mud_row)

        # De-Hiss Shelf Frequency Row
        shelf_row = QVBoxLayout()
        shelf_header = QHBoxLayout()
        shelf_title = QLabel("<b>De-Hiss Shelf Frequency (Hz)</b>")
        shelf_title.setStyleSheet("color: #ffffff;")
        shelf_desc = QLabel("<span style='color: #8a8d9b; font-size: 11px;'>(High-shelf corner frequency)</span>")

        shelf_val = self._current_settings.instrumental_dehiss_shelf_hz
        self._dehiss_shelf_val_label = QLabel(f"{int(shelf_val)} Hz")
        self._dehiss_shelf_val_label.setStyleSheet("color: #55efc4; font-weight: bold;")

        shelf_header.addWidget(shelf_title)
        shelf_header.addWidget(shelf_desc)
        shelf_header.addStretch()
        shelf_header.addWidget(self._dehiss_shelf_val_label)
        shelf_row.addLayout(shelf_header)

        self._dehiss_shelf_slider = QSlider(Qt.Orientation.Horizontal)
        self._dehiss_shelf_slider.setRange(int(DEHISS_SHELF_HZ_MIN), int(DEHISS_SHELF_HZ_MAX))
        self._dehiss_shelf_slider.setSingleStep(100)
        self._dehiss_shelf_slider.setValue(int(shelf_val))
        self._dehiss_shelf_slider.setStyleSheet(make_slider_stylesheet())
        self._dehiss_shelf_slider.valueChanged.connect(self.on_dehiss_shelf_changed)
        shelf_row.addWidget(self._dehiss_shelf_slider)
        eq_layout.addLayout(shelf_row)

        # De-Hiss Shelf Gain Row
        dehiss_gain_row = QVBoxLayout()
        dehiss_gain_header = QHBoxLayout()
        dehiss_gain_title = QLabel("<b>De-Hiss Shelf Gain (dB)</b>")
        dehiss_gain_title.setStyleSheet("color: #ffffff;")
        dehiss_gain_desc = QLabel("<span style='color: #8a8d9b; font-size: 11px;'>(High-shelf gain reduction)</span>")

        dehiss_gain_val = self._current_settings.instrumental_dehiss_gain_db
        self._dehiss_gain_val_label = QLabel(f"{dehiss_gain_val:.1f} dB")
        self._dehiss_gain_val_label.setStyleSheet("color: #ff7675; font-weight: bold;")

        dehiss_gain_header.addWidget(dehiss_gain_title)
        dehiss_gain_header.addWidget(dehiss_gain_desc)
        dehiss_gain_header.addStretch()
        dehiss_gain_header.addWidget(self._dehiss_gain_val_label)
        dehiss_gain_row.addLayout(dehiss_gain_header)

        self._dehiss_gain_slider = QSlider(Qt.Orientation.Horizontal)
        self._dehiss_gain_slider.setRange(int(DEHISS_GAIN_DB_MIN * 10), int(DEHISS_GAIN_DB_MAX * 10))
        self._dehiss_gain_slider.setValue(int(round(dehiss_gain_val * 10)))
        self._dehiss_gain_slider.setStyleSheet(make_slider_stylesheet(accent_color="#ff7675"))
        self._dehiss_gain_slider.valueChanged.connect(self.on_dehiss_gain_changed)
        dehiss_gain_row.addWidget(self._dehiss_gain_slider)
        eq_layout.addLayout(dehiss_gain_row)

        cleanup_layout.addWidget(eq_group)
        main_layout.addWidget(self._cleanup_container)

        # Group 3: Instrumental Gain Box (outside cleanup container so gain stays active)
        gain_group = QGroupBox("Mix Level")
        gain_group.setStyleSheet(
            "QGroupBox { font-weight: bold; color: #7d6dfa; border: 1px solid #2d2f3d; border-radius: 8px; margin-top: 10px; padding-top: 14px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; }"
        )
        gain_layout = QHBoxLayout(gain_group)
        gain_title = QLabel("<b>Instrumental Gain (dB):</b>")
        gain_title.setStyleSheet("color: #ffffff;")

        self._gain_spinner = QDoubleSpinBox()
        self._gain_spinner.setRange(-24.0, 24.0)
        self._gain_spinner.setSingleStep(0.5)
        self._gain_spinner.setValue(self._current_settings.instrumental_gain_db)
        self._gain_spinner.setSuffix(" dB")
        self._gain_spinner.setStyleSheet(
            "QDoubleSpinBox { background-color: #2b2d3e; color: #ffffff; border: 1px solid #3d3f52; border-radius: 4px; padding: 4px 8px; font-weight: bold; width: 100px; }"
        )
        self._gain_spinner.valueChanged.connect(self.on_gain_changed)

        gain_layout.addWidget(gain_title)
        gain_layout.addWidget(self._gain_spinner)
        gain_layout.addStretch()

        main_layout.addWidget(gain_group)

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

    # --- Properties & State Accessors ---

    def get_settings(self) -> Settings:
        """Assemble and return current panel values into a Settings object."""
        overall_enabled = self._overall_cleanup_cb.isChecked()

        if overall_enabled:
            self._current_settings.instrumental_denoise_enabled = self._denoise_widget.is_checked()
            self._current_settings.instrumental_denoise_intensity = self._denoise_widget.intensity()
            self._current_settings.instrumental_enhance_enabled = self._enhance_widget.is_checked()
            self._current_settings.instrumental_enhance_intensity = self._enhance_widget.intensity()
            self._current_settings.instrumental_mud_cut_hz = float(self._mud_cut_slider.value())
            self._current_settings.instrumental_dehiss_shelf_hz = float(self._dehiss_shelf_slider.value())
            self._current_settings.instrumental_dehiss_gain_db = self._dehiss_gain_slider.value() / 10.0
        else:
            self._current_settings.instrumental_denoise_enabled = False
            self._current_settings.instrumental_enhance_enabled = False
            self._current_settings.instrumental_dehiss_gain_db = 0.0

        self._current_settings.instrumental_gain_db = self._gain_spinner.value()

        return Settings.from_preset(self._current_settings.to_preset())

    def set_settings(self, settings: Settings | Preset) -> None:
        """Apply all control values from a Settings or Preset object to the panel."""
        preset = settings if isinstance(settings, Preset) else settings.to_preset()
        self._current_settings = Settings.from_preset(preset)
        self._block_signals = True

        try:
            overall_enabled = (
                preset.instrumental_denoise_enabled
                or preset.instrumental_enhance_enabled
                or (preset.instrumental_dehiss_gain_db != 0.0)
            )
            self._overall_cleanup_cb.setChecked(overall_enabled)
            self._cleanup_container.setEnabled(overall_enabled)

            self._denoise_widget.set_checked(preset.instrumental_denoise_enabled)
            self._denoise_widget.set_intensity(preset.instrumental_denoise_intensity)

            self._enhance_widget.set_checked(preset.instrumental_enhance_enabled)
            self._enhance_widget.set_intensity(preset.instrumental_enhance_intensity)

            mud_cut = max(MUD_CUT_HZ_MIN, min(MUD_CUT_HZ_MAX, preset.instrumental_mud_cut_hz))
            self._mud_cut_slider.setValue(int(round(mud_cut)))
            self._mud_cut_val_label.setText(f"{mud_cut:.1f} Hz")

            shelf_hz = max(DEHISS_SHELF_HZ_MIN, min(DEHISS_SHELF_HZ_MAX, preset.instrumental_dehiss_shelf_hz))
            self._dehiss_shelf_slider.setValue(int(round(shelf_hz)))
            self._dehiss_shelf_val_label.setText(f"{int(shelf_hz)} Hz")

            dehiss_gain = max(DEHISS_GAIN_DB_MIN, min(DEHISS_GAIN_DB_MAX, preset.instrumental_dehiss_gain_db))
            self._dehiss_gain_slider.setValue(int(round(dehiss_gain * 10)))
            self._dehiss_gain_val_label.setText(f"{dehiss_gain:.1f} dB")

            self._gain_spinner.setValue(preset.instrumental_gain_db)

        finally:
            self._block_signals = False

    # --- Preset Management ---

    def load_presets_list(self) -> None:
        """Refresh the preset dropdown list from app.core.presets."""
        old_block = self._block_signals
        self._block_signals = True
        try:
            self._preset_combo.clear()
            self._preset_combo.addItem("Default Preset", None)

            preset_names = presets.list_presets(self._cache_manager)
            for name in preset_names:
                self._preset_combo.addItem(name, name)
        finally:
            self._block_signals = old_block

    @Slot(int)
    def on_preset_changed(self, index: int) -> None:
        if self._block_signals or index < 0:
            return

        name = self._preset_combo.currentData()
        if name is None:
            default_preset = Preset()
            self.set_settings(default_preset)
            logger.info("Loaded Default Preset into InstrumentalPanel")
            self._emit_settings_changed()
            return

        try:
            loaded = presets.load_preset(name, self._cache_manager)
        except Exception as exc:
            logger.error("Failed to load preset %r: %s", name, exc)
            QMessageBox.warning(self, "Preset Load Error", f"Failed to load preset '{name}': {exc}")
        else:
            self.set_settings(loaded)
            logger.info("Loaded preset %r into InstrumentalPanel", name)
            self._emit_settings_changed()

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
            logger.info("Saved preset %r from InstrumentalPanel", name)
            self.load_presets_list()

            combo_index = self._preset_combo.findData(name)
            if combo_index != -1:
                self._preset_combo.setCurrentIndex(combo_index)

    # --- Control Event Slots ---

    @Slot(bool)
    def on_overall_cleanup_toggled(self, checked: bool) -> None:
        self._cleanup_container.setEnabled(checked)
        self._emit_settings_changed()

    @Slot(int)
    def on_mud_cut_changed(self, value: int) -> None:
        self._mud_cut_val_label.setText(f"{value:.1f} Hz")
        self._emit_settings_changed()

    @Slot(int)
    def on_dehiss_shelf_changed(self, value: int) -> None:
        self._dehiss_shelf_val_label.setText(f"{value} Hz")
        self._emit_settings_changed()

    @Slot(int)
    def on_dehiss_gain_changed(self, value: int) -> None:
        gain_db = value / 10.0
        self._dehiss_gain_val_label.setText(f"{gain_db:.1f} dB")
        self._emit_settings_changed()

    @Slot(float)
    def on_gain_changed(self, value: float) -> None:
        self._emit_settings_changed()

    def _emit_settings_changed(self) -> None:
        if self._block_signals:
            return
        settings = self.get_settings()
        self.settingsChanged.emit(settings)

    @Slot()
    def on_apply_clicked(self) -> None:
        settings = self.get_settings()
        logger.info("Apply / Render clicked on InstrumentalPanel with settings: %s", settings)
        self.settingsChanged.emit(settings)
        self.renderRequested.emit(settings)
