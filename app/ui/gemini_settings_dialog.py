"""Small QDialog for viewing/updating application global preferences (Gemini API key and
custom reference stems override path) after setup. Backed by app.models.gemini_settings and
app.core.reference_assets."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QFileDialog, QFrame, QHBoxLayout, QLabel,
                               QLineEdit, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from app.cache import get_logger
from app.core.reference_assets import (get_custom_reference_override_path,
                                       get_reference_stems,
                                       set_custom_reference_override_path)
from app.models import gemini_settings

logger = get_logger(__name__)


class GeminiSettingsDialog(QDialog):
    """Modal dialog for viewing/updating stored Gemini API key and custom reference override path."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Preferences & Settings")
        self.setMinimumWidth(500)
        self._init_ui()
        self._refresh_ref_status()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(16, 16, 16, 16)

        # --- Section 1: Gemini API Key ---
        gemini_card = QFrame()
        gemini_card.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; padding: 12px; }"
        )
        gemini_layout = QVBoxLayout(gemini_card)
        gemini_layout.setSpacing(8)

        gemini_header = QLabel("<b>Gemini AI API Key</b>")
        gemini_header.setStyleSheet("color: #7d6dfa; font-size: 13px;")
        gemini_layout.addWidget(gemini_header)

        info_label = QLabel(
            "Used by the AI stem-analysis QA checkpoint to suggest denoise/enhance/EQ "
            "starting values right after stem separation."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #a0a5b5; font-size: 11px;")
        gemini_layout.addWidget(info_label)

        key_row = QHBoxLayout()
        key_row.setSpacing(8)
        key_row.addWidget(QLabel("API Key:"))
        self._key_edit = QLineEdit(gemini_settings.get_gemini_api_key() or "")
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("Paste your Gemini API key here")
        key_row.addWidget(self._key_edit, 1)
        gemini_layout.addLayout(key_row)

        self._show_key_cb = QCheckBox("Show key")
        self._show_key_cb.toggled.connect(self._on_show_toggled)
        gemini_layout.addWidget(self._show_key_cb)

        layout.addWidget(gemini_card)

        # --- Section 2: Reference Audio Stems Override ---
        ref_card = QFrame()
        ref_card.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; padding: 12px; }"
        )
        ref_layout = QVBoxLayout(ref_card)
        ref_layout.setSpacing(8)

        ref_header = QLabel("<b>Custom Reference Audio Stems Directory</b>")
        ref_header.setStyleSheet("color: #55efc4; font-size: 13px;")
        ref_layout.addWidget(ref_header)

        ref_info = QLabel(
            "Optional override directory containing custom reference stems "
            "(<code>male_dry.wav</code>, <code>male_tuned.wav</code>, "
            "<code>female_dry.wav</code>, <code>female_tuned.wav</code>) for the A/B comparison view."
        )
        ref_info.setWordWrap(True)
        ref_info.setStyleSheet("color: #a0a5b5; font-size: 11px;")
        ref_layout.addWidget(ref_info)

        curr_override = get_custom_reference_override_path()
        ref_row = QHBoxLayout()
        ref_row.setSpacing(8)
        self._ref_path_edit = QLineEdit(str(curr_override) if curr_override else "")
        self._ref_path_edit.setPlaceholderText(
            "Select folder containing reference .wav stems..."
        )
        self._ref_path_edit.textChanged.connect(self._on_ref_path_changed)
        ref_row.addWidget(self._ref_path_edit, 1)

        self._browse_ref_button = QPushButton("Browse...")
        self._browse_ref_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_ref_button.clicked.connect(self.on_browse_ref_folder_clicked)
        ref_row.addWidget(self._browse_ref_button)

        self._clear_ref_button = QPushButton("Clear")
        self._clear_ref_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_ref_button.clicked.connect(self.on_clear_ref_folder_clicked)
        ref_row.addWidget(self._clear_ref_button)

        ref_layout.addLayout(ref_row)

        self._ref_status_label = QLabel()
        self._ref_status_label.setWordWrap(True)
        self._ref_status_label.setStyleSheet("font-size: 11px;")
        ref_layout.addWidget(self._ref_status_label)

        layout.addWidget(ref_card)

        # Dialog buttons
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.on_save_clicked)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    def _refresh_ref_status(self) -> None:
        raw_path = self._ref_path_edit.text().strip()
        if not raw_path:
            self._ref_status_label.setText(
                "<span style='color: #a0a5b5;'><i>Using bundled factory reference stems (/assets/factory_references/).</i></span>"
            )
            return

        path = Path(raw_path)
        if not path.is_dir():
            self._ref_status_label.setText(
                "<span style='color: #ff7675;'><i>Selected directory does not exist.</i></span>"
            )
            return

        stems = get_reference_stems(path)
        found = [k for k, p in stems.items() if p is not None]
        if len(found) == 4:
            self._ref_status_label.setText(
                "<span style='color: #55efc4;'><b>✓ All 4 reference stems present</b> in custom directory.</span>"
            )
        elif found:
            self._ref_status_label.setText(
                f"<span style='color: #fdcb6e;'><b>⚠ Partial stem coverage ({len(found)}/4 found):</b> {', '.join(found)}</span>"
            )
        else:
            self._ref_status_label.setText(
                "<span style='color: #ff7675;'><b>✕ No matching reference stems found</b> in directory.</span>"
            )

    @Slot(bool)
    def _on_show_toggled(self, checked: bool) -> None:
        self._key_edit.setEchoMode(
            QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password
        )

    @Slot(str)
    def _on_ref_path_changed(self, text: str) -> None:
        self._refresh_ref_status()

    @Slot()
    def on_browse_ref_folder_clicked(self) -> None:
        current = self._ref_path_edit.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Custom Reference Audio Stems Directory",
            current or "",
        )
        if chosen:
            self._ref_path_edit.setText(chosen)

    @Slot()
    def on_clear_ref_folder_clicked(self) -> None:
        self._ref_path_edit.clear()

    @Slot()
    def on_save_clicked(self) -> None:
        api_key = self._key_edit.text().strip()
        if api_key:
            gemini_settings.set_gemini_api_key(api_key)
            logger.info("Updated Gemini API key from settings dialog")
        else:
            gemini_settings.clear_gemini_api_key()
            logger.info("Cleared Gemini API key from settings dialog")

        ref_dir_str = self._ref_path_edit.text().strip()
        if ref_dir_str:
            ref_path = Path(ref_dir_str)
            if not ref_path.is_dir():
                QMessageBox.warning(
                    self,
                    "Invalid Directory",
                    f"The custom reference directory does not exist:\n{ref_path}",
                )
                return
            set_custom_reference_override_path(ref_path)
            logger.info(
                "Saved custom reference override path from settings dialog: %s",
                ref_path,
            )
        else:
            set_custom_reference_override_path(None)
            logger.info("Cleared custom reference override path from settings dialog")

        self.accept()
