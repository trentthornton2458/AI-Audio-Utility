"""Small QDialog for viewing/updating the Gemini API key after first-run setup, without
re-running the full SetupWizard. Backed by the same app.models.gemini_settings storage the
wizard's GeminiApiKeyPage uses."""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Slot
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from app.cache import get_logger
from app.models import gemini_settings

logger = get_logger(__name__)


class GeminiSettingsDialog(QDialog):
    """Modal dialog for viewing/updating the stored Gemini API key."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Gemini API Key")
        self.setMinimumWidth(420)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        info_label = QLabel(
            "Used by the AI stem-analysis QA checkpoint to suggest denoise/enhance/EQ "
            "starting values right after stem separation."
        )
        info_label.setWordWrap(True)
        layout.addWidget(info_label)

        key_row = QHBoxLayout()
        key_row.addWidget(QLabel("<b>API Key:</b>"))
        self._key_edit = QLineEdit(gemini_settings.get_gemini_api_key() or "")
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self._key_edit.setPlaceholderText("Paste your Gemini API key here")
        key_row.addWidget(self._key_edit, 1)
        layout.addLayout(key_row)

        self._show_key_cb = QCheckBox("Show key")
        self._show_key_cb.toggled.connect(self._on_show_toggled)
        layout.addWidget(self._show_key_cb)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.on_save_clicked)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

    @Slot(bool)
    def _on_show_toggled(self, checked: bool) -> None:
        self._key_edit.setEchoMode(QLineEdit.EchoMode.Normal if checked else QLineEdit.EchoMode.Password)

    @Slot()
    def on_save_clicked(self) -> None:
        api_key = self._key_edit.text().strip()
        if api_key:
            gemini_settings.set_gemini_api_key(api_key)
            logger.info("Updated Gemini API key from settings dialog")
        else:
            gemini_settings.clear_gemini_api_key()
            logger.info("Cleared Gemini API key from settings dialog")
        self.accept()
