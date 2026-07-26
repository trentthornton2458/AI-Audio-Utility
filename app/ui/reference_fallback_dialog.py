"""PySide6 First-Run Fallback Modal for Missing Reference Audio Assets.

Triggered when default bundled reference stems (assets/factory_references/) are missing on launch.
Allows the user to select a replacement directory or upload custom reference stems, saving the
chosen path to the custom_reference_override_path setting. Persists a 'seen' state via QSettings
so the modal only triggers once per missing-state.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.cache import get_logger
from app.core.reference_assets import (
    REFERENCE_STEM_KEYS,
    get_custom_reference_override_path,
    get_reference_stems,
    has_seen_missing_reference_modal,
    is_reference_assets_missing,
    set_custom_reference_override_path,
    set_seen_missing_reference_modal,
)

logger = get_logger(__name__)


class ReferenceFallbackDialog(QDialog):
    """Modal dialog displayed when default factory reference stems are absent on app launch."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Missing Reference Audio Assets")
        self.setMinimumWidth(540)
        self._selected_path: Optional[Path] = get_custom_reference_override_path()
        self._init_ui()
        self._apply_theme()
        self._refresh_stem_status()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header / Title Box
        header_layout = QVBoxLayout()
        header_layout.setSpacing(6)

        title_label = QLabel("<h2>⚠️  Default Reference Vocal Stems Missing</h2>")
        title_label.setStyleSheet("color: #7d6dfa; margin: 0px;")
        header_layout.addWidget(title_label)

        desc_label = QLabel(
            "The A/B Audio Comparison feature uses bundled reference vocal stems "
            "(<code>male_dry.wav</code>, <code>male_tuned.wav</code>, "
            "<code>female_dry.wav</code>, <code>female_tuned.wav</code>) to compare mastered output "
            "against human vocal anchors. Default factory reference files were not found.<br><br>"
            "You may select a folder containing replacement reference stems or upload replacement "
            "files below, or skip to proceed without reference anchors."
        )
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #d0d3e0; font-size: 12px; line-height: 1.4;")
        header_layout.addWidget(desc_label)

        layout.addLayout(header_layout)

        # Folder Selection Card
        card = QFrame()
        card.setObjectName("FolderCard")
        card.setStyleSheet("QFrame#FolderCard { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; padding: 14px; }")
        card_layout = QVBoxLayout(card)
        card_layout.setSpacing(10)

        path_label = QLabel("<b>Custom Reference Stems Directory:</b>")
        card_layout.addWidget(path_label)

        path_input_row = QHBoxLayout()
        path_input_row.setSpacing(8)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Select directory containing reference .wav files...")
        if self._selected_path:
            self._path_edit.setText(str(self._selected_path))
        self._path_edit.textChanged.connect(self._on_path_text_changed)
        path_input_row.addWidget(self._path_edit, 1)

        self._browse_dir_button = QPushButton("Browse Folder...")
        self._browse_dir_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_dir_button.clicked.connect(self.on_browse_folder_clicked)
        path_input_row.addWidget(self._browse_dir_button)

        self._upload_files_button = QPushButton("Upload Files...")
        self._upload_files_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._upload_files_button.clicked.connect(self.on_upload_files_clicked)
        path_input_row.addWidget(self._upload_files_button)

        card_layout.addLayout(path_input_row)

        # Stem Status Summary Indicator
        self._status_summary_label = QLabel()
        self._status_summary_label.setWordWrap(True)
        card_layout.addWidget(self._status_summary_label)

        layout.addWidget(card)

        # Action Buttons Row
        button_row = QHBoxLayout()
        button_row.setSpacing(10)

        self._skip_button = QPushButton("Skip for Now")
        self._skip_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._skip_button.setStyleSheet(
            "QPushButton { background-color: #3b3e54; color: #d0d3e0; font-weight: bold; border-radius: 6px; padding: 8px 16px; border: none; }"
            "QPushButton:hover { background-color: #4b4e69; }"
        )
        self._skip_button.clicked.connect(self.on_skip_clicked)
        button_row.addWidget(self._skip_button)

        button_row.addStretch()

        self._save_button = QPushButton("Save & Use Custom Stems")
        self._save_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._save_button.setStyleSheet(
            "QPushButton { background-color: #6c5ce7; color: white; font-weight: bold; border-radius: 6px; padding: 8px 20px; border: none; }"
            "QPushButton:hover { background-color: #7d6dfa; }"
            "QPushButton:disabled { background-color: #4a4b57; color: #8a8d9b; }"
        )
        self._save_button.clicked.connect(self.on_save_clicked)
        button_row.addWidget(self._save_button)

        layout.addLayout(button_row)

    def _apply_theme(self) -> None:
        self.setStyleSheet(
            "QDialog { background-color: #12131a; color: #e1e2e6; font-family: 'Segoe UI', sans-serif; }"
            "QLabel { color: #e1e2e6; }"
            "QLineEdit { background-color: #15161e; border: 1px solid #3d3f4d; color: #ffffff; padding: 6px 10px; border-radius: 4px; }"
            "QLineEdit:focus { border: 1px solid #7d6dfa; }"
            "QPushButton { background-color: #2b2d3e; color: #ffffff; border: 1px solid #3d3f4d; border-radius: 4px; padding: 6px 12px; font-weight: bold; }"
            "QPushButton:hover { background-color: #36394e; border-color: #6c5ce7; }"
        )

    def _refresh_stem_status(self) -> None:
        raw_text = self._path_edit.text().strip()
        path = Path(raw_text) if raw_text else None
        if not path or not path.is_dir():
            self._status_summary_label.setText(
                "<span style='color: #a0a5b5;'><i>No valid folder selected.</i></span>"
            )
            return

        stems = get_reference_stems(path)
        found = [k for k, p in stems.items() if p is not None]
        missing = [k for k, p in stems.items() if p is None]

        if len(found) == 4:
            self._status_summary_label.setText(
                f"<span style='color: #55efc4;'><b>✓ All 4 stem files present:</b> {', '.join(found)}</span>"
            )
        elif found:
            self._status_summary_label.setText(
                f"<span style='color: #fdcb6e;'><b>⚠ Partial stem coverage ({len(found)}/4 found):</b> {', '.join(found)}<br>"
                f"<span style='color: #ff7675;'>Missing: {', '.join(missing)}</span></span>"
            )
        else:
            self._status_summary_label.setText(
                "<span style='color: #ff7675;'><b>✕ No matching stem files found</b> in selected directory "
                "(expected files: male_dry.wav, male_tuned.wav, female_dry.wav, female_tuned.wav).</span>"
            )

    @Slot(str)
    def _on_path_text_changed(self, text: str) -> None:
        raw = text.strip()
        self._selected_path = Path(raw) if raw else None
        self._refresh_stem_status()

    @Slot()
    def on_browse_folder_clicked(self) -> None:
        chosen = QFileDialog.getExistingDirectory(
            self,
            "Select Reference Stems Directory",
            str(self._selected_path) if self._selected_path else "",
        )
        if chosen:
            self._path_edit.setText(chosen)

    @Slot()
    def on_upload_files_clicked(self) -> None:
        """Allow user to select .wav files to copy into a target custom reference directory."""
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Replacement Reference Audio Files",
            "",
            "Audio Files (*.wav);;All Files (*)",
        )
        if not files:
            return

        target_dir = self._selected_path
        if not target_dir or not target_dir.is_dir():
            chosen_dir = QFileDialog.getExistingDirectory(
                self,
                "Select Destination Directory for Custom Stems",
            )
            if not chosen_dir:
                return
            target_dir = Path(chosen_dir)
            self._path_edit.setText(str(target_dir))

        copied_count = 0
        for src_str in files:
            src = Path(src_str)
            dest = target_dir / src.name
            try:
                shutil.copy2(src, dest)
                copied_count += 1
            except Exception as exc:
                logger.warning("Failed to copy custom stem %s to %s: %s", src, dest, exc)

        logger.info("Uploaded %d reference stem files to %s", copied_count, target_dir)
        self._refresh_stem_status()

    @Slot()
    def on_save_clicked(self) -> None:
        raw = self._path_edit.text().strip()
        if not raw:
            QMessageBox.warning(self, "Invalid Directory", "Please select a custom reference stems directory.")
            return

        path = Path(raw)
        if not path.is_dir():
            QMessageBox.warning(self, "Invalid Directory", f"The directory does not exist:\n{path}")
            return

        set_custom_reference_override_path(path)
        set_seen_missing_reference_modal(True)
        logger.info("Saved custom reference override path: %s", path)
        self.accept()

    @Slot()
    def on_skip_clicked(self) -> None:
        set_seen_missing_reference_modal(True)
        logger.info("User skipped missing reference assets modal")
        self.reject()

    def get_selected_path(self) -> Optional[Path]:
        return self._selected_path if (self._selected_path and self._selected_path.is_dir()) else None


def check_reference_assets_fallback(parent: Optional[QWidget] = None) -> Optional[Path]:
    """Check if reference assets are missing and launch fallback modal if not previously seen.

    Should be invoked during startup / main window initialization. Returns the active custom
    override path if configured, else None.
    """
    if is_reference_assets_missing() and not has_seen_missing_reference_modal():
        logger.info("Default reference assets are missing; opening first-run fallback modal")
        dialog = ReferenceFallbackDialog(parent=parent)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            override_path = dialog.get_selected_path()
            if override_path:
                set_custom_reference_override_path(override_path)
                set_seen_missing_reference_modal(True)
                return override_path
        set_seen_missing_reference_modal(True)

    return get_custom_reference_override_path()
