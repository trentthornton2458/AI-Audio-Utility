"""PySide6 Render History Panel for Music Mastery Enhancer.

Displays recent renders for the active track from cache/<track_id>/renders/
with accompanying metadata JSON settings and timestamps. Selecting an entry
emits renderSelected(Path) to load it into the Cleaned waveform player.
Includes a 'Clear Cache' button that invokes CacheManager.clear_track_cache(track_id)
after a confirmation dialog.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.cache import get_logger
from app.cache.cache_manager import CacheManager

logger = get_logger(__name__)


class RenderHistoryPanel(QWidget):
    """Widget displaying recent renders for a track with settings metadata and cache clearing."""

    renderSelected = Signal(Path)

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("RenderHistoryPanel")
        self._cache_manager = cache_manager or CacheManager()
        self._track_id: Optional[str] = None

        self._init_ui()
        self._wire_events()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Header Frame
        header_card = QFrame()
        header_card.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; padding: 8px 12px; }"
        )
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(8, 6, 8, 6)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("<h3>Render History</h3>")
        title.setStyleSheet("color: #ffffff; margin: 0px;")
        subtitle = QLabel(
            "<span style='color: #a0a5b5; font-size: 11px;'>Recent mastered renders for active track</span>"
        )
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_button.setStyleSheet(
            "QPushButton { background-color: #3b3e54; color: white; font-weight: bold; padding: 5px 12px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #4b4e69; }"
        )
        header_layout.addWidget(self._refresh_button)

        self._clear_cache_button = QPushButton("Clear Cache")
        self._clear_cache_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._clear_cache_button.setStyleSheet(
            "QPushButton { background-color: #d63031; color: white; font-weight: bold; padding: 5px 12px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #e17055; }"
            "QPushButton:disabled { background-color: #4a4b57; color: #8a8d9b; }"
        )
        self._clear_cache_button.setEnabled(False)
        header_layout.addWidget(self._clear_cache_button)

        main_layout.addWidget(header_card)

        # List Widget for Renders
        self._list_widget = QListWidget()
        self._list_widget.setStyleSheet(
            "QListWidget { background-color: #181922; border: 1px solid #2d2f3d; border-radius: 8px; padding: 4px; color: #e1e2e6; }"
            "QListWidget::item { background-color: #212330; border: 1px solid #2d2f3d; border-radius: 6px; margin: 4px 2px; padding: 8px; }"
            "QListWidget::item:hover { background-color: #2a2d3e; border-color: #6c5ce7; }"
            "QListWidget::item:selected { background-color: #34384e; border-color: #7d6dfa; color: #ffffff; }"
        )
        main_layout.addWidget(self._list_widget)

        # Status / Empty Label
        self._status_label = QLabel("No track loaded.")
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_label.setStyleSheet("color: #8a8d9b; font-size: 12px; font-style: italic;")
        main_layout.addWidget(self._status_label)

    def _wire_events(self) -> None:
        self._refresh_button.clicked.connect(self.on_refresh_clicked)
        self._clear_cache_button.clicked.connect(self.on_clear_cache_clicked)
        self._list_widget.itemSelectionChanged.connect(self.on_selection_changed)

    # --- Public API Methods ---

    def set_track_id(self, track_id: Optional[str]) -> None:
        """Set active track_id and refresh render history."""
        self._track_id = track_id
        self.refresh_history()

    def get_track_id(self) -> Optional[str]:
        """Return current track_id."""
        return self._track_id

    def clear_history(self) -> None:
        """Clear list widget items."""
        self._list_widget.clear()
        self._clear_cache_button.setEnabled(False)
        self._status_label.setText("No track loaded.")
        self._status_label.setVisible(True)

    def refresh_history(self) -> None:
        """Scan cache/<track_id>/renders/ for WAV files and metadata JSONs."""
        self._list_widget.clear()

        if not self._track_id:
            self._clear_cache_button.setEnabled(False)
            self._status_label.setText("No track loaded.")
            self._status_label.setVisible(True)
            return

        renders_dir = self._cache_manager.renders_dir(self._track_id)
        if not renders_dir.exists():
            self._clear_cache_button.setEnabled(False)
            self._status_label.setText("No renders found for this track.")
            self._status_label.setVisible(True)
            return

        wav_files = sorted(renders_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)

        if not wav_files:
            self._clear_cache_button.setEnabled(True)  # Track dir exists even if no renders yet
            self._status_label.setText("No renders found for this track.")
            self._status_label.setVisible(True)
            return

        self._status_label.setVisible(False)
        self._clear_cache_button.setEnabled(True)

        for wav_path in wav_files:
            meta_path = wav_path.with_suffix(".json")
            timestamp_str, summary_str = self._parse_render_metadata(wav_path, meta_path)

            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, wav_path)

            display_text = f"Render: {wav_path.name}\nTimestamp: {timestamp_str}\nSettings: {summary_str}"
            item.setText(display_text)
            self._list_widget.addItem(item)

    # --- Private Helpers ---

    def _parse_render_metadata(self, wav_path: Path, meta_path: Path) -> tuple[str, str]:
        """Extract formatted timestamp and settings summary from metadata JSON or fallback."""
        mtime = wav_path.stat().st_mtime
        fallback_time = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")

        if not meta_path.exists():
            return fallback_time, "Default / Custom Settings (No JSON metadata)"

        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            ts_raw = data.get("timestamp")
            if ts_raw:
                try:
                    dt = datetime.fromisoformat(ts_raw)
                    formatted_ts = dt.strftime("%Y-%m-%d %H:%M:%S")
                except ValueError:
                    formatted_ts = str(ts_raw)
            else:
                formatted_ts = fallback_time

            preset = data.get("preset", {})
            if preset and isinstance(preset, dict):
                v_clean = preset.get("vocal_clean_intensity", 1.0)
                notch = preset.get("notch_depth_db", 0.0)
                lufs = preset.get("lufs_target", -14.0)
                v_gain = preset.get("vocal_gain_db", 0.0)
                i_gain = preset.get("instrumental_gain_db", 0.0)
                summary = (
                    f"Vocal Clean: {int(v_clean * 100)}% | Notch: {notch:.1f}dB | "
                    f"LUFS Target: {lufs:.1f}dB | Vocal: {v_gain:+.1f}dB | Inst: {i_gain:+.1f}dB"
                )
            else:
                summary = "Custom Settings"

            return formatted_ts, summary

        except Exception as exc:
            logger.warning("Error reading metadata from %s: %s", meta_path, exc)
            return fallback_time, "Metadata error"

    # --- Event Slots ---

    @Slot()
    def on_refresh_clicked(self) -> None:
        self.refresh_history()

    @Slot()
    def on_selection_changed(self) -> None:
        selected_items = self._list_widget.selectedItems()
        if not selected_items:
            return

        item = selected_items[0]
        wav_path = item.data(Qt.ItemDataRole.UserRole)
        if wav_path:
            path_obj = Path(wav_path)
            logger.info("Render selected from history: %s", path_obj)
            self.renderSelected.emit(path_obj)

    @Slot()
    def on_clear_cache_clicked(self) -> None:
        if not self._track_id:
            return

        confirm = QMessageBox.question(
            self,
            "Confirm Clear Cache",
            f"Are you sure you want to delete all cached stems and renders for track ID:\n{self._track_id}\n\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if confirm == QMessageBox.StandardButton.Yes:
            logger.info("User confirmed clear cache for track_id: %s", self._track_id)
            self._cache_manager.clear_track_cache(self._track_id)
            self.refresh_history()
