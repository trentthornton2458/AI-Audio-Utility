"""PySide6 A/B Compare View for Music Mastery Enhancer.

Hosts two WaveformPlayerWidget instances side by side labeled 'Original' and 'Cleaned',
allowing synchronous or independent scrubbing, playback, and A/B solo comparison
between the ingested original track and the latest mastered output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.ui.render_history_panel import RenderHistoryPanel
from app.ui.waveform_player import WaveformPlayerWidget

logger = get_logger(__name__)


class ABCompareView(QWidget):
    """Side-by-side A/B Comparison View hosting Original vs Cleaned audio players and Render History."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        super().__init__(parent)
        self._cache_manager = cache_manager or CacheManager()
        self._syncing_seek = False
        self._original_path: Optional[Path] = None
        self._cleaned_path: Optional[Path] = None

        self.setObjectName("ABCompareView")
        self._init_ui()
        self._wire_events()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Header & Master Controls Bar
        top_card = QFrame()
        top_card.setStyleSheet(
            "QFrame { background-color: #1e1f2b; border: 1px solid #2d2f3d; border-radius: 8px; padding: 8px 12px; }"
        )
        top_layout = QHBoxLayout(top_card)
        top_layout.setContentsMargins(8, 8, 8, 8)
        top_layout.setSpacing(16)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("<h2>A/B Audio Comparison</h2>")
        title.setStyleSheet("color: #ffffff; margin: 0px;")
        subtitle = QLabel("<span style='color: #a0a5b5; font-size: 11px;'>Compare original Suno track with mastered output</span>")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top_layout.addLayout(title_box)

        top_layout.addStretch()

        # Synchronized Playback Controls
        self._play_both_button = QPushButton("Play Both")
        self._play_both_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._play_both_button.setStyleSheet(
            "QPushButton { background-color: #00b894; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #00cec9; }"
        )
        self._play_both_button.clicked.connect(self.on_play_both_clicked)
        top_layout.addWidget(self._play_both_button)

        self._pause_both_button = QPushButton("Pause Both")
        self._pause_both_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pause_both_button.setStyleSheet(
            "QPushButton { background-color: #3b3e54; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #4b4e69; }"
        )
        self._pause_both_button.clicked.connect(self.on_pause_both_clicked)
        top_layout.addWidget(self._pause_both_button)

        self._sync_seek_cb = QCheckBox("Sync Playhead")
        self._sync_seek_cb.setChecked(True)
        self._sync_seek_cb.setStyleSheet("QCheckBox { color: #55efc4; font-weight: bold; }")
        top_layout.addWidget(self._sync_seek_cb)

        # Solo Selection (A/B Toggle)
        solo_frame = QFrame()
        solo_frame.setStyleSheet("QFrame { background-color: #15161e; border-radius: 4px; padding: 2px 6px; }")
        solo_layout = QHBoxLayout(solo_frame)
        solo_layout.setContentsMargins(4, 2, 4, 2)
        solo_layout.setSpacing(8)

        solo_label = QLabel("<b>Listen:</b>")
        solo_label.setStyleSheet("color: #a0a5b5; font-size: 11px;")
        solo_layout.addWidget(solo_label)

        self._btn_group = QButtonGroup(self)
        self._radio_both = QRadioButton("Both")
        self._radio_both.setChecked(True)
        self._radio_both.setStyleSheet("QRadioButton { color: #ffffff; }")

        self._radio_original = QRadioButton("Original (A)")
        self._radio_original.setStyleSheet("QRadioButton { color: #7d6dfa; font-weight: bold; }")

        self._radio_cleaned = QRadioButton("Cleaned (B)")
        self._radio_cleaned.setStyleSheet("QRadioButton { color: #55efc4; font-weight: bold; }")

        self._btn_group.addButton(self._radio_both, 0)
        self._btn_group.addButton(self._radio_original, 1)
        self._btn_group.addButton(self._radio_cleaned, 2)
        self._btn_group.idToggled.connect(self.on_solo_mode_changed)

        solo_layout.addWidget(self._radio_both)
        solo_layout.addWidget(self._radio_original)
        solo_layout.addWidget(self._radio_cleaned)
        top_layout.addWidget(solo_frame)

        main_layout.addWidget(top_card)

        # Players Side-by-Side Horizontal Layout
        players_layout = QHBoxLayout()
        players_layout.setSpacing(12)

        self._original_player = WaveformPlayerWidget(title="Original")
        self._cleaned_player = WaveformPlayerWidget(title="Cleaned")

        players_layout.addWidget(self._original_player)
        players_layout.addWidget(self._cleaned_player)

        main_layout.addLayout(players_layout)

        # Render History Panel
        self._render_history_panel = RenderHistoryPanel(cache_manager=self._cache_manager)
        main_layout.addWidget(self._render_history_panel)

    def _wire_events(self) -> None:
        self._original_player.seekRequested.connect(self.on_original_seek)
        self._cleaned_player.seekRequested.connect(self.on_cleaned_seek)
        self._render_history_panel.renderSelected.connect(self.load_cleaned)

    # --- Public API Methods ---

    def set_track_id(self, track_id: Optional[str]) -> None:
        """Set active track_id for render history panel."""
        self._render_history_panel.set_track_id(track_id)

    def refresh_history(self) -> None:
        """Refresh render history panel."""
        self._render_history_panel.refresh_history()

    def load_original(self, file_path: Union[Path, str]) -> None:
        """Load original ingested track into Original player."""
        path = Path(file_path)
        self._original_path = path
        self._original_player.load_file(path)
        logger.info("ABCompareView loaded original: %s", path)

    def load_cleaned(self, file_path: Union[Path, str]) -> None:
        """Load latest RenderJob output track into Cleaned player."""
        path = Path(file_path)
        self._cleaned_path = path
        self._cleaned_player.load_file(path)
        logger.info("ABCompareView loaded cleaned output: %s", path)

    def clear(self) -> None:
        """Clear both players."""
        self._original_path = None
        self._cleaned_path = None
        self._original_player.clear()
        self._cleaned_player.clear()

    def play_both(self) -> None:
        """Start synchronized playback on both players."""
        self._original_player.play()
        self._cleaned_player.play()

    def pause_both(self) -> None:
        """Pause playback on both players."""
        self._original_player.pause()
        self._cleaned_player.pause()

    def stop_both(self) -> None:
        """Stop playback on both players."""
        self._original_player.stop()
        self._cleaned_player.stop()

    def sync_seek(self, position_ms: int) -> None:
        """Seek both players to position_ms."""
        self._syncing_seek = True
        try:
            self._original_player.seek(position_ms)
            self._cleaned_player.seek(position_ms)
        finally:
            self._syncing_seek = False

    # --- Event Slots ---

    @Slot()
    def on_play_both_clicked(self) -> None:
        self.play_both()

    @Slot()
    def on_pause_both_clicked(self) -> None:
        self.pause_both()

    @Slot(int)
    def on_original_seek(self, position_ms: int) -> None:
        if self._sync_seek_cb.isChecked() and not self._syncing_seek:
            self._syncing_seek = True
            try:
                self._cleaned_player.seek(position_ms)
            finally:
                self._syncing_seek = False

    @Slot(int)
    def on_cleaned_seek(self, position_ms: int) -> None:
        if self._sync_seek_cb.isChecked() and not self._syncing_seek:
            self._syncing_seek = True
            try:
                self._original_player.seek(position_ms)
            finally:
                self._syncing_seek = False

    @Slot(int, bool)
    def on_solo_mode_changed(self, button_id: int, checked: bool) -> None:
        if not checked:
            return
        if button_id == 0:  # Both
            self._original_player.set_muted(False)
            self._cleaned_player.set_muted(False)
        elif button_id == 1:  # Original (A)
            self._original_player.set_muted(False)
            self._cleaned_player.set_muted(True)
        elif button_id == 2:  # Cleaned (B)
            self._original_player.set_muted(True)
            self._cleaned_player.set_muted(False)
