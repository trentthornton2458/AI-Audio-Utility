"""PySide6 A/B Compare View for Music Mastery Enhancer.

Hosts three WaveformPlayerWidget instances side by side supporting a blind comparison
between Raw Input, DSP-Humanized output (Milestone 2 humanizer pass), and a Ground Truth
Reference vocal stem (factory bundled or custom override), with synchronized scrubbing,
position drift correction, and render history.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Union

from PySide6.QtCore import Qt, Slot
from PySide6.QtWidgets import (QButtonGroup, QCheckBox, QFrame, QHBoxLayout,
                               QLabel, QPushButton, QRadioButton, QVBoxLayout,
                               QWidget)

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core.reference_assets import select_reference_stem
from app.ui.render_history_panel import RenderHistoryPanel
from app.ui.spectrogram_view import SpectrogramCompareWidget
from app.ui.waveform_player import WaveformPlayerWidget

logger = get_logger(__name__)


class ABCompareView(QWidget):
    """Side-by-side blind comparison view for Raw Input, DSP-Humanized output, and Reference Audio."""

    def __init__(
        self,
        parent: Optional[QWidget] = None,
        cache_manager: Optional[CacheManager] = None,
    ) -> None:
        super().__init__(parent)
        self._cache_manager = cache_manager or CacheManager()
        self._syncing_seek = False
        self._syncing_state = False

        self._raw_path: Optional[Path] = None
        self._humanized_path: Optional[Path] = None
        self._reference_path: Optional[Path] = None
        self._active_reference_key: Optional[str] = None
        self._vocal_metadata: Optional[dict] = None

        # Blind mode state
        self._is_revealed: bool = False
        self._slots: list[str] = ["raw", "humanized", "reference"]

        self.setObjectName("ABCompareView")
        self._init_ui()
        self._wire_events()
        self.update_blind_labels()

    # --- Backward-compatibility properties ---

    @property
    def _original_player(self) -> WaveformPlayerWidget:
        return self._get_player_for_source("raw")

    @property
    def _cleaned_player(self) -> WaveformPlayerWidget:
        return self._get_player_for_source("humanized")

    @property
    def _reference_player(self) -> WaveformPlayerWidget:
        return self._get_player_for_source("reference")

    @property
    def _original_path(self) -> Optional[Path]:
        return self._raw_path

    @_original_path.setter
    def _original_path(self, val: Optional[Path]) -> None:
        self._raw_path = val

    @property
    def _cleaned_path(self) -> Optional[Path]:
        return self._humanized_path

    @_cleaned_path.setter
    def _cleaned_path(self, val: Optional[Path]) -> None:
        self._humanized_path = val

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
        top_layout.setSpacing(12)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title = QLabel("<h2>A/B Audio Comparison</h2>")
        title.setStyleSheet("color: #ffffff; margin: 0px;")
        subtitle = QLabel(
            "<span style='color: #a0a5b5; font-size: 11px;'>Blind test: Raw Input vs DSP-Humanized vs Ground Truth Reference</span>"
        )
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top_layout.addLayout(title_box)

        top_layout.addStretch()

        # Synchronized Playback Controls
        self._play_both_button = QPushButton("Play All")
        self._play_both_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._play_both_button.setToolTip("Start synchronized playback of all players")
        self._play_both_button.setAccessibleName("Play All")
        self._play_both_button.setAccessibleDescription(
            "Starts playback on all Option A, B, and C players simultaneously."
        )
        self._play_both_button.setStyleSheet(
            "QPushButton { background-color: #00b894; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #00cec9; }"
        )
        self._play_both_button.clicked.connect(self.on_play_both_clicked)
        top_layout.addWidget(self._play_both_button)

        self._pause_both_button = QPushButton("Pause All")
        self._pause_both_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._pause_both_button.setToolTip("Pause playback of all players")
        self._pause_both_button.setAccessibleName("Pause All")
        self._pause_both_button.setAccessibleDescription(
            "Pauses playback on all Option A, B, and C players simultaneously."
        )
        self._pause_both_button.setStyleSheet(
            "QPushButton { background-color: #3b3e54; color: white; font-weight: bold; padding: 6px 14px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #4b4e69; }"
        )
        self._pause_both_button.clicked.connect(self.on_pause_both_clicked)
        top_layout.addWidget(self._pause_both_button)

        self._sync_seek_cb = QCheckBox("Sync Playhead")
        self._sync_seek_cb.setChecked(True)
        self._sync_seek_cb.setToolTip(
            "Synchronize playhead positioning across all players"
        )
        self._sync_seek_cb.setAccessibleName("Sync Playhead")
        self._sync_seek_cb.setAccessibleDescription(
            "When checked, seeking or moving any playhead will automatically seek all players to the same position."
        )
        self._sync_seek_cb.setStyleSheet(
            "QCheckBox { color: #55efc4; font-weight: bold; }"
        )
        top_layout.addWidget(self._sync_seek_cb)

        # Blind Mode Toggle & Reshuffle Buttons
        self._reveal_button = QPushButton("👁 Reveal Identities")
        self._reveal_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._reveal_button.setToolTip(
            "Toggle between blind mode and revealing option names"
        )
        self._reveal_button.setAccessibleName("Reveal Identities")
        self._reveal_button.setAccessibleDescription(
            "Toggles blind mode to reveal or hide the actual names (Raw Input, DSP-Humanized, Ground Truth Reference) of the options."
        )
        self._reveal_button.setStyleSheet(
            "QPushButton { background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #7d6dfa; }"
        )
        self._reveal_button.clicked.connect(self.toggle_reveal)
        top_layout.addWidget(self._reveal_button)

        self._shuffle_button = QPushButton("🔀 Reshuffle")
        self._shuffle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._shuffle_button.setToolTip(
            "Randomly shuffle option names and reset comparison to blind mode"
        )
        self._shuffle_button.setAccessibleName("Reshuffle Options")
        self._shuffle_button.setAccessibleDescription(
            "Randomly reshuffles which file is assigned to Option A, B, and C, resetting the comparison to blind mode."
        )
        self._shuffle_button.setStyleSheet(
            "QPushButton { background-color: #2d2f3d; color: #d0d3e0; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: none; }"
            "QPushButton:hover { background-color: #3d3f4d; color: white; }"
        )
        self._shuffle_button.clicked.connect(self.shuffle_slots)
        top_layout.addWidget(self._shuffle_button)

        # Solo Selection (A/B/C Toggle)
        solo_frame = QFrame()
        solo_frame.setStyleSheet(
            "QFrame { background-color: #15161e; border-radius: 4px; padding: 2px 6px; }"
        )
        solo_layout = QHBoxLayout(solo_frame)
        solo_layout.setContentsMargins(4, 2, 4, 2)
        solo_layout.setSpacing(8)

        solo_label = QLabel("<b>Listen:</b>")
        solo_label.setStyleSheet("color: #a0a5b5; font-size: 11px;")
        solo_layout.addWidget(solo_label)

        self._btn_group = QButtonGroup(self)
        self._radio_both = QRadioButton("All")
        self._radio_both.setChecked(True)
        self._radio_both.setStyleSheet("QRadioButton { color: #ffffff; }")

        self._radio_original = QRadioButton("Option A")
        self._radio_original.setStyleSheet(
            "QRadioButton { color: #7d6dfa; font-weight: bold; }"
        )

        self._radio_cleaned = QRadioButton("Option B")
        self._radio_cleaned.setStyleSheet(
            "QRadioButton { color: #55efc4; font-weight: bold; }"
        )

        self._radio_reference = QRadioButton("Option C")
        self._radio_reference.setStyleSheet(
            "QRadioButton { color: #fdcb6e; font-weight: bold; }"
        )

        self._btn_group.addButton(self._radio_both, 0)
        self._btn_group.addButton(self._radio_original, 1)
        self._btn_group.addButton(self._radio_cleaned, 2)
        self._btn_group.addButton(self._radio_reference, 3)
        self._btn_group.idToggled.connect(self.on_solo_mode_changed)

        solo_layout.addWidget(self._radio_both)
        solo_layout.addWidget(self._radio_original)
        solo_layout.addWidget(self._radio_cleaned)
        solo_layout.addWidget(self._radio_reference)
        top_layout.addWidget(solo_frame)

        main_layout.addWidget(top_card)

        # Players Side-by-Side Horizontal Layout
        players_layout = QHBoxLayout()
        players_layout.setSpacing(12)

        self._player_a = WaveformPlayerWidget(title="Option A")
        self._player_b = WaveformPlayerWidget(title="Option B")
        self._player_c = WaveformPlayerWidget(title="Option C")

        players_layout.addWidget(self._player_a)
        players_layout.addWidget(self._player_b)
        players_layout.addWidget(self._player_c)

        main_layout.addLayout(players_layout)

        # Spectrogram Comparison (computed post-render on the exported files)
        self._spectrogram_view = SpectrogramCompareWidget()
        main_layout.addWidget(self._spectrogram_view)

        # Render History Panel
        self._render_history_panel = RenderHistoryPanel(
            cache_manager=self._cache_manager
        )
        main_layout.addWidget(self._render_history_panel)

    def _wire_events(self) -> None:
        self._player_a.seekRequested.connect(lambda pos: self._on_player_seek(0, pos))
        self._player_b.seekRequested.connect(lambda pos: self._on_player_seek(1, pos))
        self._player_c.seekRequested.connect(lambda pos: self._on_player_seek(2, pos))

        self._player_a.positionChanged.connect(
            lambda pos: self._on_player_position_changed(0, pos)
        )
        self._player_b.positionChanged.connect(
            lambda pos: self._on_player_position_changed(1, pos)
        )
        self._player_c.positionChanged.connect(
            lambda pos: self._on_player_position_changed(2, pos)
        )

        self._player_a.playbackStateChanged.connect(
            lambda st: self._on_player_state_changed(0, st)
        )
        self._player_b.playbackStateChanged.connect(
            lambda st: self._on_player_state_changed(1, st)
        )
        self._player_c.playbackStateChanged.connect(
            lambda st: self._on_player_state_changed(2, st)
        )

        self._render_history_panel.renderSelected.connect(self.load_cleaned)

    # --- Slot & Source Mapping Helpers ---

    def _get_player_for_source(self, source_key: str) -> WaveformPlayerWidget:
        players = [self._player_a, self._player_b, self._player_c]
        for idx, key in enumerate(self._slots):
            if key == source_key:
                return players[idx]
        return self._player_a

    def _get_player_index_for_source(self, source_key: str) -> int:
        for idx, key in enumerate(self._slots):
            if key == source_key:
                return idx
        return 0

    def _get_path_for_source(self, source_key: str) -> Optional[Path]:
        if source_key == "raw":
            return self._raw_path
        elif source_key == "humanized":
            return self._humanized_path
        elif source_key == "reference":
            return self._reference_path
        return None

    def _get_source_label(self, source_key: str) -> str:
        if source_key == "raw":
            return "Raw Input"
        elif source_key == "humanized":
            return "DSP-Humanized"
        elif source_key == "reference":
            if self._active_reference_key:
                return f"Ground Truth Ref ({self._active_reference_key})"
            return "Ground Truth Reference"
        return source_key

    def update_blind_labels(self) -> None:
        """Update player widget titles and radio button text based on blind/revealed mode."""
        players = [self._player_a, self._player_b, self._player_c]
        radios = [self._radio_original, self._radio_cleaned, self._radio_reference]
        letters = ["A", "B", "C"]

        for idx in range(3):
            src_key = self._slots[idx]
            label = self._get_source_label(src_key)
            if self._is_revealed:
                players[idx]._title_label.setText(
                    f"<b>Option {letters[idx]}:</b> {label}"
                )
                radios[idx].setText(f"Option {letters[idx]} ({label})")
            else:
                players[idx]._title_label.setText(f"<b>Option {letters[idx]}</b>")
                radios[idx].setText(f"Option {letters[idx]}")

        if self._is_revealed:
            self._reveal_button.setText("🙈 Hide Identities (Blind Mode)")
            self._reveal_button.setStyleSheet(
                "QPushButton { background-color: #e17055; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: none; }"
                "QPushButton:hover { background-color: #ff7675; }"
            )
        else:
            self._reveal_button.setText("👁 Reveal Identities")
            self._reveal_button.setStyleSheet(
                "QPushButton { background-color: #6c5ce7; color: white; font-weight: bold; padding: 6px 12px; border-radius: 4px; border: none; }"
                "QPushButton:hover { background-color: #7d6dfa; }"
            )

    def _update_player_files(self) -> None:
        players = [self._player_a, self._player_b, self._player_c]
        for idx in range(3):
            src_key = self._slots[idx]
            path = self._get_path_for_source(src_key)
            if path and path.is_file():
                if players[idx].get_file_path() != path:
                    players[idx].load_file(path)
            else:
                players[idx].clear()

    @Slot()
    def toggle_reveal(self) -> None:
        """Toggle between blind mode and revealed identities."""
        self._is_revealed = not self._is_revealed
        self.update_blind_labels()

    @Slot()
    def shuffle_slots(self) -> None:
        """Reshuffle the mapping of Option A, B, C to audio sources and reset to blind mode."""
        random.shuffle(self._slots)
        self._is_revealed = False
        self._update_player_files()
        self.update_blind_labels()
        logger.info("ABCompareView reshuffled blind slots: %s", self._slots)

    # --- Public API Methods ---

    def set_track_id(self, track_id: Optional[str]) -> None:
        """Set active track_id for render history panel."""
        self._render_history_panel.set_track_id(track_id)

    def refresh_history(self) -> None:
        """Refresh render history panel."""
        self._render_history_panel.refresh_history()

    def set_vocal_metadata(self, metadata: Optional[dict]) -> None:
        """Set vocal gender/type metadata and update matching ground truth reference stem."""
        self._vocal_metadata = metadata
        self.load_reference(vocal_metadata=metadata)

    def load_original(self, file_path: Union[Path, str]) -> None:
        """Load original ingested track into Raw Input player."""
        path = Path(file_path)
        self._raw_path = path
        self._update_player_files()
        self._spectrogram_view.load_original(path)
        if not self._reference_path:
            self.load_reference(vocal_metadata=self._vocal_metadata)
        logger.info("ABCompareView loaded raw input: %s", path)

    def load_cleaned(self, file_path: Union[Path, str]) -> None:
        """Load latest RenderJob output track into DSP-Humanized player."""
        path = Path(file_path)
        self._humanized_path = path
        self._update_player_files()
        self._spectrogram_view.load_cleaned(path)
        logger.info("ABCompareView loaded DSP-humanized output: %s", path)

    def load_humanized(self, file_path: Union[Path, str]) -> None:
        """Alias for load_cleaned."""
        self.load_cleaned(file_path)

    def load_reference(
        self,
        file_path_or_key: Optional[Union[Path, str]] = None,
        vocal_metadata: Optional[dict] = None,
    ) -> None:
        """Load ground truth reference stem into Reference player."""
        if file_path_or_key and Path(file_path_or_key).is_file():
            path = Path(file_path_or_key)
            self._reference_path = path
            self._active_reference_key = path.stem
        else:
            meta = vocal_metadata or self._vocal_metadata
            key, path = select_reference_stem(vocal_metadata=meta)
            self._reference_path = path
            self._active_reference_key = key if path else None

        self._update_player_files()
        self.update_blind_labels()
        logger.info(
            "ABCompareView loaded ground truth reference (%s): %s",
            self._active_reference_key,
            self._reference_path,
        )

    def clear(self) -> None:
        """Clear all players."""
        self._raw_path = None
        self._humanized_path = None
        self._reference_path = None
        self._active_reference_key = None
        self._player_a.clear()
        self._player_b.clear()
        self._player_c.clear()
        self._spectrogram_view.clear()

    def play_both(self) -> None:
        """Start synchronized playback on all loaded players."""
        self.play_all()

    def play_all(self) -> None:
        """Start synchronized playback on all loaded players."""
        self._player_a.play()
        self._player_b.play()
        self._player_c.play()

    def pause_both(self) -> None:
        """Pause playback on all loaded players."""
        self.pause_all()

    def pause_all(self) -> None:
        """Pause playback on all loaded players."""
        self._player_a.pause()
        self._player_b.pause()
        self._player_c.pause()

    def stop_both(self) -> None:
        """Stop playback on all loaded players."""
        self.stop_all()

    def stop_all(self) -> None:
        """Stop playback on all loaded players."""
        self._player_a.stop()
        self._player_b.stop()
        self._player_c.stop()

    def sync_seek(self, position_ms: int) -> None:
        """Seek all players to position_ms."""
        self._syncing_seek = True
        try:
            self._player_a.seek(position_ms)
            self._player_b.seek(position_ms)
            self._player_c.seek(position_ms)
        finally:
            self._syncing_seek = False

    # --- Position & State Sync Event Logic ---

    def _on_player_seek(self, sender_idx: int, position_ms: int) -> None:
        if self._sync_seek_cb.isChecked() and not self._syncing_seek:
            players = [self._player_a, self._player_b, self._player_c]
            self._syncing_seek = True
            try:
                for idx, player in enumerate(players):
                    if idx != sender_idx:
                        player.seek(position_ms)
            finally:
                self._syncing_seek = False

    def _on_player_position_changed(self, sender_idx: int, position_ms: int) -> None:
        if self._sync_seek_cb.isChecked() and not self._syncing_seek:
            players = [self._player_a, self._player_b, self._player_c]
            sender = players[sender_idx]
            for idx, other in enumerate(players):
                if idx == sender_idx:
                    continue
                other_pos = other.get_position()
                drift = abs(position_ms - other_pos)
                if drift > 20:
                    if other._muted_state or not sender._muted_state:
                        self._syncing_seek = True
                        try:
                            other.seek(position_ms)
                        finally:
                            self._syncing_seek = False

    def _on_player_state_changed(self, sender_idx: int, state: object) -> None:
        from PySide6.QtMultimedia import QMediaPlayer

        if self._sync_seek_cb.isChecked() and not self._syncing_state:
            players = [self._player_a, self._player_b, self._player_c]
            self._syncing_state = True
            try:
                for idx, other in enumerate(players):
                    if idx != sender_idx:
                        if state == QMediaPlayer.PlaybackState.PlayingState:
                            other.play()
                        elif state == QMediaPlayer.PlaybackState.PausedState:
                            other.pause()
                        elif state == QMediaPlayer.PlaybackState.StoppedState:
                            other.stop()
            finally:
                self._syncing_state = False

    # --- Event Slots ---

    @Slot()
    def on_play_both_clicked(self) -> None:
        self.play_all()

    @Slot()
    def on_pause_both_clicked(self) -> None:
        self.pause_all()

    @Slot(int)
    def on_original_seek(self, position_ms: int) -> None:
        self._on_player_seek(self._get_player_index_for_source("raw"), position_ms)

    @Slot(int)
    def on_cleaned_seek(self, position_ms: int) -> None:
        self._on_player_seek(
            self._get_player_index_for_source("humanized"), position_ms
        )

    @Slot(int)
    def on_original_position_changed(self, position_ms: int) -> None:
        self._on_player_position_changed(
            self._get_player_index_for_source("raw"), position_ms
        )

    @Slot(int)
    def on_cleaned_position_changed(self, position_ms: int) -> None:
        self._on_player_position_changed(
            self._get_player_index_for_source("humanized"), position_ms
        )

    @Slot(object)
    def on_original_state_changed(self, state: object) -> None:
        self._on_player_state_changed(self._get_player_index_for_source("raw"), state)

    @Slot(object)
    def on_cleaned_state_changed(self, state: object) -> None:
        self._on_player_state_changed(
            self._get_player_index_for_source("humanized"), state
        )

    @Slot(int, bool)
    def on_solo_mode_changed(self, button_id: int, checked: bool) -> None:
        if not checked:
            return
        players = [self._player_a, self._player_b, self._player_c]

        if button_id == 0:  # All
            for p in players:
                p.set_muted(False)
        elif button_id in (1, 2, 3):  # Option A, B, C
            active_idx = button_id - 1
            for idx, p in enumerate(players):
                p.set_muted(idx != active_idx)
