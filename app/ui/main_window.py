"""PySide6 Main Application Window for Music Mastery Enhancer.

Provides the primary user interface including:
- File ingestion panel with drag-and-drop support and file picker (.wav / .mp3)
- Tabbed navigation for 'Stem Separation', 'Vocal Controls', 'Instrumental Controls', and 'A/B Compare'
- Integrated status bar displaying RenderJob stage and progress
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QSize, Qt, Signal, Slot
from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDropEvent
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.cache import get_logger
from app.cache.cache_manager import CacheManager
from app.core.ingestion import UnsupportedAudioFormatError, load_and_normalize_track
from app.models.preset import Preset
from app.models.settings import Settings
from app.ui.ab_compare_view import ABCompareView
from app.ui.instrumental_panel import InstrumentalPanel
from app.ui.vocal_panel import VocalPanel
from app.workers.render_job import RenderJob
from app.workers.separation_job import SeparationJob

logger = get_logger(__name__)

SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3"}


class FileLoadPanel(QFrame):
    """Widget panel for track ingestion supporting drag-and-drop and file picking."""

    fileSelected = Signal(Path)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.setObjectName("FileLoadPanel")
        self._init_ui()

    def _init_ui(self) -> None:
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        self._drop_label = QLabel(
            "<b>Drag & Drop Suno Audio Track Here</b><br>"
            "<span style='color: #8a8d9b;'>Supports .wav and .mp3 formats</span>"
        )
        self._drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._drop_label)

        button_layout = QHBoxLayout()
        button_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self._browse_button = QPushButton("Browse Audio File...")
        self._browse_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._browse_button.setMinimumWidth(180)
        self._browse_button.setToolTip("Open file browser to select an audio file")
        self._browse_button.setAccessibleName("Browse Audio File")
        self._browse_button.setAccessibleDescription("Opens a file dialog to select a Suno track in WAV or MP3 format.")
        self._browse_button.clicked.connect(self.on_browse_clicked)
        button_layout.addWidget(self._browse_button)

        layout.addLayout(button_layout)

        self._info_label = QLabel("No track selected")
        self._info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._info_label.setStyleSheet("color: #a0a5b5; font-size: 11px;")
        layout.addWidget(self._info_label)

        self._apply_drop_style(hover=False)

    def _apply_drop_style(self, hover: bool) -> None:
        border_color = "#6c5ce7" if hover else "#3d3f4d"
        bg_color = "#252733" if hover else "#1b1c24"
        self.setStyleSheet(
            f"QFrame#FileLoadPanel {{"
            f"  border: 2px dashed {border_color};"
            f"  border-radius: 10px;"
            f"  background-color: {bg_color};"
            f"}}"
            f"QPushButton {{"
            f"  background-color: #6c5ce7;"
            f"  color: #ffffff;"
            f"  font-weight: bold;"
            f"  border: none;"
            f"  border-radius: 6px;"
            f"  padding: 8px 16px;"
            f"}}"
            f"QPushButton:hover {{"
            f"  background-color: #7d6dfa;"
            f"}}"
            f"QPushButton:disabled {{"
            f"  background-color: #4a4b57;"
            f"  color: #8a8d9b;"
            f"}}"
        )

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                path = Path(url.toLocalFile())
                if path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                    event.acceptProposedAction()
                    self._apply_drop_style(hover=True)
                    return
        event.ignore()

    def dragLeaveEvent(self, event: QDragLeaveEvent) -> None:
        self._apply_drop_style(hover=False)
        event.accept()

    def dropEvent(self, event: QDropEvent) -> None:
        self._apply_drop_style(hover=False)
        if event.mimeData().hasUrls():
            for url in event.mimeData().urls():
                file_path = Path(url.toLocalFile())
                if file_path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS:
                    event.acceptProposedAction()
                    self.fileSelected.emit(file_path)
                    return

    @Slot()
    def on_browse_clicked(self) -> None:
        file_dialog = QFileDialog(self, "Select Suno Audio Track")
        file_dialog.setNameFilter("Audio Files (*.wav *.mp3);;WAV Files (*.wav);;MP3 Files (*.mp3);;All Files (*)")
        file_dialog.setFileMode(QFileDialog.FileMode.ExistingFile)

        if file_dialog.exec() == QFileDialog.DialogCode.Accepted:
            selected_files = file_dialog.selectedFiles()
            if selected_files:
                self.fileSelected.emit(Path(selected_files[0]))

    def set_track_info(self, input_path: Path, normalized_path: Optional[Path] = None) -> None:
        norm_status = f" | Normalized: {normalized_path.name}" if normalized_path else ""
        self._drop_label.setText(
            f"<b style='color: #55efc4;'>Selected Track: {input_path.name}</b><br>"
            f"<span style='color: #a0a5b5;'>{input_path.parent}</span>"
        )
        self._info_label.setText(f"File: {input_path.name}{norm_status}")


class StemSeparationPanel(QWidget):
    """Panel view for stem separation settings, actions, and status."""

    separationRequested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        title = QLabel("<h2>Stem Separation</h2>")
        layout.addWidget(title)

        desc = QLabel(
            "Splits Suno tracks into clean <b>Vocal</b> and <b>Instrumental</b> stems using "
            "the high-fidelity <i>BS-RoFormer</i> neural separation checkpoint."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        card = QFrame()
        card.setStyleSheet("QFrame { background-color: #21232e; border-radius: 8px; padding: 16px; }")
        card_layout = QVBoxLayout(card)

        self._model_label = QLabel("<b>Separation Model:</b> BS-RoFormer (model_bs_roformer_ep_317_sdr_12.9755.ckpt)")
        card_layout.addWidget(self._model_label)

        self._stem_status_label = QLabel("Stems status: <i>No stems generated yet. Ingest a file to run separation.</i>")
        self._stem_status_label.setWordWrap(True)
        card_layout.addWidget(self._stem_status_label)

        # Explicit Extract Stems action button
        self._extract_button = QPushButton("✂️  Extract Stems")
        self._extract_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._extract_button.setToolTip("Separate the audio track into vocal and instrumental stems")
        self._extract_button.setAccessibleName("Extract Stems")
        self._extract_button.setAccessibleDescription(
            "Extracts separate vocal and instrumental tracks from the selected file using the BS-RoFormer model."
        )
        self._extract_button.setStyleSheet(
            "QPushButton { background-color: #6c5ce7; color: white; font-weight: bold; font-size: 14px; padding: 10px 20px; border-radius: 6px; margin-top: 10px; }"
            "QPushButton:hover { background-color: #7d6dfa; }"
            "QPushButton:disabled { background-color: #4a4b57; color: #8a8d9b; }"
        )
        self._extract_button.setEnabled(False)
        self._extract_button.clicked.connect(self.on_extract_clicked)
        card_layout.addWidget(self._extract_button)

        layout.addWidget(card)
        layout.addStretch()

    def set_track_loaded(self, normalized_path: Path) -> None:
        self._stem_status_label.setText(
            f"Ready to separate track: <b>{normalized_path.name}</b><br>"
            "Outputs: <code>vocal.wav</code>, <code>instrumental.wav</code>"
        )
        self._extract_button.setEnabled(True)
        self._extract_button.setText("✂️  Extract Stems")

    def set_separating(self, stage: str) -> None:
        self._stem_status_label.setText(
            f"<b>{stage}…</b> Separating vocal and instrumental stems "
            "(this can take a few minutes on CPU)."
        )
        self._extract_button.setEnabled(False)
        self._extract_button.setText("Extracting…")

    def set_separated(self, vocal_path: Path, instrumental_path: Path) -> None:
        self._stem_status_label.setText(
            "Stems extracted:<br>"
            f"&bull; <code>{vocal_path}</code><br>"
            f"&bull; <code>{instrumental_path}</code>"
        )
        self._extract_button.setEnabled(True)
        self._extract_button.setText("✂️  Re-extract Stems")

    def set_separation_failed(self, error: str) -> None:
        self._stem_status_label.setText(f"<span style='color:#ff7675;'>Separation failed: {error}</span>")
        self._extract_button.setEnabled(True)
        self._extract_button.setText("✂️  Extract Stems")

    @Slot()
    def on_extract_clicked(self) -> None:
        self.separationRequested.emit()


class MainWindow(QMainWindow):
    """Main Application Window for Music Mastery Enhancer."""

    def __init__(self, cache_manager: Optional[CacheManager] = None) -> None:
        super().__init__()
        self.setWindowTitle("Music Mastery Enhancer")
        self.setMinimumSize(960, 680)

        self._cache_manager = cache_manager or CacheManager()
        self._current_input_path: Optional[Path] = None
        self._normalized_path: Optional[Path] = None
        self._active_render_job: Optional[RenderJob] = None
        self._active_separation_job: Optional[SeparationJob] = None

        self._init_ui()
        self._apply_global_theme()

    def _init_ui(self) -> None:
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(16)

        # Header
        header = QLabel("<h1>Music Mastery Enhancer</h1>")
        header.setStyleSheet("color: #7d6dfa; margin-bottom: 0px;")
        main_layout.addWidget(header)

        # Ingestion / File load panel
        self._file_load_panel = FileLoadPanel()
        self._file_load_panel.fileSelected.connect(self.on_file_selected)
        main_layout.addWidget(self._file_load_panel)

        # Navigation Tabs
        self._tab_widget = QTabWidget()
        self._tab_widget.setObjectName("MainTabs")

        self._stem_separation_panel = StemSeparationPanel()
        self._stem_separation_panel.separationRequested.connect(self.on_extract_stems_requested)

        # Real control panels with full slider/toggle controls
        self._vocal_panel = VocalPanel(cache_manager=self._cache_manager)
        self._vocal_panel.renderRequested.connect(self.on_vocal_render_requested)

        self._instrumental_panel = InstrumentalPanel(cache_manager=self._cache_manager)
        self._instrumental_panel.renderRequested.connect(self.on_instrumental_render_requested)

        self._ab_compare_view = ABCompareView(cache_manager=self._cache_manager)

        self._tab_widget.addTab(self._stem_separation_panel, "Stem Separation")
        self._tab_widget.addTab(self._vocal_panel, "Vocal Controls")
        self._tab_widget.addTab(self._instrumental_panel, "Instrumental Controls")
        self._tab_widget.addTab(self._ab_compare_view, "A/B Compare")

        main_layout.addWidget(self._tab_widget)

        # Master render button (uses settings from both panels)
        render_bar = QHBoxLayout()
        render_bar.setContentsMargins(0, 0, 0, 0)

        self._render_button = QPushButton("⚡  Render & Master Track")
        self._render_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._render_button.setToolTip("Start vocal/instrumental processing and master the combined output")
        self._render_button.setAccessibleName("Render and Master Track")
        self._render_button.setAccessibleDescription(
            "Applies selected neural cleanup, DSP processing, and LUFS mastering to render the final audio track."
        )
        self._render_button.setStyleSheet(
            "QPushButton { background-color: #00b894; color: white; font-weight: bold; font-size: 14px; padding: 10px 20px; border-radius: 6px; }"
            "QPushButton:hover { background-color: #00cec9; }"
            "QPushButton:disabled { background-color: #4a4b57; color: #8a8d9b; }"
        )
        self._render_button.setEnabled(False)
        self._render_button.clicked.connect(self.on_render_requested)
        render_bar.addWidget(self._render_button)

        main_layout.addLayout(render_bar)

        # Status bar setup
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)

        self._status_label = QLabel("Ready. Load a .wav or .mp3 track to begin.")
        self._status_bar.addWidget(self._status_label, 1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedWidth(200)
        self._progress_bar.setVisible(False)
        self._status_bar.addPermanentWidget(self._progress_bar)

        self._cancel_button = QPushButton("Cancel Render")
        self._cancel_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self._cancel_button.setToolTip("Cancel the active render or separation task")
        self._cancel_button.setAccessibleName("Cancel Render")
        self._cancel_button.setAccessibleDescription("Stops and cancels the current running render or separation background job.")
        self._cancel_button.setStyleSheet("background-color: #d63031; color: white; padding: 2px 8px; border-radius: 4px;")
        self._cancel_button.setVisible(False)
        self._cancel_button.clicked.connect(self.on_cancel_render_clicked)
        self._status_bar.addPermanentWidget(self._cancel_button)

    def _apply_global_theme(self) -> None:
        self.setStyleSheet(
            "QMainWindow { background-color: #12131a; font-family: 'Segoe UI', sans-serif; }"
            "QWidget { color: #e1e2e6; }"
            "QTabWidget::pane { border: 1px solid #2d2f3d; background-color: #1a1b24; border-radius: 6px; }"
            "QTabBar::tab { background-color: #15161e; color: #9a9db0; padding: 10px 20px; font-weight: bold; border-top-left-radius: 6px; border-top-right-radius: 6px; margin-right: 2px; }"
            "QTabBar::tab:selected { background-color: #1a1b24; color: #7d6dfa; border-bottom: 2px solid #7d6dfa; }"
            "QTabBar::tab:hover:!selected { background-color: #212330; color: #d0d3e0; }"
            "QStatusBar { background-color: #15161e; color: #a0a5b5; border-top: 1px solid #252733; }"
            "QProgressBar { border: 1px solid #3d3f4d; border-radius: 4px; text-align: center; background-color: #15161e; color: #ffffff; }"
            "QProgressBar::chunk { background-color: #6c5ce7; border-radius: 3px; }"
        )

    def _collect_preset(self) -> Preset:
        """Merge vocal and instrumental panel settings into a single Preset for the render job."""
        vocal_settings = self._vocal_panel.get_settings()
        instrumental_settings = self._instrumental_panel.get_settings()

        return Preset(
            # Vocal settings from VocalPanel
            vocal_denoise_enabled=vocal_settings.vocal_denoise_enabled,
            vocal_denoise_intensity=vocal_settings.vocal_denoise_intensity,
            vocal_enhance_enabled=vocal_settings.vocal_enhance_enabled,
            vocal_enhance_intensity=vocal_settings.vocal_enhance_intensity,
            vocal_clean_intensity=vocal_settings.vocal_clean_intensity,
            vocal_gain_db=vocal_settings.vocal_gain_db,
            notch_depth_db=vocal_settings.notch_depth_db,
            # Instrumental settings from InstrumentalPanel
            instrumental_denoise_enabled=instrumental_settings.instrumental_denoise_enabled,
            instrumental_denoise_intensity=instrumental_settings.instrumental_denoise_intensity,
            instrumental_enhance_enabled=instrumental_settings.instrumental_enhance_enabled,
            instrumental_enhance_intensity=instrumental_settings.instrumental_enhance_intensity,
            instrumental_mud_cut_hz=instrumental_settings.instrumental_mud_cut_hz,
            instrumental_dehiss_shelf_hz=instrumental_settings.instrumental_dehiss_shelf_hz,
            instrumental_dehiss_gain_db=instrumental_settings.instrumental_dehiss_gain_db,
            instrumental_gain_db=instrumental_settings.instrumental_gain_db,
            # Mastering (from vocal panel's LUFS target or defaults)
            lufs_target=vocal_settings.lufs_target,
        )

    @Slot(Path)
    def on_file_selected(self, input_path: Path) -> None:
        logger.info("File selected for ingestion: %s", input_path)
        self._status_label.setText(f"Ingesting track: {input_path.name}...")
        QApplication.processEvents()

        try:
            normalized_path = load_and_normalize_track(input_path, self._cache_manager)
        except UnsupportedAudioFormatError as exc:
            logger.warning("Ingestion failed: %s", exc)
            self._status_label.setText(f"Error: {exc}")
            QMessageBox.critical(self, "Invalid Audio File", str(exc))
        except Exception as exc:
            logger.exception("Unexpected error during ingestion: %s", exc)
            self._status_label.setText(f"Error ingesting file: {exc}")
            QMessageBox.critical(self, "Ingestion Error", f"Failed to ingest track: {exc}")
        else:
            self._current_input_path = input_path
            self._normalized_path = normalized_path

            self._file_load_panel.set_track_info(input_path, normalized_path)
            self._stem_separation_panel.set_track_loaded(normalized_path)
            self._render_button.setEnabled(True)
            self._ab_compare_view.load_original(normalized_path)
            self._ab_compare_view.set_track_id(normalized_path.parent.name)

            self._status_label.setText(f"Track ingested and normalized: {normalized_path.name}")
            logger.info("Track successfully normalized: %s", normalized_path)

    @Slot()
    def on_render_requested(self) -> None:
        """Triggered by the main 'Render & Master Track' button — collects from both panels."""
        if self._current_input_path is None:
            QMessageBox.warning(self, "No Track Ingested", "Please select and ingest a track before rendering.")
            return

        preset = self._collect_preset()
        self.start_render_job(preset)

    @Slot()
    def on_extract_stems_requested(self) -> None:
        """Triggered by the Stem Separation panel's 'Extract Stems' button.

        Runs ONLY ingestion + separation (no neural denoise/master), so users can pull the
        raw vocal/instrumental stems without waiting on the full CPU render pipeline.
        """
        if self._current_input_path is None:
            QMessageBox.warning(self, "No Track Ingested", "Please select and ingest a track before extracting stems.")
            return

        if self._active_separation_job is not None and self._active_separation_job.isRunning():
            logger.warning("SeparationJob is already running")
            return
        if self._active_render_job is not None and self._active_render_job.isRunning():
            QMessageBox.warning(self, "Render In Progress", "A render is already running; please wait for it to finish.")
            return

        logger.info("Starting SeparationJob for %s", self._current_input_path)
        job = SeparationJob(
            input_path=self._current_input_path,
            cache_manager=self._cache_manager,
            parent=self,
        )
        job.stageChanged.connect(self.on_render_stage_changed)
        job.progressChanged.connect(self.on_render_progress_changed)
        job.separationFinished.connect(self.on_separation_finished)
        job.failed.connect(self.on_separation_failed)
        job.cancelled.connect(self.on_render_cancelled)

        self._active_separation_job = job

        self._stem_separation_panel.set_separating("Separating")
        self._status_label.setText("Extracting stems...")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._cancel_button.setVisible(True)
        self._render_button.setEnabled(False)

        job.start()

    @Slot(Path, Path)
    def on_separation_finished(self, vocal_path: Path, instrumental_path: Path) -> None:
        logger.info("Stem separation finished: %s, %s", vocal_path, instrumental_path)
        self._status_label.setText("Stems extracted.")
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._render_button.setEnabled(True)
        self._stem_separation_panel.set_separated(vocal_path, instrumental_path)
        self._active_separation_job = None

        QMessageBox.information(
            self,
            "Stems Extracted",
            f"Vocal and instrumental stems saved to:\n{vocal_path.parent}",
        )

    @Slot(str)
    def on_separation_failed(self, error: str) -> None:
        logger.error("Stem separation failed: %s", error)
        self._status_label.setText(f"Stem separation failed: {error}")
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._render_button.setEnabled(True)
        self._stem_separation_panel.set_separation_failed(error)
        self._active_separation_job = None

        QMessageBox.critical(self, "Separation Failed", f"An error occurred during stem separation:\n{error}")

    @Slot(Settings)
    def on_vocal_render_requested(self, settings: Settings) -> None:
        """Triggered by the VocalPanel's own 'Apply / Render' button."""
        if self._current_input_path is None:
            QMessageBox.warning(self, "No Track Ingested", "Please select and ingest a track before rendering.")
            return

        preset = self._collect_preset()
        self.start_render_job(preset)

    @Slot(Settings)
    def on_instrumental_render_requested(self, settings: Settings) -> None:
        """Triggered by the InstrumentalPanel's own 'Apply / Render' button."""
        if self._current_input_path is None:
            QMessageBox.warning(self, "No Track Ingested", "Please select and ingest a track before rendering.")
            return

        preset = self._collect_preset()
        self.start_render_job(preset)

    def start_render_job(self, preset: Preset, output_path: Optional[Path] = None) -> None:
        if self._current_input_path is None:
            logger.warning("Cannot start RenderJob without current input path")
            return

        if self._active_render_job is not None and self._active_render_job.isRunning():
            logger.warning("RenderJob is already running")
            return
        if self._active_separation_job is not None and self._active_separation_job.isRunning():
            QMessageBox.warning(self, "Separation In Progress", "Stem extraction is running; please wait for it to finish.")
            return

        logger.info("Starting RenderJob for %s with preset: %s", self._current_input_path, preset)
        job = RenderJob(
            input_path=self._current_input_path,
            preset=preset,
            cache_manager=self._cache_manager,
            output_path=output_path,
            parent=self,
        )

        job.stageChanged.connect(self.on_render_stage_changed)
        job.progressChanged.connect(self.on_render_progress_changed)
        job.finished.connect(self.on_render_finished)
        job.failed.connect(self.on_render_failed)
        job.cancelled.connect(self.on_render_cancelled)

        self._active_render_job = job

        self._status_label.setText("Starting render job...")
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(True)
        self._cancel_button.setVisible(True)
        self._render_button.setEnabled(False)

        job.start()

    @Slot(str)
    def on_render_stage_changed(self, stage: str) -> None:
        self._status_label.setText(f"Rendering stage: {stage}...")

    @Slot(float)
    def on_render_progress_changed(self, progress: float) -> None:
        percentage = int(progress * 100)
        self._progress_bar.setValue(percentage)

    @Slot(Path)
    def on_render_finished(self, output_path: Path) -> None:
        logger.info("Render finished successfully: %s", output_path)
        self._status_label.setText(f"Render completed: {output_path.name}")
        self._progress_bar.setValue(100)
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._render_button.setEnabled(True)
        self._ab_compare_view.load_cleaned(output_path)
        self._ab_compare_view.refresh_history()
        self._active_render_job = None

        QMessageBox.information(
            self,
            "Render Complete",
            f"Track successfully cleaned and mastered!\nSaved to: {output_path}",
        )

    @Slot(str)
    def on_render_failed(self, error: str) -> None:
        logger.error("Render failed: %s", error)
        self._status_label.setText(f"Render failed: {error}")
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._render_button.setEnabled(True)
        self._active_render_job = None

        QMessageBox.critical(self, "Render Failed", f"An error occurred during rendering:\n{error}")

    @Slot()
    def on_render_cancelled(self) -> None:
        logger.info("Render was cancelled")
        self._status_label.setText("Render cancelled.")
        self._progress_bar.setVisible(False)
        self._cancel_button.setVisible(False)
        self._render_button.setEnabled(True)
        if self._normalized_path is not None:
            self._stem_separation_panel.set_track_loaded(self._normalized_path)
        self._active_render_job = None
        self._active_separation_job = None

    @Slot()
    def on_cancel_render_clicked(self) -> None:
        if self._active_render_job is not None and self._active_render_job.isRunning():
            self._status_label.setText("Cancelling render...")
            self._active_render_job.cancel()
        if self._active_separation_job is not None and self._active_separation_job.isRunning():
            self._status_label.setText("Cancelling separation...")
            self._active_separation_job.cancel()
