"""PySide6 QWizard-based first-run setup flow for Music Mastery Enhancer.

Pages:
1. WelcomePage: Introduces Music Mastery Enhancer and features.
2. HardwareCheckPage: Checks torch.cuda.is_available() and displays GPU/CPU status.
3. ModelDownloadPage: Downloads required neural model weights via ModelDownloader in background,
   updating per-model and overall QProgressBar widgets, with visible error state and Retry button.
4. CompletionPage: Displays completion screen with a 'Launch App' button.
"""

from __future__ import annotations

import sys
import warnings
from typing import Dict, Optional

import torch
from PySide6.QtCore import QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from app.cache import get_logger
from app.setup.model_downloader import (
    REQUIRED_MODEL_SPECS,
    ModelDownloader,
    ModelDownloadError,
)
from app.setup.installer import run_diagnostics, get_system_ram_gb, check_cuda_dll, check_pyside6_plugins

logger = get_logger(__name__)


class ModelDownloadWorker(QThread):
    """Background thread for non-blocking model download execution."""

    progress = Signal(str, float)  # model_name, fraction [0.0, 1.0]
    finished = Signal()
    failed = Signal(str, bool)  # error_reason, retryable

    def __init__(
        self,
        downloader: ModelDownloader,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._downloader = downloader

    def run(self) -> None:
        def progress_cb(model_name: str, fraction: float) -> None:
            self.progress.emit(model_name, fraction)

        try:
            logger.info("Starting background download of required models")
            self._downloader.download_required_models(progress_callback=progress_cb)
        except ModelDownloadError as exc:
            logger.warning("Model download failed: %s (retryable=%s)", exc.reason, exc.retryable)
            self.failed.emit(exc.reason, exc.retryable)
        except Exception as exc:
            logger.exception("Unexpected error during model download: %s", exc)
            self.failed.emit(str(exc), True)
        else:
            logger.info("All model downloads completed successfully")
            self.finished.emit()


class WelcomePage(QWizardPage):
    """Wizard page introducing Music Mastery Enhancer."""

    def __init__(self, parent: Optional[QWizard] = None) -> None:
        super().__init__(parent)
        self.setTitle("Welcome to Music Mastery Enhancer")
        self.setSubTitle("First-Run Setup Wizard")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        welcome_label = QLabel(
            "<b>Music Mastery Enhancer</b> cleans AI artifacts, vocal hiss, and metallic "
            "resonances from Suno-generated audio tracks using deep learning models and DSP algorithms."
        )
        welcome_label.setWordWrap(True)
        layout.addWidget(welcome_label)

        intro_text = QLabel(
            "This setup wizard will guide you through:\n"
            "• Auto-detecting CUDA GPU hardware acceleration\n"
            "• Downloading required neural model weights (BS-RoFormer & resemble-enhance)\n"
            "• Preparing your local environment"
        )
        intro_text.setWordWrap(True)
        layout.addWidget(intro_text)

        info_card = QFrame()
        info_card.setFrameShape(QFrame.Shape.StyledPanel)
        card_layout = QVBoxLayout(info_card)
        card_title = QLabel("<b>Key Features:</b>")
        card_body = QLabel(
            "• Vocal & Instrumental Stem Separation\n"
            "• Resemble-Enhance Denoising & Harmonic Reconstruction\n"
            "• Spotify Pedalboard DSP Chain & LUFS Mastering Limiter\n"
            "• Waveform A/B Comparison Player"
        )
        card_layout.addWidget(card_title)
        card_layout.addWidget(card_body)
        layout.addWidget(info_card)

        layout.addStretch()


class HardwareCheckPage(QWizardPage):
    """Wizard page checking for CUDA GPU availability."""

    def __init__(self, parent: Optional[QWizard] = None) -> None:
        super().__init__(parent)
        self.setTitle("Hardware Check")
        self.setSubTitle("Detecting graphics hardware acceleration...")
        self._status_text: str = ""
        self._gpu_detected: bool = False

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        self._status_label = QLabel("Checking hardware capability...")
        self._status_label.setWordWrap(True)
        font = self._status_label.font()
        font.setPointSize(11)
        font.setBold(True)
        self._status_label.setFont(font)
        layout.addWidget(self._status_label)

        self._detail_label = QLabel()
        self._detail_label.setWordWrap(True)
        layout.addWidget(self._detail_label)

        layout.addStretch()

    def initializePage(self) -> None:
        """Called automatically when navigating to this page."""
        super().initializePage()
        self.on_check_hardware()

    @Slot()
    def on_check_hardware(self) -> None:
        """Slot to perform CUDA hardware detection and update UI labels."""
        # Suppress PyTorch warnings during hardware check
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                cuda_available = torch.cuda.is_available()
            except Exception as e:
                logger.warning("Error checking torch.cuda.is_available: %s", e)
                cuda_available = False

        # Run environmental diagnostics
        diags = run_diagnostics()
        ram_gb = diags.get("ram_gb", 8.0)
        pyside_ok = diags.get("pyside6_plugins_ok", True)
        cuda_dll_ok = diags.get("cuda_dll_ok", True)

        detail_paras = []

        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA Device"
            except Exception:
                gpu_name = "CUDA Device"
            self._gpu_detected = True
            self._status_text = f"GPU detected: {gpu_name}"
            self._status_label.setText(self._status_text)
            self._status_label.setStyleSheet("color: #2e7d32;")
            detail_paras.append(
                "NVIDIA CUDA acceleration is available on your system. "
                "Neural stem separation and cleaning will execute using high-performance GPU acceleration."
            )

            if ram_gb < 8.0:
                detail_paras.append(
                    f"⚠️ Low-RAM Warning: Your system has less than 8 GB of RAM ({ram_gb:.1f} GB detected). "
                    "You may experience slow performance, out-of-memory errors, or system instability during stem "
                    "separation or neural cleanup. It is strongly recommended to close other open applications."
                )
            else:
                detail_paras.append(
                    f"Your system has {ram_gb:.1f} GB of RAM, which is sufficient for local neural rendering."
                )

            logger.info("Hardware check result: GPU detected (%s)", gpu_name)
        else:
            self._gpu_detected = False
            self._status_text = "No GPU detected — will run on CPU (slower)"
            self._status_label.setText(self._status_text)
            self._status_label.setStyleSheet("color: #d84315;")
            detail_paras.append(
                "No CUDA-capable GPU was detected. Processing will fall back to CPU execution. "
                "The app remains fully functional, but neural rendering passes will take longer to complete."
            )

            # Handle specific Windows missing driver check or non-Windows missing dll
            if not cuda_dll_ok:
                detail_paras.append(
                    "⚠️ Missing Driver: The NVIDIA CUDA driver library (nvcuda.dll) was not found. "
                    "If you have an NVIDIA graphics card, installing/updating the latest official NVIDIA drivers "
                    "may enable GPU acceleration."
                )

            if ram_gb < 8.0:
                detail_paras.append(
                    f"⚠️ Critical Memory Warning: Your system has low physical RAM ({ram_gb:.1f} GB detected). "
                    "Running heavy deep learning models on CPU with low memory can cause severe slowdowns, "
                    "virtual memory thrashing, or application crashes. Close background tasks to free up memory."
                )
            else:
                detail_paras.append(
                    f"Your system has {ram_gb:.1f} GB of RAM, which is sufficient for local CPU rendering."
                )

            logger.info("Hardware check result: No GPU detected, using CPU fallback.")

        # PySide6 Plugin Warnings if they are missing/corrupted in PyInstaller builds
        if not pyside_ok:
            detail_paras.append(
                "⚠️ PySide6 Plugins Diagnostic Warning: Missing or invalid PySide6 platform plugins detected. "
                "This may cause user interface rendering or startup issues."
            )

        self._detail_label.setText("\n\n".join(detail_paras))


class ModelDownloadPage(QWizardPage):
    """Wizard page executing model weight downloads with per-model & overall progress bars."""

    def __init__(
        self,
        downloader: Optional[ModelDownloader] = None,
        parent: Optional[QWizard] = None,
    ) -> None:
        super().__init__(parent)
        self.setTitle("Model Download")
        self.setSubTitle("Downloading required neural network model weights...")

        self._downloader = downloader or ModelDownloader()
        self._worker: Optional[ModelDownloadWorker] = None
        self._is_complete: bool = False
        self._model_progress_bars: Dict[str, QProgressBar] = {}
        self._model_fractions: Dict[str, float] = {}

        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        self._status_label = QLabel("Preparing to download model weights...")
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        # Per-model progress section
        models_frame = QFrame()
        models_frame.setFrameShape(QFrame.Shape.StyledPanel)
        models_layout = QVBoxLayout(models_frame)
        models_layout.setSpacing(10)

        for spec in REQUIRED_MODEL_SPECS:
            model_name = spec.name
            lbl = QLabel(f"<b>{model_name}</b> ({spec.filename})")
            pbar = QProgressBar()
            pbar.setRange(0, 100)
            pbar.setValue(0)
            models_layout.addWidget(lbl)
            models_layout.addWidget(pbar)

            self._model_progress_bars[model_name] = pbar
            self._model_fractions[model_name] = 0.0

        layout.addWidget(models_frame)

        # Overall progress bar
        overall_label = QLabel("<b>Overall Progress:</b>")
        self._overall_progress_bar = QProgressBar()
        self._overall_progress_bar.setRange(0, 100)
        self._overall_progress_bar.setValue(0)

        layout.addWidget(overall_label)
        layout.addWidget(self._overall_progress_bar)

        # Visible error state container
        self._error_frame = QFrame()
        self._error_frame.setFrameShape(QFrame.Shape.StyledPanel)
        self._error_frame.setStyleSheet("background-color: #ffebee; border: 1px solid #ef5350;")
        error_layout = QVBoxLayout(self._error_frame)

        self._error_label = QLabel()
        self._error_label.setWordWrap(True)
        self._error_label.setStyleSheet("color: #c62828; font-weight: bold;")
        error_layout.addWidget(self._error_label)

        self._retry_button = QPushButton("Retry Download")
        self._retry_button.clicked.connect(self.on_retry_clicked)
        error_layout.addWidget(self._retry_button)

        self._error_frame.setVisible(False)
        layout.addWidget(self._error_frame)

        # Skip button container
        skip_layout = QHBoxLayout()
        skip_layout.addStretch()
        self._skip_button = QPushButton("Skip Download")
        self._skip_button.clicked.connect(self.on_skip_clicked)
        skip_layout.addWidget(self._skip_button)
        layout.addLayout(skip_layout)

        layout.addStretch()

    def isComplete(self) -> bool:
        """Wizard page completion status controlling Next/Finish navigation."""
        return self._is_complete

    def initializePage(self) -> None:
        """Automatically trigger download when page becomes active if not complete."""
        super().initializePage()
        if not self._is_complete:
            self.on_start_download()

    @Slot()
    def on_start_download(self) -> None:
        """Slot to kick off the background model download worker."""
        self._error_frame.setVisible(False)
        self._status_label.setText("Downloading neural model weights...")
        self._status_label.setStyleSheet("")

        # Reset progress states
        for name, pbar in self._model_progress_bars.items():
            pbar.setValue(0)
            self._model_fractions[name] = 0.0
        self._overall_progress_bar.setValue(0)

        self._is_complete = False
        self.completeChanged.emit()

        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()

        self._worker = ModelDownloadWorker(self._downloader, parent=self)
        self._worker.progress.connect(self.on_download_progress)
        self._worker.finished.connect(self.on_download_finished)
        self._worker.failed.connect(self.on_download_failed)
        self._worker.start()

    @Slot(str, float)
    def on_download_progress(self, model_name: str, fraction: float) -> None:
        """Slot handling progress updates per model."""
        clamped_fraction = max(0.0, min(1.0, fraction))
        self._model_fractions[model_name] = clamped_fraction

        if model_name in self._model_progress_bars:
            self._model_progress_bars[model_name].setValue(int(clamped_fraction * 100))

        # Update overall progress as average of model fractions
        total_specs = len(self._model_progress_bars)
        if total_specs > 0:
            avg_fraction = sum(self._model_fractions.values()) / total_specs
            self._overall_progress_bar.setValue(int(avg_fraction * 100))

    @Slot()
    def on_download_finished(self) -> None:
        """Slot handling successful completion of model downloads."""
        for pbar in self._model_progress_bars.values():
            pbar.setValue(100)
        self._overall_progress_bar.setValue(100)

        self._status_label.setText("All required model weights downloaded and verified successfully!")
        self._status_label.setStyleSheet("color: #2e7d32; font-weight: bold;")
        self._error_frame.setVisible(False)

        self._is_complete = True
        self.completeChanged.emit()
        logger.info("ModelDownloadPage: Download complete, page marked complete.")

    @Slot(str, bool)
    def on_download_failed(self, reason: str, retryable: bool) -> None:
        """Slot handling download failures and presenting the error state and Retry button."""
        self._status_label.setText("Model download failed. Please review error below.")
        self._status_label.setStyleSheet("color: #c62828;")

        self._error_label.setText(f"Download Error: {reason}")
        self._retry_button.setVisible(retryable)
        self._error_frame.setVisible(True)

        self._is_complete = False
        self.completeChanged.emit()
        logger.warning("ModelDownloadPage: Download failed (%s), retryable=%s", reason, retryable)

    @Slot()
    def on_retry_clicked(self) -> None:
        """Slot handling click on Retry button."""
        logger.info("ModelDownloadPage: Retry clicked by user.")
        self.on_start_download()

    @Slot()
    def on_skip_clicked(self) -> None:
        """Slot handling click on Skip Download button."""
        logger.info("ModelDownloadPage: Skip clicked by user.")
        if self._worker is not None and self._worker.isRunning():
            self._worker.quit()
            self._worker.wait()
        
        for pbar in self._model_progress_bars.values():
            pbar.setValue(100)
        self._overall_progress_bar.setValue(100)

        self._status_label.setText("Model download skipped. Models will be downloaded on first use.")
        self._status_label.setStyleSheet("color: #f57f17; font-weight: bold;")
        self._error_frame.setVisible(False)
        self._skip_button.setEnabled(False)

        self._is_complete = True
        self.completeChanged.emit()


class CompletionPage(QWizardPage):
    """Wizard page signaling successful setup completion with a 'Launch App' button."""

    def __init__(self, parent: Optional[QWizard] = None) -> None:
        super().__init__(parent)
        self.setTitle("Setup Complete")
        self.setSubTitle("Music Mastery Enhancer is ready to use!")
        self._init_ui()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(16)

        congrats_label = QLabel(
            "<b>Congratulations! First-run setup is complete.</b>"
        )
        font = congrats_label.font()
        font.setPointSize(11)
        congrats_label.setFont(font)
        layout.addWidget(congrats_label)

        summary_label = QLabel(
            "• Hardware check completed\n"
            "• Neural model weights downloaded and verified\n\n"
            "Click <b>Launch App</b> below to begin enhancing Suno audio tracks."
        )
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        button_layout = QHBoxLayout()
        button_layout.addStretch()

        self._launch_button = QPushButton("Launch App")
        self._launch_button.setStyleSheet(
            "QPushButton { background-color: #1976d2; color: white; font-weight: bold; padding: 8px 16px; font-size: 14px; }"
            "QPushButton:hover { background-color: #1565c0; }"
        )
        self._launch_button.clicked.connect(self.on_launch_clicked)
        button_layout.addWidget(self._launch_button)

        layout.addLayout(button_layout)
        layout.addStretch()

    def initializePage(self) -> None:
        """Configure Finish button text on entering CompletionPage."""
        super().initializePage()
        if self.wizard() is not None:
            self.wizard().setButtonText(QWizard.WizardButton.FinishButton, "Launch App")

    @Slot()
    def on_launch_clicked(self) -> None:
        """Slot handling Launch App button click."""
        logger.info("CompletionPage: Launch App clicked, accepting wizard.")
        if self.wizard() is not None:
            self.wizard().accept()


class SetupWizard(QWizard):
    """PySide6 QWizard for first-run setup flow."""

    def __init__(
        self,
        downloader: Optional[ModelDownloader] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Music Mastery Enhancer Setup")
        self.setMinimumSize(680, 500)
        self.setWizardStyle(QWizard.WizardStyle.ModernStyle)

        self.welcome_page = WelcomePage(self)
        self.hardware_page = HardwareCheckPage(self)
        self.download_page = ModelDownloadPage(downloader=downloader, parent=self)
        self.completion_page = CompletionPage(self)

        self.addPage(self.welcome_page)
        self.addPage(self.hardware_page)
        self.addPage(self.download_page)
        self.addPage(self.completion_page)

        self.currentIdChanged.connect(self.on_page_changed)

    @Slot(int)
    def on_page_changed(self, page_id: int) -> None:
        """Slot tracking wizard page navigation changes."""
        logger.info("SetupWizard navigated to page ID %d", page_id)


# Alias for backward compatibility / explicit naming
FirstRunSetupWizard = SetupWizard


def main() -> None:
    """Helper entry point for testing or launching the setup wizard directly."""
    app = QApplication(sys.argv)
    wizard = SetupWizard()
    wizard.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
