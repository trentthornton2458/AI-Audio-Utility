"""Tests for PySide6 setup wizard UI (app.ui.setup_wizard)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from PySide6.QtWidgets import QWizard, QWizardPage

from app.setup.model_downloader import ModelDownloadError, REQUIRED_MODEL_SPECS
from app.ui.setup_wizard import (
    CompletionPage,
    HardwareCheckPage,
    ModelDownloadPage,
    SetupWizard,
    WelcomePage,
)


def test_setup_wizard_page_structure(qtbot):
    mock_downloader = MagicMock()
    wizard = SetupWizard(downloader=mock_downloader)
    qtbot.addWidget(wizard)

    page_ids = wizard.pageIds()
    assert len(page_ids) == 4
    assert isinstance(wizard.welcome_page, WelcomePage)
    assert isinstance(wizard.hardware_page, HardwareCheckPage)
    assert isinstance(wizard.download_page, ModelDownloadPage)
    assert isinstance(wizard.completion_page, CompletionPage)

    assert wizard.page(page_ids[0]) == wizard.welcome_page
    assert wizard.page(page_ids[1]) == wizard.hardware_page
    assert wizard.page(page_ids[2]) == wizard.download_page
    assert wizard.page(page_ids[3]) == wizard.completion_page


def test_hardware_check_page_gpu_detected(qtbot):
    page = HardwareCheckPage()
    qtbot.addWidget(page)

    with patch("torch.cuda.is_available", return_value=True), patch(
        "torch.cuda.get_device_name", return_value="NVIDIA GeForce RTX 4090"
    ), patch("torch.cuda.device_count", return_value=1):
        page.on_check_hardware()
        assert page._gpu_detected is True
        assert page._status_text == "GPU detected: NVIDIA GeForce RTX 4090"
        assert "GPU detected: NVIDIA GeForce RTX 4090" in page._status_label.text()


def test_hardware_check_page_cpu_fallback(qtbot):
    page = HardwareCheckPage()
    qtbot.addWidget(page)

    with patch("torch.cuda.is_available", return_value=False):
        page.on_check_hardware()
        assert page._gpu_detected is False
        assert page._status_text == "No GPU detected — will run on CPU (slower)"
        assert "No GPU detected — will run on CPU (slower)" in page._status_label.text()


def test_model_download_page_progress_and_success(qtbot):
    mock_downloader = MagicMock()
    
    mock_specs = [
        MagicMock(name="BS-RoFormer", filename="bs.ckpt"),
        MagicMock(name="resemble-enhance", filename="re.pth"),
    ]
    mock_specs[0].name = "BS-RoFormer"
    mock_specs[1].name = "resemble-enhance"
    
    with patch("app.ui.setup_wizard.REQUIRED_MODEL_SPECS", mock_specs):
        page = ModelDownloadPage(downloader=mock_downloader)
        qtbot.addWidget(page)

    assert not page.isComplete()

    # Simulate progress updates
    page.on_download_progress("BS-RoFormer", 0.5)
    assert page._model_progress_bars["BS-RoFormer"].value() == 50
    assert page._overall_progress_bar.value() == 25  # 50% of 1 of 2 models

    page.on_download_progress("resemble-enhance", 1.0)
    assert page._model_progress_bars["resemble-enhance"].value() == 100
    assert page._overall_progress_bar.value() == 75  # (50 + 100) / 2

    # Simulate download finished
    page.on_download_finished()
    assert page.isComplete()
    assert page._overall_progress_bar.value() == 100
    assert page._error_frame.isHidden()
    assert "successfully" in page._status_label.text()


def test_model_download_page_failure_and_retry(qtbot):
    mock_downloader = MagicMock()
    
    mock_specs = [
        MagicMock(name="BS-RoFormer", filename="bs.ckpt"),
        MagicMock(name="resemble-enhance", filename="re.pth"),
    ]
    mock_specs[0].name = "BS-RoFormer"
    mock_specs[1].name = "resemble-enhance"
    
    with patch("app.ui.setup_wizard.REQUIRED_MODEL_SPECS", mock_specs):
        page = ModelDownloadPage(downloader=mock_downloader)
        qtbot.addWidget(page)

    with patch.object(page, "on_start_download") as mock_start:
        page.on_download_failed("Checksum mismatch for BS-RoFormer", retryable=True)

        assert not page.isComplete()
        assert not page._error_frame.isHidden()
        assert "Checksum mismatch for BS-RoFormer" in page._error_label.text()
        assert not page._retry_button.isHidden()

        page.on_retry_clicked()
        mock_start.assert_called_once()


def test_completion_page_launch_button(qtbot):
    wizard = QWizard()
    page = CompletionPage()
    wizard.addPage(page)
    qtbot.addWidget(wizard)

    wizard.show()
    page.initializePage()
    assert wizard.buttonText(QWizard.WizardButton.FinishButton) == "Launch App"

    with patch.object(wizard, "accept") as mock_accept:
        page.on_launch_clicked()
        mock_accept.assert_called_once()
