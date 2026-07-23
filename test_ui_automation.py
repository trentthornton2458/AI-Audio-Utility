import sys
import os
import time
from pathlib import Path
import numpy as np
import soundfile as sf

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt, QTimer
from PySide6.QtTest import QTest

from app.cache.cache_manager import CacheManager
from app.ui.setup_wizard import SetupWizard
from app.ui.main_window import MainWindow

def create_dummy_wav(filepath: Path):
    sr = 44100
    duration = 0.5
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = np.sin(2 * np.pi * 440 * t)[:, np.newaxis]  # Mono tone
    sf.write(str(filepath), audio, sr, subtype="PCM_16")
    print(f"Created dummy WAV file at {filepath}")

def main():
    print("Initializing QApplication...")
    app = QApplication(sys.argv)
    app.setApplicationName("Music Mastery Enhancer - UI Automation Test")

    # Create directory for screenshots
    screenshot_dir = Path("/home/jules/verification")
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    dummy_wav_path = Path("/tmp/dummy_input.wav")
    create_dummy_wav(dummy_wav_path)

    print("Step 1: Instantiating and displaying SetupWizard...")
    wizard = SetupWizard()
    wizard.show()
    QApplication.processEvents()
    time.sleep(0.5)

    # Welcome Page Screenshot
    print("Capturing Welcome Page...")
    wizard.grab().save(str(screenshot_dir / "01_welcome_page.png"))

    # Navigate to Hardware Check
    print("Navigating to Hardware Check Page...")
    wizard.next()
    QApplication.processEvents()
    time.sleep(0.5)
    wizard.grab().save(str(screenshot_dir / "02_hardware_check_page.png"))

    # Navigate to Model Download Page
    print("Navigating to Model Download Page...")
    wizard.next()
    QApplication.processEvents()
    time.sleep(0.5)
    wizard.grab().save(str(screenshot_dir / "03_model_download_page_before_skip.png"))

    # Click skip download button to avoid downloading large files during testing
    print("Clicking Skip Download button...")
    wizard.download_page._skip_button.click()
    QApplication.processEvents()
    time.sleep(0.5)
    wizard.grab().save(str(screenshot_dir / "04_model_download_page_after_skip.png"))

    # Navigate to Completion Page
    print("Navigating to Completion Page...")
    wizard.next()
    QApplication.processEvents()
    time.sleep(0.5)
    wizard.grab().save(str(screenshot_dir / "05_completion_page.png"))

    # Accept the wizard
    print("Accepting SetupWizard...")
    wizard.accept()
    QApplication.processEvents()

    print("Step 2: Instantiating and displaying MainWindow...")
    cache_manager = CacheManager()
    main_window = MainWindow(cache_manager=cache_manager)
    main_window.show()
    QApplication.processEvents()
    time.sleep(0.5)
    main_window.grab().save(str(screenshot_dir / "06_main_window_empty.png"))

    # Simulate selecting/ingesting the dummy track
    print("Simulating track selection/ingestion...")
    main_window.on_file_selected(dummy_wav_path)
    QApplication.processEvents()
    time.sleep(0.5)
    main_window.grab().save(str(screenshot_dir / "07_main_window_track_loaded.png"))

    # Transition to Vocal Controls Tab
    print("Switching to Vocal Controls tab...")
    main_window._tab_widget.setCurrentIndex(1)
    QApplication.processEvents()
    time.sleep(0.5)
    main_window.grab().save(str(screenshot_dir / "08_vocal_controls_tab.png"))

    # Transition to Instrumental Controls Tab
    print("Switching to Instrumental Controls tab...")
    main_window._tab_widget.setCurrentIndex(2)
    QApplication.processEvents()
    time.sleep(0.5)
    main_window.grab().save(str(screenshot_dir / "09_instrumental_controls_tab.png"))

    # Transition to A/B Compare Tab
    print("Switching to A/B Compare tab...")
    main_window._tab_widget.setCurrentIndex(3)
    QApplication.processEvents()
    time.sleep(0.5)
    main_window.grab().save(str(screenshot_dir / "10_ab_compare_tab.png"))

    print("UI automation completed successfully. Screenshots saved to /home/jules/verification/")
    main_window.close()
    app.quit()

if __name__ == "__main__":
    main()
