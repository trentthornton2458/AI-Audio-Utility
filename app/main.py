"""Application entry point for Music Mastery Enhancer.

Launches MainWindow, first showing the guided SetupWizard when no cached
model weights are present (i.e. this is the user's first run).
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication, QDialog

from app.cache import configure_logging, get_logger
from app.cache.cache_manager import CacheManager
from app.models.app_config import get_app_config
from app.ui.main_window import MainWindow
from app.ui.setup_wizard import SetupWizard

logger = get_logger(__name__)


def is_first_run() -> bool:
    """Detect first run via absence of the cache/models folder.

    Checked before a CacheManager is constructed, since CacheManager.__init__
    creates the models folder as a side effect.
    """
    config = get_app_config()
    models_dir = config.cache_root / CacheManager.MODELS_DIRNAME
    return not models_dir.is_dir()


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Music Mastery Enhancer")

    first_run = is_first_run()

    cache_manager = CacheManager()
    configure_logging(cache_manager)

    if first_run:
        logger.info("First run detected (no cache/models folder); launching setup wizard")
        wizard = SetupWizard()
        if wizard.exec() != QDialog.DialogCode.Accepted:
            logger.info("Setup wizard was not completed; exiting")
            sys.exit(0)

    window = MainWindow(cache_manager=cache_manager)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
