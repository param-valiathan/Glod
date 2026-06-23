"""
Glöd — Entry point.
Swedish: glow / live embers.

Usage:
    python -m thermal_analysis.main
    OR: python thermal_analysis/main.py
"""

import sys
import logging
from pathlib import Path

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("glod")

# ── Qt app ────────────────────────────────────────────────────────────────────
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
from PyQt6.QtCore import Qt

from .ui.theme import apply_theme
from .ui.main_window import MainWindow


def main():
    # Enable High-DPI scaling
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)

    app = QApplication(sys.argv)
    app.setApplicationName("Glöd")
    app.setOrganizationName("Thermal Imaging Lab")
    app.setApplicationVersion("1.0")

    apply_theme(app)

    # Icon — look next to THIS file first, then parent directory
    icon_search = [
        Path(__file__).parent / "icon.png",
        Path(__file__).parent.parent / "icon.png",
    ]
    icon_path = next((str(p) for p in icon_search if p.exists()), None)
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))
        log.info("Icon loaded from %s", icon_path)

    window = MainWindow(icon_path=icon_path)
    window.show()

    log.info("Glöd started.")
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
