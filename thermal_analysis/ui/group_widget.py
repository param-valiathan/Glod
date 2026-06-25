"""
FolderGroupWidget — one card per experimental group in the left control panel.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QFrame, QCheckBox, QColorDialog,
    QSizePolicy,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal, Qt


_DEFAULT_COLORS = [
    "#2196F3", "#E53935", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#6D4C41", "#1E88E5",
]
_color_counter = 0


class ColorButton(QPushButton):
    """Small square button that shows the current colour and opens a picker."""

    color_changed = pyqtSignal(str)

    def __init__(self, initial: str = "#2196F3", parent=None):
        super().__init__(parent)
        self.setFixedSize(28, 28)
        self.setObjectName("SmallButton")
        self._color = initial
        self._update_swatch()
        self.clicked.connect(self._pick_color)

    @property
    def color(self) -> str:
        return self._color

    def set_color(self, hex_color: str):
        self._color = hex_color
        self._update_swatch()

    def _update_swatch(self):
        self.setStyleSheet(
            f"QPushButton {{ background-color: {self._color}; "
            f"border: 2px solid #DEE2E6; border-radius: 4px; }}"
            f"QPushButton:hover {{ border-color: #5C8A6B; }}"
        )

    def _pick_color(self):
        dlg = QColorDialog(QColor(self._color), self)
        if dlg.exec():
            chosen = dlg.currentColor().name()
            self._color = chosen
            self._update_swatch()
            self.color_changed.emit(chosen)


class FolderGroupWidget(QFrame):
    """
    One experiment group card. Contains:
      - group name field
      - colour picker
      - folder path display + Browse button
      - Export video checkbox
      - Remove button
    """

    removed = pyqtSignal(object)        # emits self
    color_changed = pyqtSignal(object)  # emits self (so MainWindow can look up group_name + color)

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        global _color_counter
        default_color = _DEFAULT_COLORS[_color_counter % len(_DEFAULT_COLORS)]
        _color_counter += 1

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("GroupCard")
        self.setStyleSheet(
            "QFrame#GroupCard { background: #FFFFFF; border: 1.5px solid #DEE2E6; "
            "border-radius: 6px; padding: 4px; }"
        )

        # ── Widgets ───────────────────────────────────────────────────────
        self._name_edit = QLineEdit(f"Group {index}")
        self._name_edit.setPlaceholderText("Group name…")
        self._name_edit.setFixedHeight(28)

        self._color_btn = ColorButton(default_color)
        self._color_btn.color_changed.connect(lambda _c: self.color_changed.emit(self))

        self._path_label = QLabel("No folder selected")
        self._path_label.setObjectName("SubtitleLabel")
        self._path_label.setWordWrap(True)
        self._path_label.setStyleSheet("color: #6C757D; font-size: 11px;")
        self._path_label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                        QSizePolicy.Policy.Preferred)

        browse_btn = QPushButton("Browse…")
        browse_btn.setObjectName("SecondaryButton")
        browse_btn.setFixedHeight(26)
        browse_btn.clicked.connect(self._browse)

        self._video_cb = QCheckBox("Export video")
        self._video_cb.setChecked(False)

        remove_btn = QPushButton("✕")
        remove_btn.setObjectName("DangerButton")
        remove_btn.setFixedSize(26, 26)
        remove_btn.setToolTip("Remove this group")
        remove_btn.clicked.connect(lambda: self.removed.emit(self))

        # ── Layout ────────────────────────────────────────────────────────
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(6)

        # Row 1: name + colour + remove
        row1 = QHBoxLayout()
        row1.addWidget(self._name_edit)
        row1.addWidget(self._color_btn)
        row1.addWidget(remove_btn)
        root.addLayout(row1)

        # Row 2: folder path
        root.addWidget(self._path_label)

        # Row 3: browse + video checkbox
        row3 = QHBoxLayout()
        row3.addWidget(browse_btn)
        row3.addStretch()
        row3.addWidget(self._video_cb)
        root.addLayout(row3)

        self._folder_path: str = ""

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def group_name(self) -> str:
        return self._name_edit.text().strip() or f"Group"

    @property
    def color(self) -> str:
        return self._color_btn.color

    @property
    def folder_path(self) -> str:
        return self._folder_path

    @property
    def export_video(self) -> bool:
        return self._video_cb.isChecked()

    def get_txt_files(self) -> list[str]:
        """Return sorted .txt files in the selected folder and one level of subdirs."""
        if not self._folder_path:
            return []
        p = Path(self._folder_path)
        files = sorted(p.glob("*.txt")) + sorted(p.glob("*/*.txt"))
        return [str(f) for f in files]

    def is_valid(self) -> bool:
        return bool(self._folder_path) and Path(self._folder_path).is_dir()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Experiment Folder", self._folder_path or "")
        if folder:
            self._folder_path = folder
            txt_count = len(list(Path(folder).glob("*.txt")))
            self._path_label.setText(f"📁 {Path(folder).name}  ({txt_count} .txt file(s))")
