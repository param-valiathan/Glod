"""
FolderGroupWidget — one card per experimental group in the left control panel.
"""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QVBoxLayout, QLabel, QLineEdit,
    QPushButton, QFileDialog, QFrame, QCheckBox, QColorDialog,
    QSizePolicy, QComboBox, QDoubleSpinBox,
)
from PyQt6.QtGui import QColor
from PyQt6.QtCore import pyqtSignal, Qt

from ..processing.roi_analysis import emissivity_offset_c
from ..utils.coat_presets import (CUSTOM_LABEL, DEFAULT_PRESET, PRESETS,
                                  emissivity_for, note_for)


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
    changed = pyqtSignal()              # name or folder changed (animal matching may differ)

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
        self._name_edit.editingFinished.connect(self.changed.emit)

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

        # Animal / coat → emissivity (see utils/coat_presets.py for the literature basis)
        self._coat_combo = QComboBox()
        self._coat_combo.addItems([lbl for lbl, _, _ in PRESETS] + [CUSTOM_LABEL])
        self._coat_combo.setCurrentText(DEFAULT_PRESET)
        self._coat_combo.setFixedHeight(26)
        self._eps_custom = QDoubleSpinBox()
        self._eps_custom.setRange(0.50, 1.00)
        self._eps_custom.setDecimals(2)
        self._eps_custom.setSingleStep(0.01)
        self._eps_custom.setValue(0.95)
        self._eps_custom.setPrefix("ε ")
        self._eps_custom.setVisible(False)
        self._coat_info = QLabel("")
        self._coat_info.setWordWrap(True)
        self._coat_info.setStyleSheet("color: #6C757D; font-size: 10px;")
        self._coat_combo.currentTextChanged.connect(self._coat_changed)
        self._eps_custom.valueChanged.connect(lambda _v: self._coat_changed())

        remove_btn = QPushButton("✕")
        remove_btn.setObjectName("DangerButton")
        remove_btn.setFixedSize(28, 28)
        remove_btn.setStyleSheet("padding: 0px; min-height: 0px; min-width: 0px;")
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

        # Row 4: animal / coat selector
        row4 = QHBoxLayout()
        row4.addWidget(QLabel("Animal / coat"))
        row4.addWidget(self._coat_combo, 1)
        row4.addWidget(self._eps_custom)
        root.addLayout(row4)
        root.addWidget(self._coat_info)

        self._folder_path: str = ""
        self._coat_changed()

    # ── Public API ────────────────────────────────────────────────────────

    @property
    def group_name(self) -> str:
        return self._name_edit.text().strip() or f"Group"

    @property
    def color(self) -> str:
        return self._color_btn.color

    @property
    def coat_label(self) -> str:
        return (f"Custom ε = {self._eps_custom.value():.2f}"
                if self._coat_combo.currentText() == CUSTOM_LABEL
                else self._coat_combo.currentText())

    @property
    def emissivity(self) -> float:
        return emissivity_for(self._coat_combo.currentText(), self._eps_custom.value())

    def _coat_changed(self, *_):
        custom = self._coat_combo.currentText() == CUSTOM_LABEL
        self._eps_custom.setVisible(custom)
        off = emissivity_offset_c(self.emissivity)
        self._coat_info.setText(
            f"ε = {self.emissivity:.2f} · shifts a 30.5 °C reading by {off:+.2f} °C "
            f"(camera assumes 0.95). " + note_for(self._coat_combo.currentText()))
        self.changed.emit()

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

    @staticmethod
    def describe_folder(folder: str) -> str:
        """"📁 name (N recordings · K session(s))" — sessions = distinct recording dates."""
        from ..utils.naming import split_stem
        p = Path(folder)
        files = list(p.glob("*.txt")) + list(p.glob("*/*.txt"))
        dates = {split_stem(f.stem)[1] for f in files} - {None}
        s = f" · {len(dates)} session(s)" if dates else ""
        return f"📁 {p.name}  ({len(files)} recording(s){s})"

    def is_valid(self) -> bool:
        return bool(self._folder_path) and Path(self._folder_path).is_dir()

    # ── Slots ─────────────────────────────────────────────────────────────

    def _browse(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Experiment Folder", self._folder_path or "")
        if folder:
            self._folder_path = folder
            self._path_label.setText(self.describe_folder(folder))
            self.changed.emit()
