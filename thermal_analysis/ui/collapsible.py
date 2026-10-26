"""
CollapsibleSection — a titled section whose body can be folded away.
"""

from __future__ import annotations

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QFrame, QSizePolicy
from PyQt6.QtCore import Qt, QSettings, pyqtSignal


class CollapsibleSection(QWidget):
    """Header button (arrow + title + optional summary) above a foldable body."""

    toggled = pyqtSignal(bool)   # True when expanded

    def __init__(self, title: str, expanded: bool = False,
                 settings_key: str = "", parent=None):
        super().__init__(parent)
        self._title = title
        self._summary = ""
        self._key = f"sections/{settings_key}" if settings_key else ""

        if self._key:
            stored = QSettings("Glod", "glod").value(self._key, None)
            if stored is not None:
                expanded = str(stored).lower() == "true"

        self._header = QPushButton()
        self._header.setObjectName("SectionHeader")
        self._header.setCheckable(True)
        self._header.setChecked(expanded)
        self._header.setCursor(Qt.CursorShape.PointingHandCursor)
        self._header.setSizePolicy(QSizePolicy.Policy.Expanding,
                                   QSizePolicy.Policy.Fixed)
        self._header.clicked.connect(self._on_clicked)

        self._body = QFrame()
        self._body.setObjectName("SectionBody")
        self._body_layout = QVBoxLayout(self._body)
        self._body_layout.setContentsMargins(8, 4, 8, 8)
        self._body_layout.setSpacing(4)
        self._body.setVisible(expanded)

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)
        root.addWidget(self._header)
        root.addWidget(self._body)

        self._refresh_header()

    # ── API ───────────────────────────────────────────────────────────────

    def body_layout(self) -> QVBoxLayout:
        return self._body_layout

    def add_widget(self, widget: QWidget) -> None:
        self._body_layout.addWidget(widget)

    def set_summary(self, text: str) -> None:
        """One-line description shown next to the title while collapsed."""
        self._summary = text
        self._refresh_header()

    def is_expanded(self) -> bool:
        return self._header.isChecked()

    def set_expanded(self, expanded: bool) -> None:
        self._header.setChecked(expanded)
        self._on_clicked(expanded)

    # ── Internals ─────────────────────────────────────────────────────────

    def _on_clicked(self, checked: bool) -> None:
        self._body.setVisible(checked)
        self._refresh_header()
        if self._key:
            QSettings("Glod", "glod").setValue(self._key, checked)
        self.toggled.emit(checked)

    def _refresh_header(self) -> None:
        arrow = "▾" if self._header.isChecked() else "▸"
        text = f"{arrow}  {self._title}"
        if self._summary and not self._header.isChecked():
            text += f"   ·  {self._summary}"
        self._header.setText(text.replace("&", "&&"))   # '&' would become a mnemonic
