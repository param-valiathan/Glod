"""
AnimalRegistry — shared group/animal state between LiveCaptureTab and MainWindow.

MainWindow holds one singleton instance.  LiveCaptureTab writes to it;
the Analysis panel reads via the groups_changed signal.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from PyQt6.QtCore import QObject, pyqtSignal


@dataclass
class AnimalConfig:
    camera_id: str        # e.g. "COM3"
    animal_name: str
    group_name: str
    group_color: str = "#2196F3"


class AnimalRegistry(QObject):
    """Holds groups and per-camera animal assignments for a session."""

    groups_changed = pyqtSignal()   # emitted on add/remove/rename

    def __init__(self, parent=None):
        super().__init__(parent)
        self._groups: list[tuple[str, str]] = []   # (name, color)
        self._animals: list[AnimalConfig] = []

    # ── Groups ─────────────────────────────────────────────────────────────

    def add_group(self, name: str, color: str = "#2196F3") -> None:
        if not any(g[0] == name for g in self._groups):
            self._groups.append((name, color))
            self.groups_changed.emit()

    def remove_group(self, name: str) -> None:
        self._groups = [g for g in self._groups if g[0] != name]
        self._animals = [a for a in self._animals if a.group_name != name]
        self.groups_changed.emit()

    def set_group_color(self, name: str, color: str) -> None:
        self._groups = [
            (n, color if n == name else c) for n, c in self._groups
        ]
        for a in self._animals:
            if a.group_name == name:
                a.group_color = color
        self.groups_changed.emit()

    def groups(self) -> list[tuple[str, str]]:
        return list(self._groups)

    def group_color(self, name: str) -> str:
        for n, c in self._groups:
            if n == name:
                return c
        return "#2196F3"

    # ── Animals ────────────────────────────────────────────────────────────

    def assign(self, config: AnimalConfig) -> None:
        self._animals = [a for a in self._animals if a.camera_id != config.camera_id]
        self._animals.append(config)

    def unassign(self, camera_id: str) -> None:
        self._animals = [a for a in self._animals if a.camera_id != camera_id]

    def animals(self) -> list[AnimalConfig]:
        return list(self._animals)

    def get_animal(self, camera_id: str) -> Optional[AnimalConfig]:
        return next((a for a in self._animals if a.camera_id == camera_id), None)

    def clear(self) -> None:
        self._animals.clear()
        self._groups.clear()
        self.groups_changed.emit()
