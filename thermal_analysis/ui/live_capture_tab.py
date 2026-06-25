"""
LiveCaptureTab — the "Live Capture" tab added to the right panel of MainWindow.

Layout
------
  Horizontal splitter:
    Left  (280 px) : controls (detect, settings, groups, record/stop, status)
    Right           : QScrollArea grid of CameraPanel widgets (2 columns)
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from pathlib import Path

from PyQt6.QtCore import Qt, QSettings, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFrame,
    QGridLayout, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSpinBox, QSplitter, QTextEdit, QVBoxLayout, QWidget,
    QFileDialog, QApplication,
)
from PyQt6.QtGui import QColor

from .camera_panel import CameraPanel
from ..capture.camera_discovery import list_cameras, list_all_serial_ports, probe
from ..capture.capture_worker import CaptureWorker, ram_warning
from ..capture.recording_manager import RecordingManager
from ..utils.animal_registry import AnimalRegistry, AnimalConfig
from ..utils.config import (
    KNOWN_CAMERA_VIDS, RECORDINGS_SUBDIR,
    T_MIN_DEFAULT, T_MAX_DEFAULT, COLORMAP_DEFAULT,
)

log = logging.getLogger(__name__)

_SETTINGS_KEY_PORTS = "camera_ports"


def _h_rule() -> QFrame:
    """Thin horizontal divider line for use inside layouts."""
    f = QFrame()
    f.setFrameShape(QFrame.Shape.HLine)
    f.setFrameShadow(QFrame.Shadow.Plain)
    f.setStyleSheet("color: #DEE2E6; max-height: 1px; margin: 2px 0;")
    return f

_GROUP_COLORS = [
    "#2196F3", "#E53935", "#43A047", "#FB8C00",
    "#8E24AA", "#00ACC1", "#6D4C41", "#1E88E5",
]


class LiveCaptureTab(QWidget):
    """Main widget for the Live Capture tab."""

    send_to_analysis = pyqtSignal(str)   # emits path of recordings/ folder

    def __init__(
        self,
        registry: AnimalRegistry,
        output_dir: str = "",
        t_min: float = T_MIN_DEFAULT,
        t_max: float = T_MAX_DEFAULT,
        colormap: str = COLORMAP_DEFAULT,
        parent=None,
    ):
        super().__init__(parent)
        self._registry = registry
        self._output_dir = output_dir
        self._t_min = t_min
        self._t_max = t_max
        self._colormap = colormap

        self._panels: dict[str, CameraPanel] = {}
        self._preview_workers: dict[str, CaptureWorker] = {}
        self._manager = RecordingManager(parent=self)
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_timer)
        self._elapsed_s = 0

        self._build_ui()
        self._connect_signals()
        self._load_remembered_ports()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_control_strip())
        splitter.addWidget(self._build_camera_area())
        splitter.setSizes([280, 900])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _build_control_strip(self) -> QWidget:
        panel = QFrame()
        panel.setFixedWidth(280)
        panel.setObjectName("ControlPanel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        # ── Detect ─────────────────────────────────────────────────────────
        self._detect_btn = QPushButton("  Detect Cameras")
        self._detect_btn.setObjectName("SecondaryButton")
        self._detect_btn.setFixedHeight(32)
        self._detect_btn.clicked.connect(self._detect_cameras)
        layout.addWidget(self._detect_btn)

        self._detect_lbl = QLabel("No cameras detected.")
        self._detect_lbl.setWordWrap(True)
        self._detect_lbl.setStyleSheet("font-size: 11px; color: #6C757D; padding: 0 2px;")
        layout.addWidget(self._detect_lbl)

        # ── Preview ────────────────────────────────────────────────────────
        self._preview_btn = QPushButton("▶  Live Preview")
        self._preview_btn.setObjectName("SecondaryButton")
        self._preview_btn.setFixedHeight(32)
        self._preview_btn.setEnabled(False)
        self._preview_btn.setToolTip(
            "Start live feed from cameras without saving to disk.\n"
            "Green dot = frames arriving.  Press again to stop.")
        self._preview_btn.clicked.connect(self._toggle_preview)
        layout.addWidget(self._preview_btn)

        layout.addWidget(_h_rule())

        # ── Capture settings ───────────────────────────────────────────────
        sec1 = QLabel("CAPTURE SETTINGS")
        sec1.setObjectName("DividerLabel")
        layout.addWidget(sec1)
        layout.addWidget(self._build_settings_group())

        layout.addWidget(_h_rule())

        # ── Groups ─────────────────────────────────────────────────────────
        sec2 = QLabel("GROUPS")
        sec2.setObjectName("DividerLabel")
        layout.addWidget(sec2)
        layout.addWidget(self._build_groups_group())

        layout.addWidget(_h_rule())

        # ── Record / Stop  (full-width, prominent) ─────────────────────────
        self._record_btn = QPushButton("⏺   Record")
        self._record_btn.setObjectName("RecordButton")
        self._record_btn.clicked.connect(self._start_recording)
        layout.addWidget(self._record_btn)

        self._stop_btn = QPushButton("⏹   Stop Recording")
        self._stop_btn.setObjectName("StopRecordButton")
        self._stop_btn.setVisible(False)
        self._stop_btn.clicked.connect(self._stop_recording)
        layout.addWidget(self._stop_btn)

        self._status_lbl = QLabel("Idle")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._status_lbl.setStyleSheet("font-size: 11px; color: #6C757D; padding: 2px 0;")
        layout.addWidget(self._status_lbl)

        self._send_btn = QPushButton("→  Send to Analysis")
        self._send_btn.setObjectName("SecondaryButton")
        self._send_btn.setEnabled(False)
        self._send_btn.clicked.connect(self._send_to_analysis)
        layout.addWidget(self._send_btn)

        layout.addStretch()

        # ── Camera log ─────────────────────────────────────────────────────
        log_lbl = QLabel("CAMERA LOG")
        log_lbl.setObjectName("DividerLabel")
        layout.addWidget(log_lbl)

        self._inline_log = QTextEdit()
        self._inline_log.setReadOnly(True)
        self._inline_log.setFixedHeight(100)
        self._inline_log.setToolTip("Camera connection and error log")
        layout.addWidget(self._inline_log)

        return panel

    def _build_settings_group(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        fps_row = QHBoxLayout()
        fps_lbl = QLabel("FPS:")
        fps_lbl.setFixedWidth(80)
        fps_row.addWidget(fps_lbl)
        self._fps_spin = QSpinBox()
        self._fps_spin.setRange(1, 25)
        self._fps_spin.setValue(5)
        self._fps_spin.setFixedWidth(64)
        self._fps_spin.setToolTip(
            "Target FPS per camera. Keep ≤5 on a USB hub to avoid\n"
            "bus contention. Each camera staggers its start by 0.3 s.")
        fps_row.addWidget(self._fps_spin)
        fps_row.addWidget(QLabel("(≤5 on USB hub)"))
        fps_row.addStretch()
        layout.addLayout(fps_row)

        dur_row = QHBoxLayout()
        dur_lbl = QLabel("Duration (s):")
        dur_lbl.setFixedWidth(80)
        dur_row.addWidget(dur_lbl)
        self._dur_spin = QSpinBox()
        self._dur_spin.setRange(10, 7200)
        self._dur_spin.setValue(300)
        self._dur_spin.setFixedWidth(64)
        dur_row.addWidget(self._dur_spin)
        dur_row.addStretch()
        layout.addLayout(dur_row)

        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output:"))
        self._out_lbl = QLabel(self._output_dir or "(not set)")
        self._out_lbl.setStyleSheet("font-size: 10px; color: #6C757D;")
        self._out_lbl.setWordWrap(True)
        out_row.addWidget(self._out_lbl, stretch=1)
        browse_btn = QPushButton("…")
        browse_btn.setObjectName("SmallButton")
        browse_btn.setFixedSize(26, 24)
        browse_btn.clicked.connect(self._browse_output)
        out_row.addWidget(browse_btn)
        layout.addLayout(out_row)

        return w

    def _build_groups_group(self) -> QWidget:
        w = QWidget()
        layout = QVBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self._group_list = QListWidget()
        self._group_list.setFixedHeight(78)
        layout.addWidget(self._group_list)

        btn_row = QHBoxLayout()
        add_btn = QPushButton("✚ Add Group")
        add_btn.setObjectName("SecondaryButton")
        add_btn.setFixedHeight(28)
        add_btn.clicked.connect(self._add_group)
        btn_row.addWidget(add_btn)

        rm_btn = QPushButton("✕ Remove")
        rm_btn.setObjectName("DangerButton")
        rm_btn.setFixedHeight(28)
        rm_btn.clicked.connect(self._remove_group)
        btn_row.addWidget(rm_btn)
        layout.addLayout(btn_row)

        return w

    def _build_camera_area(self) -> QWidget:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._grid_widget = QWidget()
        self._grid_widget.setStyleSheet("background: #F5F5F5;")
        self._grid_layout = QGridLayout(self._grid_widget)
        self._grid_layout.setContentsMargins(8, 8, 8, 8)
        self._grid_layout.setSpacing(8)

        scroll.setWidget(self._grid_widget)
        outer_layout.addWidget(scroll)

        self._no_cam_lbl = QLabel(
            "No cameras detected.\n\nClick \"Detect Cameras\" to scan for connected "
            "Waveshare MI48x3 USB thermal cameras.\n\n"
            "If nothing is found, run find_camera_ports.py to identify ports manually."
        )
        self._no_cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_cam_lbl.setStyleSheet("color: #999; font-size: 12px; padding: 40px;")
        self._grid_layout.addWidget(self._no_cam_lbl, 0, 0, 1, 2)

        return outer

    # ── Signal connections ─────────────────────────────────────────────────

    def _connect_signals(self):
        self._manager.log_message.connect(self._log)
        self._manager.all_saved.connect(self._on_all_saved)
        self._manager.save_error.connect(
            lambda msg: QMessageBox.warning(self, "Save Error", msg))
        self._manager.camera_error.connect(
            lambda cam, msg: self._log(f"✗ [{cam}] {msg}"))
        self._registry.groups_changed.connect(self._refresh_group_list)

    # ── Camera detection ───────────────────────────────────────────────────

    def _detect_cameras(self):
        self._detect_btn.setEnabled(False)
        self._detect_lbl.setText("Scanning…")
        QApplication.processEvents()

        cameras = list_cameras(known_vids=KNOWN_CAMERA_VIDS)

        if not cameras:
            cameras = self._manual_selection_dialog()

        if cameras:
            self._populate_panels(cameras)
            self._remember_ports([c["id"] for c in cameras])
            self._detect_lbl.setText(f"Found {len(cameras)} camera(s).")
            self._detect_lbl.setStyleSheet("font-size: 11px; color: #43A047;")
        else:
            self._detect_lbl.setText("No cameras selected.")
            self._detect_lbl.setStyleSheet("font-size: 11px; color: #E53935;")

        self._detect_btn.setEnabled(True)

    def _manual_selection_dialog(self) -> list[dict]:
        all_ports = list_all_serial_ports()
        if not all_ports:
            QMessageBox.information(
                self, "No Ports",
                "No COM ports found at all.\n\nMake sure the camera is plugged in "
                "and the USB driver is installed.\n\nRun find_camera_ports.py "
                "for diagnostics."
            )
            return []

        dlg = _PortSelectionDialog(all_ports, parent=self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            selected = dlg.selected_ports()
            return [{"id": p, "label": f"Manual ({p})", "source": "manual"}
                    for p in selected]
        return []

    def _populate_panels(self, cameras: list[dict]):
        existing_ids = {c["id"] for c in cameras}
        for cam_id in list(self._panels.keys()):
            if cam_id not in existing_ids:
                panel = self._panels.pop(cam_id)
                panel.setParent(None)
                panel.deleteLater()

        groups = self._registry.groups()
        for cam in cameras:
            if cam["id"] not in self._panels:
                panel = CameraPanel(
                    camera_id=cam["id"],
                    groups=groups,
                    fps=self._fps_spin.value(),
                    t_min=self._t_min,
                    t_max=self._t_max,
                    colormap=self._colormap,
                    parent=self,
                )
                panel.assignment_changed.connect(self._on_assignment_changed)
                self._panels[cam["id"]] = panel

        if self._panels and self._no_cam_lbl is not None:
            self._grid_layout.removeWidget(self._no_cam_lbl)
            self._no_cam_lbl.setVisible(False)
            self._no_cam_lbl = None
        elif not self._panels and self._no_cam_lbl is not None:
            self._no_cam_lbl.setVisible(True)

        for i, (cam_id, panel) in enumerate(self._panels.items()):
            row, col = divmod(i, 2)
            self._grid_layout.addWidget(panel, row, col)

        self._preview_btn.setEnabled(bool(self._panels))

        # Probe non-auto cameras in background daemon threads (avoids blocking the GUI)
        for cam in cameras:
            if cam["source"] != "auto":
                self._probe_async(cam["id"])

    # ── Live preview (no recording) ────────────────────────────────────────

    def _toggle_preview(self):
        if self._preview_workers:
            self._stop_preview()
        else:
            self._start_preview()

    def _start_preview(self):
        # Enumerate ports on the main thread (same pattern as RecordingManager)
        port_map: dict = {}
        try:
            from senxor import list_senxor  # type: ignore[import]
            all_ports = list_senxor("serial")
            port_map = {p.device: p for p in all_ports}
            self._log(f"Preview: found ports {list(port_map.keys())}")
        except Exception as exc:
            self._log(f"Preview: camera enumeration failed — {exc}")

        fps = self._fps_spin.value()
        for i, (cam_id, panel) in enumerate(self._panels.items()):
            panel.set_status_state("connecting")
            w = CaptureWorker(
                camera_id=cam_id,
                target_fps=fps,
                duration_s=86400,          # runs until stopped
                start_delay_s=i * 0.3,
                port_obj=port_map.get(cam_id),
                preview_only=True,
                parent=self,
            )
            w.frame_ready.connect(
                lambda cid, frame, elapsed, p=panel: p.on_frame(frame, elapsed))
            w.connected.connect(
                lambda cid, p=panel: p.set_status_state("streaming"))
            w.error.connect(
                lambda cid, msg, p=panel: (
                    p.set_status_state("error"),
                    self._log(f"✗ [{cid}] {msg}"),
                ))
            w.log_message.connect(self._log)
            self._preview_workers[cam_id] = w
            w.start()

        self._preview_btn.setText("■  Stop Preview")
        self._record_btn.setEnabled(False)

    def _stop_preview(self):
        for w in self._preview_workers.values():
            w.stop()
        for w in self._preview_workers.values():
            w.wait(3000)
        self._preview_workers.clear()
        for panel in self._panels.values():
            panel.set_status_state("idle")
        self._preview_btn.setText("▶  Preview")
        self._preview_btn.setEnabled(bool(self._panels))
        self._record_btn.setEnabled(True)

    def _probe_async(self, cam_id: str):
        """Probe a camera port in a background daemon thread."""
        def _do():
            ok = probe(cam_id)
            self._log(
                f"[{cam_id}] Probe "
                f"{'OK — camera responds' if ok else 'FAILED — check connection'}")
        threading.Thread(target=_do, daemon=True).start()

    # ── Recording ──────────────────────────────────────────────────────────

    def _start_recording(self):
        if not self._panels:
            QMessageBox.warning(self, "No Cameras", "Detect cameras first.")
            return
        if not self._output_dir:
            QMessageBox.warning(self, "No Output Dir",
                                "Set an output directory first.")
            return

        # Stop preview so the COM ports are free for recording workers
        if self._preview_workers:
            self._log("Stopping preview before recording…")
            self._stop_preview()

        configs = []
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, (cam_id, panel) in enumerate(self._panels.items()):
            name = panel._animal_edit.text().strip()
            if not name or name == cam_id:
                name = f"{_ts}_{i + 1:02d}"
                panel._animal_edit.setText(name)
            if not panel.group_name:
                QMessageBox.warning(
                    self, "Missing Group",
                    f"Camera {cam_id} has no group assigned.")
                return
            configs.append(AnimalConfig(
                camera_id=cam_id,
                animal_name=name,
                group_name=panel.group_name,
                group_color=self._registry.group_color(panel.group_name),
            ))

        fps = self._fps_spin.value()
        duration = self._dur_spin.value()

        warn = ram_warning(len(configs), fps, duration)
        if warn:
            ret = QMessageBox.warning(
                self, "RAM Warning", warn + "\n\nContinue?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if ret != QMessageBox.StandardButton.Yes:
                return

        if len(configs) > 1:
            self._log(
                f"Multiple cameras ({len(configs)}): starts staggered by 0.3 s each "
                "to reduce USB hub contention.")

        for panel in self._panels.values():
            panel.reset()
            panel.set_status(True)

        # prepare() creates workers but does NOT start threads yet.
        # Connect frame_ready signals before start() so no frames are missed.
        self._manager.prepare(configs, fps, duration, self._output_dir)

        for cam_id, worker in self._manager._workers.items():
            panel = self._panels.get(cam_id)
            if panel:
                worker.frame_ready.connect(
                    lambda cid, frame, elapsed, p=panel:
                    p.on_frame(frame, elapsed)
                )

        # Start threads now that signals are all wired
        self._manager.start()

        self._record_btn.setVisible(False)
        self._stop_btn.setVisible(True)
        self._preview_btn.setEnabled(False)
        self._elapsed_s = 0
        self._elapsed_timer.start()
        self._send_btn.setEnabled(False)

    def _stop_recording(self):
        self._manager.stop_all()
        self._elapsed_timer.stop()
        self._status_lbl.setText("Stopping…")
        self._stop_btn.setEnabled(False)

    @pyqtSlot(list)
    def _on_all_saved(self, paths: list):
        self._elapsed_timer.stop()
        self._record_btn.setVisible(True)
        self._stop_btn.setVisible(False)
        self._stop_btn.setEnabled(True)
        self._preview_btn.setEnabled(bool(self._panels))
        for panel in self._panels.values():
            panel.set_status(False)
        n = len(paths)
        self._status_lbl.setText(f"Done — {n} file(s) saved.")
        self._send_btn.setEnabled(n > 0)
        self._log(f"✓ Recording complete. {n} file(s) saved.")

    def _tick_timer(self):
        self._elapsed_s += 1
        dur = self._dur_spin.value()
        m, s = divmod(self._elapsed_s, 60)
        dm, ds = divmod(dur, 60)
        self._status_lbl.setText(f"Recording {m}:{s:02d} / {dm}:{ds:02d}")

    # ── Groups ─────────────────────────────────────────────────────────────

    def _add_group(self):
        from PyQt6.QtWidgets import QInputDialog
        name, ok = QInputDialog.getText(self, "Add Group", "Group name:")
        if ok and name.strip():
            name = name.strip()
            color = _GROUP_COLORS[len(self._registry.groups()) % len(_GROUP_COLORS)]
            self._registry.add_group(name, color)

    def _remove_group(self):
        item = self._group_list.currentItem()
        if item:
            self._registry.remove_group(item.text())

    def _refresh_group_list(self):
        self._group_list.clear()
        for name, color in self._registry.groups():
            item = QListWidgetItem(name)
            item.setForeground(QColor(color))
            self._group_list.addItem(item)
        groups = self._registry.groups()
        for panel in self._panels.values():
            panel.update_groups(groups)

    # ── Assignments ────────────────────────────────────────────────────────

    def _on_assignment_changed(self, cam_id: str, animal: str, group: str):
        self._registry.assign(AnimalConfig(
            camera_id=cam_id,
            animal_name=animal,
            group_name=group,
            group_color=self._registry.group_color(group),
        ))

    # ── Helpers ────────────────────────────────────────────────────────────

    def _browse_output(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._output_dir or "")
        if folder:
            self._output_dir = folder
            self._out_lbl.setText(folder)

    def _send_to_analysis(self):
        recordings_path = str(Path(self._output_dir) / RECORDINGS_SUBDIR)
        self.send_to_analysis.emit(recordings_path)

    def _log(self, msg: str):
        log.info(msg)
        self._inline_log.append(msg)
        sb = self._inline_log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _remember_ports(self, ports: list[str]):
        s = QSettings("Glod", "glod")
        s.setValue(_SETTINGS_KEY_PORTS, ports)

    def _load_remembered_ports(self):
        s = QSettings("Glod", "glod")
        ports = s.value(_SETTINGS_KEY_PORTS, [])
        if ports:
            cameras = [{"id": p, "label": f"Remembered ({p})", "source": "filter"}
                       for p in ports]
            self._populate_panels(cameras)
            self._detect_lbl.setText(
                f"Loaded {len(cameras)} remembered port(s). Click Detect to refresh.")
            self._detect_lbl.setStyleSheet("font-size: 11px; color: #6C757D;")

    def update_display_settings(self, t_min: float, t_max: float, colormap: str):
        self._t_min = t_min
        self._t_max = t_max
        self._colormap = colormap
        fps = self._fps_spin.value()
        for panel in self._panels.values():
            panel.update_settings(fps, t_min, t_max, colormap)

    def set_output_dir(self, path: str):
        self._output_dir = path
        self._out_lbl.setText(path or "(not set)")


# ── Manual port selection dialog ───────────────────────────────────────────────

class _PortSelectionDialog(QDialog):
    def __init__(self, ports: list[dict], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Camera Ports")
        self.setMinimumWidth(360)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "No cameras were auto-detected.\n"
            "Tick the COM ports that are thermal cameras:"
        ))

        self._checks: list[tuple[QCheckBox, str]] = []
        for p in ports:
            cb = QCheckBox(p["label"])
            self._checks.append((cb, p["id"]))
            layout.addWidget(cb)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def selected_ports(self) -> list[str]:
        return [port_id for cb, port_id in self._checks if cb.isChecked()]
