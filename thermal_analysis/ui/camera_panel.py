"""
CameraPanel — per-camera card for the Live Capture tab.

Displays:
  • Thermal heatmap  (pyqtgraph ImageItem, real-time)
  • Draggable ROI overlay (green RectROI — drag to reposition in Fixed mode;
    tracks hottest pixel in Dynamic mode)
  • Draggable Arena overlay (orange RectROI — drag to set the analysis region)
  • Live ROI-max + ROI-mean plot (60-second rolling window)
  • Per-camera colour-scaling controls (T min/max, colormap)
  • Animal name + group assignment controls
  • ROI mode + size controls
  • Arena enable + size controls (position set by dragging the orange box)
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QHBoxLayout, QLabel,
    QLineEdit, QSpinBox, QVBoxLayout, QWidget, QSizePolicy,
)

log = logging.getLogger(__name__)

try:
    import pyqtgraph as pg
    _HAS_PG = True
    pg.setConfigOption("background", "#1A1A2E")
    pg.setConfigOption("foreground", "#E0E0E0")
except ImportError:
    _HAS_PG = False
    log.warning("pyqtgraph not installed — live camera display unavailable.")

from ..utils.config import (
    CAMERA_WIDTH, CAMERA_HEIGHT, COLORMAPS,
    T_MIN_DEFAULT, T_MAX_DEFAULT, COLORMAP_DEFAULT,
)


class CameraPanel(QFrame):
    """
    One card per detected camera.

    Signals
    -------
    assignment_changed(camera_id, animal_name, group_name)
    """

    assignment_changed = pyqtSignal(str, str, str)

    PLOT_WINDOW_S = 60

    def __init__(
        self,
        camera_id: str,
        groups: list[tuple[str, str]],
        fps: int = 10,
        t_min: float = T_MIN_DEFAULT,
        t_max: float = T_MAX_DEFAULT,
        colormap: str = COLORMAP_DEFAULT,
        parent=None,
    ):
        super().__init__(parent)
        self._camera_id = camera_id
        self._fps = fps
        self._t_min = t_min
        self._t_max = t_max
        self._colormap_name = colormap

        self._frame_count = 0
        self._display_pending = False
        self._plot_buf: deque = deque(maxlen=fps * self.PLOT_WINDOW_S)

        self._img_item: Optional[object] = None
        self._roi_item: Optional[object] = None
        self._arena_item: Optional[object] = None
        self._plot_curve: Optional[object] = None
        self._plot_curve_mean: Optional[object] = None
        self._colormap_lut: Optional[np.ndarray] = None

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setObjectName("GroupCard")
        self.setStyleSheet(
            "QFrame#GroupCard { background: #FFFFFF; border: 1.5px solid #DEE2E6; "
            "border-radius: 6px; }"
        )
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._build_ui(groups)

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self, groups: list[tuple[str, str]]):
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 8, 10, 8)
        root.setSpacing(4)

        # Row 1: status • camera ID • group
        row1 = QHBoxLayout()
        row1.setSpacing(6)
        self._status_dot = QLabel("●")
        self._status_dot.setStyleSheet("color: #CCCCCC; font-size: 16px;")
        self._status_dot.setToolTip("Idle — not streaming")
        self._status_dot.setFixedWidth(20)
        row1.addWidget(self._status_dot)

        cam_lbl = QLabel(self._camera_id)
        cam_lbl.setStyleSheet("font-weight: 700; font-size: 13px; color: #212529;")
        row1.addWidget(cam_lbl)
        row1.addStretch()

        grp_lbl = QLabel("Group:")
        grp_lbl.setStyleSheet("font-size: 11px; color: #6C757D;")
        row1.addWidget(grp_lbl)
        self._group_combo = QComboBox()
        self._group_combo.setFixedWidth(100)
        for name, _ in groups:
            self._group_combo.addItem(name)
        self._group_combo.currentTextChanged.connect(self._emit_assignment)
        row1.addWidget(self._group_combo)
        root.addLayout(row1)

        # Row 2: animal # + name
        row2 = QHBoxLayout()
        row2.setSpacing(6)
        num_lbl = QLabel("Animal #")
        num_lbl.setStyleSheet("font-size: 11px; color: #6C757D;")
        row2.addWidget(num_lbl)
        self._animal_num = QSpinBox()
        self._animal_num.setRange(1, 99)
        self._animal_num.setFixedWidth(52)
        self._animal_num.setToolTip("Auto-fills Name field below")
        self._animal_num.valueChanged.connect(self._on_animal_num_changed)
        row2.addWidget(self._animal_num)
        name_lbl = QLabel("Name:")
        name_lbl.setStyleSheet("font-size: 11px; color: #6C757D;")
        row2.addWidget(name_lbl)
        self._animal_edit = QLineEdit()
        self._animal_edit.setPlaceholderText("Mouse_01  (auto if blank)")
        self._animal_edit.editingFinished.connect(self._emit_assignment)
        row2.addWidget(self._animal_edit, stretch=1)
        root.addLayout(row2)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #DEE2E6; margin: 2px 0;")
        root.addWidget(line)

        if _HAS_PG:
            root.addWidget(self._build_heatmap())
            root.addWidget(self._build_live_plot())
        else:
            no_pg = QLabel("Install pyqtgraph for live display\n(run setup_glod_env.bat)")
            no_pg.setAlignment(Qt.AlignmentFlag.AlignCenter)
            no_pg.setStyleSheet("color: #999; font-size: 11px; padding: 20px;")
            root.addWidget(no_pg)

        root.addWidget(self._build_roi_controls())
        root.addWidget(self._build_color_controls())
        root.addWidget(self._build_arena_controls())

        status_row = QHBoxLayout()
        self._frame_lbl = QLabel("Frame: 0")
        self._frame_lbl.setStyleSheet("font-size: 10px; color: #6C757D;")
        self._peak_lbl = QLabel("ROI: --.-°C  mean --.-°C")
        self._peak_lbl.setStyleSheet("font-size: 10px; color: #6C757D;")
        status_row.addWidget(self._frame_lbl)
        status_row.addStretch()
        status_row.addWidget(self._peak_lbl)
        root.addLayout(status_row)

    def _build_heatmap(self) -> QWidget:
        glw = pg.GraphicsLayoutWidget()
        glw.setFixedHeight(200)

        vb = glw.addViewBox()
        vb.setAspectLocked(True)
        vb.invertY(True)

        self._img_item = pg.ImageItem()
        self._colormap_lut = self._make_lut(self._colormap_name)
        self._img_item.setLookupTable(self._colormap_lut)
        self._img_item.setLevels([self._t_min, self._t_max])
        vb.addItem(self._img_item)

        # Arena overlay (orange) — always below ROI in z-order
        self._arena_item = pg.RectROI(
            [0, 0], [CAMERA_WIDTH, CAMERA_HEIGHT],
            pen=pg.mkPen("#FFB300", width=2.0),
            movable=True,
            resizable=True,
        )
        self._arena_item.setVisible(False)
        vb.addItem(self._arena_item)
        self._arena_item.sigRegionChanged.connect(self._sync_arena_spinboxes)

        # ROI overlay (green) — visible in both modes; interactive only in Fixed
        self._roi_item = pg.RectROI(
            [35, 26], [10, 10],
            pen=pg.mkPen("#00FF88", width=1.5),
            movable=True,
            resizable=True,
        )
        self._roi_item.setVisible(True)
        # Start in Dynamic mode: not user-draggable
        self._roi_item.translatable = False
        vb.addItem(self._roi_item)
        self._roi_item.sigRegionChanged.connect(self._sync_roi_spinboxes)

        return glw

    def _build_live_plot(self) -> QWidget:
        pw = pg.PlotWidget()
        pw.setFixedHeight(110)
        pw.setLabel("left", "ROI temp", units="°C")
        pw.setLabel("bottom", "t", units="s")
        pw.showGrid(x=False, y=True, alpha=0.3)
        pw.addLegend(offset=(5, 5))
        self._plot_curve = pw.plot(pen=pg.mkPen("#68A07B", width=1.5), name="max")
        self._plot_curve_mean = pw.plot(pen=pg.mkPen("#5B8DD9", width=1.2), name="mean")
        return pw

    def _build_roi_controls(self) -> QWidget:
        """ROI mode selector + W/H size controls. Position set by dragging the green box."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        roi_lbl = QLabel("ROI:")
        roi_lbl.setStyleSheet("font-size: 11px; color: #495057;")
        row.addWidget(roi_lbl)

        self._roi_mode = QComboBox()
        self._roi_mode.addItems(["Dynamic", "Fixed"])
        self._roi_mode.setMinimumWidth(85)
        self._roi_mode.setToolTip(
            "Dynamic: ROI follows the hottest pixel each frame.\n"
            "Fixed: drag the green box to set position.")
        self._roi_mode.currentTextChanged.connect(self._on_roi_mode_changed)
        row.addWidget(self._roi_mode)

        for lbl, attr in [("W", "_roi_w"), ("H", "_roi_h")]:
            lbl_w = QLabel(lbl)
            lbl_w.setStyleSheet("font-size: 11px; color: #495057;")
            row.addWidget(lbl_w)
            sb = QSpinBox()
            sb.setRange(1, 4096)
            sb.setValue(10)
            sb.setMinimumWidth(65)
            sb.valueChanged.connect(self._sync_roi_size_from_spinbox)
            setattr(self, attr, sb)
            row.addWidget(sb)

        row.addStretch()
        return w

    def _build_color_controls(self) -> QWidget:
        """Per-camera colour scaling: T min, T max, colormap."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        t_lbl = QLabel("T:")
        t_lbl.setStyleSheet("font-size: 11px; color: #495057;")
        row.addWidget(t_lbl)

        self._t_min_spin = QDoubleSpinBox()
        self._t_min_spin.setRange(-50, 200)
        self._t_min_spin.setValue(self._t_min)
        self._t_min_spin.setDecimals(1)
        self._t_min_spin.setMinimumWidth(68)
        self._t_min_spin.setToolTip("Colormap minimum (°C)")
        self._t_min_spin.valueChanged.connect(self._on_levels_changed)
        row.addWidget(self._t_min_spin)

        sep = QLabel("–")
        sep.setStyleSheet("font-size: 11px; color: #6C757D;")
        row.addWidget(sep)

        self._t_max_spin = QDoubleSpinBox()
        self._t_max_spin.setRange(-50, 200)
        self._t_max_spin.setValue(self._t_max)
        self._t_max_spin.setDecimals(1)
        self._t_max_spin.setMinimumWidth(68)
        self._t_max_spin.setToolTip("Colormap maximum (°C)")
        self._t_max_spin.valueChanged.connect(self._on_levels_changed)
        row.addWidget(self._t_max_spin)

        deg = QLabel("°C")
        deg.setStyleSheet("font-size: 11px; color: #6C757D;")
        row.addWidget(deg)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(COLORMAPS)
        self._cmap_combo.setCurrentText(self._colormap_name)
        self._cmap_combo.setMinimumWidth(95)
        self._cmap_combo.currentTextChanged.connect(self._on_colormap_changed)
        row.addWidget(self._cmap_combo)

        row.addStretch()
        return w

    def _build_arena_controls(self) -> QWidget:
        """Arena checkbox + W/H size controls. Position set by dragging the orange box."""
        w = QWidget()
        row = QHBoxLayout(w)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        self._arena_cb = QCheckBox("Arena")
        self._arena_cb.setStyleSheet("font-size: 11px; color: #495057;")
        self._arena_cb.setToolTip(
            "Restrict ROI to this bounding box.\n"
            "Drag the orange rectangle on the heatmap to define the arena.\n"
            "W/H set the initial size; position is fully mouse-driven.")
        self._arena_cb.toggled.connect(self._on_arena_toggled)
        row.addWidget(self._arena_cb)

        for lbl, attr in [("W", "_arena_w"), ("H", "_arena_h")]:
            lbl_w = QLabel(lbl)
            lbl_w.setStyleSheet("font-size: 11px; color: #495057;")
            row.addWidget(lbl_w)
            sb = QSpinBox()
            sb.setRange(1, 4096)
            sb.setValue(CAMERA_WIDTH if lbl == "W" else CAMERA_HEIGHT)
            sb.setMinimumWidth(65)
            sb.setEnabled(False)
            sb.valueChanged.connect(self._sync_arena_size_from_spinbox)
            setattr(self, attr, sb)
            row.addWidget(sb)

        row.addStretch()
        return w

    # ── Public API ─────────────────────────────────────────────────────────────

    @property
    def camera_id(self) -> str:
        return self._camera_id

    @property
    def animal_name(self) -> str:
        return self._animal_edit.text().strip() or self._camera_id

    @property
    def group_name(self) -> str:
        return self._group_combo.currentText()

    @property
    def arena_bounds(self) -> tuple:
        """(enabled, x, y, w, h) — x/y from item position, w/h from spinboxes."""
        enabled = self._arena_cb.isChecked()
        if self._arena_item is not None:
            pos = self._arena_item.pos()
            x, y = int(pos.x()), int(pos.y())
        else:
            x, y = 0, 0
        return (enabled, x, y, self._arena_w.value(), self._arena_h.value())

    def update_groups(self, groups: list[tuple[str, str]]) -> None:
        current = self._group_combo.currentText()
        self._group_combo.blockSignals(True)
        self._group_combo.clear()
        for name, _ in groups:
            self._group_combo.addItem(name)
        idx = self._group_combo.findText(current)
        if idx >= 0:
            self._group_combo.setCurrentIndex(idx)
        self._group_combo.blockSignals(False)

    def set_status_state(self, state: str) -> None:
        _STATES = {
            "idle":       ("#CCCCCC", "Idle — not streaming"),
            "connecting": ("#FB8C00", "Connecting to camera…"),
            "streaming":  ("#43A047", "Live — receiving frames"),
            "recording":  ("#E53935", "Recording"),
            "error":      ("#B71C1C", "Error — check USB / camera"),
        }
        color, tip = _STATES.get(state, ("#CCCCCC", ""))
        self._status_dot.setStyleSheet(f"color: {color}; font-size: 14px;")
        self._status_dot.setToolTip(tip)

    def set_status(self, recording: bool) -> None:
        self.set_status_state("recording" if recording else "idle")

    def update_settings(self, fps: int, t_min: float, t_max: float, colormap: str):
        self._fps = fps
        self._t_min = t_min
        self._t_max = t_max
        self._colormap_name = colormap
        self._plot_buf = deque(maxlen=fps * self.PLOT_WINDOW_S)
        for spin, val in [
            (self._t_min_spin, t_min),
            (self._t_max_spin, t_max),
        ]:
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)
        self._cmap_combo.blockSignals(True)
        self._cmap_combo.setCurrentText(colormap)
        self._cmap_combo.blockSignals(False)
        if self._img_item is not None:
            self._img_item.setLevels([t_min, t_max])
            self._colormap_lut = self._make_lut(colormap)
            self._img_item.setLookupTable(self._colormap_lut)

    # ── Frame handler ──────────────────────────────────────────────────────────

    def on_frame(self, frame: np.ndarray, elapsed_s: float,
                 apply_emissivity: bool = False,
                 epsilon_mouse: float = 0.93,
                 epsilon_sensor: float = 1.0) -> None:
        if apply_emissivity:
            from ..processing.roi_analysis import apply_emissivity_correction_array
            frame = apply_emissivity_correction_array(frame, epsilon_mouse, epsilon_sensor)

        roi_max, roi_mean = self._compute_roi_stats(frame)
        self._plot_buf.append((elapsed_s, roi_max, roi_mean))
        self._frame_count += 1

        self._frame_lbl.setText(f"Frame: {self._frame_count}")
        self._peak_lbl.setText(f"ROI: {roi_max:.1f}°C  mean {roi_mean:.1f}°C")

        if _HAS_PG and not self._display_pending:
            self._display_pending = True
            if self._img_item is not None:
                self._img_item.setImage(frame.T, autoLevels=False)
            if self._plot_buf:
                times = np.array([p[0] for p in self._plot_buf])
                maxes = np.array([p[1] for p in self._plot_buf])
                means = np.array([p[2] for p in self._plot_buf])
                if self._plot_curve is not None:
                    self._plot_curve.setData(times, maxes)
                if self._plot_curve_mean is not None:
                    self._plot_curve_mean.setData(times, means)
            self._display_pending = False

        # Dynamic mode: move green ROI box to hottest pixel each frame
        if _HAS_PG and self._roi_item is not None and self._roi_mode.currentText() == "Dynamic":
            self._update_dynamic_roi(frame)

    def reset(self) -> None:
        self._frame_count = 0
        self._plot_buf.clear()
        self._frame_lbl.setText("Frame: 0")
        self._peak_lbl.setText("ROI: --.-°C  mean --.-°C")
        if self._plot_curve is not None:
            self._plot_curve.setData([], [])
        if self._plot_curve_mean is not None:
            self._plot_curve_mean.setData([], [])

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _on_animal_num_changed(self, value: int) -> None:
        self._animal_edit.setText(f"Mouse_{value:02d}")
        self._emit_assignment()

    def _compute_roi_stats(self, frame: np.ndarray) -> tuple:
        """Return (roi_max, roi_mean), respecting arena mask and ROI mode.

        When the arena is active pixels outside it are set to (frame.min() - 1).
        roi_max is always from inside the arena.
        roi_mean only averages pixels that are genuine sensor readings (>= frame.min()),
        so border overlap never dilutes the mean with sentinel values.
        """
        from ..processing.roi_analysis import (
            find_roi_dynamic, find_roi_static, apply_arena_mask)

        arena_active = self._arena_cb.isChecked() and self._arena_item is not None
        # Record original minimum before masking so we can filter sentinels later
        real_min = float(frame.min())

        grid = frame
        if arena_active:
            pos = self._arena_item.pos()
            grid = apply_arena_mask(
                frame,
                int(pos.x()), int(pos.y()),
                self._arena_w.value(), self._arena_h.value(),
            )

        w, h = self._roi_w.value(), self._roi_h.value()
        if self._roi_mode.currentText() == "Dynamic":
            roi = find_roi_dynamic(grid, w, h)
        else:
            pos = self._roi_item.pos() if self._roi_item is not None else None
            x = int(pos.x()) if pos is not None else 0
            y = int(pos.y()) if pos is not None else 0
            roi = find_roi_static(grid, x, y, w, h)

        if roi.size == 0:
            return float(frame.max()), float(frame.mean())

        roi_max = float(roi.max())
        if arena_active:
            # Exclude sentinel pixels (set to real_min - 1) from the mean
            valid = roi[roi >= real_min]
            roi_mean = float(valid.mean()) if valid.size > 0 else roi_max
        else:
            roi_mean = float(roi.mean())
        return roi_max, roi_mean

    def _update_dynamic_roi(self, frame: np.ndarray) -> None:
        """Move the green ROI overlay to the hottest pixel, clamped inside the arena."""
        from ..processing.roi_analysis import find_max_pixel, apply_arena_mask

        arena_active = self._arena_cb.isChecked() and self._arena_item is not None
        grid = frame
        if arena_active:
            apos = self._arena_item.pos()
            ax, ay = int(apos.x()), int(apos.y())
            aw, ah = self._arena_w.value(), self._arena_h.value()
            grid = apply_arena_mask(frame, ax, ay, aw, ah)

        row, col, _ = find_max_pixel(grid)
        rw, rh = self._roi_w.value(), self._roi_h.value()

        # Centre ROI on hotspot
        x = col - rw // 2
        y = row - rh // 2

        if arena_active:
            # Clamp so the entire ROI box stays within the orange arena rectangle
            x = max(ax, min(x, ax + aw - rw))
            y = max(ay, min(y, ay + ah - rh))
        else:
            x = max(0, x)
            y = max(0, y)

        self._roi_item.blockSignals(True)
        self._roi_item.setPos([x, y])
        self._roi_item.setSize([rw, rh])
        self._roi_item.blockSignals(False)

    def _on_roi_mode_changed(self, mode: str) -> None:
        fixed = (mode == "Fixed")
        if self._roi_item is not None:
            # Always visible; only user-draggable in Fixed mode
            self._roi_item.translatable = fixed

    def _sync_roi_spinboxes(self):
        """Update W/H spinboxes when the green ROI box is resized by dragging."""
        if self._roi_item is None:
            return
        size = self._roi_item.size()
        for sb, val in [
            (self._roi_w, max(1, int(size.x()))),
            (self._roi_h, max(1, int(size.y()))),
        ]:
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)

    def _sync_roi_size_from_spinbox(self):
        """Resize the green ROI box when W/H spinboxes change (keeps current position)."""
        if self._roi_item is None:
            return
        self._roi_item.blockSignals(True)
        self._roi_item.setSize([self._roi_w.value(), self._roi_h.value()])
        self._roi_item.blockSignals(False)

    def _on_levels_changed(self):
        if self._img_item is not None:
            self._img_item.setLevels(
                [self._t_min_spin.value(), self._t_max_spin.value()])

    def _on_colormap_changed(self, name: str):
        if self._img_item is not None:
            self._colormap_lut = self._make_lut(name)
            self._img_item.setLookupTable(self._colormap_lut)

    def _on_arena_toggled(self, enabled: bool):
        self._arena_w.setEnabled(enabled)
        self._arena_h.setEnabled(enabled)
        if self._arena_item is not None:
            self._arena_item.setVisible(enabled)
            self._arena_item.translatable = enabled

    def _sync_arena_spinboxes(self):
        """Update W/H spinboxes when the orange arena box is resized by dragging."""
        if self._arena_item is None:
            return
        size = self._arena_item.size()
        for sb, val in [
            (self._arena_w, max(1, int(size.x()))),
            (self._arena_h, max(1, int(size.y()))),
        ]:
            sb.blockSignals(True)
            sb.setValue(val)
            sb.blockSignals(False)

    def _sync_arena_size_from_spinbox(self):
        """Resize the orange arena box when W/H spinboxes change."""
        if self._arena_item is None:
            return
        self._arena_item.blockSignals(True)
        self._arena_item.setSize([self._arena_w.value(), self._arena_h.value()])
        self._arena_item.blockSignals(False)

    def _emit_assignment(self):
        self.assignment_changed.emit(
            self._camera_id, self.animal_name, self.group_name)

    @staticmethod
    def _make_lut(colormap_name: str) -> np.ndarray:
        try:
            import matplotlib
            cmap = matplotlib.colormaps[colormap_name]
            lut = (cmap(np.linspace(0, 1, 256))[:, :3] * 255).astype(np.uint8)
            return lut
        except Exception:
            return np.tile(np.arange(256, dtype=np.uint8).reshape(-1, 1), (1, 3))
