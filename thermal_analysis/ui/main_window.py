"""
Glöd — Main application window.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QScrollArea, QVBoxLayout,
    QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QProgressBar, QRadioButton,
    QButtonGroup, QCheckBox, QFrame, QTabWidget, QTextEdit,
    QFileDialog, QMessageBox, QStackedWidget,
)
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, QSettings, pyqtSlot

from .plot_panel import PlotScrollPanel
from .group_widget import FolderGroupWidget
from .live_capture_tab import LiveCaptureTab
from .collapsible import CollapsibleSection
from ..processing.workers import ParseWorker, VideoWorker, GroupData
from ..utils.config import (
    AnalysisSettings, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    METADATA_COLS, ROI_DEFAULT_W, ROI_DEFAULT_H, ROI_DEFAULT_X,
    ROI_DEFAULT_Y,
    COLORMAPS, COLORMAP_DEFAULT,
    T_MIN_DEFAULT, T_MAX_DEFAULT,
    TCORE_SAMPLING_INTERVAL_SEC, TCORE_AVERAGING_WINDOW_MIN,
    TCORE_SLOPE, TCORE_INTERCEPT, EPSILON_SENSOR,
    BASELINE_END_SEC, SETTLE_S, MIN_COVERAGE, RECORDINGS_SUBDIR,
)
from ..utils.animal_registry import AnimalRegistry
from ..utils import naming
from ..utils.naming import animal_from_stem
from ..processing.publication import write_figure_source_data

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Glöd main window."""

    def __init__(self, icon_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Glöd — Thermal Imaging Analysis")
        self.setMinimumSize(1000, 640)
        self.resize(1400, 820)

        if icon_path and Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))

        self._group_widgets: List[FolderGroupWidget] = []
        self._parse_worker: Optional[ParseWorker] = None
        self._video_worker: Optional[VideoWorker] = None
        self._all_group_data: dict = {}
        self._stats_results: dict = {}
        self._output_dir: str = self._load_project_folder()
        self._run_dir: Optional[Path] = None   # analysis/<timestamp>/ of the current run
        self._run_settings: Optional[AnalysisSettings] = None
        self._registry = AnimalRegistry(parent=self)

        self._build_ui()
        self._connect_signals()
        self._set_phase(str(QSettings("Glod", "glod").value("phase", "record")))

    # ─────────────────────────────────────────────────────────────────────
    # Project folder (single user-chosen root for recordings + analyses)
    # ─────────────────────────────────────────────────────────────────────

    def _load_project_folder(self) -> str:
        """Return the stored project folder, asking the user on first launch."""
        stored = str(QSettings("Glod", "glod").value("project_folder", "") or "")
        if stored and Path(stored).is_dir():
            return stored
        start = stored or str(Path.home())
        folder = QFileDialog.getExistingDirectory(
            None, "Choose a project folder — recordings and results are saved here",
            start)
        if not folder:
            folder = stored or str(Path.home() / "Glod")
            Path(folder).mkdir(parents=True, exist_ok=True)
        QSettings("Glod", "glod").setValue("project_folder", folder)
        return folder

    def _set_project_label(self, folder: str):
        """Show the project folder, middle-elided so long paths never overflow."""
        from PyQt6.QtCore import Qt as _Qt
        fm = self._project_lbl.fontMetrics()
        self._project_lbl.setText(fm.elidedText(folder, _Qt.TextElideMode.ElideMiddle, 340))
        self._project_lbl.setToolTip(folder)

    def _change_project_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Choose project folder", self._output_dir)
        if not folder:
            return
        self._output_dir = folder
        QSettings("Glod", "glod").setValue("project_folder", folder)
        self._set_project_label(folder)
        self._live_tab.set_output_dir(folder)
        self._log(f"Project folder → {folder}")

    # ─────────────────────────────────────────────────────────────────────
    # UI Construction
    # ─────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QVBoxLayout(central)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # ── Header bar ────────────────────────────────────────────────────
        root_layout.addWidget(self._build_header())

        # ── Phase pages: Record (live capture) | Analyse ──────────────────
        self._phase_stack = QStackedWidget()

        self._live_tab = LiveCaptureTab(
            registry=self._registry,
            output_dir=self._output_dir,
            parent=self,
        )
        self._live_tab.send_to_analysis.connect(
            lambda path: self._load_from_recordings(path, auto_run=True))
        self._phase_stack.addWidget(self._live_tab)          # index 0

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_view_panel())
        splitter.setSizes([430, 970])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        self._phase_stack.addWidget(splitter)                # index 1

        root_layout.addWidget(self._phase_stack, stretch=1)

        # ── Status bar ────────────────────────────────────────────────────
        self._open_run_btn = QPushButton("Open results folder")
        self._open_run_btn.setObjectName("SmallButton")
        self._open_run_btn.setVisible(False)
        self._open_run_btn.clicked.connect(self._open_run_dir)
        self.statusBar().addPermanentWidget(self._open_run_btn)
        self.statusBar().showMessage("Ready.")

    def _set_phase(self, phase: str):
        analyse = phase == "analyse"
        self._phase_stack.setCurrentIndex(1 if analyse else 0)
        self._phase_record_btn.setChecked(not analyse)
        self._phase_analyse_btn.setChecked(analyse)
        QSettings("Glod", "glod").setValue("phase", "analyse" if analyse else "record")

    def _open_run_dir(self):
        if self._run_dir and self._run_dir.is_dir():
            os.startfile(str(self._run_dir))

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(
            "QFrame { background: #EEF2F0; border-bottom: 1px solid #DEE2E6; }")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)

        title = QLabel("Glöd")
        title.setObjectName("TitleLabel")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)
        layout.addSpacing(20)

        # Phase switch
        self._phase_record_btn = QPushButton("Record")
        self._phase_analyse_btn = QPushButton("Analyse")
        for btn, phase in ((self._phase_record_btn, "record"),
                           (self._phase_analyse_btn, "analyse")):
            btn.setObjectName("PhaseButton")
            btn.setCheckable(True)
            btn.setFixedHeight(30)
            btn.clicked.connect(lambda _=False, p=phase: self._set_phase(p))
            layout.addWidget(btn)

        layout.addStretch()

        # Project folder
        folder_caption = QLabel("Project folder")
        folder_caption.setStyleSheet("color: #6C757D; font-size: 11px;")
        layout.addWidget(folder_caption)
        self._project_lbl = QLabel()
        self._project_lbl.setMaximumWidth(360)
        self._set_project_label(self._output_dir)
        self._project_lbl.setTextInteractionFlags(
            Qt.TextInteractionFlag.NoTextInteraction)
        layout.addWidget(self._project_lbl)
        change_btn = QPushButton("Change…")
        change_btn.setObjectName("SecondaryButton")
        change_btn.setFixedHeight(30)
        change_btn.clicked.connect(self._change_project_folder)
        layout.addWidget(change_btn)
        layout.addSpacing(8)

        # About button
        about_btn = QPushButton("About")
        about_btn.setObjectName("SecondaryButton")
        about_btn.setFixedHeight(30)
        about_btn.clicked.connect(self._show_about)
        layout.addWidget(about_btn)

        return header

    def _build_control_panel(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setMinimumWidth(425)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("ControlScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("ControlPanel")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        # 1. Groups — the primary input
        layout.addWidget(self._build_groups_section())

        # 1b. Time-point comparison (collapsed; summary shows the chosen times)
        self._sec_tp = CollapsibleSection("Time-point comparison",
                                          settings_key="timepoints")
        self._sec_tp.add_widget(self._build_timepoint_group())
        layout.addWidget(self._sec_tp)

        # 2. Video export (collapsed)
        video_sec = CollapsibleSection("Video export", settings_key="video")
        video_sec.add_widget(self._build_display_group())
        video_sec.add_widget(self._build_video_actions())
        layout.addWidget(video_sec)

        # 3. Advanced settings (collapsed; sensible defaults)
        adv = CollapsibleSection("Advanced settings", settings_key="advanced")
        self._sec_roi = CollapsibleSection("ROI & arena", expanded=True,
                                           settings_key="roi")
        self._sec_roi.add_widget(self._build_roi_group())
        self._sec_tcore = CollapsibleSection("T_core estimation",
                                             settings_key="tcore")
        self._sec_tcore.add_widget(self._build_tcore_group())
        self._sec_emiss = CollapsibleSection("Emissivity correction",
                                             settings_key="emissivity")
        self._sec_emiss.add_widget(self._build_emissivity_group())
        self._sec_fmt = CollapsibleSection("Recording file format",
                                           settings_key="format")
        self._sec_fmt.add_widget(self._build_camera_group())
        for sec in (self._sec_roi, self._sec_tcore, self._sec_emiss, self._sec_fmt):
            adv.add_widget(sec)
        layout.addWidget(adv)
        layout.addStretch()

        for w in (self._roi_body, self._roi_static,
                  self._tcore_cb, self._emiss_cb):
            w.toggled.connect(lambda _: self._refresh_summaries())
        for w in (self._roi_w, self._roi_h, self._cam_width, self._cam_height,
                  self._cam_fps):
            w.valueChanged.connect(lambda _: self._refresh_summaries())
        self._refresh_summaries()

        scroll.setWidget(container)
        wrapper_layout.addWidget(scroll, 1)

        # Pinned footer: Run + progress are always visible
        wrapper_layout.addWidget(self._build_actions())

        footer = QLabel("P.Valiathan 2026  ·  Karolinska Institutet")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(
            "color: #B8C4BB; font-size: 9px; padding: 4px 0;"
            "border-top: 1px solid #DEE2E6; background: #EEF2F0;"
        )
        wrapper_layout.addWidget(footer)

        return wrapper

    # ── Parameter group builders ──────────────────────────────────────────

    def _build_camera_group(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(3)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Width (px)"))
        self._cam_width = QSpinBox()
        self._cam_width.setRange(1, 4096)
        self._cam_width.setValue(CAMERA_WIDTH)
        self._cam_width.setToolTip("Pixel columns per frame (Waveshare MI48x3 default: 80)")
        row1.addWidget(self._cam_width)
        row1.addWidget(QLabel("Height (px)"))
        self._cam_height = QSpinBox()
        self._cam_height.setRange(1, 4096)
        self._cam_height.setValue(CAMERA_HEIGHT)
        self._cam_height.setToolTip("Pixel rows per frame (Waveshare MI48x3 default: 62)")
        row1.addWidget(self._cam_height)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("FPS"))
        self._cam_fps = QDoubleSpinBox()
        self._cam_fps.setRange(0.1, 120.0)
        self._cam_fps.setValue(CAMERA_FPS)
        self._cam_fps.setDecimals(1)
        self._cam_fps.setToolTip("Camera frames per second (used for T_core algorithm)")
        row2.addWidget(self._cam_fps)
        row2.addWidget(QLabel("Meta cols"))
        self._meta_cols = QSpinBox()
        self._meta_cols.setRange(0, 64)
        self._meta_cols.setValue(METADATA_COLS)
        self._meta_cols.setToolTip(
            "Number of metadata columns before pixel data (MI48x3: 8)")
        row2.addWidget(self._meta_cols)
        layout.addLayout(row2)

        return box

    def _build_roi_group(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(3)

        # Mode selection — radio buttons (Body tracking is the recommended default)
        self._roi_mode_grp = QButtonGroup()
        self._roi_body = QRadioButton("Body (auto-tracking, recommended)")
        self._roi_body.setToolTip(
            "Segments the mouse body in every frame (adaptive threshold, tail removed)\n"
            "and reads its temperature. Frames where the body is mostly out of view are\n"
            "ignored instead of appearing as temperature dips. Needs no settings.")
        self._roi_body.setChecked(True)
        self._roi_mode_grp.addButton(self._roi_body)
        layout.addWidget(self._roi_body)

        self._roi_static = QRadioButton("Fixed box (restrained / anaesthetised animal)")
        self._roi_static.setToolTip(
            "Reads the temperature from a fixed rectangle (X, Y, width, height).\n"
            "Use only when the animal does not move.")
        self._roi_mode_grp.addButton(self._roi_static)
        layout.addWidget(self._roi_static)
        coord_row = QHBoxLayout()
        coord_row.addWidget(QLabel("X"))
        self._roi_x = QSpinBox(); self._roi_x.setRange(0, 4096); self._roi_x.setValue(ROI_DEFAULT_X)
        coord_row.addWidget(self._roi_x)
        coord_row.addWidget(QLabel("Y"))
        self._roi_y = QSpinBox(); self._roi_y.setRange(0, 4096); self._roi_y.setValue(ROI_DEFAULT_Y)
        coord_row.addWidget(self._roi_y)
        layout.addLayout(coord_row)

        size_row = QHBoxLayout()
        size_row.addWidget(QLabel("Width"))
        self._roi_w = QSpinBox(); self._roi_w.setRange(1, 4096); self._roi_w.setValue(ROI_DEFAULT_W)
        size_row.addWidget(self._roi_w)
        size_row.addWidget(QLabel("Height"))
        self._roi_h = QSpinBox(); self._roi_h.setRange(1, 4096); self._roi_h.setValue(ROI_DEFAULT_H)
        size_row.addWidget(self._roi_h)
        layout.addLayout(size_row)

        # Wire all mode buttons to the same toggle method
        for btn in (self._roi_body, self._roi_static):
            btn.toggled.connect(lambda _: self._update_roi_controls())
        self._update_roi_controls()  # apply initial state

        # ── Settling period ───────────────────────────────────────────────
        settle_row = QHBoxLayout()
        settle_row.addWidget(QLabel("Settling period"))
        self._settle_s = QDoubleSpinBox()
        self._settle_s.setRange(0.0, 600.0)
        self._settle_s.setDecimals(0)
        self._settle_s.setSingleStep(15.0)
        self._settle_s.setValue(SETTLE_S)
        self._settle_s.setSuffix(" s")
        self._settle_s.setToolTip(
            "Seconds after placing the mouse in the box that are excluded from the\n"
            "baseline, T_core and per-animal statistics (shaded grey in plots).\n"
            "On the reference recordings the body signal ran 0.3–0.5 °C low for the\n"
            "first ~45–90 s (handling/transfer). 0 = use everything.")
        settle_row.addWidget(self._settle_s)
        settle_row.addStretch()
        layout.addLayout(settle_row)

        cov_row = QHBoxLayout()
        cov_row.addWidget(QLabel("Min. body tracked"))
        self._min_cov = QDoubleSpinBox()
        self._min_cov.setRange(0.0, 100.0)
        self._min_cov.setDecimals(0)
        self._min_cov.setValue(MIN_COVERAGE * 100)
        self._min_cov.setSuffix(" %")
        self._min_cov.setToolTip(
            "Recordings in which the body was tracked in fewer frames than this (after the\n"
            "settling period) are left out of group statistics and listed as EXCLUDED in the\n"
            "tracking-QC table. Untracked frames are never filled in.")
        cov_row.addWidget(self._min_cov)
        cov_row.addStretch()
        layout.addLayout(cov_row)

        # ── Arena ─────────────────────────────────────────────────────────
        arena_hdr = QHBoxLayout()
        self._arena_cb = QCheckBox("Arena (restrict ROI to box)")
        self._arena_cb.setChecked(False)
        self._arena_cb.setToolTip(
            "Mask pixels outside the arena before ROI detection.\n"
            "Set X/Y/W/H below or drag the orange box in the live preview.")
        self._arena_cb.toggled.connect(self._toggle_arena_coords)
        arena_hdr.addWidget(self._arena_cb)
        layout.addLayout(arena_hdr)

        self._arena_box = QWidget()
        arena_box_layout = QVBoxLayout(self._arena_box)
        arena_box_layout.setContentsMargins(0, 0, 0, 0)
        arena_box_layout.setSpacing(3)

        arena_coord_row = QHBoxLayout()
        arena_coord_row.addWidget(QLabel("X"))
        self._arena_x = QSpinBox()
        self._arena_x.setRange(0, 4096)
        self._arena_x.setValue(0)
        arena_coord_row.addWidget(self._arena_x)
        arena_coord_row.addWidget(QLabel("Y"))
        self._arena_y = QSpinBox()
        self._arena_y.setRange(0, 4096)
        self._arena_y.setValue(0)
        arena_coord_row.addWidget(self._arena_y)
        arena_box_layout.addLayout(arena_coord_row)

        arena_size_row = QHBoxLayout()
        arena_size_row.addWidget(QLabel("W"))
        self._arena_w = QSpinBox()
        self._arena_w.setRange(1, 4096)
        self._arena_w.setValue(CAMERA_WIDTH)
        arena_size_row.addWidget(self._arena_w)
        arena_size_row.addWidget(QLabel("H"))
        self._arena_h = QSpinBox()
        self._arena_h.setRange(1, 4096)
        self._arena_h.setValue(CAMERA_HEIGHT)
        arena_size_row.addWidget(self._arena_h)
        arena_box_layout.addLayout(arena_size_row)
        layout.addWidget(self._arena_box)

        self._toggle_arena_coords(False)

        return box

    def _update_roi_controls(self):
        static = self._roi_static.isChecked()
        for w in (self._roi_x, self._roi_y, self._roi_w, self._roi_h):
            w.setEnabled(static)

    def _toggle_arena_coords(self, enabled: bool):
        for w in [self._arena_x, self._arena_y, self._arena_w, self._arena_h]:
            w.setEnabled(enabled)
        self._arena_box.setVisible(enabled)

    def _build_display_group(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(3)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap"))
        self._colormap = QComboBox()
        self._colormap.addItems(COLORMAPS)
        self._colormap.setCurrentText(COLORMAP_DEFAULT)
        cmap_row.addWidget(self._colormap)
        layout.addLayout(cmap_row)

        range_row = QHBoxLayout()
        range_row.addWidget(QLabel("T Min (°C)"))
        self._t_min = QDoubleSpinBox()
        self._t_min.setRange(-50, 200); self._t_min.setValue(T_MIN_DEFAULT); self._t_min.setDecimals(1)
        range_row.addWidget(self._t_min)
        range_row.addWidget(QLabel("T Max (°C)"))
        self._t_max = QDoubleSpinBox()
        self._t_max.setRange(-50, 200); self._t_max.setValue(T_MAX_DEFAULT); self._t_max.setDecimals(1)
        range_row.addWidget(self._t_max)
        layout.addLayout(range_row)

        return box

    def _build_timepoint_group(self) -> QWidget:
        """Up to three user-set times (minutes from recording start)."""
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        hint = QLabel("Compare temperature between groups at up to 3 times "
                      "after the start of each recording.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #6C757D; font-size: 11px;")
        layout.addWidget(hint)

        self._tp_checks: list = []
        self._tp_spins: list = []
        for i, default in enumerate((5.0, 15.0, 30.0)):
            row = QHBoxLayout()
            cb = QCheckBox(f"Time point {i + 1}")
            sp = QDoubleSpinBox()
            sp.setRange(0.0, 1440.0)
            sp.setDecimals(1)
            sp.setValue(default)
            sp.setSuffix(" min")
            sp.setEnabled(False)
            cb.toggled.connect(sp.setEnabled)
            cb.toggled.connect(lambda _: self._refresh_summaries())
            sp.valueChanged.connect(lambda _: self._refresh_summaries())
            row.addWidget(cb)
            row.addStretch()
            row.addWidget(sp)
            layout.addLayout(row)
            self._tp_checks.append(cb)
            self._tp_spins.append(sp)

        sig_row = QHBoxLayout()
        sig_row.addWidget(QLabel("Signal"))
        self._tp_signal = QComboBox()
        self._tp_signal.addItem("ROI max temperature (°C)", "roi_max")
        self._tp_signal.addItem("Baseline-normalised ROI max (Δ°C)", "normalized_roi_max")
        self._tp_signal.addItem("Estimated T_core (°C)", "tcore_estimated")
        sig_row.addWidget(self._tp_signal, stretch=1)
        layout.addLayout(sig_row)

        win_row = QHBoxLayout()
        win_row.addWidget(QLabel("Averaging window"))
        self._tp_window = QDoubleSpinBox()
        self._tp_window.setRange(1.0, 600.0)
        self._tp_window.setDecimals(0)
        self._tp_window.setValue(30.0)
        self._tp_window.setSuffix(" s")
        self._tp_window.setToolTip(
            "Mean over this window centred on each time point. "
            "Recordings that end before a time point are excluded from it.")
        win_row.addWidget(self._tp_window)
        layout.addLayout(win_row)
        return box

    def _timepoints_s(self) -> list:
        vals = {round(sp.value() * 60.0, 3)
                for cb, sp in zip(self._tp_checks, self._tp_spins) if cb.isChecked()}
        return sorted(vals)

    def _refresh_summaries(self):
        """One-line state shown on collapsed section headers."""
        tps = self._timepoints_s()
        self._sec_tp.set_summary(
            ", ".join(f"{t / 60:g} min" for t in tps) if tps else "off")
        self._sec_roi.set_summary(
            "Body tracking" if self._roi_body.isChecked()
            else f"Fixed box · {self._roi_w.value()}×{self._roi_h.value()}")
        self._sec_tcore.set_summary("on" if self._tcore_cb.isChecked() else "off")
        self._sec_emiss.set_summary("on" if self._emiss_cb.isChecked() else "off")
        self._sec_fmt.set_summary(
            f"{self._cam_width.value()}×{self._cam_height.value()} @ "
            f"{self._cam_fps.value():g} fps")

    def _build_emissivity_group(self) -> QWidget:
        box = QWidget()
        layout = QVBoxLayout(box)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self._emiss_cb = QCheckBox("Correct for coat emissivity")
        self._emiss_cb.setChecked(True)
        self._emiss_cb.setToolTip(
            "Choose the animal / coat on each group card (default C57BL/6).\n"
            "Applied to the temperature readings only; estimated T_core always uses the\n"
            "uncorrected sensor values its regression was calibrated on.")
        layout.addWidget(self._emiss_cb)

        sens_row = QHBoxLayout()
        sens_row.addWidget(QLabel("Camera setting ε"))
        self._eps_sensor = QDoubleSpinBox()
        self._eps_sensor.setRange(0.50, 1.0)
        self._eps_sensor.setDecimals(2)
        self._eps_sensor.setSingleStep(0.01)
        self._eps_sensor.setValue(EPSILON_SENSOR)
        self._eps_sensor.setToolTip(
            "Emissivity the camera itself assumed when it converted radiance to °C.\n"
            "The MI48x3 EMISSIVITY register reads 95 % on all three cameras\n"
            "(factory default; Glöd never changes it). Change only if you set it differently.")
        sens_row.addWidget(self._eps_sensor)
        sens_row.addStretch()
        layout.addLayout(sens_row)
        note = QLabel("Animal / coat is chosen on each group card. Fur ε in the literature "
                      "spans 0.86–0.98; a wrong value shifts all readings by a nearly constant "
                      "offset, so it matters for absolute temperatures, not for within-animal changes.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #6C757D; font-size: 10px;")
        layout.addWidget(note)

        return box
    def _build_tcore_group(self) -> QWidget:
        box = QWidget()
        box.setToolTip("van der Vinne et al. (2020) Sci Rep 10:20680")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(3)

        enable_row = QHBoxLayout()
        self._tcore_cb = QCheckBox("Compute estimated T_core")
        self._tcore_cb.setChecked(True)
        self._tcore_cb.setToolTip(
            "Uses van der Vinne et al. (2020) Sci Rep 10:20680.\n"
            "Group-average params: slope=0.93, intercept=7.1°C.\n"
            "Between-animal error ≈ ±0.9°C.")
        enable_row.addWidget(self._tcore_cb)
        layout.addLayout(enable_row)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Sampling (s)"))
        self._tc_sample = QDoubleSpinBox()
        self._tc_sample.setRange(1, 3600); self._tc_sample.setValue(TCORE_SAMPLING_INTERVAL_SEC)
        self._tc_sample.setDecimals(0)
        self._tc_sample.setToolTip("Duration of each sampling block (paper: 60 s)")
        row1.addWidget(self._tc_sample)
        row1.addWidget(QLabel("Window (min)"))
        self._tc_window = QDoubleSpinBox()
        self._tc_window.setRange(0.5, 720); self._tc_window.setValue(TCORE_AVERAGING_WINDOW_MIN)
        self._tc_window.setDecimals(0)
        self._tc_window.setToolTip("Rolling average window (paper: 30 min)")
        row1.addWidget(self._tc_window)
        layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Slope"))
        self._tc_slope = QDoubleSpinBox()
        self._tc_slope.setRange(-10, 10); self._tc_slope.setValue(TCORE_SLOPE)
        self._tc_slope.setDecimals(3)
        row2.addWidget(self._tc_slope)
        row2.addWidget(QLabel("Intercept (°C)"))
        self._tc_intercept = QDoubleSpinBox()
        self._tc_intercept.setRange(-50, 50); self._tc_intercept.setValue(TCORE_INTERCEPT)
        self._tc_intercept.setDecimals(2)
        row2.addWidget(self._tc_intercept)
        layout.addLayout(row2)

        return box

    def _build_groups_section(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        lbl = QLabel("Experiment Groups")
        lbl.setObjectName("SectionLabel")
        header_row.addWidget(lbl)
        header_row.addStretch()
        add_btn = QPushButton("✚  Add Group")
        add_btn.setObjectName("SecondaryButton")
        add_btn.setFixedHeight(28)
        add_btn.clicked.connect(self._add_group)
        header_row.addWidget(add_btn)
        layout.addLayout(header_row)

        load_rec_btn = QPushButton("Load from Recordings…")
        load_rec_btn.setObjectName("SecondaryButton")
        load_rec_btn.setFixedHeight(26)
        load_rec_btn.setToolTip(
            "Scan the recordings/ subfolder of the output directory and "
            "auto-populate group cards from saved live-capture sessions."
        )
        load_rec_btn.clicked.connect(self._load_from_recordings_dialog)
        layout.addWidget(load_rec_btn)

        # Paired design: same animals measured in several groups (e.g. ±drug)
        self._paired_cb = QCheckBox("Paired design (same animals in groups)")
        self._paired_cb.setToolTip(
            "Tick when the same mice were recorded under different conditions\n"
            "(e.g. with and without drug). Animals are matched by name across groups\n"
            "and compared with paired statistics. Auto-ticked when the same animal\n"
            "names are found in more than one group.")
        self._paired_cb.clicked.connect(self._on_paired_clicked)
        pair_row = QHBoxLayout()
        pair_row.addWidget(self._paired_cb, stretch=1)
        help_btn = QPushButton("?")
        help_btn.setObjectName("SmallButton")
        help_btn.setFixedSize(24, 24)
        help_btn.setToolTip("How to set up experiments that reuse animals")
        help_btn.clicked.connect(self._show_pairing_help)
        pair_row.addWidget(help_btn)
        layout.addLayout(pair_row)
        self._pair_status = QLabel("")
        self._pair_status.setWordWrap(True)
        self._pair_status.setStyleSheet("color: #6C757D; font-size: 11px;")
        self._pair_status.setVisible(False)
        layout.addWidget(self._pair_status)
        self._pair_user_set = False   # True once the user toggled the box themselves

        self._groups_layout = QVBoxLayout()
        self._groups_layout.setSpacing(6)
        layout.addLayout(self._groups_layout)

        return container

    def _build_actions(self) -> QWidget:
        """Pinned footer: Run button + progress (always visible)."""
        container = QWidget()
        container.setObjectName("ActionsBar")
        container.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setObjectName("RunButton")
        self._run_btn.setFixedHeight(40)
        self._run_btn.clicked.connect(self._run_analysis)
        layout.addWidget(self._run_btn)

        self._parse_progress = QProgressBar()
        self._parse_progress.setValue(0)
        self._parse_progress.setTextVisible(True)
        layout.addWidget(self._parse_progress)

        return container

    def _build_video_actions(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 4, 0, 0)
        layout.setSpacing(4)

        self._video_btn = QPushButton("Export Video")
        self._video_btn.setObjectName("SecondaryButton")
        self._video_btn.setFixedHeight(30)
        self._video_btn.setToolTip(
            "Saved automatically to the results folder of the last analysis.")
        self._video_btn.clicked.connect(self._export_video_prompt)
        layout.addWidget(self._video_btn)

        self._video_progress = QProgressBar()
        self._video_progress.setValue(0)
        self._video_progress.setTextVisible(True)
        layout.addWidget(self._video_progress)

        return container

    def _build_view_panel(self) -> QWidget:
        """Results area: plots (category tabs) and the console log."""
        self._tabs = QTabWidget()

        self._plot_scroll = PlotScrollPanel()
        self._tabs.addTab(self._plot_scroll, "Results")

        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._tabs.addTab(self._log_edit, "Log")

        return self._tabs

    # ─────────────────────────────────────────────────────────────────────
    # Signal connections
    # ─────────────────────────────────────────────────────────────────────

    def _connect_signals(self):
        pass  # connections made inline during UI build

    # ─────────────────────────────────────────────────────────────────────
    # Group management
    # ─────────────────────────────────────────────────────────────────────

    def _add_group(self):
        idx = len(self._group_widgets) + 1
        gw = FolderGroupWidget(idx)
        gw.removed.connect(self._remove_group)
        gw.color_changed.connect(self._on_group_color_changed)
        gw.changed.connect(self._refresh_pairing)
        self._group_widgets.append(gw)
        self._groups_layout.addWidget(gw)
        self._refresh_pairing()

    def _load_from_recordings_dialog(self):
        """Let the user pick a recordings/ folder, then auto-populate group cards."""
        default = str(Path(self._output_dir) / RECORDINGS_SUBDIR)
        folder = QFileDialog.getExistingDirectory(
            self, "Select a recordings folder or a Glöd project folder", default)
        if folder:
            # A whole project folder is accepted: use its recordings/ subfolder.
            sub = Path(folder) / RECORDINGS_SUBDIR
            self._load_from_recordings(str(sub if sub.is_dir() else folder))

    def _load_from_recordings(self, recordings_path: str, auto_run: bool = False):
        """
        Scan recordings_path for group subdirectories and create a
        FolderGroupWidget for each one found.  Each group dir may contain
        animal subdirs (matched by get_txt_files's */*.txt glob).
        """
        base = Path(recordings_path)
        if not base.is_dir():
            QMessageBox.warning(
                self, "Folder Not Found",
                f"Recordings folder not found:\n{recordings_path}")
            return

        group_dirs = sorted(d for d in base.iterdir() if d.is_dir())
        if not group_dirs:
            QMessageBox.information(
                self, "No Groups Found",
                f"No group subdirectories found in:\n{recordings_path}")
            return

        added = 0
        existing_names = {gw.group_name for gw in self._group_widgets}
        for group_dir in group_dirs:
            if group_dir.name in existing_names:
                continue
            gw = FolderGroupWidget(len(self._group_widgets) + 1, parent=self)
            gw._name_edit.setText(group_dir.name)
            gw._folder_path = str(group_dir)
            gw._path_label.setText(FolderGroupWidget.describe_folder(str(group_dir)))
            gw.removed.connect(self._remove_group)
            gw.color_changed.connect(self._on_group_color_changed)
            gw.changed.connect(self._refresh_pairing)
            self._group_widgets.append(gw)
            self._groups_layout.addWidget(gw)
            added += 1

        if added:
            self._refresh_pairing()
            self._set_phase("analyse")
            self._tabs.setCurrentWidget(self._plot_scroll)
            self._log(f"✓ Loaded {added} group(s) from {recordings_path}")
            if auto_run:
                self._run_analysis()
        else:
            self._log("No new groups found (already loaded or folder empty).")

    def _animals_by_group(self) -> dict:
        """{group name: {animal key: display name}} from the recordings on disk."""
        out: dict = {}
        for gw in self._group_widgets:
            if not gw.is_valid():
                continue
            animals = out.setdefault(gw.group_name, {})
            for f in gw.get_txt_files():
                a = naming.animal_from_stem(Path(f).stem)
                animals.setdefault(naming.safe(a).lower(), a)
        return out

    def _show_pairing_help(self):
        QMessageBox.information(
            self, "Reusing animals across conditions",
            "<b>Same mouse, different conditions (e.g. with / without drug)</b>"
            "<ol>"
            "<li><b>One group per condition</b> - e.g. <i>Vehicle</i> and <i>Drug</i>.</li>"
            "<li><b>Same animal name</b> for the same mouse in every group "
            "(e.g. <i>M1</i> in both). Names are matched ignoring upper/lower case.</li>"
            "<li>Record each condition on its own day or session - groups from "
            "different sessions are found automatically.</li>"
            "<li>In Analyse, Glöd ticks <i>paired design</i> for you and lists "
            "which animals matched.</li>"
            "</ol>"
            "Animals missing from a group are left out of the paired tests.<br>"
            "Different mice in each group? Use different names - nothing else to do.")

    def _on_paired_clicked(self, checked: bool):
        self._pair_user_set = True
        self._refresh_pairing()

    def _refresh_pairing(self):
        """Detect animals shared between groups; auto-tick and explain the pairing."""
        by_group = self._animals_by_group()
        names = list(by_group)
        if len(names) < 2:
            self._pair_status.setVisible(False)
            if not self._pair_user_set:
                self._paired_cb.setChecked(False)
            return
        all_keys = set().union(*[set(v) for v in by_group.values()])
        shared = set.intersection(*[set(v) for v in by_group.values()])
        in_several = {k for k in all_keys if sum(k in v for v in by_group.values()) > 1}
        if in_several and not self._pair_user_set:
            self._paired_cb.setChecked(True)

        if not self._paired_cb.isChecked():
            if in_several:
                self._pair_status.setText(
                    f"{len(in_several)} animal name(s) appear in more than one group — "
                    "tick the box above to compare them as paired.")
                self._pair_status.setVisible(True)
            else:
                self._pair_status.setVisible(False)
            return

        display = {}
        for v in by_group.values():
            display.update(v)
        lines = [f"✓ {len(shared)} animal(s) found in all {len(names)} groups "
                 "will be compared as paired."]
        if len(shared) < 2:
            lines = ["⚠ Fewer than 2 animals appear in every group — paired statistics "
                     "need matching animal names across groups."]
        loose = sorted(display[k] for k in all_keys - shared)
        if loose:
            shown = ", ".join(loose[:6]) + (" …" if len(loose) > 6 else "")
            lines.append(f"Not in every group (excluded from paired tests): {shown}.")
        self._pair_status.setText("\n".join(lines))
        self._pair_status.setVisible(True)

    def _remove_group(self, widget: FolderGroupWidget):
        if widget in self._group_widgets:
            name = widget.group_name
            self._group_widgets.remove(widget)
            self._groups_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
            self._refresh_pairing()
            if name in self._all_group_data:
                del self._all_group_data[name]
                self._replot_from_existing()

    def _on_group_color_changed(self, widget: FolderGroupWidget):
        gd = self._all_group_data.get(widget.group_name)
        if gd is not None:
            gd.color = widget.color
            self._replot_from_existing()

    def _replot_from_existing(self):
        if not self._all_group_data:
            return
        self._log("Replotting with updated settings…")
        self._generate_plots(self._all_group_data, self._stats_results)

    # ─────────────────────────────────────────────────────────────────────
    # Run analysis
    # ─────────────────────────────────────────────────────────────────────

    def _collect_settings(self) -> AnalysisSettings:
        roi_mode = "static" if self._roi_static.isChecked() else "body"
        return AnalysisSettings(
            camera_width=self._cam_width.value(),
            camera_height=self._cam_height.value(),
            metadata_cols=self._meta_cols.value(),
            fps=self._cam_fps.value(),
            roi_mode=roi_mode,
            roi_x=self._roi_x.value(),
            roi_y=self._roi_y.value(),
            roi_w=self._roi_w.value(),
            roi_h=self._roi_h.value(),
            colormap=self._colormap.currentText(),
            t_min=self._t_min.value(),
            t_max=self._t_max.value(),
            apply_emissivity=self._emiss_cb.isChecked(),
            epsilon_sensor=self._eps_sensor.value(),
            tcore_enabled=self._tcore_cb.isChecked(),
            tcore_sampling_interval_sec=self._tc_sample.value(),
            tcore_averaging_window_min=self._tc_window.value(),
            tcore_slope=self._tc_slope.value(),
            tcore_intercept=self._tc_intercept.value(),
            baseline_end_sec=BASELINE_END_SEC,
            settle_s=self._settle_s.value(),
            min_coverage=self._min_cov.value() / 100.0,
            paired_design=self._paired_cb.isChecked(),
            timepoints_s=self._timepoints_s(),
            timepoint_signal=self._tp_signal.currentData(),
            timepoint_window_s=self._tp_window.value(),
            arena_enabled=self._arena_cb.isChecked(),
            arena_x=self._arena_x.value(),
            arena_y=self._arena_y.value(),
            arena_w=self._arena_w.value(),
            arena_h=self._arena_h.value(),
            output_dir=self._output_dir,
        )

    def _run_analysis(self):
        valid_groups = [gw for gw in self._group_widgets if gw.is_valid()]
        if not valid_groups:
            QMessageBox.warning(self, "No Groups",
                                "Please add at least one experiment group with a valid folder.")
            return

        if self._parse_worker and self._parse_worker.isRunning():
            QMessageBox.information(self, "Running", "Analysis already in progress.")
            return

        settings = self._collect_settings()
        if settings.timepoint_signal == "tcore_estimated" and settings.timepoints_s \
                and not settings.tcore_enabled:
            QMessageBox.warning(self, "T_core disabled",
                                "Time-point signal is Estimated T_core, but T_core "
                                "estimation is off (Advanced settings → T_core).")
            return
        if settings.paired_design:
            by_group = self._animals_by_group()
            shared = (set.intersection(*[set(v) for v in by_group.values()])
                      if len(by_group) >= 2 else set())
            if len(shared) < 2:
                QMessageBox.warning(
                    self, "Paired design",
                    "Paired design is ticked, but fewer than 2 animals have the same "
                    "name in every group.\n\nUse identical animal names for the same "
                    "mouse in each group, or untick the paired option.")
                return
        self._run_settings = settings
        self._parse_progress.setValue(0)
        self._plot_scroll.clear()
        self._log_edit.clear()
        self._log("Starting analysis…")

        groups = []
        for gw in valid_groups:
            files = gw.get_txt_files()
            if not files:
                self._log(f"⚠ {gw.group_name}: no .txt files found in folder — skipped.")
                continue
            self._log(f"  {gw.group_name}: {len(files)} file(s) found")
            gdata = GroupData(gw.group_name, gw.color, files)
            gdata.epsilon = gw.emissivity
            gdata.coat_label = gw.coat_label
            groups.append(gdata)

        if not groups:
            QMessageBox.warning(self, "No Data", "No .txt files found in any folder.")
            return

        # One timestamped folder per run: <project>/analysis/<ts>/
        self._run_dir = (Path(self._output_dir) / "analysis"
                         / datetime.now().strftime("%Y-%m-%d_%H%M%S"))
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._open_run_btn.setVisible(False)
        self._write_settings_json(settings, groups)

        self._parse_worker = ParseWorker(groups, settings, str(self._run_dir))
        self._parse_worker.progress.connect(self._parse_progress.setValue)
        self._parse_worker.log_message.connect(self._log)
        self._parse_worker.finished.connect(self._on_analysis_done)
        self._parse_worker.error.connect(self._on_worker_error)
        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")
        self.statusBar().showMessage("Analysis running — please wait…")
        self._parse_worker.start()

    def _restore_run_button(self):
        self._run_btn.setEnabled(True)
        self._run_btn.setText("Run Analysis")

    @pyqtSlot(dict, dict)
    def _on_analysis_done(self, all_group_data: dict, stats_results: dict):
        self._all_group_data = all_group_data
        self._stats_results = stats_results
        self._restore_run_button()
        self._log("✓ Analysis complete — generating plots…")
        self._tabs.setCurrentWidget(self._plot_scroll)
        self._generate_plots(all_group_data, stats_results)
        n_plots = len(self._plot_scroll._canvases)
        saved = self._autosave_results()
        self.statusBar().showMessage(
            f"Analysis complete — {n_plots} plot(s) saved to {self._run_dir}"
            if saved else f"Analysis complete — {n_plots} plot(s); see Log for save errors.")
        self._open_run_btn.setVisible(True)

        # Auto-trigger video export for groups that requested it
        wanted = [gw.group_name for gw in self._group_widgets
                  if gw.export_video and gw.is_valid()]
        if wanted:
            self._export_videos(wanted)

    @pyqtSlot(str)
    def _on_worker_error(self, msg: str):
        self._restore_run_button()
        self._log(f"✗ Error: {msg}")
        self.statusBar().showMessage("Analysis failed — see Console Log.")
        QMessageBox.critical(self, "Analysis Error", msg)

    # ─────────────────────────────────────────────────────────────────────
    # Plot generation
    # ─────────────────────────────────────────────────────────────────────

    def _generate_plots(self, all_group_data: dict, stats_results: dict):
        from ..plotting.figure_builder import (
            build_timecourse, build_tcore_estimated, build_animal_overview,
            build_tracking_qc, build_group_comparison_bars, build_timepoint_bars,
        )

        self._plot_scroll.clear()
        n_groups = len(all_group_data)
        rs = getattr(self, "_run_settings", None) or self._collect_settings()

        try:
            # ── Group time courses (every animal + group mean) ────────────
            self._plot_scroll.add_figure(
                build_timecourse(all_group_data, "delta"), "Temperature change", "Overview")
            self._plot_scroll.add_figure(
                build_timecourse(all_group_data, "abs"), "Absolute temperature", "Overview")

            if any(gd.tcore_mean is not None for gd in all_group_data.values()):
                self._plot_scroll.add_figure(
                    build_tcore_estimated(all_group_data), "Estimated T_core", "T_core")

            # ── Tracking quality across all animals ───────────────────────
            qc_df = stats_results.get("qc_df")
            if qc_df is not None and len(qc_df):
                self._plot_scroll.add_figure(
                    build_tracking_qc(all_group_data, qc_df, rs.min_coverage),
                    "Tracking quality", "Quality")

            # ── One overview per recording ────────────────────────────────
            for g_name, gd in all_group_data.items():
                for stem, roi_df in gd.roi_dfs.items():
                    if len(roi_df) == 0:
                        continue
                    self._plot_scroll.add_figure(
                        build_animal_overview(
                            roi_df, f"{g_name} — {animal_from_stem(stem)}",
                            self._cam_width.value(), self._cam_height.value(),
                            gd.per_file_metrics.get(stem)),
                        f"Animal {g_name} {animal_from_stem(stem)}", "Per animal")

            # ── Time-point comparison (when time points were set) ─────────
            if rs.timepoints_s and stats_results.get("timepoint_df") is not None:
                self._plot_scroll.add_figure(
                    build_timepoint_bars(all_group_data, stats_results, rs),
                    "Time-point comparison", "Time points")

            # ── Group comparison (only with 2+ groups) ────────────────────
            if n_groups >= 2 and stats_results.get("per_file_df") is not None:
                self._plot_scroll.add_figure(
                    build_group_comparison_bars(all_group_data, stats_results),
                    "Group Comparison Bars", "Statistics")

            self._log(f"✓ {len(self._plot_scroll._canvases)} plots generated.")

        except Exception as exc:
            self._log(f"⚠ Plot generation error: {exc}")
            log.exception("Plot generation error", exc_info=exc)
    # ─────────────────────────────────────────────────────────────────────
    # Video export
    # ─────────────────────────────────────────────────────────────────────

    def _export_video_prompt(self):
        if not self._all_group_data:
            QMessageBox.information(self, "No Data",
                                    "Run analysis first before exporting video.")
            return
        self._export_videos(list(self._all_group_data.keys()))

    def _video_jobs(self, group_name: str) -> list:
        """One group-tile job plus one job per animal for ``group_name``.

        Names: <date>_<group>_tile.mp4 and <date>_<group>_<animal>.mp4,
        all inside <run folder>/videos/<group>/.
        """
        from ..utils import naming
        gd = self._all_group_data.get(group_name)
        file_paths = list(gd.file_paths) if gd is not None else []
        if not file_paths:
            gw = next((w for w in self._group_widgets
                       if w.group_name == group_name), None)
            file_paths = gw.get_txt_files() if gw else []
        if not file_paths:
            return []

        run_date = naming.today()
        base = self._run_dir or (Path(self._output_dir) / "analysis")
        vdir = base / "videos" / naming.safe(group_name)
        vdir.mkdir(parents=True, exist_ok=True)

        animals = [naming.animal_from_stem(Path(f).stem) for f in file_paths]
        dates = [naming.date_from_stem(Path(f).stem, run_date) for f in file_paths]
        jobs = []
        if len(file_paths) > 1:
            jobs.append({
                "file_paths": file_paths, "labels": animals,
                "group": group_name, "date": dates[0],
                "output_path": str(vdir / naming.build(
                    "tile", "mp4", date=dates[0], group=group_name)),
            })
        for fp, animal, date in zip(file_paths, animals, dates):
            jobs.append({
                "file_paths": [fp], "labels": [animal],
                "group": group_name, "date": date,
                "output_path": str(vdir / naming.build(
                    "roi_video", "mp4", date=date, group=group_name, animal=animal)),
            })
        return jobs

    def _export_videos(self, group_names: list):
        if self._video_worker and self._video_worker.isRunning():
            self._log("Video export already running.")
            return
        jobs = []
        for name in group_names:
            group_jobs = self._video_jobs(name)
            if not group_jobs:
                self._log(f"⚠ {name}: no .txt files — no video exported.")
            jobs += group_jobs
        if not jobs:
            return
        self._video_progress.setValue(0)
        self._video_worker = VideoWorker(jobs, self._collect_settings())
        self._video_worker.progress.connect(self._video_progress.setValue)
        self._video_worker.log_message.connect(self._log)
        self._video_worker.finished.connect(
            lambda p: self._log(f"✓ Videos saved → {p}"))
        self._video_worker.error.connect(
            lambda e: self._log(f"✗ Video error: {e}"))
        self._video_worker.start()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _write_settings_json(self, settings: AnalysisSettings, groups: list):
        """Record exactly what was run, next to the results."""
        try:
            payload = {
                "started": datetime.now().isoformat(timespec="seconds"),
                "settings": dataclasses.asdict(settings),
                "groups": {g.name: [str(f) for f in g.file_paths] for g in groups},
                "emissivity_by_group": {g.name: {"coat": g.coat_label, "epsilon": g.epsilon}
                                        for g in groups},
            }
            (self._run_dir / "settings.json").write_text(
                json.dumps(payload, indent=2), encoding="utf-8")
        except Exception as exc:
            self._log(f"⚠ Could not write settings.json: {exc}")

    def _autosave_results(self) -> bool:
        """Save every plot as PNG and the log into the current run folder.

        CSV data is already written by ParseWorker into <run>/data/.
        Returns True when nothing failed.
        """
        if self._run_dir is None:
            return False
        plots_dir = self._run_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        src_dir = self._run_dir / "source_data"
        src_dir.mkdir(parents=True, exist_ok=True)
        ok = True

        for i, (fig, title) in enumerate(self._plot_scroll.figures()):
            fname = naming.build(f"{i + 1:02d}_{title or 'plot'}", "png",
                                 date=self._run_dir.name[:10])
            try:
                fig.savefig(str(plots_dir / fname),
                            dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            except Exception as exc:
                ok = False
                self._log(f"⚠ Could not save plot '{title}': {exc}")
            # Source data for the figure, same stem as the PNG
            try:
                write_figure_source_data(
                    fig, title, src_dir / (Path(fname).stem + ".csv"),
                    self._stats_results.get("per_file_df"),
                    self._stats_results.get("timepoint_df"))
            except Exception as exc:
                ok = False
                self._log(f"⚠ Could not save source data for '{title}': {exc}")

        self._log(f"✓ Results saved → {self._run_dir}")
        try:
            (self._run_dir / "log.txt").write_text(
                self._log_edit.toPlainText(), encoding="utf-8")
        except Exception as exc:
            ok = False
            self._log(f"⚠ Could not write log.txt: {exc}")
        return ok

    def _log(self, msg: str):
        ts = datetime.now().strftime("%H:%M:%S")
        self._log_edit.append(f"<span style='color:#68A07B'>[{ts}]</span> {msg}")

    def _show_about(self):
        msg = QMessageBox(self)
        msg.setWindowTitle("About Glöd")
        msg.setText(
            "<b>Glöd v1.0</b><br>"
            "Thermal Imaging ROI Analysis<br><br>"
            "<i>Swedish: glow / live embers</i><br><br>"
            "<b>Author</b><br>"
            "Param Valiathan<br>"
            "Ernfors Lab, Karolinska Institutet<br><br>"
            "T<sub>core</sub> estimation algorithm:<br>"
            "van der Vinne et al. (2020)<br>"
            "<i>Sci Rep</i> 10:20680<br>"
            "DOI: 10.1038/s41598-020-77786-5<br><br>"
            "Waveshare MI48x3 thermal camera<br>"
            "80 × 62 pixels"
        )
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()

    def closeEvent(self, event):
        if self._live_tab._manager.is_running():
            self._live_tab._manager.stop_all()
        for worker in [self._parse_worker, self._video_worker]:
            if worker and worker.isRunning():
                worker.requestInterruption()
                worker.wait(3000)
        event.accept()
