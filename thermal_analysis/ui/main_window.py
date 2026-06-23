"""
Glöd — Main application window.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QSplitter, QScrollArea, QVBoxLayout,
    QHBoxLayout, QLabel, QSpinBox, QDoubleSpinBox, QComboBox,
    QPushButton, QProgressBar, QGroupBox, QRadioButton,
    QButtonGroup, QCheckBox, QFrame, QTabWidget, QTextEdit,
    QFileDialog, QMessageBox,
)
from PyQt6.QtGui import QIcon, QFont
from PyQt6.QtCore import Qt, pyqtSlot

from .plot_panel import PlotScrollPanel
from .group_widget import FolderGroupWidget
from ..processing.workers import ParseWorker, VideoWorker, GroupData
from ..utils.config import (
    AnalysisSettings, CAMERA_WIDTH, CAMERA_HEIGHT, CAMERA_FPS,
    METADATA_COLS, ROI_DEFAULT_W, ROI_DEFAULT_H, ROI_DEFAULT_X,
    ROI_DEFAULT_Y, COLORMAPS, COLORMAP_DEFAULT,
    T_MIN_DEFAULT, T_MAX_DEFAULT,
    TCORE_SAMPLING_INTERVAL_SEC, TCORE_AVERAGING_WINDOW_MIN,
    TCORE_SLOPE, TCORE_INTERCEPT,
    BASELINE_END_SEC, OUTPUT_DIR_PREFIX,
)

log = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """Glöd main window."""

    def __init__(self, icon_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Glöd — Thermal Imaging Analysis")
        self.setMinimumSize(1100, 700)
        self.resize(1400, 820)

        if icon_path and Path(icon_path).exists():
            self.setWindowIcon(QIcon(icon_path))

        self._group_widgets: List[FolderGroupWidget] = []
        self._parse_worker: Optional[ParseWorker] = None
        self._video_worker: Optional[VideoWorker] = None
        self._all_group_data: dict = {}
        self._stats_results: dict = {}
        self._output_dir: str = str(Path.home() / "glod_output")

        self._build_ui()
        self._connect_signals()

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

        # ── Main splitter (left control | right view) ─────────────────────
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        splitter.addWidget(self._build_control_panel())
        splitter.addWidget(self._build_view_panel())
        splitter.setSizes([340, 1060])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        root_layout.addWidget(splitter, stretch=1)

        # ── Status bar ────────────────────────────────────────────────────
        self.statusBar().showMessage("Ready — add experimental groups to begin.")

    def _build_header(self) -> QWidget:
        header = QFrame()
        header.setFixedHeight(56)
        header.setStyleSheet(
            "QFrame { background: #EEF2F0; border-bottom: 1px solid #DEE2E6; }")
        layout = QHBoxLayout(header)
        layout.setContentsMargins(16, 0, 16, 0)

        # Ember/glow icon placeholder (replaced by real icon if available)
        icon_label = QLabel("🔥")
        icon_label.setFont(QFont("Segoe UI Emoji", 22))
        layout.addWidget(icon_label)

        title = QLabel("Glöd")
        title.setObjectName("TitleLabel")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        layout.addWidget(title)

        sub = QLabel("Thermal Imaging ROI Analysis")
        sub.setObjectName("SubtitleLabel")
        sub.setStyleSheet("color: #6C757D; font-size: 12px; margin-left: 6px;")
        layout.addWidget(sub)

        layout.addStretch()

        # About button
        about_btn = QPushButton("About")
        about_btn.setObjectName("SecondaryButton")
        about_btn.setFixedHeight(30)
        about_btn.clicked.connect(self._show_about)
        layout.addWidget(about_btn)

        return header

    def _build_control_panel(self) -> QWidget:
        wrapper = QWidget()
        wrapper.setMinimumWidth(260)
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.setSpacing(0)

        scroll = QScrollArea()
        scroll.setObjectName("ControlScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        container = QWidget()
        container.setObjectName("ControlPanel")
        container.setStyleSheet("background: #EEF2F0;")
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self._build_camera_group())
        layout.addWidget(self._build_roi_group())
        layout.addWidget(self._build_display_group())
        layout.addWidget(self._build_tcore_group())
        layout.addWidget(self._build_groups_section())
        layout.addWidget(self._build_actions())
        layout.addStretch()

        scroll.setWidget(container)
        wrapper_layout.addWidget(scroll, 1)

        footer = QLabel("P.Valiathan 2026  ·  Karolinska Institutet")
        footer.setAlignment(Qt.AlignmentFlag.AlignCenter)
        footer.setStyleSheet(
            "color: #B8C4BB; font-size: 9px; padding: 4px 0;"
            "border-top: 1px solid #DEE2E6; background: #EEF2F0;"
        )
        wrapper_layout.addWidget(footer)

        return wrapper

    # ── Parameter group builders ──────────────────────────────────────────

    def _build_camera_group(self) -> QGroupBox:
        box = QGroupBox("Camera Settings")
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

    def _build_roi_group(self) -> QGroupBox:
        box = QGroupBox("ROI Settings")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(6, 4, 6, 6)
        layout.setSpacing(3)

        mode_row = QHBoxLayout()
        self._roi_dynamic = QRadioButton("Dynamic (hottest pixel)")
        self._roi_static = QRadioButton("Fixed box")
        self._roi_dynamic.setChecked(True)
        self._roi_mode_grp = QButtonGroup()
        self._roi_mode_grp.addButton(self._roi_dynamic)
        self._roi_mode_grp.addButton(self._roi_static)
        mode_row.addWidget(self._roi_dynamic)
        mode_row.addWidget(self._roi_static)
        layout.addLayout(mode_row)

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

        # Enable/disable coordinate fields based on mode
        self._roi_dynamic.toggled.connect(lambda checked: self._toggle_roi_coords(not checked))
        self._toggle_roi_coords(False)  # dynamic is default

        return box

    def _toggle_roi_coords(self, enabled: bool):
        for w in [self._roi_x, self._roi_y]:
            w.setEnabled(enabled)

    def _build_display_group(self) -> QGroupBox:
        box = QGroupBox("Display Settings")
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

        emiss_row = QHBoxLayout()
        self._emiss_cb = QCheckBox("Emissivity correction")
        self._emiss_cb.setChecked(False)
        emiss_row.addWidget(self._emiss_cb)
        layout.addLayout(emiss_row)

        emiss_vals = QHBoxLayout()
        emiss_vals.addWidget(QLabel("ε mouse"))
        self._eps_mouse = QDoubleSpinBox()
        self._eps_mouse.setRange(0.01, 1.0); self._eps_mouse.setValue(0.93); self._eps_mouse.setDecimals(2)
        emiss_vals.addWidget(self._eps_mouse)
        emiss_vals.addWidget(QLabel("ε sensor"))
        self._eps_sensor = QDoubleSpinBox()
        self._eps_sensor.setRange(0.01, 1.0); self._eps_sensor.setValue(1.0); self._eps_sensor.setDecimals(2)
        emiss_vals.addWidget(self._eps_sensor)
        layout.addLayout(emiss_vals)

        return box

    def _build_tcore_group(self) -> QGroupBox:
        box = QGroupBox("T_core Estimation")
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

        self._groups_layout = QVBoxLayout()
        self._groups_layout.setSpacing(6)
        layout.addLayout(self._groups_layout)

        return container

    def _build_actions(self) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setObjectName("RunButton")
        self._run_btn.setFixedHeight(36)
        self._run_btn.clicked.connect(self._run_analysis)
        btn_row.addWidget(self._run_btn)

        self._video_btn = QPushButton("Export Video")
        self._video_btn.setFixedHeight(36)
        self._video_btn.setObjectName("SecondaryButton")
        self._video_btn.clicked.connect(self._export_video_prompt)
        btn_row.addWidget(self._video_btn)
        layout.addLayout(btn_row)

        # Output dir row
        out_row = QHBoxLayout()
        out_row.addWidget(QLabel("Output dir:"))
        self._out_dir_label = QLabel(self._output_dir)
        self._out_dir_label.setStyleSheet("font-size: 11px; color: #6C757D;")
        self._out_dir_label.setWordWrap(True)
        out_row.addWidget(self._out_dir_label, stretch=1)
        browse_out_btn = QPushButton("…")
        browse_out_btn.setObjectName("SmallButton")
        browse_out_btn.setFixedSize(28, 26)
        browse_out_btn.clicked.connect(self._browse_output_dir)
        out_row.addWidget(browse_out_btn)
        layout.addLayout(out_row)

        # Progress bars
        parse_row = QHBoxLayout()
        parse_row.addWidget(QLabel("Parsing:"))
        self._parse_progress = QProgressBar()
        self._parse_progress.setValue(0)
        self._parse_progress.setTextVisible(True)
        parse_row.addWidget(self._parse_progress)
        layout.addLayout(parse_row)

        video_row = QHBoxLayout()
        video_row.addWidget(QLabel("Video:  "))
        self._video_progress = QProgressBar()
        self._video_progress.setValue(0)
        self._video_progress.setTextVisible(True)
        video_row.addWidget(self._video_progress)
        layout.addLayout(video_row)

        return container

    def _build_view_panel(self) -> QWidget:
        outer = QWidget()
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.setSpacing(0)

        # ── Toolbar row above tabs (Save All button lives here) ────────────
        toolbar = QFrame()
        toolbar.setFixedHeight(38)
        toolbar.setStyleSheet(
            "QFrame { background: #EEF2F0; border-bottom: 1px solid #DEE2E6; }")
        tb_layout = QHBoxLayout(toolbar)
        tb_layout.setContentsMargins(8, 0, 8, 0)
        tb_layout.addStretch()

        self._save_all_btn = QPushButton("💾  Save All")
        self._save_all_btn.setObjectName("SecondaryButton")
        self._save_all_btn.setFixedHeight(28)
        self._save_all_btn.setEnabled(False)
        self._save_all_btn.setToolTip(
            "Save all plots as PNG images and CSV data to a chosen folder")
        self._save_all_btn.clicked.connect(self._save_all)
        tb_layout.addWidget(self._save_all_btn)

        outer_layout.addWidget(toolbar)

        # ── Analysis Plots tab ────────────────────────────────────────────
        self._tabs = QTabWidget()
        self._plot_scroll = PlotScrollPanel()
        self._tabs.addTab(self._plot_scroll, "Analysis Plots")

        # ── Console Log tab ───────────────────────────────────────────────
        self._log_edit = QTextEdit()
        self._log_edit.setReadOnly(True)
        self._tabs.addTab(self._log_edit, "Console Log")

        outer_layout.addWidget(self._tabs, stretch=1)
        return outer

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
        self._group_widgets.append(gw)
        self._groups_layout.addWidget(gw)

    def _remove_group(self, widget: FolderGroupWidget):
        if widget in self._group_widgets:
            name = widget.group_name
            self._group_widgets.remove(widget)
            self._groups_layout.removeWidget(widget)
            widget.setParent(None)
            widget.deleteLater()
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
        return AnalysisSettings(
            camera_width=self._cam_width.value(),
            camera_height=self._cam_height.value(),
            metadata_cols=self._meta_cols.value(),
            fps=self._cam_fps.value(),
            roi_mode="dynamic" if self._roi_dynamic.isChecked() else "static",
            roi_x=self._roi_x.value(),
            roi_y=self._roi_y.value(),
            roi_w=self._roi_w.value(),
            roi_h=self._roi_h.value(),
            colormap=self._colormap.currentText(),
            t_min=self._t_min.value(),
            t_max=self._t_max.value(),
            apply_emissivity=self._emiss_cb.isChecked(),
            epsilon_mouse=self._eps_mouse.value(),
            epsilon_sensor=self._eps_sensor.value(),
            tcore_enabled=self._tcore_cb.isChecked(),
            tcore_sampling_interval_sec=self._tc_sample.value(),
            tcore_averaging_window_min=self._tc_window.value(),
            tcore_slope=self._tc_slope.value(),
            tcore_intercept=self._tc_intercept.value(),
            baseline_end_sec=BASELINE_END_SEC,
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
            groups.append(gdata)

        if not groups:
            QMessageBox.warning(self, "No Data", "No .txt files found in any folder.")
            return

        self._parse_worker = ParseWorker(groups, settings, self._output_dir)
        self._parse_worker.progress.connect(self._parse_progress.setValue)
        self._parse_worker.log_message.connect(self._log)
        self._parse_worker.finished.connect(self._on_analysis_done)
        self._parse_worker.error.connect(self._on_worker_error)
        self._run_btn.setEnabled(False)
        self._run_btn.setText("Running…")
        self._save_all_btn.setEnabled(False)
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
        self._tabs.setCurrentIndex(0)
        self._generate_plots(all_group_data, stats_results)
        n_plots = len(self._plot_scroll._canvases)
        self._save_all_btn.setEnabled(True)
        self.statusBar().showMessage(
            f"Analysis complete — {n_plots} plot(s) generated.")

        # Auto-trigger video export for groups that requested it
        for gw in self._group_widgets:
            if gw.export_video and gw.is_valid():
                self._export_video_for_group(gw)

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
            build_normalized_roi, build_rate_of_change, build_roi_max_vs_mean,
            build_max_pixel_temperature, build_max_pixel_movement,
            build_absolute_comparison, build_tcore_estimated,
            build_tcore_rate_of_change, build_group_comparison_bars,
            build_group_violin_box,
        )

        self._plot_scroll.clear()
        n_groups = len(all_group_data)

        try:
            # ── Multi-group time-series plots ─────────────────────────────
            self._plot_scroll.add_figure(
                build_normalized_roi(all_group_data), "Normalised ROI")
            self._plot_scroll.add_figure(
                build_rate_of_change(all_group_data), "Rate of Change (ROI)")
            self._plot_scroll.add_figure(
                build_absolute_comparison(all_group_data), "Absolute ROI")

            if any(gd.tcore_mean is not None for gd in all_group_data.values()):
                self._plot_scroll.add_figure(
                    build_tcore_estimated(all_group_data), "Estimated T_core")
                self._plot_scroll.add_figure(
                    build_tcore_rate_of_change(all_group_data), "T_core Rate of Change")

            # ── Per-file plots (first 6 camera files to avoid overload) ───
            shown = 0
            for g_name, gd in all_group_data.items():
                for stem, roi_df in gd.roi_dfs.items():
                    if shown >= 6:
                        break
                    if len(roi_df) > 0:
                        self._plot_scroll.add_figure(
                            build_roi_max_vs_mean(roi_df, f"{g_name} — {stem}"),
                            f"ROI Max/Mean: {stem}")
                        self._plot_scroll.add_figure(
                            build_max_pixel_temperature(roi_df, f"{g_name} — {stem}"),
                            f"Max Pixel: {stem}")
                        w = self._cam_width.value()
                        h = self._cam_height.value()
                        self._plot_scroll.add_figure(
                            build_max_pixel_movement(roi_df, w, h, f"{g_name} — {stem}"),
                            f"Trajectory: {stem}")
                    shown += 1

            # ── Group comparison (only with 2+ groups) ────────────────────
            if n_groups >= 2 and stats_results.get("per_file_df") is not None:
                self._plot_scroll.add_figure(
                    build_group_comparison_bars(all_group_data, stats_results),
                    "Group Comparison Bars")
                self._plot_scroll.add_figure(
                    build_group_violin_box(all_group_data, stats_results),
                    "Group Violin / Box")

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

        group_names = list(self._all_group_data.keys())
        # If only one group, use it directly
        if len(group_names) == 1:
            self._export_video_for_name(group_names[0])
        else:
            from PyQt6.QtWidgets import QInputDialog
            chosen, ok = QInputDialog.getItem(
                self, "Select Group", "Export video for group:", group_names, 0, False)
            if ok:
                self._export_video_for_name(chosen)

    def _export_video_for_name(self, group_name: str):
        gd = self._all_group_data.get(group_name)
        if gd is None:
            return
        # Prefer the full paths stored in GroupData; fall back to group widget
        file_paths = gd.file_paths
        if not file_paths:
            gw = next((w for w in self._group_widgets if w.group_name == group_name), None)
            file_paths = gw.get_txt_files() if gw else []
        if not file_paths:
            QMessageBox.warning(self, "No Files", f"No .txt files found for {group_name}")
            return

        out_path = str(Path(self._output_dir) /
                       f"{OUTPUT_DIR_PREFIX}_{group_name}_roi_video.mp4")
        settings = self._collect_settings()
        self._start_video_export(file_paths, out_path, settings)

    def _export_video_for_group(self, gw: FolderGroupWidget):
        file_paths = gw.get_txt_files()
        if not file_paths:
            return
        out_path = str(Path(self._output_dir) /
                       f"{OUTPUT_DIR_PREFIX}_{gw.group_name}_roi_video.mp4")
        settings = self._collect_settings()
        self._start_video_export(file_paths, out_path, settings)

    def _start_video_export(self, file_paths, out_path, settings):
        if self._video_worker and self._video_worker.isRunning():
            self._log("Video export already running.")
            return
        self._video_progress.setValue(0)
        self._video_worker = VideoWorker(file_paths, out_path, settings)
        self._video_worker.progress.connect(self._video_progress.setValue)
        self._video_worker.log_message.connect(self._log)
        self._video_worker.finished.connect(
            lambda p: self._log(f"✓ Video saved: {p}"))
        self._video_worker.error.connect(
            lambda e: self._log(f"✗ Video error: {e}"))
        self._video_worker.start()

    # ─────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────

    def _save_all(self):
        """Save every displayed plot as a PNG and export CSV data to a chosen folder."""
        if not self._all_group_data:
            QMessageBox.information(self, "Nothing to Save",
                                    "Run analysis first.")
            return

        folder = QFileDialog.getExistingDirectory(
            self, "Choose folder to save plots and CSV files", self._output_dir)
        if not folder:
            return

        import pandas as pd
        from pathlib import Path

        save_dir = Path(folder)
        errors: list[str] = []

        # ── Save each plot as PNG ─────────────────────────────────────────
        for i, (fig, title) in enumerate(self._plot_scroll.figures()):
            safe = "".join(c if c.isalnum() or c in " _-" else "_" for c in title)
            png_path = save_dir / f"{i + 1:02d}_{safe or 'plot'}.png"
            try:
                fig.savefig(str(png_path), dpi=150, bbox_inches="tight",
                            facecolor=fig.get_facecolor())
            except Exception as exc:
                errors.append(f"PNG '{title}': {exc}")

        # ── Save per-file ROI CSVs and group aggregates ───────────────────
        for g_name, gd in self._all_group_data.items():
            safe_g = "".join(c if c.isalnum() or c in " _-" else "_" for c in g_name)

            for stem, roi_df in gd.roi_dfs.items():
                safe_s = "".join(c if c.isalnum() or c in " _-" else "_" for c in stem)
                try:
                    roi_df.to_csv(str(save_dir / f"{safe_g}_{safe_s}_roi.csv"), index=False)
                except Exception as exc:
                    errors.append(f"ROI CSV '{stem}': {exc}")

            for stem, tcore_df in gd.tcore_dfs.items():
                if tcore_df is None:
                    continue
                safe_s = "".join(c if c.isalnum() or c in " _-" else "_" for c in stem)
                try:
                    tcore_df.to_csv(
                        str(save_dir / f"{safe_g}_{safe_s}_tcore.csv"), index=False)
                except Exception as exc:
                    errors.append(f"T_core CSV '{stem}': {exc}")

            if gd.common_t is not None and gd.roi_mean is not None:
                try:
                    agg: dict = {
                        "time_s": gd.common_t,
                        "roi_mean": gd.roi_mean,
                        "roi_sem": gd.roi_sem if gd.roi_sem is not None
                                   else [None] * len(gd.common_t),
                    }
                    if gd.tcore_mean is not None:
                        agg["tcore_mean"] = gd.tcore_mean
                        agg["tcore_sem"] = (gd.tcore_sem if gd.tcore_sem is not None
                                            else [None] * len(gd.common_t))
                    pd.DataFrame(agg).to_csv(
                        str(save_dir / f"{safe_g}_group_agg.csv"), index=False)
                except Exception as exc:
                    errors.append(f"Agg CSV '{g_name}': {exc}")

        # ── Save stats per-file table ─────────────────────────────────────
        per_file_df = self._stats_results.get("per_file_df")
        if per_file_df is not None and not per_file_df.empty:
            try:
                per_file_df.to_csv(str(save_dir / "stats_per_file.csv"), index=False)
            except Exception as exc:
                errors.append(f"Stats CSV: {exc}")

        n_plots = len(self._plot_scroll.figures())
        if errors:
            QMessageBox.warning(
                self, "Save Completed With Warnings",
                f"Saved with {len(errors)} error(s):\n" + "\n".join(errors[:6]))
        else:
            self._log(f"✓ Saved {n_plots} plot(s) and CSV data → {folder}")
            QMessageBox.information(self, "Saved",
                                    f"All files saved to:\n{folder}")

    def _browse_output_dir(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self._output_dir)
        if folder:
            self._output_dir = folder
            self._out_dir_label.setText(folder)

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
        for worker in [self._parse_worker, self._video_worker]:
            if worker and worker.isRunning():
                worker.requestInterruption()
                worker.wait(3000)
        event.accept()
