"""
QThread workers for Glöd.

ParseWorker  — parallel file parsing + group aggregation + stats
VideoWorker  — multi-panel video export
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from PyQt6.QtCore import QThread, pyqtSignal

from .parser import parse_files_parallel, result_to_dataframes
from .roi_analysis import smooth_series
from .stats import GroupStatsTester, extract_per_file_metrics
from .video_exporter import export_video
from ..utils.config import AnalysisSettings, DOWN_SAMPLE_PERIOD_SEC, OUTPUT_DIR_PREFIX

log = logging.getLogger(__name__)


# ── Data structures shared between worker and main thread ────────────────────

class GroupData:
    """Holds all computed data for one experimental group."""
    def __init__(self, name: str, color: str, file_paths: List[str]):
        self.name = name
        self.color = color
        self.file_paths = file_paths

        # Set after parsing
        self.roi_dfs: Dict[str, pd.DataFrame] = {}      # stem → roi_df
        self.tcore_dfs: Dict[str, Optional[pd.DataFrame]] = {}

        # Group aggregation (common time axis at 30-s steps)
        self.common_t: Optional[np.ndarray] = None
        self.roi_mean: Optional[np.ndarray] = None
        self.roi_sem: Optional[np.ndarray] = None
        self.tcore_mean: Optional[np.ndarray] = None
        self.tcore_sem: Optional[np.ndarray] = None

        self.per_file_metrics: Dict[str, dict] = {}


# ── Parse Worker ─────────────────────────────────────────────────────────────

class ParseWorker(QThread):
    """
    Background worker that:
      1. Parses all files in parallel (ProcessPoolExecutor)
      2. Aggregates group statistics
      3. Runs statistical tests
      4. Saves CSV outputs
      5. Emits results to the main thread
    """

    # Signals
    progress = pyqtSignal(int)          # 0-100 overall progress
    log_message = pyqtSignal(str)       # console log line
    finished = pyqtSignal(dict, dict)   # (all_group_data, stats_results)
    error = pyqtSignal(str)             # fatal error message

    def __init__(self, groups: List[GroupData], settings: AnalysisSettings,
                 output_root: str, parent=None):
        super().__init__(parent)
        self.groups = groups
        self.settings = settings
        self.output_root = Path(output_root)

    def run(self):
        try:
            self._run()
        except Exception as exc:
            log.exception("ParseWorker fatal error: %s", exc)
            self.error.emit(str(exc))

    def _run(self):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = self.output_root / f"{OUTPUT_DIR_PREFIX}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_files = [fp for g in self.groups for fp in g.file_paths]
        total_files = len(all_files) or 1
        files_done = 0

        all_group_data: Dict[str, GroupData] = {}

        for g in self.groups:
            self.log_message.emit(f"[{g.name}] Parsing {len(g.file_paths)} file(s)…")
            g_out_dir = out_dir / g.name
            g_out_dir.mkdir(parents=True, exist_ok=True)

            # ── Parallel parsing ──────────────────────────────────────────
            results = parse_files_parallel(
                g.file_paths,
                self.settings,
                progress_callback=None,   # overall progress emitted per-file below
            )

            for path, result in results.items():
                stem = Path(path).stem
                files_done += 1

                if result is None:
                    self.log_message.emit(f"  ⚠ Skipped {stem} (parse error)")
                    self.progress.emit(int(files_done / total_files * 80))
                    continue

                roi_df, tcore_df = result_to_dataframes(result)
                g.roi_dfs[stem] = roi_df
                g.tcore_dfs[stem] = tcore_df

                # ── Per-file CSV export ───────────────────────────────────
                roi_df.to_csv(g_out_dir / f"{stem}_roi_data.csv", index=False)
                self.log_message.emit(
                    f"  ✓ {stem}: {result['n_frames']} frames"
                )

                if tcore_df is not None:
                    s = self.settings
                    header = (
                        f"# Settings: fps={s.fps}, sampling_interval_sec="
                        f"{s.tcore_sampling_interval_sec}, averaging_window_min="
                        f"{s.tcore_averaging_window_min}, "
                        f"slope={s.tcore_slope}, intercept={s.tcore_intercept}\n"
                    )
                    tc_path = g_out_dir / f"{stem}_tcore_estimated.csv"
                    with open(tc_path, "w") as fh:
                        fh.write(header)
                    tcore_df.to_csv(tc_path, index=False, mode="a")

                # ── Per-file metrics ──────────────────────────────────────
                g.per_file_metrics[stem] = extract_per_file_metrics(
                    g.name, stem, roi_df, tcore_df,
                    baseline_end_sec=self.settings.baseline_end_sec,
                )

                self.progress.emit(int(files_done / total_files * 80))

            # ── Group aggregation ─────────────────────────────────────────
            if g.roi_dfs:
                self._aggregate_group(g)
                self._save_group_csvs(g, g_out_dir)
                self.log_message.emit(f"[{g.name}] Aggregation complete.")

            all_group_data[g.name] = g

        self.progress.emit(85)

        # ── Statistical tests ─────────────────────────────────────────────
        self.log_message.emit("Running statistical tests…")
        # Build mock all_group_stats for GroupStatsTester
        mock_stats = {
            g.name: {"per_file_metrics": g.per_file_metrics}
            for g in self.groups
        }
        stats_results = GroupStatsTester.run_all(mock_stats)

        # ── Save stats CSVs ───────────────────────────────────────────────
        self._save_stats_csvs(stats_results, out_dir)

        self.progress.emit(95)

        # ── Multi-group time-series CSV ───────────────────────────────────
        self._save_multgroup_timeseries(all_group_data, out_dir)

        self.progress.emit(100)
        self.log_message.emit(f"✓ All outputs saved to: {out_dir}")
        self.finished.emit(
            {g_name: gd for g_name, gd in all_group_data.items()},
            stats_results,
        )

    def _aggregate_group(self, g: GroupData):
        """Interpolate all files to a common 30-s time axis and compute mean ± SEM."""
        roi_series = []
        tcore_series = []
        max_t = 0.0

        for stem, roi_df in g.roi_dfs.items():
            if len(roi_df) > 0:
                max_t = max(max_t, roi_df["time_s"].iloc[-1])

        if max_t <= 0:
            return

        common_t = np.arange(0, max_t, DOWN_SAMPLE_PERIOD_SEC)
        g.common_t = common_t

        for stem, roi_df in g.roi_dfs.items():
            if len(roi_df) < 2:
                continue
            t = roi_df["time_s"].to_numpy()
            interp_roi = np.interp(common_t, t, roi_df["normalized_roi_max"].to_numpy())
            # Two-stage smoothing (Combine temp apap.py pattern)
            interp_roi = smooth_series(interp_roi)
            roi_series.append(interp_roi)

            tdf = g.tcore_dfs.get(stem)
            if tdf is not None and len(tdf) >= 2:
                t_tc = tdf["time_s"].to_numpy()
                interp_tc = np.interp(common_t, t_tc, tdf["tcore_estimated"].to_numpy())
                tcore_series.append(interp_tc)

        if roi_series:
            mat = np.vstack(roi_series)
            g.roi_mean = mat.mean(axis=0)
            g.roi_sem = (sp_stats.sem(mat, axis=0) if mat.shape[0] > 1
                         else np.zeros(len(common_t)))

        if tcore_series:
            mat_tc = np.vstack(tcore_series)
            g.tcore_mean = mat_tc.mean(axis=0)
            g.tcore_sem = (sp_stats.sem(mat_tc, axis=0) if mat_tc.shape[0] > 1
                           else np.zeros(len(common_t)))

    def _save_group_csvs(self, g: GroupData, g_out_dir: Path):
        if g.common_t is not None and g.roi_mean is not None:
            roi_summary = pd.DataFrame({
                "time_s": g.common_t,
                "roi_mean": g.roi_mean,
                "roi_sem": g.roi_sem if g.roi_sem is not None else 0.0,
            })
            roi_summary.to_csv(g_out_dir / "group_roi_summary.csv", index=False)

        if g.common_t is not None and g.tcore_mean is not None:
            tc_summary = pd.DataFrame({
                "time_s": g.common_t,
                "tcore_mean": g.tcore_mean,
                "tcore_sem": g.tcore_sem if g.tcore_sem is not None else 0.0,
            })
            tc_summary.to_csv(g_out_dir / "group_tcore_summary.csv", index=False)

    def _save_stats_csvs(self, stats_results: dict, out_dir: Path):
        if stats_results.get("per_file_df") is not None:
            stats_results["per_file_df"].to_csv(
                out_dir / "per_file_metrics.csv", index=False)

        if stats_results.get("test_rows"):
            pd.DataFrame(stats_results["test_rows"]).to_csv(
                out_dir / "statistical_tests.csv", index=False)

        if stats_results.get("posthoc_rows"):
            pd.DataFrame(stats_results["posthoc_rows"]).to_csv(
                out_dir / "posthoc_tests.csv", index=False)

    def _save_multgroup_timeseries(self, all_group_data: dict, out_dir: Path):
        dfs = []
        for g_name, gd in all_group_data.items():
            if gd.common_t is None:
                continue
            col_prefix = g_name.replace(" ", "_")
            d = {"time_s": gd.common_t}
            if gd.roi_mean is not None:
                d[f"{col_prefix}_roi_mean"] = gd.roi_mean
                d[f"{col_prefix}_roi_sem"] = gd.roi_sem if gd.roi_sem is not None else 0.0
            if gd.tcore_mean is not None:
                d[f"{col_prefix}_tcore_mean"] = gd.tcore_mean
                d[f"{col_prefix}_tcore_sem"] = (gd.tcore_sem
                                                  if gd.tcore_sem is not None else 0.0)
            dfs.append(pd.DataFrame(d))

        if dfs:
            from functools import reduce
            merged = reduce(lambda a, b: pd.merge(a, b, on="time_s", how="outer"), dfs)
            merged.sort_values("time_s").to_csv(
                out_dir / "multi_group_timeseries.csv", index=False)


# ── Video Worker ──────────────────────────────────────────────────────────────

class VideoWorker(QThread):
    """Background worker for video export."""

    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, file_paths: List[str], output_path: str,
                 settings: AnalysisSettings, parent=None):
        super().__init__(parent)
        self.file_paths = file_paths
        self.output_path = output_path
        self.settings = settings

    def run(self):
        try:
            self.log_message.emit(
                f"Exporting video: {len(self.file_paths)} cameras → {self.output_path}")
            ok = export_video(
                self.file_paths,
                self.output_path,
                self.settings,
                progress_callback=lambda p: self.progress.emit(p),
            )
            if ok:
                self.finished.emit(self.output_path)
            else:
                self.error.emit("Video export failed — check log for details.")
        except Exception as exc:
            log.exception("VideoWorker fatal error: %s", exc)
            self.error.emit(str(exc))
