"""
QThread workers for Glöd.

ParseWorker  — parallel file parsing + group aggregation + stats
VideoWorker  — multi-panel video export
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from PyQt6.QtCore import QThread, pyqtSignal

from .parser import parse_files_parallel, result_to_dataframes
from .roi_analysis import bin_mean, emissivity_offset_c, nan_interp
from .stats import (GroupStatsTester, MIN_BIN_COVERAGE, extract_per_file_metrics, qc_flag,
                    run_timepoint_stats, timepoint_label)
from .video_exporter import export_video
from .publication import write_run_package
from ..utils import naming
from ..utils.config import AnalysisSettings, DOWN_SAMPLE_PERIOD_SEC

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
        self.animal_series: Dict[str, dict] = {}   # stem → 30-s bin series (abs/norm/tcore)
        self.abs_mean: Optional[np.ndarray] = None
        self.abs_sem: Optional[np.ndarray] = None
        self.coat_label: Optional[str] = None   # animal/coat preset chosen for this group
        self.epsilon: Optional[float] = None     # fur/skin emissivity used for this group
        self.settle_s: float = 0.0   # seconds excluded after placing the animal (plots shade it)


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

    # ── Naming (see utils/naming.py) ──────────────────────────────────────
    def _run_date(self) -> str:
        # output_root is <project>/analysis/<YYYY-MM-DD_HHMMSS>
        name = self.output_root.name
        return name[:10] if len(name) >= 10 else naming.today()

    def _name(self, group: str, stem, kind: str) -> str:
        """<date>_<group>[_<animal>]_<kind>.csv ; date = recording date if known."""
        if stem is None:
            return naming.build(kind, "csv", date=self._run_date(), group=group)
        animal, rec_date = naming.split_stem(stem)
        return naming.build(kind, "csv", date=rec_date or self._run_date(),
                            group=group, animal=animal)

    def _run_name(self, kind: str) -> str:
        return naming.build(kind, "csv", date=self._run_date())

    def run(self):
        try:
            self._run()
        except Exception as exc:
            log.exception("ParseWorker fatal error: %s", exc)
            self.error.emit(str(exc))

    def _run(self):
        # output_root is the per-run folder chosen by the GUI; CSVs go in data/
        out_dir = self.output_root / "data"
        out_dir.mkdir(parents=True, exist_ok=True)

        all_files = [fp for g in self.groups for fp in g.file_paths]
        total_files = len(all_files) or 1
        files_done = 0

        all_group_data: Dict[str, GroupData] = {}
        emis_by_group: Dict[str, dict] = {}

        for g in self.groups:
            self.log_message.emit(f"[{g.name}] Parsing {len(g.file_paths)} file(s)…")
            g_out_dir = out_dir / g.name
            g_out_dir.mkdir(parents=True, exist_ok=True)

            # ── Parallel parsing (emissivity is chosen per group) ─────────
            eps_g = g.epsilon if g.epsilon else self.settings.epsilon_mouse
            settings_g = dataclasses.replace(
                self.settings, epsilon_mouse=eps_g,
                coat_label=g.coat_label or self.settings.coat_label)
            if self.settings.apply_emissivity:
                off = emissivity_offset_c(eps_g, self.settings.epsilon_sensor)
                self.log_message.emit(
                    f"[{g.name}] emissivity ε = {eps_g:.2f} ({settings_g.coat_label}); sensor "
                    f"assumed {self.settings.epsilon_sensor:.2f} → temperatures shifted "
                    f"{off:+.2f} °C at 30.5 °C (T_core uses uncorrected sensor values).")
            emis_by_group[g.name] = {"coat": settings_g.coat_label, "epsilon": eps_g}
            results = parse_files_parallel(
                g.file_paths,
                settings_g,
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
                roi_df.to_csv(g_out_dir / self._name(g.name, stem, "roi_data"), index=False)
                fps_m = result.get("fps_measured", self.settings.fps)
                msg = (f"  ✓ {stem}: {result['n_frames']} frames, "
                       f"measured {fps_m:.2f} fps")
                if result.get("pauses"):
                    tot = sum(e - s for s, e in result["pauses"])
                    self.log_message.emit(
                        f"    ⏸ {stem}: recording was paused {len(result['pauses'])}× "
                        f"({tot:.0f} s in total). Analysed as ONE continuous recording: the pause "
                        "is cut out of the time axis and the data continue as if nothing happened.")
                trk = result.get("tracking")
                if trk:
                    ok = float(np.mean(trk["valid"])) * 100
                    msg += f", body tracked in {ok:.0f}% of frames"
                self.log_message.emit(msg)
                if abs(fps_m - self.settings.fps) > 0.15 * self.settings.fps:
                    self.log_message.emit(
                        f"    ⚠ measured rate {fps_m:.2f} fps differs from the {self.settings.fps:g} fps "
                        "setting; all time windows use the measured rate.")

                if tcore_df is not None:
                    s = self.settings
                    header = (
                        f"# Settings: fps_measured={fps_m:.3f}, settle_s={s.settle_s:g}, "
                        f"sampling_interval_sec="
                        f"{s.tcore_sampling_interval_sec}, averaging_window_min="
                        f"{s.tcore_averaging_window_min}, "
                        f"slope={s.tcore_slope}, intercept={s.tcore_intercept}\n"
                    )
                    tc_path = g_out_dir / self._name(g.name, stem, "tcore_estimated")
                    with open(tc_path, "w") as fh:
                        fh.write(header)
                    tcore_df.to_csv(tc_path, index=False, mode="a")

                # ── Per-file metrics ──────────────────────────────────────
                g.per_file_metrics[stem] = extract_per_file_metrics(
                    g.name, stem, roi_df, tcore_df,
                    baseline_end_sec=self.settings.baseline_end_sec,
                    settle_s=self.settings.settle_s,
                )
                m = g.per_file_metrics[stem]
                m["qc_flag"] = qc_flag(m, self.settings.min_coverage)
                if m["qc_flag"] != "ok":
                    self.log_message.emit(f"    ⚠ {stem}: {m['qc_flag']}")

                self.progress.emit(int(files_done / total_files * 80))

            # ── Multi-session check: same animal recorded more than once ──
            by_animal: Dict[str, list] = {}
            for stem in g.roi_dfs:
                a, d = naming.split_stem(stem)
                by_animal.setdefault(a, []).append(d or "?")
            dates = sorted({naming.split_stem(s)[1] or "?" for s in g.roi_dfs})
            if len(dates) > 1:
                self.log_message.emit(
                    f"[{g.name}] recordings from {len(dates)} sessions: {', '.join(dates)}")
            repeated = {a: d for a, d in by_animal.items() if len(d) > 1}
            if repeated:
                self.log_message.emit(
                    f"  ⚠ [{g.name}] animals with several recordings "
                    f"({', '.join(sorted(repeated))}): statistics treat each recording "
                    "as independent — use one recording per animal for inference.")

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
        # Recordings with too little tracked body are listed in the QC table but kept
        # out of the group statistics (min_coverage setting).
        mock_stats = {
            g.name: {"per_file_metrics": {
                s: m for s, m in g.per_file_metrics.items()
                if not str(m.get("qc_flag", "")).startswith("EXCLUDED")}}
            for g in self.groups
        }
        paired = bool(self.settings.paired_design) and len(self.groups) >= 2
        stats_results = GroupStatsTester.run_all(mock_stats, paired=paired)
        qc_rows = [{"group": g.name, "animal": naming.animal_from_stem(s),
                    "recording": s, **m}
                   for g in self.groups for s, m in g.per_file_metrics.items()]
        stats_results["qc_df"] = pd.DataFrame(qc_rows)
        stats_results["emissivity_by_group"] = emis_by_group
        if paired:
            m = stats_results.get("matching_df")
            if m is not None and len(m):
                n_ok = int(m["included_in_paired_analysis"].sum())
                self.log_message.emit(
                    f"Paired design: {n_ok} animal(s) present in all "
                    f"{len(self.groups)} groups are compared; "
                    f"{len(m) - n_ok} unmatched animal(s) excluded.")

        # ── Time-point comparison (optional) ──────────────────────────────
        if self.settings.timepoints_s:
            self.log_message.emit("Comparing user-set time points…")
            stats_results.update(run_timepoint_stats(
                all_group_data, self.settings.timepoints_s,
                self.settings.timepoint_signal, self.settings.timepoint_window_s,
                paired=paired))
            tp = stats_results["timepoint_df"]
            for (g, t), sub in tp.groupby(["group", "timepoint_s"]):
                missing = int(sub["value"].isna().sum())
                if missing:
                    self.log_message.emit(
                        f"  ⚠ {g}: {missing}/{len(sub)} recording(s) end before "
                        f"{timepoint_label(t)} and are excluded from that time point")

        # ── Save stats CSVs ───────────────────────────────────────────────
        try:
            write_run_package(self.output_root, self.groups, stats_results,
                              self.settings, self._run_date())
        except Exception as exc:
            log.exception("Publication package failed: %s", exc)
            self.log_message.emit(f"⚠ Could not write statistics package: {exc}")

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
        abs_series = []
        tcore_series = []
        max_t = 0.0

        for stem, roi_df in g.roi_dfs.items():
            if len(roi_df) > 0:
                max_t = max(max_t, roi_df["time_s"].iloc[-1])

        if max_t <= 0:
            return

        common_t = np.arange(0, max_t, DOWN_SAMPLE_PERIOD_SEC)
        g.common_t = common_t

        settle = float(getattr(self.settings, "settle_s", 0.0))
        g.settle_s = settle
        half = DOWN_SAMPLE_PERIOD_SEC / 2.0
        g.animal_series = {}

        def _binned(t, y):
            """30-s bin means of tracked frames; bins with < 20 % valid frames and the
            settling period are NaN (shown as gaps, never as dips)."""
            m = bin_mean(common_t, t, y, half)
            cov = bin_mean(common_t, t, np.isfinite(y).astype(float), half)
            m[(cov < MIN_BIN_COVERAGE) | (common_t + half <= settle)] = np.nan
            return m

        for stem, roi_df in g.roi_dfs.items():
            if len(roi_df) < 2:
                continue
            t = roi_df["time_s"].to_numpy()
            roi = roi_df["roi_max"].to_numpy()
            bw = (t >= settle) & (t <= self.settings.baseline_end_sec) & np.isfinite(roi)
            base_c = float(roi[bw].mean()) if bw.any() else np.nan
            ser = {"abs": _binned(t, roi),
                   "norm": _binned(t, roi_df["normalized_roi_max"].to_numpy()),
                   "baseline_c": base_c,
                   "excluded": str(g.per_file_metrics.get(stem, {}).get(
                       "qc_flag", "")).startswith("EXCLUDED")}
            tdf = g.tcore_dfs.get(stem)
            if tdf is not None and len(tdf) >= 2:
                ser["tcore"] = nan_interp(common_t, tdf["time_s"].to_numpy(),
                                          tdf["tcore_estimated"].to_numpy())
                ser["tskin"] = nan_interp(common_t, tdf["time_s"].to_numpy(),
                                          tdf["tskin_max"].to_numpy())
            ser["delta"] = ser["abs"] - base_c      # °C relative to the animal's own baseline
            g.animal_series[stem] = ser
            if ser["excluded"]:
                continue          # low-coverage animals stay out of group curves
            roi_series.append(ser["norm"])
            abs_series.append(ser["abs"])
            if "tcore" in ser:
                tcore_series.append(ser["tcore"])
        def _mean_sem(rows):
            mat = np.vstack(rows)
            n = np.sum(np.isfinite(mat), axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(n > 0, np.nansum(mat, axis=0) / np.maximum(n, 1), np.nan)
                sd = np.sqrt(np.nansum((mat - mean) ** 2, axis=0) / np.maximum(n - 1, 1))
                sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
            return mean, np.where(n > 0, sem, np.nan)

        if roi_series:
            g.roi_mean, g.roi_sem = _mean_sem(roi_series)
            g.abs_mean, g.abs_sem = _mean_sem(abs_series)
        if tcore_series:
            g.tcore_mean, g.tcore_sem = _mean_sem(tcore_series)

    def _save_group_csvs(self, g: GroupData, g_out_dir: Path):
        if g.common_t is not None and g.roi_mean is not None:
            roi_summary = pd.DataFrame({
                "time_s": g.common_t,
                "roi_mean": g.roi_mean,
                "roi_sem": g.roi_sem if g.roi_sem is not None else 0.0,
            })
            roi_summary.to_csv(
                g_out_dir / self._name(g.name, None, "group_roi_summary"), index=False)

        if g.common_t is not None and g.tcore_mean is not None:
            tc_summary = pd.DataFrame({
                "time_s": g.common_t,
                "tcore_mean": g.tcore_mean,
                "tcore_sem": g.tcore_sem if g.tcore_sem is not None else 0.0,
            })
            tc_summary.to_csv(
                g_out_dir / self._name(g.name, None, "group_tcore_summary"), index=False)

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
                out_dir / self._run_name("multi_group_timeseries"), index=False)


# ── Video Worker ──────────────────────────────────────────────────────────────

class VideoWorker(QThread):
    """Background worker for video export.

    ``jobs`` is a list of dicts: {file_paths, output_path, labels, group, date}.
    Each job produces one video (group tile or single animal).
    """

    progress = pyqtSignal(int)
    log_message = pyqtSignal(str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, jobs: List[dict], settings: AnalysisSettings, parent=None):
        super().__init__(parent)
        self.jobs = jobs
        self.settings = settings

    def run(self):
        try:
            n = len(self.jobs)
            done, failed = [], []
            for i, job in enumerate(self.jobs):
                self.log_message.emit(
                    f"Exporting video {i + 1}/{n}: {Path(job['output_path']).name}")
                ok = export_video(
                    job["file_paths"], job["output_path"], self.settings,
                    progress_callback=lambda p, i=i: self.progress.emit(
                        int((i * 100 + p) / n)),
                    labels=job.get("labels"),
                    group_name=job.get("group", ""),
                    date_str=job.get("date", ""),
                )
                (done if ok else failed).append(job["output_path"])
            self.progress.emit(100)
            if failed:
                self.error.emit("Video export failed for: "
                                + ", ".join(Path(f).name for f in failed))
            if done:
                self.finished.emit(str(Path(done[0]).parent))
        except Exception as exc:
            log.exception("VideoWorker fatal error: %s", exc)
            self.error.emit(str(exc))