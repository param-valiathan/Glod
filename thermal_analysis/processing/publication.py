"""
Publication-ready exports for a Glöd analysis run (Qt-free).

Layout written into <run folder>:
  source_data/   one tidy CSV per figure (exactly what is drawn), named like the PNG
  statistics/    per-animal metrics, descriptives, normality, tests, post-hoc, report (.md)
  <date>_analysis_manifest.csv   every input file (animal, group, date, frames, SHA-256)
  README.txt     data dictionary and reproducibility notes

All CSVs are UTF-8, comma-separated, one header row, full numeric precision, so they
import directly into R, Python, Prism or Excel and can be deposited as supplementary
source data.
"""

from __future__ import annotations

import dataclasses
import hashlib
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection

from . import stats as glod_stats
from ..utils import naming

# ── Figure source data ────────────────────────────────────────────────────────

_COLS = ["figure", "panel", "panel_title", "series", "element",
         "x", "y", "x_label", "y_label"]


def _poly_band(path_vertices: np.ndarray):
    """Split a fill_between polygon into (x, lower, upper)."""
    v = np.asarray(path_vertices)
    n = (len(v) - 3) // 2
    if n < 1:
        return None
    first = v[1:n + 1]                  # (x, y1)
    second = v[n + 2:2 * n + 2][::-1]   # (x, y2) — stored in reverse
    lo = np.minimum(first[:, 1], second[:, 1])
    hi = np.maximum(first[:, 1], second[:, 1])
    return first[:, 0], lo, hi


def figure_to_frame(fig, title: str) -> pd.DataFrame:
    """Extract every plotted element of ``fig`` into one tidy DataFrame."""
    rows: List[dict] = []

    for p_idx, ax in enumerate(fig.axes):
        panel = f"panel_{p_idx + 1}"
        base = {"figure": title, "panel": panel, "panel_title": ax.get_title(),
                "x_label": ax.get_xlabel(), "y_label": ax.get_ylabel()}

        def add(series, element, xs, ys):
            for x, y in zip(xs, ys):
                rows.append({**base, "series": series, "element": element,
                             "x": x, "y": y})

        # Lines (data series, fits, annotation lines)
        for i, ln in enumerate(ax.get_lines()):
            label = ln.get_label()
            if label.startswith("_"):
                continue   # matplotlib-internal / decorative (brackets, errorbar caps)
            xs, ys = ln.get_xdata(), ln.get_ydata()
            if len(xs) < 2 and len(xs) == 0:
                continue
            add(label or f"series_{i + 1}", "line", xs, ys)

        # Shaded bands (mean ± SEM etc.)
        for j, coll in enumerate(ax.collections):
            cname = type(coll).__name__
            if isinstance(coll, PolyCollection):
                for k, pth in enumerate(coll.get_paths()):
                    band = _poly_band(pth.vertices)
                    if band is None:
                        continue
                    x, lo, hi = band
                    lab = coll.get_label()
                    lab = f"band_{j + 1}" if (not lab or lab.startswith("_")) else lab
                    add(lab, "band_lower", x, lo)
                    add(lab, "band_upper", x, hi)
            elif cname == "PathCollection":
                offs = np.asarray(coll.get_offsets())
                if len(offs):
                    lab = coll.get_label()
                    lab = f"points_{j + 1}" if (not lab or lab.startswith("_")) else lab
                    add(lab, "point", offs[:, 0], offs[:, 1])

        # Bars (+ their tick labels)
        ticks = {float(t): l.get_text()
                 for t, l in zip(ax.get_xticks(), ax.get_xticklabels())}
        for patch in ax.patches:
            if type(patch).__name__ != "Rectangle":
                continue
            x_c = patch.get_x() + patch.get_width() / 2
            lab = ticks.get(float(round(x_c, 6)), "")
            rows.append({**base, "series": lab or f"bar_{round(x_c, 3)}",
                         "element": "bar", "x": x_c, "y": patch.get_height()})

    return pd.DataFrame(rows, columns=_COLS)


def write_figure_source_data(fig, title: str, path: Path,
                             per_file_df: pd.DataFrame | None = None,
                             timepoint_df: pd.DataFrame | None = None) -> None:
    """Write source data for one figure.

    Statistics figures (bars / violin / box) are built from the per-animal metric
    table, so the exact per-animal values behind them are written instead of
    reconstructed geometry.
    """
    src = getattr(fig, "_glod_source", None)
    if src is not None:           # figure carries its own exact source table
        src.to_csv(path, index=False)
        return
    if timepoint_df is not None and not timepoint_df.empty and "Time-point" in title:
        out = timepoint_df.rename(columns={"file": "animal_recording"}).copy()
        out["animal"] = out["animal_recording"].map(naming.animal_from_stem)
        out.to_csv(path, index=False)
        return
    if per_file_df is not None and not per_file_df.empty and (
            "Group Comparison" in title or "Violin" in title):
        cols = ["group", "file"] + [m for m in glod_stats.METRICS
                                    if m in per_file_df.columns]
        out = per_file_df[cols].rename(columns={"file": "animal_recording"})
        out["animal"] = out["animal_recording"].map(naming.animal_from_stem)
        out["recording_date"] = out["animal_recording"].map(
            lambda s: naming.split_stem(s)[1] or "")
        out.to_csv(path, index=False)
        return
    figure_to_frame(fig, title).to_csv(path, index=False)


# ── Statistics package ────────────────────────────────────────────────────────

def _fmt(v, nd=3):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "NA" if np.isnan(f) else f"{f:.{nd}f}"


def write_statistics_package(stats_dir: Path, stats_results: dict,
                             settings, date: str) -> None:
    stats_dir.mkdir(parents=True, exist_ok=True)

    def w(df: pd.DataFrame, kind: str):
        if df is not None and len(df):
            df.to_csv(stats_dir / naming.build(kind, "csv", date=date), index=False)

    per_file = stats_results.get("per_file_df")
    if per_file is not None and len(per_file):
        pf = per_file.copy()
        if "animal" not in pf.columns:
            pf.insert(2, "animal", pf["file"].map(naming.animal_from_stem))
        pf.insert(3, "recording_date", pf["file"].map(
            lambda s: naming.split_stem(s)[1] or ""))
        w(pf, "per_animal_metrics")

    if stats_results.get("paired"):
        w(stats_results.get("matching_df"), "paired_animal_matching")
        rec = stats_results.get("recording_df")
        if rec is not None and len(rec):
            rec = rec.copy()
            rec.insert(2, "animal", rec["file"].map(naming.animal_from_stem))
            w(rec, "per_recording_metrics")
    qc = stats_results.get("qc_df")
    if qc is not None and len(qc):
        cols = ["group", "animal", "recording", "coverage_pct", "longest_gap_s", "fps_measured",
                "tracking_bias_rho", "tracking_bias_p", "tracking_bias_n", "n_pauses", "paused_s", "qc_flag"]
        w(qc[[c for c in cols if c in qc.columns]], "tracking_qc")
    w(stats_results.get("descriptive_df"), "descriptive_statistics")
    w(stats_results.get("normality_df"), "normality_tests")
    w(pd.DataFrame(stats_results.get("test_rows", [])), "statistical_tests")
    w(pd.DataFrame(stats_results.get("posthoc_rows", [])), "posthoc_tests")

    # Time-point comparison (only present when time points were set)
    if len(stats_results.get("timepoint_df", [])):
        w(stats_results["timepoint_df"].assign(
            animal=lambda d: d["file"].map(naming.animal_from_stem)),
          "timepoint_values_per_animal")
        w(stats_results.get("timepoint_desc_df"), "timepoint_descriptive_statistics")
        w(pd.DataFrame(stats_results.get("timepoint_test_rows", [])),
          "timepoint_between_group_tests")
        w(pd.DataFrame(stats_results.get("timepoint_posthoc_rows", [])),
          "timepoint_between_group_posthoc")
        w(pd.DataFrame(stats_results.get("timepoint_within_rows", [])),
          "timepoint_within_group_tests")

    (stats_dir / naming.build("statistics_report", "md", date=date)).write_text(
        _build_report(stats_results, settings, date), encoding="utf-8")


def _emissivity_sentence(stats_results: dict, s) -> str:
    by_group = stats_results.get("emissivity_by_group") or {}
    eps = {f"{v['epsilon']:.2f}" for v in by_group.values()} or {f"{s.epsilon_mouse:.2f}"}
    who = (", ".join(f"{g}: ε = {v['epsilon']:.2f} ({v['coat']})" for g, v in by_group.items())
           if len(eps) > 1 else f"ε = {next(iter(eps))}")
    return (f". Temperatures were corrected for the emissivity of the coat ({who}; sensor "
            f"setting ε = {s.epsilon_sensor:.2f}; reflected temperature estimated from the "
            "image background; literature range for rodent fur 0.86–0.98, "
            "McGowan et al. 2018). Estimated core temperature used uncorrected sensor values")


def _methods_paragraph(stats_results: dict, settings) -> str:
    s = settings
    roi = {"body": "the animal's body, segmented in every frame (adaptive Otsu threshold, "
                   "morphological opening to exclude the tail, largest connected component; "
                   "the hottest pixel of the eroded body core after 1-px Gaussian smoothing). "
                   "Frames in which the body was mostly out of view, merged with warm bedding, "
                   "or a Hampel outlier (3-s window, 4 MAD) were excluded and gaps ≤ 2 s were "
                   "bridged by linear interpolation",
           "static": f"a fixed {s.roi_w}×{s.roi_h}-pixel region of interest at ({s.roi_x}, {s.roi_y})"
          }.get(s.roi_mode, "the animal's body")
    text = [
        f"Thermal image sequences ({s.camera_width}×{s.camera_height} px) "
        f"were analysed with Glöd. Surface temperature was "
        f"extracted from {roi}"
        + (f", restricted to an arena of {s.arena_w}×{s.arena_h} px"
           if s.arena_enabled else "")
        + (_emissivity_sentence(stats_results, s) if s.apply_emissivity else "")
        + (f". The first {s.settle_s:g} s after placing the animal in the arena were "
           "excluded as a settling period" if s.settle_s > 0 else "")
        + f". Signals were normalised to the mean over {max(s.settle_s, 0):g}–"
          f"{s.baseline_end_sec:g} s (baseline). All temporal windows were defined in "
          "seconds using the frame rate measured from the recorded timestamps."
    ]
    if s.tcore_enabled:
        text.append(
            "Core temperature was estimated from maximal skin temperature following "
            f"van der Vinne et al. (2020, Sci Rep 10:20680): maximum over "
            f"{s.tcore_sampling_interval_sec:g}-s blocks, a {s.tcore_averaging_window_min:g}-min "
            f"rolling mean, and T_core = {s.tcore_slope:g} × T_skin + {s.tcore_intercept:g} °C "
            "(group-average calibration; between-animal error ≈ ±0.9 °C).")

    rows = stats_results.get("test_rows", [])
    paired = bool(stats_results.get("paired"))
    if rows:
        names = sorted({r["test_name"] for r in rows}
                       | {r["test_name"] for r in stats_results.get("posthoc_rows", [])})
        if paired:
            text.append(
                "The same animals were measured under every condition, so conditions "
                "were compared with paired tests on animals present in all conditions "
                "(repeated recordings of an animal were averaged). Two conditions: "
                "paired t-test, or Wilcoxon signed-rank test when the paired differences "
                "failed the Shapiro–Wilk test and n ≥ 6 (below this a signed-rank test "
                "cannot reach p < 0.05). Three or more conditions: repeated-measures "
                "ANOVA with Greenhouse–Geisser correction, or the Friedman test if a "
                "condition failed Shapiro–Wilk (n ≥ 5), followed by pairwise paired "
                "tests with Holm correction when the omnibus test was significant. "
                "All tests were two-sided (α = 0.05). Effect sizes: Cohen's d_z, "
                "rank-biserial r, partial η² or Kendall's W.")
        else:
            text.append(
                "Per-animal summary metrics (one value per recording; unit of analysis = "
                "animal) were compared between independent groups. Welch's t-test (two "
                "groups) or one-way ANOVA (≥ 3 groups) was used by default; the "
                "Mann–Whitney U or Kruskal–Wallis test was used instead when a group "
                "failed the Shapiro–Wilk test (n ≥ 3) and the rank test could reach "
                "p < 0.05 at the sample sizes available. Significant omnibus tests were "
                "followed by Tukey HSD (after ANOVA) or Dunn's test with Holm correction "
                "(after Kruskal–Wallis). All tests were two-sided (α = 0.05). Effect "
                "sizes: Cohen's d, rank-biserial r, η² or ε².")
        text.append(f"Tests used in this run: {', '.join(names)}.")
    if s.timepoints_s:
        from .stats import TIMEPOINT_SIGNALS, timepoint_label
        sig = TIMEPOINT_SIGNALS.get(s.timepoint_signal, (s.timepoint_signal, ""))[0].lower()
        tps = ", ".join(timepoint_label(t) for t in s.timepoints_s)
        text.append(
            f"Temperature was additionally compared at predefined time points ({tps} "
            f"from recording start) using the {sig}, averaged over a "
            f"{s.timepoint_window_s:g}-s window centred on each time point; recordings "
            "ending before a time point were excluded from it. Conditions were compared "
            "at each time point with the tests described above (Holm-adjusted across "
            "time points)"
            + ("; changes across time points within each group were tested with "
               "repeated-measures ANOVA or Friedman (three time points) and pairwise "
               "paired t-tests or Wilcoxon signed-rank tests, Holm-corrected."
               if len(s.timepoints_s) > 1 else "."))
    text.append("Analysis software: Glöd (Python; NumPy, SciPy, pandas).")
    return " ".join(text)

def _build_report(stats_results: dict, settings, date: str) -> str:
    L: List[str] = [f"# Glöd statistics report — {date}", ""]
    L += ["## Suggested methods text", "", _methods_paragraph(stats_results, settings), ""]

    mt = stats_results.get("matching_df")
    if stats_results.get("paired") and mt is not None and len(mt):
        ok = mt[mt["included_in_paired_analysis"]]
        bad = mt[~mt["included_in_paired_analysis"]]
        L += ["## Paired design — animal matching", "",
              f"{len(ok)} animal(s) were measured in every group and enter the paired "
              "analysis (repeated recordings averaged per animal).", ""]
        if len(bad):
            L.append("Excluded (not present in all groups): "
                     + ", ".join(bad["animal"].astype(str)) + ".")
            L.append("")

    qc = stats_results.get("qc_df")
    if qc is not None and len(qc):
        L += ["## Tracking quality", "",
              "Statistics use tracked frames only (no imputation). Each animal's value is the "
              "mean of 30-s bins that have ≥ 20 % tracked frames, so periods that are easier "
              "to track do not dominate. `coverage_pct` = tracked frames after the settling "
              f"period; animals below {100 * settings.min_coverage:.0f} % are excluded. "
              "`tracking_bias` is the Spearman correlation between a bin's tracked fraction "
              "and its temperature (a significant, large value would indicate selection bias).", "",
              "| Group | Animal | Coverage % | Longest gap (s) | Bias rho (p) | Verdict |",
              "|---|---|---|---|---|---|"]
        for r in qc.itertuples():
            L.append(f"| {r.group} | {r.animal} | {_fmt(r.coverage_pct, 1)} | "
                     f"{_fmt(r.longest_gap_s, 1)} | {_fmt(r.tracking_bias_rho, 2)} "
                     f"({_fmt(r.tracking_bias_p, 3)}) | {r.qc_flag} |")
        L.append("")
        for r in qc.to_dict("records"):
            if str(r.get("qc_flag", "")).startswith("CHECK"):
                L += [f"**{r['group']} · {r['animal']}** — " + glod_stats.bias_sentence(r), ""]

    desc = stats_results.get("descriptive_df")
    if desc is not None and len(desc):
        L += ["## Descriptive statistics", "",
              "Per metric and group; values are mean ± SD (n); median [Q1, Q3]; 95 % CI.", "",
              "| Metric | Unit | Group | n | Mean ± SD | Median [Q1, Q3] | 95 % CI |",
              "|---|---|---|---|---|---|---|"]
        for r in desc.itertuples():
            L.append(f"| {r.metric} | {r.unit} | {r.group} | {r.n} | "
                     f"{_fmt(r.mean)} ± {_fmt(r.sd)} | "
                     f"{_fmt(r.median)} [{_fmt(r.q1)}, {_fmt(r.q3)}] | "
                     f"[{_fmt(r.ci95_low)}, {_fmt(r.ci95_high)}] |")
        L.append("")

    rows = stats_results.get("test_rows", [])
    L += ["## Group comparisons", ""]
    if not rows:
        L += ["No between-group tests were run (a single group was analysed, or a group "
              "had no data for a metric).", ""]
    else:
        L += ["| Metric | Comparison | Test | Statistic | df | p | Holm-adj. p | Effect size | n |",
              "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            comp = (r["group_A"] if not r["group_B"]
                    else f"{r['group_A']} vs {r['group_B']}")
            L.append(
                f"| {r['metric']} | {comp} | {r['test_name']} | {_fmt(r['statistic'])} | "
                f"{r['df'] if r['df'] == r['df'] else 'NA'} | {r['p_text']} | "
                f"{_fmt(r['p_holm_across_metrics'])} | "
                f"{_fmt(r['effect_size'])} ({r['effect_size_type']}) | "
                f"{r['n_A']}{'+' + str(r['n_B']) if r['n_B'] else ''} |")
        L += ["", "Test selection basis per metric:", ""]
        for r in rows:
            L.append(f"- {r['metric']}: {r['test_selection_basis']}")
        L.append("")

    tp_rows = stats_results.get("timepoint_test_rows", [])
    tp_df = stats_results.get("timepoint_df")
    if tp_df is not None and len(tp_df):
        L += ["## Time-point comparison", "",
              f"Signal: {settings.timepoint_signal}; window: {settings.timepoint_window_s:g} s "
              "centred on each time point (time from recording start).", ""]
        d = stats_results.get("timepoint_desc_df")
        if d is not None and len(d):
            L += ["| Time point | Group | n | Mean ± SD | Median [Q1, Q3] | 95 % CI |",
                  "|---|---|---|---|---|---|"]
            for r in d.itertuples():
                L.append(f"| {r.timepoint} | {r.group} | {r.n} | {_fmt(r.mean)} ± {_fmt(r.sd)} | "
                         f"{_fmt(r.median)} [{_fmt(r.q1)}, {_fmt(r.q3)}] | "
                         f"[{_fmt(r.ci95_low)}, {_fmt(r.ci95_high)}] |")
            L.append("")
        if tp_rows:
            L += ["Between groups:", "",
                  "| Time point | Comparison | Test | Statistic | df | p | Holm-adj. p | Effect size |",
                  "|---|---|---|---|---|---|---|---|"]
            for r in tp_rows:
                comp = r["group_A"] if not r["group_B"] else f"{r['group_A']} vs {r['group_B']}"
                L.append(f"| {r['timepoint']} | {comp} | {r['test_name']} | {_fmt(r['statistic'])} | "
                         f"{r['df'] if r['df'] == r['df'] else 'NA'} | {r['p_text']} | "
                         f"{_fmt(r['p_holm_across_metrics'])} | "
                         f"{_fmt(r['effect_size'])} ({r['effect_size_type']}) |")
            L.append("")
        tpp = stats_results.get("timepoint_posthoc_rows", [])
        if tpp:
            L += ["Between-group post-hoc (Tukey / Dunn / Holm, see correction column):", "",
                  "| Time point | Pair | Test | p (corrected) |", "|---|---|---|---|"]
            for r in tpp:
                L.append(f"| {r['timepoint']} | {r['group_A']} vs {r['group_B']} | "
                         f"{r['test_name']} | {_fmt(r['p_corrected'], 4)} |")
            L.append("")
        wr = stats_results.get("timepoint_within_rows", [])
        if wr:
            L += ["Within groups (repeated measures, complete cases):", "",
                  "| Group | Comparison | Test | Statistic | p (corrected) | Effect size | n |",
                  "|---|---|---|---|---|---|---|"]
            for r in wr:
                L.append(f"| {r['group']} | {r['comparison']} | {r['test_name']} | "
                         f"{_fmt(r['statistic'])} | {r['p_text']} | "
                         f"{_fmt(r['effect_size'])} ({r['effect_size_type']}) | {r['n_pairs']} |")
            L.append("")

    ph = stats_results.get("posthoc_rows", [])
    if ph:
        L += ["## Post-hoc pairwise tests", "",
              "| Metric | Pair | Test | Statistic | p (raw) | p (corrected) | Correction | Effect size |",
              "|---|---|---|---|---|---|---|---|"]
        for r in ph:
            L.append(f"| {r['metric']} | {r['group_A']} vs {r['group_B']} | {r['test_name']} | "
                     f"{_fmt(r['statistic'])} | {_fmt(r['p_value'], 4)} | "
                     f"{_fmt(r['p_corrected'], 4)} | {r['correction']} | "
                     f"{_fmt(r['effect_size'])} ({r['effect_size_type']}) |")
        L.append("")

    L += ["## Notes", "",
          ("- Paired design: unit of analysis is the animal; each animal contributes one "
           "value per condition." if stats_results.get("paired") else
           "- Unit of analysis is one recording (animal). Repeated recordings of the same "
           "animal are not independent; tick the paired design option if the same animals "
           "appear in several groups, or analyse one recording per animal."),
          "- Test choice favours power at small n: parametric tests are the default and a "
          "rank test is used only when normality fails and the rank test can reach p < 0.05; "
          "tests flagged 'interpret with caution' rest on very few animals.",
          "- p-values in `statistical_tests.csv` are unadjusted within a metric; "
          "`p_holm_across_metrics` additionally adjusts across all metrics tested in this run.",
          "- Significance codes: ns p ≥ 0.05, * < 0.05, ** < 0.01, *** < 0.001, **** < 0.0001.",
          "- Exact values, normality tests and per-animal data are in the CSV files "
          "next to this report.", ""]
    return "\n".join(L)


# ── Manifest, README ──────────────────────────────────────────────────────────

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "unreadable"


def write_manifest(run_dir: Path, groups: list, settings, date: str) -> None:
    """One row per input recording, with a SHA-256 for provenance."""
    rows = []
    for g in groups:
        for fp in g.file_paths:
            stem = Path(fp).stem
            roi_df = g.roi_dfs.get(stem)
            rows.append({
                "group": g.name,
                "animal": naming.animal_from_stem(stem),
                "recording_date": naming.split_stem(stem)[1] or "",
                "file": str(fp),
                "sha256": _sha256(str(fp)),
                "frames_analysed": len(roi_df) if roi_df is not None else 0,
                "duration_s": (float(roi_df["time_s"].iloc[-1])
                               if roi_df is not None and len(roi_df) else np.nan),
                "status": "analysed" if roi_df is not None else "skipped (parse error)",
            })
    pd.DataFrame(rows).to_csv(
        run_dir / naming.build("analysis_manifest", "csv", date=date), index=False)


def write_readme(run_dir: Path, settings, date: str) -> None:
    import scipy
    try:
        import matplotlib
        mpl = matplotlib.__version__
    except Exception:
        mpl = "n/a"
    lines = [
        f"Glöd analysis run — {date}",
        f"Created: {datetime.now().isoformat(timespec='seconds')}",
        f"Software: Python {sys.version.split()[0]} · NumPy {np.__version__} · "
        f"pandas {pd.__version__} · SciPy {scipy.__version__} · "
        f"matplotlib {mpl} · {platform.platform()}",
        "",
        "FOLDERS",
        "  data/          per-animal ROI and T_core time series; group mean ± SEM series",
        "  plots/         figures as PNG (150 dpi)",
        "  source_data/   one CSV per figure, same stem as the PNG: the plotted values",
        "  statistics/    tracking-QC table, per-animal metrics, descriptive statistics, Shapiro-Wilk tests,",
        "                 group tests, post-hoc tests and a statistics report (.md) with",
        "                 suggested methods text",
        "  videos/        group tile and per-animal videos",
        "  settings.json  every analysis parameter used",
        f"  {date}_analysis_manifest.csv  input files with SHA-256 hashes",
        "",
        "FILE NAMES   <date>_<group>[_<animal>]_<kind>.<ext>   (date = recording date)",
        "",
        "SOURCE-DATA COLUMNS (tidy/long format)",
        "  figure, panel, panel_title   which plot and sub-panel",
        "  series                       legend label (or bar/group name)",
        "  element                      line | band_lower | band_upper (mean ± SEM etc.) |",
        "                               point | bar",
        "  x, y                         plotted coordinates; x_label / y_label carry units",
        "  Statistics figures list the per-animal values behind the plot instead.",
        "",
        "METRIC DEFINITIONS",
    ]
    for m, (d, u) in glod_stats.METRIC_INFO.items():
        lines.append(f"  {m}  [{u}]  {d}")
    lines += [
        "",
        "STATISTICS COLUMNS",
        "  statistic, df, p_value (exact, unrounded), p_text, p_corrected + correction,",
        "  p_holm_across_metrics, effect_size + effect_size_type, mean_diff_A_minus_B with",
        "  95 % CI (Welch), n_A/n_B, mean/sd/sem/median per group,",
        "  test_selection_basis (why parametric / non-parametric was chosen).",
        "",
    ]
    (run_dir / "README.txt").write_text("\n".join(lines), encoding="utf-8")


def write_run_package(run_dir: Path, groups: list, stats_results: dict,
                      settings, date: str) -> None:
    """Statistics + manifest + README (everything not tied to a rendered figure)."""
    write_statistics_package(run_dir / "statistics", stats_results, settings, date)
    write_manifest(run_dir, groups, settings, date)
    write_readme(run_dir, settings, date)
