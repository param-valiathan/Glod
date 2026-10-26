"""
Matplotlib figure builders for Glöd.

All functions return a matplotlib.figure.Figure.
Figures use a clean Scandinavian white background with muted charcoal text.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import textwrap
from ..processing.stats import bias_flagged, bias_sentence
from scipy.ndimage import gaussian_filter1d


# ── Style constants ───────────────────────────────────────────────────────────
_BG = "#FFFFFF"
_TEXT = "#212529"
_GRID = "#DEE2E6"
_PANEL_BG = "#F8F9FA"
_ACCENT = "#5C8A6B"
_SEM_ALPHA = 0.20
_LINE_W = 2.0
_FONT = {"family": "DejaVu Sans", "size": 11}
_TITLE_SIZE = 13
_LABEL_SIZE = 11
_TICK_SIZE = 10

matplotlib.rc("font", **_FONT)
matplotlib.rc("axes", facecolor=_BG, edgecolor=_GRID, labelcolor=_TEXT,
               titlecolor=_TEXT, grid=True)
matplotlib.rc("grid", color=_GRID, linewidth=0.7, linestyle="--", alpha=0.7)
matplotlib.rc("xtick", color=_TEXT, labelsize=_TICK_SIZE)
matplotlib.rc("ytick", color=_TEXT, labelsize=_TICK_SIZE)
matplotlib.rc("legend", frameon=True, framealpha=0.9, edgecolor=_GRID,
               fontsize=10, labelcolor=_TEXT)
matplotlib.rc("figure", facecolor=_PANEL_BG, edgecolor=_PANEL_BG)


def _new_fig(nrows=1, ncols=1, figsize=(10, 4.5), **kw) -> tuple[Figure, any]:
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                              facecolor=_PANEL_BG, **kw)
    fig.patch.set_facecolor(_PANEL_BG)
    return fig, axes


def _style_ax(ax, xlabel="", ylabel="", title=""):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=_LABEL_SIZE, color=_TEXT)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=_LABEL_SIZE, color=_TEXT)
    if title:
        ax.set_title(title, fontsize=_TITLE_SIZE, color=_TEXT, fontweight="bold", pad=8)


def _sig_bracket(ax, x1, x2, y, label, color="black", lw=1.2):
    """Draw a significance bracket above two bars."""
    h = y * 0.03
    ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], lw=lw, color=color)
    ax.text((x1 + x2) / 2, y + h * 1.1, label, ha="center", va="bottom",
            fontsize=10, color=color)


def _shade_settle(ax, settle_s: float) -> None:
    """Grey band over the settling period that is excluded from baseline / metrics."""
    if settle_s and settle_s > 0:
        ax.axvspan(0, settle_s, color="#ADB5BD", alpha=0.18, lw=0, zorder=0)
        ymax = ax.get_ylim()[1]
        ax.text(settle_s / 2, ymax, "settling\n(excluded)", ha="center", va="top",
                fontsize=7, color="#6C757D")


def _settle_of(roi_df: pd.DataFrame) -> float:
    """Settling period recorded in a per-file frame (0 when the column is absent)."""
    if "in_settle" in roi_df.columns and roi_df["in_settle"].any():
        return float(roi_df.loc[roi_df["in_settle"].astype(bool), "time_s"].max())
    return 0.0


# ── 1. Group time courses: every animal + group mean ─────────────────────────

_EXCLUDED_GREY = "#ADB5BD"


def _animal_label(stem: str) -> str:
    from ..utils.naming import animal_from_stem
    return animal_from_stem(stem)


def _mean_sem_nan(rows):
    mat = np.vstack(rows)
    n = np.isfinite(mat).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, np.nansum(mat, axis=0) / np.maximum(n, 1), np.nan)
        sd = np.sqrt(np.nansum((mat - mean) ** 2, axis=0) / np.maximum(n - 1, 1))
        sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mean, sem, n


def build_timecourse(all_group_data: dict, kind: str = "delta") -> Figure:
    """
    Body temperature over time, one thin line per animal plus the group mean ± SEM.

    kind="delta": change from the animal's own post-settling baseline, in °C;  kind="abs": °C.
    Individual animals are drawn so n and between-animal spread are visible (SEM of a
    handful of animals alone hides them).  Gaps are untracked periods, never dips;
    animals excluded for low tracking coverage are dashed grey and not in the mean.
    """
    from matplotlib.lines import Line2D
    norm = kind == "delta"
    scale = 1.0
    fig, ax = _new_fig(figsize=(11, 5))
    _style_ax(ax, "Time since placement (s)",
              "Δ body temperature from baseline (°C)" if norm else "Body temperature (°C)",
              "Body temperature — change from baseline" if norm
              else "Body temperature — absolute")
    handles, settle = [], 0.0
    for g_name, gd in all_group_data.items():
        if gd.common_t is None or not gd.animal_series:
            continue
        t, color = gd.common_t, gd.color
        rows = []
        for stem, ser in gd.animal_series.items():
            y = ser[kind] * scale
            if ser["excluded"]:
                ax.plot(t, y, color=_EXCLUDED_GREY, lw=1, ls="--", alpha=0.8,
                        label=f"{g_name} · {_animal_label(stem)} (excluded)")
            else:
                ax.plot(t, y, color=color, lw=1.0, alpha=0.45,
                        label=f"{g_name} · {_animal_label(stem)}")
                rows.append(y)
        if rows:
            mean, sem, n = _mean_sem_nan(rows)
            if len(rows) > 1:
                ax.fill_between(t, mean - sem, mean + sem, color=color,
                                alpha=_SEM_ALPHA, lw=0, label=f"{g_name} mean ± SEM")
            ax.plot(t, mean, color=color, lw=2.8, label=f"{g_name} mean")
            handles.append(Line2D([], [], color=color, lw=2.8,
                                  label=f"{g_name} (n={len(rows)})"))
        settle = max(settle, gd.settle_s)
    if norm:
        ax.axhline(0, color="#6C757D", lw=0.8)
    _shade_settle(ax, settle)
    handles.append(Line2D([], [], color="#6C757D", lw=1, alpha=0.6, label="individual animals"))
    ax.legend(handles=handles, loc="best")
    fig.tight_layout(pad=1.5)
    return fig


# ── 2. Estimated T_core ───────────────────────────────────────────────────────

def build_tcore_estimated(all_group_data: dict) -> Figure:
    """T_skin,max (top) and estimated T_core (bottom): animals + group mean ± SEM."""
    from matplotlib.lines import Line2D
    fig, (ax1, ax2) = _new_fig(2, 1, figsize=(11, 8), sharex=True)
    _style_ax(ax1, "", "T_skin,max per block (°C)", "Skin temperature proxy (T_skin,max)")
    _style_ax(ax2, "Time since placement (s)", "Estimated T_core (°C)",
              "Estimated core body temperature")
    handles, settle = [], 0.0
    for g_name, gd in all_group_data.items():
        series = [(s, v) for s, v in gd.animal_series.items() if "tcore" in v]
        if not series or gd.common_t is None:
            continue
        t, color = gd.common_t, gd.color
        rows_c, rows_s = [], []
        for stem, ser in series:
            if ser["excluded"]:
                continue
            for ax, key, store in ((ax1, "tskin", rows_s), (ax2, "tcore", rows_c)):
                ax.plot(t, ser[key], color=color, lw=1.0, alpha=0.45,
                        label=f"{g_name} · {_animal_label(stem)}")
                store.append(ser[key])
        if rows_c:
            for ax, rows in ((ax1, rows_s), (ax2, rows_c)):
                mean, sem, _ = _mean_sem_nan(rows)
                if len(rows) > 1:
                    ax.fill_between(t, mean - sem, mean + sem, color=color,
                                    alpha=_SEM_ALPHA, lw=0, label=f"{g_name} mean ± SEM")
                ax.plot(t, mean, color=color, lw=2.8, label=f"{g_name} mean")
            handles.append(Line2D([], [], color=color, lw=2.8,
                                  label=f"{g_name} (n={len(rows_c)})"))
        settle = max(settle, gd.settle_s)
    _shade_settle(ax1, settle)
    _shade_settle(ax2, settle)
    ax2.legend(handles=handles)
    fig.text(0.5, 0.01,
             "T_core estimated via van der Vinne et al. (2020) Sci Rep 10:20680 "
             "(slope 0.93, intercept 7.1 °C; between-animal error ≈ ±0.9 °C — use for within-animal "
             "changes and group comparisons, not absolute core temperature).",
             ha="center", va="bottom", fontsize=8, color="#6C757D", style="italic", wrap=True)
    fig.tight_layout(pad=1.5, rect=[0, 0.05, 1, 1])
    return fig


# ── 3. Per-animal overview ────────────────────────────────────────────────────

def _runs(mask: np.ndarray, t: np.ndarray, min_len_s: float = 0.5):
    """(start, end) times of runs where mask is True and lasting ≥ min_len_s."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(int), [0]]))
    out = []
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        t0, t1 = t[s], t[min(e, len(t) - 1)]
        if t1 - t0 >= min_len_s:
            out.append((t0, t1))
    return out


def build_animal_overview(roi_df: pd.DataFrame, title: str, cam_width: int,
                          cam_height: int, qc: dict | None = None) -> Figure:
    """
    One figure per recording that shows everything needed to trust (or distrust) it:
    temperature with untracked stretches marked, centroid speed, tracked/untracked
    raster, and where the animal spent its time in view.
    """
    fig = plt.figure(figsize=(12.5, 7), facecolor=_PANEL_BG)
    gs = fig.add_gridspec(3, 2, width_ratios=[2.7, 1], height_ratios=[3, 1.3, 0.55],
                          hspace=0.14, wspace=0.16)
    ax_t = fig.add_subplot(gs[0, 0])
    ax_s = fig.add_subplot(gs[1, 0], sharex=ax_t)
    ax_c = fig.add_subplot(gs[2, 0], sharex=ax_t)
    ax_m = fig.add_subplot(gs[0, 1])
    ax_b = fig.add_subplot(gs[1:, 1])
    _style_ax(ax_t, "", "Body temperature (°C)")
    _style_ax(ax_s, "", "Speed (px/s)")
    _style_ax(ax_c, "Time since placement (s)", "")
    _style_ax(ax_m, "Column (px)", "Row (px)", "Time in view")

    t = roi_df["time_s"].to_numpy()
    roi = roi_df["roi_max"].to_numpy()
    valid = np.isfinite(roi)
    settle = _settle_of(roi_df)
    pauses = (roi_df.loc[roi_df["gap_before_s"] > 0, "time_s"].to_numpy()
              if "gap_before_s" in roi_df.columns else np.array([]))

    # untracked stretches
    for a, b in _runs(~valid, t):
        ax_t.axvspan(a, b, color="#E07B5A", alpha=0.18, lw=0, zorder=0)
    ax_t.plot(t[valid], roi[valid], ".", ms=1.6, color="#9AA5A0", alpha=0.6,
              label="tracked frames", zorder=1)
    sm = (pd.Series(roi, index=pd.to_timedelta(t, unit="s"))
          .rolling("5s", center=True, min_periods=3).median().to_numpy())
    ax_t.plot(t, sm, color=_ACCENT, lw=2.0, label="body, 5-s rolling median", zorder=3)
    if "head_max" in roi_df.columns and np.isfinite(roi_df["head_max"]).any():
        hs = (pd.Series(roi_df["head_max"].to_numpy(), index=pd.to_timedelta(t, unit="s"))
              .rolling("5s", center=True, min_periods=3).median().to_numpy())
        ax_t.plot(t, hs, color="#E07B5A", lw=1.2, ls="--", label="head (secondary readout)",
                  zorder=3)
    post = t >= settle
    edges = np.arange(settle, t[-1] + 30, 30.0)
    cen, mu = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (t >= lo) & (t < hi) & valid
        if sel.sum() >= 0.2 * 30 * (len(t) / max(t[-1], 1)):
            cen.append((lo + hi) / 2); mu.append(roi[sel].mean())
    if cen:
        ax_t.plot(cen, mu, "o-", color="#212529", lw=1.2, ms=4, zorder=4, label="30-s bin mean")
    ax_t.plot([], [], color="#E07B5A", alpha=0.4, lw=8, label="body not tracked")
    if len(pauses):
        ax_t.plot([], [], color="#6C757D", lw=1, ls=":", label="recorder pause (cut out)")
    ax_t.legend(loc="lower right", ncol=2, fontsize=8)
    plt.setp(ax_t.get_xticklabels(), visible=False)

    # speed
    if "speed_px_s" in roi_df.columns:
        sp = roi_df["speed_px_s"].to_numpy()
        ax_s.plot(t, sp, color="#4C78A8", lw=0.8)
        bl = roi_df.attrs.get("body_len_px")
        if bl and np.isfinite(bl):
            ax_s.axhline(0.25 * bl, color="#6C757D", lw=0.8, ls="--")
            ax_s.text(t[0] + 0.5 * (t[-1] - t[0]), 0.25 * bl, "moving ≥ 0.25 body length/s", fontsize=7,
                      color="#6C757D", va="bottom", ha="center", backgroundcolor="white")
    plt.setp(ax_s.get_xticklabels(), visible=False)

    # tracking raster
    ax_c.fill_between(t, 0, 1, where=valid, step="mid", color=_ACCENT, alpha=0.7, lw=0)
    ax_c.fill_between(t, 0, 1, where=~valid, step="mid", color="#E07B5A", alpha=0.7, lw=0)
    ax_c.set_yticks([]); ax_c.set_ylim(0, 1); ax_c.grid(False)
    ax_c.set_ylabel("tracked", fontsize=8)
    for ax in (ax_t, ax_s, ax_c):
        ax.set_xlim(t[0], t[-1])
    for ax in (ax_t, ax_s, ax_c):
        if settle > 0:
            ax.axvspan(0, settle, color="#ADB5BD", alpha=0.18, lw=0, zorder=0)
        for tp in pauses:        # thin marker only: the data simply continue after a pause
            ax.axvline(tp, color="#6C757D", lw=1, ls=":", zorder=5)
    if settle > 0:
        ax_t.text(settle / 2, ax_t.get_ylim()[1], "settling\n(excluded)", ha="center",
                  va="top", fontsize=7, color="#6C757D")

    # occupancy
    rows = roi_df["max_row"].to_numpy().astype(int)
    cols = roi_df["max_col"].to_numpy().astype(int)
    ok = (rows >= 0) & (cols >= 0)
    freq = np.zeros((cam_height, cam_width), dtype=np.float32)
    if ok.any():
        np.add.at(freq, (rows[ok].clip(0, cam_height - 1), cols[ok].clip(0, cam_width - 1)), 1)
    freq = gaussian_filter1d(gaussian_filter1d(freq, 1.5, axis=0), 1.5, axis=1)
    im = ax_m.imshow(100 * freq / max(freq.sum(), 1), origin="upper", cmap="magma",
                     extent=[0, cam_width, cam_height, 0], aspect="equal")
    ax_m.grid(False)
    plt.colorbar(im, ax=ax_m, label="% of tracked time per px", shrink=0.8)

    # visibility check: does a window's temperature depend on how much of it was tracked?
    flagged = bool(qc) and bias_flagged(qc)
    wc, wm = [], []
    fps_est = len(t) / max(t[-1], 1.0)
    for lo in np.arange(settle, t[-1] - 15, 30.0):
        sel = (t >= lo) & (t < lo + 30)
        okw = sel & valid
        frac = okw.sum() / max(sel.sum(), 1)
        if frac >= 0.2 and okw.any():
            wc.append(100 * frac); wm.append(roi[okw].mean())
    _style_ax(ax_b, "Share of 30-s window tracked (%)",
              "Window mean temperature (°C)", "Visibility check")
    col = "#C0392B" if flagged else "#4C78A8"
    ax_b.scatter(wc, wm, color=col, edgecolor="white", s=34, zorder=3)
    if len(wc) >= 3 and np.ptp(wc) > 0:
        k, c0 = np.polyfit(wc, wm, 1)
        xx = np.array([min(wc), max(wc)])
        ax_b.plot(xx, k * xx + c0, color=col, lw=1.4, ls="--")
    ax_b.set_title("Visibility check" + (" — FLAGGED" if flagged else " — passed"),
                   fontsize=10, color=col, fontweight="bold")

    sub = []
    if qc:
        sub.append(f"tracked {qc.get('coverage_pct', float('nan')):.0f}% after settling")
        if np.isfinite(qc.get("mean_roi_max", np.nan)):
            sub.append(f"mean {qc['mean_roi_max']:.2f} °C")
        if np.isfinite(qc.get("pct_time_moving", np.nan)):
            sub.append(f"moving {qc['pct_time_moving']:.0f}% of tracked time")
    fig.suptitle(title + ("\n" + "  ·  ".join(sub) if sub else ""), fontsize=_TITLE_SIZE,
                 fontweight="bold", color=_TEXT)
    note = bias_sentence(qc) if qc else ""
    if note:
        fig.text(0.07, 0.012, textwrap.fill(note, 170), fontsize=8.5, va="bottom",
                 color="#C0392B" if flagged else "#6C757D")
    fig.subplots_adjust(left=0.07, right=0.95, top=0.89, bottom=0.17 if note else 0.08)
    keep = [c for c in ("time_s", "roi_max", "roi_mean", "body_valid", "body_area_px",
                        "speed_px_s", "moving", "max_row", "max_col", "in_settle")
            if c in roi_df.columns]
    fig._glod_source = roi_df[keep].rename(columns={"max_row": "centroid_row",
                                                    "max_col": "centroid_col"})
    return fig


# ── 4. Tracking quality across animals ────────────────────────────────────────

def build_tracking_qc(all_group_data: dict, qc_df: pd.DataFrame, min_coverage: float) -> Figure:
    """Share of post-settling frames in which the body was tracked, per animal."""
    qc = qc_df.reset_index(drop=True)
    n = len(qc)
    # Footnote: what the bars mean and why any animal carries a ⚠ — the flag is NOT about low
    # coverage (a well-tracked animal can be flagged), so it is spelled out under the plot.
    notes = ["Bars: share of the recording (after settling) in which the body was tracked — higher is better. "
             "A ⚠ is a separate test: does an animal's temperature depend on how visible it was?"]
    for r in qc.to_dict("records"):
        if str(r.get("qc_flag", "")).startswith("CHECK"):
            notes.append(f"{r['group']} · {r['animal']}: " + bias_sentence(r).replace("Why flagged: ", ""))
        elif str(r.get("qc_flag", "")).startswith("EXCLUDED"):
            notes.append(f"{r['group']} · {r['animal']}: {r['qc_flag']}.")
    wrapped = "\n".join(textwrap.fill(s, 125, subsequent_indent="   ") for s in notes)
    n_lines = wrapped.count("\n") + 1
    fig, ax = _new_fig(figsize=(9.5, max(2.8, 0.38 * n + 1.5 + 0.2 * n_lines)))
    _style_ax(ax, "Frames with body tracked after settling (%)", "", "Tracking quality")
    y = np.arange(n)[::-1]
    colors = [all_group_data[g].color if g in all_group_data else "#6C757D" for g in qc["group"]]
    ax.barh(y, qc["coverage_pct"].fillna(0), color=colors, alpha=0.85, height=0.62, zorder=2)
    ax.axvline(100 * min_coverage, color="#C0392B", lw=1.2, ls="--", zorder=3)
    ax.text(100 * min_coverage + 1, -0.62, "exclusion threshold", color="#C0392B",
            fontsize=8, va="center")
    ax.set_ylim(-0.9, n - 0.4)
    for yi, r in zip(y, qc.itertuples()):
        flag = str(r.qc_flag)
        mark = "✗ excluded" if flag.startswith("EXCLUDED") else "⚠ see note" if flag.startswith(
            "CHECK") else ""
        ax.text(min(float(r.coverage_pct) if np.isfinite(r.coverage_pct) else 0, 100) + 1,
                yi, f"{r.coverage_pct:.0f}%  {mark}", va="center", fontsize=8, color=_TEXT)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{g} · {a}" for g, a in zip(qc["group"], qc["animal"])], fontsize=9)
    ax.set_xlim(0, 118)
    ax.grid(axis="y", visible=False)
    bottom = min(0.55, (0.2 * n_lines + 0.25) / fig.get_figheight())
    fig.tight_layout(pad=1.5, rect=[0, bottom, 1, 1])
    fig.text(0.02, 0.01, wrapped, fontsize=8, va="bottom", color="#495057")
    fig._glod_source = qc[[c for c in ("group", "animal", "recording", "coverage_pct", "longest_gap_s",
                                       "tracking_bias_rho", "tracking_bias_p", "qc_flag")
                           if c in qc.columns]]
    return fig


def _metric_grid(n: int, per_row: int = 3):
    """Figure with up to ``per_row`` metric panels per row (constrained layout, so
    labels stay clear of neighbouring panels when the canvas is resized)."""
    ncols = min(n, per_row)
    nrows = -(-n // per_row)
    fig, axes = _new_fig(nrows, ncols, figsize=(4.3 * ncols, 5.0 * nrows),
                         squeeze=False)
    fig.set_layout_engine("constrained")
    flat = list(axes.ravel())
    for extra in flat[n:]:
        extra.set_visible(False)
    return fig, flat[:n]


# ── 9. Group comparison bar charts ───────────────────────────────────────────

_METRIC_LABELS = {
    "mean_roi_max":            "Mean body temp.\n(°C, after settling)",
    "mean_roi_max_resting":    "Mean body temp., resting\n(°C, not moving)",
    "mean_head_roi_max":       "Mean head temp.\n(°C, secondary)",
    "mean_normalized_roi_max": "Mean Δ from baseline\n(fraction)",
    "peak_normalized_roi_max": "Peak Δ from baseline\n(fraction, 30-s bins)",
    "slope_roi_max_c_per_min": "Temperature slope\n(°C/min)",
    "mean_tcore_estimated":    "Mean T_core est. (°C)",
    "peak_tcore_estimated":    "Peak T_core est. (°C)",
    "mean_speed_px_s":         "Mean speed (px/s)",
    "pct_time_moving":         "Time moving (%)",
}


def build_group_comparison_bars(all_group_data: dict, stats_results: dict) -> Figure:
    """
    Side-by-side bar charts per metric; individual data points overlaid;
    significance brackets between bars.
    """
    metrics = list(_METRIC_LABELS.keys())
    n_metrics = len(metrics)
    group_names = list(all_group_data.keys())
    n_groups = len(group_names)
    colors = [gd.color for gd in all_group_data.values()]

    per_file_df = stats_results.get("per_file_df")
    if per_file_df is None or per_file_df.empty:
        fig, ax = _new_fig()
        ax.text(0.5, 0.5, "No data for group comparison",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = _metric_grid(n_metrics)

    test_rows = stats_results.get("test_rows", [])
    ph_rows = stats_results.get("posthoc_rows", [])
    paired = bool(stats_results.get("paired"))

    rng = np.random.default_rng(42)

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        _style_ax(ax, "", _METRIC_LABELS[metric])
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(group_names, rotation=20, ha="right", fontsize=9)

        group_vals = {}
        bar_tops = []

        for g_idx, (g_name, gd) in enumerate(all_group_data.items()):
            vals = per_file_df.loc[per_file_df["group"] == g_name, metric].dropna().to_numpy()
            group_vals[g_name] = vals
            mean = float(np.mean(vals)) if len(vals) > 0 else 0.0
            sem  = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            color = colors[g_idx]

            ax.bar(g_idx, mean, color=color, alpha=0.75, width=0.55,
                   edgecolor="white", linewidth=1.2, zorder=2)
            ax.errorbar(g_idx, mean, yerr=sem, fmt="none", color=_TEXT,
                        capsize=5, linewidth=1.5, zorder=3)

            # Individual points (no jitter when paired, so lines connect cleanly)
            if len(vals) > 0:
                jitter = (np.zeros(len(vals)) if paired
                          else rng.uniform(-0.12, 0.12, size=len(vals)))
                ax.scatter(np.full(len(vals), g_idx) + jitter, vals,
                           color=color, edgecolors=_TEXT, linewidths=0.6,
                           s=28, zorder=4, alpha=0.9)
            bar_tops.append(mean + sem)

        # Paired design: one line per animal across conditions
        if paired:
            sub = per_file_df[per_file_df["group"].isin(group_names)]
            for _, rows_a in sub.groupby("file"):
                pts = {group_names.index(r.group): getattr(r, metric)
                       for r in rows_a.itertuples() if not pd.isna(getattr(r, metric))}
                xs = sorted(pts)
                if len(xs) > 1:
                    ax.plot(xs, [pts[x] for x in xs], color="#6C757D",
                            lw=0.9, alpha=0.7, zorder=3.5)
            bar_tops += [float(np.nanmax(group_vals[g])) for g in group_names
                         if len(group_vals.get(g, [])) > 0]

        # ── Significance brackets ─────────────────────────────────────────
        y_max = max(bar_tops) if bar_tops else 1.0
        y_bracket = y_max * 1.08

        # Find relevant test rows for this metric
        for row in (test_rows + ph_rows):
            if row.get("metric") != metric:
                continue
            sig = row.get("significance", "ns")
            gA = row.get("group_A", "")
            gB = row.get("group_B", "")
            if gA not in group_names or gB not in group_names:
                continue
            xA = group_names.index(gA)
            xB = group_names.index(gB)
            _sig_bracket(ax, xA, xB, y_bracket, sig, color=_TEXT)
            y_bracket *= 1.12  # stack brackets upward

        # Test label at bottom
        test_name = next((r["test_name"] for r in test_rows
                          if r.get("metric") == metric), "")
        if test_name:
            ax.set_xlabel(f"({test_name})", fontsize=8, color="#6C757D", style="italic")

    fig.suptitle("Group Comparison — Summary Metrics", fontsize=_TITLE_SIZE,
                 fontweight="bold", color=_TEXT)
    return fig


# ── 11. Time-point comparison bars ───────────────────────────────────────────

def build_timepoint_bars(all_group_data: dict, stats_results: dict, settings) -> Figure:
    """
    Grouped bar chart: one cluster per user-set time point, one bar per group
    (mean ± SEM, individual animals overlaid), with between-group significance
    brackets taken from the time-point tests.
    """
    from ..processing.stats import TIMEPOINT_SIGNALS, timepoint_label

    tp_df = stats_results.get("timepoint_df")
    tps = [float(t) for t in settings.timepoints_s]
    group_names = list(all_group_data.keys())
    n_groups = len(group_names)
    sig_name, unit = TIMEPOINT_SIGNALS.get(settings.timepoint_signal, ("", ""))

    fig, ax = _new_fig(figsize=(max(6.0, 2.6 * len(tps) + 2.0), 5.2))
    if tp_df is None or tp_df.empty:
        ax.text(0.5, 0.5, "No data for time-point comparison",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    _style_ax(ax, "", f"{sig_name} ({unit})")
    width = 0.8 / n_groups
    rng = np.random.default_rng(42)
    ph = (stats_results.get("timepoint_test_rows", [])
          + stats_results.get("timepoint_posthoc_rows", []))
    tops = {}
    paired = bool(stats_results.get("paired"))

    for ti, t in enumerate(tps):
        label = timepoint_label(t)
        tops[ti] = []
        for gi, (g, gd) in enumerate(all_group_data.items()):
            vals = tp_df[(tp_df.group == g) & (tp_df.timepoint_s == t)]["value"] \
                .dropna().to_numpy()
            x = ti + (gi - (n_groups - 1) / 2) * width
            mean = float(np.mean(vals)) if len(vals) else np.nan
            sem = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
            if len(vals):
                ax.bar(x, mean, width=width * 0.9, color=gd.color, alpha=0.75,
                       edgecolor="white", linewidth=1.0, zorder=2,
                       label=g if ti == 0 else None)
                ax.errorbar(x, mean, yerr=sem, fmt="none", color=_TEXT,
                            capsize=4, linewidth=1.4, zorder=3)
                jit = (np.zeros(len(vals)) if paired
                       else rng.uniform(-width * 0.2, width * 0.2, len(vals)))
                ax.scatter(x + jit, vals,
                           color=gd.color, edgecolors=_TEXT, linewidths=0.6,
                           s=24, zorder=4, alpha=0.9)
                tops[ti].append(mean + sem)

        # Paired design: connect each animal across conditions at this time point
        if paired:
            sub = tp_df[tp_df.timepoint_s == t]
            for _, rows_a in sub.groupby("file"):
                pts = [(ti + (group_names.index(r.group) - (n_groups - 1) / 2) * width,
                        r.value) for r in rows_a.itertuples() if not pd.isna(r.value)]
                if len(pts) > 1:
                    pts.sort()
                    ax.plot(*zip(*pts), color="#6C757D", lw=0.9, alpha=0.7, zorder=3.5)
                    tops[ti].append(max(v for _, v in pts))

        # Brackets for this time point (2 groups: test row; 3+: post-hoc rows)
        y0 = max(tops[ti]) if tops[ti] else 0.0
        span = (ax.get_ylim()[1] - ax.get_ylim()[0]) or 1.0
        y = y0 + span * 0.04
        for r in ph:
            if r.get("timepoint") != label:
                continue
            gA, gB = r.get("group_A", ""), r.get("group_B", "")
            if gA not in group_names or gB not in group_names:
                continue
            xa = ti + (group_names.index(gA) - (n_groups - 1) / 2) * width
            xb = ti + (group_names.index(gB) - (n_groups - 1) / 2) * width
            _sig_bracket(ax, xa, xb, y, r.get("significance", "ns"), color=_TEXT)
            y += span * 0.08

    ax.set_xticks(range(len(tps)))
    ax.set_xticklabels([timepoint_label(t) for t in tps], fontsize=_LABEL_SIZE)
    ax.margins(y=0.15)
    if n_groups > 1:
        ax.legend(loc="best")
    ax.set_title(f"Temperature at set time points — mean ± SEM, "
                 f"{settings.timepoint_window_s:g}-s window",
                 fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT, pad=8)
    fig.tight_layout(pad=1.5)
    return fig
