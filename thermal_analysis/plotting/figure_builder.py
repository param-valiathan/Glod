"""
Matplotlib figure builders for Glöd (10 plot types).

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
from scipy.ndimage import gaussian_filter1d

from ..processing.roi_analysis import poly_fit, rate_of_change, smooth_series
from ..utils.config import POLY_DEGREE

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


# ── 1. Normalised ROI time series (Combine temp apap.py style) ───────────────

def build_normalized_roi(all_group_data: dict) -> Figure:
    """Multi-group normalised ROI max over time with SEM shaded bands + poly fit."""
    fig, ax = _new_fig(figsize=(11, 5))
    _style_ax(ax, "Time (s)", "Normalised ROI Max (Δ from baseline)",
              "Normalised ROI Temperature — All Groups")

    handles = []
    for g_name, gd in all_group_data.items():
        if gd.common_t is None or gd.roi_mean is None:
            continue
        t = gd.common_t
        mean = gd.roi_mean
        sem = gd.roi_sem if gd.roi_sem is not None else np.zeros_like(mean)
        color = gd.color

        ax.fill_between(t, mean - sem, mean + sem, alpha=_SEM_ALPHA, color=color)
        line, = ax.plot(t, mean, color=color, lw=_LINE_W, label=g_name)

        # Polynomial fit overlay
        fit = poly_fit(t, smooth_series(mean), degree=POLY_DEGREE)
        ax.plot(t, fit, color=color, lw=1.0, ls="--", alpha=0.7)

        handles.append(mpatches.Patch(color=color, label=g_name))

    ax.axhline(0, color=_GRID, lw=1.0, ls="-")
    ax.legend(handles=handles)
    fig.tight_layout(pad=1.5)
    return fig


# ── 2. Rate of change of ROI (°C/min) ───────────────────────────────────────

def build_rate_of_change(all_group_data: dict) -> Figure:
    """ROI rate of change in °C/min, Gaussian-smoothed, multi-group."""
    fig, ax = _new_fig(figsize=(11, 5))
    _style_ax(ax, "Time (s)", "Rate of Change (°C / min)",
              "ROI Temperature Rate of Change — All Groups")

    handles = []
    for g_name, gd in all_group_data.items():
        if gd.common_t is None or gd.roi_mean is None:
            continue
        t = gd.common_t
        roc = rate_of_change(t, gd.roi_mean)
        roc_sem = (rate_of_change(t, gd.roi_sem)
                   if gd.roi_sem is not None else np.zeros_like(roc))
        color = gd.color

        ax.fill_between(t, roc - abs(roc_sem), roc + abs(roc_sem),
                        alpha=_SEM_ALPHA, color=color)
        ax.plot(t, roc, color=color, lw=_LINE_W, label=g_name)
        handles.append(mpatches.Patch(color=color, label=g_name))

    ax.axhline(0, color=_GRID, lw=1.0, ls="-")
    ax.legend(handles=handles)
    fig.tight_layout(pad=1.5)
    return fig


# ── 3. ROI Max vs ROI Mean (per camera file) ─────────────────────────────────

def build_roi_max_vs_mean(roi_df: pd.DataFrame, title: str = "") -> Figure:
    """Per-camera ROI max (solid) vs ROI mean (dashed) with poly fit."""
    fig, ax = _new_fig(figsize=(10, 4.5))
    _style_ax(ax, "Time (s)", "Temperature (°C)",
              title or "ROI Max vs ROI Mean")

    t = roi_df["time_s"].to_numpy()
    roi_max = roi_df["roi_max_smoothed"].to_numpy()
    roi_mean = roi_df["roi_mean"].to_numpy()

    ax.plot(t, roi_max, color=_ACCENT, lw=_LINE_W, label="ROI Max (smoothed)")
    ax.plot(t, roi_mean, color="#E07B5A", lw=_LINE_W, ls="--", label="ROI Mean")

    fit_max = poly_fit(t, roi_max, POLY_DEGREE)
    ax.plot(t, fit_max, color=_ACCENT, lw=1.0, ls=":", alpha=0.8, label="Poly fit (Max)")

    ax.legend()
    fig.tight_layout(pad=1.5)
    return fig


# ── 4. Max pixel temperature (per camera) ────────────────────────────────────

def build_max_pixel_temperature(roi_df: pd.DataFrame, title: str = "") -> Figure:
    """Max pixel temperature with ±SEM band and polynomial fit."""
    fig, ax = _new_fig(figsize=(10, 4.5))
    _style_ax(ax, "Time (s)", "Temperature (°C)",
              title or "Max Pixel Temperature")

    t = roi_df["time_s"].to_numpy()
    max_t = roi_df["max_temp"].to_numpy()
    roi_max = roi_df["roi_max_smoothed"].to_numpy()

    # Rolling SEM over 11-point window as proxy error band
    sem_band = pd.Series(max_t).rolling(11, min_periods=1, center=True).std().fillna(0).to_numpy()

    ax.fill_between(t, max_t - sem_band * 0.5, max_t + sem_band * 0.5,
                    alpha=0.15, color="#E07B5A", label="±0.5 SD band")
    ax.plot(t, max_t, color="#E07B5A", lw=_LINE_W, alpha=0.8, label="Max Pixel Temp")
    ax.plot(t, roi_max, color=_ACCENT, lw=_LINE_W, ls="--", label="ROI Max (smoothed)")

    fit = poly_fit(t, max_t, POLY_DEGREE)
    ax.plot(t, fit, color="#E07B5A", lw=1.2, ls=":", label="Poly fit")

    ax.legend()
    fig.tight_layout(pad=1.5)
    return fig


# ── 5. Max pixel spatial trajectory ──────────────────────────────────────────

def build_max_pixel_movement(roi_df: pd.DataFrame, cam_width: int,
                              cam_height: int, title: str = "") -> Figure:
    """Spatial trajectory + visit-frequency heatmap of the hottest pixel."""
    fig, (ax1, ax2) = _new_fig(1, 2, figsize=(12, 5))
    _style_ax(ax1, "Column (px)", "Row (px)", "Max Pixel Trajectory")
    _style_ax(ax2, "Column (px)", "Row (px)", "Visit Frequency Heatmap")

    rows = roi_df["max_row"].to_numpy().astype(int)
    cols = roi_df["max_col"].to_numpy().astype(int)
    temps = roi_df["max_temp"].to_numpy()

    # ── Trajectory scatter (coloured by temperature) ──────────────────────
    sc = ax1.scatter(cols, rows, c=temps, cmap="inferno", s=6, alpha=0.6,
                     vmin=temps.min(), vmax=temps.max())
    # Draw trajectory line
    ax1.plot(cols, rows, color="#C8D0CC", lw=0.5, alpha=0.4, zorder=0)
    ax1.invert_yaxis()
    ax1.set_xlim(0, cam_width)
    ax1.set_ylim(cam_height, 0)
    plt.colorbar(sc, ax=ax1, label="Temperature (°C)", shrink=0.85)

    # ── Frequency heatmap ─────────────────────────────────────────────────
    freq = np.zeros((cam_height, cam_width), dtype=np.float32)
    np.add.at(freq, (rows.clip(0, cam_height - 1), cols.clip(0, cam_width - 1)), 1)
    freq_smooth = gaussian_filter1d(gaussian_filter1d(freq, sigma=1, axis=0), sigma=1, axis=1)
    im = ax2.imshow(freq_smooth, origin="upper", cmap="hot",
                    extent=[0, cam_width, cam_height, 0], aspect="auto")
    plt.colorbar(im, ax=ax2, label="Visit count", shrink=0.85)

    if title:
        fig.suptitle(title, fontsize=_TITLE_SIZE, fontweight="bold", color=_TEXT)
    fig.tight_layout(pad=1.5)
    return fig


# ── 6. Absolute temperature comparison ───────────────────────────────────────

def build_absolute_comparison(all_group_data: dict) -> Figure:
    """All groups' un-normalised ROI max on one axis for absolute temperature reference."""
    fig, ax = _new_fig(figsize=(11, 5))
    _style_ax(ax, "Time (s)", "ROI Max Temperature (°C)",
              "Absolute ROI Max Temperature — All Groups")

    handles = []
    for g_name, gd in all_group_data.items():
        # Reconstruct absolute mean from per-file dfs
        roi_series = []
        common_t_list = []
        for stem, roi_df in gd.roi_dfs.items():
            if len(roi_df) > 0:
                roi_series.append(roi_df["roi_max"].to_numpy())
                common_t_list.append(roi_df["time_s"].to_numpy())
        if not roi_series:
            continue

        max_t = max(t[-1] for t in common_t_list)
        t_common = np.arange(0, max_t, 30.0)
        interp = [np.interp(t_common, ct, rs) for ct, rs in zip(common_t_list, roi_series)]
        mean = np.mean(interp, axis=0)
        sem = (np.std(interp, axis=0) / np.sqrt(len(interp))
               if len(interp) > 1 else np.zeros_like(mean))

        color = gd.color
        ax.fill_between(t_common, mean - sem, mean + sem, alpha=_SEM_ALPHA, color=color)
        ax.plot(t_common, mean, color=color, lw=_LINE_W, label=g_name)
        handles.append(mpatches.Patch(color=color, label=g_name))

    ax.legend(handles=handles)
    fig.tight_layout(pad=1.5)
    return fig


# ── 7. Estimated T_core time series ──────────────────────────────────────────

def build_tcore_estimated(all_group_data: dict) -> Figure:
    """Dual panel: T_skin,max rolling avg (top) and estimated T_core (bottom)."""
    fig, (ax1, ax2) = _new_fig(2, 1, figsize=(11, 8), sharex=True)
    _style_ax(ax1, "", "T_skin,max (°C, 30-min rolling avg)",
              "Skin Temperature Proxy (T_skin,max)")
    _style_ax(ax2, "Time (s)", "Estimated T_core (°C)",
              "Estimated Core Body Temperature")

    handles = []
    for g_name, gd in all_group_data.items():
        # Build aligned list; skip None / too-short entries explicitly to avoid
        # key-order mismatches when some files failed T_core estimation.
        valid = [
            (tdf["time_s"].to_numpy(),
             tdf["tcore_estimated"].to_numpy(),
             tdf["tskin_max"].to_numpy())
            for tdf in gd.tcore_dfs.values()
            if tdf is not None and len(tdf) >= 2
        ]
        if not valid:
            continue

        # Use the longest time axis as the common reference
        t_ref = max(valid, key=lambda x: len(x[0]))[0]

        tcore_mat = np.array([np.interp(t_ref, t, tc) for t, tc, _ in valid])
        tskin_mat = np.array([np.interp(t_ref, t, ts) for t, _, ts in valid])

        tc_mean = tcore_mat.mean(axis=0)
        ts_mean = tskin_mat.mean(axis=0)

        if len(valid) > 1:
            tc_sem = tcore_mat.std(axis=0) / np.sqrt(len(valid))
            ts_sem = tskin_mat.std(axis=0) / np.sqrt(len(valid))
        else:
            tc_sem = np.zeros_like(tc_mean)
            ts_sem = np.zeros_like(ts_mean)

        color = gd.color
        ax1.fill_between(t_ref, ts_mean - ts_sem, ts_mean + ts_sem,
                         alpha=_SEM_ALPHA, color=color)
        ax1.plot(t_ref, ts_mean, color=color, lw=_LINE_W)

        ax2.fill_between(t_ref, tc_mean - tc_sem, tc_mean + tc_sem,
                         alpha=_SEM_ALPHA, color=color)
        ax2.plot(t_ref, tc_mean, color=color, lw=_LINE_W, label=g_name)

        handles.append(mpatches.Patch(color=color, label=g_name))

    ax2.legend(handles=handles)
    fig.text(0.5, 0.01,
             "T_core estimated via van der Vinne et al. (2020) Sci Rep 10:20680.\n"
             "Group-average: slope=0.93, intercept=7.1°C. Between-animal error ≈ ±0.9°C.",
             ha="center", va="bottom", fontsize=8, color="#6C757D", style="italic")
    fig.tight_layout(pad=1.5, rect=[0, 0.05, 1, 1])
    return fig


# ── 8. T_core rate of change ─────────────────────────────────────────────────

def build_tcore_rate_of_change(all_group_data: dict) -> Figure:
    """Rate of change of estimated T_core in °C/min."""
    fig, ax = _new_fig(figsize=(11, 5))
    _style_ax(ax, "Time (s)", "d(T_core_est) / dt  (°C / min)",
              "Estimated Core Temperature Rate of Change — All Groups")

    handles = []
    for g_name, gd in all_group_data.items():
        if gd.common_t is None or gd.tcore_mean is None:
            continue
        t = gd.common_t
        roc = rate_of_change(t, gd.tcore_mean)
        color = gd.color
        ax.plot(t, roc, color=color, lw=_LINE_W, label=g_name)
        handles.append(mpatches.Patch(color=color, label=g_name))

    ax.axhline(0, color=_GRID, lw=1.0)
    ax.legend(handles=handles)
    fig.tight_layout(pad=1.5)
    return fig


# ── 9. Group comparison bar charts ───────────────────────────────────────────

_METRIC_LABELS = {
    "peak_normalized_roi_max": "Peak Norm. ROI Max\n(Δ from baseline)",
    "auc_normalized_roi_max":  "AUC Norm. ROI Max\n(Δ·s)",
    "time_to_peak_s":          "Time to Peak (s)",
    "peak_tcore_estimated":    "Peak T_core est. (°C)",
    "auc_tcore_estimated":     "AUC T_core est. (°C·s)",
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

    fig, axes = _new_fig(1, n_metrics, figsize=(3.5 * n_metrics, 5.5))
    if n_metrics == 1:
        axes = [axes]

    test_rows = stats_results.get("test_rows", [])
    ph_rows = stats_results.get("posthoc_rows", [])

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

            # Jittered individual points
            if len(vals) > 0:
                jitter = rng.uniform(-0.12, 0.12, size=len(vals))
                ax.scatter(np.full(len(vals), g_idx) + jitter, vals,
                           color=color, edgecolors=_TEXT, linewidths=0.6,
                           s=28, zorder=4, alpha=0.9)
            bar_tops.append(mean + sem)

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
    fig.tight_layout(pad=1.5)
    return fig


# ── 10. Violin + box plot ─────────────────────────────────────────────────────

def build_group_violin_box(all_group_data: dict, stats_results: dict) -> Figure:
    """Violin + box plot overlay per metric showing distribution of per-file values."""
    metrics = list(_METRIC_LABELS.keys())
    n_metrics = len(metrics)
    group_names = list(all_group_data.keys())
    n_groups = len(group_names)
    colors = [gd.color for gd in all_group_data.values()]

    per_file_df = stats_results.get("per_file_df")
    if per_file_df is None or per_file_df.empty:
        fig, ax = _new_fig()
        ax.text(0.5, 0.5, "No data for violin plot",
                ha="center", va="center", transform=ax.transAxes)
        return fig

    fig, axes = _new_fig(1, n_metrics, figsize=(3.5 * n_metrics, 5.5))
    if n_metrics == 1:
        axes = [axes]

    ph_rows = stats_results.get("posthoc_rows", []) or stats_results.get("test_rows", [])

    for ax_idx, metric in enumerate(metrics):
        ax = axes[ax_idx]
        _style_ax(ax, "", _METRIC_LABELS[metric])
        ax.set_xticks(range(n_groups))
        ax.set_xticklabels(group_names, rotation=20, ha="right", fontsize=9)

        group_vals = []
        bar_tops = []
        for g_name in group_names:
            vals = per_file_df.loc[per_file_df["group"] == g_name, metric].dropna().to_numpy()
            group_vals.append(vals)
            bar_tops.append(vals.max() if len(vals) > 0 else 0.0)

        for g_idx, (vals, color) in enumerate(zip(group_vals, colors)):
            if len(vals) >= 3:
                parts = ax.violinplot([vals], positions=[g_idx], widths=0.5,
                                       showmedians=False, showextrema=False)
                for pc in parts["bodies"]:
                    pc.set_facecolor(color)
                    pc.set_alpha(0.35)
                    pc.set_edgecolor("none")

            if len(vals) > 0:
                ax.boxplot([vals], positions=[g_idx], widths=0.28,
                                patch_artist=True, notch=False,
                                boxprops=dict(facecolor=color, alpha=0.6,
                                              edgecolor=_TEXT, linewidth=1.2),
                                medianprops=dict(color="white", linewidth=2),
                                whiskerprops=dict(color=_TEXT, linewidth=1.2),
                                capprops=dict(color=_TEXT, linewidth=1.2),
                                flierprops=dict(marker="o", color=color,
                                                markersize=4, alpha=0.6))

        # Significance brackets
        y_max = max(bar_tops) if bar_tops else 1.0
        y_bracket = y_max * 1.08
        for row in ph_rows:
            if row.get("metric") != metric:
                continue
            sig = row.get("significance", "ns")
            gA, gB = row.get("group_A", ""), row.get("group_B", "")
            if gA not in group_names or gB not in group_names:
                continue
            _sig_bracket(ax, group_names.index(gA), group_names.index(gB),
                         y_bracket, sig)
            y_bracket *= 1.12

    fig.suptitle("Group Distribution — Violin + Box Plots", fontsize=_TITLE_SIZE,
                 fontweight="bold", color=_TEXT)
    fig.tight_layout(pad=1.5)
    return fig
