"""
Statistical tests for group-wise comparison of ROI and T_core metrics.

Design is chosen for small cohorts (typically n = 3-10 mice):
  Independent groups
    2 groups  → Welch t-test by default; Mann-Whitney U only when Shapiro-Wilk fails
                AND a rank test can reach p < 0.05 at these n (3 vs 3 cannot)
    3+ groups → one-way ANOVA (Kruskal-Wallis under the same rule), then Tukey HSD
                (Dunn + Holm after Kruskal-Wallis) when the omnibus p < 0.05
  Paired design (same animals in every condition; animals matched by name)
    2 conditions → paired t-test (Wilcoxon signed-rank if differences non-normal and n >= 6)
    3+           → repeated-measures ANOVA with Greenhouse-Geisser (Friedman if non-normal,
                   n >= 5), then pairwise paired tests with Holm correction

Every number needed to report a result in a paper (n, df, statistic, exact p,
effect size, assumption checks, correction method) is returned in the result tables.
Unit of analysis: one animal (paired) or one recording file (independent groups).
"""

import itertools
import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..utils import naming


# ── Effect size helpers ───────────────────────────────────────────────────────

def _cohens_d(a, b):
    """Cohen's d for two independent samples (pooled std)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    pooled_std = np.sqrt(((na - 1) * np.var(a, ddof=1) + (nb - 1) * np.var(b, ddof=1))
                          / (na + nb - 2))
    if pooled_std == 0:
        return np.nan
    return (np.mean(a) - np.mean(b)) / pooled_std


def _rank_biserial(a, b):
    """Rank-biserial correlation as effect size for Mann-Whitney U."""
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return np.nan
    u_stat, _ = sp_stats.mannwhitneyu(a, b, alternative='two-sided')
    return 1 - (2 * u_stat) / (na * nb)


def _eta_squared(f_stat, groups):
    """Eta-squared effect size for one-way ANOVA."""
    all_vals = np.concatenate(groups)
    ns = [len(g) for g in groups]
    grand_mean = np.mean(all_vals)
    ss_between = sum(n * (np.mean(g) - grand_mean) ** 2 for n, g in zip(ns, groups))
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total if ss_total != 0 else np.nan


def _welch_df_and_ci(a, b, conf=0.95):
    """Welch–Satterthwaite df and CI of the mean difference (A − B)."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan, np.nan, np.nan
    va, vb = np.var(a, ddof=1) / na, np.var(b, ddof=1) / nb
    se = np.sqrt(va + vb)
    if se == 0:
        return np.nan, np.nan, np.nan
    df = (va + vb) ** 2 / (va ** 2 / (na - 1) + vb ** 2 / (nb - 1))
    tcrit = sp_stats.t.ppf(0.5 + conf / 2, df)
    diff = np.mean(a) - np.mean(b)
    return float(df), float(diff - tcrit * se), float(diff + tcrit * se)


def _significance_label(p):
    if p < 0.0001:
        return "****"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def p_text(p):
    """APA-style p-value text: 'p < .001' or 'p = .032'."""
    if p is None or (isinstance(p, float) and np.isnan(p)):
        return "NA"
    if p < 0.001:
        return "< 0.001"
    return f"= {p:.3f}"


def _f(x):
    """float or NaN → plain float (keeps full precision for the CSV)."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


# ── Per-file scalar extraction ────────────────────────────────────────────────

METRICS = [
    "mean_roi_max",
    "mean_roi_max_resting",
    "mean_head_roi_max",
    "mean_normalized_roi_max",
    "peak_normalized_roi_max",
    "slope_roi_max_c_per_min",
    "mean_tcore_estimated",
    "peak_tcore_estimated",
    "mean_speed_px_s",
    "pct_time_moving",
]

# Human-readable definitions and units (written to the data dictionary)
METRIC_INFO = {
    "mean_roi_max": ("Mean body temperature after the settling period (mean of 30-s bin means "
                     "of tracked frames, so every part of the recording counts equally)", "°C"),
    "mean_roi_max_resting": ("Same as mean_roi_max but only frames in which the animal was not moving "
                             "(speed <= 0.25 body lengths/s); controls for the movement dependence of "
                             "the reading (moving frames read cooler on the reference recordings)", "°C"),
    "mean_head_roi_max": ("Secondary readout: mean temperature of the head region (front 30 % of the "
                          "body axis) after settling; noisier and cooler than the body readout, "
                          "less dependent on visibility — use as a robustness check", "°C"),
    "mean_normalized_roi_max": ("Mean baseline-normalised body temperature after settling "
                                "(fractional change from baseline; ×100 for %)", "fraction"),
    "peak_normalized_roi_max": ("Highest 30-s bin mean of the baseline-normalised body temperature",
                                "fraction"),
    "slope_roi_max_c_per_min": ("Theil–Sen (robust) slope of the 30-s bin means after settling",
                                "°C/min"),
    "mean_tcore_estimated": ("Mean estimated core temperature (van der Vinne et al. 2020)", "°C"),
    "peak_tcore_estimated": ("Maximum estimated core temperature", "°C"),
    "mean_speed_px_s": ("Mean centroid speed over 1-s windows (tracked frames only); "
                        "pixels, not cm — compare only between identical camera/arena set-ups",
                        "px/s"),
    "pct_time_moving": ("Percentage of tracked time with speed > 0.25 body lengths/s", "%"),
}

BIN_S = 30.0          # time bin used for all per-animal summaries
MIN_BIN_COVERAGE = 0.2  # a bin needs ≥ this fraction of valid frames to count


def _bins(t, y, settle_s, fps):
    """30-s bin means of y over the post-settle recording.

    Returns (centres_s, means, coverage) — means are NaN for bins with too little valid
    data.  Equal weight per time bin means periods that are easy to track (mouse in
    view) cannot dominate the animal's summary value.
    """
    t_end = float(t[-1]) if len(t) else 0.0
    edges = np.arange(settle_s, t_end + BIN_S, BIN_S)
    centres, means, cov = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if hi - lo < BIN_S / 2 and len(centres):
            break
        sel = (t >= lo) & (t < hi)
        ok = sel & np.isfinite(y)
        exp = max(1.0, (min(hi, t_end) - lo) * fps)
        c = ok.sum() / exp
        centres.append((lo + min(hi, t_end)) / 2)
        cov.append(min(c, 1.0))
        means.append(float(y[ok].mean()) if c >= MIN_BIN_COVERAGE else np.nan)
    return np.array(centres), np.array(means), np.array(cov)


def extract_per_file_metrics(group_name: str, file_stem: str,
                              roi_df: pd.DataFrame,
                              tcore_df: pd.DataFrame,
                              baseline_end_sec: float = 300.0,
                              settle_s: float = 0.0) -> dict:
    """
    Per-animal scalar summaries (the unit of statistics) plus tracking-quality columns.

    Handling of untracked frames (no imputation):
      * the settling period and untracked (NaN) frames never enter a metric;
      * each metric is the mean/peak of 30-s *bin* means, bins with < 20 % valid frames
        are dropped, so a mouse that is easier to track at some times cannot bias the
        summary towards those times;
      * ``coverage_pct`` and ``longest_gap_s`` describe how much of the recording the
        value rests on; ``tracking_bias_rho/p`` is the Spearman correlation between a
        bin's coverage and its temperature — a significant value warns that visibility
        and temperature are related (selection bias) and the animal needs inspection.
    """
    out = {"group": group_name, "file": file_stem}
    nan_keys = [m for m in METRICS if not m.endswith(("_tcore_estimated",))]
    for k in nan_keys + ["coverage_pct", "longest_gap_s", "tracking_bias_rho",
                         "tracking_bias_p", "tracking_bias_n", "fps_measured", "n_pauses",
                         "paused_s"]:
        out[k] = np.nan
    out["qc_flag"] = "no data"

    if roi_df is not None and len(roi_df):
        t = roi_df["time_s"].to_numpy()
        fps = 1.0 / np.median(np.diff(t)) if len(t) > 2 else 10.0
        out["fps_measured"] = float(fps)
        if "gap_before_s" in roi_df.columns:        # recorder pauses (cut from the time axis)
            g_ = roi_df["gap_before_s"].to_numpy()
            out["n_pauses"] = int((g_ > 0).sum())
            out["paused_s"] = float(g_.sum())
        roi = roi_df["roi_max"].to_numpy()
        norm = roi_df["normalized_roi_max"].to_numpy()
        post = t >= settle_s
        valid = np.isfinite(roi)

        if post.any():
            out["coverage_pct"] = float(100 * valid[post].mean())
            # longest run of untracked frames after settling
            bad = (~valid & post).astype(int)
            run_edges = np.flatnonzero(np.diff(np.concatenate([[0], bad, [0]])))
            if len(run_edges):
                starts, ends = run_edges[::2], run_edges[1::2]
                out["longest_gap_s"] = float(max(
                    t[min(e, len(t) - 1)] - t[s] for s, e in zip(starts, ends)))
            else:
                out["longest_gap_s"] = 0.0

            c, m_abs, cov = _bins(t, roi, settle_s, fps)
            _, m_norm, _ = _bins(t, norm, settle_s, fps)
            ok = np.isfinite(m_abs)
            if ok.any():
                out["mean_roi_max"] = float(m_abs[ok].mean())
                out["mean_normalized_roi_max"] = float(m_norm[ok].mean())
                out["peak_normalized_roi_max"] = float(np.nanmax(m_norm))
            if "moving" in roi_df.columns:
                rest = np.where(roi_df["moving"].to_numpy() == 0, roi, np.nan)
                _, m_rest, _ = _bins(t, rest, settle_s, fps)
                if np.isfinite(m_rest).any():
                    out["mean_roi_max_resting"] = float(np.nanmean(m_rest))
            if "head_max" in roi_df.columns:
                _, m_head, _ = _bins(t, roi_df["head_max"].to_numpy(), settle_s, fps)
                if np.isfinite(m_head).any():
                    out["mean_head_roi_max"] = float(np.nanmean(m_head))
            if ok.sum() >= 4:
                out["slope_roi_max_c_per_min"] = float(
                    sp_stats.theilslopes(m_abs[ok], c[ok] / 60.0)[0])
            if ok.sum() >= 6 and np.ptp(cov[ok]) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    rho, p = sp_stats.spearmanr(cov[ok], m_abs[ok])
                out["tracking_bias_rho"], out["tracking_bias_p"] = float(rho), float(p)
                out["tracking_bias_n"] = int(ok.sum())

            if "speed_px_s" in roi_df.columns:
                sp = roi_df["speed_px_s"].to_numpy()[post]
                mv = roi_df["moving"].to_numpy()[post]
                if np.isfinite(sp).any():
                    out["mean_speed_px_s"] = float(np.nanmean(sp))
                    out["pct_time_moving"] = float(100 * np.nanmean(mv))

    if tcore_df is not None and len(tcore_df) > 0:
        tcore = tcore_df["tcore_estimated"].to_numpy()
        out["peak_tcore_estimated"] = float(np.nanmax(tcore))
        out["mean_tcore_estimated"] = float(np.nanmean(tcore))
    else:
        out["peak_tcore_estimated"] = out["mean_tcore_estimated"] = np.nan
    return out


def bias_flagged(m: dict) -> bool:
    """True when a significant, large link between visibility and temperature was found."""
    p, rho = m.get("tracking_bias_p"), m.get("tracking_bias_rho")
    return bool(p is not None and rho is not None and np.isfinite(p) and np.isfinite(rho)
                and p < 0.05 and abs(rho) >= 0.5)


def bias_sentence(m: dict) -> str:
    """Plain-language statement of the visibility–temperature check for one animal."""
    rho, p, n = m.get("tracking_bias_rho"), m.get("tracking_bias_p"), m.get("tracking_bias_n")
    if rho is None or not np.isfinite(rho):
        return "Visibility check not possible (too few 30-s windows)."
    ptxt = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    if bias_flagged(m):
        direction = "lower" if rho > 0 else "higher"
        return (f"Why flagged: in the {n} thirty-second windows after settling, windows in which the "
                f"body was tracked for a smaller share of the time gave {direction} temperatures "
                f"(Spearman rho = {rho:.2f}, {ptxt}). This is a different question from overall "
                "coverage: even when most frames are tracked, partly visible or fast-moving stretches "
                "can bias the mean. The values are not wrong, but the mean may under-represent the "
                "animal's true level; inspect the scatter and consider the resting-only metric.")
    return (f"Visibility check passed: how much of a 30-s window was tracked is not clearly related "
            f"to its temperature (rho = {rho:.2f}, {ptxt}, {n} windows).")


def qc_flag(m: dict, min_coverage: float) -> str:
    """Plain-language QC verdict for one animal (see ``extract_per_file_metrics``)."""
    cov = m.get("coverage_pct")
    if cov is None or not np.isfinite(cov):
        return "no data"
    if cov < 100 * min_coverage:
        return f"EXCLUDED: body tracked in only {cov:.0f}% of frames (< {100 * min_coverage:.0f}%)"
    if bias_flagged(m):
        p, rho = m["tracking_bias_p"], m["tracking_bias_rho"]
        return ("CHECK: temperature depends on how visible the animal was "
                f"(rho = {rho:.2f}, {'p < 0.001' if p < 0.001 else f'p = {p:.3f}'}); "
                "see the note under the animal overview")
    return "ok"


# ── Paired (same-animal, several conditions) design ──────────────────────────

def animal_key(name: str) -> str:
    """Case-insensitive, filename-safe key used to match one animal across groups."""
    return naming.safe(naming.animal_from_stem(name)).lower()


def collapse_to_animals(per_file_df: pd.DataFrame, group_names: list):
    """One row per animal × group (repeated recordings averaged), restricted to
    animals present in every group.

    Returns (animal_df, matching_df).  matching_df lists every animal with its
    recording count per group and whether it enters the paired analysis.
    """
    df = per_file_df.copy()
    df["animal"] = df["file"].map(naming.animal_from_stem)
    df["_key"] = df["animal"].map(lambda a: naming.safe(a).lower())
    metric_cols = [m for m in METRICS if m in df.columns]
    agg = (df.groupby(["group", "_key"], sort=False)
             .agg(animal=("animal", "first"), n_recordings=("file", "count"),
                  **{m: (m, "mean") for m in metric_cols})
             .reset_index())
    present = agg.groupby("_key")["group"].apply(set)
    complete = {k for k, gs in present.items() if gs >= set(group_names)}

    match_rows = []
    for key, sub in agg.groupby("_key", sort=False):
        row = {"animal": sub["animal"].iloc[0]}
        for g in group_names:
            m = sub[sub["group"] == g]
            row[f"recordings_in_{g}"] = int(m["n_recordings"].iloc[0]) if len(m) else 0
        row["in_all_groups"] = key in complete
        row["included_in_paired_analysis"] = key in complete
        match_rows.append(row)
    matching_df = pd.DataFrame(match_rows)

    animal_df = agg[agg["_key"].isin(complete)].drop(columns="_key").copy()
    animal_df.insert(1, "file", animal_df["animal"])
    # keep group order stable
    animal_df["group"] = pd.Categorical(animal_df["group"], group_names, ordered=True)
    animal_df = animal_df.sort_values(["group", "animal"]).reset_index(drop=True)
    animal_df["group"] = animal_df["group"].astype(str)
    return animal_df, matching_df


def holm_adjust(pvals) -> list:
    """Holm step-down adjusted p-values (same order as input).

    Controls family-wise error like Bonferroni but is uniformly more powerful.
    """
    p = np.asarray([_f(x) for x in pvals], dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj.tolist()


def _wilcoxon_can_detect(n: int) -> bool:
    """Smallest two-sided exact Wilcoxon p is 2/2^n; below n = 6 it exceeds 0.05."""
    return n >= 6


def _mwu_can_detect(n1: int, n2: int) -> bool:
    """Smallest two-sided exact Mann-Whitney p is 2/C(n1+n2, n1)."""
    from math import comb
    return 2.0 / comb(n1 + n2, n1) < 0.05


def _paired_compare(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired comparison of x vs y (same animals, same order).

    Paired t-test unless the paired differences fail Shapiro–Wilk (n ≥ 3) *and*
    a Wilcoxon signed-rank test could actually reach p < 0.05 at this n
    (needs n ≥ 6); otherwise the rank test would be unable to detect anything
    and the t-test is used (flagged in ``basis``).
    """
    n = len(x)
    d = x - y
    zero_var = n >= 2 and np.ptp(d) == 0
    verdict = "untestable"
    if n >= 3 and not zero_var:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            verdict = "normal" if sp_stats.shapiro(d)[1] >= 0.05 else "non-normal"
    use_t = (not zero_var) and n >= 2 and (
        verdict in ("normal", "untestable")
        or (verdict == "non-normal" and not _wilcoxon_can_detect(n)))
    lo = hi = np.nan
    if use_t:
        stat, p = sp_stats.ttest_rel(x, y)
        sd = np.std(d, ddof=1)
        eff, et = (np.mean(d) / sd if sd > 0 else np.nan), "Cohen's d_z"
        se = sd / np.sqrt(n)
        tcrit = sp_stats.t.ppf(0.975, n - 1)
        lo, hi = float(np.mean(d) - tcrit * se), float(np.mean(d) + tcrit * se)
        test, df = "Paired t-test", n - 1
        basis = {
            "normal": "paired differences passed Shapiro-Wilk (p >= 0.05) -> parametric",
            "untestable": "n < 3 pairs (normality untestable) -> parametric; interpret "
                          "with caution",
            "non-normal": "differences failed Shapiro-Wilk but n < 6 pairs, where a "
                          "signed-rank test cannot reach p < 0.05 -> paired t-test; "
                          "interpret with caution",
        }[verdict]
    else:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stat, p = sp_stats.wilcoxon(x, y)
        except ValueError:                       # every difference is zero
            stat, p = np.nan, 1.0
        nz = d[d != 0]
        if len(nz):
            ranks = sp_stats.rankdata(np.abs(nz))
            eff = float((ranks[nz > 0].sum() - ranks[nz < 0].sum()) / ranks.sum())
        else:
            eff = np.nan
        et, test, df = "Rank-biserial r", "Wilcoxon signed-rank", np.nan
        basis = ("paired differences failed Shapiro-Wilk -> non-parametric"
                 if verdict == "non-normal" else
                 "identical paired differences (zero variance) -> non-parametric")
    return {"test_name": test, "statistic": _f(stat), "df": _f(df), "p": float(p),
            "effect": _f(eff), "effect_type": et, "diff": float(np.mean(d)),
            "ci_lo": lo, "ci_hi": hi, "basis": basis}


def _rm_anova(wide: pd.DataFrame) -> dict:
    """One-way repeated-measures ANOVA with Greenhouse–Geisser correction.

    ``wide``: rows = animals (complete), columns = conditions.
    """
    Y = wide.to_numpy(dtype=float)
    n, k = Y.shape
    gm = Y.mean()
    ss_cond = n * np.sum((Y.mean(axis=0) - gm) ** 2)
    ss_subj = k * np.sum((Y.mean(axis=1) - gm) ** 2)
    ss_err = np.sum((Y - gm) ** 2) - ss_cond - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    if ss_err <= 0 or df2 <= 0:
        return {"F": np.nan, "df1": df1, "df2": df2, "p": np.nan, "eps": np.nan,
                "eta_p2": np.nan}
    F = (ss_cond / df1) / (ss_err / df2)
    # Greenhouse–Geisser epsilon from the covariance of the conditions
    S = np.cov(Y, rowvar=False)
    Sc = S - S.mean(axis=0, keepdims=True) - S.mean(axis=1, keepdims=True) + S.mean()
    den = (k - 1) * np.sum(Sc ** 2)
    eps = float(np.clip(np.sum(np.diag(Sc)) ** 2 / den, 1.0 / (k - 1), 1.0)) if den > 0 else 1.0
    p = float(sp_stats.f.sf(F, df1 * eps, df2 * eps))
    return {"F": float(F), "df1": df1 * eps, "df2": df2 * eps, "p": p, "eps": eps,
            "eta_p2": float(ss_cond / (ss_cond + ss_err))}


def _omnibus_paired(wide: pd.DataFrame) -> dict:
    """Omnibus test over 3+ repeated conditions (complete cases).

    RM-ANOVA (Greenhouse–Geisser) when each condition passes Shapiro–Wilk or n is
    too small to judge (n < 5); Friedman otherwise.
    """
    n, k = wide.shape
    verdict = "untestable"
    if n >= 5:
        verdict = "normal"
        for col in wide.columns:
            v = wide[col].to_numpy()
            if np.ptp(v) > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    if sp_stats.shapiro(v)[1] < 0.05:
                        verdict = "non-normal"
                        break
    if verdict == "non-normal":
        stat, p = sp_stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
        return {"test_name": "Friedman", "statistic": _f(stat), "df": k - 1,
                "p": float(p), "effect": _f(stat / (n * (k - 1))),
                "effect_type": "Kendall's W",
                "basis": "a condition failed Shapiro-Wilk (n >= 5) -> non-parametric"}
    r = _rm_anova(wide)
    return {"test_name": "Repeated-measures ANOVA (Greenhouse-Geisser)",
            "statistic": _f(r["F"]), "df": f"{r['df1']:.2f}, {r['df2']:.2f}",
            "p": r["p"], "effect": _f(r["eta_p2"]), "effect_type": "Partial eta-squared",
            "basis": ("all conditions passed Shapiro-Wilk -> parametric"
                      if verdict == "normal" else
                      "n < 5 animals (normality untestable) -> parametric; interpret "
                      "with caution")}


def _paired_group_tests(label: str, wide: pd.DataFrame, group_names: list):
    """Paired tests across conditions for one metric / time point.

    ``wide``: rows = animals, columns = groups (complete cases only).
    2 groups → paired t / Wilcoxon.  3+ → omnibus (RM-ANOVA or Friedman), then
    pairwise paired tests with Holm correction when the omnibus p < 0.05.
    Row keys match the unpaired tables so plots and reports are shared.
    """
    n = len(wide)
    rows, posthoc = [], []
    if n < 2 or len(group_names) < 2:
        return rows, posthoc

    def pair_row(g1, g2, res, test_suffix="", p_corr=None, correction=None):
        a, b = _describe(wide[g1].to_numpy()), _describe(wide[g2].to_numpy())
        pc = res["p"] if p_corr is None else p_corr
        return {
            "metric": label, "group_A": g1, "group_B": g2,
            "test_name": res["test_name"] + test_suffix,
            "test_selection_basis": res["basis"], "alternative": "two-sided",
            "statistic": res["statistic"], "df": res["df"],
            "p_value": res["p"], "p_text": p_text(pc), "p_corrected": pc,
            "correction": correction or "none (single comparison)",
            "significance": _significance_label(pc),
            "effect_size": res["effect"], "effect_size_type": res["effect_type"],
            "mean_diff_A_minus_B": res["diff"],
            "mean_diff_ci95_low": res["ci_lo"], "mean_diff_ci95_high": res["ci_hi"],
            "n_A": n, "n_B": n,
            "mean_A": a["mean"], "sd_A": a["sd"], "sem_A": a["sem"], "median_A": a["median"],
            "mean_B": b["mean"], "sd_B": b["sd"], "sem_B": b["sem"], "median_B": b["median"],
        }

    if len(group_names) == 2:
        g1, g2 = group_names
        rows.append(pair_row(g1, g2, _paired_compare(wide[g1].to_numpy(),
                                                     wide[g2].to_numpy())))
        return rows, posthoc

    k = len(group_names)
    if n < 3:
        return rows, posthoc
    om = _omnibus_paired(wide[group_names])
    rows.append({
        "metric": label, "group_A": " vs ".join(group_names), "group_B": "",
        "test_name": om["test_name"], "test_selection_basis": om["basis"],
        "alternative": "omnibus", "statistic": om["statistic"], "df": om["df"],
        "p_value": om["p"], "p_text": p_text(om["p"]), "p_corrected": om["p"],
        "correction": "none (omnibus)", "significance": _significance_label(om["p"]),
        "effect_size": om["effect"], "effect_size_type": om["effect_type"],
        "mean_diff_A_minus_B": np.nan, "mean_diff_ci95_low": np.nan,
        "mean_diff_ci95_high": np.nan, "n_A": n, "n_B": 0,
        "mean_A": np.nan, "sd_A": np.nan, "sem_A": np.nan, "median_A": np.nan,
        "mean_B": np.nan, "sd_B": np.nan, "sem_B": np.nan, "median_B": np.nan,
    })
    if om["p"] < 0.05:
        pairs = list(itertools.combinations(group_names, 2))
        results = [_paired_compare(wide[a].to_numpy(), wide[b].to_numpy())
                   for a, b in pairs]
        adj = holm_adjust([r["p"] for r in results])
        for (g1, g2), res, pc in zip(pairs, results, adj):
            r = pair_row(g1, g2, res, " (post-hoc)", pc, f"Holm (m = {len(pairs)})")
            for key in ("test_selection_basis", "mean_A", "sd_A", "sem_A", "median_A",
                        "mean_B", "sd_B", "sem_B", "median_B"):
                r.pop(key, None)
            posthoc.append(r)
    return rows, posthoc

# ── Time-point comparison ────────────────────────────────────────────────────

TIMEPOINT_SIGNALS = {
    "roi_max": ("ROI max temperature", "°C"),
    "normalized_roi_max": ("Baseline-normalised ROI max", "Δ°C"),
    "tcore_estimated": ("Estimated T_core", "°C"),
}


def timepoint_label(t_s: float) -> str:
    """'t=5 min' / 't=90 s' — stable label used in tables and plots."""
    if t_s >= 60 and abs(t_s / 60 - round(t_s / 60)) < 1e-9:
        return f"t={int(round(t_s / 60))} min"
    if t_s >= 60:
        return f"t={round(t_s / 60, 2):g} min"
    return f"t={t_s:g} s"


def extract_timepoint_values(roi_df, tcore_df, timepoints_s, signal: str,
                             window_s: float) -> dict:
    """Mean of ``signal`` over [t − w/2, t + w/2] for each time point.

    NaN when the recording ended before the time point or no sample falls in
    the window (so short recordings are excluded, never extrapolated).
    """
    df = tcore_df if signal == "tcore_estimated" else roi_df
    out = {}
    for t in timepoints_s:
        val = np.nan
        if df is not None and len(df) and signal in df.columns:
            ts = df["time_s"].to_numpy()
            if ts[-1] >= t:
                m = (ts >= t - window_s / 2) & (ts <= t + window_s / 2)
                if m.any():
                    val = float(np.nanmean(df[signal].to_numpy()[m]))
        out[float(t)] = val
    return out


def run_timepoint_stats(all_group_data: dict, timepoints_s, signal: str,
                        window_s: float, paired: bool = False) -> dict:
    """Between-group tests at each time point + within-group tests across them.

    Returns dict with:
      timepoint_df           long: group, file, animal, timepoint_s, label, value
      timepoint_desc_df      n, mean, SD, SEM, CI, median, IQR per time point × group
      timepoint_test_rows    between-group tests (one per time point)
      timepoint_posthoc_rows between-group pairwise (3+ groups, omnibus p < 0.05)
      timepoint_within_rows  repeated-measures tests across time points per group
    """
    tps = [float(t) for t in timepoints_s]
    rows = []
    for g_name, gd in all_group_data.items():
        for stem, roi_df in gd.roi_dfs.items():
            vals = extract_timepoint_values(
                roi_df, gd.tcore_dfs.get(stem), tps, signal, window_s)
            for t, v in vals.items():
                rows.append({"group": g_name, "file": stem,
                             "timepoint_s": t, "label": timepoint_label(t),
                             "signal": signal, "window_s": window_s, "value": v})
    tp_df = pd.DataFrame(rows, columns=["group", "file", "timepoint_s", "label",
                                        "signal", "window_s", "value"])
    group_names = list(all_group_data.keys())
    if paired and not tp_df.empty:
        # one value per animal × group × time point; only animals in every group
        tp_df["animal"] = tp_df["file"].map(naming.animal_from_stem)
        tp_df["_key"] = tp_df["animal"].map(lambda a: naming.safe(a).lower())
        have = tp_df.groupby("_key")["group"].apply(set)
        keep = {k for k, gs in have.items() if gs >= set(group_names)}
        tp_df = (tp_df[tp_df["_key"].isin(keep)]
                 .groupby(["group", "_key", "timepoint_s", "label", "signal", "window_s"],
                          sort=False)
                 .agg(animal=("animal", "first"), n_recordings=("file", "count"),
                      value=("value", "mean"))
                 .reset_index().drop(columns="_key"))
        tp_df.insert(1, "file", tp_df["animal"])
    result = {"timepoint_df": tp_df, "timepoint_desc_df": pd.DataFrame(),
              "timepoint_test_rows": [], "timepoint_posthoc_rows": [],
              "timepoint_within_rows": []}
    if tp_df.empty:
        return result

    unit = TIMEPOINT_SIGNALS.get(signal, ("", ""))[1]

    desc = []
    for t in tps:
        for g in group_names:
            arr = tp_df[(tp_df.group == g) & (tp_df.timepoint_s == t)]["value"].dropna().to_numpy()
            desc.append({"timepoint": timepoint_label(t), "timepoint_s": t,
                         "group": g, "unit": unit, **_describe(arr)})
    result["timepoint_desc_df"] = pd.DataFrame(desc)

    # Between groups at each time point
    for t in tps:
        label = timepoint_label(t)
        samples = {g: tp_df[(tp_df.group == g) & (tp_df.timepoint_s == t)]["value"]
                   .dropna().to_numpy() for g in group_names}
        if paired:
            wide = (tp_df[tp_df.timepoint_s == t]
                    .pivot(index="file", columns="group", values="value")
                    .reindex(columns=group_names).dropna())
            rows_, ph = _paired_group_tests(label, wide, group_names)
            result["timepoint_test_rows"].extend(rows_)
            result["timepoint_posthoc_rows"].extend(ph)
            continue
        if len(group_names) < 2 or any(len(v) == 0 for v in samples.values()):
            continue
        if len(group_names) == 2:
            result["timepoint_test_rows"].extend(
                GroupStatsTester._two_group_test(label, samples))
        else:
            rows_, ph = GroupStatsTester._multi_group_test(label, samples)
            result["timepoint_test_rows"].extend(rows_)
            result["timepoint_posthoc_rows"].extend(ph)
    GroupStatsTester._holm_across_metrics(result["timepoint_test_rows"])
    for r in result["timepoint_test_rows"]:
        r["timepoint"] = r["metric"]
        r["metric"] = f"{signal} @ {r['metric']}"
    for r in result["timepoint_posthoc_rows"]:
        r["timepoint"] = r["metric"]
        r["metric"] = f"{signal} @ {r['metric']}"

    # Within each group across time points (paired by animal recording)
    if len(tps) >= 2:
        for g in group_names:
            wide = (tp_df[tp_df.group == g]
                    .pivot(index="file", columns="timepoint_s", values="value")
                    .reindex(columns=tps))
            if len(wide.dropna(how="all")) < 2:
                continue
            result["timepoint_within_rows"].extend(
                _within_group_tests(g, wide, tps))
    return result


def _within_group_tests(group: str, wide: pd.DataFrame, tps: list) -> list:
    """Repeated-measures tests across time points within one group.

    Omnibus (3 time points, recordings with all of them): RM-ANOVA or Friedman via
    ``_omnibus_paired``.  Pairwise: ``_paired_compare`` on recordings that have
    both time points; Holm-corrected across the pairs.
    """
    rows = []
    full = wide.dropna()
    n = len(full)
    if len(tps) == 3 and n >= 3:
        om = _omnibus_paired(full)
        rows.append({
            "group": group, "comparison": "all time points",
            "test_name": om["test_name"], "statistic": om["statistic"], "df": om["df"],
            "p_value": om["p"], "p_corrected": om["p"], "p_text": p_text(om["p"]),
            "correction": "none (omnibus)", "significance": _significance_label(om["p"]),
            "effect_size": om["effect"], "effect_size_type": om["effect_type"],
            "n_pairs": n, "mean_diff": np.nan,
            "mean_diff_ci95_low": np.nan, "mean_diff_ci95_high": np.nan,
        })

    pairs = list(itertools.combinations(tps, 2))
    results = []
    for a, b in pairs:
        both = wide[[a, b]].dropna()     # complete cases for this pair only
        if len(both) < 2:
            continue
        results.append((a, b, len(both),
                        _paired_compare(both[a].to_numpy(), both[b].to_numpy())))
    adj = holm_adjust([r[3]["p"] for r in results]) if results else []
    for (a, b, npair, res), pc in zip(results, adj):
        rows.append({
            "group": group,
            "comparison": f"{timepoint_label(a)} vs {timepoint_label(b)}",
            "test_name": res["test_name"] + (" (post-hoc)" if len(tps) == 3 else ""),
            "statistic": res["statistic"], "df": res["df"], "p_value": res["p"],
            "p_corrected": pc, "p_text": p_text(pc),
            "correction": f"Holm (m = {len(results)})" if len(results) > 1 else "none",
            "significance": _significance_label(pc),
            "effect_size": res["effect"], "effect_size_type": res["effect_type"],
            "n_pairs": npair, "mean_diff": res["diff"],
            "mean_diff_ci95_low": res["ci_lo"], "mean_diff_ci95_high": res["ci_hi"],
        })
    return rows

# ── Descriptive statistics & assumption checks ───────────────────────────────

def _describe(arr: np.ndarray) -> dict:
    n = len(arr)
    mean = float(np.mean(arr)) if n else np.nan
    sd = float(np.std(arr, ddof=1)) if n > 1 else np.nan
    sem = float(sp_stats.sem(arr)) if n > 1 else np.nan
    if n > 1 and not np.isnan(sem):
        tcrit = sp_stats.t.ppf(0.975, n - 1)
        lo, hi = mean - tcrit * sem, mean + tcrit * sem
    else:
        lo = hi = np.nan
    return {
        "n": n, "mean": mean, "sd": sd, "sem": sem,
        "ci95_low": lo, "ci95_high": hi,
        "median": float(np.median(arr)) if n else np.nan,
        "q1": float(np.percentile(arr, 25)) if n else np.nan,
        "q3": float(np.percentile(arr, 75)) if n else np.nan,
        "min": float(np.min(arr)) if n else np.nan,
        "max": float(np.max(arr)) if n else np.nan,
    }


def descriptive_table(per_file_df: pd.DataFrame, group_names: list) -> pd.DataFrame:
    """One row per metric × group: n, mean, SD, SEM, 95 % CI, median, IQR, range."""
    rows = []
    for metric in METRICS:
        if metric not in per_file_df.columns:
            continue
        for g in group_names:
            arr = per_file_df.loc[per_file_df["group"] == g, metric].dropna().to_numpy()
            rows.append({"metric": metric, "group": g,
                         "unit": METRIC_INFO[metric][1], **_describe(arr)})
    return pd.DataFrame(rows)


def normality_table(per_file_df: pd.DataFrame, group_names: list) -> pd.DataFrame:
    """Shapiro–Wilk W and p per metric × group (n ≥ 3 required)."""
    rows = []
    for metric in METRICS:
        if metric not in per_file_df.columns:
            continue
        for g in group_names:
            arr = per_file_df.loc[per_file_df["group"] == g, metric].dropna().to_numpy()
            w = p = np.nan
            note = ""
            if len(arr) < GroupStatsTester.NORMALITY_MIN_N:
                note = f"not tested (n < {GroupStatsTester.NORMALITY_MIN_N})"
            elif np.ptp(arr) == 0:
                note = "not tested (zero variance)"
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    w, p = sp_stats.shapiro(arr)
                note = "normal (p ≥ 0.05)" if p >= 0.05 else "non-normal (p < 0.05)"
            rows.append({"metric": metric, "group": g, "n": len(arr),
                         "shapiro_W": _f(w), "shapiro_p": _f(p), "interpretation": note})
    return pd.DataFrame(rows)


# ── Main statistical testing engine ──────────────────────────────────────────

class GroupStatsTester:
    """Run appropriate statistical tests comparing groups on scalar per-file metrics."""

    NORMALITY_MIN_N = 3  # minimum n per group to run Shapiro-Wilk

    @classmethod
    def run_all(cls, all_group_stats: dict, paired: bool = False) -> dict:
        """
        Parameters
        ----------
        all_group_stats : {group_name: {..., per_file_metrics: {file: {metric: value}}}}

        Returns
        -------
        dict with keys:
          'test_rows'      : list of dicts (one per metric × comparison)
          'posthoc_rows'   : list of dicts (pairwise if 3+ groups)
          'per_file_df'    : pd.DataFrame, one row per animal/file
          'descriptive_df' : n, mean, SD, SEM, 95 % CI, median, IQR, range per metric × group
          'normality_df'   : Shapiro-Wilk per metric × group
        """
        all_rows = []
        for group_name, gdata in all_group_stats.items():
            for file_name, metrics in gdata.get("per_file_metrics", {}).items():
                row = {"group": group_name, "file": file_name}
                row.update(metrics)
                all_rows.append(row)
        per_file_df = pd.DataFrame(all_rows)

        group_names = list(all_group_stats.keys())
        n_groups = len(group_names)

        # Paired design: same animals in several groups (conditions).  Repeated
        # recordings are averaged per animal and only animals present in every
        # group are analysed; tests are paired.
        recording_df, matching_df = per_file_df, None
        if paired and len(per_file_df):
            per_file_df, matching_df = collapse_to_animals(per_file_df, group_names)

        test_rows = []
        posthoc_rows = []

        testable_metrics = [m for m in METRICS
                            if m in per_file_df.columns
                            and per_file_df[m].notna().any()]

        for metric in testable_metrics:
            if paired:
                wide = (per_file_df.pivot(index="file", columns="group", values=metric)
                        .reindex(columns=group_names).dropna())
                rows_, ph = _paired_group_tests(metric, wide, group_names)
                test_rows.extend(rows_)
                posthoc_rows.extend(ph)
                continue
            group_samples = {
                g: per_file_df.loc[per_file_df["group"] == g, metric]
                              .dropna().to_numpy()
                for g in group_names
            }
            if any(len(v) == 0 for v in group_samples.values()):
                continue

            if n_groups == 2:
                test_rows.extend(cls._two_group_test(metric, group_samples))
            elif n_groups >= 3:
                rows, ph_rows = cls._multi_group_test(metric, group_samples)
                test_rows.extend(rows)
                posthoc_rows.extend(ph_rows)

        cls._holm_across_metrics(test_rows)

        if len(per_file_df):
            descriptive_df = descriptive_table(per_file_df, group_names)
            normality_df = normality_table(per_file_df, group_names)
        else:
            descriptive_df = normality_df = pd.DataFrame()

        return {
            "test_rows": test_rows,
            "posthoc_rows": posthoc_rows,
            "per_file_df": per_file_df,
            "descriptive_df": descriptive_df,
            "normality_df": normality_df,
            "paired": paired,
            "recording_df": recording_df,
            "matching_df": matching_df,
        }

    @staticmethod
    def _holm_across_metrics(test_rows: list) -> None:
        """Add Holm-adjusted p across the metrics tested (family = this analysis run)."""
        idx = [i for i, r in enumerate(test_rows) if not np.isnan(_f(r["p_value"]))]
        order = sorted(idx, key=lambda i: test_rows[i]["p_value"])
        m = len(order)
        running = 0.0
        for rank, i in enumerate(order):
            adj = min(1.0, (m - rank) * test_rows[i]["p_value"])
            running = max(running, adj)
            test_rows[i]["p_holm_across_metrics"] = running
        for r in test_rows:
            r.setdefault("p_holm_across_metrics", np.nan)

    @classmethod
    def _normality_verdict(cls, samples: dict) -> str:
        """'normal' | 'non-normal' | 'untestable' (no group has n >= 3)."""
        tested = False
        for arr in samples.values():
            if len(arr) >= cls.NORMALITY_MIN_N and np.ptp(arr) > 0:
                tested = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p = sp_stats.shapiro(arr)
                if p < 0.05:
                    return "non-normal"
        return "normal" if tested else "untestable"

    @classmethod
    def _is_normal(cls, samples: dict) -> bool:
        return cls._normality_verdict(samples) == "normal"

    @classmethod
    def _choose_parametric(cls, samples: dict, design: str):
        """Return (use_parametric, basis).

        Parametric (Welch / ANOVA) is the default.  A rank test is used only when
        normality was tested and failed *and* the rank test can actually reach
        p < 0.05 at these sample sizes (with 3 vs 3 animals a Mann-Whitney test
        cannot go below p = 0.10, so it would never detect any effect).
        """
        sizes = [len(a) for a in samples.values()]
        if min(sizes) < 2:
            return False, "a group has n = 1 (no variance estimate) -> rank test"
        verdict = cls._normality_verdict(samples)
        if verdict == "normal":
            return True, "all groups passed Shapiro-Wilk (p >= 0.05) -> parametric"
        if verdict == "untestable":
            return True, (f"no group has n >= {cls.NORMALITY_MIN_N} (normality "
                          "untestable) -> parametric; interpret with caution")
        detectable = (_mwu_can_detect(*sizes) if design == "two"
                      else (sum(sizes) >= 9 and min(sizes) >= 3))
        if detectable:
            return False, "at least one group failed Shapiro-Wilk (p < 0.05) -> non-parametric"
        return True, ("at least one group failed Shapiro-Wilk but n is too small for a "
                      "rank test to reach p < 0.05 -> parametric; interpret with caution")

    @classmethod
    def _two_group_test(cls, metric: str, samples: dict) -> list:
        group_names = list(samples.keys())
        g1_name, g2_name = group_names[0], group_names[1]
        g1, g2 = samples[g1_name], samples[g2_name]

        normal, basis = cls._choose_parametric(samples, "two")
        df = diff = ci_lo = ci_hi = np.nan

        if normal:
            stat, p = sp_stats.ttest_ind(g1, g2, equal_var=False)
            effect = _cohens_d(g1, g2)
            effect_type = "Cohen's d"
            test_name = "Welch t-test"
            df, ci_lo, ci_hi = _welch_df_and_ci(g1, g2)
            diff = float(np.mean(g1) - np.mean(g2))
        else:
            stat, p = sp_stats.mannwhitneyu(g1, g2, alternative='two-sided')
            effect = _rank_biserial(g1, g2)
            effect_type = "Rank-biserial r"
            test_name = "Mann-Whitney U"

        s1, s2 = _describe(g1), _describe(g2)
        return [{
            "metric": metric,
            "group_A": g1_name,
            "group_B": g2_name,
            "test_name": test_name,
            "test_selection_basis": basis,
            "alternative": "two-sided",
            "statistic": _f(stat),
            "df": _f(df),
            "p_value": _f(p),
            "p_text": p_text(float(p)),
            "p_corrected": _f(p),
            "correction": "none (single comparison)",
            "significance": _significance_label(p),
            "effect_size": _f(effect),
            "effect_size_type": effect_type,
            "mean_diff_A_minus_B": _f(diff),
            "mean_diff_ci95_low": _f(ci_lo),
            "mean_diff_ci95_high": _f(ci_hi),
            "n_A": len(g1),
            "n_B": len(g2),
            "mean_A": s1["mean"], "sd_A": s1["sd"], "sem_A": s1["sem"],
            "median_A": s1["median"],
            "mean_B": s2["mean"], "sd_B": s2["sd"], "sem_B": s2["sem"],
            "median_B": s2["median"],
        }]

    @classmethod
    def _multi_group_test(cls, metric: str, samples: dict):
        group_names = list(samples.keys())
        arrays = [samples[g] for g in group_names]
        normal, basis = cls._choose_parametric(samples, "multi")
        k = len(arrays)
        n_total = sum(len(a) for a in arrays)

        if normal:
            stat, p = sp_stats.f_oneway(*arrays)
            effect = _eta_squared(stat, arrays)
            effect_type = "Eta-squared"
            test_name = "One-way ANOVA"
            df = f"{k - 1}, {n_total - k}"
        else:
            stat, p = sp_stats.kruskal(*arrays)
            effect = float(stat) / (n_total - 1) if n_total > 1 else np.nan
            effect_type = "Epsilon-squared"
            test_name = "Kruskal-Wallis"
            df = f"{k - 1}"

        test_row = {
            "metric": metric,
            "group_A": " vs ".join(group_names),
            "group_B": "",
            "test_name": test_name,
            "test_selection_basis": basis,
            "alternative": "omnibus",
            "statistic": _f(stat),
            "df": df,
            "p_value": _f(p),
            "p_text": p_text(float(p)),
            "p_corrected": _f(p),
            "correction": "none (omnibus)",
            "significance": _significance_label(p),
            "effect_size": _f(effect),
            "effect_size_type": effect_type,
            "mean_diff_A_minus_B": np.nan,
            "mean_diff_ci95_low": np.nan,
            "mean_diff_ci95_high": np.nan,
            "n_A": n_total,
            "n_B": 0,
            "mean_A": np.nan, "sd_A": np.nan, "sem_A": np.nan, "median_A": np.nan,
            "mean_B": np.nan, "sd_B": np.nan, "sem_B": np.nan, "median_B": np.nan,
        }

        posthoc_rows = []
        if p < 0.05:
            posthoc_rows = cls._posthoc(metric, samples, normal)

        return [test_row], posthoc_rows

    @classmethod
    def _posthoc(cls, metric: str, samples: dict, normal: bool) -> list:
        """Pairwise post-hoc tests after a significant omnibus test.

        After ANOVA: Tukey–Kramer HSD (handles unequal n; less conservative than
        Bonferroni).  After Kruskal–Wallis: Dunn's test on the pooled ranks with
        Holm correction.
        """
        names = list(samples.keys())
        pairs = list(itertools.combinations(range(len(names)), 2))
        rows = []

        if normal:
            res = sp_stats.tukey_hsd(*[samples[g] for g in names])
            test_name, correction = "Tukey HSD (post-hoc)", "Tukey family-wise"
            stats_p = [(float(res.statistic[i, j]), float(res.pvalue[i, j]),
                        float(res.pvalue[i, j])) for i, j in pairs]
        else:
            arrays = [samples[g] for g in names]
            allv = np.concatenate(arrays)
            N = len(allv)
            ranks = sp_stats.rankdata(allv)
            _, counts = np.unique(allv, return_counts=True)
            tie = np.sum(counts ** 3 - counts) / (12.0 * (N - 1))
            S2 = N * (N + 1) / 12.0 - tie
            bounds = np.cumsum([0] + [len(a) for a in arrays])
            rbar = [ranks[bounds[i]:bounds[i + 1]].mean() for i in range(len(arrays))]
            raw = []
            for i, j in pairs:
                se = np.sqrt(S2 * (1 / len(arrays[i]) + 1 / len(arrays[j])))
                z = (rbar[i] - rbar[j]) / se if se > 0 else 0.0
                raw.append((z, 2 * sp_stats.norm.sf(abs(z))))
            adj = holm_adjust([r[1] for r in raw])
            test_name, correction = "Dunn (post-hoc)", f"Holm (m = {len(pairs)})"
            stats_p = [(z, p, a) for (z, p), a in zip(raw, adj)]

        for (i, j), (stat, p_raw, p_corr) in zip(pairs, stats_p):
            g1, g2 = samples[names[i]], samples[names[j]]
            if normal:
                effect, effect_type = _cohens_d(g1, g2), "Cohen's d"
                diff = float(np.mean(g1) - np.mean(g2))
            else:
                effect, effect_type = _rank_biserial(g1, g2), "Rank-biserial r"
                diff = np.nan
            rows.append({
                "metric": metric,
                "group_A": names[i],
                "group_B": names[j],
                "test_name": test_name,
                "alternative": "two-sided",
                "statistic": _f(stat),
                "df": np.nan,
                "p_value": _f(p_raw),
                "p_corrected": _f(p_corr),
                "p_text": p_text(float(p_corr)),
                "correction": correction,
                "significance": _significance_label(p_corr),
                "effect_size": _f(effect),
                "effect_size_type": effect_type,
                "mean_diff_A_minus_B": _f(diff),
                "mean_diff_ci95_low": np.nan,
                "mean_diff_ci95_high": np.nan,
                "n_A": len(g1),
                "n_B": len(g2),
            })
        return rows