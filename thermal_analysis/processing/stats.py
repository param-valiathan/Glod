"""
Statistical tests for group-wise comparison of ROI and T_core metrics.

Test selection:
  2 groups  → Shapiro-Wilk normality → Welch t-test  OR  Mann-Whitney U
  3+ groups → Shapiro-Wilk normality → one-way ANOVA OR  Kruskal-Wallis
                                        + post-hoc Tukey HSD / Dunn's test
"""

import warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


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
    k = len(groups)
    all_vals = np.concatenate(groups)
    ns = [len(g) for g in groups]
    grand_mean = np.mean(all_vals)
    ss_between = sum(n * (np.mean(g) - grand_mean) ** 2 for n, g in zip(ns, groups))
    ss_total = np.sum((all_vals - grand_mean) ** 2)
    return ss_between / ss_total if ss_total != 0 else np.nan


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


# ── Per-file scalar extraction ────────────────────────────────────────────────

METRICS = [
    "baseline_roi_max",
    "peak_roi_max",
    "peak_normalized_roi_max",
    "auc_normalized_roi_max",
    "time_to_peak_s",
    "peak_tcore_estimated",
    "auc_tcore_estimated",
    "mean_tcore_estimated",
]


def extract_per_file_metrics(group_name: str, file_stem: str,
                              roi_df: pd.DataFrame,
                              tcore_df: pd.DataFrame,
                              baseline_end_sec: float = 300.0) -> dict:
    """
    Compute scalar summary statistics from per-frame and T_core DataFrames.
    These scalars are the unit of statistical analysis (n = files per group).
    """
    result = {"group": group_name, "file": file_stem}

    if roi_df is not None and len(roi_df) > 0:
        t = roi_df["time_s"].to_numpy()
        roi_max = roi_df["roi_max"].to_numpy()
        norm_roi = roi_df["normalized_roi_max"].to_numpy()
        smoothed = roi_df.get("roi_max_smoothed", roi_df["roi_max"]).to_numpy()

        baseline_mask = t <= baseline_end_sec
        result["baseline_roi_max"] = float(roi_max[baseline_mask].mean()
                                            if baseline_mask.any() else roi_max.mean())
        result["peak_roi_max"] = float(roi_max.max())
        result["peak_normalized_roi_max"] = float(norm_roi.max())
        result["auc_normalized_roi_max"] = float(np.trapezoid(norm_roi, t))
        peak_idx = int(np.argmax(smoothed))
        result["time_to_peak_s"] = float(t[peak_idx])
    else:
        for k in ["baseline_roi_max", "peak_roi_max", "peak_normalized_roi_max",
                  "auc_normalized_roi_max", "time_to_peak_s"]:
            result[k] = np.nan

    if tcore_df is not None and len(tcore_df) > 0:
        t_tc = tcore_df["time_s"].to_numpy()
        tcore = tcore_df["tcore_estimated"].to_numpy()
        result["peak_tcore_estimated"] = float(tcore.max())
        result["auc_tcore_estimated"] = float(np.trapezoid(tcore, t_tc))
        result["mean_tcore_estimated"] = float(tcore.mean())
    else:
        for k in ["peak_tcore_estimated", "auc_tcore_estimated", "mean_tcore_estimated"]:
            result[k] = np.nan

    return result


# ── Main statistical testing engine ──────────────────────────────────────────

class GroupStatsTester:
    """Run appropriate statistical tests comparing groups on scalar per-file metrics."""

    NORMALITY_MIN_N = 3  # minimum n per group to run Shapiro-Wilk

    @classmethod
    def run_all(cls, all_group_stats: dict) -> dict:
        """
        Parameters
        ----------
        all_group_stats : {group_name: {..., per_file_metrics: {file: {metric: value}}}}

        Returns
        -------
        dict with keys:
          'test_rows'    : list of dicts (one per metric × comparison) → statistical_tests.csv
          'posthoc_rows' : list of dicts (pairwise if 3+ groups) → posthoc_tests.csv
          'per_file_df'  : pd.DataFrame → per_file_metrics.csv
        """
        # Build per-file DataFrame
        all_rows = []
        for group_name, gdata in all_group_stats.items():
            for file_name, metrics in gdata.get("per_file_metrics", {}).items():
                row = {"group": group_name, "file": file_name}
                row.update(metrics)
                all_rows.append(row)
        per_file_df = pd.DataFrame(all_rows)

        group_names = list(all_group_stats.keys())
        n_groups = len(group_names)

        test_rows = []
        posthoc_rows = []

        testable_metrics = [m for m in METRICS
                            if m in per_file_df.columns
                            and per_file_df[m].notna().any()]

        for metric in testable_metrics:
            group_samples = {
                g: per_file_df.loc[per_file_df["group"] == g, metric]
                              .dropna().to_numpy()
                for g in group_names
            }
            # Skip if any group has no data
            if any(len(v) == 0 for v in group_samples.values()):
                continue

            if n_groups == 2:
                rows = cls._two_group_test(metric, group_samples)
                test_rows.extend(rows)
            elif n_groups >= 3:
                rows, ph_rows = cls._multi_group_test(metric, group_samples)
                test_rows.extend(rows)
                posthoc_rows.extend(ph_rows)

        return {
            "test_rows": test_rows,
            "posthoc_rows": posthoc_rows,
            "per_file_df": per_file_df,
        }

    @classmethod
    def _is_normal(cls, samples: dict) -> bool:
        """Return True if all testable groups pass Shapiro-Wilk.
        Returns False (non-parametric) if no group has n >= NORMALITY_MIN_N."""
        tested = False
        for arr in samples.values():
            if len(arr) >= cls.NORMALITY_MIN_N:
                tested = True
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, p = sp_stats.shapiro(arr)
                if p < 0.05:
                    return False
        return tested  # False when all groups are too small → non-parametric

    @classmethod
    def _two_group_test(cls, metric: str, samples: dict) -> list:
        group_names = list(samples.keys())
        g1_name, g2_name = group_names[0], group_names[1]
        g1, g2 = samples[g1_name], samples[g2_name]

        normal = cls._is_normal(samples)

        if normal:
            stat, p = sp_stats.ttest_ind(g1, g2, equal_var=False)
            effect = _cohens_d(g1, g2)
            effect_type = "Cohen's d"
            test_name = "Welch t-test"
        else:
            stat, p = sp_stats.mannwhitneyu(g1, g2, alternative='two-sided')
            effect = _rank_biserial(g1, g2)
            effect_type = "Rank-biserial r"
            test_name = "Mann-Whitney U"

        return [{
            "metric": metric,
            "group_A": g1_name,
            "group_B": g2_name,
            "test_name": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "p_corrected": round(float(p), 4),
            "significance": _significance_label(p),
            "effect_size": round(float(effect), 3) if not np.isnan(effect) else "NA",
            "effect_size_type": effect_type,
            "n_A": len(g1),
            "n_B": len(g2),
            "mean_A": round(float(np.mean(g1)), 4),
            "sem_A": round(float(sp_stats.sem(g1)), 4) if len(g1) > 1 else 0.0,
            "mean_B": round(float(np.mean(g2)), 4),
            "sem_B": round(float(sp_stats.sem(g2)), 4) if len(g2) > 1 else 0.0,
        }]

    @classmethod
    def _multi_group_test(cls, metric: str, samples: dict):
        group_names = list(samples.keys())
        arrays = [samples[g] for g in group_names]
        normal = cls._is_normal(samples)

        if normal:
            stat, p = sp_stats.f_oneway(*arrays)
            effect = _eta_squared(stat, arrays)
            effect_type = "Eta-squared"
            test_name = "One-way ANOVA"
        else:
            stat, p = sp_stats.kruskal(*arrays)
            effect = np.nan
            effect_type = "NA"
            test_name = "Kruskal-Wallis"

        # Group-level summary row
        test_row = {
            "metric": metric,
            "group_A": " vs ".join(group_names),
            "group_B": "",
            "test_name": test_name,
            "statistic": round(float(stat), 4),
            "p_value": round(float(p), 4),
            "p_corrected": round(float(p), 4),
            "significance": _significance_label(p),
            "effect_size": round(float(effect), 3) if not np.isnan(effect) else "NA",
            "effect_size_type": effect_type,
            "n_A": sum(len(a) for a in arrays),
            "n_B": 0,
            "mean_A": "NA", "sem_A": "NA", "mean_B": "NA", "sem_B": "NA",
        }

        posthoc_rows = []
        if p < 0.05:
            posthoc_rows = cls._posthoc(metric, samples, normal)

        return [test_row], posthoc_rows

    @classmethod
    def _posthoc(cls, metric: str, samples: dict, normal: bool) -> list:
        """Pairwise post-hoc tests with Bonferroni correction."""
        import itertools
        pairs = list(itertools.combinations(samples.keys(), 2))
        n_comparisons = len(pairs)
        rows = []

        for g1_name, g2_name in pairs:
            g1, g2 = samples[g1_name], samples[g2_name]
            if normal:
                stat, p_raw = sp_stats.ttest_ind(g1, g2, equal_var=False)
                test_name = "Welch t-test (post-hoc)"
            else:
                stat, p_raw = sp_stats.mannwhitneyu(g1, g2, alternative='two-sided')
                test_name = "Mann-Whitney U (post-hoc)"

            p_corr = min(1.0, p_raw * n_comparisons)  # Bonferroni
            rows.append({
                "metric": metric,
                "group_A": g1_name,
                "group_B": g2_name,
                "test_name": test_name,
                "statistic": round(float(stat), 4),
                "p_value": round(float(p_raw), 4),
                "p_corrected": round(float(p_corr), 4),
                "significance": _significance_label(p_corr),
                "n_A": len(g1),
                "n_B": len(g2),
            })

        return rows
