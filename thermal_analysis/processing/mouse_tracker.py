"""
Body-mask mouse tracking for Glöd (Qt-free; NumPy/SciPy only).

Why this exists
---------------
The "hottest pixel" and fixed-threshold blob trackers fail in three ways on real
MI48 recordings of freely moving mice:

1. On the body's flat temperature plateau the argmax is decided by sensor noise,
   so the tracked position hops across the whole animal.
2. When the mouse leaves the field of view (only a tail or a body sliver is
   left) the "hottest object" is the tail / bedding, which reads several °C
   colder.  This appears as sharp dips although nothing physiological happened.
3. Group statistics were computed in *frames*, assuming a nominal frame rate,
   while the real rate (~6 Hz on the test rig) differs.

Approach
--------
Per frame: adaptive (Otsu) threshold, morphological opening to remove the thin
tail, largest component = body.  Temperature is read only from the eroded body
core.  Frames that are not a trustworthy view of the body (body mostly out of
frame, mask merged with warm bedding, no contrast, statistical outlier) are
flagged invalid and become NaN; they are never allowed to pull the trace down.
Every temporal window is defined in seconds and converted with the *measured*
frame rate.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

# Tunables (seconds / °C / fractions) — deliberately not exposed in the UI.
SEG_SIGMA = 1.0            # px, smoothing used for segmentation and max reading
THR_BAND_C = 1.0           # per-frame Otsu threshold may deviate ±this from the recording median
MIN_CONTRAST_C = 3.0       # body mean must exceed background by this much
MIN_AREA_PX = 60           # absolute lower bound for a body mask
AREA_REF_WINDOW_S = 20.0   # window for the reference body area (rolling median)
AREA_LOW_FRAC = 0.55       # body smaller than this × reference → mostly out of view
AREA_HIGH_FRAC = 1.6       # larger than this × reference → merged with bedding / glitch
HOT_BAND_C = 2.0           # "warm body" = within this of the recording's typical body temperature
MAX_EDGE_FRAC = 0.35      # fraction of mask pixels on the image border
HAMPEL_WINDOW_S = 3.0
HAMPEL_NSIGMA = 4.0
HAMPEL_MIN_MAD_C = 0.15
MAX_GAP_S = 2.0            # gaps up to this long are bridged by interpolation


PAUSE_GAP_S = 5.0          # a break between consecutive frames longer than this is a pause


def find_pauses(time_s: np.ndarray, min_gap_s: float = PAUSE_GAP_S) -> list:
    """
    (start_s, end_s) of every break in the recording longer than ``min_gap_s`` — a pause
    in the recorder (or a long camera stall).  Frames on either side belong to the same
    recording; nothing is interpolated or smoothed across the break and it is neither
    "tracked" nor "untracked" time.
    """
    time_s = np.asarray(time_s, dtype=float)
    if len(time_s) < 2:
        return []
    idx = np.flatnonzero(np.diff(time_s) > min_gap_s)
    return [(float(time_s[i]), float(time_s[i + 1])) for i in idx]


def collapse_pauses(time_s: np.ndarray, min_gap_s: float = PAUSE_GAP_S):
    """
    Remove recorder pauses from the time axis so the recording continues from where it
    stopped "as if nothing happened".  The raw file keeps its real timestamps; the analysis
    works on recorded time.

    Returns ``(time_s_continuous, gaps)`` where ``gaps`` has, for every frame, the length of
    the pause immediately before it (0 for ordinary frames).  After a pause the next frame
    follows the last one by one normal frame interval.
    """
    time_s = np.asarray(time_s, dtype=float)
    gaps = np.zeros(len(time_s))
    if len(time_s) < 3:
        return time_s.copy(), gaps
    dt = np.diff(time_s)
    normal = float(np.median(dt[dt > 0])) if (dt > 0).any() else 0.0
    jump = np.flatnonzero(dt > min_gap_s)
    gaps[jump + 1] = dt[jump]
    shift = np.zeros(len(time_s))
    for i in jump:
        shift[i + 1:] += dt[i] - normal
    return time_s - shift, gaps


def estimate_fps(time_s: np.ndarray, default: float = 10.0) -> float:
    """Measured frame rate from timestamps (median inter-frame interval)."""
    if len(time_s) < 3:
        return default
    dt = np.diff(time_s)
    dt = dt[dt > 0]
    if len(dt) == 0:
        return default
    return float(1.0 / np.median(dt))


def otsu_threshold(img: np.ndarray, nbins: int = 64) -> float:
    """Otsu threshold of a 2-D array, histogram-based, pure NumPy."""
    hist, edges = np.histogram(img, bins=nbins)
    centres = (edges[:-1] + edges[1:]) / 2.0
    w1 = np.cumsum(hist)
    w2 = w1[-1] - w1
    m1 = np.cumsum(hist * centres)
    mu1 = m1 / np.maximum(w1, 1)
    mu2 = (m1[-1] - m1) / np.maximum(w2, 1)
    return float(centres[np.argmax(w1 * w2 * (mu1 - mu2) ** 2)])


def _rolling_nan_median(x: np.ndarray, win: int) -> np.ndarray:
    """Centred rolling median that ignores NaN."""
    import pandas as pd
    return (pd.Series(x).rolling(max(3, win | 1), center=True, min_periods=1)
            .median().to_numpy())


def _hampel_outliers(x: np.ndarray, win: int) -> np.ndarray:
    """True where x deviates from its rolling median by > k·MAD (NaN → False)."""
    import pandas as pd
    s = pd.Series(x)
    w = max(3, win | 1)
    med = s.rolling(w, center=True, min_periods=3).median()
    mad = (s - med).abs().rolling(w, center=True, min_periods=3).median() * 1.4826
    thr = np.maximum(HAMPEL_NSIGMA * mad.to_numpy(), HAMPEL_MIN_MAD_C)
    dev = np.abs((s - med).to_numpy())
    return np.nan_to_num(dev > thr, nan=False).astype(bool)


def bridge_gaps(values: np.ndarray, time_s: np.ndarray, max_gap_s: float = MAX_GAP_S) -> np.ndarray:
    """
    Linearly interpolate NaN gaps no longer than ``max_gap_s`` seconds
    (measured on the time axis, so it is frame-rate independent).
    Longer gaps stay NaN — absent data is not invented.
    """
    out = values.astype(np.float64).copy()
    good = ~np.isnan(out)
    if good.sum() < 2:
        return out
    filled = np.interp(time_s, time_s[good], out[good])
    idx = np.flatnonzero(good)
    # gap length = time between the valid frames on either side
    right = np.searchsorted(idx, np.arange(len(out)))
    left = right - 1
    ok_edges = (left >= 0) & (right < len(idx))
    gap = np.full(len(out), np.inf)
    gap[ok_edges] = time_s[idx[right[ok_edges]]] - time_s[idx[left[ok_edges]]]
    bridge = (~good) & (gap <= max_gap_s)
    out[bridge] = filled[bridge]
    return out


def _head_readout(g: np.ndarray, masks: np.ndarray, valid: np.ndarray, thr_ref: float):
    """
    Secondary readout: hottest pixel of the head region (front 30 % of the body axis).

    The body axis is the principal axis of the mask; the rear end is the one the tail
    attaches to (warm pixels connected to the body but removed by the opening), the head
    is the opposite end.  Frames where the tail attachment cannot be seen give NaN.
    Returns (head_max, head_clipped) — clipped = head region touches the image border.
    It is a secondary readout: on 80 x 62 px recordings the head (~10 px) is cooler and noisier than\n    the whole-body reading, but less dependent on how visible the animal is.
    """
    n = len(masks)
    head_max = np.full(n, np.nan)
    clipped = np.zeros(n, dtype=bool)
    for i in np.flatnonzero(valid):
        body = masks[i]
        ys, xs = np.nonzero(body)
        if len(ys) < 30:
            continue
        cy, cx = ys.mean(), xs.mean()
        _, vecs = np.linalg.eigh(np.cov(np.stack([ys - cy, xs - cx])))
        u = vecs[:, 1]
        p = (ys - cy) * u[0] + (xs - cx) * u[1]
        length = np.ptp(p)
        if length < 8:
            continue
        lab, _ = ndi.label(g[i] > thr_ref - 1.0)
        counts = np.bincount(lab[body])
        counts[0] = 0
        resid = (lab == counts.argmax()) & ~ndi.binary_dilation(body)
        ry, rx = np.nonzero(resid)
        if len(ry) < 6:
            continue
        pr = (ry - cy) * u[0] + (rx - cx) * u[1]
        far = pr[np.abs(pr) > 0.35 * length]
        if len(far) < 4:
            continue
        ph = -np.sign(np.median(far)) * p            # increases toward the head
        sel = ph > ph.max() - 0.30 * length
        head = np.zeros_like(body)
        head[ys[sel], xs[sel]] = True
        core = head & ndi.binary_erosion(body)
        reg = core if core.sum() >= 8 else head
        if reg.sum() < 8:
            continue
        head_max[i] = g[i][reg].max()
        clipped[i] = head[0].any() or head[-1].any() or head[:, 0].any() or head[:, -1].any()
    return head_max, clipped


def track_body(stack: np.ndarray, time_s: np.ndarray, fps: float | None = None) -> dict:
    """
    Locate the mouse body in every frame and read its temperature.

    Parameters
    ----------
    stack  : (N, H, W) °C
    time_s : (N,) seconds since the first frame
    fps    : measured frame rate (estimated from ``time_s`` when None)

    Returns a dict of length-N arrays:
      roi_max   hottest body-core temperature (σ=1 smoothed), NaN where invalid
      roi_mean  mean body-core temperature, NaN where invalid
      body_p90  90th percentile of body-core temperature
      max_row, max_col  location of roi_max (-1 where invalid)
      cent_row, cent_col  body centroid (NaN where no mask)
      body_area, edge_frac, background  per-frame diagnostics
      valid     bool — frame is a trustworthy view of the body
      reason    int8 code — 0 ok, 1 no mask, 2 low contrast, 3 area low,
                4 area high, 5 clipped by edge, 6 outlier
      fps       measured frame rate (float)
    """
    n, h, w = stack.shape
    time_s = np.asarray(time_s, dtype=np.float64)
    fps = float(fps) if fps else estimate_fps(time_s)

    g = ndi.gaussian_filter(stack.astype(np.float32, copy=False),
                            sigma=(0, SEG_SIGMA, SEG_SIGMA))
    thr_raw = np.array([otsu_threshold(g[i]) for i in range(n)])
    bg = np.percentile(g.reshape(n, -1), 25, axis=1)
    t0 = float(np.median(thr_raw))
    thr = np.clip(thr_raw, t0 - THR_BAND_C, t0 + THR_BAND_C)

    area = np.zeros(n)
    edge = np.zeros(n)
    roi_max = np.full(n, np.nan)
    roi_mean = np.full(n, np.nan)
    p90 = np.full(n, np.nan)
    max_row = np.full(n, -1, dtype=int)
    max_col = np.full(n, -1, dtype=int)
    cent_r = np.full(n, np.nan)
    cent_c = np.full(n, np.nan)
    reason = np.zeros(n, dtype=np.int8)
    contrast_ok = np.ones(n, dtype=bool)
    masks = np.zeros((n, h, w), dtype=bool)   # body mask per frame (video outlines, pass 2)

    for i in range(n):
        m = ndi.binary_opening(g[i] > thr[i], iterations=2)
        lab, k = ndi.label(m)
        if k == 0:
            reason[i] = 1
            continue
        sizes = ndi.sum(m, lab, range(1, k + 1))
        body = lab == (int(np.argmax(sizes)) + 1)
        masks[i] = body
        a = int(body.sum())
        area[i] = a
        edge[i] = (body[0].sum() + body[-1].sum() + body[:, 0].sum() + body[:, -1].sum()) / a
        cy, cx = ndi.center_of_mass(body)
        cent_r[i], cent_c[i] = cy, cx

        core = ndi.binary_erosion(body, iterations=1)
        if core.sum() < 10:
            core = body
        vals = stack[i][core]
        if vals.mean() - bg[i] < MIN_CONTRAST_C:
            contrast_ok[i] = False
        gv = np.where(core, g[i], -np.inf)
        j = int(np.argmax(gv))
        roi_max[i] = gv.flat[j]
        max_row[i], max_col[i] = divmod(j, w)
        roi_mean[i] = vals.mean()
        p90[i] = np.percentile(vals, 90)

    # ── Second pass: how much of the body is actually warm and in view? ──────
    # A tail or a sliver at the border passes the Otsu mask (area looks normal)
    # but is several °C colder than the trunk.  Reference = the recording's own
    # typical body temperature (upper quartile of per-frame p90).
    t_body = float(np.nanpercentile(p90[reason == 0], 75)) if (reason == 0).any() else np.nan
    hot_area = np.zeros(n)
    if not np.isnan(t_body):
        hot_area = (masks & (g > t_body - HOT_BAND_C)).sum(axis=(1, 2)).astype(float)

    # ── Frame validity (all windows in seconds → frames via measured fps) ────
    ref_win = int(round(AREA_REF_WINDOW_S * fps))
    area_ref = _rolling_nan_median(np.where(area > 0, area, np.nan), ref_win)
    hot_ref = _rolling_nan_median(np.where(hot_area > 0, hot_area, np.nan), ref_win)
    valid = np.ones(n, dtype=bool)
    valid &= reason == 0
    reason[(reason == 0) & ~contrast_ok] = 2
    valid &= contrast_ok
    low = valid & ((area < AREA_LOW_FRAC * area_ref) | (area < MIN_AREA_PX)
                   | (hot_area < AREA_LOW_FRAC * hot_ref))
    reason[low] = 3
    valid &= ~low
    high = valid & (area > AREA_HIGH_FRAC * area_ref)
    reason[high] = 4
    valid &= ~high
    clipped = valid & (edge > MAX_EDGE_FRAC)
    reason[clipped] = 5
    valid &= ~clipped

    roi_max = np.where(valid, roi_max, np.nan)
    out = _hampel_outliers(roi_max, int(round(HAMPEL_WINDOW_S * fps)))
    reason[out & valid] = 6
    valid &= ~out
    for arr in (roi_max, roi_mean, p90):
        arr[~valid] = np.nan
    max_row[~valid] = -1
    max_col[~valid] = -1

    # Secondary head readout (same Hampel screening as the body trace)
    head_max, head_clipped = _head_readout(g, masks, valid, t0)
    head_max[_hampel_outliers(head_max, int(round(HAMPEL_WINDOW_S * fps)))] = np.nan

    return {
        "head_max": head_max, "head_clipped": head_clipped,
        "roi_max": roi_max, "roi_mean": roi_mean, "body_p90": p90,
        "max_row": max_row, "max_col": max_col,
        "cent_row": cent_r, "cent_col": cent_c,
        "body_area": area, "edge_frac": edge, "background": bg,
        "valid": valid, "reason": reason, "fps": fps, "masks": masks,
    }


MOVING_BODY_LENGTHS_PER_S = 0.25   # "moving" = centroid speed above this many body lengths / s
SPEED_LAG_S = 1.0                  # displacement is measured over this interval (suppresses jitter)


def compute_activity(cent_row: np.ndarray, cent_col: np.ndarray, time_s: np.ndarray,
                     valid: np.ndarray, body_area: np.ndarray, fps: float):
    """
    Centroid speed (px/s) from displacement over ~1 s, using tracked frames only.

    Returns (speed_px_s, moving, body_len_px).  ``speed``/``moving`` are NaN where
    either end of the 1-s window was not tracked (unknown activity is never
    counted as rest).  Moving threshold is scale-free: 0.25 body lengths / s,
    with body length = equivalent diameter of the median tracked body area.
    """
    n = len(time_s)
    speed = np.full(n, np.nan)
    moving = np.full(n, np.nan)
    ok_area = body_area[valid & (body_area > 0)]
    body_len = float(np.sqrt(4 * np.median(ok_area) / np.pi)) if len(ok_area) else np.nan
    lag = max(1, int(round(SPEED_LAG_S * fps)))
    if n <= lag or not np.isfinite(body_len):
        return speed, moving, body_len
    r = np.where(valid, cent_row, np.nan)
    c = np.where(valid, cent_col, np.nan)
    dt = time_s[lag:] - time_s[:-lag]
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.hypot(r[lag:] - r[:-lag], c[lag:] - c[:-lag]) / dt
    s[dt > 1.5 * lag / fps] = np.nan          # window straddles a recording hiccup
    speed[lag // 2: lag // 2 + len(s)] = s
    fin = np.isfinite(speed)
    moving[fin] = (speed[fin] > MOVING_BODY_LENGTHS_PER_S * body_len).astype(float)
    return speed, moving, body_len
