"""
Core ROI analysis algorithms adapted from legacy scripts.

Sources:
  - MultisiteTempArduino 2.py  (emissivity correction, ROI detection, smoothing)
  - Combine temp apap.py       (group smoothing, rate-of-change, normalization)
  - van der Vinne et al. (2020) Scientific Reports 10:20680  (T_core estimation)
"""

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter as _savgol
from scipy.ndimage import gaussian_filter1d


# ── Emissivity correction ─────────────────────────────────────────────────────

def correct_emissivity(temp_c, eps_true: float, eps_sensor: float = 0.95,
                       t_refl_c: float = 22.0):
    """
    Re-evaluate temperatures read by a camera that assumed ``eps_sensor`` for a surface whose
    emissivity is ``eps_true``, including the radiance reflected from the surroundings.

    The sensor converts total radiance W = ε·σT⁴ + (1−ε)·σT_refl⁴ to a temperature using its
    own EMISSIVITY register (``eps_sensor``; 0.95 on the MI48 cameras unless changed), so

        W = ε_s·T_s⁴ + (1−ε_s)·T_r⁴          (T_s = value in the recording)
        T_true⁴ = (W − (1−ε)·T_r⁴) / ε = (ε_s·T_s⁴ + (ε−ε_s)·T_r⁴) / ε

    Temperatures in kelvin; the band-integrated T⁴ law is an approximation for an 8–14 µm
    sensor, adequate for offsets of a few tenths of a degree.  Identity when ε = ε_s.
    NaN-safe; accepts scalars or arrays.
    """
    arr = np.asarray(temp_c, dtype=np.float64)
    if abs(eps_true - eps_sensor) < 1e-9:
        return arr.copy() if arr.ndim else float(arr)
    ts = arr + 273.15
    tr4 = (t_refl_c + 273.15) ** 4
    t4 = (eps_sensor * ts ** 4 + (eps_true - eps_sensor) * tr4) / eps_true
    out = np.power(np.maximum(t4, 0.0), 0.25) - 273.15
    return out if arr.ndim else float(out)


def emissivity_offset_c(eps_true: float, eps_sensor: float = 0.95,
                        t_obj_c: float = 30.5, t_refl_c: float = 22.0) -> float:
    """Temperature change (°C) the correction applies to a typical mouse surface reading."""
    return float(correct_emissivity(t_obj_c, eps_true, eps_sensor, t_refl_c) - t_obj_c)

# ── Spatial smoothing ─────────────────────────────────────────────────────────

def spatial_smooth(grid_2d: np.ndarray, sigma: float = 0.5) -> np.ndarray:
    """Gaussian spatial smoothing on a 2D thermal grid."""
    return gaussian_filter(grid_2d.astype(np.float64), sigma=sigma)


# ── Max pixel detection (MultisiteTempArduino 2.py, lines 101-108) ───────────

def find_max_pixel(grid_2d: np.ndarray):
    """Return (row, col) of the hottest pixel and its temperature."""
    idx = np.argmax(grid_2d)
    row, col = np.unravel_index(idx, grid_2d.shape)
    return int(row), int(col), float(grid_2d[row, col])


# ── ROI extraction (MultisiteTempArduino 2.py, lines 111-132) ────────────────

def find_roi_dynamic(grid_2d: np.ndarray, roi_w: int, roi_h: int) -> np.ndarray:
    """
    Centre an roi_w × roi_h region on the hottest pixel (clamped to grid bounds).
    Returns the 2D ROI sub-array.
    """
    h, w = grid_2d.shape
    row, col, _ = find_max_pixel(grid_2d)

    r0 = max(0, row - roi_h // 2)
    r1 = min(h, r0 + roi_h)
    r0 = max(0, r1 - roi_h)

    c0 = max(0, col - roi_w // 2)
    c1 = min(w, c0 + roi_w)
    c0 = max(0, c1 - roi_w)

    return grid_2d[r0:r1, c0:c1]


def find_roi_static(grid_2d: np.ndarray, roi_x: int, roi_y: int,
                    roi_w: int, roi_h: int) -> np.ndarray:
    """Slice a fixed bounding box from the grid."""
    h, w = grid_2d.shape
    x0 = max(0, roi_x)
    y0 = max(0, roi_y)
    x1 = min(w, x0 + roi_w)
    y1 = min(h, y0 + roi_h)
    return grid_2d[y0:y1, x0:x1]


# ── Temporal smoothing (MultisiteTempArduino 2.py, lines ~410) ───────────────

def temporal_smooth(series: np.ndarray, window: int = 11, poly: int = 3) -> np.ndarray:
    """
    Savitzky-Golay temporal smoothing.
    Falls back to a simple moving average if the series is too short.
    """
    n = len(series)
    if n < 4:
        return series.copy()
    w = window if window % 2 == 1 else window + 1
    w = min(w, n if n % 2 == 1 else n - 1)
    w = max(w, poly + 2 if (poly + 2) % 2 == 1 else poly + 3)
    try:
        return _savgol(series, window_length=w, polyorder=poly)
    except Exception:
        return pd.Series(series).rolling(window=max(3, window // 2),
                                          min_periods=1, center=True).mean().to_numpy()


# ── Normalization (MultisiteTempArduino 2.py, lines ~430) ────────────────────

def normalize_to_baseline(series: np.ndarray, time_s: np.ndarray,
                           baseline_end_sec: float = 300.0,
                           baseline_start_sec: float = 0.0) -> np.ndarray:
    """
    Baseline-subtract and normalize: (x - baseline_mean) / baseline_mean
    Returns fractional change from baseline (0 = at baseline).

    The baseline is the NaN-ignoring mean over [baseline_start_sec, baseline_end_sec]
    (start = end of the settling period, so the transient after placing the animal
    in the box does not define "normal").  Falls back to the whole recording when the
    window holds no data.
    """
    finite = np.isfinite(series)
    mask = (time_s >= baseline_start_sec) & (time_s <= baseline_end_sec) & finite
    if mask.sum() == 0:
        mask = finite
    if mask.sum() == 0:
        return np.full(len(series), np.nan)
    baseline_mean = float(np.mean(series[mask]))
    if baseline_mean == 0:
        return series - float(np.nanmean(series))
    return (series - baseline_mean) / baseline_mean


# ── NaN-aware helpers (frame-rate independent: everything is on the time axis) ─

def nan_interp(grid: np.ndarray, t: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Interpolate y(t) onto ``grid`` using finite samples only; NaN outside their range."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(t) & np.isfinite(y)
    if ok.sum() < 2:
        return np.full(len(grid), np.nan)
    return np.interp(grid, t[ok], y[ok], left=np.nan, right=np.nan)

def bin_mean(grid: np.ndarray, t: np.ndarray, y: np.ndarray, half_width: float) -> np.ndarray:
    """
    Mean of the finite samples of y(t) within ±half_width seconds of each grid point
    (NaN if none).  Unlike point interpolation this uses every valid frame in the bin,
    so a single noisy frame cannot define a grid value.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    out = np.full(len(grid), np.nan)
    if len(t) == 0:
        return out
    csum = np.concatenate([[0.0], np.cumsum(y)])
    lo = np.searchsorted(t, grid - half_width, side="left")
    hi = np.searchsorted(t, grid + half_width, side="right")
    cnt = hi - lo
    good = cnt > 0
    out[good] = (csum[hi[good]] - csum[lo[good]]) / cnt[good]
    return out


def smooth_series_nan(series: np.ndarray, roll_window: int = 15,
                      gauss_sigma: float = 3.0) -> np.ndarray:
    """``smooth_series`` applied only to the finite stretch; NaNs are kept as NaN."""
    series = np.asarray(series, dtype=float)
    ok = np.isfinite(series)
    out = np.full(len(series), np.nan)
    if ok.sum() == 0:
        return out
    idx = np.flatnonzero(ok)
    # smooth each contiguous finite run separately so gaps never leak into data
    for run in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
        out[run] = smooth_series(series[run], roll_window, gauss_sigma)
    return out


# ── Polynomial fitting (MultisiteTempArduino 2.py, lines ~460) ───────────────

def poly_fit(time_s: np.ndarray, values: np.ndarray, degree: int = 5):
    """Fit a polynomial and return the fitted values (NaN where ``values`` is NaN)."""
    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    ok = np.isfinite(values) & np.isfinite(time_s)
    if ok.sum() < degree + 2:
        return values.copy()
    out = np.full(len(values), np.nan)
    try:
        coeffs = np.polyfit(time_s[ok], values[ok], degree)
        out[ok] = np.polyval(coeffs, time_s[ok])
        return out
    except Exception:
        return values.copy()


# ── Group-level smoothing (Combine temp apap.py, _smooth_series) ─────────────

def smooth_series(series: np.ndarray, roll_window: int = 15,
                  gauss_sigma: float = 3.0) -> np.ndarray:
    """Two-stage smoothing: rolling mean then Gaussian (from Combine temp apap.py)."""
    s = pd.Series(series).rolling(window=roll_window, min_periods=1,
                                   center=True).mean().to_numpy()
    return gaussian_filter1d(s, sigma=gauss_sigma)


# ── Rate of change (Combine temp apap.py, plot_rate_of_change) ───────────────

def rate_of_change(time_s: np.ndarray, values: np.ndarray,
                   gauss_sigma: float = 3.0) -> np.ndarray:
    """
    Compute smoothed rate of change in °C/min.
    Uses np.gradient then Gaussian smoothing (from Combine temp apap.py).
    """
    time_s = np.asarray(time_s, dtype=float)
    values = np.asarray(values, dtype=float)
    ok = np.isfinite(values)
    if not ok.all():
        # differentiate the finite stretch only; keep NaN elsewhere
        out = np.full(len(values), np.nan)
        if ok.sum() >= 3:
            out[ok] = rate_of_change(time_s[ok], values[ok], gauss_sigma)
        return out
    if len(time_s) < 3:
        return np.zeros_like(values)
    dt_min = np.diff(time_s) / 60.0  # seconds → minutes
    dv = np.diff(values)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        rate = np.where(dt_min != 0, dv / dt_min, 0.0)
    # Pad to original length and smooth
    rate_full = np.concatenate([[rate[0]], rate])
    return gaussian_filter1d(rate_full, sigma=gauss_sigma)


# ── T_core estimation (van der Vinne et al., 2020) ───────────────────────────

def compute_tcore_estimate(roi_max_series: np.ndarray,
                            fps: float = 10.0,
                            sampling_interval_sec: float = 60.0,
                            averaging_window_min: float = 30.0,
                            slope: float = 0.93,
                            intercept: float = 7.1,
                            time_s: np.ndarray | None = None,
                            settle_s: float = 0.0):
    """
    Estimate core body temperature from skin temperature time series.

    van der Vinne et al. (2020) Scientific Reports 10:20680:
      1. Divide frames into sampling_interval_sec blocks
      2. Take the MAXIMUM pixel temperature per block → T_skin,max[i]
      3. Rolling average of T_skin,max over averaging_window_min
      4. T_core_estimated = slope * T_skin_max_rolling + intercept

    Parameters
    ----------
    roi_max_series : (N,) array of per-frame ROI max temperatures (°C)
    fps            : camera frames per second (default 10)
    sampling_interval_sec : seconds per sampling block (default 60 s)
    averaging_window_min  : rolling average window length in minutes (default 30 min)
    slope          : linear regression slope (paper group average 0.93)
    intercept      : linear regression intercept in °C (paper group average 7.1)

    Returns
    -------
    sample_times_s     : (M,) array — centre of each sampling block in seconds
    tskin_max          : (M,) array — max T_skin per sampling block
    tcore_estimated    : (M,) array — estimated T_core
    """
    if time_s is not None:
        return _tcore_time_based(roi_max_series, np.asarray(time_s, dtype=float), fps,
                                 sampling_interval_sec, averaging_window_min,
                                 slope, intercept, settle_s)

    frames_per_sample = max(1, int(round(sampling_interval_sec * fps)))
    n_samples = len(roi_max_series) // frames_per_sample

    if n_samples == 0:
        # Recording shorter than one sampling block — use whole series as one sample
        tskin_max = np.array([float(roi_max_series.max())])
        sample_times_s = np.array([len(roi_max_series) / (2.0 * fps)])
        tcore_estimated = slope * tskin_max + intercept
        return sample_times_s, tskin_max, tcore_estimated

    def _block_max(block):
        valid = block[~np.isnan(block)]
        return float(valid.max()) if len(valid) > 0 else np.nan

    tskin_max = np.array([
        _block_max(roi_max_series[i * frames_per_sample:(i + 1) * frames_per_sample])
        for i in range(n_samples)
    ])

    # Centre of each block
    sample_times_s = (np.arange(n_samples) + 0.5) * sampling_interval_sec

    # Rolling mean over averaging_window_min
    samples_per_window = max(1, int(round(averaging_window_min * 60.0 / sampling_interval_sec)))
    tskin_max_rolling = (
        pd.Series(tskin_max)
        .rolling(window=samples_per_window, min_periods=1, center=False)
        .mean()
        .to_numpy()
    )

    tcore_estimated = slope * tskin_max_rolling + intercept

    return sample_times_s, tskin_max, tcore_estimated


def _tcore_time_based(roi_max_series, time_s, fps, interval_s, window_min,
                      slope, intercept, settle_s, min_coverage: float = 0.3):
    """
    T_core estimate with sampling blocks defined in *seconds* on the real time axis.

    Block k covers [k·interval, (k+1)·interval).  Frames before ``settle_s`` and
    invalid (NaN) frames are ignored.  A block is kept only if its valid frames cover
    ≥ ``min_coverage`` of the post-settle block duration, so a block where the animal
    was mostly out of view cannot report a spuriously low maximum.  Blocks with no
    usable data are dropped (not zero-filled).  The trailing partial block is kept
    when at least half an interval long.
    """
    y = np.where(time_s >= settle_s, np.asarray(roi_max_series, dtype=float), np.nan)
    t_end = float(time_s[-1]) if len(time_s) else 0.0
    n_blocks = int(np.ceil(t_end / interval_s - 0.5)) if t_end > 0 else 0
    n_blocks = max(n_blocks, 1)
    centres, tskin = [], []
    for k in range(n_blocks):
        b0, b1 = k * interval_s, min((k + 1) * interval_s, t_end)
        eff = b1 - max(b0, settle_s)
        if eff <= 0:
            continue
        sel = (time_s >= b0) & (time_s < b1) & np.isfinite(y)
        if sel.sum() < min_coverage * eff * fps or sel.sum() == 0:
            continue
        centres.append((max(b0, settle_s) + b1) / 2.0)
        tskin.append(float(y[sel].max()))
    if not tskin:
        return np.array([]), np.array([]), np.array([])
    tskin = np.array(tskin)
    centres = np.array(centres)
    n_win = max(1, int(round(window_min * 60.0 / interval_s)))
    rolled = pd.Series(tskin).rolling(window=n_win, min_periods=1).mean().to_numpy()
    return centres, tskin, slope * rolled + intercept
