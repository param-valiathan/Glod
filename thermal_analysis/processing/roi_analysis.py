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


# ── Emissivity correction (MultisiteTempArduino 2.py, lines 56-99) ───────────

def apply_emissivity_correction(t_obs_c: float, eps_sensor: float = 1.0,
                                 eps_true: float = 0.93) -> float:
    """
    Stefan-Boltzmann emissivity correction.
    T_true = (T_obs_K^4 * eps_sensor / eps_true)^(1/4) - 273.15
    """
    t_k = t_obs_c + 273.15
    t_true_k = (t_k ** 4 * eps_sensor / eps_true) ** 0.25
    return t_true_k - 273.15


def apply_emissivity_correction_array(arr: np.ndarray, eps_sensor: float = 1.0,
                                       eps_true: float = 0.93) -> np.ndarray:
    """Vectorised emissivity correction over a numpy array."""
    t_k = arr + 273.15
    return (t_k ** 4 * eps_sensor / eps_true) ** 0.25 - 273.15


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
                           baseline_end_sec: float = 300.0) -> np.ndarray:
    """
    Baseline-subtract and normalize: (x - baseline_mean) / baseline_mean
    Returns fractional change from baseline (0 = at baseline).
    """
    mask = time_s <= baseline_end_sec
    if mask.sum() == 0:
        mask = np.ones(len(series), dtype=bool)
    baseline_mean = series[mask].mean()
    if baseline_mean == 0:
        return series - series[mask].mean()
    return (series - baseline_mean) / baseline_mean


# ── Polynomial fitting (MultisiteTempArduino 2.py, lines ~460) ───────────────

def poly_fit(time_s: np.ndarray, values: np.ndarray, degree: int = 5):
    """Fit a polynomial and return the fitted values."""
    if len(time_s) < degree + 2:
        return values.copy()
    try:
        coeffs = np.polyfit(time_s, values, degree)
        return np.polyval(coeffs, time_s)
    except np.RankWarning:
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
                            intercept: float = 7.1):
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
    frames_per_sample = max(1, int(round(sampling_interval_sec * fps)))
    n_samples = len(roi_max_series) // frames_per_sample

    if n_samples == 0:
        # Recording shorter than one sampling block — use whole series as one sample
        tskin_max = np.array([float(roi_max_series.max())])
        sample_times_s = np.array([len(roi_max_series) / (2.0 * fps)])
        tcore_estimated = slope * tskin_max + intercept
        return sample_times_s, tskin_max, tcore_estimated

    tskin_max = np.array([
        roi_max_series[i * frames_per_sample:(i + 1) * frames_per_sample].max()
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


# ── Per-frame ROI processing (combines all steps) ────────────────────────────

def process_frame(grid_raw: np.ndarray, settings) -> dict:
    """
    Full per-frame processing pipeline.

    Returns dict with: roi_mean, roi_max, max_row, max_col, max_temp
    """
    grid = spatial_smooth(grid_raw, sigma=settings.spatial_sigma)

    # Optional emissivity correction on full grid
    if settings.apply_emissivity:
        grid = apply_emissivity_correction_array(grid, settings.epsilon_sensor,
                                                  settings.epsilon_mouse)

    max_row, max_col, max_temp = find_max_pixel(grid)

    if settings.roi_mode == "static":
        roi = find_roi_static(grid, settings.roi_x, settings.roi_y,
                              settings.roi_w, settings.roi_h)
    else:
        roi = find_roi_dynamic(grid, settings.roi_w, settings.roi_h)

    roi_mean = float(roi.mean()) if roi.size > 0 else float(max_temp)
    roi_max = float(roi.max()) if roi.size > 0 else float(max_temp)

    return {
        "roi_mean": roi_mean,
        "roi_max": roi_max,
        "max_row": max_row,
        "max_col": max_col,
        "max_temp": max_temp,
    }
