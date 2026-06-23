"""
High-performance thermal data parser for Glöd.

Strategy
--------
1. `pandas.read_csv()` loads the entire file in one C-accelerated call.
2. Pixel columns are extracted as a single (N, H*W) NumPy matrix and reshaped
   to (N, H, W) in one vectorised operation — no Python-level frame loop.
3. Spatial smoothing is applied across the whole 3D stack simultaneously.
4. ROI extraction is a simple NumPy slice over the frame axis.
5. `ProcessPoolExecutor` allows multiple files to be parsed in parallel across
   CPU cores (or, if CuPy is available, the heavy maths moves to the GPU).

GPU note
--------
Set USE_GPU = True (auto-detected below) to move the 3D convolution and ROI
maths to the GPU via CuPy.  The result is transferred back to NumPy before
returning.  Falls back silently to NumPy if CuPy is not installed.
"""

from __future__ import annotations

import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .roi_analysis import (
    find_max_pixel,
    find_roi_dynamic,
    find_roi_static,
    apply_emissivity_correction_array,
    temporal_smooth,
    normalize_to_baseline,
    compute_tcore_estimate,
)

log = logging.getLogger(__name__)

# ── Optional GPU support (CuPy) ───────────────────────────────────────────────
try:
    import cupy as cp
    from cupyx.scipy.ndimage import gaussian_filter as cp_gauss
    USE_GPU = True
    log.info("CuPy detected — GPU acceleration enabled for thermal processing.")
except ImportError:
    USE_GPU = False
    log.debug("CuPy not available — using CPU (NumPy) processing.")


# ── GPU-aware Gaussian filter ─────────────────────────────────────────────────

def _gauss3d(stack: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply per-frame Gaussian spatial smoothing to a (N, H, W) stack.
    Uses GPU (CuPy) if available, otherwise NumPy/SciPy on CPU.
    sigma is applied only on axes 1 and 2 (H, W); axis 0 (frames) is not blurred.
    """
    if USE_GPU and sigma > 0:
        try:
            gpu_stack = cp.asarray(stack, dtype=cp.float32)
            smoothed = cp_gauss(gpu_stack, sigma=(0, sigma, sigma))
            return cp.asnumpy(smoothed)
        except Exception as exc:
            log.warning("GPU smoothing failed (%s), falling back to CPU.", exc)

    if sigma > 0:
        return gaussian_filter(stack.astype(np.float64), sigma=(0, sigma, sigma))
    return stack.astype(np.float64)


# ── Core file-parsing function (runs in worker process) ──────────────────────

def _parse_single_file(path: str, settings_dict: dict) -> Optional[dict]:
    """
    Parse one thermal camera .txt file into per-frame ROI metrics.

    Designed to run inside ProcessPoolExecutor (no Qt objects allowed here).
    Returns a serialisable dict or None on failure.
    """
    import numpy as np
    import pandas as pd
    from scipy.ndimage import gaussian_filter
    from pathlib import Path

    # Re-import local modules (worker process has its own namespace)
    import sys, os
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from processing.roi_analysis import (
        find_max_pixel, find_roi_dynamic, find_roi_static,
        apply_emissivity_correction_array, temporal_smooth,
        normalize_to_baseline, compute_tcore_estimate,
    )

    try:
        meta_cols = settings_dict["metadata_cols"]
        height    = settings_dict["camera_height"]
        width     = settings_dict["camera_width"]
        n_pixels  = height * width

        # ── 1. Bulk load with pandas (C-accelerated CSV reader) ───────────
        df_raw = pd.read_csv(path, header=None, dtype=str, on_bad_lines="skip",
                              engine="c", low_memory=False)

        if df_raw.empty or df_raw.shape[1] < meta_cols + n_pixels:
            log.warning("File %s has too few columns (%d); expected ≥ %d",
                        path, df_raw.shape[1], meta_cols + n_pixels)
            return None

        # ── 2. Parse timestamps from column 0 ─────────────────────────────
        timestamps = pd.to_datetime(df_raw.iloc[:, 0], errors="coerce", utc=True)
        valid_mask = timestamps.notna().to_numpy()
        df_raw = df_raw.loc[valid_mask].reset_index(drop=True)
        timestamps = timestamps[valid_mask].reset_index(drop=True)

        if len(df_raw) == 0:
            log.warning("No valid frames in %s", path)
            return None

        n_frames = len(df_raw)

        # ── 3. Extract pixel block as float32 NumPy array ─────────────────
        pixel_cols = df_raw.iloc[:, meta_cols: meta_cols + n_pixels]
        pixels = pixel_cols.to_numpy(dtype=np.float32, na_value=np.nan)
        del df_raw, pixel_cols   # free the large string DataFrame immediately

        # Drop frames where any pixel is NaN
        valid_rows = ~np.isnan(pixels).any(axis=1)
        pixels = pixels[valid_rows]   # boolean-index → new contiguous array
        timestamps = timestamps[valid_rows.tolist()]
        del valid_rows
        n_frames = len(pixels)

        if n_frames == 0:
            log.warning("All frames invalid in %s", path)
            return None

        # ── 4. Reshape to (N, H, W) and promote to float64 once ───────────
        # Single promotion here means no further copies are needed downstream.
        stack = pixels.reshape(n_frames, height, width).astype(np.float64)
        del pixels   # float32 source no longer needed

        # ── 5. Optional emissivity correction — all in-place ──────────────
        if settings_dict["apply_emissivity"]:
            eps_s = settings_dict["epsilon_sensor"]
            eps_m = settings_dict["epsilon_mouse"]
            stack += 273.15                       # °C → K
            np.power(stack, 4, out=stack)         # T_K^4
            stack *= eps_s / eps_m                # emissivity scale
            np.power(stack, 0.25, out=stack)      # back to T_K
            stack -= 273.15                       # K → °C

        # ── 6. Spatial smoothing over entire 3D stack ──────────────────────
        sigma = settings_dict["spatial_sigma"]
        if sigma > 0:
            try:
                if USE_GPU:
                    import cupy as cp
                    from cupyx.scipy.ndimage import gaussian_filter as cp_gauss
                    gpu_s = cp.asarray(stack, dtype=cp.float32)
                    stack = cp.asnumpy(cp_gauss(gpu_s, sigma=(0, sigma, sigma)))
                    del gpu_s
                    cp.get_default_memory_pool().free_all_blocks()
                else:
                    # stack is already float64; gaussian_filter writes a new array
                    stack = gaussian_filter(stack, sigma=(0, sigma, sigma))
            except Exception:
                stack = gaussian_filter(stack, sigma=(0, sigma, sigma))

        # ── 7. ROI and max-pixel extraction ───────────────────────────────
        roi_mode = settings_dict["roi_mode"]
        roi_x = settings_dict["roi_x"]
        roi_y = settings_dict["roi_y"]
        roi_w = settings_dict["roi_w"]
        roi_h = settings_dict["roi_h"]

        if roi_mode == "static":
            y0 = max(0, roi_y)
            y1 = min(height, y0 + roi_h)
            x0 = max(0, roi_x)
            x1 = min(width, x0 + roi_w)
            roi_stack = stack[:, y0:y1, x0:x1]  # (N, roi_h, roi_w)
            roi_means = roi_stack.mean(axis=(1, 2))
            roi_maxes = roi_stack.max(axis=(1, 2))
        else:
            # Dynamic: per-frame, find hottest pixel → extract ROI around it
            flat = stack.reshape(n_frames, -1)
            hot_idx = flat.argmax(axis=1)
            hot_rows = hot_idx // width
            hot_cols = hot_idx % width
            roi_means = np.empty(n_frames, dtype=np.float64)
            roi_maxes = np.empty(n_frames, dtype=np.float64)
            rh2, rw2 = roi_h // 2, roi_w // 2
            for i in range(n_frames):
                r, c = int(hot_rows[i]), int(hot_cols[i])
                r0 = max(0, r - rh2); r1 = min(height, r0 + roi_h); r0 = max(0, r1 - roi_h)
                c0 = max(0, c - rw2); c1 = min(width, c0 + roi_w); c0 = max(0, c1 - roi_w)
                patch = stack[i, r0:r1, c0:c1]
                roi_means[i] = patch.mean()
                roi_maxes[i] = patch.max()
            hot_rows = hot_rows.astype(int)
            hot_cols = hot_cols.astype(int)

        # Max pixel per frame (vectorised)
        flat_max = stack.reshape(n_frames, -1)
        max_indices = flat_max.argmax(axis=1)
        max_rows = (max_indices // width).astype(int)
        max_cols = (max_indices % width).astype(int)
        max_temps = flat_max[np.arange(n_frames), max_indices]

        # ── 8. Time axis ──────────────────────────────────────────────────
        t0 = timestamps.iloc[0]
        time_s = np.array([(ts - t0).total_seconds() for ts in timestamps])

        # ── 9. Temporal smoothing ─────────────────────────────────────────
        sw = settings_dict["savgol_window"]
        roi_max_smoothed = temporal_smooth(roi_maxes, window=sw)

        # ── 10. Normalisation ─────────────────────────────────────────────
        baseline_end = settings_dict["baseline_end_sec"]
        normalized_roi_max = normalize_to_baseline(roi_maxes, time_s, baseline_end)

        # ── 11. T_core estimation ─────────────────────────────────────────
        tcore_result = None
        if settings_dict["tcore_enabled"]:
            sample_times, tskin_max, tcore_est = compute_tcore_estimate(
                roi_maxes,
                fps=settings_dict["fps"],
                sampling_interval_sec=settings_dict["tcore_sampling_interval_sec"],
                averaging_window_min=settings_dict["tcore_averaging_window_min"],
                slope=settings_dict["tcore_slope"],
                intercept=settings_dict["tcore_intercept"],
            )
            tcore_result = {
                "time_s": sample_times.tolist(),
                "tskin_max": tskin_max.tolist(),
                "tcore_estimated": tcore_est.tolist(),
            }

        return {
            "path": path,
            "n_frames": n_frames,
            "time_s": time_s.tolist(),
            "roi_mean": roi_means.tolist(),
            "roi_max": roi_maxes.tolist(),
            "roi_max_smoothed": roi_max_smoothed.tolist(),
            "normalized_roi_max": normalized_roi_max.tolist(),
            "max_row": max_rows.tolist(),
            "max_col": max_cols.tolist(),
            "max_temp": max_temps.tolist(),
            "tcore": tcore_result,
            # Store raw stack reference as path for video export
            "stack_shape": list(stack.shape),
        }

    except Exception as exc:
        log.exception("Error parsing file %s: %s", path, exc)
        return None


def result_to_dataframes(result: dict) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """Convert a _parse_single_file result dict to (roi_df, tcore_df)."""
    roi_df = pd.DataFrame({
        "time_s": result["time_s"],
        "roi_mean": result["roi_mean"],
        "roi_max": result["roi_max"],
        "roi_max_smoothed": result["roi_max_smoothed"],
        "normalized_roi_max": result["normalized_roi_max"],
        "max_row": result["max_row"],
        "max_col": result["max_col"],
        "max_temp": result["max_temp"],
    })

    tcore_df = None
    if result.get("tcore"):
        tc = result["tcore"]
        tcore_df = pd.DataFrame({
            "time_s": tc["time_s"],
            "tskin_max": tc["tskin_max"],
            "tcore_estimated": tc["tcore_estimated"],
        })

    return roi_df, tcore_df


# ── Public parallel parsing API ───────────────────────────────────────────────

def parse_files_parallel(
    file_paths: List[str],
    settings,
    progress_callback: Optional[Callable[[int], None]] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, dict]:
    """
    Parse multiple thermal camera files in parallel using ProcessPoolExecutor.

    Parameters
    ----------
    file_paths      : list of absolute paths to .txt files
    settings        : AnalysisSettings instance
    progress_callback : called with integer (0-100) as files complete
    max_workers     : number of parallel processes (default: min(n_files, CPU_count))

    Returns
    -------
    dict mapping file path → result dict (or None if parsing failed)
    """
    if not file_paths:
        return {}

    settings_dict = asdict(settings)
    n_files = len(file_paths)
    workers = max_workers or min(n_files, max(1, os.cpu_count() or 1))

    results: Dict[str, Optional[dict]] = {}
    completed = 0

    if workers == 1 or n_files == 1:
        # Single-threaded path (avoids ProcessPoolExecutor overhead for 1 file)
        for path in file_paths:
            results[path] = _parse_single_file(path, settings_dict)
            completed += 1
            if progress_callback:
                progress_callback(int(completed / n_files * 100))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_path = {
                executor.submit(_parse_single_file, p, settings_dict): p
                for p in file_paths
            }
            for future in as_completed(future_to_path):
                path = future_to_path[future]
                try:
                    results[path] = future.result()
                except Exception as exc:
                    log.exception("Worker for %s raised: %s", path, exc)
                    results[path] = None
                completed += 1
                if progress_callback:
                    progress_callback(int(completed / n_files * 100))

    return results


def load_raw_frames(path: str, settings) -> Optional[np.ndarray]:
    """
    Load raw (N, H, W) pixel stack from a .txt file — used by VideoExporter.
    Returns None on error.
    """
    from dataclasses import asdict as _asdict
    sd = _asdict(settings)
    meta_cols = sd["metadata_cols"]
    height    = sd["camera_height"]
    width     = sd["camera_width"]
    n_pixels  = height * width

    try:
        df_raw = pd.read_csv(path, header=None, dtype=np.float32,
                              usecols=list(range(meta_cols, meta_cols + n_pixels)),
                              on_bad_lines="skip", engine="c", low_memory=False)
        pixels = df_raw.to_numpy(dtype=np.float32)
        del df_raw   # free string DataFrame before working with pixel array
        valid = ~np.isnan(pixels).any(axis=1)
        result = pixels[valid].reshape(-1, height, width)
        del pixels
        return result
    except Exception as exc:
        log.error("Failed to load frames from %s: %s", path, exc)
        return None
