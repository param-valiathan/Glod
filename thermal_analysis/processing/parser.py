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
    temporal_smooth, correct_emissivity,
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
        temporal_smooth, correct_emissivity,
        normalize_to_baseline, compute_tcore_estimate,
    )
    from processing.mouse_tracker import (track_body, bridge_gaps, estimate_fps,
                                          compute_activity, find_pauses, collapse_pauses)

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

        # ── 6b. Time axis and *measured* frame rate ───────────────────────
        # Every temporal window below is defined in seconds and converted with
        # the real rate (the rig delivers ~6 Hz, not the nominal 10 Hz).
        t0 = timestamps.iloc[0]
        time_s = (timestamps - t0).dt.total_seconds().to_numpy()
        # A recorder pause is cut out of the time axis: the recording simply continues from
        # where it stopped.  Raw timestamps stay in the file; ``gap_before_s`` keeps the record.
        pauses = find_pauses(time_s)
        time_s, gap_before = collapse_pauses(time_s)
        fps_nominal = float(settings_dict["fps"])
        fps = estimate_fps(time_s, default=fps_nominal)
        settle_s = float(settings_dict.get("settle_s", 0.0))
        body_info = None

        # ── 7. ROI and max-pixel extraction ───────────────────────────────
        roi_mode = settings_dict["roi_mode"]
        roi_x = settings_dict["roi_x"]
        roi_y = settings_dict["roi_y"]
        roi_w = settings_dict["roi_w"]
        roi_h = settings_dict["roi_h"]

        if roi_mode == "static":
            # Fixed box — for restrained / anaesthetised animals that do not move
            y0 = max(0, roi_y)
            y1 = min(height, y0 + roi_h)
            x0 = max(0, roi_x)
            x1 = min(width, x0 + roi_w)
            roi_stack = stack[:, y0:y1, x0:x1]  # (N, roi_h, roi_w)
            roi_means = roi_stack.mean(axis=(1, 2))
            roi_maxes = roi_stack.max(axis=(1, 2))
            flat_max = stack.reshape(n_frames, -1)
            max_indices = flat_max.argmax(axis=1)
            max_rows = (max_indices // width).astype(int)
            max_cols = (max_indices % width).astype(int)
            max_temps = flat_max[np.arange(n_frames), max_indices]
            head_max = np.full(n_frames, np.nan)
            head_clipped = np.zeros(n_frames, dtype=bool)
        else:
            # Body tracking (default): adaptive segmentation + quality gating
            # (see mouse_tracker.py).  Invalid frames are NaN, never dips.
            if settings_dict.get("arena_enabled"):
                # Everything outside the arena rectangle is replaced by the frame
                # median so warm objects outside the box cannot be tracked.
                inside = np.zeros((height, width), dtype=bool)
                ax0, ay0 = max(0, settings_dict["arena_x"]), max(0, settings_dict["arena_y"])
                inside[ay0:ay0 + settings_dict["arena_h"], ax0:ax0 + settings_dict["arena_w"]] = True
                med = np.median(stack.reshape(n_frames, -1), axis=1)
                stack[:, ~inside] = med[:, None]
            roi_mode = "body"
            body_info = track_body(stack, time_s, fps)
            roi_means = body_info["roi_mean"]
            roi_maxes = body_info["roi_max"]
            valid = body_info["valid"]
            # Trajectory = body centroid (stable), not the noisy hottest pixel
            max_rows = np.where(valid, np.round(body_info["cent_row"]), -1).astype(int)
            max_cols = np.where(valid, np.round(body_info["cent_col"]), -1).astype(int)
            max_temps = roi_maxes.copy()
            head_max = body_info["head_max"]
            head_clipped = body_info["head_clipped"]
            speed, moving, body_len = compute_activity(
                body_info["cent_row"], body_info["cent_col"], time_s, valid,
                body_info["body_area"], fps)
        # ── 8. Coat emissivity correction of the temperature series ─────────
        # Segmentation above ran on the sensor values; the correction (physics in
        # roi_analysis.correct_emissivity) is applied to the read-out temperatures only.
        # T_core keeps the uncorrected series: its regression was calibrated on raw camera
        # skin temperatures, so correcting the input would shift the estimate.
        roi_maxes_sensor = roi_maxes.copy()
        emis_info = None
        if settings_dict["apply_emissivity"]:
            eps_s = float(settings_dict["epsilon_sensor"])
            eps_m = float(settings_dict["epsilon_mouse"])
            t_refl = settings_dict.get("reflected_temp_c")
            if t_refl is None:       # room/bedding temperature ≈ cold floor of the image
                t_refl = float(np.median(np.percentile(stack.reshape(n_frames, -1), 10, axis=1)))
            def _fix(a):
                return correct_emissivity(a, eps_m, eps_s, t_refl)
            roi_means, roi_maxes, max_temps, head_max = (
                _fix(roi_means), _fix(roi_maxes), _fix(max_temps), _fix(head_max))
            emis_info = {"eps_true": eps_m, "eps_sensor": eps_s, "t_refl_c": t_refl}
            if body_info is not None:
                body_info["body_p90"] = _fix(body_info["body_p90"])

        # ── 9. Temporal smoothing (window defined at the 10 FPS reference) ─
        # Gaps ≤ 2 s are bridged by time-based interpolation; longer gaps stay
        # NaN (absent data is shown as absent, never as a dip).  The Savitzky-
        # Golay window is rescaled to the measured rate so it spans the same
        # number of seconds regardless of camera frame rate.
        sw = int(settings_dict["savgol_window"])
        sw_eff = max(5, int(round(sw * fps / 10.0)) | 1)
        bridged = bridge_gaps(roi_maxes, time_s)
        gaps = np.isnan(bridged)
        if gaps.all():
            roi_max_smoothed = bridged.copy()
        else:
            filled = np.interp(time_s, time_s[~gaps], bridged[~gaps])
            roi_max_smoothed = temporal_smooth(filled, window=sw_eff)
            roi_max_smoothed[gaps] = np.nan
        # ── 10. Normalisation (baseline starts after the settling period) ──
        baseline_end = settings_dict["baseline_end_sec"]
        baseline_start = settle_s if settle_s < baseline_end else 0.0
        normalized_roi_max = normalize_to_baseline(roi_maxes, time_s, baseline_end,
                                                   baseline_start)

        # ── 11. T_core estimation ─────────────────────────────────────────
        tcore_result = None
        if settings_dict["tcore_enabled"]:
            sample_times, tskin_max, tcore_est = compute_tcore_estimate(
                roi_maxes_sensor,
                fps=fps,
                sampling_interval_sec=settings_dict["tcore_sampling_interval_sec"],
                averaging_window_min=settings_dict["tcore_averaging_window_min"],
                slope=settings_dict["tcore_slope"],
                intercept=settings_dict["tcore_intercept"],
                time_s=time_s,
                settle_s=settle_s,
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
            "roi_max_sensor": roi_maxes_sensor.tolist(),
            "head_max": head_max.tolist(),
            "head_clipped": head_clipped.tolist(),
            "emissivity": emis_info,
            "roi_max_smoothed": roi_max_smoothed.tolist(),
            "normalized_roi_max": normalized_roi_max.tolist(),
            "max_row": max_rows.tolist(),
            "max_col": max_cols.tolist(),
            "max_temp": max_temps.tolist(),
            "tcore": tcore_result,
            # Store raw stack reference as path for video export
            "stack_shape": list(stack.shape),
            "fps_measured": fps,
            "settle_s": settle_s,
            "in_settle": (time_s < settle_s).tolist(),
            "gap_before_s": gap_before.tolist(),
            "pauses": pauses,
            "tracking": (None if body_info is None else {
                "valid": body_info["valid"].tolist(),
                "reason": body_info["reason"].tolist(),
                "body_area": body_info["body_area"].tolist(),
                "body_p90": body_info["body_p90"].tolist(),
                "speed_px_s": speed.tolist(),
                "moving": moving.tolist(),
                "body_len_px": body_len,
            }),
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
    roi_df["in_settle"] = result["in_settle"]
    roi_df["gap_before_s"] = result["gap_before_s"]
    roi_df["roi_max_sensor"] = result["roi_max_sensor"]
    roi_df["head_max"] = result["head_max"]
    roi_df["head_clipped"] = result["head_clipped"]
    trk = result.get("tracking")
    if trk:
        roi_df["body_valid"] = trk["valid"]
        roi_df["body_area_px"] = trk["body_area"]
        roi_df["body_p90"] = trk["body_p90"]
        roi_df["speed_px_s"] = trk["speed_px_s"]
        roi_df["moving"] = trk["moving"]
        roi_df.attrs["body_len_px"] = trk["body_len_px"]

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


def measure_fps(path: str, default: float = 10.0, n_rows: int = 1200) -> float:
    """Frame rate of a recording from its own timestamps (first ``n_rows`` frames)."""
    from .mouse_tracker import estimate_fps
    try:
        col = pd.read_csv(path, header=None, usecols=[0], dtype=str,
                          nrows=n_rows, on_bad_lines="skip")
        ts = pd.to_datetime(col.iloc[:, 0], errors="coerce", utc=True).dropna()
        if len(ts) < 3:
            return default
        return estimate_fps((ts - ts.iloc[0]).dt.total_seconds().to_numpy(), default)
    except Exception:
        return default


def load_raw_frames(path: str, settings, return_times: bool = False):
    """
    Load raw (N, H, W) pixel stack from a .txt file — used by VideoExporter.
    Returns None on error.  With ``return_times`` returns ``(stack, time_s)`` where ``time_s``
    are the real per-frame times in seconds (None if they cannot be aligned), so a recorder
    pause is visible to the caller.
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
        if not return_times:
            return result
        times = None
        try:
            col = pd.read_csv(path, header=None, usecols=[0], dtype=str,
                              on_bad_lines="skip", engine="c")
            ts = pd.to_datetime(col.iloc[:, 0], errors="coerce", utc=True)
            if len(ts) == len(valid) and ts.notna().all():
                times = (ts[valid] - ts.iloc[0]).dt.total_seconds().to_numpy()
        except Exception:
            times = None
        return result, times
    except Exception as exc:
        log.error("Failed to load frames from %s: %s", path, exc)
        return None
