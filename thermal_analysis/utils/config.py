"""Application-wide default constants for Glöd."""
from dataclasses import dataclass, field
from typing import List


# ── Camera defaults ──────────────────────────────────────────────────────────
CAMERA_WIDTH: int = 80
CAMERA_HEIGHT: int = 62
METADATA_COLS: int = 8          # columns before pixel data in each line
CAMERA_FPS: float = 10.0

# ── ROI defaults ─────────────────────────────────────────────────────────────
ROI_MODE_DYNAMIC: str = "dynamic"
ROI_MODE_STATIC: str = "static"
ROI_DEFAULT_MODE: str = ROI_MODE_DYNAMIC
ROI_DEFAULT_W: int = 10
ROI_DEFAULT_H: int = 10
ROI_DEFAULT_X: int = 35
ROI_DEFAULT_Y: int = 26

# ── Thermal display ──────────────────────────────────────────────────────────
T_MIN_DEFAULT: float = 22.5
T_MAX_DEFAULT: float = 35.0
COLORMAP_DEFAULT: str = "inferno"
COLORMAPS: List[str] = ["inferno", "plasma", "viridis", "magma", "hot", "jet"]

# ── Emissivity correction ────────────────────────────────────────────────────
APPLY_EMISSIVITY: bool = False
EPSILON_MOUSE: float = 0.93
EPSILON_SENSOR: float = 1.00

# ── T_core estimation (van der Vinne et al., 2020) ───────────────────────────
TCORE_ENABLED: bool = True
TCORE_SAMPLING_INTERVAL_SEC: float = 60.0   # take max T_skin per this interval
TCORE_AVERAGING_WINDOW_MIN: float = 30.0    # rolling average over this window
TCORE_SLOPE: float = 0.93
TCORE_INTERCEPT: float = 7.1               # °C

# ── Signal processing ────────────────────────────────────────────────────────
SPATIAL_SIGMA: float = 0.5          # Gaussian spatial smoothing
SAVGOL_WINDOW: int = 11             # Savitzky-Golay temporal smoothing window
SAVGOL_POLY: int = 3
ROLLING_WINDOW: int = 15            # group-level rolling mean (Combine temp)
GAUSSIAN_SIGMA_GROUP: float = 3.0   # group-level Gaussian polish

# ── Analysis ─────────────────────────────────────────────────────────────────
BASELINE_END_SEC: float = 300.0     # first N seconds used as baseline
POLY_DEGREE: int = 5
DOWN_SAMPLE_PERIOD_SEC: float = 30.0  # interpolation step for group aggregation

# ── Statistical tests ────────────────────────────────────────────────────────
NORMALITY_MIN_N: int = 3            # min n per group to apply Shapiro-Wilk

# ── Video export ─────────────────────────────────────────────────────────────
VIDEO_PANEL_W: int = 320
VIDEO_PANEL_H: int = 240
VIDEO_HEADER_H: int = 36
VIDEO_FPS_DEFAULT: float = 10.0

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR_PREFIX: str = "glod_output"


@dataclass
class AnalysisSettings:
    """All user-configurable parameters passed from UI to workers."""
    camera_width: int = CAMERA_WIDTH
    camera_height: int = CAMERA_HEIGHT
    metadata_cols: int = METADATA_COLS
    fps: float = CAMERA_FPS

    roi_mode: str = ROI_DEFAULT_MODE
    roi_x: int = ROI_DEFAULT_X
    roi_y: int = ROI_DEFAULT_Y
    roi_w: int = ROI_DEFAULT_W
    roi_h: int = ROI_DEFAULT_H

    colormap: str = COLORMAP_DEFAULT
    t_min: float = T_MIN_DEFAULT
    t_max: float = T_MAX_DEFAULT

    apply_emissivity: bool = APPLY_EMISSIVITY
    epsilon_mouse: float = EPSILON_MOUSE
    epsilon_sensor: float = EPSILON_SENSOR

    tcore_enabled: bool = TCORE_ENABLED
    tcore_sampling_interval_sec: float = TCORE_SAMPLING_INTERVAL_SEC
    tcore_averaging_window_min: float = TCORE_AVERAGING_WINDOW_MIN
    tcore_slope: float = TCORE_SLOPE
    tcore_intercept: float = TCORE_INTERCEPT

    baseline_end_sec: float = BASELINE_END_SEC
    spatial_sigma: float = SPATIAL_SIGMA
    savgol_window: int = SAVGOL_WINDOW

    export_video: bool = False
    output_dir: str = ""
