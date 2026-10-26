"""Application-wide default constants for Glöd."""
from dataclasses import dataclass, field
from typing import List


# ── Camera defaults ──────────────────────────────────────────────────────────
CAMERA_WIDTH: int = 80
CAMERA_HEIGHT: int = 62
METADATA_COLS: int = 8          # columns before pixel data in each line
CAMERA_FPS: float = 10.0

# ── ROI defaults ─────────────────────────────────────────────────────────────
ROI_MODE_STATIC: str = "static"   # fixed box, for restrained / anaesthetised animals
ROI_MODE_BODY: str = "body"        # adaptive body segmentation (processing/mouse_tracker.py)
ROI_DEFAULT_MODE: str = ROI_MODE_BODY
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
APPLY_EMISSIVITY: bool = True        # applied to the temperature series (segmentation uses raw data)
EPSILON_MOUSE: float = 0.95          # C57BL/6 preset (see utils/coat_presets.py)
EPSILON_SENSOR: float = 0.95         # MI48 EMISSIVITY register as read from the cameras

# ── T_core estimation (van der Vinne et al., 2020) ───────────────────────────
TCORE_ENABLED: bool = True
TCORE_SAMPLING_INTERVAL_SEC: float = 60.0   # take max T_skin per this interval
TCORE_AVERAGING_WINDOW_MIN: float = 30.0    # rolling average over this window
TCORE_SLOPE: float = 0.93
TCORE_INTERCEPT: float = 7.1               # °C

# ── Arena ────────────────────────────────────────────────────────────────────
ARENA_ENABLED: bool = False
ARENA_X: int = 0
ARENA_Y: int = 0
ARENA_W: int = CAMERA_WIDTH
ARENA_H: int = CAMERA_HEIGHT

# ── Signal processing ────────────────────────────────────────────────────────
SPATIAL_SIGMA: float = 0.5          # Gaussian spatial smoothing
SAVGOL_WINDOW: int = 11             # Savitzky-Golay temporal smoothing window
SAVGOL_POLY: int = 3
ROLLING_WINDOW: int = 15            # group-level rolling mean (Combine temp)
GAUSSIAN_SIGMA_GROUP: float = 3.0   # group-level Gaussian polish

# ── Analysis ─────────────────────────────────────────────────────────────────
BASELINE_END_SEC: float = 300.0     # first N seconds used as baseline
# Seconds after placing the mouse in the box that are excluded from baseline,
# T_core and per-animal metrics.  On the reference 10-min recordings of 3 untreated
# mice the body signal ran 0.3–0.5 °C low for the first ~45–90 s (handling/transfer).
SETTLE_S: float = 90.0
# Recordings whose body was tracked in less than this fraction of post-settle frames are
# excluded from group statistics (still listed in the tracking-QC table).
MIN_COVERAGE: float = 0.5
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

# ── Live capture ──────────────────────────────────────────────────────────────
CAMERA_BASE_FPS: int = 25           # MI48x3 base frame rate (FPS = 25 / divider)
RECORDINGS_SUBDIR: str = "recordings"
# USB VID:PID for Waveshare / Meridian MI48x3 cameras.
# Run find_camera_ports.py with cameras plugged in to confirm values,
# then update this tuple and CLAUDE.md.
KNOWN_CAMERA_VIDS: tuple = (0x0416,)   # Confirmed: VID=0x0416 PID=0xb020 (Winbond/SiGe CDC), COM6/COM7/COM14


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
    coat_label: str = "Mouse · C57BL/6 (black)"
    reflected_temp_c: float | None = None   # None → estimated from the image background

    tcore_enabled: bool = TCORE_ENABLED
    tcore_sampling_interval_sec: float = TCORE_SAMPLING_INTERVAL_SEC
    tcore_averaging_window_min: float = TCORE_AVERAGING_WINDOW_MIN
    tcore_slope: float = TCORE_SLOPE
    tcore_intercept: float = TCORE_INTERCEPT

    baseline_end_sec: float = BASELINE_END_SEC
    settle_s: float = SETTLE_S
    min_coverage: float = MIN_COVERAGE
    spatial_sigma: float = SPATIAL_SIGMA
    savgol_window: int = SAVGOL_WINDOW

    # Same animals measured in several groups/conditions → paired statistics
    paired_design: bool = False

    # Time-point comparison: up to 3 times (s from recording start); empty = off
    timepoints_s: list = field(default_factory=list)
    timepoint_signal: str = "roi_max"      # roi_max | normalized_roi_max | tcore_estimated
    timepoint_window_s: float = 30.0       # mean over [t - w/2, t + w/2]

    arena_enabled: bool = ARENA_ENABLED
    arena_x: int = ARENA_X
    arena_y: int = ARENA_Y
    arena_w: int = ARENA_W
    arena_h: int = ARENA_H

    export_video: bool = False
    output_dir: str = ""
