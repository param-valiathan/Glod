"""
Multi-panel thermal video exporter for Glöd.

Creates a side-by-side panel video (one panel per camera file) with:
  - Thermal colormap rendering scaled to user-defined T range
  - ROI bounding box overlay
  - Frame counter and ROI max temperature overlay
  - Synchronized across all cameras (min common frame count)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, List, Optional

import cv2
import numpy as np

from .parser import load_raw_frames

log = logging.getLogger(__name__)

# Font used for OpenCV text overlays
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _resolve_colormap(name: str):
    """Resolve a matplotlib colormap by name once; cache the callable."""
    import matplotlib
    try:
        return matplotlib.colormaps[name]
    except KeyError:
        return matplotlib.colormaps["inferno"]


def _apply_colormap(frame_hw: np.ndarray, t_min: float, t_max: float,
                    cmap) -> np.ndarray:
    """
    Convert a single (H, W) float frame to an (H, W, 3) BGR uint8 image.
    ``cmap`` must be a pre-resolved matplotlib colormap callable.
    """
    rng = t_max - t_min
    if rng == 0:
        norm = np.zeros(frame_hw.shape, dtype=np.float32)
    else:
        norm = np.clip((frame_hw - t_min) / rng, 0.0, 1.0).astype(np.float32)

    rgba = cmap(norm)                               # (H, W, 4) float64
    rgb  = (rgba[:, :, :3] * 255).astype(np.uint8) # (H, W, 3) uint8 RGB
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def export_video(
    file_paths: List[str],
    output_path: str,
    settings,
    progress_callback: Optional[Callable[[int], None]] = None,
) -> bool:
    """
    Export a multi-panel thermal video to ``output_path``.

    Parameters
    ----------
    file_paths      : ordered list of .txt paths (one panel each)
    output_path     : destination .mp4 path
    settings        : AnalysisSettings
    progress_callback : called with int 0-100

    Returns True on success, False on error.
    """
    if not file_paths:
        log.error("No files provided for video export.")
        return False

    n_cams = len(file_paths)
    panel_w = 320
    panel_h = int(panel_w * settings.camera_height / settings.camera_width)
    header_h = 36
    canvas_w = panel_w * n_cams
    canvas_h = panel_h + header_h

    fps_out = settings.fps if settings.fps > 0 else 10.0

    # ── Try to open VideoWriter ───────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps_out,
                              (canvas_w, canvas_h))
    if not writer.isOpened():
        log.warning("mp4v codec unavailable, falling back to XVID.")
        out_path_avi = str(output_path).replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(out_path_avi, fourcc, fps_out,
                                  (canvas_w, canvas_h))
        if not writer.isOpened():
            log.error("Could not open any VideoWriter codec.")
            return False
        output_path = out_path_avi

    # ── Load all camera stacks ────────────────────────────────────────────
    stacks = []
    for fp in file_paths:
        stack = load_raw_frames(fp, settings)
        if stack is None:
            log.error("Failed to load frames from %s", fp)
            writer.release()
            return False
        stacks.append(stack)

    n_frames = min(s.shape[0] for s in stacks)
    log.info("Exporting %d frames × %d cameras → %s", n_frames, n_cams, output_path)

    t_min = settings.t_min
    t_max = settings.t_max
    # Resolve colormap once — avoids dict lookup on every frame × camera
    cmap  = _resolve_colormap(settings.colormap)

    roi_mode = settings.roi_mode
    roi_x, roi_y = settings.roi_x, settings.roi_y
    roi_w, roi_h = settings.roi_w, settings.roi_h

    h_orig = settings.camera_height
    w_orig = settings.camera_width
    scale_x = panel_w / w_orig
    scale_y = panel_h / h_orig

    # Scaled static ROI coordinates (computed once)
    sx0 = int(roi_x * scale_x)
    sy0 = int(roi_y * scale_y)
    sx1 = int((roi_x + roi_w) * scale_x)
    sy1 = int((roi_y + roi_h) * scale_y)

    # Pre-allocate canvas once; only image area is zeroed per frame
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

    for frame_idx in range(n_frames):
        # Reset header and image areas in-place (no new allocation)
        canvas[:header_h, :] = (30, 30, 30)
        canvas[header_h:, :] = 0

        header_text = f"Glod  |  Frame {frame_idx + 1:04d} / {n_frames:04d}"
        cv2.putText(canvas, header_text, (10, 24), _FONT, 0.55, (200, 210, 205), 1,
                    cv2.LINE_AA)

        for cam_idx, stack in enumerate(stacks):
            # Avoid copy when dtype already matches (stacks from load_raw_frames are float32)
            frame = stack[frame_idx].astype(np.float32, copy=False)
            img_bgr = _apply_colormap(frame, t_min, t_max, cmap)
            img_panel = cv2.resize(img_bgr, (panel_w, panel_h),
                                   interpolation=cv2.INTER_LINEAR)

            # ── ROI box ───────────────────────────────────────────────────
            if roi_mode == "static":
                cv2.rectangle(img_panel, (sx0, sy0), (sx1, sy1), (255, 255, 255), 2)
            else:
                # Dynamic: find hottest pixel in this frame → draw ROI
                # ravel() returns a view (no copy) for C-contiguous frames
                flat = frame.ravel()
                hot_idx = int(flat.argmax())
                hr, hc = hot_idx // w_orig, hot_idx % w_orig
                rh2, rw2 = roi_h // 2, roi_w // 2
                r0 = max(0, hr - rh2); r1 = min(h_orig, r0 + roi_h)
                c0 = max(0, hc - rw2); c1 = min(w_orig, c0 + roi_w)
                dsx0 = int(c0 * scale_x); dsy0 = int(r0 * scale_y)
                dsx1 = int(c1 * scale_x); dsy1 = int(r1 * scale_y)
                cv2.rectangle(img_panel, (dsx0, dsy0), (dsx1, dsy1),
                               (255, 255, 255), 2)

            # ── Temperature annotation ────────────────────────────────────
            if roi_mode == "static":
                roi_temp = float(frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].max())
            else:
                roi_temp = float(frame.max())
            cv2.putText(img_panel, f"{roi_temp:.1f}C",
                        (6, panel_h - 8), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cam_label = Path(file_paths[cam_idx]).stem[-12:]
            cv2.putText(img_panel, cam_label, (6, 18), _FONT, 0.4,
                        (200, 210, 205), 1, cv2.LINE_AA)

            # ── Paste panel into canvas ───────────────────────────────────
            x_off = cam_idx * panel_w
            canvas[header_h:, x_off:x_off + panel_w] = img_panel

        writer.write(canvas)

        if progress_callback and frame_idx % max(1, n_frames // 100) == 0:
            progress_callback(int(frame_idx / n_frames * 100))

    writer.release()
    if progress_callback:
        progress_callback(100)
    log.info("Video export complete: %s", output_path)
    return True
