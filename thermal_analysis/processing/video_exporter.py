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

from .mouse_tracker import collapse_pauses, track_body
from .parser import load_raw_frames, measure_fps
from ..utils.naming import animal_from_stem

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


def _render_strip(tr: dict, w: int, h: int, fps: float, settle_s: float) -> dict:
    """Pre-render the temperature trace shown under a panel (cursor added per frame)."""
    y = tr["roi_max"]
    n = len(y)
    base = np.full((h, w, 3), 30, dtype=np.uint8)
    fin = np.isfinite(y)
    lo, hi = (np.nanpercentile(y, [1, 99]) if fin.any() else (0.0, 1.0))
    pad = max(0.3, 0.1 * (hi - lo))
    lo, hi = lo - pad, hi + pad
    xs = np.linspace(4, w - 5, n)
    ys = (h - 14) - (np.nan_to_num(y, nan=lo) - lo) / (hi - lo) * (h - 24)
    if settle_s > 0:
        x1 = int(4 + (w - 9) * min(1.0, settle_s * fps / max(n - 1, 1)))
        base[:, 4:x1] = (52, 52, 52)
    for i in np.flatnonzero(~tr["valid"]):              # untracked = red tick at the bottom
        base[h - 8:h - 2, int(xs[i])] = (60, 60, 255)
    seg = np.flatnonzero(np.diff(np.concatenate([[0], fin.astype(int), [0]])))
    for s, e in zip(seg[::2], seg[1::2]):               # tracked runs = continuous polyline
        if e - s >= 2:
            pts = np.stack([xs[s:e], ys[s:e]], axis=1).astype(np.int32)
            cv2.polylines(base, [pts], False, (110, 200, 120), 1, cv2.LINE_AA)
    cv2.putText(base, f"{hi - pad:.1f}", (w - 34, 11), _FONT, 0.32, (170, 170, 170), 1)
    cv2.putText(base, f"{lo + pad:.1f}", (w - 34, h - 16), _FONT, 0.32, (170, 170, 170), 1)
    return {"img": base, "xs": xs, "ys": ys, "n": n}


def _render_colorbar(w: int, h: int, t_min: float, t_max: float, cmap) -> np.ndarray:
    """Shared temperature scale drawn under the panels."""
    bar = np.full((h, w, 3), 30, dtype=np.uint8)
    x0, x1 = 60, w - 60
    grad = np.tile(np.linspace(0, 1, x1 - x0, dtype=np.float32), (h - 12, 1))
    rgb = (cmap(grad)[:, :, :3] * 255).astype(np.uint8)
    bar[3:h - 9, x0:x1] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    for frac in (0, 0.25, 0.5, 0.75, 1):
        x = int(x0 + frac * (x1 - x0 - 1))
        cv2.putText(bar, f"{t_min + frac * (t_max - t_min):.1f}", (x - 12, h - 1),
                    _FONT, 0.3, (200, 200, 200), 1)
    cv2.putText(bar, "deg C", (8, h // 2 + 4), _FONT, 0.38, (200, 200, 200), 1)
    return bar


def export_video(
    file_paths: List[str],
    output_path: str,
    settings,
    progress_callback: Optional[Callable[[int], None]] = None,
    labels: Optional[List[str]] = None,
    group_name: str = "",
    date_str: str = "",
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
    body_mode = settings.roi_mode != "static"
    strip_h = 64 if body_mode else 0      # live temperature trace under each panel
    cbar_h = 26                           # shared colour scale
    canvas_h = header_h + panel_h + strip_h + cbar_h

    # Play back in real time: use the rate measured from the recordings' own
    # timestamps (the nominal setting can be far off, e.g. 6 Hz vs 10 Hz).
    nominal = settings.fps if settings.fps > 0 else 10.0
    fps_out = float(np.median([measure_fps(fp, nominal) for fp in file_paths]))
    log.info("Video frame rate: %.2f fps (nominal setting %.1f)", fps_out, nominal)

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
    frame_times = []     # real per-frame times (a recorder pause shows up as a jump)
    for fp in file_paths:
        loaded = load_raw_frames(fp, settings, return_times=True)
        stack, ftimes = loaded if loaded is not None else (None, None)
        if stack is None:
            log.error("Failed to load frames from %s", fp)
            writer.release()
            return False
        stacks.append(stack)
        frame_times.append(ftimes)

    n_frames = min(s.shape[0] for s in stacks)
    log.info("Exporting %d frames × %d cameras → %s", n_frames, n_cams, output_path)

    # Body tracking per camera (same tracker as the analysis) for the overlay
    trackers = None
    strips = None
    if body_mode:
        trackers = [track_body(s.astype(np.float64),
                               collapse_pauses(ft)[0] if ft is not None else np.arange(s.shape[0]) / fps_out,
                               fps_out)
                    for s, ft in zip(stacks, frame_times)]
        strips = [_render_strip(tr, panel_w, strip_h, fps_out, settings.settle_s)
                  for tr in trackers]
    cbar = _render_colorbar(canvas_w, cbar_h, settings.t_min, settings.t_max,
                            _resolve_colormap(settings.colormap))

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
    ref_times = None
    if frame_times and frame_times[0] is not None:
        ref_times = collapse_pauses(frame_times[0])[0]

    for frame_idx in range(n_frames):
        # Reset header and image areas in-place (no new allocation)
        canvas[:header_h, :] = (30, 30, 30)
        canvas[header_h:, :] = 0
        canvas[header_h + panel_h + strip_h:, :] = cbar

        # Recording clock with recorder pauses cut out (continues as if nothing happened)
        t_real = float(ref_times[frame_idx]) if ref_times is not None else frame_idx / fps_out
        elapsed = int(t_real)
        header_text = "  |  ".join(filter(None, [
            "Glod", group_name, date_str,
            f"t {elapsed // 60:02d}:{elapsed % 60:02d}",
            "SETTLING (excluded)" if t_real < settings.settle_s else "",
            f"Frame {frame_idx + 1:04d} / {n_frames:04d}",
        ]))
        cv2.putText(canvas, header_text, (10, 24), _FONT, 0.55, (200, 210, 205), 1,
                    cv2.LINE_AA)

        for cam_idx, stack in enumerate(stacks):
            # Avoid copy when dtype already matches (stacks from load_raw_frames are float32)
            frame = stack[frame_idx].astype(np.float32, copy=False)
            img_bgr = _apply_colormap(frame, t_min, t_max, cmap)
            img_panel = cv2.resize(img_bgr, (panel_w, panel_h),
                                   interpolation=cv2.INTER_LINEAR)

            # ── ROI box / tracked body ────────────────────────────────────
            tr = trackers[cam_idx] if trackers else None
            if tr is not None:
                ok = bool(tr["valid"][frame_idx])
                cr, cc = tr["cent_row"][frame_idx], tr["cent_col"][frame_idx]
                if np.isfinite(cr):
                    # green = used for analysis, red = rejected frame (not in the trace)
                    col = (80, 220, 80) if ok else (60, 60, 255)
                    cx, cy = int(cc * scale_x), int(cr * scale_y)
                    cv2.drawMarker(img_panel, (cx, cy), col, cv2.MARKER_CROSS, 14, 2)
                    mk = cv2.resize(tr["masks"][frame_idx].astype(np.uint8), (panel_w, panel_h),
                                    interpolation=cv2.INTER_NEAREST)
                    cnts, _ = cv2.findContours(mk, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img_panel, cnts, -1, col, 1, cv2.LINE_AA)
                if ok:
                    hx = int((tr["max_col"][frame_idx] + 0.5) * scale_x)
                    hy = int((tr["max_row"][frame_idx] + 0.5) * scale_y)
                    cv2.circle(img_panel, (hx, hy), 6, (255, 255, 255), 1, cv2.LINE_AA)
            elif roi_mode == "static":
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
            if tr is not None:
                v = tr["roi_max"][frame_idx]
                label = f"{v:.1f}C" if np.isfinite(v) else "body not tracked"
            elif roi_mode == "static":
                label = f"{float(frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w].max()):.1f}C"
            else:
                label = f"{float(frame.max()):.1f}C"
            cv2.putText(img_panel, label,
                        (6, panel_h - 8), _FONT, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            cam_label = (labels[cam_idx] if labels and cam_idx < len(labels)
                         else animal_from_stem(Path(file_paths[cam_idx]).stem))
            cv2.putText(img_panel, cam_label, (6, 20), _FONT, 0.6,
                        (255, 255, 255), 2, cv2.LINE_AA)

            # ── Paste panel into canvas ───────────────────────────────────
            x_off = cam_idx * panel_w
            canvas[header_h:header_h + panel_h, x_off:x_off + panel_w] = img_panel
            if strips is not None:
                st = strips[cam_idx]
                sc = st["img"].copy()
                cx_ = int(st["xs"][min(frame_idx, st["n"] - 1)])
                cv2.line(sc, (cx_, 0), (cx_, strip_h - 1), (255, 255, 255), 1)
                canvas[header_h + panel_h:header_h + panel_h + strip_h,
                       x_off:x_off + panel_w] = sc

        writer.write(canvas)

        if progress_callback and frame_idx % max(1, n_frames // 100) == 0:
            progress_callback(int(frame_idx / n_frames * 100))

    writer.release()
    if progress_callback:
        progress_callback(100)
    log.info("Video export complete: %s", output_path)
    return True
