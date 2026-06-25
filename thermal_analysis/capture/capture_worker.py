"""
CaptureWorker — one QThread per connected thermal camera.

Robustness design
-----------------
* Camera detection is dynamic: re-calls list_senxor() at connect time and
  matches by .device string, so COM port numbers don't need to be hardcoded.
  Works on any computer where pysenxor-lite can enumerate the cameras.
* Non-blocking reads (block=False) with a 20 ms poll so stop() is honoured
  within 20 ms even when the camera stops sending frames (e.g. USB disconnect).
* start_delay_s staggers USB bus negotiation across cameras on a shared hub.
* Retry on camera-not-found: retries list_senxor() up to CONNECT_RETRIES times
  before giving up (handles the brief window where the OS is still enumerating
  after plug-in).
* Consecutive-error limit: if dev.read() fails MAX_CONSECUTIVE_ERRORS times in
  a row the worker stops cleanly rather than spinning forever.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

log = logging.getLogger(__name__)

_RAM_WARN_BYTES      = 1_073_741_824  # 1 GB across all cameras
_POLL_INTERVAL_S     = 0.02           # 20 ms non-blocking poll interval
_CONNECT_RETRIES     = 4              # number of times to retry list_senxor()
_CONNECT_RETRY_DELAY = 1.5            # seconds between retries
_MAX_CONSECUTIVE_ERR = 10             # consecutive read failures before stopping


class CaptureWorker(QThread):
    """
    Streams frames from a single camera port.

    Signals
    -------
    frame_ready(camera_id, frame_array, elapsed_s)
        Emitted for every frame received.  frame_array is (62, 80) float32 °C.
    log_message(str)
    finished(camera_id, buffer)
        buffer is a list of (iso_ts: str, frame: ndarray, header, frame_n: int)
    error(camera_id, message)
    """

    frame_ready = pyqtSignal(str, np.ndarray, float)
    log_message  = pyqtSignal(str)
    connected    = pyqtSignal(str)        # emitted once streaming starts
    finished     = pyqtSignal(str, list)
    error        = pyqtSignal(str, str)

    def __init__(
        self,
        camera_id: str,
        target_fps: int = 5,
        duration_s: int = 300,
        start_delay_s: float = 0.0,
        port_obj=None,         # SerialPort from list_senxor(); pass from main thread to avoid races
        preview_only: bool = False,   # skip buffering — live feed without saving
        parent=None,
    ):
        super().__init__(parent)
        self._camera_id     = camera_id
        self._target_fps    = max(1, min(target_fps, 25))
        self._duration_s    = duration_s
        self._start_delay_s = start_delay_s
        self._port_obj      = port_obj
        self._preview_only  = preview_only
        self._stop_event    = threading.Event()

    # ── Public API ─────────────────────────────────────────────────────────

    def stop(self):
        self._stop_event.set()

    @property
    def camera_id(self) -> str:
        return self._camera_id

    # ── QThread run ────────────────────────────────────────────────────────

    def run(self):
        # Staggered start — interruptible 20 ms ticks
        if self._start_delay_s > 0:
            self.log_message.emit(
                f"[{self._camera_id}] Waiting {self._start_delay_s:.1f} s "
                "before connect (USB hub stagger)…")
            deadline = time.monotonic() + self._start_delay_s
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    return
                time.sleep(_POLL_INTERVAL_S)

        # ── Connect with retry ─────────────────────────────────────────────
        dev = self._connect_with_retry()
        if dev is None:
            return  # error already emitted

        # ── Set FPS via hardware divider ───────────────────────────────────
        divider = round(25 / self._target_fps)
        try:
            dev.set_frame_rate_divider(divider)
            actual_divider = dev.get_frame_rate_divider()
            actual_fps = 25 / actual_divider
            self.log_message.emit(
                f"[{self._camera_id}] FPS divider={actual_divider} "
                f"-> {actual_fps:.1f} FPS")
        except Exception as exc:
            self.log_message.emit(
                f"[{self._camera_id}] Could not set FPS divider ({exc}) "
                "— using camera default")

        # ── Start streaming ────────────────────────────────────────────────
        try:
            dev.start_stream()
        except Exception as exc:
            self.error.emit(self._camera_id, f"start_stream failed: {exc}")
            self._close(dev)
            return

        self.log_message.emit(
            f"[{self._camera_id}] Streaming for {self._duration_s} s…")
        self.connected.emit(self._camera_id)

        # ── Frame capture loop ─────────────────────────────────────────────
        t0: Optional[datetime] = None
        frame_n = 0
        buf: list = []
        consecutive_errors = 0

        while not self._stop_event.is_set():
            try:
                header, frame = dev.read(block=False)
                consecutive_errors = 0          # reset on any successful call
            except Exception as exc:
                consecutive_errors += 1
                self.log_message.emit(
                    f"[{self._camera_id}] Read error ({consecutive_errors}/"
                    f"{_MAX_CONSECUTIVE_ERR}): {exc}")
                if consecutive_errors >= _MAX_CONSECUTIVE_ERR:
                    self.error.emit(
                        self._camera_id,
                        f"Camera disconnected after {consecutive_errors} "
                        "consecutive read errors.")
                    break
                time.sleep(_POLL_INTERVAL_S)
                continue

            if frame is None:
                time.sleep(_POLL_INTERVAL_S)
                continue

            now = datetime.now(timezone.utc)
            if t0 is None:
                t0 = now

            elapsed = (now - t0).total_seconds()
            if elapsed > self._duration_s:
                break

            frame_copy = frame.copy()
            if not self._preview_only:
                buf.append((now.isoformat(), frame_copy, header, frame_n))
            frame_n += 1
            self.frame_ready.emit(self._camera_id, frame_copy, elapsed)

        self._close(dev)
        self.log_message.emit(
            f"[{self._camera_id}] Done — {frame_n} frames captured.")
        self.finished.emit(self._camera_id, buf)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _connect_with_retry(self):
        """
        Connect to self._camera_id with up to _CONNECT_RETRIES attempts.

        If a SerialPort object was passed at construction (port_obj), use it
        directly for the first attempt — this avoids concurrent list_senxor()
        calls from multiple worker threads, which was the root cause of frames
        never arriving in the Qt app.  On failure, fall back to re-enumeration.

        Returns a Senxor device or None (error already emitted).
        """
        from senxor import list_senxor, connect  # type: ignore[import]

        port_obj = self._port_obj   # may be None if not pre-enumerated

        for attempt in range(1, _CONNECT_RETRIES + 1):
            if self._stop_event.is_set():
                return None
            try:
                if port_obj is None:
                    # Re-enumerate — safe on retry since other workers are already
                    # connected by this point (stagger ensures serialised start)
                    all_ports = list_senxor("serial")
                    port_obj = next(
                        (p for p in all_ports if p.device == self._camera_id),
                        None)

                if port_obj is not None:
                    dev = connect(port_obj)
                    self.log_message.emit(
                        f"[{self._camera_id}] Connected (attempt {attempt})")
                    return dev

                self.log_message.emit(
                    f"[{self._camera_id}] Not found in list_senxor "
                    f"(attempt {attempt}/{_CONNECT_RETRIES}), "
                    f"retrying in {_CONNECT_RETRY_DELAY:.1f} s...")
                port_obj = None  # force re-enumerate next attempt

            except Exception as exc:
                self.log_message.emit(
                    f"[{self._camera_id}] Connect attempt {attempt} failed: {exc}")
                port_obj = None  # force re-enumerate next attempt

            # Interruptible wait before next attempt
            deadline = time.monotonic() + _CONNECT_RETRY_DELAY
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    return None
                time.sleep(_POLL_INTERVAL_S)

        self.error.emit(
            self._camera_id,
            f"Camera not found after {_CONNECT_RETRIES} attempts. "
            "Check it is plugged in and the driver is installed.")
        return None

    @staticmethod
    def _close(dev) -> None:
        """Stop stream and close device, ignoring errors."""
        try:
            dev.stop_stream()
        except Exception:
            pass
        try:
            dev.close()
        except Exception:
            pass


def estimate_ram_bytes(n_cameras: int, fps: int, duration_s: int) -> int:
    """Estimate buffer RAM: n_cameras × fps × duration × 62 × 80 × 4 bytes."""
    return n_cameras * fps * duration_s * 62 * 80 * 4


def ram_warning(n_cameras: int, fps: int, duration_s: int) -> Optional[str]:
    """Return a warning string if estimated RAM exceeds 1 GB, else None."""
    total = estimate_ram_bytes(n_cameras, fps, duration_s)
    if total > _RAM_WARN_BYTES:
        gb = total / 1_073_741_824
        return (
            f"Estimated capture buffer: {gb:.1f} GB "
            f"({n_cameras} camera(s) × {fps} FPS × {duration_s} s). "
            "This may exhaust system RAM. Consider reducing duration or FPS."
        )
    return None
