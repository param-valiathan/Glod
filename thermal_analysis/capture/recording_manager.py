"""
RecordingManager — coordinates multiple CaptureWorkers and saves recordings.

Lives on the main thread.  SaveWorker runs disk I/O in a separate QThread
so the GUI stays responsive during file writing.

Saved file format matches the Waveshare EVK Viewer output exactly:
  col 0      : ISO 8601 timestamp (with timezone)
  cols 1–7   : numeric metadata (from header where available, else zeros)
  cols 8–4967: pixel temperatures °C, row-major (80 × 62 = 4960 values)

Start sequencing
----------------
Call prepare() first to create workers, then connect any additional signals
(e.g. frame_ready → CameraPanel.on_frame), then call start() to launch
threads.  This avoids a race where camera 0 (no start delay) emits
frame_ready before external slots are connected.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from .capture_worker import CaptureWorker

log = logging.getLogger(__name__)

# Seconds between each successive camera start — staggers USB bus negotiation.
_STAGGER_S: float = 0.3


# ── Save worker ────────────────────────────────────────────────────────────────

class SaveWorker(QThread):
    """Writes one .txt file per camera buffer in a background thread."""

    log_message = pyqtSignal(str)
    finished    = pyqtSignal(list)   # list of saved file paths
    error       = pyqtSignal(str)

    def __init__(self, jobs: list[dict], parent=None):
        """
        jobs: list of {
          "buf":       [(iso_ts, frame_ndarray, header, frame_n), ...],
          "save_path": Path,
        }
        """
        super().__init__(parent)
        self._jobs = jobs

    def run(self):
        saved = []
        for job in self._jobs:
            path: Path = job["save_path"]
            buf: list = job["buf"]
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                self.log_message.emit(f"Saving {path.name} ({len(buf)} frames)…")
                _write_file(path, buf)
                saved.append(str(path))
                self.log_message.emit(f"✓ Saved {path}")
            except Exception as exc:
                self.error.emit(f"Failed to save {path}: {exc}")
        self.finished.emit(saved)


def _write_file(path: Path, buf: list) -> None:
    """Write one recording CSV in EVK Viewer format."""
    rows = []
    for iso_ts, frame, header, frame_n in buf:
        meta = _extract_meta(header, frame_n)
        pixels = frame.flatten()  # (4960,) row-major
        row = (
            f"{iso_ts},"
            + ",".join(str(m) for m in meta)
            + ","
            + ",".join(f"{v:.4f}" for v in pixels)
        )
        rows.append(row)
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _extract_meta(header, frame_n: int) -> list:
    """
    Return [frame_n, voltage, ambient_t, hw_ts, sens, filt, ref_t] from the
    pysenxor-lite header uint16 array.  Falls back to zeros for unknown fields.
    """
    if header is None:
        return [frame_n, 0, 0, 0, 0, 0, 0]
    try:
        h = np.asarray(header, dtype=np.uint16)
        return [
            int(h[0]) if len(h) > 0 else frame_n,
            int(h[1]) if len(h) > 1 else 0,
            int(h[2]) if len(h) > 2 else 0,
            int(h[3]) if len(h) > 3 else 0,
            int(h[4]) if len(h) > 4 else 0,
            int(h[5]) if len(h) > 5 else 0,
            int(h[6]) if len(h) > 6 else 0,
        ]
    except Exception:
        return [frame_n, 0, 0, 0, 0, 0, 0]


# ── Recording manager ──────────────────────────────────────────────────────────

class RecordingManager(QObject):
    """
    Owns all CaptureWorker instances for a recording session.

    Usage:
        manager.prepare(configs, fps, duration_s, output_dir)
        # connect manager._workers[cam_id].frame_ready here
        manager.start()
    """

    log_message  = pyqtSignal(str)
    all_saved    = pyqtSignal(list)   # list of saved file paths
    save_error   = pyqtSignal(str)
    camera_error = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._workers: dict[str, CaptureWorker] = {}
        self._buffers: dict[str, list] = {}
        self._animal_map: dict[str, object] = {}
        self._output_dir: str = ""
        self._timestamp: str = ""
        self._save_worker: Optional[SaveWorker] = None
        self._expected: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def prepare(
        self,
        animal_configs: list,
        fps: int,
        duration_s: int,
        output_dir: str,
    ) -> None:
        """
        Create workers with staggered start delays (0.3 s per camera).
        Does NOT start threads — call start() after connecting frame signals.
        """
        self._output_dir = output_dir
        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._buffers.clear()
        self._workers.clear()
        self._animal_map = {ac.camera_id: ac for ac in animal_configs}
        self._expected = len(animal_configs)

        # Enumerate ports ONCE on the main thread before starting any workers.
        # Calling list_senxor() from multiple threads simultaneously causes a
        # race in pysenxor-lite's global state and is the root cause of cameras
        # connecting but never emitting frames in the Qt app.
        port_map: dict = {}
        try:
            from senxor import list_senxor  # type: ignore[import]
            all_ports = list_senxor("serial")
            port_map = {p.device: p for p in all_ports}
            found = [ac.camera_id for ac in animal_configs if ac.camera_id in port_map]
            missing = [ac.camera_id for ac in animal_configs if ac.camera_id not in port_map]
            if found:
                self.log_message.emit(f"Pre-enumerated ports: {found}")
            if missing:
                self.log_message.emit(
                    f"Warning: cameras not found during pre-enum (will retry): {missing}")
        except Exception as exc:
            self.log_message.emit(
                f"list_senxor() failed on main thread ({exc}); "
                "workers will enumerate individually.")

        for i, ac in enumerate(animal_configs):
            delay = i * _STAGGER_S
            w = CaptureWorker(
                ac.camera_id, fps, duration_s,
                start_delay_s=delay,
                port_obj=port_map.get(ac.camera_id),  # pre-enumerated SerialPort or None
                parent=self,
            )
            w.log_message.connect(self.log_message)
            w.frame_ready.connect(self._on_frame_ready)
            w.finished.connect(self._on_worker_finished)
            w.error.connect(self._on_worker_error)
            self._workers[ac.camera_id] = w

        self.log_message.emit(
            f"Prepared {len(animal_configs)} camera(s): "
            f"{fps} FPS, {duration_s} s, "
            f"stagger {_STAGGER_S:.1f} s/camera"
        )

    def start(self) -> None:
        """Start all prepared workers.  Call prepare() first."""
        for w in self._workers.values():
            w.start()
        self.log_message.emit("Recording started.")

    def stop_all(self) -> None:
        for w in self._workers.values():
            w.stop()

    def is_running(self) -> bool:
        return any(w.isRunning() for w in self._workers.values())

    # ── Internal slots ─────────────────────────────────────────────────────

    def _on_frame_ready(self, camera_id: str, frame: np.ndarray, elapsed: float):
        pass  # frames forwarded via CaptureWorker.frame_ready directly to CameraPanel

    def _on_worker_finished(self, camera_id: str, buf: list):
        self._buffers[camera_id] = buf
        self.log_message.emit(
            f"[{camera_id}] capture done — {len(buf)} frames buffered.")
        if len(self._buffers) == self._expected:
            self._launch_save()

    def _on_worker_error(self, camera_id: str, msg: str):
        self.camera_error.emit(camera_id, msg)
        self.log_message.emit(f"✗ [{camera_id}] {msg}")
        self._buffers.setdefault(camera_id, [])
        if len(self._buffers) == self._expected:
            self._launch_save()

    def _launch_save(self):
        jobs = []
        for camera_id, buf in self._buffers.items():
            if not buf:
                continue
            ac = self._animal_map.get(camera_id)
            if ac is None:
                continue
            safe_group  = _safe(ac.group_name)
            safe_animal = _safe(ac.animal_name)
            fname = f"{safe_animal}_{self._timestamp}.txt"
            save_path = (
                Path(self._output_dir)
                / "recordings"
                / safe_group
                / safe_animal
                / fname
            )
            jobs.append({"buf": buf, "save_path": save_path})

        if not jobs:
            self.log_message.emit("No data to save.")
            self.all_saved.emit([])
            return

        self._save_worker = SaveWorker(jobs, parent=self)
        self._save_worker.log_message.connect(self.log_message)
        self._save_worker.finished.connect(self.all_saved)
        self._save_worker.error.connect(self.save_error)
        self._save_worker.start()


def _safe(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in name).strip("_") or "unnamed"
