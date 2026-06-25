"""
Camera discovery for Waveshare MI48x3 USB thermal cameras.

Three-layer fallback:
  1. pysenxor-lite list_senxor("serial") — library-native, filters by VID/PID internally
  2. pyserial description-string + known-VID filter — catches firmware variants
  3. Returns [] → caller opens manual selection dialog
"""

from __future__ import annotations

import logging

log = logging.getLogger(__name__)

KNOWN_DESCRIPTIONS = ("senxor", "mi48", "waveshare", "meridian", "thermal")


def list_cameras(known_vids: tuple = ()) -> list[dict]:
    """
    Return a list of detected thermal camera ports.

    Each entry: {"id": "COM3", "label": "SenXor (COM3)", "source": "auto|filter"}

    Returns [] when nothing is found — the caller should open a manual
    selection dialog (Layer 3).
    """
    # Layer 1 — pysenxor-lite
    # list_senxor("serial") returns SerialPort objects; extract .device for the
    # string id ("COM6") that is used as dict key, QSettings value, and UI label.
    # CaptureWorker.run() calls list_senxor() again to get the object for connect().
    try:
        from senxor import list_senxor  # type: ignore[import]
        ports = list_senxor("serial")
        if ports:
            log.info("Layer 1 found %d camera(s): %s", len(ports), ports)
            return [
                {"id": p.device, "label": f"SenXor ({p.device})", "source": "auto"}
                for p in ports
            ]
    except Exception as exc:
        log.debug("Layer 1 (list_senxor) failed: %s", exc)

    # Layer 2 — pyserial description / VID filter
    candidates = _fallback_list(known_vids)
    if candidates:
        log.info("Layer 2 found %d candidate(s): %s", len(candidates), candidates)
        return [{"id": p, "label": f"Thermal? ({p})", "source": "filter"} for p in candidates]

    log.info("No cameras found via auto-detection.")
    return []


def _fallback_list(known_vids: tuple = ()) -> list[str]:
    try:
        import serial.tools.list_ports  # type: ignore[import]
    except ImportError:
        log.warning("pyserial not installed — Layer 2 unavailable.")
        return []

    candidates: list[str] = []
    for p in serial.tools.list_ports.comports():
        desc = (p.description or "").lower()
        if any(k in desc for k in KNOWN_DESCRIPTIONS):
            candidates.append(p.device)
        elif known_vids and p.vid in known_vids:
            candidates.append(p.device)
    return candidates


def probe(port_id: str) -> bool:
    """
    Attempt a brief connection to confirm the port responds as a SenXor camera.
    Returns True on success, False on any error.
    """
    try:
        from senxor import list_senxor, connect  # type: ignore[import]
        all_ports = list_senxor("serial")
        port_obj = next((p for p in all_ports if p.device == port_id), None)
        if port_obj is None:
            return False
        dev = connect(port_obj)
        dev.start_stream()
        header, frame = dev.read(block=False)
        dev.close()
        return frame is not None
    except Exception as exc:
        log.debug("Probe failed on %s: %s", port_id, exc)
        return False


def list_all_serial_ports() -> list[dict]:
    """Return all COM ports for the manual selection dialog."""
    try:
        import serial.tools.list_ports  # type: ignore[import]
        return [
            {"id": p.device, "label": f"{p.device} — {p.description or 'Unknown'}"}
            for p in sorted(serial.tools.list_ports.comports(), key=lambda x: x.device)
        ]
    except ImportError:
        return []
