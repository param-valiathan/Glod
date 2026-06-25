"""
Run this script once with all thermal cameras plugged in to discover their
USB VID:PID values. Update KNOWN_CAMERA_VIDS in thermal_analysis/utils/config.py
and CLAUDE.md with the confirmed values.

Usage:
    launch_glod.bat  (or run via setup_glod_env.bat environment)
"""

import sys

try:
    import serial.tools.list_ports
except ImportError:
    print("pyserial not installed. Run setup_glod_env.bat first.")
    sys.exit(1)

print(f"\n{'PORT':<10} {'VID':<8} {'PID':<8} {'DESCRIPTION'}")
print("-" * 70)
ports = list(serial.tools.list_ports.comports())
if not ports:
    print("No COM ports found. Is the camera plugged in?")
else:
    for p in sorted(ports, key=lambda x: x.device):
        vid = f"{p.vid:#06x}" if p.vid is not None else "N/A"
        pid = f"{p.pid:#06x}" if p.pid is not None else "N/A"
        print(f"{p.device:<10} {vid:<8} {pid:<8} {p.description or ''}")

print()

try:
    from senxor import list_senxor
    found = list_senxor("serial")
    print(f"pysenxor-lite detected: {found if found else '(none)'}")
except Exception as e:
    print(f"pysenxor-lite list_senxor failed: {e}")
