"""
Single source of truth for every name Glöd writes to disk.

Convention (all outputs):  <YYYY-MM-DD>_<group>_<animal>_<kind>.<ext>
Group-level files omit <animal>; run-level files omit <group> too.
Recordings are stored as   recordings/<group>/<animal>/<animal>_<YYYYMMDD_HHMMSS>.txt
Qt-free so it can be imported from worker processes.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Optional, Tuple

_TS_RE = re.compile(r"_(\d{8})_(\d{6})$")


def safe(name: str) -> str:
    """Filesystem-safe token: letters, digits, '-' and '_' only."""
    out = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(name))
    return out.strip("_") or "unnamed"


def split_stem(stem: str) -> Tuple[str, Optional[str]]:
    """'Mouse_01_20261026_143012' -> ('Mouse_01', '2026-10-26').

    Stems without a trailing recording timestamp return (stem, None).
    """
    m = _TS_RE.search(stem)
    if not m:
        return stem, None
    d = m.group(1)
    return stem[:m.start()], f"{d[:4]}-{d[4:6]}-{d[6:]}"


def animal_from_stem(stem: str) -> str:
    return split_stem(stem)[0]


def date_from_stem(stem: str, fallback: Optional[str] = None) -> str:
    return split_stem(stem)[1] or fallback or today()


def today() -> str:
    return datetime.now().strftime("%Y-%m-%d")


def build(kind: str, ext: str, *, date: str, group: str = "",
          animal: str = "") -> str:
    """<date>[_<group>][_<animal>]_<kind>.<ext>"""
    parts = [date] + [safe(p) for p in (group, animal) if p] + [safe(kind)]
    return "_".join(parts) + "." + ext.lstrip(".")
