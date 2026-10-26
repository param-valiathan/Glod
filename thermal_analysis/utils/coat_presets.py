"""
Animal / coat presets for the emissivity correction.

What the literature does — and does not — support (searched Oct 2026):

* Infrared thermography of rodent fur is commonly done with ε ≈ 0.94–0.98
  (e.g. ε = 0.98 in a C57BL/6NCrl study, 0.94 for fur and 0.98 for skin in another);
  older work assumed 0.95–1.00 for mammal pelage.
* McGowan et al. 2018 (Quant. InfraRed Thermography J., doi:10.1080/17686733.2018.1437239)
  measured pelage of 22 mammal species: mean ε = 0.86 ± 0.01, unrelated to taxonomy or hair
  length/diameter, and concluded that a single value (0.98) is unlikely to be accurate.
* No study found reports ε separately for black (C57BL/6), white (BALB/c, albino) or brown/agouti
  coats.  Visible colour does not set emissivity in the 8–14 µm band, so colour-specific
  presets would be invented.  What *is* documented to change fur emissivity: diet (Marks et al.,
  Dietary effects on pelage emissivity in mammals), dirt/wetness (lowers it), viewing angle, and
  bare skin versus fur (skin ≈ 0.98).
* Fur ε therefore enters as an uncertain *offset*: for a 30.5 °C surface in a 22 °C room,
  ε = 0.86 reads 0.85 °C warmer than ε = 0.95 and ε = 0.98 reads 0.25 °C cooler.  The offset is
  nearly constant, so it matters for absolute temperatures and cross-strain comparisons, hardly
  for within-animal changes.

The cameras' own EMISSIVITY register (``senxor`` field) reads 95 % on all three Waveshare MI48x3
units and the app never changes it, so recorded temperatures already assume ε = 0.95; a preset of
0.95 therefore leaves the data unchanged.
"""

from __future__ import annotations

SENSOR_EMISSIVITY = 0.95      # MI48 EMISSIVITY register (reset value), read from the cameras
CUSTOM_LABEL = "Custom value…"

_FUR_NOTE = ("Fur in the 8–14 µm band: literature range 0.86–0.98 (typically 0.94–0.98); "
             "coat colour has no published effect — adjust only if you calibrated your own set-up.")
_SKIN_NOTE = "Bare skin ≈ 0.98 (nude, hairless or shaved animals)."

# (label, emissivity, note)
PRESETS = [
    ("Mouse · C57BL/6 (black)",              0.95, _FUR_NOTE),
    ("Mouse · BALB/c / albino (white)",      0.95, _FUR_NOTE),
    ("Mouse · agouti / brown (129, CBA, C3H)", 0.95, _FUR_NOTE),
    ("Mouse · nude / shaved (bare skin)",    0.98, _SKIN_NOTE),
    ("Rat · pigmented (Long-Evans, brown)",  0.95, _FUR_NOTE),
    ("Rat · albino (Sprague-Dawley, Wistar)", 0.95, _FUR_NOTE),
    ("Rat · hairless / shaved (bare skin)",  0.98, _SKIN_NOTE),
    ("Conservative fur estimate (McGowan 2018 mean)", 0.86,
     "Measured mean pelage emissivity over 22 mammal species (0.86 ± 0.01)."),
]
DEFAULT_PRESET = PRESETS[0][0]

_BY_LABEL = {label: (eps, note) for label, eps, note in PRESETS}


def emissivity_for(label: str, custom: float | None = None) -> float:
    """Emissivity for a preset label (or the custom value for ``CUSTOM_LABEL``)."""
    if label == CUSTOM_LABEL:
        return float(custom) if custom is not None else _BY_LABEL[DEFAULT_PRESET][0]
    return _BY_LABEL.get(label, _BY_LABEL[DEFAULT_PRESET])[0]


def note_for(label: str) -> str:
    return _BY_LABEL.get(label, ("", "Your own emissivity value."))[1]
