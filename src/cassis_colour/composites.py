# -*- coding: utf-8 -*-
"""RGB composites of a CaSSIS cube for visual inspection of colour variations.

Four composites are defined, each channel stretched between its 2nd and 98th
percentile:

==============  ======================================================
``npb``         R = NIR, G = PAN, B = BLU: the standard CaSSIS colour view
``band_depths``  R = BD_PAN, G = BD_RED, B = c3: ferric edge, ferrous
                shoulder and cubic shape term
``ratios``      R = RED/PAN, G = NIR/RED, B = c3: the ratio counterpart of
                ``band_depths``
``shape``       R = theta, G = phi, B = c3: direction of the CLR shape vector
==============  ======================================================

``composites_from_cube`` returns the composites as ``(lines, samples, 3)``
float32 arrays in [0, 1]; ``save_composites`` writes them as PNG files plus a
four-panel overview and needs matplotlib (the ``plot`` extra).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import parameters as sp

COMPOSITES = {
    "npb": "NPB colour (R = NIR, G = PAN, B = BLU)",
    "band_depths": "Band depths (R = BD_PAN, G = BD_RED, B = c3)",
    "ratios": "Band ratios (R = RED/PAN, G = NIR/RED, B = c3)",
    "shape": "CLR shape (R = theta, G = phi, B = c3)",
}
"""Composite names, in output order, with the channel assignments."""


def stretch_percentile(arr, lo=2.0, hi=98.0) -> np.ndarray:
    """Linear stretch of ``arr`` between its ``lo`` and ``hi`` percentiles to [0, 1].

    Percentiles are taken over the finite values; NaN becomes 0. An array
    without finite values or without contrast returns zeros.
    """
    a = np.asarray(arr, dtype=np.float64)
    finite = np.isfinite(a)
    if not finite.any():
        return np.zeros(a.shape, dtype=np.float32)
    vlo, vhi = np.percentile(a[finite], [lo, hi])
    if vhi <= vlo:
        return np.zeros(a.shape, dtype=np.float32)
    x = (a - vlo) / (vhi - vlo)
    return np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0).astype(np.float32)


def make_rgb(r, g, b, lo=2.0, hi=98.0) -> np.ndarray:
    """Stack three planes into a stretched ``(..., 3)`` RGB array."""
    return np.dstack([stretch_percentile(r, lo, hi),
                      stretch_percentile(g, lo, hi),
                      stretch_percentile(b, lo, hi)])


def composites_from_bands(bands, lo=2.0, hi=98.0) -> dict:
    """Return ``{name: rgb}`` for the four composites from a dict of filter planes."""
    BLU, PAN, RED, NIR = bands["BLU"], bands["PAN"], bands["RED"], bands["NIR"]
    bd_pan = sp.band_depth_PAN(BLU, PAN, RED, NIR)
    bd_red = sp.band_depth_RED(BLU, PAN, RED, NIR)
    c3 = sp.legendre(BLU, PAN, RED, NIR)["c3"]
    ang = sp.simplex_angles(BLU, PAN, RED, NIR)
    rat = sp.band_ratios(BLU, PAN, RED, NIR)
    return {
        "npb": make_rgb(NIR, PAN, BLU, lo, hi),
        "band_depths": make_rgb(bd_pan, bd_red, c3, lo, hi),
        "ratios": make_rgb(rat["ratio_RED_PAN"], rat["ratio_NIR_RED"], c3, lo, hi),
        "shape": make_rgb(ang["theta"], ang["phi"], c3, lo, hi),
    }


def composites_from_cube(cube, lo=2.0, hi=98.0) -> dict:
    """Composites for a :class:`cassis_colour.reader.CassisCube`."""
    return composites_from_bands(cube.bands, lo, hi)


def save_composites(comps, out_dir, image_id, panel=True, dpi=130) -> list:
    """Write each composite as ``<out_dir>/<image_id>_<name>.png``.

    With ``panel`` a two-by-two overview ``<image_id>_composites.png`` is
    written as well. Returns the list of written paths. Requires matplotlib.
    """
    from matplotlib.figure import Figure
    from matplotlib.image import imsave

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, rgb in comps.items():
        p = out_dir / f"{image_id}_{name}.png"
        imsave(p, np.clip(rgb, 0.0, 1.0))
        written.append(p)
    if panel:
        names = [n for n in COMPOSITES if n in comps]
        fig = Figure(figsize=(13, 13))
        axes = fig.subplots(2, 2)
        for ax, name in zip(axes.ravel(), names):
            ax.imshow(comps[name], interpolation="nearest")
            ax.set_title(COMPOSITES[name], fontsize=12, fontweight="bold")
            ax.set_axis_off()
        for ax in axes.ravel()[len(names):]:
            ax.set_axis_off()
        fig.suptitle(f"CaSSIS composites: {image_id}", fontsize=15, y=0.99)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        p = out_dir / f"{image_id}_composites.png"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        written.append(p)
    return written
