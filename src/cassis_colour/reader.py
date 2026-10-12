# -*- coding: utf-8 -*-
"""Read CaSSIS ISIS3 cubes (``*.cubeit.cub``) into named filter planes.

CaSSIS products distributed through the ESA Planetary Science Archive are
ISIS3 cubes with one band per filter. Four-band colour cubes hold the filters
in the order NIR, RED, PAN, BLU; three-band "NPB" cubes hold NIR, PAN, BLU.
The ``FilterName`` entries of the ISIS3 ``BandBin`` group are exposed by GDAL
as band descriptions, which this reader uses to map bands to filters; when
they are missing it falls back to that positional order.

Cubes are read with rasterio (GDAL's ISIS3 driver), which handles both the
band-sequential and the tiled core formats and supports decimated reads, so a
large cube can be inspected at reduced resolution without loading it fully.
``is_complete`` checks a download against the core size declared in the
label, because GDAL fills missing data of a truncated cube with zeros instead
of raising an error.
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .parameters import FILTERS, WAVELENGTHS_NM  # noqa: F401  (re-exported)

POSITIONAL_ORDER = {
    4: ("NIR", "RED", "PAN", "BLU"),
    3: ("NIR", "PAN", "BLU"),
    2: ("PAN", "BLU"),
    1: ("PAN",),
}
"""Band order assumed when a cube carries no filter names."""

_PIXEL_BYTES = {
    "UNSIGNEDBYTE": 1, "SIGNEDWORD": 2, "UNSIGNEDWORD": 2,
    "SIGNEDINTEGER": 4, "UNSIGNEDINTEGER": 4, "REAL": 4, "DOUBLE": 8,
}

_LABEL_HEAD_BYTES = 1 << 20


# --------------------------------------------------------------------------- #
# Label parsing and completeness
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CoreLayout:
    """Byte layout of the ISIS3 cube core, read from the label."""

    start_byte: int          # 1-based offset of the first core byte
    format: str              # "BandSequential" or "Tile"
    samples: int
    lines: int
    bands: int
    pixel_type: str          # ISIS3 pixel type, e.g. "Real"
    bytes_per_pixel: int
    tile_samples: int | None
    tile_lines: int | None
    core_file: str | None    # detached core file name, if any

    @property
    def core_bytes(self) -> int:
        if self.format.lower() == "tile" and self.tile_samples and self.tile_lines:
            ns = math.ceil(self.samples / self.tile_samples) * self.tile_samples
            nl = math.ceil(self.lines / self.tile_lines) * self.tile_lines
        else:
            ns, nl = self.samples, self.lines
        return ns * nl * self.bands * self.bytes_per_pixel

    @property
    def end_byte(self) -> int:
        """1-based offset of the last core byte."""
        return self.start_byte - 1 + self.core_bytes


def _label_value(block: str, key: str) -> str | None:
    m = re.search(r"^\s*" + re.escape(key) + r"\s*=\s*\"?([^\s\"<]+)", block, re.M | re.I)
    return m.group(1) if m else None


def core_layout(path) -> CoreLayout:
    """Parse the ISIS3 label of ``path`` and return the layout of its core."""
    with open(path, "rb") as f:
        head = f.read(_LABEL_HEAD_BYTES).decode("latin-1")
    m = re.search(r"^\s*Object\s*=\s*Core\s*$(.*?)^\s*End_Object", head, re.M | re.S | re.I)
    if m is None:
        raise ValueError(f"{path}: no ISIS3 'Object = Core' block found in the label")
    block = m.group(1)

    def need(key):
        v = _label_value(block, key)
        if v is None:
            raise ValueError(f"{path}: label key {key!r} missing from the Core object")
        return v

    pixel_type = need("Type")
    try:
        bpp = _PIXEL_BYTES[pixel_type.upper()]
    except KeyError:
        raise ValueError(f"{path}: unsupported ISIS3 pixel type {pixel_type!r}") from None
    ts, tl = _label_value(block, "TileSamples"), _label_value(block, "TileLines")
    return CoreLayout(
        start_byte=int(_label_value(block, "StartByte") or 1),
        format=_label_value(block, "Format") or "BandSequential",
        samples=int(need("Samples")),
        lines=int(need("Lines")),
        bands=int(need("Bands")),
        pixel_type=pixel_type,
        bytes_per_pixel=bpp,
        tile_samples=int(ts) if ts else None,
        tile_lines=int(tl) if tl else None,
        core_file=_label_value(block, "^Core"),
    )


def is_complete(path) -> bool:
    """True if the file holds every byte of the cube core declared in its label.

    A partially downloaded cube is shorter than ``StartByte - 1 + core size``.
    Objects stored after the core (history, tables, the original label) make
    the file longer, which is fine. Detached cores (``^Core``) are checked in
    the referenced file.
    """
    path = Path(path)
    layout = core_layout(path)
    target = path if layout.core_file is None else path.parent / layout.core_file
    return target.stat().st_size >= layout.end_byte


# --------------------------------------------------------------------------- #
# Band identification
# --------------------------------------------------------------------------- #
_ALIASES = {"BLU": "BLU", "BLUE": "BLU", "PAN": "PAN", "PANCHROMATIC": "PAN",
            "RED": "RED", "NIR": "NIR", "INFRARED": "NIR"}


def filter_from_description(desc) -> str | None:
    """Map a band description such as ``"NIR"`` or ``"Blue"`` to a filter name."""
    if not desc:
        return None
    s = str(desc).upper()
    for word in re.split(r"[^A-Z]+", s):
        if word in _ALIASES:
            return _ALIASES[word]
    for f in ("NIR", "BLU", "PAN", "RED"):
        if f in s:
            return f
    return None


def band_mapping(count: int, descriptions=None) -> dict:
    """Return ``{filter: 1-based band index}`` for a cube.

    Filter names are taken from the band descriptions; bands that remain
    unidentified are assigned from ``POSITIONAL_ORDER`` for the band count.
    """
    mapping = {}
    descs = tuple(descriptions) if descriptions else (None,) * count
    for i, d in enumerate(descs, start=1):
        f = filter_from_description(d)
        if f and f not in mapping:
            mapping[f] = i
    if len(mapping) < count:
        used = set(mapping.values())
        for i, f in enumerate(POSITIONAL_ORDER.get(count, ()), start=1):
            if f not in mapping and i not in used:
                mapping[f] = i
    if not mapping:
        raise ValueError(f"cannot identify CaSSIS filters in a {count}-band cube")
    return mapping


# --------------------------------------------------------------------------- #
# Reading
# --------------------------------------------------------------------------- #
@dataclass
class CassisCube:
    """Filter planes of one CaSSIS cube.

    ``bands`` maps each of BLU, PAN, RED and NIR to a float32 array of shape
    ``shape``; filters absent from the file are all-NaN planes and
    ``filters_present`` lists the ones that exist. ``full_shape`` is the
    ``(lines, samples)`` size of the file and ``scale`` the decimation factor
    applied when reading (1.0 for full resolution).
    """

    path: str
    bands: dict
    filters_present: tuple
    shape: tuple
    full_shape: tuple
    scale: float = 1.0

    def __getitem__(self, name):
        return self.bands[name]

    def stack(self) -> np.ndarray:
        """Wavelength-ascending ``(4, lines, samples)`` array: BLU, PAN, RED, NIR."""
        return np.stack([self.bands[f] for f in FILTERS], axis=0)


def _mask_invalid(arr, lo=-1.0, hi=1.0):
    arr[(arr < lo) | (arr > hi) | ~np.isfinite(arr)] = np.nan
    return arr


def read_cassis_cube(path, max_side=None, clip_invalid=True) -> CassisCube:
    """Read a CaSSIS cube into named float32 filter planes.

    ``max_side`` decimates the read (nearest neighbour) so that the longer
    image side has at most that many pixels; ``None`` reads at full
    resolution. With ``clip_invalid`` (default) values outside [-1, 1] and
    non-finite values, which include the ISIS3 special pixels (NULL, LRS,
    LIS, HRS, HIS), become NaN.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.errors import NotGeoreferencedWarning

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.open(str(path)) as ds:
            full = (ds.height, ds.width)
            scale = 1.0 if not max_side else max(1.0, max(full) / float(max_side))
            out = (max(1, int(round(ds.height / scale))), max(1, int(round(ds.width / scale))))
            mapping = band_mapping(ds.count, ds.descriptions)
            present = tuple(f for f in FILTERS if f in mapping)
            bands = {}
            nan_plane = None
            for f in FILTERS:
                if f in mapping:
                    arr = ds.read(mapping[f], out_shape=out,
                                  resampling=Resampling.nearest).astype(np.float32)
                    bands[f] = _mask_invalid(arr) if clip_invalid else arr
                else:
                    if nan_plane is None:
                        nan_plane = np.full(out, np.nan, dtype=np.float32)
                    bands[f] = nan_plane
    return CassisCube(path=str(path), bands=bands, filters_present=present,
                      shape=out, full_shape=full, scale=scale)
