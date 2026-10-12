# -*- coding: utf-8 -*-
"""Colour and spectral-shape parameters for the four CaSSIS filters.

CaSSIS (Colour and Stereo Surface Imaging System, ExoMars Trace Gas Orbiter)
images the surface of Mars through four broadband filters below 1 micron:

    BLU  494.8 nm    PAN  678.2 nm    RED  836.0 nm    NIR  939.3 nm

(effective band centres, see ``WAVELENGTHS_NM``). Every function takes the four
filter reflectances in wavelength-ascending order (BLU, PAN, RED, NIR), as
scalars or as numpy arrays of one common shape (for example whole image
planes), and returns the parameter, or a dict of parameters, with that shape.
The module needs only numpy. ``compute_all_parameters`` returns every
parameter in one flat dict and ``parameters_from_cube`` does the same for a
cube returned by :func:`cassis_colour.reader.read_cassis_cube`.

Parameter families
------------------
band ratios        ratio_NIR_RED, ratio_RED_PAN, ratio_NIR_BLU, ratio_PAN_BLU, ratio_RED_BLU
band depths        BD_PAN (678 nm against a BLU-NIR continuum),
                   BD_RED (836 nm against a PAN-NIR continuum)
spectral slopes    S_495_678, S_678_836, S_836_939, S_495_939 (reflectance per nm)
CLR                clr_BLU, clr_PAN, clr_RED, clr_NIR (centred log-ratios) and
                   log_g (log of the geometric mean reflectance)
simplex angles     theta, phi: direction of the CLR vector in isometric
                   log-ratio (Helmert) coordinates
curvature          kappa_PAN, kappa_RED: second differences at PAN and at RED
spectral moments   M0 (sum of reflectances), M1 (mean wavelength),
                   M2 (spread), M3 (skewness)
Legendre           c0, c1, c2, c3: expansion of the four-point spectrum in
                   Legendre polynomials of the normalised wavelength
entropy, rho       spectral entropy and L2 norm of the normalised spectrum

Brightness invariance
---------------------
Unchanged by a common brightness factor (``SCALE_INVARIANT``): the ratios,
the band depths, the CLR values, theta, phi, M1, M2, M3, entropy and rho.
Scaling with brightness: M0, log_g, the slopes, the curvatures and the
Legendre coefficients.
"""

from __future__ import annotations

import numpy as np

FILTERS = ("BLU", "PAN", "RED", "NIR")
"""CaSSIS filter names in wavelength-ascending order."""

WAVELENGTHS_NM = {"BLU": 494.8, "PAN": 678.2, "RED": 836.0, "NIR": 939.3}
"""Effective band-centre wavelength of each filter, nm."""

_WL = np.array([WAVELENGTHS_NM[f] for f in FILTERS], dtype=np.float64)

# Linear-continuum weights at PAN (between BLU and NIR) and at RED (between
# PAN and NIR), fixed by the band centres.
_W_PAN_BLU = (_WL[3] - _WL[1]) / (_WL[3] - _WL[0])   # 0.58740
_W_PAN_NIR = 1.0 - _W_PAN_BLU                          # 0.41260
_W_RED_PAN = (_WL[3] - _WL[2]) / (_WL[3] - _WL[1])    # 0.39563
_W_RED_NIR = 1.0 - _W_RED_PAN                          # 0.60437

# Helmert (isometric log-ratio) orthonormal basis for a four-part composition.
_HELMERT = np.array([
    [1 / np.sqrt(2), -1 / np.sqrt(2), 0.0, 0.0],
    [1 / np.sqrt(6), 1 / np.sqrt(6), -2 / np.sqrt(6), 0.0],
    [1 / np.sqrt(12), 1 / np.sqrt(12), 1 / np.sqrt(12), -3 / np.sqrt(12)],
], dtype=np.float64)

PARAMETER_NAMES = (
    "ratio_NIR_RED", "ratio_RED_PAN", "ratio_NIR_BLU", "ratio_PAN_BLU", "ratio_RED_BLU",
    "BD_PAN", "BD_RED",
    "S_495_678", "S_678_836", "S_836_939", "S_495_939",
    "clr_BLU", "clr_PAN", "clr_RED", "clr_NIR", "log_g",
    "theta", "phi",
    "kappa_PAN", "kappa_RED",
    "M0", "M1", "M2", "M3",
    "c0", "c1", "c2", "c3",
    "entropy", "rho",
)
"""Keys returned by :func:`compute_all_parameters`, in order."""

SCALE_INVARIANT = (
    "ratio_NIR_RED", "ratio_RED_PAN", "ratio_NIR_BLU", "ratio_PAN_BLU", "ratio_RED_BLU",
    "BD_PAN", "BD_RED",
    "clr_BLU", "clr_PAN", "clr_RED", "clr_NIR",
    "theta", "phi",
    "M1", "M2", "M3",
    "entropy", "rho",
)
"""Parameters that do not change when all four reflectances are multiplied by
the same factor."""


def _stack(BLU, PAN, RED, NIR):
    """Stack the four bands along a new leading axis, shape (4, ...)."""
    return np.stack([np.asarray(b, dtype=np.float64) for b in (BLU, PAN, RED, NIR)], axis=0)


# --------------------------------------------------------------------------- #
# Band ratios
# --------------------------------------------------------------------------- #
def band_ratios(BLU, PAN, RED, NIR):
    """Return the five CaSSIS band ratios as a dict.

    NIR/RED responds to the short-wavelength side of the 1 micron ferrous
    (Fe2+) absorption, RED/PAN to the ferric (Fe3+) absorption edge; the three
    ratios against BLU measure the overall red slope.
    """
    BLU, PAN, RED, NIR = (np.asarray(b, dtype=np.float64) for b in (BLU, PAN, RED, NIR))
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "ratio_NIR_RED": NIR / RED,
            "ratio_RED_PAN": RED / PAN,
            "ratio_NIR_BLU": NIR / BLU,
            "ratio_PAN_BLU": PAN / BLU,
            "ratio_RED_BLU": RED / BLU,
        }


# --------------------------------------------------------------------------- #
# Band depths (continuum-removed, Clark and Roush 1984)
# --------------------------------------------------------------------------- #
def band_depth_PAN(BLU, PAN, RED, NIR):
    """Band depth at PAN (678 nm) against a linear BLU-NIR continuum.

    ``1 - PAN / (0.5874 BLU + 0.4126 NIR)``; positive where the spectrum is
    concave between BLU and NIR, as across the ferric absorption edge.
    """
    BLU, PAN, NIR = (np.asarray(b, dtype=np.float64) for b in (BLU, PAN, NIR))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - PAN / (_W_PAN_BLU * BLU + _W_PAN_NIR * NIR)


def band_depth_RED(BLU, PAN, RED, NIR):
    """Band depth at RED (836 nm) against a linear PAN-NIR continuum.

    ``1 - RED / (0.3956 PAN + 0.6044 NIR)``; responds to the short-wavelength
    shoulder of the 1 micron ferrous (Fe2+) band.
    """
    PAN, RED, NIR = (np.asarray(b, dtype=np.float64) for b in (PAN, RED, NIR))
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - RED / (_W_RED_PAN * PAN + _W_RED_NIR * NIR)


# --------------------------------------------------------------------------- #
# Spectral slopes (reflectance per nm)
# --------------------------------------------------------------------------- #
def spectral_slopes(BLU, PAN, RED, NIR):
    """Return the four band-to-band spectral slopes as a dict (units: 1/nm)."""
    BLU, PAN, RED, NIR = (np.asarray(b, dtype=np.float64) for b in (BLU, PAN, RED, NIR))
    return {
        "S_495_678": (PAN - BLU) / (_WL[1] - _WL[0]),
        "S_678_836": (RED - PAN) / (_WL[2] - _WL[1]),
        "S_836_939": (NIR - RED) / (_WL[3] - _WL[2]),
        "S_495_939": (NIR - BLU) / (_WL[3] - _WL[0]),
    }


# --------------------------------------------------------------------------- #
# Centred log-ratio (compositional representation of the spectral shape)
# --------------------------------------------------------------------------- #
def clr(BLU, PAN, RED, NIR):
    """Centred log-ratios of the four bands and the log geometric mean.

    Returns ``{clr_BLU, clr_PAN, clr_RED, clr_NIR, log_g}`` where
    ``clr_i = ln R_i - log_g`` and ``log_g = mean_i ln R_i`` (Aitchison 1986).
    The four CLR values sum to zero and carry the spectral shape; ``log_g``
    carries the brightness. Reflectances must be positive; non-positive values
    give NaN or infinite results.
    """
    R = _stack(BLU, PAN, RED, NIR)
    with np.errstate(divide="ignore", invalid="ignore"):
        logR = np.log(R)
        log_g = logR.mean(axis=0)
        c = logR - log_g
    return {"clr_BLU": c[0], "clr_PAN": c[1], "clr_RED": c[2], "clr_NIR": c[3], "log_g": log_g}


def simplex_angles(BLU, PAN, RED, NIR):
    """Polar angle ``theta`` and azimuth ``phi`` of the CLR vector.

    The CLR vector is expressed in isometric log-ratio coordinates with the
    Helmert basis (Egozcue et al. 2003) and converted to spherical angles:
    ``theta`` in [0, pi] from the third axis, ``phi`` in (-pi, pi] in the
    plane of the first two. They describe the direction of the spectral shape
    irrespective of its contrast.
    """
    c = clr(BLU, PAN, RED, NIR)
    C = np.stack([c["clr_BLU"], c["clr_PAN"], c["clr_RED"], c["clr_NIR"]], axis=0)
    ilr = np.tensordot(_HELMERT, C, axes=(1, 0))          # shape (3, ...)
    x, y, z = ilr[0], ilr[1], ilr[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        norm = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(np.clip(z / norm, -1.0, 1.0))
        phi = np.arctan2(y, x)
    return {"theta": theta, "phi": phi}


# --------------------------------------------------------------------------- #
# Curvature (second difference on the non-uniform wavelength grid)
# --------------------------------------------------------------------------- #
def _second_diff(L, C, R, xL, xC, xR):
    num = 2.0 * (R * (xC - xL) - C * (xR - xL) + L * (xR - xC))
    den = (xR - xC) * (xC - xL) * (xR - xL)
    return num / den


def curvature(BLU, PAN, RED, NIR):
    """Second differences of reflectance against wavelength (units: 1/nm^2).

    ``kappa_PAN`` uses BLU, PAN, RED and ``kappa_RED`` uses PAN, RED, NIR.
    Negative values mark a local peak, positive values a local trough.
    """
    BLU, PAN, RED, NIR = (np.asarray(b, dtype=np.float64) for b in (BLU, PAN, RED, NIR))
    return {
        "kappa_PAN": _second_diff(BLU, PAN, RED, _WL[0], _WL[1], _WL[2]),
        "kappa_RED": _second_diff(PAN, RED, NIR, _WL[1], _WL[2], _WL[3]),
    }


# --------------------------------------------------------------------------- #
# Spectral moments
# --------------------------------------------------------------------------- #
def moments(BLU, PAN, RED, NIR):
    """Moments of the spectrum treated as a distribution over wavelength.

    With ``p_i = R_i / M0``: ``M0`` is the sum of the four reflectances,
    ``M1`` the reflectance-weighted mean wavelength (nm), ``M2`` the variance
    about it (nm^2) and ``M3`` the standardised third moment (skewness).
    """
    R = _stack(BLU, PAN, RED, NIR)
    with np.errstate(divide="ignore", invalid="ignore"):
        M0 = R.sum(axis=0)
        p = R / M0
        wl = _WL.reshape((4,) + (1,) * (R.ndim - 1))
        M1 = (p * wl).sum(axis=0)
        d = wl - M1
        M2 = (p * d ** 2).sum(axis=0)
        M3 = (p * (d / np.sqrt(M2)) ** 3).sum(axis=0)
    return {"M0": M0, "M1": M1, "M2": M2, "M3": M3}


# --------------------------------------------------------------------------- #
# Legendre coefficients
# --------------------------------------------------------------------------- #
def _legendre_inv():
    x = 2.0 * (_WL - _WL.min()) / (_WL.max() - _WL.min()) - 1.0
    B = np.empty((4, 4))
    B[:, 0] = 1.0
    B[:, 1] = x
    B[:, 2] = 0.5 * (3 * x ** 2 - 1)
    B[:, 3] = 0.5 * (5 * x ** 3 - 3 * x)
    return np.linalg.inv(B)


_BINV = _legendre_inv()


def legendre(BLU, PAN, RED, NIR):
    """Coefficients ``c0..c3`` of the four-point spectrum in Legendre polynomials.

    The wavelength is mapped to ``x = 2 (lambda - 494.8) / (939.3 - 494.8) - 1``
    and the four reflectances are expanded exactly as
    ``R(x) = c0 P0(x) + c1 P1(x) + c2 P2(x) + c3 P3(x)``: ``c0`` is a mean
    level, ``c1`` a linear tilt, ``c2`` a curvature and ``c3`` a cubic term.
    """
    R = _stack(BLU, PAN, RED, NIR)
    C = np.tensordot(_BINV, R, axes=(1, 0))               # shape (4, ...)
    return {"c0": C[0], "c1": C[1], "c2": C[2], "c3": C[3]}


# --------------------------------------------------------------------------- #
# Entropy and L2 norm of the normalised spectrum
# --------------------------------------------------------------------------- #
def entropy_rho(BLU, PAN, RED, NIR):
    """Spectral entropy ``-sum p_i ln p_i`` and ``rho = sqrt(sum p_i^2)``.

    Both measure how evenly the reflectance is spread over the four filters:
    a flat spectrum gives entropy ``ln 4`` and rho 0.5.
    """
    R = _stack(BLU, PAN, RED, NIR)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = R / R.sum(axis=0)
        ent = -(p * np.log(np.where(p > 0, p, 1.0))).sum(axis=0)
        rho = np.sqrt((p ** 2).sum(axis=0))
    return {"entropy": ent, "rho": rho}


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def compute_all_parameters(BLU, PAN, RED, NIR):
    """Compute every parameter and return one flat dict keyed by ``PARAMETER_NAMES``.

    Inputs are the four filter reflectances, scalars or arrays of one shape.
    """
    out = {}
    out.update(band_ratios(BLU, PAN, RED, NIR))
    out["BD_PAN"] = band_depth_PAN(BLU, PAN, RED, NIR)
    out["BD_RED"] = band_depth_RED(BLU, PAN, RED, NIR)
    out.update(spectral_slopes(BLU, PAN, RED, NIR))
    out.update(clr(BLU, PAN, RED, NIR))
    out.update(simplex_angles(BLU, PAN, RED, NIR))
    out.update(curvature(BLU, PAN, RED, NIR))
    out.update(moments(BLU, PAN, RED, NIR))
    out.update(legendre(BLU, PAN, RED, NIR))
    out.update(entropy_rho(BLU, PAN, RED, NIR))
    return {k: out[k] for k in PARAMETER_NAMES}


def parameters_from_cube(cube, names=None):
    """Compute parameters for a cube from :func:`cassis_colour.reader.read_cassis_cube`.

    ``cube`` may also be a plain dict with the keys BLU, PAN, RED and NIR.
    ``names`` optionally restricts the result to a subset of ``PARAMETER_NAMES``.
    Filters absent from the cube (for example RED in a three-band NPB product)
    are all-NaN planes, so parameters that need them come out NaN.
    """
    bands = cube.bands if hasattr(cube, "bands") else cube
    params = compute_all_parameters(bands["BLU"], bands["PAN"], bands["RED"], bands["NIR"])
    if names is None:
        return params
    unknown = [n for n in names if n not in PARAMETER_NAMES]
    if unknown:
        raise KeyError(f"unknown parameter name(s): {', '.join(unknown)}")
    return {n: params[n] for n in names}
