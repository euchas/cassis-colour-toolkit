"""cassis-colour-toolkit: read CaSSIS colour cubes, compute colour parameters, build composites."""

from ._version import __version__
from .composites import (COMPOSITES, composites_from_bands, composites_from_cube,
                         make_rgb, save_composites, stretch_percentile)
from .parameters import (FILTERS, PARAMETER_NAMES, SCALE_INVARIANT, WAVELENGTHS_NM,
                         band_depth_PAN, band_depth_RED, band_ratios, clr,
                         compute_all_parameters, curvature, entropy_rho, legendre,
                         moments, parameters_from_cube, simplex_angles, spectral_slopes)
from .reader import (POSITIONAL_ORDER, CassisCube, CoreLayout, band_mapping, core_layout,
                     filter_from_description, is_complete, read_cassis_cube)

__all__ = [
    "__version__",
    "FILTERS", "WAVELENGTHS_NM", "PARAMETER_NAMES", "SCALE_INVARIANT", "POSITIONAL_ORDER",
    "band_ratios", "band_depth_PAN", "band_depth_RED", "spectral_slopes", "clr",
    "simplex_angles", "curvature", "moments", "legendre", "entropy_rho",
    "compute_all_parameters", "parameters_from_cube",
    "CassisCube", "CoreLayout", "read_cassis_cube", "is_complete", "core_layout",
    "band_mapping", "filter_from_description",
    "COMPOSITES", "stretch_percentile", "make_rgb", "composites_from_bands",
    "composites_from_cube", "save_composites",
]
