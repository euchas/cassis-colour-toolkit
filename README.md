# cassis-colour-toolkit

Python tools for the four-filter colour cubes of CaSSIS, the Colour and Stereo
Surface Imaging System on the ExoMars Trace Gas Orbiter (Thomas et al. 2017):

| Module | What it does |
|---|---|
| `cassis_colour.reader` | reads CaSSIS ISIS3 cubes (`*.cubeit.cub`) into named BLU, PAN, RED and NIR planes; identifies the filters from the label, reads decimated, and checks that a download is complete |
| `cassis_colour.parameters` | 30 colour and spectral-shape parameters of the four-point spectrum: band ratios, continuum-removed band depths, slopes, centred log-ratios and simplex angles, curvature, spectral moments, Legendre coefficients, entropy (pure numpy) |
| `cassis_colour.composites` | four percentile-stretched RGB composites: NPB colour, band depths, ratios, CLR shape |
| `cassis_colour.cli` | the `cassis-colour` command with the sub-commands `info`, `params` and `composites` |

The package was consolidated from research scripts; the parameter definitions
are unchanged. It works on the reflectance (I/F) values stored in the cube and
applies no photometric or atmospheric correction and no map projection.

## Install

```bash
pip install git+https://github.com/euchas/cassis-colour-toolkit
pip install "cassis-colour-toolkit[plot] @ git+https://github.com/euchas/cassis-colour-toolkit"   # matplotlib, for the composite PNGs
```

Requires Python 3.10 or newer, numpy and rasterio (GDAL's ISIS3 driver reads
the cubes). For development: `git clone`, then `pip install -e .[test,plot]`
and `pytest -q`.

## Quickstart

Input: a CaSSIS colour cube from the ESA Planetary Science Archive
(https://archives.esac.esa.int/psa, mission ExoMars 2016, instrument CaSSIS),
for example `MY36_012345_123_0.cubeit.cub`. Four-band cubes hold the filters
NIR, RED, PAN, BLU; three-band NPB cubes hold NIR, PAN, BLU (parameters that
need RED come out NaN).

```bash
cassis-colour info MY36_012345_123_0.cubeit.cub                      # label, completeness, per-filter statistics
cassis-colour params MY36_012345_123_0.cubeit.cub --out params.npz   # all 30 parameter planes
cassis-colour params MY36_012345_123_0.cubeit.cub --pixel 1200 800 --names BD_PAN,ratio_NIR_RED
cassis-colour composites MY36_012345_123_0.cubeit.cub --out-dir composites/
```

```python
from cassis_colour import (read_cassis_cube, is_complete, parameters_from_cube,
                           composites_from_cube, save_composites)

is_complete("MY36_012345_123_0.cubeit.cub")      # every core byte declared in the label is present
cube = read_cassis_cube("MY36_012345_123_0.cubeit.cub", max_side=3000)   # at most 3000 px on the long side
cube.filters_present                              # ('BLU', 'PAN', 'RED', 'NIR')
cube["NIR"]                                       # (lines, samples) float32, NaN where invalid
params = parameters_from_cube(cube)               # dict of 30 planes, names as in the table below
params["BD_PAN"], params["ratio_NIR_RED"]
comps = composites_from_cube(cube)                # 'npb', 'band_depths', 'ratios', 'shape' -> (lines, samples, 3)
save_composites(comps, "composites", "MY36_012345_123_0")
```

The parameter functions also take plain numbers or arrays of one common
shape, always in wavelength order BLU, PAN, RED, NIR:

```python
from cassis_colour import band_depth_PAN, compute_all_parameters
band_depth_PAN(0.061, 0.198, 0.241, 0.247)
compute_all_parameters(blu, pan, red, nir)        # {'ratio_NIR_RED': ..., 'BD_PAN': ..., ...}
```

## Parameters

Filter centres used throughout: BLU 494.8 nm, PAN 678.2 nm, RED 836.0 nm,
NIR 939.3 nm (`WAVELENGTHS_NM`), the effective central wavelengths of the
flight filters as quoted in the CaSSIS literature (filter characterisation:
Roloff et al. 2017). `R_i` is the reflectance in filter `i`;
`p_i = R_i / (BLU + PAN + RED + NIR)`; `lambda_i` the filter centre.
"Invariant" marks parameters unchanged by a common brightness factor
(`SCALE_INVARIANT`); the others scale with brightness.

| Name | Definition | Filters | Invariant |
|---|---|---|---|
| `ratio_NIR_RED`, `ratio_RED_PAN`, `ratio_NIR_BLU`, `ratio_PAN_BLU`, `ratio_RED_BLU` | `NIR/RED`, `RED/PAN`, `NIR/BLU`, `PAN/BLU`, `RED/BLU` | two each | yes |
| `BD_PAN` | `1 - PAN / (0.5874 BLU + 0.4126 NIR)`: band depth at 678 nm against the linear BLU-NIR continuum (Clark and Roush 1984); ferric absorption edge | BLU, PAN, NIR | yes |
| `BD_RED` | `1 - RED / (0.3956 PAN + 0.6044 NIR)`: band depth at 836 nm against the PAN-NIR continuum; short-wavelength shoulder of the 1 micron ferrous band | PAN, RED, NIR | yes |
| `S_495_678`, `S_678_836`, `S_836_939`, `S_495_939` | `(R_j - R_i) / (lambda_j - lambda_i)`, reflectance per nm | two each | no |
| `clr_BLU`, `clr_PAN`, `clr_RED`, `clr_NIR` | `ln R_i - mean_j ln R_j`, centred log-ratios (Aitchison 1986); they sum to zero | all | yes |
| `log_g` | `mean_j ln R_j`, log of the geometric mean reflectance | all | no |
| `theta`, `phi` | polar and azimuth angle of the CLR vector in isometric log-ratio coordinates with the Helmert basis (Egozcue et al. 2003); the direction of the spectral shape | all | yes |
| `kappa_PAN`, `kappa_RED` | second difference of `R(lambda)` at PAN (from BLU, PAN, RED) and at RED (from PAN, RED, NIR), per nm^2 | three each | no |
| `M0` | `BLU + PAN + RED + NIR` | all | no |
| `M1`, `M2`, `M3` | `sum p_i lambda_i` (mean wavelength, nm), `sum p_i (lambda_i - M1)^2` (nm^2), `sum p_i ((lambda_i - M1) / sqrt(M2))^3` (skewness) | all | yes |
| `c0`, `c1`, `c2`, `c3` | coefficients of the exact expansion `R = sum c_k P_k(x)` in Legendre polynomials, `x = 2 (lambda - 494.8) / (939.3 - 494.8) - 1`: level, tilt, curvature, cubic term | all | no |
| `entropy` | `-sum p_i ln p_i` (`ln 4` for a flat spectrum) | all | yes |
| `rho` | `sqrt(sum p_i^2)` (0.5 for a flat spectrum) | all | yes |

## Composites

| Name | R | G | B |
|---|---|---|---|
| `npb` | NIR | PAN | BLU |
| `band_depths` | `BD_PAN` | `BD_RED` | `c3` |
| `ratios` | `RED/PAN` | `NIR/RED` | `c3` |
| `shape` | `theta` | `phi` | `c3` |

Each channel is stretched linearly between its 2nd and 98th percentile over
the finite pixels (`--percentiles` on the command line); NaN becomes black.
`save_composites` writes one PNG per composite and a two-by-two overview.

## Reading cubes

* Filters are identified from the band descriptions GDAL takes from the
  `BandBin/FilterName` entries of the ISIS3 label. Without them the order
  NIR, RED, PAN, BLU (four bands) or NIR, PAN, BLU (three bands) is assumed.
* Values outside [-1, 1] and non-finite values, which include the ISIS3
  special pixels (NULL, LRS, LIS, HRS, HIS), become NaN (`clip_invalid=False`
  keeps them).
* `max_side` reads the cube decimated by nearest-neighbour sampling, so a
  large cube can be screened without loading it fully; `CassisCube.scale`
  records the factor.
* `is_complete(path)` compares the file size with the core size declared in
  the label (`core_layout(path)` returns that layout). This matters because
  GDAL returns zeros for the missing part of a truncated cube instead of
  raising an error, so an incomplete download would otherwise look like a
  dark image.

## Known limitations and assumptions

| Assumption | Value | Where |
|---|---|---|
| Filter centre wavelengths | 494.8, 678.2, 836.0, 939.3 nm (effective centres; slightly different values appear in different characterisation papers); the continuum weights, slope denominators, curvature grid and Legendre grid all derive from them | `src/cassis_colour/parameters.py`: `WAVELENGTHS_NM` |
| Valid reflectance range | [-1, 1]; everything else NaN | `src/cassis_colour/reader.py`: `read_cassis_cube()` |
| Non-positive reflectance | NaN or infinite values in the log-ratio family (`clr_*`, `log_g`, `theta`, `phi`) | `parameters.py`: `clr()` |
| Brightness and atmosphere | the invariant parameters cancel a multiplicative factor only; additive (atmospheric, stray-light) contributions are not removed | `parameters.py` |
| Decimation | nearest-neighbour sampling, no averaging | `reader.py`: `read_cassis_cube()` |
| Composite stretch | 2nd to 98th percentile per channel, computed on the decimated planes when `max_side` is used | `src/cassis_colour/composites.py`: `stretch_percentile()` |
| Cube format | attached-label ISIS3 cubes as distributed by the PSA; band-sequential and tiled cores; detached cores are handled by `is_complete()` only | `reader.py`: `core_layout()` |

## Tests

```bash
pip install -e .[test,plot]
pytest -q
```

The tests write their own small ISIS3 cubes (band-sequential and tiled, with
and without filter names, three-band and truncated) and check the reader,
every parameter family against closed-form values, the composites and the
command line tool. No mission data and no network access are needed.

## Citing

If this software is useful in your work, please cite it (see `CITATION.cff`):

> Valantinas, A. (2026). cassis-colour-toolkit: reader, colour parameters and
> RGB composites for CaSSIS cubes (version 0.1.0).
> https://github.com/euchas/cassis-colour-toolkit

and the instrument and methods it builds on:

* CaSSIS: Thomas, N. et al. (2017), Space Science Reviews 212, 1897-1944,
  https://doi.org/10.1007/s11214-017-0421-1; on-ground performance and
  filter characterisation, Roloff, V. et al. (2017), Space Science Reviews
  212, 1871-1896, https://doi.org/10.1007/s11214-017-0404-2; colour
  capabilities and products, Tornabene, L. L. et al. (2018), Space Science
  Reviews 214, 18, https://doi.org/10.1007/s11214-017-0436-7.
* Band depth: Clark, R. N. and Roush, T. L. (1984), Journal of Geophysical
  Research 89(B7), 6329-6340, https://doi.org/10.1029/JB089iB07p06329.
* Log-ratio representation: Aitchison, J. (1986), The Statistical Analysis
  of Compositional Data, Chapman and Hall,
  https://doi.org/10.1007/978-94-009-4109-0; Egozcue, J. J. et al. (2003),
  Mathematical Geology 35(3), 279-300, https://doi.org/10.1023/A:1023818214614.

## Acknowledgement

Much of the code was written with an AI coding assistant (Claude, Anthropic)
under the direction of the author, who reviewed the code and results.

## License

MIT, see `LICENSE`.
