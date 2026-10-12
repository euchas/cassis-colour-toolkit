import numpy as np
import pytest

from cassis_colour import parameters as sp

WL = np.array([sp.WAVELENGTHS_NM[f] for f in sp.FILTERS])


def test_linear_spectrum_has_zero_band_depths_and_curvature():
    R = 0.1 + 0.0002 * (WL - WL[0])                  # reflectance linear in wavelength
    assert sp.band_depth_PAN(*R) == pytest.approx(0.0, abs=1e-12)
    assert sp.band_depth_RED(*R) == pytest.approx(0.0, abs=1e-12)
    k = sp.curvature(*R)
    assert k["kappa_PAN"] == pytest.approx(0.0, abs=1e-12)
    assert k["kappa_RED"] == pytest.approx(0.0, abs=1e-12)
    for v in sp.spectral_slopes(*R).values():
        assert v == pytest.approx(0.0002)
    leg = sp.legendre(*R)
    assert leg["c2"] == pytest.approx(0.0, abs=1e-12)
    assert leg["c3"] == pytest.approx(0.0, abs=1e-12)


def test_scale_invariance():
    rng = np.random.default_rng(1)
    R = rng.uniform(0.05, 0.4, size=(4, 5, 7))
    a = sp.compute_all_parameters(*R)
    b = sp.compute_all_parameters(*(3.7 * R))
    for k in sp.SCALE_INVARIANT:
        assert np.allclose(a[k], b[k]), k
    for k in ("M0", "S_495_678", "S_678_836", "S_836_939", "S_495_939",
              "kappa_PAN", "kappa_RED", "c0", "c1", "c2", "c3"):
        assert np.allclose(b[k], 3.7 * a[k]), k
    assert np.allclose(b["log_g"], a["log_g"] + np.log(3.7))


def test_clr_angles_and_legendre_round_trip():
    R = np.array([0.061, 0.198, 0.241, 0.247])
    c = sp.clr(*R)
    assert c["clr_BLU"] + c["clr_PAN"] + c["clr_RED"] + c["clr_NIR"] == pytest.approx(0.0, abs=1e-12)
    assert c["log_g"] == pytest.approx(np.log(R).mean())
    ang = sp.simplex_angles(*R)
    assert 0.0 <= ang["theta"] <= np.pi
    assert -np.pi < ang["phi"] <= np.pi
    leg = sp.legendre(*R)
    x = 2.0 * (WL - WL.min()) / (WL.max() - WL.min()) - 1.0
    recon = (leg["c0"] + leg["c1"] * x + leg["c2"] * 0.5 * (3 * x ** 2 - 1)
             + leg["c3"] * 0.5 * (5 * x ** 3 - 3 * x))
    assert np.allclose(recon, R)
    # non-positive reflectance gives non-finite log-ratio values, not an exception
    bad = sp.clr(0.0, 0.2, 0.2, 0.2)
    assert np.isnan(bad["clr_BLU"]) and not np.isfinite(bad["log_g"])
    assert np.isnan(sp.simplex_angles(-0.1, 0.2, 0.2, 0.2)["theta"])


def test_flat_spectrum_moments_and_entropy():
    R = (0.2, 0.2, 0.2, 0.2)
    m = sp.moments(*R)
    d = WL - WL.mean()
    assert m["M0"] == pytest.approx(0.8)
    assert m["M1"] == pytest.approx(WL.mean())
    assert m["M2"] == pytest.approx((d ** 2).mean())
    assert m["M3"] == pytest.approx((d ** 3).mean() / (d ** 2).mean() ** 1.5)
    e = sp.entropy_rho(*R)
    assert e["entropy"] == pytest.approx(np.log(4.0))
    assert e["rho"] == pytest.approx(0.5)


def test_compute_all_names_shapes_and_values():
    R = (0.1, 0.2, 0.25, 0.3)
    out = sp.compute_all_parameters(*R)
    assert tuple(out) == sp.PARAMETER_NAMES
    assert out["ratio_NIR_RED"] == pytest.approx(1.2)
    assert out["ratio_RED_PAN"] == pytest.approx(1.25)
    assert out["ratio_PAN_BLU"] == pytest.approx(2.0)
    w = (WL[3] - WL[1]) / (WL[3] - WL[0])
    assert out["BD_PAN"] == pytest.approx(1.0 - 0.2 / (w * 0.1 + (1.0 - w) * 0.3))
    w2 = (WL[3] - WL[2]) / (WL[3] - WL[1])
    assert out["BD_RED"] == pytest.approx(1.0 - 0.25 / (w2 * 0.2 + (1.0 - w2) * 0.3))
    assert out["S_495_678"] == pytest.approx(0.1 / (WL[1] - WL[0]))
    assert out["M0"] == pytest.approx(0.85)
    # arrays of one shape give arrays of that shape
    planes = [np.full((3, 4), v) for v in R]
    arr = sp.compute_all_parameters(*planes)
    assert all(np.shape(v) == (3, 4) for v in arr.values())
    assert np.allclose(arr["BD_PAN"], out["BD_PAN"])
    # cube-style input, subset selection and an absent filter
    bands = {"BLU": planes[0], "PAN": planes[1], "RED": np.full((3, 4), np.nan), "NIR": planes[3]}
    sub = sp.parameters_from_cube(bands, ["ratio_NIR_BLU", "BD_RED"])
    assert tuple(sub) == ("ratio_NIR_BLU", "BD_RED")
    assert np.allclose(sub["ratio_NIR_BLU"], 3.0)
    assert np.isnan(sub["BD_RED"]).all()
    with pytest.raises(KeyError):
        sp.parameters_from_cube(bands, ["no_such_parameter"])
