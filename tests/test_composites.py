import numpy as np
import pytest

from cassis_colour import composites as comp


def test_stretch_and_composite_shapes(scene):
    a = np.array([[np.nan, 0.0], [1.0, 2.0]])
    s = comp.stretch_percentile(a, 0.0, 100.0)
    assert s.dtype == np.float32
    assert s[0, 0] == 0.0 and s[0, 1] == 0.0 and s[1, 1] == 1.0
    assert s[1, 0] == pytest.approx(0.5)
    assert np.all(comp.stretch_percentile(np.full((3, 3), 0.4)) == 0.0)      # no contrast
    assert np.all(comp.stretch_percentile(np.full((3, 3), np.nan)) == 0.0)   # nothing finite
    rgb = comp.make_rgb(a, a, a)
    assert rgb.shape == (2, 2, 3)

    comps = comp.composites_from_bands(scene)
    assert tuple(comps) == tuple(comp.COMPOSITES)
    for name, rgb in comps.items():
        assert rgb.shape == scene["BLU"].shape + (3,), name
        assert rgb.dtype == np.float32
        assert np.all((rgb >= 0.0) & (rgb <= 1.0))
        assert rgb.max() == 1.0


def test_save_composites_writes_png(tmp_path, scene):
    pytest.importorskip("matplotlib")
    comps = comp.composites_from_bands(scene)
    written = comp.save_composites(comps, tmp_path / "out", "IMG", panel=True)
    names = sorted(p.name for p in written)
    assert names == sorted(["IMG_npb.png", "IMG_band_depths.png", "IMG_ratios.png",
                            "IMG_shape.png", "IMG_composites.png"])
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    assert len(comp.save_composites(comps, tmp_path / "out2", "IMG", panel=False)) == 4
