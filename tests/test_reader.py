import numpy as np
import pytest

from cassis_colour import reader as rd

pytestmark = pytest.mark.skipif(pytest.importorskip("rasterio", reason="rasterio not installed") is None,
                                reason="rasterio not installed")


def test_named_bands_match_written_planes(cube4, scene):
    cube = rd.read_cassis_cube(cube4)
    assert cube.filters_present == ("BLU", "PAN", "RED", "NIR")
    assert cube.shape == cube.full_shape == scene["BLU"].shape
    assert cube.scale == 1.0
    for f in rd.FILTERS:
        arr = cube.bands[f]
        assert arr.dtype == np.float32
        ok = np.isfinite(arr)
        assert np.allclose(arr[ok], scene[f][ok], atol=1e-6)
    # the invalid pixels written by the fixture are NaN
    assert all(np.isnan(cube.bands[f][0, 0]) for f in rd.FILTERS)
    assert np.isnan(cube.bands["PAN"][1, 1]) and np.isnan(cube.bands["BLU"][2, 2])
    assert np.isfinite(cube.bands["RED"][1, 1])
    assert cube["NIR"] is cube.bands["NIR"]
    assert cube.stack().shape == (4,) + cube.shape


def test_clip_invalid_can_be_disabled(cube4):
    cube = rd.read_cassis_cube(cube4, clip_invalid=False)
    assert cube.bands["PAN"][1, 1] == pytest.approx(1.5)
    assert cube.bands["BLU"][2, 2] == pytest.approx(-2.0)
    assert cube.bands["NIR"][0, 0] < -1e38


def test_three_band_npb_cube(cube3, scene):
    cube = rd.read_cassis_cube(cube3)
    assert cube.filters_present == ("BLU", "PAN", "NIR")
    assert np.isnan(cube.bands["RED"]).all()
    for f in ("BLU", "PAN", "NIR"):
        assert np.allclose(cube.bands[f], scene[f], atol=1e-6)


def test_positional_fallback_and_tiled_format(cube_positional, cube_tiled, scene):
    for path in (cube_positional, cube_tiled):
        cube = rd.read_cassis_cube(path)
        assert cube.filters_present == ("BLU", "PAN", "RED", "NIR")
        assert cube.shape == scene["BLU"].shape
        for f in rd.FILTERS:
            assert np.allclose(cube.bands[f], scene[f], atol=1e-6), (path, f)


def test_decimated_read(cube4):
    cube = rd.read_cassis_cube(cube4, max_side=8)      # 12 x 16 -> 6 x 8
    assert cube.scale == pytest.approx(2.0)
    assert cube.shape == (6, 8)
    assert cube.full_shape == (12, 16)
    assert all(cube.bands[f].shape == (6, 8) for f in rd.FILTERS)


def test_core_layout_and_is_complete(cube4, cube_tiled, cube_truncated):
    lay = rd.core_layout(cube4)
    assert (lay.bands, lay.lines, lay.samples) == (4, 12, 16)
    assert lay.pixel_type.lower() == "real" and lay.bytes_per_pixel == 4
    assert lay.format == "BandSequential" and lay.start_byte == 65537
    assert lay.core_bytes == 4 * 12 * 16 * 4
    assert lay.end_byte == 65536 + lay.core_bytes
    tiled = rd.core_layout(cube_tiled)
    assert tiled.format == "Tile" and (tiled.tile_samples, tiled.tile_lines) == (8, 8)
    assert tiled.core_bytes == 4 * 16 * 16 * 4          # 12 lines padded to 2 tiles of 8
    assert rd.is_complete(cube4)
    assert rd.is_complete(cube_tiled)
    assert not rd.is_complete(cube_truncated)


def test_band_mapping_and_descriptions():
    assert rd.band_mapping(4, ("NIR", "RED", "PAN", "BLU")) == {"NIR": 1, "RED": 2, "PAN": 3, "BLU": 4}
    assert rd.band_mapping(3, ("Blue", "Panchromatic", "Near-infrared")) == {"BLU": 1, "PAN": 2, "NIR": 3}
    assert rd.band_mapping(4, None) == {"NIR": 1, "RED": 2, "PAN": 3, "BLU": 4}
    assert rd.band_mapping(3) == {"NIR": 1, "PAN": 2, "BLU": 3}
    # partial descriptions: identified bands keep their index, the rest is positional
    assert rd.band_mapping(4, (None, None, "PAN", None)) == {"PAN": 3, "NIR": 1, "RED": 2, "BLU": 4}
    assert rd.filter_from_description("CaSSIS RED filter") == "RED"
    assert rd.filter_from_description("") is None and rd.filter_from_description(None) is None
    with pytest.raises(ValueError):
        rd.band_mapping(7, None)
