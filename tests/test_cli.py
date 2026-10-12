import numpy as np
import pytest

from cassis_colour import cli

pytestmark = pytest.mark.skipif(pytest.importorskip("rasterio", reason="rasterio not installed") is None,
                                reason="rasterio not installed")


def test_info_and_params_commands(cube4, cube_truncated, tmp_path, capsys):
    assert cli.main(["info", str(cube4)]) == 0
    out = capsys.readouterr().out
    assert "complete: yes" in out
    assert "filters present: BLU PAN RED NIR" in out
    assert "core: 4 bands x 12 lines x 16 samples" in out

    assert cli.main(["info", str(cube_truncated)]) == 0
    assert "complete: NO" in capsys.readouterr().out

    npz = tmp_path / "params.npz"
    assert cli.main(["params", str(cube4), "--out", str(npz), "--names", "BD_PAN,ratio_NIR_RED"]) == 0
    with np.load(npz) as z:
        assert set(z.files) == {"BD_PAN", "ratio_NIR_RED"}
        assert z["BD_PAN"].shape == (12, 16) and z["BD_PAN"].dtype == np.float32
    assert "scene median" in capsys.readouterr().out

    assert cli.main(["params", str(cube4), "--pixel", "5", "6"]) == 0
    out = capsys.readouterr().out
    assert "line 5 sample 6" in out and "BD_PAN" in out and "rho" in out

    assert cli.image_id("MY36_012345_123_0.cubeit.cub") == "MY36_012345_123_0"
    assert cli.image_id("scene.cub") == "scene"


def test_composites_command(cube4, tmp_path, capsys):
    pytest.importorskip("matplotlib")
    assert cli.main(["composites", str(cube4), "--out-dir", str(tmp_path), "--max-side", "8"]) == 0
    names = sorted(p.name for p in tmp_path.glob("*.png"))
    assert names == sorted(["MY36_012345_123_0_npb.png", "MY36_012345_123_0_band_depths.png",
                            "MY36_012345_123_0_ratios.png", "MY36_012345_123_0_shape.png",
                            "MY36_012345_123_0_composites.png"])
    assert capsys.readouterr().out.count("wrote") == 5
