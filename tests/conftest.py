"""Shared fixtures: small synthetic ISIS3 cubes written without GDAL."""

import math
import shutil

import numpy as np
import pytest

LINES, SAMPLES = 12, 16
LABEL_BYTES = 65536
ISIS_NULL = np.float32(-3.4028227e38)      # ISIS3 NULL special pixel (float32)


def write_isis3_cube(path, data, filters=None, centers=None, fmt="BandSequential",
                     tile=(8, 8), byteorder="Lsb", label_bytes=LABEL_BYTES):
    """Write ``data`` (bands, lines, samples) as an attached-label ISIS3 cube."""
    nb, nl, ns = data.shape
    lines = ["Object = IsisCube", "  Object = Core", f"    StartByte   = {label_bytes + 1}",
             f"    Format      = {fmt}"]
    if fmt == "Tile":
        lines += [f"    TileSamples = {tile[0]}", f"    TileLines   = {tile[1]}"]
    lines += ["", "    Group = Dimensions", f"      Samples = {ns}", f"      Lines   = {nl}",
              f"      Bands   = {nb}", "    End_Group", "", "    Group = Pixels",
              "      Type       = Real", f"      ByteOrder  = {byteorder}",
              "      Base       = 0.0", "      Multiplier = 1.0", "    End_Group", "  End_Object"]
    if filters:
        lines += ["", "  Group = BandBin", "    FilterName = (" + ", ".join(filters) + ")"]
        if centers:
            lines += ["    Center     = (" + ", ".join(str(c) for c in centers) + ") <nm>"]
        lines += ["  End_Group"]
    lines += ["End_Object", "", "Object = Label", f"  Bytes = {label_bytes}", "End_Object", "End", ""]
    label = "\n".join(lines).encode("ascii")
    assert len(label) < label_bytes
    dt = "<f4" if byteorder == "Lsb" else ">f4"
    if fmt == "Tile":
        ts, tl = tile
        ntx, nty = math.ceil(ns / ts), math.ceil(nl / tl)
        padded = np.full((nb, nty * tl, ntx * ts), ISIS_NULL, dtype=np.float32)
        padded[:, :nl, :ns] = data
        blocks = [padded[b, ty * tl:(ty + 1) * tl, tx * ts:(tx + 1) * ts].ravel()
                  for b in range(nb) for ty in range(nty) for tx in range(ntx)]
        raw = np.concatenate(blocks)
    else:
        raw = np.asarray(data, dtype=np.float32).ravel()
    with open(path, "wb") as f:
        f.write(label.ljust(label_bytes, b"\0"))
        f.write(np.ascontiguousarray(raw, dtype=dt).tobytes())
    return path


def make_scene(lines=LINES, samples=SAMPLES):
    """Smooth synthetic reflectance planes for BLU, PAN, RED, NIR (all in (0, 1))."""
    y = np.linspace(0.0, 1.0, lines)[:, None]
    x = np.linspace(0.0, 1.0, samples)[None, :]
    one = np.ones((lines, samples))
    return {
        "BLU": (0.06 + 0.04 * x) * one,
        "PAN": (0.16 + 0.06 * y) * one,
        "RED": (0.22 + 0.05 * x * y) * one,
        "NIR": (0.24 + 0.04 * x) * one,
    }


def _raw_stack(scene, order):
    return np.stack([scene[f] for f in order], axis=0).astype(np.float32)


@pytest.fixture(scope="session")
def scene():
    return make_scene()


@pytest.fixture(scope="session")
def cube4(tmp_path_factory, scene):
    """Four-band cube (NIR, RED, PAN, BLU) with filter names and a few invalid pixels."""
    data = _raw_stack(scene, ("NIR", "RED", "PAN", "BLU"))
    data[:, 0, 0] = ISIS_NULL          # NULL in every band
    data[2, 1, 1] = 1.5                # PAN above the valid range
    data[3, 2, 2] = -2.0               # BLU below the valid range
    p = tmp_path_factory.mktemp("cubes") / "MY36_012345_123_0.cubeit.cub"
    return write_isis3_cube(p, data, ("NIR", "RED", "PAN", "BLU"), (939.3, 836.0, 678.2, 494.8))


@pytest.fixture(scope="session")
def cube3(tmp_path_factory, scene):
    """Three-band NPB cube (NIR, PAN, BLU)."""
    p = tmp_path_factory.mktemp("cubes") / "npb.cubeit.cub"
    return write_isis3_cube(p, _raw_stack(scene, ("NIR", "PAN", "BLU")),
                            ("NIR", "PAN", "BLU"), (939.3, 678.2, 494.8))


@pytest.fixture(scope="session")
def cube_positional(tmp_path_factory, scene):
    """Four-band cube without a BandBin group (positional band order)."""
    p = tmp_path_factory.mktemp("cubes") / "nobandbin.cub"
    return write_isis3_cube(p, _raw_stack(scene, ("NIR", "RED", "PAN", "BLU")))


@pytest.fixture(scope="session")
def cube_tiled(tmp_path_factory, scene):
    """Four-band cube in the tiled core format (8 x 8 tiles, padded)."""
    p = tmp_path_factory.mktemp("cubes") / "tiled.cub"
    return write_isis3_cube(p, _raw_stack(scene, ("NIR", "RED", "PAN", "BLU")),
                            ("NIR", "RED", "PAN", "BLU"), fmt="Tile")


@pytest.fixture(scope="session")
def cube_truncated(tmp_path_factory, cube4):
    """Copy of ``cube4`` missing its last 100 bytes (an incomplete download)."""
    p = tmp_path_factory.mktemp("cubes") / "truncated.cubeit.cub"
    shutil.copyfile(cube4, p)
    with open(p, "r+b") as f:
        f.seek(0, 2)
        f.truncate(f.tell() - 100)
    return p
