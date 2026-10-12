# -*- coding: utf-8 -*-
"""Command line interface: ``cassis-colour {info,params,composites}``."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from ._version import __version__
from .composites import composites_from_cube, save_composites
from .parameters import FILTERS, PARAMETER_NAMES, parameters_from_cube
from .reader import core_layout, is_complete, read_cassis_cube


def image_id(path) -> str:
    """Product id from a file name: ``MY36_012345_123_0.cubeit.cub`` -> ``MY36_012345_123_0``."""
    name = Path(path).name
    for suffix in (".cubeit.cub", ".cub"):
        if name.lower().endswith(suffix):
            return name[: -len(suffix)]
    return Path(name).stem


def _stats(a):
    v = a[np.isfinite(a)]
    if v.size == 0:
        return "absent"
    return (f"valid {v.size}/{a.size}  min/median/max "
            f"{v.min():.4f}/{np.median(v):.4f}/{v.max():.4f}")


def cmd_info(args):
    layout = core_layout(args.cube)
    complete = is_complete(args.cube)
    print(args.cube)
    print(f"core: {layout.bands} bands x {layout.lines} lines x {layout.samples} samples, "
          f"{layout.pixel_type}, {layout.format}")
    print("complete:", "yes" if complete else "NO (file shorter than the core declared in the label)")
    cube = read_cassis_cube(args.cube, max_side=args.max_side)
    print(f"read at {cube.shape[0]} x {cube.shape[1]} (scale {cube.scale:.2f})")
    print("filters present:", " ".join(cube.filters_present))
    for f in FILTERS:
        print(f"{f}: {_stats(cube.bands[f])}")
    return 0


def cmd_params(args):
    names = [n.strip() for n in args.names.split(",")] if args.names else None
    cube = read_cassis_cube(args.cube, max_side=args.max_side)
    if args.pixel is not None:
        line, sample = args.pixel
        bands = {f: cube.bands[f][line, sample] for f in FILTERS}
        print(f"{args.cube}  line {line} sample {sample}")
        for f in FILTERS:
            print(f"  {f:14s} = {bands[f]:+.5f}")
        for k, v in parameters_from_cube(bands, names).items():
            print(f"  {k:14s} = {float(v):+.5f}")
        return 0
    params = parameters_from_cube(cube, names)
    if args.out:
        np.savez_compressed(args.out, **{k: np.asarray(v, dtype=np.float32) for k, v in params.items()})
        print(f"wrote {args.out} ({len(params)} parameters, {cube.shape[0]} x {cube.shape[1]})")
    print(f"{'parameter':14s}  scene median")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN planes of absent filters
        for k, v in params.items():
            print(f"{k:14s}  {np.nanmedian(v):+.5f}")
    return 0


def cmd_composites(args):
    cube = read_cassis_cube(args.cube, max_side=args.max_side)
    lo, hi = args.percentiles
    comps = composites_from_cube(cube, lo, hi)
    written = save_composites(comps, args.out_dir, image_id(args.cube), panel=not args.no_panel)
    for p in written:
        print("wrote", p)
    return 0


def build_parser():
    p = argparse.ArgumentParser(prog="cassis-colour",
                                description="CaSSIS colour cubes: inspection, colour parameters and RGB composites.")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("info", help="label summary, completeness check and per-filter statistics")
    s.add_argument("cube")
    s.add_argument("--max-side", type=int, default=2000,
                   help="read decimated so the longer side has at most this many pixels (default 2000; 0 = full)")
    s.set_defaults(func=cmd_info)

    s = sub.add_parser("params", help="compute the colour parameters")
    s.add_argument("cube")
    s.add_argument("--out", help="write the parameter planes to this .npz file")
    s.add_argument("--pixel", nargs=2, type=int, metavar=("LINE", "SAMPLE"),
                   help="print the reflectances and parameters of one pixel instead")
    s.add_argument("--names", help="comma-separated subset of: " + ", ".join(PARAMETER_NAMES))
    s.add_argument("--max-side", type=int, default=0, help="decimate the read (0 = full resolution)")
    s.set_defaults(func=cmd_params)

    s = sub.add_parser("composites", help="write the four RGB composites as PNG (needs matplotlib)")
    s.add_argument("cube")
    s.add_argument("--out-dir", default=".")
    s.add_argument("--max-side", type=int, default=0, help="decimate the read (0 = full resolution)")
    s.add_argument("--percentiles", nargs=2, type=float, default=(2.0, 98.0), metavar=("LO", "HI"),
                   help="stretch limits per channel (default 2 98)")
    s.add_argument("--no-panel", action="store_true", help="skip the two-by-two overview figure")
    s.set_defaults(func=cmd_composites)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "max_side", None) == 0:
        args.max_side = None
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
