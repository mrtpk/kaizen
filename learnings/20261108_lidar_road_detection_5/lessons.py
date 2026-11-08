"""LiDAR Road Detection, part 5: what held up.

Re-runs the key checks of parts 1 to 4 and prints one scorecard.
Every number in the post comes from this script.

Run:  uv run --with numpy --with pillow python lessons.py
It reuses the scripts of parts 1 to 4 (sibling folders in the kaizen repo).
"""

import contextlib
import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import numpy as np

HERE = Path(__file__).resolve().parent


def sibling(folder: str, name: str) -> ModuleType:
    """Import a script of an earlier part of the series."""
    spec = importlib.util.spec_from_file_location(name, HERE.parent / folder / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


part1 = sibling("20261108_lidar_road_detection_1", "top_view_grid")
part2 = sibling("20261108_lidar_road_detection_2", "labels_top_view")
part3 = sibling("20261108_lidar_road_detection_3", "context_module")
part4 = sibling("20261108_lidar_road_detection_4", "fewer_beams")


def main() -> None:
    points = part1.load_points()
    grid = part1.top_view(points)
    occupied = grid[0] > 0

    print("== Scorecard ==")
    print("1. grid: %d x %d cells of %.1f m; %d of %d scan points fall in it"
          % (part1.ROWS, part1.COLS, part1.CELL, part1.in_grid(points).sum(), len(points)))
    print("   occupied cells: %.1f %%" % (occupied.mean() * 100))

    _, calib, road = part2.load()
    road_pts, kept = part2.dense_road(points, calib, road)
    sparse = part2.grid_cells(road_pts)
    dense = part2.grid_cells(np.vstack([road_pts[:, :4], kept[:, :4]]))
    print("2. road cells: %d from points alone, %d with pseudo points (x%.1f)"
          % (sparse.sum(), dense.sum(), dense.sum() / sparse.sum()))

    w, h = part3.receptive_fields(part3.DILATIONS)[-1]
    print("3. context receptive field %d x %d on a %d x %d input" % (w, h, part3.CTX_W, part3.CTX_H))

    ring = part4.rings(points)
    p16 = points[ring % 4 == 0]
    occ16 = part1.top_view(p16)[0] > 0
    print("4. occupied cells with 16 of 64 rings: %.1f %% (was %.1f %%)" % (occ16.mean() * 100, occupied.mean() * 100))
    far = part4.band_occupancy(occ16)[-1]
    print("   at 36-46 m with 16 rings: %.1f %% of cells occupied" % far)

    checks = [
        ("vectorised grid equals the loop", bool(np.allclose(grid, part1.top_view_loop(points), atol=1e-5))),
        ("sparse road cells stay road in the dense label", bool(np.all(dense[sparse]))),
        ("receptive field is larger than the context input", w > part3.CTX_W and h > part3.CTX_H),
        ("16-ring cells are a subset of 64-ring cells", bool(np.all(occupied[occ16]))),
    ]
    print("\n== Checks ==")
    for name, ok in checks:
        print(f"{name}. Is pass? {ok}")
    print("All checks hold. Is pass?", all(ok for _, ok in checks))


if __name__ == "__main__":
    main()
