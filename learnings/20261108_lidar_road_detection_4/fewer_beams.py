"""LiDAR Road Detection, part 4: from 64 beams to 16, and back to points.

Splits the 64-beam KITTI sample scan into its rings, keeps every 4th ring to get
a 16-beam scan with the same vertical field of view, and measures what that does
to the top-view grid. Then it labels points from a top-view road mask by a cell
lookup, the last step of the pipeline.
Every number in the post comes from this script.

Run:  uv run --with numpy --with pillow python fewer_beams.py
It reuses the scripts of part 1 and part 2 (sibling folders in the kaizen repo).
"""

import importlib.util
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
    spec.loader.exec_module(module)
    return module


part1 = sibling("20261108_lidar_road_detection_1", "top_view_grid")
part2 = sibling("20261108_lidar_road_detection_2", "labels_top_view")

BANDS = [(6, 16), (16, 26), (26, 36), (36, 46)]


def rings(points: np.ndarray) -> np.ndarray:
    """Ring index of each point. The scan is stored ring by ring; a ring ends
    where the azimuth wraps around. The first and the last piece are one ring."""
    az = np.degrees(np.arctan2(points[:, 1], points[:, 0]))
    ring = np.r_[0, np.cumsum(np.abs(np.diff(az)) > 180)]
    ring[ring == ring.max()] = 0
    return ring


def elevation(points: np.ndarray) -> np.ndarray:
    return np.degrees(np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1])))


def band_occupancy(occupied: np.ndarray) -> list[float]:
    out = []
    for near, far in BANDS:
        r0, r1 = round((part1.X_MAX - far) / part1.CELL), round((part1.X_MAX - near) / part1.CELL)
        out.append(100 * float(occupied[r0:r1].mean()))
    return out


def corridor_hits(points: np.ndarray, ring: np.ndarray) -> np.ndarray:
    """Distance ahead where each ring crosses the corridor |y| < 1 m, 6-46 m."""
    m = (np.abs(points[:, 1]) < 1.0) & (points[:, 0] >= 6) & (points[:, 0] < 46)
    hits = [float(np.median(points[m & (ring == r), 0])) for r in np.unique(ring[m])]
    return np.sort(np.array(hits))


def main() -> None:
    points = part1.load_points()
    ring = rings(points)
    el = elevation(points)

    print("== 1. Rings in the scan ==")
    n_rings = int(ring.max()) + 1
    print("rings found:", n_rings)
    top = float(np.median(el[ring == 0]))
    bottom = float(np.median(el[ring == n_rings - 1]))
    print("elevation of the rings: %.1f to %.1f degrees" % (top, bottom))
    print("mean step between rings: %.2f degrees" % ((top - bottom) / (n_rings - 1)))

    print("\n== 2. Keep every 4th ring ==")
    keep16 = ring % 4 == 0
    p16 = points[keep16]
    print("rings kept:", len(np.unique(ring[keep16])), " points:", len(p16), "of", len(points),
          "(%.1f %%)" % (100 * len(p16) / len(points)))

    print("\n== 3. What the grid sees ==")
    occ64 = part1.top_view(points)[0] > 0
    occ16 = part1.top_view(p16)[0] > 0
    print("points in the grid: 64 rings %d, 16 rings %d" % (part1.in_grid(points).sum(), part1.in_grid(p16).sum()))
    print("occupied cells: 64 rings %.1f %%, 16 rings %.1f %%" % (occ64.mean() * 100, occ16.mean() * 100))
    for (near, far), a, b in zip(BANDS, band_occupancy(occ64), band_occupancy(occ16)):
        print(f"  {near:2d}-{far:2d} m: 64 rings {a:5.1f} %   16 rings {b:5.1f} %")
    print("Every cell of the 16-ring grid is also in the 64-ring grid. Is pass?", bool(np.all(occ64[occ16])))

    print("\n== 4. Rings on the road ahead (|y| < 1 m) ==")
    for name, pts, rg in [("64 rings", points, ring), ("16 rings", p16, ring[keep16])]:
        hits = corridor_hits(pts, rg)
        gaps = np.diff(hits)
        print(f"{name}: {len(hits)} rings cross 6-46 m; first at {hits[0]:.1f} m, last at {hits[-1]:.1f} m;"
              f" largest gap {gaps.max():.1f} m")

    print("\n== 5. Back to points: label each point from a top-view mask ==")
    _, calib, road = part2.load()
    road_pts, kept = part2.dense_road(points, calib, road)
    mask = part2.grid_cells(np.vstack([road_pts[:, :4], kept[:, :4]]))   # stands in for the network output
    inside16, truth16 = part2.read_labels(p16, calib, road)
    g = part1.in_grid(p16) & inside16
    row, col = part1.cell_index(p16[g])
    lookup = mask[row, col]
    agree = float((lookup == truth16[g]).mean())
    print("16-ring points in the grid and the image:", int(g.sum()))
    print("labelled road by the mask:", int(lookup.sum()), " by the camera label:", int(truth16[g].sum()))
    print("mask lookup agrees with the camera label: %.1f %% of the points" % (100 * agree))
    extra = int((lookup & ~truth16[g]).sum())
    missed = int((~lookup & truth16[g]).sum())
    print("disagreements: road by lookup only %d, road by camera only %d" % (extra, missed))

    print("\n== 6. The time budget ==")
    ms = 12
    print("12 ms per frame = %.1f frames per second" % (1000 / ms))


if __name__ == "__main__":
    main()
