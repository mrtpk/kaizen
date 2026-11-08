"""LiDAR Road Detection, part 1: from points to a top-view grid.

Turns one KITTI scan (the sample frame of lidar-buster) into the six top-view
images of LoDNN: number of points, mean reflectance, and min / max / mean / std
of the elevation, on a 400 x 200 grid of 0.1 m cells.
Every number in the post comes from this script.

Run:  uv run --with numpy python top_view_grid.py
The sample frame is downloaded once into ~/.cache/kaizen-lidar/. To use a local
copy, set LIDAR_SAMPLE_DIR to the folder that holds umm_000015.bin.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np

SAMPLE_URL = ("https://raw.githubusercontent.com/mrtpk/lidar-buster/"
              "1bb1d59d2d470af4c08350cbd912081728cd01eb/tutorial/sample_data/")

# The grid of the LoDNN paper: 40 m ahead, 20 m wide, 0.1 m cells.
X_MIN, X_MAX = 6.0, 46.0      # forward, metres
Y_MIN, Y_MAX = -10.0, 10.0    # left is positive, metres
CELL = 0.1                    # metres
ROWS = round((X_MAX - X_MIN) / CELL)   # 400
COLS = round((Y_MAX - Y_MIN) / CELL)   # 200
CHANNELS = ["number of points", "mean reflectance", "max elevation",
            "min elevation", "mean elevation", "std of elevation"]


def sample_file(name: str) -> Path:
    """Return a local path to a lidar-buster sample file; download it once."""
    local = os.environ.get("LIDAR_SAMPLE_DIR")
    if local:
        return Path(local) / name
    cache = Path.home() / ".cache" / "kaizen-lidar"
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if not path.exists():
        urllib.request.urlretrieve(SAMPLE_URL + name, path)
    return path


def load_points() -> np.ndarray:
    """KITTI scan: N x 4 float32 (x, y, z, reflectance)."""
    return np.fromfile(sample_file("umm_000015.bin"), dtype=np.float32).reshape(-1, 4)


def in_grid(points: np.ndarray) -> np.ndarray:
    """Boolean mask of the points inside the grid area."""
    x, y = points[:, 0], points[:, 1]
    return (x > X_MIN) & (x <= X_MAX) & (y > Y_MIN) & (y <= Y_MAX)


def cell_index(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Row 0 is the far edge (46 m), column 0 is the left edge (+10 m)."""
    row = np.floor((X_MAX - points[:, 0]) / CELL).astype(int)
    col = np.floor((Y_MAX - points[:, 1]) / CELL).astype(int)
    return np.clip(row, 0, ROWS - 1), np.clip(col, 0, COLS - 1)


def top_view(points: np.ndarray) -> np.ndarray:
    """Six 400 x 200 images, vectorised. Empty cells are 0 in every channel."""
    p = points[in_grid(points)]
    row, col = cell_index(p)
    flat = row * COLS + col
    n = ROWS * COLS
    z = p[:, 2].astype(np.float64)
    r = p[:, 3].astype(np.float64)

    count = np.bincount(flat, minlength=n).astype(np.float64)
    occupied = count > 0
    safe = np.where(occupied, count, 1.0)
    mean_r = np.bincount(flat, weights=r, minlength=n) / safe
    mean_z = np.bincount(flat, weights=z, minlength=n) / safe
    mean_z2 = np.bincount(flat, weights=z * z, minlength=n) / safe
    std_z = np.sqrt(np.clip(mean_z2 - mean_z ** 2, 0.0, None))
    max_z = np.full(n, -np.inf)
    min_z = np.full(n, np.inf)
    np.maximum.at(max_z, flat, z)
    np.minimum.at(min_z, flat, z)

    images = np.stack([count, mean_r, max_z, min_z, mean_z, std_z])
    images[:, ~occupied] = 0.0
    return images.reshape(6, ROWS, COLS)


def top_view_loop(points: np.ndarray) -> np.ndarray:
    """The same six images with a plain Python loop over the cells."""
    p = points[in_grid(points)]
    row, col = cell_index(p)
    cells: dict[tuple[int, int], list[tuple[float, float]]] = {}
    for i in range(len(p)):
        cells.setdefault((int(row[i]), int(col[i])), []).append((float(p[i, 2]), float(p[i, 3])))
    images = np.zeros((6, ROWS, COLS))
    for (i, j), values in cells.items():
        z = np.array([v[0] for v in values])
        r = np.array([v[1] for v in values])
        images[:, i, j] = [len(z), r.mean(), z.max(), z.min(), z.mean(), z.std()]
    return images


def main() -> None:
    points = load_points()
    print("== 1. The scan ==")
    print("points in the scan:", len(points))
    print("reflectance range: %.2f to %.2f" % (points[:, 3].min(), points[:, 3].max()))

    print("\n== 2. The grid ==")
    print(f"grid: x {X_MIN:g} to {X_MAX:g} m, y {Y_MIN:g} to {Y_MAX:g} m, cell {CELL} m")
    print("rows x cols:", ROWS, "x", COLS, "=", ROWS * COLS, "cells")
    mask = in_grid(points)
    print("points inside the grid:", int(mask.sum()), "(%.1f %% of the scan)" % (mask.mean() * 100))
    # Worked example: one point and its cell.
    k = int(np.nonzero(mask)[0][0])
    pr, pc = cell_index(points[k:k + 1])
    print("example point x=%.2f y=%.2f z=%.2f -> row %d, col %d"
          % (points[k, 0], points[k, 1], points[k, 2], pr[0], pc[0]))

    print("\n== 3. Six images, vectorised vs loop ==")
    fast = top_view(points)
    slow = top_view_loop(points)
    for name, a, b in zip(CHANNELS, fast, slow):
        print(f"{name:18s} max |diff| = {np.abs(a - b).max():.2e}")
    print("Vectorised top view equals the loop. Is pass?", bool(np.allclose(fast, slow, atol=1e-5)))
    occ = fast[0] > 0
    for name, a in zip(CHANNELS, fast):
        print(f"{name:18s} range over occupied cells: {a[occ].min():.2f} .. {a[occ].max():.2f}")

    print("\n== 4. The overwrite trap ==")
    p = points[mask]
    row, col = cell_index(p)
    naive = np.zeros((ROWS, COLS))
    naive[row, col] += 1                      # repeated cells are written once
    proper = np.zeros((ROWS, COLS))
    np.add.at(proper, (row, col), 1)          # repeated cells are added up
    print("cells with more than one point:", int((proper > 1).sum()))
    print("naive count of points:", int(naive.sum()), " np.add.at count:", int(proper.sum()))

    print("\n== 5. How sparse is the grid? ==")
    occupied = fast[0] > 0
    print("occupied cells: %d of %d (%.1f %%)" % (occupied.sum(), ROWS * COLS, occupied.mean() * 100))
    print("most points in one cell:", int(fast[0].max()))
    for near, far in [(6, 16), (16, 26), (26, 36), (36, 46)]:
        r0, r1 = round((X_MAX - far) / CELL), round((X_MAX - near) / CELL)
        print(f"  {near:2d}-{far:2d} m: {occupied[r0:r1].mean() * 100:5.1f} % of cells occupied")
    zero_mean = occupied & (np.abs(fast[4]) < 0.05)
    print("occupied cells with |mean elevation| < 0.05 m:", int(zero_mean.sum()))


if __name__ == "__main__":
    main()
