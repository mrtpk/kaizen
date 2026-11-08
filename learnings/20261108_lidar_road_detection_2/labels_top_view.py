"""LiDAR Road Detection, part 2: labels for a top view.

KITTI gives the road label in the camera image. This script moves it to the
top-view grid: project each LiDAR point into the image, read the label under it,
add pseudo road points on small circles around each road point, and grid them.
Every number in the post comes from this script.

Run:  uv run --with numpy --with pillow python labels_top_view.py
The sample files are downloaded once into ~/.cache/kaizen-lidar/. To use a local
copy, set LIDAR_SAMPLE_DIR to the folder that holds them.
"""

import os
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

SAMPLE_URL = ("https://raw.githubusercontent.com/mrtpk/lidar-buster/"
              "1bb1d59d2d470af4c08350cbd912081728cd01eb/tutorial/sample_data/")
X_MIN, X_MAX, Y_MIN, Y_MAX, CELL = 6.0, 46.0, -10.0, 10.0, 0.1
ROWS, COLS = 400, 200


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


def read_calib(path: Path) -> dict[str, np.ndarray]:
    """KITTI calib file: 'key: v1 v2 ...' per line."""
    calib = {}
    for line in path.read_text().splitlines():
        key, _, value = line.partition(":")
        try:
            calib[key] = np.array([float(v) for v in value.split()])
        except ValueError:
            pass
    return calib


def project(points: np.ndarray, calib: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pixel column u, row v (floored) and depth of each point in camera 2.

    x_img = P2 . [R0 . Tr . [x, y, z, 1], 1]
    """
    tr = calib["Tr_velo_to_cam"].reshape(3, 4)
    r0 = calib["R0_rect"].reshape(3, 3)
    p2 = calib["P2"].reshape(3, 4)
    xyz1 = np.c_[points[:, :3], np.ones(len(points))].T
    cam = r0 @ (tr @ xyz1)
    img = p2 @ np.vstack([cam, np.ones(cam.shape[1])])
    depth = img[2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = np.floor(img[0] / depth)
        v = np.floor(img[1] / depth)
    return u, v, depth


def read_labels(points: np.ndarray, calib: dict[str, np.ndarray], road: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each point: is it inside the image, and is it on the road label?"""
    h, w = road.shape
    u, v, depth = project(points, calib)
    inside = (points[:, 0] >= 0) & (depth > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
    on_road = np.zeros(len(points), dtype=bool)
    on_road[inside] = road[v[inside].astype(int), u[inside].astype(int)]
    return inside, on_road


def ring_points(point: np.ndarray, r: float = 1.0, dr: float = 0.2, dphi: float = 10.0) -> np.ndarray:
    """Pseudo points on circles around one road point (same z and reflectance)."""
    phi = np.radians(np.arange(0.0, 360.0, dphi))
    radii = np.arange(dr, r, dr)
    rr, pp = np.meshgrid(radii, phi)
    out = np.repeat(point[None, :], rr.size, axis=0).astype(np.float64)
    out[:, 0] += (rr * np.cos(pp)).ravel()
    out[:, 1] += (rr * np.sin(pp)).ravel()
    return out


def grid_cells(points: np.ndarray) -> np.ndarray:
    """Boolean 400 x 200 image: cells that hold at least one of the points."""
    x, y = points[:, 0], points[:, 1]
    keep = (x > X_MIN) & (x <= X_MAX) & (y > Y_MIN) & (y <= Y_MAX)
    row = np.clip(np.floor((X_MAX - x[keep]) / CELL).astype(int), 0, ROWS - 1)
    col = np.clip(np.floor((Y_MAX - y[keep]) / CELL).astype(int), 0, COLS - 1)
    grid = np.zeros((ROWS, COLS), dtype=bool)
    grid[row, col] = True
    return grid


def load() -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    points = np.fromfile(sample_file("umm_000015.bin"), dtype=np.float32).reshape(-1, 4)
    calib = read_calib(sample_file("umm_000015.txt"))
    label = np.array(Image.open(sample_file("umm_road_000015.png")).convert("RGB"))
    road = label[:, :, 2] == 255          # magenta (255, 0, 255) is road, red is not road
    return points, calib, road


def dense_road(points: np.ndarray, calib: dict[str, np.ndarray], road: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Road points, and the pseudo points around them that also land on the road label."""
    inside, on_road = read_labels(points, calib, road)
    road_pts = points[on_road]
    pseudo = np.vstack([ring_points(p) for p in road_pts])
    p_inside, p_road = read_labels(pseudo, calib, road)
    return road_pts, pseudo[p_road]


def main() -> None:
    points, calib, road = load()
    h, w = road.shape

    print("== 1. The camera label ==")
    print("label image:", w, "x", h, "pixels")
    print("road pixels: %.1f %% of the image" % (road.mean() * 100))

    print("\n== 2. Project the points into the image ==")
    inside, on_road = read_labels(points, calib, road)
    print("points in the scan:", len(points))
    print("points that land inside the image:", int(inside.sum()))
    print("of these, on the road label:", int(on_road.sum()))
    tr = calib["Tr_velo_to_cam"].reshape(3, 4)
    print("Tr_velo_to_cam translation (m): %.3f %.3f %.3f" % tuple(tr[:, 3]))
    # Worked example: the first road point.
    k = int(np.nonzero(on_road)[0][0])
    u, v, depth = project(points[k:k + 1], calib)
    print("example road point x=%.2f y=%.2f z=%.2f -> pixel (u=%d, v=%d), depth %.2f m"
          % (points[k, 0], points[k, 1], points[k, 2], u[0], v[0], depth[0]))

    print("\n== 3. A sparse top-view label ==")
    seen = grid_cells(points[inside])
    sparse = grid_cells(points[on_road])
    print("grid cells with a labelled point:", int(seen.sum()))
    print("grid cells with a road point:", int(sparse.sum()))

    print("\n== 4. Pseudo points on circles ==")
    one = ring_points(points[k])
    n_dir, n_rad = len(np.arange(0.0, 360.0, 10.0)), len(np.arange(0.2, 1.0, 0.2))
    print("pseudo points per road point:", len(one), f"= {n_rad} radii (0.2 to 0.8 m) x {n_dir} directions (every 10 degrees)")
    road_pts, kept = dense_road(points, calib, road)
    print("pseudo points made:", len(road_pts) * len(one))
    print("pseudo points that land on the road label:", len(kept),
          "(%.1f %%)" % (100 * len(kept) / (len(road_pts) * len(one))))
    dense = grid_cells(np.vstack([road_pts[:, :4], kept[:, :4]]))
    print("road cells, sparse:", int(sparse.sum()), " dense:", int(dense.sum()),
          " (x%.1f)" % (dense.sum() / sparse.sum()))
    print("Every sparse road cell is still road in the dense label. Is pass?",
          bool(np.all(dense[sparse])))

    print("\n== 5. Where are the road cells? ==")
    for near, far in [(6, 16), (16, 26), (26, 36), (36, 46)]:
        r0, r1 = round((X_MAX - far) / CELL), round((X_MAX - near) / CELL)
        a, b = int(sparse[r0:r1].sum()), int(dense[r0:r1].sum())
        print(f"  {near:2d}-{far:2d} m: sparse {a:5d}  dense {b:5d} road cells  (x{b / a:.1f})")


if __name__ == "__main__":
    main()
