"""Figure 1: the scorecard of the series.

Values: lessons.py: grid 400 x 200, 15515 of 116044 points; 9.4 % occupied; road cells 3319 -> 28401;
receptive field 129 x 255 on 100 x 200; 16 rings 2.6 %; all four checks True.
"""

import contextlib
import io

import numpy as np
from _lrd import script
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram

with contextlib.redirect_stdout(io.StringIO()):
    les = script("20261108_lidar_road_detection_5", "lessons")
p1, p2, p3, p4 = les.part1, les.part2, les.part3, les.part4
pts = p1.load_points()
grid = p1.top_view(pts)
occ = grid[0] > 0
_, calib, road = p2.load()
road_pts, kept = p2.dense_road(pts, calib, road)
sparse = p2.grid_cells(road_pts)
dense = p2.grid_cells(np.vstack([road_pts[:, :4], kept[:, :4]]))
w, h = p3.receptive_fields(p3.DILATIONS)[-1]
ring = p4.rings(pts)
occ16 = p1.top_view(pts[ring % 4 == 0])[0] > 0

rows = [
    ("1", "top-view grid", f"{p1.ROWS} x {p1.COLS} cells; {occ.mean() * 100:.1f} % occupied",
     "vectorised grid = loop", bool(np.allclose(grid, p1.top_view_loop(pts), atol=1e-5)), BLUE),
    ("2", "labels", f"road cells {int(sparse.sum())} -> {int(dense.sum())}",
     "sparse road stays road", bool(np.all(dense[sparse])), AMBER),
    ("3", "network", f"receptive field {w} x {h} on {p3.CTX_W} x {p3.CTX_H}",
     "field larger than input", w > p3.CTX_W and h > p3.CTX_H, VIOLET),
    ("4", "16 beams", f"occupied {occ16.mean() * 100:.1f} % instead of {occ.mean() * 100:.1f} %",
     "16-ring cells within 64", bool(np.all(occ[occ16])), EMERALD),
]

d = Diagram(width=1100)
d.header("The series in four checks",
         "LiDAR Road Detection  ·  part 5  ·  lessons.py re-runs the checks of parts 1 to 4",
         params="one KITTI scan  ·  all numbers from the scripts of the series")
y1 = 116
d.panel(y1, 330, 1, "What did each part show, and does its check still hold?",
        "All four checks print True.")
cols = [70, 110, 290, 680]
for x, t in zip(cols[1:], ["part", "what it showed", "check"]):
    d.text(x, y1 + 86, t, 12.5, INK, "bold")
for i, (n, name, value, check, ok, col) in enumerate(rows):
    yy = y1 + 104 + i * 52
    d.rect(60, yy, 980, 42, fill=col[1], stroke=col[0], rx=8, sw=1.2)
    d.badge(84, yy + 21, n, col[0], r=11, size=11)
    d.text(cols[1], yy + 26, name, 13, col[0], "bold")
    d.text(cols[2], yy + 26, value, 12, INK, mono=True)
    d.text(cols[3], yy + 26, check, 12.5, INK)
    d.pill(920, yy + 11, 100, f"{ok}", EMERALD if ok else MUTE)
y = y1 + 350
d.legend(30, y, [(BLUE, "input"), (AMBER, "labels"), (VIOLET, "network"), (EMERALD, "sensor")])
end = d.footnotes(30, y + 40, ["The results on the cart (94 % F1, 95 % average accuracy, 83 FPS) are not in this table: no script here can reproduce them."])
d.save(height=end + 10)
