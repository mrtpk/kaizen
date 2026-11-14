"""Figure 2: from points to six images in one pass, and the overwrite trap.

Values: top_view_grid.py, sections 2 and 4: 15515 points in the grid,
4078 cells with more than one point, naive count 7536, np.add.at count 15515.
"""

import numpy as np
from _lrd1 import script
from svglib import AMBER, BLUE, EMERALD, MUTE, PINK, Diagram

g = script("20261108_lidar_road_detection_1", "top_view_grid")
pts = g.load_points()
p = pts[g.in_grid(pts)]
row, col = g.cell_index(p)
naive = np.zeros((g.ROWS, g.COLS))
naive[row, col] += 1
proper = np.zeros((g.ROWS, g.COLS))
np.add.at(proper, (row, col), 1)

d = Diagram(width=1100)
d.header("Points to six images: one flat index, then group-by",
         "LiDAR Road Detection  ·  part 1  ·  the vectorised top view",
         params="flat = row * 200 + col  ·  np.bincount  ·  np.maximum.at  ·  np.minimum.at")

BW, BH = 300, 92
XS = [60, 400, 740]


def row3(y, boxes):
    for i, (t, lines, c) in enumerate(boxes):
        d.box(XS[i], y, BW, BH, t, lines, col=c, mono_size=12)
        if i < 2:
            d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, MUTE)


y1 = 116
d.panel(y1, 190, 1, "How do six images come from one pass?",
        "Give every point one flat cell index, then add up per index.")
row3(y1 + 76, [
    ("points in the grid", [f"{len(p)} points", "x, y, z, reflectance"], EMERALD),
    ("flat cell index", ["row * 200 + col", "0 .. 79999"], BLUE),
    ("group by cell", ["bincount: count, sums", "maximum.at, minimum.at"], BLUE),
])

y2 = y1 + 210
d.panel(y2, 190, 2, "Why not grid[row, col] += 1?",
        "Fancy-index assignment writes a repeated cell once. np.add.at adds every point.")
row3(y2 + 76, [
    ("cells with > 1 point", [f"{int((proper > 1).sum())} cells", "repeated indices"], AMBER),
    ("grid[row, col] += 1", [f"counts {int(naive.sum())} points", "repeats are lost"], PINK),
    ("np.add.at(grid, ..., 1)", [f"counts {int(proper.sum())} points", "every point is added"], EMERALD),
])

y = y2 + 210
d.legend(30, y, [(EMERALD, "correct result"), (BLUE, "vectorised step"), (AMBER, "the cause"), (PINK, "the trap")])
end = d.footnotes(30, y + 40, ["Counts from the sample scan. The vectorised images equal a plain Python loop over the "
                               "cells."])
d.save(height=end + 10)
