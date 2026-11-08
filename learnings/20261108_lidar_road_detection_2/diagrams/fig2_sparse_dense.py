"""Figure 2: the top-view label, from points alone and with pseudo points.

Values: labels_top_view.py, sections 3 and 4: 6565 labelled cells, 3319 road cells
from points alone, 28401 road cells with pseudo points (x8.6).
"""

import numpy as np
from _lrd import heatmap, script
from svglib import AMBER, BLUE, Diagram

s = script("20261108_lidar_road_detection_2", "labels_top_view")
points, calib, road = s.load()
inside, on_road = s.read_labels(points, calib, road)
seen = s.grid_cells(points[inside]).astype(float)
sparse = s.grid_cells(points[on_road]).astype(float)
road_pts, kept = s.dense_road(points, calib, road)
dense = s.grid_cells(np.vstack([road_pts[:, :4], kept[:, :4]])).astype(float)

d = Diagram(width=1100)
d.header("The top-view road label: sparse points, then pseudo points",
         "LiDAR Road Detection  ·  part 2  ·  one KITTI scan on the 400 x 200 grid",
         params="pseudo points: circles of radius 0.2, 0.4, 0.6, 0.8 m, every 10 degrees, around each road point")
XS = [110, 440, 770]
y1 = 116
d.panel(y1, 520, 1, "Which cells get a road label?",
        f"From points alone {int(sparse.sum())} cells; with pseudo points {int(dense.sum())} cells, "
        f"{dense.sum() / sparse.sum():.1f} times more.")
heatmap(d, XS[0], y1 + 96, seen, "cells with a labelled "
                                 "point", BLUE[0], px=1.0, lo=0, hi=1, note=f"{int(seen.sum())} cells")
heatmap(d, XS[1], y1 + 96, sparse, "road cells, points alone", AMBER[0], px=1.0, lo=0, hi=1,
        note=f"{int(sparse.sum())} cells")
heatmap(d, XS[2], y1 + 96, dense, "road cells, with pseudo "
                                  "points", AMBER[0], px=1.0, lo=0, hi=1, note=f"{int(dense.sum())} cells")
y = y1 + 540
d.legend(30, y, [(BLUE, "cell with a point inside the camera view"), (AMBER, "road cell")])
end = d.footnotes(30, y + 40, ["Drawn as blocks of 4 x 4 cells; a block is coloured when any of its cells is set. The "
                               "far edge (46 m) is at the top."])
d.save(height=end + 10)
