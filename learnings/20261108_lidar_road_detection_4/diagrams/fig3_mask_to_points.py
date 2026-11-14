"""Figure 3: from a top-view road map back to the points.

Values: fewer_beams.py, section 5: 3321 points of the 16-ring scan in the grid and the image;
2066 road by the mask lookup, 1914 road by the camera label; 95.4 % agree.
"""

import contextlib
import io

import numpy as np
from _lrd4 import script
from svglib import AMBER, BLUE, EMERALD, MUTE, VIOLET, Diagram

with contextlib.redirect_stdout(io.StringIO()):
    f = script("20261108_lidar_road_detection_4", "fewer_beams")
pts = f.part1.load_points()
ring = f.rings(pts)
p16 = pts[ring % 4 == 0]
_, calib, road = f.part2.load()
road_pts, kept = f.part2.dense_road(pts, calib, road)
mask = f.part2.grid_cells(np.vstack([road_pts[:, :4], kept[:, :4]]))
inside, truth = f.part2.read_labels(p16, calib, road)
g = f.part1.in_grid(p16) & inside
r, c = f.part1.cell_index(p16[g])
lookup = mask[r, c]
agree = 100 * float((lookup == truth[g]).mean())

d = Diagram(width=1100)
d.header("From the road map back to the points",
         "LiDAR Road Detection  ·  part 4  ·  the last step of the pipeline",
         params="label of a point = road map[row, col] of its cell")
BW, BH = 300, 92
XS = [60, 400, 740]


def row3(y0, boxes):
    for i, (t, lines, col) in enumerate(boxes):
        d.box(XS[i], y0, BW, BH, t, lines, col=col, mono_size=12)
        if i < 2:
            d.arrow(XS[i] + BW, y0 + BH / 2, XS[i + 1] - 2, y0 + BH / 2, MUTE)


y1 = 116
d.panel(y1, 190, 1, "What happens after the network?",
        "The road map labels the points, and the road points can be drawn on the camera image.")
row3(y1 + 76, [
    ("road map", ["400 x 200 cells", "probability of road"], VIOLET),
    ("points", ["cell of each point", "-> road or not road"], BLUE),
    ("camera image", ["project road points", "for display"], EMERALD),
])
y2 = y1 + 210
d.panel(y2, 190, 2, "How well does a grid label the points?",
        f"With the dense label of part 2 as the map, the lookup agrees with the camera label for {agree:.1f} % of the "
        "points.")
row3(y2 + 76, [
    ("16-ring points", [f"{int(g.sum())} in the grid", "and in the image"], BLUE),
    ("road by lookup", [f"{int(lookup.sum())} points", "from the grid cell"], AMBER),
    ("road by camera", [f"{int(truth[g].sum())} points", f"agree: {agree:.1f} %"], AMBER),
])
y = y2 + 210
d.legend(30, y, [(VIOLET, "network output"), (BLUE, "points"), (AMBER, "road labels"), (EMERALD, "display")])
end = d.footnotes(30, y + 40, ["Panel 2 uses the label of part 2 in place of a trained network, so it measures the "
                               "grid step only."])
d.save(height=end + 10)
