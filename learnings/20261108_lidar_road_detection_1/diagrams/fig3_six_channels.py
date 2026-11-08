"""Figure 3: the six top-view images of one scan.

Values: top_view_grid.py, top_view() on the sample scan; blocks of 4 x 4 cells (0.4 m).
"""

from _lrd import heatmap, script
from svglib import BLUE, Diagram

g = script("20261108_lidar_road_detection_1", "top_view_grid")
img = g.top_view(g.load_points())
occ_all = img[0] > 0
names = ["number of points", "mean reflectance", "max elevation",
         "min elevation", "mean elevation", "std of elevation"]

d = Diagram(width=1100)
d.header("Six images of one scan",
         "LiDAR Road Detection  ·  part 1  ·  the input of the network, one KITTI scan",
         params="400 x 200 cells, drawn as blocks of 4 x 4 cells  ·  gray = no point  ·  darker = larger value")

XS = [110, 440, 770]
y1 = 116
d.panel(y1, 520, 1, "What do the count, the reflectance and the max elevation show?",
        "Points pile up near the car; the far half of the grid is almost empty.")
for i in range(3):
    v = img[i]
    occ = v[occ_all]
    heatmap(d, XS[i], y1 + 96, v, names[i], BLUE[0], px=1.0, mask=occ_all,
            note=f"{occ.min():.2f} .. {occ.max():.2f}")

y2 = y1 + 540
d.panel(y2, 520, 2, "And the other elevation statistics?",
        "The ring pattern of the sensor is visible in every image; std is zero where a cell holds one point.")
for i in range(3, 6):
    v = img[i]
    occ = v[occ_all]
    heatmap(d, XS[i - 3], y2 + 96, v, names[i], BLUE[0], px=1.0, mask=occ_all,
            note=f"{occ.min():.2f} .. {occ.max():.2f}")

y = y2 + 540
d.legend(30, y, [(BLUE, "value of the cell (light = low, dark = high)")])
end = d.footnotes(30, y + 40, [
    "Each image is drawn with its own range (below it). Elevation is in metres in the LiDAR frame.",
    "A block shows the mean over its occupied cells. The far edge (46 m) is at the top.",
])
d.save(height=end + 10)
