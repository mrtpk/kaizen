"""Figure 0: the map of the series.

Values: the part titles of the series; the grid size from top_view_grid.py, section 2.
"""

from svglib import AMBER, BLUE, EMERALD, MUTE, VIOLET, Diagram

d = Diagram(width=1100)
d.header("LiDAR Road Detection: the map of the series",
         "LiDAR Road Detection  ·  part 1  ·  five parts, from a point cloud to road points",
         params="method: LoDNN (Caltagirone et al.)  ·  grid 400 x 200 cells of 0.1 m  ·  built at Infosys, 2018")

BW, BH = 300, 78
XS = [60, 400, 740]


def row(y, boxes):
    for i, (title, lines, col) in enumerate(boxes):
        d.box(XS[i], y, BW, BH, title, lines, col=col, mono_size=12)
        if i < len(boxes) - 1:
            d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, MUTE)


y1 = 116
d.panel(y1, 176, 1, "What goes into the network?",
        "Parts 1 and 2 build the input images and the labels on the same top-view grid.")
row(y1 + 76, [
    ("1. Points to a grid", ["scan -> 6 top-view images", "count, reflectance, z stats"], BLUE),
    ("2. Labels for a top view", ["camera label -> points", "-> top-view road cells"], AMBER),
])

y2 = y1 + 196
d.panel(y2, 176, 2, "What comes out, and what held up?",
        "Parts 3 to 5 cover the network, the 16-beam sensor, and a scorecard.")
row(y2 + 76, [
    ("3. Seeing far", ["dilated context module,", "view larger than the grid"], VIOLET),
    ("4. 64 beams to 16", ["sparser grid, and the", "mask back to points"], EMERALD),
    ("5. What held up", ["the checks again,", "and what was lost"], MUTE),
])

y = y2 + 196
d.legend(30, y, [(BLUE, "input"), (AMBER, "labels"), (VIOLET, "network"), (EMERALD, "sensor and output")])
end = d.footnotes(30, y + 40, ["The method is LoDNN by Caltagirone et al. The series re-builds its steps with new scripts on one public KITTI scan."])
d.save(height=end + 10)
