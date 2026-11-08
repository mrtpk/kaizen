"""Figure 1: from a LiDAR point to a pixel of the camera label.

Values: labels_top_view.py, section 2: example road point x=58.91 y=0.06 z=-0.62 ->
pixel (609, 187), depth 58.63 m; 116044 points, 18969 inside the image, 7501 on the road label.
"""

import numpy as np
from _lrd import script
from svglib import AMBER, BLUE, EMERALD, MUTE, VIOLET, Diagram

s = script("20261108_lidar_road_detection_2", "labels_top_view")
points, calib, road = s.load()
inside, on_road = s.read_labels(points, calib, road)
k = int(np.nonzero(on_road)[0][0])
u, v, depth = s.project(points[k:k + 1], calib)
x, y, z = (float(c) for c in points[k, :3])

d = Diagram(width=1100)
d.header("From a LiDAR point to a pixel of the road label",
         "LiDAR Road Detection  ·  part 2  ·  KITTI calibration: Tr_velo_to_cam, R0_rect, P2",
         params="pixel = P2 · [ R0 · Tr · [x, y, z, 1] ; 1 ],  then divide by the depth")

BW, BH = 300, 92
XS = [60, 400, 740]


def row3(y0, boxes):
    for i, (t, lines, c) in enumerate(boxes):
        d.box(XS[i], y0, BW, BH, t, lines, col=c, mono_size=12)
        if i < 2:
            d.arrow(XS[i] + BW, y0 + BH / 2, XS[i + 1] - 2, y0 + BH / 2, MUTE)


y1 = 116
d.panel(y1, 190, 1, "Where does one road point land in the image?",
        "Move it to the camera frame, rectify it, project it, divide by the depth.")
row3(y1 + 76, [
    ("LiDAR point", [f"x={x:.2f} y={y:.2f} z={z:.2f}", "w = 1 (homogeneous)"], EMERALD),
    ("camera frame", ["Tr: 3 x 4, then R0: 3 x 3", f"depth = {float(depth[0]):.2f} m"], BLUE),
    ("pixel", ["P2: 3 x 4, / depth", f"u = {int(u[0])}, v = {int(v[0])}"], VIOLET),
])

y2 = y1 + 210
d.panel(y2, 190, 2, "How many points get a label?",
        "Only the points in front of the camera and inside the image can read the label.")
row3(y2 + 76, [
    ("all points", [f"{len(points)} points", "360 degrees around"], MUTE),
    ("inside the image", [f"{int(inside.sum())} points", "the camera view"], BLUE),
    ("on the road label", [f"{int(on_road.sum())} points", "magenta pixels"], AMBER),
])

y = y2 + 210
d.legend(30, y, [(EMERALD, "LiDAR frame"), (BLUE, "camera frame"), (VIOLET, "image"), (AMBER, "road")])
end = d.footnotes(30, y + 40, ["The example is the first point of the scan that lands on the road label. It is beyond "
                               "the 46 m grid."])
d.save(height=end + 10)
