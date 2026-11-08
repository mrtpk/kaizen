"""Figure 3: road cells by distance, sparse vs dense.

Values: labels_top_view.py, section 5: 6-16 m 2089 / 9043 (x4.3), 16-26 m 743 / 8819 (x11.9),
26-36 m 344 / 6868 (x20.0), 36-46 m 143 / 3671 (x25.7) road cells (sparse / dense).
"""

import numpy as np
from _lrd import hbars, script
from svglib import AMBER, GRAY_MID, Diagram

s = script("20261108_lidar_road_detection_2", "labels_top_view")
points, calib, road = s.load()
inside, on_road = s.read_labels(points, calib, road)
sparse = s.grid_cells(points[on_road])
road_pts, kept = s.dense_road(points, calib, road)
dense = s.grid_cells(np.vstack([road_pts[:, :4], kept[:, :4]]))

rows = []
ratios = []
for near, far in [(6, 16), (16, 26), (26, 36), (36, 46)]:
    r0, r1 = round((s.X_MAX - far) / s.CELL), round((s.X_MAX - near) / s.CELL)
    a, b = int(sparse[r0:r1].sum()), int(dense[r0:r1].sum())
    ratios.append(b / a)
    rows.append((f"{near}-{far} m, points alone", a, str(a), GRAY_MID))
    rows.append((f"{near}-{far} m, with pseudo", b, str(b), AMBER[0]))

d = Diagram(width=1100)
d.header("Pseudo points help most where the rings are far apart",
         "LiDAR Road Detection  ·  part 2  ·  road cells per 10 m band",
         params="each band is 100 rows x 200 columns = 20000 cells")
y1 = 116
d.panel(y1, 300, 1, "How many road cells does each band get?",
        f"Near the car, {ratios[0]:.1f} times more road cells; at 36-46 m, {ratios[-1]:.1f} times more.")
hbars(d, 60, y1 + 74, rows, vmax=10000, width=560, label_w=240)
y = y1 + 320
d.legend(30, y, [(GRAY_MID, "road cells from points alone"), (AMBER, "road cells with pseudo points")])
end = d.footnotes(30, y + 40, ["One KITTI scan. Pseudo points are kept only when they also land on the road label."])
d.save(height=end + 10)
