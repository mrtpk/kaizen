"""Figure 4: how sparse the grid is, by distance.

Values: top_view_grid.py, section 5: 7536 of 80000 cells occupied (9.4 %);
6-16 m 25.4 %, 16-26 m 7.2 %, 26-36 m 3.3 %, 36-46 m 1.8 %.
"""

from _lrd1 import hbars, script
from svglib import BLUE, Diagram

g = script("20261108_lidar_road_detection_1", "top_view_grid")
occ = g.top_view(g.load_points())[0] > 0
rows = []
for near, far in [(6, 16), (16, 26), (26, 36), (36, 46)]:
    r0, r1 = round((g.X_MAX - far) / g.CELL), round((g.X_MAX - near) / g.CELL)
    v = 100 * float(occ[r0:r1].mean())
    rows.append((f"{near}-{far} m ahead", v, f"{v:.1f} %", BLUE[0]))

d = Diagram(width=1100)
d.header("Most cells are empty, and more so far away",
         "LiDAR Road Detection  ·  part 1  ·  occupied cells of one scan",
         params=f"occupied: {int(occ.sum())} of {occ.size} cells = {100 * occ.mean():.1f} %")
y1 = 116
d.panel(y1, 190, 1, "Which share of the cells holds at least one point?",
        "A quarter near the car, under 2 % at the far edge.")
hbars(d, 60, y1 + 74, rows, vmax=30, width=600, label_w=180)
y = y1 + 210
d.legend(30, y, [(BLUE, "share of occupied cells in a 10 m band")])
end = d.footnotes(30, y + 40, ["One 64-beam KITTI scan. Each band is 100 rows x 200 columns."])
d.save(height=end + 10)
