"""Figure 2: occupied cells per band, 64 rings vs 16 rings.

Values: fewer_beams.py, section 3: 9.4 % -> 2.6 %; 6-16 m 25.4 / 7.1, 16-26 m 7.2 / 1.8,
26-36 m 3.3 / 0.7, 36-46 m 1.8 / 0.7 (% of cells, 64 / 16 rings).
"""

import contextlib
import io

from _lrd import hbars, script
from svglib import BLUE, EMERALD, Diagram

with contextlib.redirect_stdout(io.StringIO()):
    f = script("20261108_lidar_road_detection_4", "fewer_beams")
pts = f.part1.load_points()
ring = f.rings(pts)
occ64 = f.part1.top_view(pts)[0] > 0
occ16 = f.part1.top_view(pts[ring % 4 == 0])[0] > 0
a, b = f.band_occupancy(occ64), f.band_occupancy(occ16)
rows = []
for (near, far), va, vb in zip(f.BANDS, a, b):
    rows.append((f"{near}-{far} m, 64 rings", va, f"{va:.1f} %", BLUE[0]))
    rows.append((f"{near}-{far} m, 16 rings", vb, f"{vb:.1f} %", EMERALD[0]))

d = Diagram(width=1100)
d.header(f"A quarter of the rings: {100 * occ16.mean():.1f} % of the cells instead of {100 * occ64.mean():.1f} %",
         "LiDAR Road Detection  ·  part 4  ·  occupied cells of the 400 x 200 grid",
         params=f"all cells: 64 rings {100 * occ64.mean():.1f} %  ·  16 rings {100 * occ16.mean():.1f} %")
y1 = 116
d.panel(y1, 300, 1, "Which share of the cells holds a point?",
        f"Near the car {b[0]:.1f} % instead of {a[0]:.1f} %; at the far edge {b[-1]:.1f} % instead of {a[-1]:.1f} %.")
hbars(d, 60, y1 + 74, rows, vmax=30, width=600, label_w=200)
y = y1 + 320
d.legend(30, y, [(BLUE, "64 rings"), (EMERALD, "16 rings (every 4th)")])
end = d.footnotes(30, y + 40, ["Every cell of the 16-ring grid is also occupied in the 64-ring grid. One KITTI scan."])
d.save(height=end + 10)
