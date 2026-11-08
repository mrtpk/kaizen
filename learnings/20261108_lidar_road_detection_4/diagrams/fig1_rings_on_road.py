"""Figure 1: where the rings cross the road ahead, 64 rings vs every 4th ring.

Values: fewer_beams.py, section 4: 64 rings -> 37 crossings from 6.1 to 41.5 m, largest gap 5.8 m;
16 rings -> 9 crossings from 6.5 to 36.6 m, largest gap 11.2 m.
"""

import contextlib
import io

import numpy as np
from _lrd import script
from svglib import BLUE, EMERALD, INK, LINE, MUTE, Diagram

with contextlib.redirect_stdout(io.StringIO()):
    f = script("20261108_lidar_road_detection_4", "fewer_beams")
pts = f.part1.load_points()
ring = f.rings(pts)
keep = ring % 4 == 0
h64 = f.corridor_hits(pts, ring)
h16 = f.corridor_hits(pts[keep], ring[keep])

d = Diagram(width=1100)
d.header("Fewer rings, wider gaps on the road ahead",
         "LiDAR Road Detection  ·  part 4  ·  where each ring crosses the corridor |y| < 1 m in front of the car",
         params="16 rings = every 4th ring of the 64-ring KITTI scan (same vertical field of view)")

X0, X1 = 140, 1040
S = (X1 - X0) / 40.0


def strip(y, hits, col, name):
    d.text(X0 - 14, y + 5, name, 13, INK, "bold", "end")
    d.line(X0, y, X1, y, LINE, sw=2)
    for h in hits:
        x = X0 + (h - 6) * S
        d.line(x, y - 12, x, y + 12, col, sw=2.4)
    g = np.diff(hits)
    i = int(np.argmax(g))
    xa, xb = X0 + (hits[i] - 6) * S, X0 + (hits[i + 1] - 6) * S
    d.line(xa, y + 24, xb, y + 24, col, sw=1.2, dash="4 3")
    d.text((xa + xb) / 2, y + 42, f"largest gap {g[i]:.1f} m", 12, col, "bold", "middle", mono=True)
    d.text(X1, y - 20, f"{len(hits)} crossings, {hits[0]:.1f} to {hits[-1]:.1f} m", 12, MUTE, anchor="end", mono=True)


y1 = 116
d.panel(y1, 290, 1, "Where do the rings hit the road ahead?",
        f"With 64 rings, {len(h64)} rings cross 6-46 m; with 16 rings, {len(h16)}. The gaps grow with distance.")
strip(y1 + 110, h64, BLUE[0], "64 rings")
strip(y1 + 200, h16, EMERALD[0], "16 rings")
for m in range(6, 47, 10):
    x = X0 + (m - 6) * S
    d.text(x, y1 + 272, f"{m} m", 12, MUTE, anchor="middle", mono=True)

y = y1 + 310
d.legend(30, y, [(BLUE, "a ring of the 64-ring scan"), (EMERALD, "a ring of the 16-ring scan")])
end = d.footnotes(30, y + 40, ["A crossing is the median distance of a ring's points with |y| < 1 m. One scan; a real 16-beam sensor also has another field of view."])
d.save(height=end + 10)
