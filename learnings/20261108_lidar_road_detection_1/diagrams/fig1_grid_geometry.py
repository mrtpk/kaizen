"""Figure 1: the top-view grid and how a point finds its cell.

Values: top_view_grid.py, section 2: grid x 6 to 46 m, y -10 to 10 m, cell 0.1 m,
400 x 200 cells; example point x=41.95 y=2.72 -> row 40, col 72.
"""

from _lrd import script
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, Diagram

g = script("20261108_lidar_road_detection_1", "top_view_grid")
pts = g.load_points()
k = int(g.in_grid(pts).nonzero()[0][0])
r, c = g.cell_index(pts[k:k + 1])
ex, ey = float(pts[k, 0]), float(pts[k, 1])

d = Diagram(width=1100)
d.header("The top-view grid: 40 m ahead, 20 m wide, 0.1 m cells",
         "LiDAR Road Detection  ·  part 1  ·  the grid of the LoDNN paper",
         params=f"x {g.X_MIN:g}..{g.X_MAX:g} m  ·  y {g.Y_MIN:g}..{g.Y_MAX:g} m  ·  cell {g.CELL} m  ·  {g.ROWS} rows "
         f"x {g.COLS} cols = {g.ROWS * g.COLS} cells")

y1 = 116
d.panel(y1, 520, 1, "Where is the grid?", "In front of the car: 6 m to 46 m ahead, 10 m to each side. Row 0 is the far "
                                          "edge.")
S = 8.0                      # pixels per metre
gx, gy = 120, y1 + 80        # top-left corner of the grid drawing (x = 46 m, y = +10 m)
gw, gh = 20 * S, 40 * S
d.rect(gx, gy, gw, gh, fill=BLUE[1], stroke=BLUE[0], rx=0, sw=1.5)
for m in range(0, 41, 10):
    yy = gy + m * S
    d.line(gx - 6, yy, gx, yy, INK)
    d.text(gx - 10, yy + 4, f"{46 - m} m", 12, INK, anchor="end", mono=True)
d.text(gx, gy - 8, "+10 m", 12, INK, anchor="middle", mono=True)
d.text(gx + gw, gy - 8, "-10 m", 12, INK, anchor="middle", mono=True)
car_y = gy + gh + 6 * S
d.rect(gx + gw / 2 - 12, car_y - 4, 24, 36, fill=AMBER[1], stroke=AMBER[0], rx=4, sw=1.5)
d.text(gx + gw / 2 + 22, car_y + 18, "LiDAR (x = 0)", 12, AMBER[0], "bold")
d.line(gx + gw / 2, gy + gh, gx + gw / 2, car_y - 4, LINE, dash="3 3")
px_, py_ = gx + (10 - ey) * S, gy + (46 - ex) * S
d.circle(px_, py_, 5, EMERALD[0])
d.line(px_, py_, gx + gw + 14, py_, EMERALD[0], dash="3 3")
d.text(gx + gw + 20, py_ + 4, f"point x={ex:.2f}, y={ey:.2f}", 12, EMERALD[0], "bold", mono=True)

tx = 560
d.text(tx, y1 + 100, "Axes (LiDAR frame)", 14, INK, "bold")
for i, s in enumerate(["x: forward, the distance ahead", "y: to the left", "z: up (the elevation)",
                       "", "rows: 40 m / 0.1 m = 400", "cols: 20 m / 0.1 m = 200",
                       "", "Only the points inside the box are used."]):
    d.text(tx, y1 + 126 + i * 22, s, 13, INK, mono=i in (4, 5))

y2 = y1 + 540
d.panel(y2, 170, 2, "How does a point find its cell?", "Two floors, one per axis. Far rows and left columns come "
                                                       "first.")
BW, BH = 300, 78
XS = [60, 400, 740]
boxes = [
    ("point", [f"x = {ex:.2f} m", f"y = {ey:.2f} m"], EMERALD),
    ("row = floor((46 - x) / 0.1)", [f"floor({46 - ex:.2f} / 0.1)", f"= {int(r[0])}"], BLUE),
    ("col = floor((10 - y) / 0.1)", [f"floor({10 - ey:.2f} / 0.1)", f"= {int(c[0])}"], BLUE),
]
for i, (t, lines, col) in enumerate(boxes):
    d.box(XS[i], y2 + 74, BW, BH, t, lines, col=col, mono_size=12.5)
    if i < 2:
        d.arrow(XS[i] + BW, y2 + 74 + BH / 2, XS[i + 1] - 2, y2 + 74 + BH / 2, MUTE)

y = y2 + 190
d.legend(30, y, [(BLUE, "grid"), (EMERALD, "a point of the scan"), (AMBER, "the sensor")])
end = d.footnotes(30, y + 40, ["Drawn to scale, 8 px per metre. The example is the first in-grid point of the sample "
                               "scan."])
d.save(height=end + 10)
