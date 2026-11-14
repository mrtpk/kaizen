"""Figure 2: receptive field growth of the context module against its input size.

Values: context_module.py, sections 1 to 3: receptive fields 3 x 3 ... 129 x 255;
context input 100 x 200; 1.29 and 1.27 times the input; whole network 266 x 518 cells.
"""

from _lrd3 import script
from svglib import AMBER, INK, VIOLET, Diagram

c = script("20261108_lidar_road_detection_3", "context_module")
rf = c.receptive_fields(c.DILATIONS)

d = Diagram(width=1100)
d.header("Seven layers grow a view larger than the input",
         "LiDAR Road Detection  ·  part 3  ·  receptive field of the context module, in cells of its input",
         params=f"context input {c.CTX_W} x {c.CTX_H} (width x height)  ·  last layer {rf[-1][0]} x {rf[-1][1]}")

y1 = 116
LX, BX, SCALE = 60, 200, 2.6
d.panel(y1, 470, 1, "How wide and how tall is the view after each layer?",
        f"After layer 7: {rf[-1][0] / c.CTX_W:.2f} times the input width, {rf[-1][1] / c.CTX_H:.2f} times its height.")
top = y1 + 84
for i, (w, h) in enumerate(rf, start=1):
    yy = top + (i - 1) * 50
    d.text(LX + 120, yy + 13, f"layer {i}  width", 12.5, INK, anchor="end")
    d.rect(BX, yy, w * SCALE, 16, fill=AMBER[0], stroke=None, rx=3)
    d.text(BX + w * SCALE + 8, yy + 13, str(w), 12.5, INK, mono=True)
    d.text(LX + 120, yy + 33, "height", 12.5, INK, anchor="end")
    d.rect(BX, yy + 20, h * SCALE, 16, fill=VIOLET[0], stroke=None, rx=3)
    d.text(BX + h * SCALE + 8, yy + 33, str(h), 12.5, INK, mono=True)
bottom = top + 7 * 50
for v, label, col in [(c.CTX_W, f"input width {c.CTX_W}", AMBER[0]), (c.CTX_H, f"input height {c.CTX_H}", VIOLET[0])]:
    x = BX + v * SCALE
    d.line(x, top - 14, x, bottom - 8, col, sw=1.4, dash="5 4")
    d.text(x, top - 20, label, 12, col, "bold", "middle")

y = y1 + 490
d.legend(30, y, [(AMBER, "width (across the road)"), (VIOLET, "height (distance ahead)")])
end = d.footnotes(30, y + 40, [
    "Dashed lines: the size of the context input, after one 2 x 2 pooling of the 200 x 400 grid.",
    f"The field is centred on its output: an output at the far edge reaches {(rf[-1][1] - 1) // 2 + 1} of the "
    f"{c.CTX_H} rows.",
])
d.save(height=end + 10)
