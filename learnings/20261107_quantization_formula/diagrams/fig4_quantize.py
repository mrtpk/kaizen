"""Figure 4: the int8 quantization formula for the float range [-1, 3].

Values: quantization.py -> scale = 0.015686, zero_point = -64;
-1.0 -> -128, 0.0 -> -64, 1.0 -> 0, 3.0 -> 127, 3.5 -> 127 (clipped).
"""

from _numberline import NumberLine
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, Diagram

S, Z = 4 / 255, -64

d = Diagram(width=1100)
d.header("Quantizing a float to int8",
         "Quantization Basics  ·  float range [-1, 3]  ·  signed int8 range [-128, 127]",
         params="scale = 0.015686    zero_point = -64")

# both lines share one x mapping: x(r) for floats, x(q) = x(scale * (q - zero_point)) for integers
R0, R1, X0, X1 = -1.25, 3.75, 200, 1020


def xr(r):
    return X0 + (r - R0) / (R1 - R0) * (X1 - X0)


y = 116
d.panel(y, 300, 1, "Where does each float land on the int8 line?",
        "Divide by scale, add zero_point, round, then clip to [-128, 127].")
fl = NumberLine(d, X0, X1, y + 120, R0, R1, [-1, 0, 1, 2, 3], name="float")
iy = y + 250
d.line(xr(S * (-128 - Z)), iy, xr(S * (127 - Z)), iy, EMERALD[0], sw=2)
d.text(X0 - 14, iy + 5, "int8", 13, INK, "bold", anchor="end")
for q in (-128, -64, 0, 64, 127):
    x = xr(S * (q - Z))
    d.line(x, iy - 6, x, iy + 6, EMERALD[0], sw=1.4)
    d.text(x, iy + 24, str(q), 12, MUTE, anchor="middle", mono=True)

pairs = [(-1.0, -128, BLUE), (0.0, -64, AMBER), (1.0, 0, BLUE), (3.0, 127, BLUE)]
for r, q, col in pairs:
    xf, xq = xr(r), xr(S * (q - Z))
    fl.point(r, f"{r:g}", col)
    d.arrow(xf, y + 148, xq, iy - 10, col, sw=1.3, dash="4 3")
    d.circle(xq, iy, 6, EMERALD[0])
d.text(xr(0) + 10, y + 192, "zero_point = -64", 12.5, AMBER, "bold", mono=True)

xc = xr(3.5)
fl.point(3.5, "3.5", PINK)
x127 = xr(S * (127 - Z))
d.path_arrow(f"M {xc:.1f},{y + 130} L {xc:.1f},{iy} L {x127 + 10:.1f},{iy}", PINK, sw=1.3, dash="4 3")
d.text(xc + 8, y + 190, "clipped", 12.5, PINK, "bold", mono=True)

y += 320
d.panel(y, 170, 2, "Which formulas give these numbers?", "Two parameters from the range, then one formula each way.")
lines = [
    "scale      = (r_max - r_min) / (127 - (-128))   = 4 / 255             = 0.015686",
    "zero_point = round(-128 - r_min / scale)         = round(-64.25)       = -64",
    "quantize   q = clip(round(r / scale) + zero_point, -128, 127)",
    "dequantize r = scale * (q - zero_point)",
]
for i, s in enumerate(lines):
    d.text(68, y + 82 + i * 22, s, 13, INK, mono=True)

y += 190
d.legend(30, y, [(BLUE, "float value"), (EMERALD, "int8 value"), (AMBER, "zero point (float 0.0)"), (PINK, "clipped")])
y = d.footnotes(30, y + 40, ["Rounding zero_point moves the int8 line by a quarter step, so -1.0 and 3.0 come back as -1.0039 and 2.9961."])
d.save(height=y + 10)
