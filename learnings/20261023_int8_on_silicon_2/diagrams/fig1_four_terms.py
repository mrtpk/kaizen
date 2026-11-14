"""Figure 1: where the zero points go. The four terms of sum (q_in - z_in)(q_w - z_w).

Values: zero_points.py (3x3 window, z_in = -64, z_w = 0, sum of weights = -412, bias = 1500).
"""

from svglib import AMBER, BLUE, EMERALD, NEUTRAL, Diagram

d = Diagram(width=1100)
d.header("Where do the zero points go?",
         "Int8 on Silicon  ·  part 2  ·  one 3x3 window  ·  z_in = -64  ·  z_w = 0",
         params="sum (q_in - z_in)(q_w - z_w) = sum q_in*q_w - z_w*sum q_in - z_in*sum q_w + N*z_in*z_w")

y: float = 116
d.panel(y, 290, 1, "What happens to each of the four terms?",
        "One is the MAC, one is a constant for the bias, and two are zero for int8 weights.")
W2, H2 = 470, 76
cards = [
    (60, y + 76, "sum q_in * q_w = -705", ["the MAC computes it", "on raw int8 inputs"], BLUE),
    (570, y + 76, "-z_w * sum q_in = 0", ["int8 weights are symmetric:", "z_w = 0"], NEUTRAL),
    (60, y + 180, "-z_in * sum q_w = -26368", ["depends only on the weights:", "fold it into the bias"], AMBER),
    (570, y + 180, "N * z_in * z_w = 0", ["z_w = 0 again"], NEUTRAL),
]
for x, cy, title, lines, col in cards:
    d.box(x, cy, W2, H2, title, lines, col=col, tsize=14, mono_size=12.5)

y += 310
d.panel(y, 170, 2, "What does the hardware see?",
        "A raw int8 MAC and a new bias. The input zero point is gone.")
BW = 290
d.box(60, y + 74, BW, 72, "folded bias", ["1500 - (-64)(-412)", "= -24868"], col=AMBER, mono_size=12.5)
d.arrow(60 + BW, y + 110, 410 - 2, y + 110, AMBER)
d.box(410, y + 74, BW, 72, "raw MAC + folded bias", ["-705 + (-24868)", "= -25573"], col=BLUE, mono_size=12.5)
d.arrow(410 + BW, y + 110, 760 - 2, y + 110, BLUE)
d.box(760, y + 74, BW, 72, "reference", ["-27073 + 1500", "= -25573  (equal)"], col=EMERALD, mono_size=12.5)

y += 190
d.legend(30, y, [(BLUE, "runtime MAC"), (AMBER, "compile-time constant"), (NEUTRAL, "zero for int8 weights"),
                 (EMERALD, "matches the reference")])
y = d.footnotes(
    30, y + 40, ["Reference: the sum with (q_in - z_in) inside the loop, as in TFLite's reference convolution."])
d.save(height=y + 10)
