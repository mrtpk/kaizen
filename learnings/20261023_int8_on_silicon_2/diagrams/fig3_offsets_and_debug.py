"""Figure 3: where the input offset and the weight offset go.

Values: zero_points.py (weights with z_w = 3 -> q_w - z_w in [-131, 124]; [-255, 255] in general).
Counts: one 3x3 convolution over 64 input channels has 3 * 3 * 64 = 576 multiply-accumulates per output.
"""

from figkit import row
from svglib import AMBER, BLUE, EMERALD, NEUTRAL, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Moving work from the hardware to the compiler",
         "Int8 on Silicon  ·  part 2  ·  the input offset and the weight offset",
         params="one 3x3 convolution over 64 input channels: 576 multiply-accumulates per output")

y = 116
d.panel(y, 170, 1, "Where is the input offset added?",
        "Inside the loop it is one addition per multiply. Folded into the bias it is one addition per output.")
row(d, y + 74, [
    ("offset in the loop", ["(q_in - z_in) * q_w", "576 additions per output"], PINK),
    ("offset folded", ["bias' = bias - z_in * sum(q_w)", "computed once, at compile time"], AMBER),
    ("hardware sees", ["raw q_in * q_w, then", "1 bias addition per output"], EMERALD),
], NEUTRAL[0])

y += 190
d.panel(y, 170, 2, "What about a weight zero point?",
        "Subtract it from the weights before writing them. The MAC never sees it, but the weights need one more bit.")
row(d, y + 74, [
    ("weights with z_w = 3", ["q_w in [-128, 127]", ""], BLUE),
    ("written for the hardware", ["q_w - z_w", "in [-131, 124]"], AMBER),
    ("in general", ["q_w - z_w in [-255, 255]", "9 bits instead of 8"], VIOLET),
], NEUTRAL[0])

y += 190
d.legend(30, y, [(PINK, "extra work in the MAC loop"), (AMBER, "compile-time constants"),
         (EMERALD, "what the hardware does"), (BLUE, "stored weights"), (VIOLET, "cost")])
y = d.footnotes(
    30, y + 40, ["Int8 weights in TFLite are symmetric (z_w = 0), so the weight offset matters only for older uint8 "
                 "models."])
d.save(height=y + 10)
