"""Figure 3: where the offsets go, and how the before/after dumps split a bug in two.

Values: zero_points.py (weights with z_w = 3 -> q_w - z_w in [-131, 124]; [-255, 255] in general).
Counts: one 3x3 convolution over 64 input channels has 3 * 3 * 64 = 576 multiply-accumulates per output.
"""

from figkit import row
from svglib import AMBER, BLUE, EMERALD, NEUTRAL, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Moving work from the hardware to the compiler",
         "Int8 on Silicon  ·  part 2  ·  input offset, weight offset, and a debugging aid",
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
        "Subtract it from the weights offline. The MAC never sees it, but the stored weights need one more bit.")
row(d, y + 74, [
    ("weights with z_w = 3", ["q_w in [-128, 127]", ""], BLUE),
    ("stored offline", ["q_w - z_w", "in [-131, 124]"], AMBER),
    ("in general", ["q_w - z_w in [-255, 255]", "9 bits instead of 8"], VIOLET),
], NEUTRAL[0])

y += 190
d.panel(y, 170, 3, "A layer differs on the hardware: where is the bug?",
        "The reference model dumped the accumulator before and after the correction, for every layer.")
row(d, y + 74, [
    ("compare 'before'", ["raw MAC sums", "on hardware vs golden"], BLUE),
    ("before differs", ["MAC array or", "data movement"], PINK),
    ("only after differs", ["constants: bias, alpha,", "pad value, zero points"], AMBER),
])

y += 190
d.legend(30, y, [(PINK, "work or bug in the MAC path"), (AMBER, "compile-time constants"), (EMERALD, "what the hardware does"), (VIOLET, "cost")])
y = d.footnotes(30, y + 40, ["Int8 weights in TFLite are symmetric (z_w = 0), so the weight offset matters only for older uint8 models."])
d.save(height=y + 10)
