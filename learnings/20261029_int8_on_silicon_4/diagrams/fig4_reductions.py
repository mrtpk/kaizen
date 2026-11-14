"""Figure 4: three averages, three ways to run them without a divider.

Values: no_divider.py (global pool over 7x7 = 49: mean multiplier floor(2^20 / 49) = 21399);
average pool and mean over channels as convolutions with all-ones weights (part 4 text).
"""

from figkit import row, table, table_height
from svglib import AMBER, BLUE, EMERALD, INK, NEUTRAL, PINK, Diagram

d = Diagram(width=1100)
d.header("Three averages, no divider",
         "Int8 on Silicon  ·  part 4  ·  average pool, mean over channels, global average pool",
         params="divide by N  ->  multiply by 1/N, and fold 1/N into alpha")

y: float = 116
d.panel(y, 170, 1, "How is a mean over C channels a convolution?",
        "A 1x1 convolution with C input channels, one output channel, and every weight equal to 1.")
row(d, y + 74, [
    ("C values at one pixel", ["q[0], q[1], ... q[C-1]", "one int8 value per channel"], BLUE),
    ("1x1 conv, weights = 1", ["sum of the C values", "minus C * z_in (bias)"], BLUE),
    ("x alpha, >> 20", ["alpha = s_in / (s_out C)", "      * 2^20"], AMBER),
], NEUTRAL[0])

y += 190
rows = [
    ("average", "runs as", "the 1/N goes into", "roundings"),
    ("average pool k x k", "depthwise conv, all-ones k x k weights", "alpha, N = k * k", "1"),
    ("mean over C channels", "1x1 conv, all-ones weights", "alpha, N = C", "1"),
    ("global average pool", "its own kernel: sum, x floor(2^20 / N)", "a mean multiplier, 21399 for 7x7", "2"),
]
H = table_height(len(rows))
d.panel(y, H, 2, "Which average used which kernel?",
        "Two of them are plain convolutions. Global pooling had its own kernel and one rounding too many.")
table(d, y + 92, rows, [80, 330, 700, 960],
      lambda i, j, v: (PINK if v == "2" else EMERALD) if j == 3 else INK)

y += H + 20
d.legend(30, y, [(BLUE, "MAC array"), (AMBER, "alpha"), (EMERALD, "one rounding"), (PINK, "two roundings")])
y = d.footnotes(
    30, y + 40, ["The MAC array sums, the post-processing unit divides. No new hardware for any of the three."])
d.save(height=y + 10)
