"""Figure 3: global average pooling with two roundings vs one.

Values: no_divider.py, section 2: global 7x7 pool, s_in 0.05 -> s_out 0.02:
mean multiplier floor(2^20 / 49) = 21399; two roundings 60.51 % differ from exact (max 2); one rounding 0 %.
"""

from figkit import row
from svglib import AMBER, EMERALD, PINK, Diagram

d = Diagram(width=1100)
d.header("Global average pooling: one rounding too many",
         "Int8 on Silicon  ·  part 4  ·  mean over 7x7 = 49 values  ·  s_in 0.05 -> s_out 0.02",
         params="100,000 random 7x7 inputs, compared with exact rounding")

y = 116
d.panel(y, 170, 1, "What does the global pool kernel do?",
        "It rounds the mean in input units, then rescales by s_in / s_out = 2.5, so the first rounding error grows "
        "2.5x.")
row(d, y + 74, [
    ("round 1: mean", ["sum * floor(2^20 / 49)", ">> 20, in input units"], AMBER),
    ("round 2: rescale", ["(mean - z_in) * alpha", ">> 20, + z_out"], AMBER),
    ("vs exact", ["60.5 % of outputs differ", "by up to 2"], PINK),
], AMBER[0])

y += 190
d.panel(y, 170, 2, "What does one rounding give?",
        "Fold 1/49 and s_in / s_out into one alpha, and round once.")
row(d, y + 74, [
    ("one alpha", ["s_in / (s_out * 49)", "* 2^20"], AMBER),
    ("one rounding", ["(sum - 49 z_in) * alpha", ">> 20, + z_out"], AMBER),
    ("vs exact", ["0 % of outputs differ", ""], EMERALD),
], AMBER[0])

y += 190
d.legend(30, y, [(AMBER, "a rounding step"), (PINK, "two roundings"), (EMERALD, "one rounding")])
y = d.footnotes(30, y + 40, ["Global pool: s_in = 0.05, z_in = -10, s_out = 0.02, z_out = 5."])
d.save(height=y + 10)
