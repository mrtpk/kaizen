"""Figure 3: global average pooling with two roundings, and bilinear resize weights.

Values: no_divider.py, sections 2 and 5: global 7x7 pool, s_in 0.05 -> s_out 0.02:
two roundings 60.51 % differ from exact (max 2), one rounding 0 %;
resize 8x8 -> 16x16: weights sum to 1024, 0 of 256 differ from TFLite;
8x8 -> 13x13: weights sum to 1021..1024, 26 of 169 differ by 1.
"""

from svglib import AMBER, EMERALD, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Two small leaks: double rounding and truncated weights",
         "Int8 on Silicon  ·  part 4  ·  global average pool and bilinear resize",
         params="mean over 7x7 = 49 values  ·  resize weights at 2^10")

BW, BH = 290, 72
XS = [60, 410, 760]
y = 116
d.panel(y, 170, 1, "Global average pool: why does one more rounding matter?",
        "The mean is rounded in input units, then scaled by s_in / s_out = 2.5, so its rounding error grows 2.5x.")
d.box(XS[0], y + 74, BW, BH, "round 1: mean", ["sum * floor(2^20 / 49)", ">> 20, in input units"], col=AMBER, mono_size=12)
d.arrow(XS[0] + BW, y + 110, XS[1] - 2, y + 110, AMBER)
d.box(XS[1], y + 74, BW, BH, "round 2: rescale", ["(mean - z_in) * alpha", ">> 20, + z_out"], col=AMBER, mono_size=12)
d.arrow(XS[1] + BW, y + 110, XS[2] - 2, y + 110, PINK)
d.box(XS[2], y + 74, BW, BH, "vs exact", ["two roundings: 60.5 %", "one rounding: 0 %"], col=PINK, mono_size=12)

y += 190
d.panel(y, 170, 2, "Bilinear resize: do the four weights add up to 1?",
        "Only when the weights are exact in 10 bits. Each weight was truncated on its own.")
d.box(XS[0], y + 74, BW, BH, "8x8 -> 16x16", ["weights 1/4, 3/4: exact", "sum = 1024"], col=EMERALD, mono_size=12)
d.box(XS[1], y + 74, BW, BH, "8x8 -> 13x13", ["weights truncated", "sum = 1021 .. 1024"], col=PINK, mono_size=12)
d.box(XS[2], y + 74, BW, BH, "vs TFLite integer", ["x2: 0 of 256 differ", "x13/8: 26 of 169, by 1"], col=VIOLET, mono_size=12)

y += 190
d.legend(30, y, [(AMBER, "a rounding step"), (PINK, "the leak"), (EMERALD, "exact"), (VIOLET, "TFLite")])
y = d.footnotes(30, y + 40, [
    "Global pool: s_in = 0.05, z_in = -10, s_out = 0.02, z_out = 5, 100,000 random 7x7 inputs.",
    "TFLite builds the four resize weights from integer coordinates, so they always sum to 2^20.",
])
d.save(height=y + 10)
