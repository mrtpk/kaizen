"""Figure 2: TFLite (two roundings) vs ours (one rounding) for the same M.

Values: one_multiply_one_shift.py: M = 0.0023050238, q31 = 1267200256, shift = -8,
alpha = 2416; ours != TFLite 3.02 % (max 1); TFLite != exact 0.18 %;
alpha rounded (2417): ours != exact 0.02 %.
TFLite: v2.16.1 default build (double rounding), gemmlowp fixedpoint.h.
"""

from svglib import AMBER, BLUE, EMERALD, NEUTRAL, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Two ways to multiply by M = 0.0023050 with integers",
         "Int8 on Silicon  ·  part 1  ·  same accumulator, same M, different rounding",
         params="s_in = 0.0235   s_w = 0.0041   s_out = 0.0418   M = s_in * s_w / s_out")

BW, BH = 290, 72
XS = [60, 410, 760]


def row(y, boxes, arrow_col):
    for i, (title, lines, col) in enumerate(boxes):
        d.box(XS[i], y, BW, BH, title, lines, col=col, mono_size=12)
        if i < 2:
            d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, arrow_col)


y = 116
d.panel(y, 170, 1, "How does TFLite do it?",
        "A 31-bit mantissa and a per-channel shift. Two roundings.")
row(y + 74, [
    ("compile time", ["frexp(M) -> Q31", "q31 = 1267200256, shift = -8"], NEUTRAL),
    ("rounding #1", ["high 32 bits of", "2 * acc * q31 (int64)"], AMBER),
    ("rounding #2", ["divide by 2^8", "round half away from 0"], AMBER),
], VIOLET)

y += 190
d.panel(y, 170, 2, "How did we do it?",
        "One integer alpha with a fixed shift of 20. One rounding.")
row(y + 74, [
    ("compile time", ["alpha = int(M * 2^20)", "= 2416, shift = 20"], NEUTRAL),
    ("one multiply", ["alpha * acc", "fits in int32"], BLUE),
    ("one rounding", [">> 20 with a mask", "round half away from 0"], AMBER),
], BLUE)

y += 190
d.panel(y, 160, 3, "How different are the results?",
        "Never more than 1 apart. Most of the gap comes from truncating alpha, not from the rounding.")
chips = [
    ("ours vs TFLite", "3.02 % differ, by 1", PINK),
    ("TFLite vs exact", "0.18 % differ", VIOLET),
    ("ours (alpha 2417) vs exact", "0.02 % differ", EMERALD),
]
for i, (title, value, col) in enumerate(chips):
    d.box(XS[i], y + 74, BW, 60, title, [value], col=col, mono_size=13)

y += 180
d.legend(30, y, [(NEUTRAL, "compile time"), (AMBER, "a rounding step"), (BLUE, "our runtime path"),
                 (VIOLET, "TFLite"), (PINK, "mismatch"), (EMERALD, "rounded alpha")])
y = d.footnotes(30, y + 40, [
    "200,000 random accumulators in [-60000, 60000]. Exact = round half away from zero of acc * M.",
    "TFLite: default build. A TFLITE_SINGLE_ROUNDING build flag switches to one rounding.",
])
d.save(height=y + 10)
