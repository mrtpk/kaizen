"""Figure 3: the scorecard of gaps found by the scripts of this series.

Values: parts 1-6 scripts (one_multiply_one_shift.py, one_unit_many_ops.py, no_divider.py, lessons.py).
"""

from svglib import EMERALD, INK, NEUTRAL, PINK, Diagram

d = Diagram(width=1100)
d.header("What the scripts of this series found",
         "Int8 on Silicon  ·  part 6  ·  measured in NumPy against exact rounding or TFLite",
         params="most gaps are a truncation or a missing check; the fixes are one line each")

rows = [
    ("part", "gap", "measured effect", "fix"),
    ("1", "alpha truncated", "2.83 % vs exact", "round(): 0.02 %"),
    ("3", "multiply split, 2^10 alphas", "75 %, max 7 (SE block)", "all of M on one input"),
    ("3", "constant truncated in bias", "85 of 256, by 1", "round()"),
    ("4", "average pool border", "28 of 28, max 37", "divide by valid count"),
    ("4", "global pool, two roundings", "60.5 %, max 2", "one rounding: 0 %"),
    ("4", "tables truncated", "113 of 256, by 1", "round()"),
    ("4", "softmax cast truncated", "82 % of values", "round()"),
    ("4", "resize weights truncated", "26 of 169, by 1", "integer complements"),
    ("6", "max pool pads with z", "28 of 64, max 47", "skip padded points"),
    ("6", "concat copies bytes", "real error up to 6.4", "check the scales"),
    ("6", "overflow guard in int32", "127 becomes -128", "check in int64"),
]
y = 116
H = 96 + 26 * len(rows)
d.panel(y, H, 1, "Where could the reference model be tighter?",
        "Eleven places, each small enough that final model accuracy stayed on par with TFLite.")
cols = [80, 150, 480, 760]
for i, r in enumerate(rows):
    ry = y + 92 + i * 26
    for j, (x, v) in enumerate(zip(cols, r)):
        if i == 0:
            d.text(x, ry, v, 13, INK, "bold")
        else:
            col = INK if j < 2 else (PINK if j == 2 else EMERALD)
            d.text(x, ry, v, 12.5, col, mono=j >= 2)
    if i == 0:
        d.line(70, ry + 8, 1040, ry + 8, NEUTRAL[0])

y += H + 20
d.legend(30, y, [(PINK, "effect"), (EMERALD, "fix")])
y = d.footnotes(30, y + 40, [
    "Synthetic inputs and example scales; the effect on a real model depends on its scales and data.",
    "Max pool: only when the input to the pool can be negative (no ReLU before it).",
])
d.save(height=y + 10)
