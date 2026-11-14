"""Figure 2: multiply with the square-root split, and what it costs.

Values: one_unit_many_ops.py, section "multiply": alphas 147 and 88 (example),
228 and 17 (SE block); error rates against exact rounding for four variants.
"""

from svglib import AMBER, EMERALD, INK, NEUTRAL, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Multiply: splitting M into two square roots",
         "Int8 on Silicon  ·  part 3  ·  M = s1 * s2 / s_out = (s1 / sqrt(s_out)) * (s2 / sqrt(s_out))",
         params="out = clip( round_shift(((q1 - z1) * a1) * ((q2 - z2) * a2), 20) + z_out ),  a_i = int(s_i / "
                "sqrt(s_out) * 2^10)")

BW, BH = 290, 92
XS = [60, 410, 760]
y = 116
d.panel(y, 210, 1, "How does the split work?",
        "Each input gets half of the scale, at 2^10, so the product fits the 32-bit accumulator and lands at 2^20.")
d.box(XS[0], y + 80, BW, BH, "input 1", ["(q1 - z1) * a1",
      "a1 = 0.05 / sqrt(0.12) * 2^10", "= 147"], col=AMBER, mono_size=12)
d.box(XS[1], y + 80, BW, BH, "input 2", ["(q2 - z2) * a2",
      "a2 = 0.03 / sqrt(0.12) * 2^10", "= 88"], col=AMBER, mono_size=12)
d.box(XS[2], y + 80, BW, BH, "product", ["x1 * x2 (int32)", ">> 20, + z_out, clip"], col=VIOLET, mono_size=12)
d.arrow(XS[1] + BW, y + 126, XS[2] - 2, y + 126, NEUTRAL[0])
x1c, x3c, yb = XS[0] + BW / 2, XS[2] + BW / 2, y + 80 + BH
d.path_arrow(
    f"M {x1c:.1f},{yb:.1f} L {x1c:.1f},{yb + 14:.1f} L {x3c:.1f},{yb + 14:.1f} L {x3c:.1f},{yb + 2:.1f}", NEUTRAL[0])

y += 230
rows = [
    ("variant", "example (a = 147, 88)", "SE block x * sigmoid (a = 228, 17)"),
    ("split, truncated a (ours)", "45.9 %, max 2", "75.2 %, max 7"),
    ("split, rounded a", "17.1 %, max 1", "19.5 %, max 1"),
    ("all of M on one input, 2^20", "3.9 %, max 1", "0.05 %, max 1"),
    ("TFLite", "0.23 %, max 1", "0.14 %, max 1"),
]
H = 96 + 26 * len(rows)
d.panel(y, H, 2, "What does it cost?",
        "Room to go further: a = 17 is a 5-bit multiplier. Outputs that differ from exact rounding, 200,000 pairs:")
cols = [80, 440, 720]
for i, r in enumerate(rows):
    ry = y + 92 + i * 26
    for j, (x, v) in enumerate(zip(cols, r)):
        if i == 0:
            d.text(x, ry, v, 13, INK, "bold")
        else:
            col = PINK if i == 1 else (EMERALD if i == 3 else INK)
            d.text(x, ry, v, 13, col, mono=j > 0)
    if i == 0:
        d.line(70, ry + 8, 1040, ry + 8, NEUTRAL[0])

y += H + 20
d.legend(30, y, [(AMBER, "per-input constants"), (VIOLET, "the one rounding"), (PINK, "split, truncated alphas"),
                 (EMERALD, "best variant")])
y = d.footnotes(30, y + 40, [
    "SE block: s1 = 0.05, z1 = -10; sigmoid output s2 = 1/256, z2 = -128; s_out = 0.05.",
    "Example: s1 = 0.05, z1 = -10; s2 = 0.03, z2 = 5; s_out = 0.12, z_out = -20.",
])
d.save(height=y + 10)
