"""Figure 3: the ReLU family and operations with a constant, on the same unit.

Values: one_unit_many_ops.py, sections "an operand that is a constant" and "the ReLU family".
"""

from svglib import AMBER, EMERALD, INK, NEUTRAL, PINK, Diagram

d = Diagram(width=1100)
d.header("One kernel for the ReLU family, and constants folded into the unit",
         "Int8 on Silicon  ·  part 3  ·  s_in = s_out = 1/16, zero points 0 (ReLU family)",
         params="PReLU:  v = q - z_in;  v >= 0: round_shift(v * a_pos, 16);  v < 0: round_shift(slope_q * v * a_neg, "
                "16);  + z_out, clip")


def table(y, rows, cols, highlight):
    for i, r in enumerate(rows):
        ry = y + i * 26
        for j, (x, v) in enumerate(zip(cols, r)):
            if i == 0:
                d.text(x, ry, v, 13, INK, "bold")
            else:
                d.text(x, ry, v, 12.5, highlight.get(i, INK) if j == len(r) - 1 else INK, mono=j > 0)
        if i == 0:
            d.line(70, ry + 8, 1040, ry + 8, NEUTRAL[0])


y = 116
rows = [
    ("operation", "slope (int8, scale)", "clip max", "differs from exact"),
    ("ReLU", "0", "127", "0 of 256"),
    ("ReLU6", "0", "q(6.0) = 96", "0 of 256"),
    ("min(x, 2.0)", "1, scale 1.0", "q(2.0) = 32", "0 of 256"),
    ("LeakyReLU 0.1", "1, scale 0.1", "127", "13 of 256, by 1"),
    ("PReLU 0.25", "64, scale 1/256", "127", "0 of 256"),
]
H = 96 + 26 * len(rows)
d.panel(y, H, 1, "How do five activations become one kernel?",
        "Each is a slope for negative inputs plus a clip. Only the constants change.")
table(y + 92, rows, [80, 330, 600, 820], {4: AMBER})

y += H + 20
rows = [
    ("operation", "bias", "alpha (shift 16)", "differs from exact"),
    ("x + 0.75", "int(0.75 / 0.02) - 10 = 27", "0.02 / 0.03 * 2^16", "85 of 256, by 1"),
    ("x * -1.5", "-z_in = -10", "-1.5 * 0.02 / 0.03 * 2^16", "0 of 256"),
]
H = 96 + 26 * len(rows)
d.panel(y, H, 2, "What if one operand is a constant?",
        "Add a constant: it goes into the bias. Multiply by a constant: it goes into alpha. One pass, no second "
        "tensor.")
table(y + 92, rows, [80, 330, 600, 820], {1: PINK, 2: EMERALD})

y += H + 20
d.legend(30, y, [(EMERALD, "exact"), (AMBER, "off by 1, truncated alpha"), (PINK, "off by 1, truncated constant")])
y = d.footnotes(30, y + 40, [
    "x + 0.75: s_in = 0.02, z_in = 10, s_out = 0.03, z_out = -5. 0.75 / 0.02 = 37.5 has no exact integer in input "
    "units.",
])
d.save(height=y + 10)
