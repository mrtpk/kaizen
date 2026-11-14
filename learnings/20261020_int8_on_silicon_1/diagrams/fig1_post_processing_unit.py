"""Figure 1: one output through the post-processing unit.

Values: one_multiply_one_shift.py, section "one output through the post-processing unit":
acc 12345, bias -500, alpha 2416, shift 20, zero_point -12 -> 15.
"""

from svglib import AMBER, BLUE, EMERALD, MUTE, VIOLET, Diagram

d = Diagram(width=1100)
d.header("One multiply, one shift: the post-processing unit",
         "Int8 on Silicon  ·  part 1  ·  every layer ends in this unit, once per output value",
         params="y = clip( round_shift((acc + bias[c]) * alpha[c], 20) + zero_point, min, max )")

BW, BH = 290, 72
XS = [60, 410, 760]


def row(y, boxes):
    for i, (title, lines, col) in enumerate(boxes):
        d.box(XS[i], y, BW, BH, title, lines, col=col, mono_size=12.5)
        if i < 2:
            d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, MUTE)


y1 = 116
d.panel(y1, 170, 1, "What does the unit do with the int32 accumulator?",
        "It adds a per-channel bias, then multiplies by a per-channel integer alpha.")
row(y1 + 74, [
    ("int8 MACs", ["acc = sum(q_in * q_w)", "int32: 12345"], BLUE),
    ("+ bias[c]", ["12345 + (-500)", "= 11845"], AMBER),
    ("x alpha[c]", ["11845 x 2416", "= 28,617,520"], AMBER),
])

y2 = y1 + 210
d.panel(y2, 170, 2, "How does it get back to int8?",
        "One rounding shift by 20, add the output zero point, clip to the int8 range.")
row(y2 + 74, [
    (">> 20, rounded", ["28,617,520 / 2^20", "= 27.292 -> 27"], VIOLET),
    ("+ zero_point", ["27 + (-12)", "= 15"], EMERALD),
    ("clip", ["to [-128, 127]", "-> 15 (int8)"], EMERALD),
])

y = y2 + 190
d.legend(30, y, [(BLUE, "int32 data"), (AMBER, "per-channel constant (compile time)"),
                 (VIOLET, "the one rounding"), (EMERALD, "int8 output")])
y = d.footnotes(30, y + 40, [
    "Float check: (acc + bias) * M + zero_point = 15.303, with M = 0.0023050.",
    "The bias value here is an example. Part 2 shows where the real bias comes from.",
])
d.save(height=y + 10)
