"""Figure 4: every operation of the series as constants for the same post-processing unit.

Sources: parts 1-4 of the series (conv: part 1-2; add, multiply, constants, ReLU family: part 3;
pooling and mean: part 4). Shifts are the ones the hardware supported for each operation.
"""

from figkit import table, table_height
from svglib import AMBER, BLUE, EMERALD, INK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Every operation is a set of constants",
         "Int8 on Silicon  ·  part 3  ·  the same unit: (x + bias) * alpha, rounding shift, + z_out, clip",
         params="the compiler writes these constants per layer; the hardware never changes")

rows = [
    ("operation", "passes", "bias", "alpha", "shift"),
    ("convolution, dense", "1", "bias - z_in * sum(w)", "M * 2^20, per channel", "20"),
    ("add, subtract", "3", "-z1, -z2, then 0", "s_i / s_out * 2^20", "20"),
    ("multiply", "3", "-z1, -z2", "s_i / sqrt(s_out) * 2^10", "20"),
    ("add a constant c", "1", "c / s_in - z_in", "s_in / s_out * 2^16", "16"),
    ("multiply by k", "1", "-z_in", "k * s_in / s_out * 2^16", "16"),
    ("ReLU family", "1", "-z_in", "two alphas (x >= 0, x < 0)", "16"),
    ("average pool k x k", "1", "-z_in * k * k", "s_in / (s_out k^2) * 2^20", "20"),
    ("mean over C channels", "1", "-z_in * C", "s_in / (s_out C) * 2^20", "20"),
    ("requantize", "1", "-z_in", "s_in / s_out * 2^20", "20"),
]


def color(i, j, v):
    if j == 4:
        return EMERALD if v == "20" else VIOLET
    if j == 3:
        return AMBER
    if j == 2:
        return BLUE
    return INK


y = 116
H = table_height(len(rows))
d.panel(y, H, 1, "What changes from one operation to the next?",
        "Only the constants and the number of passes. Two shift amounts from the hardware's list cover every row.")
table(d, y + 92, rows, [80, 300, 390, 620, 960], color)

y += H + 20
d.legend(30, y, [(BLUE, "bias (zero points, constants)"),
         (AMBER, "alpha (scales)"), (EMERALD, "shift 20"), (VIOLET, "shift 16")])
y = d.footnotes(30, y + 40, [
    "Multiply: each input pass uses 2^10, so the product of the two lands at 2^20.",
    "Simplified: tables (sigmoid, tanh) and softmax are not on this unit; see part 4.",
])
d.save(height=y + 10)
