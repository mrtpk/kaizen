"""Figure 4: where the multiplier M comes from.

Values: one_multiply_one_shift.py: s_in 0.0235, s_w 0.0041, s_out 0.0418, M = 0.0023050.
"""

from figkit import row
from svglib import AMBER, BLUE, EMERALD, NEUTRAL, PINK, Diagram

d = Diagram(width=1100)
d.header("Where does M come from?",
         "Int8 on Silicon  ·  part 1  ·  one output of a convolution",
         params="r = s (q - z)   ->   r_out = sum r_in * r_w = s_in s_w sum (q_in - z_in)(q_w - z_w)")

y: float = 116
d.panel(y, 170, 1, "What is integer, and what is not?",
        "The sum over the window is all integers. Only the scales are real numbers.")
row(d, y + 74, [
    ("int8 inputs, int8 weights", ["q_in, q_w", "with zero points z_in, z_w"], BLUE),
    ("int32 accumulator", ["acc = sum (q_in - z_in)", "      * (q_w - z_w)"], BLUE),
    ("real scales", ["s_in = 0.0235, s_w = 0.0041", "s_out = 0.0418"], AMBER),
], NEUTRAL[0])

y += 190
d.panel(y, 170, 2, "How do they meet in the output?",
        "Divide by s_out and add z_out. All the real numbers collapse into one multiplier M.")
row(d, y + 74, [
    ("M", ["s_in * s_w / s_out", "= 0.0023050"], AMBER),
    ("int8 output", ["q_out = z_out + M * acc", ""], EMERALD),
    ("the problem", ["M is a real number;", "the hardware has no float"], PINK),
], NEUTRAL[0])

y += 190
d.legend(30, y, [(BLUE, "integers"), (AMBER, "real numbers"), (EMERALD, "output"), (PINK, "what parts 1-3 solve")])
y = d.footnotes(30, y + 40, ["A per-channel weight scale gives one M per output channel."])
d.save(height=y + 10)
