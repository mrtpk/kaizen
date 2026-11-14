"""Figure 2: non-linear functions as 256-entry tables, and softmax on the host.

Values: no_divider.py, sections 3 and 4: sigmoid table (s_in 0.08, out 1/256, -128):
113 of 256 entries differ between truncated and rounded tables; hard-swish 75 of 256;
softmax: 82.2 % of values differ, argmax unchanged.
"""

from svglib import AMBER, BLUE, EMERALD, NEUTRAL, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Non-linear functions: a table, or the host",
         "Int8 on Silicon  ·  part 4  ·  sigmoid input s = 0.08, z = 0  ·  output s = 1/256, z = -128",
         params="an int8 input has only 256 values, so any one-input function is a 256-entry table")

BW, BH = 290, 72
XS = [60, 410, 760]
y: float = 116
d.panel(y, 170, 1, "How does a sigmoid run with no exp?",
        "The compiler computes the function for all 256 inputs once. The hardware does one table read.")
d.box(XS[0], y + 74, BW, BH, "int8 input q", ["one of 256 values", "-128 .. 127"], col=BLUE, mono_size=12)
d.arrow(XS[0] + BW, y + 110, XS[1] - 2, y + 110, BLUE)
d.box(XS[1], y + 74, BW, BH, "table[q + 128]", ["computed in float", "at compile time"], col=AMBER, mono_size=12)
d.arrow(XS[1] + BW, y + 110, XS[2] - 2, y + 110, AMBER)
d.box(XS[2], y + 74, BW, BH, "int8 output", ["sigmoid, tanh,", "hard-swish, rsqrt"], col=EMERALD, mono_size=12)

y += 190
d.panel(y, 170, 2, "Does the table round or truncate?",
        "Our tables truncated with int(), except tanh. Truncation can move an entry one step toward zero.")
d.box(XS[0], y + 74, BW, BH, "sigmoid table", ["113 of 256 entries", "differ by 1"], col=PINK, mono_size=12)
d.box(XS[1], y + 74, BW, BH, "hard-swish table", ["75 of 256 entries", "differ by 1"], col=PINK, mono_size=12)
d.box(XS[2], y + 74, BW, BH, "example", ["q = 20: 84 (trunc.)", "vs 85 (rounded)"], col=NEUTRAL, mono_size=12)

y += 190
d.panel(y, 170, 3, "What about softmax?",
        "It needs a sum over the whole row, so it ran in float on the host. The final int8 cast truncated too.")
d.box(XS[0], y + 74, BW, BH, "host, float", ["exp, sum, divide", "per row"], col=VIOLET, mono_size=12)
d.arrow(XS[0] + BW, y + 110, XS[1] - 2, y + 110, VIOLET)
d.box(XS[1], y + 74, BW, BH, "int(p * 256 - 128)", ["82 % of values", "1 off from rounding"], col=PINK, mono_size=12)
d.arrow(XS[1] + BW, y + 110, XS[2] - 2, y + 110, PINK)
d.box(XS[2], y + 74, BW, BH, "argmax", ["unchanged in", "more than 99 % of rows"], col=EMERALD, mono_size=12)

y += 190
d.legend(30, y, [(AMBER, "compile time"), (PINK, "truncation effect"), (VIOLET, "host"), (EMERALD, "result")])
y = d.footnotes(
    30, y + 40, ["int() truncates toward zero. For the negative values of a softmax output (z = -128), it rounds up."])
d.save(height=y + 10)
