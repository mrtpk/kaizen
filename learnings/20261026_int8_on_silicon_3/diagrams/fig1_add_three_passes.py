"""Figure 1: add as three sets of constants for the same unit, one rounding.

Values: one_unit_many_ops.py, section "add": s1 0.05/z1 -10, s2 0.03/z2 5, s_out 0.07/z_out -3;
alpha1 748982, alpha2 449389; q1 = 40, q2 = -70 -> 1; 0 of 200,000 differ from TFLite.
"""

from svglib import AMBER, BLUE, EMERALD, MUTE, NEUTRAL, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Add: three sets of constants, one rounding",
         "Int8 on Silicon  ·  part 3  ·  s1 = 0.05, z1 = -10  ·  s2 = 0.03, z2 = 5  ·  s_out = 0.07, z_out = -3",
         params="out = clip( round_shift((q1 - z1) * alpha1 + (q2 - z2) * alpha2, 20) + z_out )")

BW, BH = 290, 92
XS = [60, 410, 760]

y: float = 116
d.panel(y, 210, 1, "How does the unit add two tensors with different scales?",
        "The bias slot removes each zero point, alpha rescales each input to the output scale, and one shift ends it.")
d.box(XS[0], y + 80, BW, BH, "input 1", ["bias = -z1 = 10",
      "alpha1 = 0.05/0.07 * 2^20", "= 748982"], col=AMBER, mono_size=12)
d.box(XS[1], y + 80, BW, BH, "input 2", ["bias = -z2 = -5",
      "alpha2 = 0.03/0.07 * 2^20", "= 449389"], col=AMBER, mono_size=12)
d.box(XS[2], y + 80, BW, BH, "sum", ["x1 + x2", ">> 20 (the one rounding)", "+ z_out, clip"], col=VIOLET, mono_size=12)
d.arrow(XS[1] + BW, y + 126, XS[2] - 2, y + 126, MUTE)
x1c, x3c, yb = XS[0] + BW / 2, XS[2] + BW / 2, y + 80 + BH
d.path_arrow(f"M {x1c:.1f},{yb:.1f} L {x1c:.1f},{yb + 14:.1f} L {x3c:.1f},{yb + 14:.1f} L {x3c:.1f},{yb + 2:.1f}", MUTE)

y += 230
d.panel(y, 170, 2, "What does one example give?",
        "q1 = 40 and q2 = -70: the float answer is 0.571, and the unit gives 1.")
d.box(XS[0], y + 74, BW, 72, "scaled inputs", ["50 * 748982", "-75 * 449389"], col=BLUE, mono_size=12)
d.arrow(XS[0] + BW, y + 110, XS[1] - 2, y + 110, BLUE)
d.box(XS[1], y + 74, BW, 72, "sum, >> 20", ["3,744,925 / 2^20", "= 3.57 -> 4"], col=VIOLET, mono_size=12)
d.arrow(XS[1] + BW, y + 110, XS[2] - 2, y + 110, VIOLET)
d.box(XS[2], y + 74, BW, 72, "+ z_out", ["4 + (-3) = 1", "float: 0.571 -> 1"], col=EMERALD, mono_size=12)

y += 190
d.panel(y, 170, 3, "How close is it to TFLite?",
        "Identical for these scales. alpha has about 20 bits when s_in / s_out is near 1.")
d.box(XS[0], y + 74, BW, 72, "ours", ["one rounding", "shift 20"], col=VIOLET, mono_size=12)
d.box(XS[1], y + 74, BW, 72, "TFLite", ["left shift 20, three",
      "multipliers, two roundings each"], col=NEUTRAL, mono_size=12)
d.box(XS[2], y + 74, BW, 72, "200,000 random pairs", ["0 differ"], col=EMERALD, mono_size=12.5)

y += 190
d.legend(30, y, [(AMBER, "per-input constants"), (VIOLET, "the one rounding"),
         (BLUE, "int32 data"), (EMERALD, "int8 output")])
y = d.footnotes(
    30, y + 40, ["TFLite shifts both inputs left by 20 and uses three multipliers, each with two roundings."])
d.save(height=y + 10)
