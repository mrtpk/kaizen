"""Figure 1: average pooling as a convolution with all-ones weights, and its border.

Values: no_divider.py, section 1: alpha = int(2^20 / 9) = 116508; 3x3 SAME on 8x8;
inner 0 of 36 differ; border 28 of 28 differ, max 37; corner TFLite 30, ours 8.
"""

from svglib import AMBER, BLUE, EMERALD, INK, PINK, VIOLET, Diagram, _c, _t

d = Diagram(width=1100)
d.header("Average pooling without a divider",
         "Int8 on Silicon  ·  part 4  ·  3x3 average pool, SAME padding, on an 8x8 input",
         params="average = sum * (1 / 9)  ->  depthwise conv with all-ones weights, alpha = 2^20 / 9 = 116508")

BW, BH = 290, 72
XS = [60, 410, 760]
y: float = 116
d.panel(y, 170, 1, "How do we divide by 9 with no divider?",
        "We do not. The 1/9 goes into alpha, and the pool becomes a convolution with weights of 1.")
d.box(XS[0], y + 74, BW, BH, "all-ones 3x3 weights",
      ["sum of the window", "(minus z_in, folded)"], col=BLUE, mono_size=12)
d.arrow(XS[0] + BW, y + 110, XS[1] - 2, y + 110, BLUE)
d.box(XS[1], y + 74, BW, BH, "x alpha, >> 20", ["alpha = int(2^20 / 9)", "= 116508"], col=AMBER, mono_size=12)
d.arrow(XS[1] + BW, y + 110, XS[2] - 2, y + 110, AMBER)
d.box(XS[2], y + 74, BW, BH, "inner outputs", ["0 of 36 differ", "from TFLite"], col=EMERALD, mono_size=12)

y += 190
d.panel(y, 230, 2, "What happens at the border?",
        "TFLite averages only the valid points. A convolution always divides by 9, with the padding counted as 0.0.")
C = 40
x0, y0 = 80, y + 76
for r in range(3):
    for c in range(3):
        padded = r == 0 or c == 0
        col = PINK if padded else BLUE
        d.rect(x0 + c * C, y0 + r * C, C - 4, C - 4, fill=_t(col), stroke=col, rx=4, sw=1.2)
        d.text(x0 + c * C + 18, y0 + r * C + 23, "pad" if padded else "q", 11, _c(col), "bold", "middle", mono=True)
d.text(260, y0 + 18, "corner window: 4 valid points, 5 padded points", 13, INK, "bold")
d.text(260, y0 + 46, "TFLite:  sum of 4 / 4  = 30", 13, VIOLET, "bold", mono=True)
d.text(260, y0 + 70, "ours:    sum of 9 / 9  = 8   (5 points are 0.0)", 13, PINK, "bold", mono=True)
d.text(260, y0 + 106, "All 28 border outputs differ, by up to 37. VALID padding has no border, so it "
                      "matches.", 12.5, INK)

y += 250
d.legend(30, y, [(BLUE, "valid input"), (PINK, "padding / differs"),
         (AMBER, "alpha"), (VIOLET, "TFLite"), (EMERALD, "matches")])
y = d.footnotes(30, y + 40, ["Input and output share s = 0.05 and z = -10 here, so the pool needs no rescale."])
d.save(height=y + 10)
