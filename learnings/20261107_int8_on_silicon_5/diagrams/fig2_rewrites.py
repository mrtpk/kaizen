"""Figure 2: three rewrites, each checked in NumPy.

Values: frontend.py: SAME 3x3 stride 2 on 8x8 -> pad top/bottom 0/1; R-T-R-DepthToSpace == DepthToSpace CRD;
Split into 3 == 3 Slices.
"""

from svglib import AMBER, EMERALD, NEUTRAL, Diagram

d = Diagram(width=1100)
d.header("Three rewrites, each checked in NumPy",
         "Int8 on Silicon  ·  part 5  ·  the graph changes, the output does not",
         params="every rewrite is checked: the new graph gives the same int8 output as the old one")

BW, BH = 290, 72
XS = [60, 410, 760]


def row(y, boxes, col_arrow):
    for i, (title, lines, col) in enumerate(boxes):
        d.box(XS[i], y, BW, BH, title, lines, col=col, mono_size=12)
        if i < 2:
            d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, col_arrow)


y = 116
d.panel(y, 170, 1, "SAME convolution -> Pad + VALID convolution",
        "3x3, stride 2, on 8x8: TensorFlow pads 0 on top and 1 at the bottom. The pad value is z_in.")
row(y + 74, [
    ("before", ["Conv SAME", "stride 2"], NEUTRAL),
    ("Pad", ["top 0, bottom 1", "left 0, right 1, value z_in"], AMBER),
    ("Conv VALID", ["same output", "Is pass? True"], EMERALD),
], AMBER)

y += 190
d.panel(y, 170, 2, "Reshape, Transpose, Reshape, DepthToSpace -> one DepthToSpace",
        "A PixelShuffle written in TFLite ops. The pass replaces the 4 layers with one CRD-order DepthToSpace.")
row(y + 74, [
    ("before", ["Reshape, Transpose,", "Reshape, DepthToSpace"], NEUTRAL),
    ("after", ["DepthToSpace", "mode CRD"], AMBER),
    ("check", ["same output", "Is pass? True"], EMERALD),
], AMBER)

y += 190
d.panel(y, 170, 3, "Split -> Slices",
        "A Split has many outputs. A Slice has one, which fits one layer, one output.")
row(y + 74, [
    ("before", ["Split", "3 outputs"], NEUTRAL),
    ("after", ["3 Slices", "1 output each"], AMBER),
    ("check", ["same output", "Is pass? True"], EMERALD),
], AMBER)

y += 190
d.legend(30, y, [(NEUTRAL, "original layers"), (AMBER, "rewritten"), (EMERALD, "checked equal")])
y = d.footnotes(30, y + 40, ["TensorFlow puts the extra pad pixel at the bottom and right (see the older note on TensorFlow padding)."])
d.save(height=y + 10)
