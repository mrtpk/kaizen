"""Figure 1: the compiler frontend, from a TFLite file to layers, constants and golden files.

Pass list from the reference model's optimise step (see the post); hardware-only passes marked.
"""

from svglib import AMBER, BLUE, EMERALD, INK, NEUTRAL, VIOLET, Diagram

d = Diagram(width=1100)
d.header("From a TFLite file to hardware layers",
         "Int8 on Silicon  ·  part 5  ·  the compiler frontend of the reference model",
         params="TFLite flatbuffer -> graph -> rewrite passes -> layers in execution order -> config, constants, golden files")

BW, BH = 290, 72
XS = [60, 410, 760]


def row(y, boxes):
    for i, (title, lines, col) in enumerate(boxes):
        d.box(XS[i], y, BW, BH, title, lines, col=col, mono_size=12)
        if i < 2:
            d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, NEUTRAL[0])


y = 116
d.panel(y, 170, 1, "What goes in?",
        "An int8 TFLite model. Every tensor already has its scale and zero point.")
row(y + 74, [
    ("TFLite flatbuffer", ["ops, tensors,", "scales, zero points"], BLUE),
    ("graph", ["one node per op,", "one edge per tensor"], BLUE),
    ("rewrite passes", ["make the graph fit", "the hardware"], AMBER),
])

y += 190
rows = [
    ("pass", "what it does"),
    ("remove input Quantize", "the hardware gets int8 input directly"),
    ("Split -> Slices", "one output per layer"),
    ("remove Identity, unused layers", "less work, fewer layers"),
    ("SAME -> Pad + VALID  (hardware)", "the pad value becomes an explicit constant"),
    ("R-T-R-DepthToSpace -> DepthToSpace CRD  (hardware)", "4 layers become 1"),
]
H = 96 + 26 * len(rows)
d.panel(y, H, 2, "Which rewrites ran?", "Five passes. The last two ran only when exporting for the hardware.")
for i, (a, b) in enumerate(rows):
    ry = y + 92 + i * 26
    if i == 0:
        d.text(80, ry, a, 13, INK, "bold")
        d.text(560, ry, b, 13, INK, "bold")
        d.line(70, ry + 8, 1040, ry + 8, NEUTRAL[0])
    else:
        col = AMBER if "(hardware)" in a else INK
        d.text(80, ry, a, 12.5, col, mono=True)
        d.text(560, ry, b, 12.5, INK)

y += H + 20
d.panel(y, 170, 3, "What comes out?",
        "Layers in TFLite execution order, and three kinds of files for the hardware team.")
row(y + 74, [
    ("layer config", ["type, shapes, stride,", "inputs (JSON)"], VIOLET),
    ("constants", ["weights, bias, alpha,", "shift, zero point, pad"], VIOLET),
    ("golden files", ["inputs and outputs", "of every layer"], EMERALD),
])

y += 190
d.legend(30, y, [(BLUE, "model"), (AMBER, "rewrite"), (VIOLET, "for the hardware"), (EMERALD, "for verification")])
y = d.footnotes(30, y + 40, ["Simplified: the pass list is the one in the reference model's optimise step; a few passes were disabled."])
d.save(height=y + 10)
