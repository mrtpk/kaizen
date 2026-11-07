"""Figure 3: partitioning between accelerator and host, and the verification loop.

Values: frontend.py, section 4 (the six-op example graph and its four subgraphs).
"""

from svglib import AMBER, BLUE, EMERALD, INK, NEUTRAL, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Which ops run where, and how we knew they were right",
         "Int8 on Silicon  ·  part 5  ·  partitioning and verification",
         params="order: conv1 -> conv2 -> argmax -> conv3 -> add -> softmax   (argmax, softmax: not on the accelerator)")

y = 116
rows = [
    ("op", "inputs", "runs on", "boundary rule only", "with the span rule"),
    ("conv1", "-", "accelerator", "subgraph A", "subgraph 1"),
    ("conv2", "conv1", "accelerator", "subgraph A", "subgraph 1"),
    ("argmax", "conv1", "host", "subgraph B", "subgraph 2"),
    ("conv3", "conv2", "accelerator", "subgraph A", "subgraph 3"),
    ("add", "conv3, argmax", "accelerator", "subgraph A  (wrong)", "subgraph 3"),
    ("softmax", "add", "host", "subgraph C", "subgraph 4"),
]
H = 96 + 26 * len(rows)
d.panel(y, H, 1, "How is the model split between accelerator and host?",
        "Cut every edge across the boundary, and every edge that jumps over a host op in execution order.")
cols = [80, 220, 420, 590, 820]
for i, r in enumerate(rows):
    ry = y + 92 + i * 26
    for j, (x, v) in enumerate(zip(cols, r)):
        if i == 0:
            d.text(x, ry, v, 13, INK, "bold")
        else:
            col = INK
            if j == 2:
                col = BLUE if v == "accelerator" else VIOLET
            if j == 3 and "wrong" in v:
                col = PINK
            if j == 4:
                col = EMERALD
            d.text(x, ry, v, 12.5, col, mono=j != 2)
    if i == 0:
        d.line(70, ry + 8, 1040, ry + 8, NEUTRAL[0])

y += H + 20
BW, BH = 290, 72
XS = [60, 410, 760]
d.panel(y, 170, 2, "How did we know each layer was right?",
        "The reference model wrote golden files. The hardware and TFLite were each compared against it.")
d.box(XS[0], y + 74, BW, BH, "reference model", ["inputs, weights, constants,", "outputs of every layer"], col=AMBER, mono_size=12)
d.arrow(XS[0] + BW, y + 110, XS[1] - 2, y + 110, AMBER)
d.box(XS[1], y + 74, BW, BH, "hardware (FPGA)", ["same input, same weights", "diff against golden: must be 0"], col=BLUE, mono_size=12)
d.arrow(XS[1] + BW, y + 110, XS[2] - 2, y + 110, AMBER)
d.box(XS[2], y + 74, BW, BH, "TFLite", ["layer by layer, and", "accuracy on datasets"], col=VIOLET, mono_size=12)

y += 190
d.legend(30, y, [(BLUE, "accelerator"), (VIOLET, "host / TFLite"), (AMBER, "reference model"), (EMERALD, "correct split"), (PINK, "wrong split")])
y = d.footnotes(30, y + 40, ["Bit-exact target: the hardware. TFLite was the accuracy target, not a bit-exact one."])
d.save(height=y + 10)
