"""Figure 1: the LoDNN network, as I read the paper's figure, and its dilation table.

Values: context_module.py, sections 1 and 4: grid 200 x 400 (W x H), context input 100 x 200,
feature maps 6 -> 32 -> 128 -> 32 -> 2, dilations (1,1) ... (32,64), receptive fields 3 x 3 ... 129 x 255.
"""

from _lrd import script
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, VIOLET, Diagram

c = script("20261108_lidar_road_detection_3", "context_module")
rf = c.receptive_fields(c.DILATIONS)

d = Diagram(width=1100)
d.header("LoDNN: encoder, dilated context module, decoder",
         "LiDAR Road Detection  ·  part 3  ·  the network of Caltagirone et al., shapes as width x height x maps",
         params=f"input {c.GRID_W} x {c.GRID_H} x 6  ·  one 2 x 2 max pooling  ·  context at {c.CTX_W} x {c.CTX_H}  ·  "
         f"output {c.GRID_W} x {c.GRID_H} x 2")

BW, BH = 300, 108
XS = [60, 400, 740]
y1 = 116
d.panel(y1, 206, 1, "What are the three blocks?",
        "Shrink once, look far with dilated convolutions, grow back to the grid.")
boxes = [
    ("encoder", ["3x3 conv 6 -> 32, ELU", "3x3 conv 32 -> 32, ELU", f"max pool -> {c.CTX_W} x {c.CTX_H} x 32"], BLUE),
    ("context module", ["7 dilated 3x3 convs, 128 maps", "ELU + spatial dropout", "1x1 conv 128 -> 32"], VIOLET),
    ("decoder", [f"max unpool -> {c.GRID_W} x "
                 f"{c.GRID_H}", "3x3 conv 32 -> 32, ELU", "3x3 conv 32 -> 2, softmax"], EMERALD),
]
for i, (t, lines, col) in enumerate(boxes):
    d.box(XS[i], y1 + 76, BW, BH, t, lines, col=col, mono_size=12)
    if i < 2:
        d.arrow(XS[i] + BW, y1 + 76 + BH / 2, XS[i + 1] - 2, y1 + 76 + BH / 2, MUTE)

y2 = y1 + 226
d.panel(y2, 230, 2, "How does the context module grow its view?",
        "From layer 2 on, the height dilation is twice the width; from layer 3 on, each layer doubles it.")
cols = [90, 260, 470, 680]
heads = ["layer", "dilation (w, h)", "receptive field (w x h)", "kernel"]
for x, h in zip(cols, heads):
    d.text(x, y2 + 82, h, 12.5, INK, "bold")
for i, ((dw, dh), (w, h)) in enumerate(zip(c.DILATIONS, rf), start=1):
    yy = y2 + 104 + (i - 1) * 16
    d.text(cols[0], yy, str(i), 12, INK, mono=True)
    d.text(cols[1], yy, f"({dw}, {dh})", 12, AMBER[0], mono=True)
    d.text(cols[2], yy, f"{w} x {h}", 12, VIOLET[0], mono=True)
    d.text(cols[3], yy, "3x3", 12, INK, mono=True)

y = y2 + 250
d.legend(30, y, [(BLUE, "encoder"), (VIOLET, "context module"), (EMERALD, "decoder"), (AMBER, "dilation")])
end = d.footnotes(30, y + 40, [
    "Layer sizes are my reading of the paper's figure: every convolution outside the 1x1 is taken as 3x3.",
    "The dilation table and the receptive fields are the paper's table, re-computed by the script.",
])
d.save(height=end + 10)
