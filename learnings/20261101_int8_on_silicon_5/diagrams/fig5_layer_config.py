"""Figure 5: what the hardware gets for one convolution layer: a config entry and a constants file.

Simplified: key names are generic, values are illustrative and taken from parts 1-2
(alpha 2416, folded bias -24868, pad value z_in = -64, output zero point -12, shift 20).
"""

from figkit import code
from svglib import AMBER, BLUE, INK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("One layer, as the hardware sees it",
         "Int8 on Silicon  ·  part 5  ·  a 3x3 convolution after the SAME -> Pad + VALID rewrite",
         params="the layer config says what to run; the constants say how to finish it")

y = 116
d.panel(y, 330, 1, "What goes into the layer config and the constants?",
        "Shapes and links in a JSON entry; per-channel numbers for the post-processing unit in a constants file.")
code(d, 70, y + 74, 450, [
    "{",
    '  "op": "CONVOLUTION",',
    '  "layer_id": 7,',
    '  "inputs": [6],',
    '  "input_rows": 56, "input_cols": 56,',
    '  "in_channels": 64,',
    '  "out_channels": 128,',
    '  "kernel": 3, "stride": 1,',
    '  "padding": "VALID", "groups": 1',
    "}",
], color_fn=lambda i, s: VIOLET if '"' in s else INK)
code(d, 580, y + 74, 450, [
    "pad_value\t\t-64",
    "output_zp\t\t-12",
    "shift\t\t20",
    "clip\t\t-128 127",
    "alpha\t0\t2416",
    "bias\t0\t-24868",
    "alpha\t1\t...",
    "bias\t1\t...",
], color_fn=lambda i, s: AMBER if s.startswith(("alpha", "bias")) else BLUE, tabs=(0, 110, 180))

y += 350
d.legend(30, y, [(VIOLET, "layer config (JSON)"), (BLUE, "per-layer constants"), (AMBER, "per-channel constants")])
y = d.footnotes(30, y + 40, [
    "Simplified: key names are generic and the values are illustrative, taken from the examples in parts 1 and 2.",
    "Layer 6 is the explicit Pad layer that feeds this convolution.",
])
d.save(height=y + 10)
