"""Figure 4: the golden files: how much gets written, and what a line looks like.

Levels: the reference model's intermediate levels (none / default / all / one layer / default + outputs).
Line format: "<channel> <index> : <value>" for tensors, "<out ch> <in ch> : <value> 0x<hex>" for weights.
Values in the example lines are illustrative. printf %x on an int8 promoted to int prints -5 as 0xfffffffb.
"""

from figkit import code, table, table_height
from svglib import AMBER, BLUE, INK, MUTE, Diagram

d = Diagram(width=1100)
d.header("Golden files a human can read",
         "Int8 on Silicon  ·  part 5  ·  what the reference model wrote for the hardware team",
         params="plain text, one value per line; the amount of output is a setting")

rows = [
    ("level", "what it writes"),
    ("none", "nothing; builds the model and the layer config"),
    ("default", "constants and weights of every layer"),
    ("all", "default, plus the input and output of every layer"),
    ("one layer", "everything for chosen layers (by id or type), default for the rest"),
    ("default + outputs", "default, plus the outputs of the model"),
]
y: float = 116
H = table_height(len(rows))
d.panel(y, H, 1, "How much gets written?",
        "Five levels. 'One layer' was the one for chasing a bug without drowning in files.")
table(d, y + 92, rows, [80, 330], lambda i, j, v: AMBER if i == 4 else INK)

y += H + 20
d.panel(y, 230, 2, "What does a line look like?",
        "Tensors: channel, index, value. Weights: output channel, input channel, value, and the same value in hex.")
code(d, 70, y + 74, 450, [
    "# layer output (channel index : value)",
    "0 0 : -37",
    "0 1 : 12",
    "0 2 : 127",
    "1 0 : -128",
], color_fn=lambda i, s: MUTE if s.startswith("#") else INK)
code(d, 580, y + 74, 450, [
    "# weights (out ch, in ch : value hex)",
    "0 0 : 89 0x59",
    "0 1 : 35 0x23",
    "0 2 : -5 0xfffffffb",
    "1 0 : -59 0xffffffc5",
], color_fn=lambda i, s: MUTE if s.startswith("#") else BLUE)

y += 250
d.legend(30, y, [(AMBER, "the level for debugging"), (BLUE, "weights")])
y = d.footnotes(
    30, y + 40, ["Illustrative values. Not compact, but a hardware engineer can grep a channel and compare by eye."])
d.save(height=y + 10)
