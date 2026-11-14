"""Figure 4: the design decisions that held up, with the part that shows each one.

Values: parts 1-5 of the series (scripts and text).
"""

from figkit import table, table_height
from svglib import BLUE, EMERALD, INK, Diagram

d = Diagram(width=1100)
d.header("What held up",
         "Int8 on Silicon  ·  part 6  ·  the decisions I would make again",
         params="each one is backed by a script or a check in the part where it appears")

rows = [
    ("decision", "part", "why it held up"),
    ("one post-processing unit for every operation", "1, 3", "nine operations, one thing to verify"),
    ("a fixed shift of 20, one rounding", "1, 6", "largest shift with 4 bits of headroom"),
    ("zero points folded into the bias", "2", "the MAC array multiplies raw int8"),
    ("pad with the input zero point", "2", "the folded bias stays exact at the border"),
    ("the ReLU family as one PReLU kernel", "3", "five activations, one kernel"),
    ("averages as convolutions with 1/N in alpha", "4", "no divider needed"),
    ("256-entry tables for one-input functions", "4", "one read per value"),
    ("graph rewrites that keep the int8 output equal", "5", "the graph changes, the output does not"),
    ("the span rule for the accelerator/host split", "5", "no subgraph waits for the host"),
    ("golden files in plain text", "5", "a mismatch points to a layer and a channel"),
    ("two targets: hardware bit-exact, TFLite accuracy", "5", "never mixed up"),
]
y: float = 116
H = table_height(len(rows))
d.panel(y, H, 1, "Which decisions would I make again?",
        "All of these. The gaps (figure 3) are in the details around them, not in the decisions.")
table(d, y + 92, rows, [80, 560, 650], lambda i, j, v: BLUE if j == 1 else (EMERALD if j == 2 else INK))

y += H + 20
d.legend(30, y, [(BLUE, "part of the series"), (EMERALD, "the evidence")])
y = d.footnotes(
    30, y + 40, ["The gaps (truncated constants, one extra rounding, border rules, checks in int32) are in figure 3."])
d.save(height=y + 10)
