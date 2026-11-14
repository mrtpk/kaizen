"""Figure 3: map_point in four steps, mapping 1.0 on [-1, 3] to [-128, 127].

Values: quantization.py, map_point(); 1.0 -> 2.0 -> 0.5 -> 127.5 -> -0.5.
"""

from _numberline import NumberLine
from svglib import AMBER, BLUE, EMERALD, MUTE, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Mapping a point from one number line to another",
         "Quantization Basics  ·  map_point(1.0, src = [-1, 3], dst = [-128, 127])",
         params="move to 0  ->  shrink to 0..1  ->  stretch to the destination length  ->  move to the destination start")

X0, X1 = 400, 1000
y = 116
PITCH = 112
rows = [
    (-1, 3, [-1, 0, 1, 2, 3], 1.0, "1.0", BLUE, "source [-1, 3]", None),
    (0, 4, [0, 1, 2, 3, 4], 2.0, "2.0", AMBER, "[0, 4]", "1. translate by -src_min (+1)"),
    (0, 1, [0, 0.25, 0.5, 0.75, 1], 0.5, "0.5", VIOLET, "[0, 1]", "2. scale by 1 / src_length (1 / 4)"),
    (0, 255, [0, 64, 128, 192, 255], 127.5, "127.5", VIOLET, "[0, 255]", "3. scale by dst_length (255)"),
    (-128, 127, [-128, -64, 0, 64, 127], -0.5, "-0.5", EMERALD, "destination", "4. translate by dst_min (-128)"),
]
H = 50 + PITCH * len(rows)
d.panel(y, H, 1, "Where does 1.0 land on the int8 line?",
        "At -0.5. The point stays at the same fraction of each line: halfway.")
for i, (lo, hi, ticks, v, label, col, name, step) in enumerate(rows):
    ly = y + 110 + i * PITCH
    nl = NumberLine(d, X0, X1, ly, lo, hi, ticks)
    nl.point(v, label, col)
    d.text(X0 - 30, ly + 5, name, 13, col, "bold", anchor="end", mono=True)
    if step:
        d.text(60, ly - 46, step, 12.5, MUTE, "bold")
    if i < len(rows) - 1:
        xm = nl.px(v)
        d.arrow(xm, ly + 32, xm, ly + PITCH - 30, MUTE, sw=1.2, dash="4 3")

y += H + 20
d.legend(30, y, [(BLUE, "source line"), (AMBER, "after translate"), (VIOLET, "after scale"), (EMERALD, "destination line")])
y = d.footnotes(30, y + 40, ["The int8 line has 256 integer points, so -0.5 must be rounded to an integer. Figure 4 shows that step."])
d.save(height=y + 10)
