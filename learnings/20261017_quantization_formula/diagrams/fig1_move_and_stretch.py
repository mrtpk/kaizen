"""Figure 1: translate (add) moves a point; scale (multiply) stretches it.

Values: quantization.py, section "moving and stretching a point".
"""

from _numberline import NumberLine
from svglib import AMBER, BLUE, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Moving and stretching a point on a number line",
         "Quantization Basics  ·  the two operations behind every formula in this note",
         params="translate(value, offset) = value + offset    scale(value, factor) = value * factor")

TICKS = range(-6, 15, 2)
X0, X1 = 150, 1040

# panel 1: translate
y: float = 116
d.panel(y, 190, 1, "What does translate do?", "Adding an offset moves the point along the line. The distance is the "
                                              "offset.")
nl = NumberLine(d, X0, X1, y + 140, -6, 14, TICKS)
x4 = nl.point(4, "4", BLUE)
x9 = nl.point(9, "9", AMBER)
xm1 = nl.point(-1, "-1", AMBER)
d.arrow(x4 + 8, y + 100, x9 - 4, y + 100, AMBER)
d.arrow_label((x4 + x9) / 2, y + 100, "+5", AMBER)
d.arrow(x4 - 8, y + 100, xm1 + 4, y + 100, AMBER)
d.arrow_label((x4 + xm1) / 2, y + 100, "-5", AMBER)

# panel 2: scale
y = 326
d.panel(y, 210, 2, "What does scale do?",
        "Multiplying by a factor stretches (factor > 1) or shrinks (factor < 1) the distance from 0.")
nl = NumberLine(d, X0, X1, y + 160, -6, 14, TICKS)
x4 = nl.point(4, "4", BLUE)
x12 = nl.point(12, "12", VIOLET)
x2 = nl.point(2, "2", VIOLET)
d.arrow(x4 + 8, y + 120, x12 - 4, y + 120, VIOLET)
d.arrow_label((x4 + x12) / 2, y + 120, "x3", VIOLET)
d.arrow(x4 - 8, y + 92, x2 + 4, y + 92, VIOLET)
d.arrow_label((x4 + x2) / 2, y + 92, "x0.5", VIOLET)

y = 556
d.legend(30, y, [(BLUE, "starting point"), (AMBER, "after translate"), (VIOLET, "after scale")])
y = d.footnotes(30, y + 40, ["Scale stretches the distance from 0, so 0 itself never moves."])
d.save(height=y + 10)
