"""Figure 2: percentage is two scale operations (25 of 500 -> 0.05 -> 5 %).

Values: quantization.py, percentage(25, 500) = 5.0.
"""

from _numberline import NumberLine
from svglib import BLUE, EMERALD, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Percentage is two stretches",
         "Quantization Basics  ·  25 out of 500 is 5 %",
         params="percentage(value, max_value) = scale(scale(value, 1 / max_value), 100)")

X0, X1 = 260, 1000
y = 116
d.panel(y, 420, 1, "How does 25 on a 0..500 line become 5 on a 0..100 line?",
        "Shrink the line to 0..1, then stretch it to 0..100. The point keeps its place: 5 % of the way.")

rows = [
    (y + 130, 0, 500, [0, 100, 200, 300, 400, 500], 25, "25", BLUE, "value line"),
    (y + 250, 0, 1, [0, 0.2, 0.4, 0.6, 0.8, 1], 0.05, "0.05", VIOLET, "normalized"),
    (y + 370, 0, 100, [0, 20, 40, 60, 80, 100], 5, "5", EMERALD, "percent line"),
]
xs = []
for ly, lo, hi, ticks, v, label, col, name in rows:
    nl = NumberLine(d, X0, X1, ly, lo, hi, ticks, name=name)
    xs.append(nl.point(v, label, col))

steps = [("scale by 1 / 500", VIOLET), ("scale by 100", EMERALD)]
for i, (label, col) in enumerate(steps):
    y_top, y_bot = rows[i][0] + 34, rows[i + 1][0] - 34
    d.arrow(xs[i], y_top, xs[i + 1], y_bot, col)
    d.text(xs[i] + 22, (y_top + y_bot) / 2 + 4, label, 12.5, col, "bold", mono=True)

y = 556
d.legend(30, y, [(BLUE, "value (source line)"), (VIOLET, "normalized (0..1)"), (EMERALD, "percent (destination line)")])
y = d.footnotes(
    30, y + 40, ["All three lines start at 0, so two scale operations are enough. Figure 3 handles lines that do not "
                 "start at 0."])
d.save(height=y + 10)
