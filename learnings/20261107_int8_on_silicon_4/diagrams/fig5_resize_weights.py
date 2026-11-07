"""Figure 5: the four bilinear weights of one output point, exact vs truncated.

Values: no_divider.py, "one output point of the 8x8 -> 13x13 resize": point (1, 1),
dy = dx = 0.4231; exact x 1024: 340.83, 249.94, 249.94, 183.29; truncated: 340, 249, 249, 183 (sum 1021).
"""

from figkit import table, table_height
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, Diagram

d = Diagram(width=1100)
d.header("Four weights that should add up to 1",
         "Int8 on Silicon  ·  part 4  ·  bilinear resize 8x8 -> 13x13, output point (1, 1)",
         params="out = (w_ll * q_ll + w_lu * q_lu + w_rl * q_rl + w_ru * q_ru) >> 10,   weights in units of 1/1024")

# panel 1: geometry
y = 116
d.panel(y, 260, 1, "Where does the output point fall?",
        "Between four input pixels, 0.4231 of a pixel to the right and down. Each weight is the area opposite to its corner.")
x0, y0, S = 120, y + 80, 150
d.rect(x0, y0, S, S, fill="#ffffff", stroke=BLUE, rx=2, sw=1.3)
for cx, cy, lab, w in [(x0, y0, "q_ll", "340.83"), (x0, y0 + S, "q_lu", "249.94"),
                       (x0 + S, y0, "q_rl", "249.94"), (x0 + S, y0 + S, "q_ru", "183.29")]:
    d.circle(cx, cy, 6, BLUE[0])
    d.text(cx + (-14 if cx == x0 else 14), cy + (-10 if cy == y0 else 22), lab, 12, BLUE, "bold",
           "end" if cx == x0 else "start", mono=True)
px, py = x0 + 0.4231 * S, y0 + 0.4231 * S
d.line(px, y0, px, y0 + S, AMBER, sw=1, dash="3 3")
d.line(x0, py, x0 + S, py, AMBER, sw=1, dash="3 3")
d.circle(px, py, 6, AMBER[0])
d.text(px + 10, py - 8, "out", 12, AMBER, "bold", mono=True)
d.text(x0 + 0.5 * S, y0 - 10, "dx = dy = 0.4231", 12, MUTE, anchor="middle", mono=True)
d.text(380, y0 + 30, "w_ll = (1 - dy)(1 - dx) = 0.3328  ->  340.83 / 1024", 12.5, INK, mono=True)
d.text(380, y0 + 56, "w_lu = dy (1 - dx)       = 0.2441  ->  249.94 / 1024", 12.5, INK, mono=True)
d.text(380, y0 + 82, "w_rl = (1 - dy) dx       = 0.2441  ->  249.94 / 1024", 12.5, INK, mono=True)
d.text(380, y0 + 108, "w_ru = dy dx             = 0.1790  ->  183.29 / 1024", 12.5, INK, mono=True)
d.text(380, y0 + 140, "exact sum: 1024.00", 12.5, EMERALD, "bold", mono=True)

# panel 2: truncation
y += 280
rows = [
    ("weight", "exact x 1024", "ours: int(), one by one", "lost"),
    ("w_ll", "340.83", "340", "0.83"),
    ("w_lu", "249.94", "249", "0.94"),
    ("w_rl", "249.94", "249", "0.94"),
    ("w_ru", "183.29", "183", "0.29"),
    ("sum", "1024.00", "1021", "3.00"),
]
H = table_height(len(rows))
d.panel(y, H, 2, "What happens when each weight is truncated on its own?",
        "Each loses up to one unit. Together they lose 3 of 1024: the output is pulled slightly toward 0.")
table(d, y + 92, rows, [80, 330, 600, 900],
      lambda i, j, v: PINK if (i == 5 and j >= 2) else (EMERALD if i == 5 else INK))

y += H + 20
d.legend(30, y, [(BLUE, "input pixels"), (AMBER, "output point"), (EMERALD, "exact"), (PINK, "truncated")])
y = d.footnotes(30, y + 40, ["TFLite computes the weights from integer coordinates as (1024 - f) and f, so they always sum to exactly 2^20."])
d.save(height=y + 10)
