"""Figure 3: what a fixed shift of 20 costs (precision) and why it fits (bit budget).

Values: one_multiply_one_shift.py, section "what a fixed shift costs";
overflow only when |acc * M| > 2^31 / 2^20 = 2048.
"""

from svglib import AMBER, EMERALD, INK, NEUTRAL, PINK, VIOLET, Diagram, _c, _t

d = Diagram(width=1100)
d.header("What does a fixed shift of 20 cost?",
         "Int8 on Silicon  ·  part 1  ·  alpha = int(M * 2^20)",
         params="TFLite keeps 31 bits for every M; a fixed shift keeps fewer bits as M gets smaller")

# panel 1: precision table
y: float = 116
rows = [
    ("0.5", "524288", "20", "+0.000 %"),
    ("0.1", "104857", "17", "-0.001 %"),
    ("0.01", "10485", "14", "-0.007 %"),
    ("0.0023 (our example)", "2416", "12", "-0.041 %"),
    ("0.001", "1048", "11", "-0.055 %"),
    ("0.0001", "104", "7", "-0.818 %"),
]
H = 110 + 24 * len(rows)
d.panel(y, H, 1, "How precise is alpha?",
        "Fine for M above about 0.001. Below that, alpha has few bits and the error grows fast.")
cols = [80, 400, 600, 800]
for x, h in zip(cols, ["M", "alpha", "bits in alpha", "alpha error"]):
    d.text(x, y + 84, h, 13, INK, "bold")
d.line(70, y + 92, 1040, y + 92, NEUTRAL[0])
for i, r in enumerate(rows):
    ry = y + 114 + i * 24
    col = PINK if r[0] == "0.0001" else INK
    for x, v in zip(cols, r):
        d.text(x, ry, v, 13, col, mono=True)

# panel 2: bit budget of the int32 product
y += H + 20
d.panel(y, 190, 2, "Why does alpha * acc fit in int32?",
        "alpha * acc is about (acc * M) * 2^20: the output with 20 fraction bits. It overflows only past 2048.")
X0, BW = 60, 980
bits = [(1, "sign", NEUTRAL), (4, "spare", AMBER), (7, "int8 magnitude", EMERALD),
        (20, "fraction (shifted out)", VIOLET)]
px = BW / 32
bx = float(X0)
for n, label, col in bits:
    w = n * px
    d.rect(bx, y + 96, w, 40, fill=_t(col), stroke=col, rx=3, sw=1.3)
    d.text(bx + w / 2, y + 88, f"{n} bit" + ("s" if n > 1 else ""), 12, _c(col), "bold", "middle", mono=True)
    d.text(bx + w / 2, y + 160, label, 12.5, _c(col), "bold", "middle")
    bx += w

y += 210
d.legend(30, y, [(EMERALD, "output bits"), (VIOLET, "fraction bits"),
         (AMBER, "spare bits"), (PINK, "large alpha error")])
y = d.footnotes(30, y + 40, [
    "Overflow happens only when |acc * M| > 2^31 / 2^20 = 2048, 16x beyond the int8 range (128) before clipping.",
    "Elementwise ops have M near 1 but tiny inputs (|q - zp| <= 255), so they fit too.",
])
d.save(height=y + 10)
