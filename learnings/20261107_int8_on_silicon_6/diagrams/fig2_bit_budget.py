"""Figure 2: alpha and the product share the same 32 bits.

Values: lessons.py, section 4.
"""

from svglib import AMBER, EMERALD, INK, NEUTRAL, PINK, VIOLET, Diagram

d = Diagram(width=1100)
d.header("The real limit: the width of the product",
         "Int8 on Silicon  ·  part 6  ·  alpha * acc ~ output * 2^shift must fit in the product",
         params="bits of alpha ~ shift + log2(M)        bits of the product ~ shift + output bits")


def table(y, rows, cols, color_row):
    for i, r in enumerate(rows):
        ry = y + i * 26
        for j, (x, v) in enumerate(zip(cols, r)):
            if i == 0:
                d.text(x, ry, v, 13, INK, "bold")
            else:
                d.text(x, ry, v, 12.5, color_row(i, r), mono=True)
        if i == 0:
            d.line(70, ry + 8, 1040, ry + 8, NEUTRAL[0])


y = 116
rows = [
    ("shift", "overflow when |output| >", "alpha bits, M = 0.0023", "alpha bits, M = 0.0001"),
    ("16", "32,768", "8", "3"),
    ("20  (ours)", "2,048", "12", "7"),
    ("23", "256", "15", "10"),
    ("24", "128", "16", "11"),
    ("28", "8", "20", "15"),
]
H = 96 + 26 * len(rows)
d.panel(y, H, 1, "With a 32-bit product, can a bigger shift buy precision?",
        "Only by giving up range. Past 24 the int8 range itself overflows. 20 keeps 4 spare bits.")
table(y + 92, rows, [80, 300, 560, 820],
      lambda i, r: EMERALD if "ours" in r[0] else (PINK if r[0] == "28" else (AMBER if r[0] == "24" else INK)))

y += H + 20
rows = [
    ("product width", "largest shift (int8 + 4 spare bits)", "alpha bits, M = 0.0001"),
    ("32 bits (ours)", "20", "7"),
    ("40 bits", "28", "15"),
    ("48 bits", "36", "23"),
]
H = 96 + 26 * len(rows)
d.panel(y, H, 2, "What would a wider product buy?",
        "8 more bits of product give 8 more bits of alpha, for every M. TFLite uses a 64-bit product.")
table(y + 92, rows, [80, 400, 820], lambda i, r: EMERALD if "ours" in r[0] else VIOLET)

y += H + 20
d.legend(30, y, [(EMERALD, "our choice"), (AMBER, "no spare bits"), (PINK, "int8 range overflows"), (VIOLET, "wider product")])
y = d.footnotes(30, y + 40, ["Shift 20 was the best a 32-bit product allows. The precision for small M was limited by the product width."])
d.save(height=y + 10)
