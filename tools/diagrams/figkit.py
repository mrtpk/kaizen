"""Small layout helpers shared by kaizen figures (built on svglib).

row():   up to 3 boxes side by side, with arrows between them
table(): a header row, a hairline, and mono rows; color_fn picks a color per cell
code():  a white code card with mono lines
"""

from svglib import INK, NEUTRAL, WHITE

BW, BH = 290, 72
XS = [60, 410, 760]


def row(d, y, boxes, arrow_col=None, bh=BH, mono_size=12):
    """boxes: list of (title, [lines], color). At most 3."""
    for i, (title, lines, col) in enumerate(boxes):
        d.box(XS[i], y, BW, bh, title, lines, col=col, mono_size=mono_size)
        if arrow_col is not None and i < len(boxes) - 1:
            d.arrow(XS[i] + BW, y + bh / 2, XS[i + 1] - 2, y + bh / 2, arrow_col)


def table(d, y, rows, cols, color_fn=None, pitch=26, size=12.5):
    """rows[0] is the header. color_fn(i, j, value) -> color for body cells."""
    for i, r in enumerate(rows):
        ry = y + i * pitch
        for j, (x, v) in enumerate(zip(cols, r)):
            if i == 0:
                d.text(x, ry, v, 13, INK, "bold")
            else:
                col = color_fn(i, j, v) if color_fn else INK
                d.text(x, ry, v, size, col, mono=j > 0)
        if i == 0:
            d.line(70, ry + 8, 1040, ry + 8, NEUTRAL[0])
    return y + pitch * len(rows)


def table_height(n_rows, pitch=26):
    return 96 + pitch * n_rows


def code(d, x, y, w, lines, size=12.5, pitch=19, color_fn=None, tabs=(0, 160, 260)):
    """A white card with mono lines. Returns the height.

    Runs of spaces collapse in the PNG renderer, so use "\t" to start a new column;
    `tabs` gives the x offset of each column. Leading spaces are kept as an indent.
    """
    h = 24 + pitch * len(lines)
    d.rect(x, y, w, h, fill=WHITE, stroke=NEUTRAL[0], rx=8, sw=1)
    cw = size * 0.602
    for i, s in enumerate(lines):
        col = color_fn(i, s) if color_fn else INK
        indent = (len(s) - len(s.lstrip(" "))) * cw
        for k, part in enumerate(s.lstrip(" ").split("\t")):
            d.text(x + 16 + indent + tabs[k], y + 28 + i * pitch, part, size, col, mono=True)
    return h
