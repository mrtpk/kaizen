"""Figure 2: the padding trap. The folded bias assumes every window point holds z_in.

Values: zero_points.py: 5x5 input, 3x3 SAME conv, z_in = -64; corner window has 5 padded
points whose weights sum to -40; pad 0 -> corner off by -2560 (-3.94 int8 steps);
16 of 25 outputs wrong, all on the border.
"""

from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, Diagram, _c, _t

d = Diagram(width=1100)
d.header("Why the padding must be the zero point",
         "Int8 on Silicon  ·  part 2  ·  5x5 input  ·  3x3 conv with SAME padding  ·  z_in = -64",
         params="folded bias = bias - z_in * sum(all 9 weights): it assumes every point of the window holds z_in")

C = 40


def window(x0, y0, pad_value, col_pad):
    """Top-left 3x3 window: the first row and column are padding."""
    for r in range(3):
        for c in range(3):
            padded = r == 0 or c == 0
            col = col_pad if padded else BLUE
            d.rect(x0 + c * C, y0 + r * C, C - 4, C - 4, fill=_t(col), stroke=col, rx=4, sw=1.2)
            label = str(pad_value) if padded else "q"
            d.text(x0 + c * C + (C - 4) / 2, y0 + r * C + 23, label, 12, _c(col) if padded else INK,
                   "bold" if padded else "normal", "middle", mono=True)


# panel 1: pad with z_in
y = 116
d.panel(y, 190, 1, "What if we pad with z_in?",
        "Each padded point adds z_in * w, and the folded bias removes exactly that. The corner is exact.")
window(80, y + 66, -64, AMBER)
d.text(250, y + 96, "padded points hold -64 = real 0.0", 13, AMBER, "bold")
d.text(250, y + 120, "MAC adds  -64 * (sum of the 5 padded weights) = -64 * -40 = 2560", 12.5, INK, mono=True)
d.text(250, y + 144, "folded bias removes the same 2560", 12.5, INK, mono=True)
d.text(250, y + 168, "corner output = reference  (-50248)", 12.5, EMERALD, "bold", mono=True)

# panel 2: pad with 0
y += 210
d.panel(y, 190, 2, "What if we pad with 0?",
        "The MAC adds nothing for the padded points, but the bias still removes 2560. The corner is off.")
window(80, y + 66, 0, PINK)
d.text(250, y + 96, "padded points hold 0 = real 1.0039 (not 0.0)", 13, PINK, "bold")
d.text(250, y + 120, "MAC adds  0", 12.5, INK, mono=True)
d.text(250, y + 144, "folded bias still removes 2560", 12.5, INK, mono=True)
d.text(250, y + 168, "corner output = -52808: off by -2560 = -3.94 int8 steps", 12.5, PINK, "bold", mono=True)

# panel 3: which outputs are wrong
y += 210
d.panel(y, 290, 3, "Which outputs go wrong?",
        "Every output whose window touches the padding: all 16 border outputs of the 5x5 image.")
x0, y0 = 80, y + 70
for r in range(5):
    for c in range(5):
        border = r in (0, 4) or c in (0, 4)
        col = PINK if border else EMERALD
        d.rect(x0 + c * C, y0 + r * C, C - 4, C - 4, fill=_t(col), stroke=col, rx=4, sw=1.2)
d.text(320, y0 + 50, "16 border outputs wrong with pad 0", 13, PINK, "bold")
d.text(320, y0 + 76, "9 inner outputs exact either way", 13, EMERALD, "bold")
d.text(320, y0 + 116, "Small images and stride-2 layers have a large border share,", 12.5, MUTE)
d.text(320, y0 + 136, "so this is not a corner case in practice.", 12.5, MUTE)

y += 310
d.legend(30, y, [(BLUE, "image value"), (AMBER, "padding = z_in"), (PINK, "padding = 0 / wrong"), (EMERALD, "exact")])
y = d.footnotes(
    30, y + 40, ["TFLite skips padded points, which is the same as real 0.0. Padding with z_in gives the same result."])
d.save(height=y + 10)
