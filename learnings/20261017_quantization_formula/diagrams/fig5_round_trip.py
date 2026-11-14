"""Figure 5: round-trip error, clipping error, and symmetric vs asymmetric ranges.

Values: quantization.py -> scale = 0.015686, zero_point = -64;
1.0 -> q 0 -> 1.0039 (error +0.0039, max scale / 2 = 0.007843);
3.5 -> q 127 -> 2.9961 (error -0.5039);
symmetric |r| <= 3: scale = 0.023622, zero_point = 0, 1.0 -> 42.
"""

from _numberline import NumberLine
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, Diagram

S, Z = 4 / 255, -64

d = Diagram(width=1100)
d.header("What do we lose when we go to int8 and back?",
         "Quantization Basics  ·  float range [-1, 3]  ·  scale = 0.015686  ·  zero_point = -64",
         params="dequantize(quantize(r)) = r + error")

# panel 1: rounding error, zoomed in around 1.0
y = 116
d.panel(y, 200, 1, "Inside the range: how big is the error?",
        "At most half a step (scale / 2 = 0.007843). 1.0 comes back as 1.0039.")
lo, hi = 0.975, 1.03
x_band = 160 + (S * (0 - Z) - S / 2 - lo) / (hi - lo) * 860
d.rect(x_band, y + 122, S * 860 / (hi - lo), 16, fill=PINK[1], stroke=None, rx=2)
nl = NumberLine(d, 160, 1020, y + 130, lo, hi, [])
x0q = nl.px(S * (0 - Z))
for q in (-1, 0, 1, 2):
    r = S * (q - Z)
    if lo <= r <= hi:
        x = nl.px(r)
        d.line(x, y + 118, x, y + 142, EMERALD[0], sw=2)
        d.text(x, y + 162, f"q = {q}", 12, EMERALD, "bold", anchor="middle", mono=True)
        d.text(x, y + 180, f"{r:.4f}", 12, MUTE, anchor="middle", mono=True)
x1 = nl.point(1.0, "1.0", BLUE)
d.arrow(x1 + 6, y + 96, x0q - 3, y + 96, PINK, sw=1.4)
d.arrow_label((x1 + x0q) / 2 + 30, y + 96, "+0.0039", PINK)

# panel 2: clipping error
y += 220
d.panel(y, 170, 2, "Outside the range: how big is the error?",
        "Anything. 3.5 is clipped to q = 127 and comes back as 2.9961 (error -0.5039).")
nl = NumberLine(d, 160, 1020, y + 120, -1.25, 3.75, [-1, 0, 1, 2, 3])
xe = nl.px(S * (127 - Z))
d.rect(nl.px(S * (-128 - Z)), y + 112, xe - nl.px(S * (-128 - Z)), 16, fill=EMERALD[1], stroke=None, rx=2)
d.text(nl.px(1.0), y + 104, "representable range", 12, EMERALD, "bold", anchor="middle")
x35 = nl.point(3.5, "3.5", PINK)
d.circle(xe, y + 120, 6, EMERALD[0])
d.text(xe, y + 164, "2.9961", 12, EMERALD, "bold", anchor="middle", mono=True)
d.arrow(x35 - 8, y + 150, xe + 30, y + 150, PINK, sw=1.4)

# panel 3: symmetric vs asymmetric
y += 190
d.panel(y, 196, 3, "Why not keep zero_point = 0 (symmetric)?",
        "It wastes levels when the range is lopsided. For [-1, 3], symmetric covers [-3, 3].")
rows = [
    ("asymmetric", -1.0, 3.0, "scale 0.015686 · zero_point -64 · 256 levels on [-1, 3]", BLUE),
    ("symmetric", -3.0, 3.0, "scale 0.023622 · zero_point 0 · 1/3 of the levels unused", AMBER),
]
for i, (name, a, b, note, col) in enumerate(rows):
    ly = y + 96 + i * 54
    d.text(170, ly + 4, name, 13, col, "bold", anchor="end")
    nl = NumberLine(d, 190, 640, ly, -3.25, 3.25, [] if i == 0 else [-3, -1, 0, 1, 3])
    d.rect(nl.px(a), ly - 7, nl.px(b) - nl.px(a), 14, fill=col[1], stroke=col, rx=3, sw=1.2)
    if name == "symmetric":
        d.rect(nl.px(-3.0), ly - 7, nl.px(-1.0) - nl.px(-3.0), 14, fill=PINK[1], stroke=PINK, rx=3, sw=1.2, dash="3 2")
    d.text(670, ly + 4, note, 12, INK, mono=True)

y += 216
d.legend(30, y, [(BLUE, "float value"), (EMERALD, "int8 levels"), (PINK, "error / unused"), (AMBER, "symmetric range")])
y = d.footnotes(30, y + 40, ["Symmetric uses -127..127 so that r and -r map to q and -q. Weights are often symmetric; activations after ReLU are not."])
d.save(height=y + 10)
