"""Figure 5: round_shift at the bit level, and why a plain shift is not enough.

Values: one_multiply_one_shift.py, section "rounding at the bit level":
28,617,520 -> 27 (exact 27.292); 30,006,720 -> 29 (plain >> 28, exact 28.617);
-28,617,520 -> -27 (plain >> -28, exact -27.292).
"""

from svglib import AMBER, BLUE, EMERALD, INK, MUTE, PINK, Diagram, _c, _t

d = Diagram(width=1100)
d.header("Rounding with a shift, a mask and the sign",
         "Int8 on Silicon  ·  part 1  ·  round_shift(x, 20)",
         params="r = (|x| >> 20) + ((|x| >> 19) & 1);   if x < 0: r = -r")

X0 = 60


def word(y, x, label_int, label_bit):
    """Draw |x| as 32 bits: [12 integer bits][bit 19][19 lower bits]."""
    a = abs(x)
    bits = format(a, "032b")
    cw = 28
    for i, ch in enumerate(bits):
        pos = 31 - i
        col = BLUE if pos >= 20 else (AMBER if pos == 19 else (AMBER[0], "#f3f4f6", None))
        fill = _t(col) if pos >= 19 else "#f3f4f6"
        stroke = _c(col) if pos >= 19 else "#d1d5db"
        d.rect(X0 + i * cw, y, cw - 3, 30, fill=fill, stroke=stroke, rx=3, sw=1)
        d.text(X0 + i * cw + (cw - 3) / 2, y + 20, ch, 12.5, INK if pos >= 19 else MUTE, anchor="middle", mono=True)
    d.text(X0 + 6 * cw, y - 8, label_int, 12, BLUE, "bold", "middle")
    d.text(X0 + 12 * cw + 12, y + 50, label_bit, 12, AMBER, "bold", "middle")
    d.text(X0 + 22 * cw, y - 8, "19 lower bits, dropped", 12, MUTE, "bold", "middle")


y = 116
d.panel(y, 180, 1, "28,617,520 >> 20: which bits matter?",
        "The 12 high bits are the integer part, 27. The first dropped bit is 0, so 27 is already the nearest integer.")
word(y + 96, 28_617_520, "integer part = 27", "first dropped bit = 0")

y += 200
d.panel(y, 180, 2, "30,006,720 >> 20: when does it round up?",
        "Exact value 28.617. The first dropped bit is 1, so the result is 28 + 1 = 29. A plain >> gives 28.")
word(y + 96, 30_006_720, "integer part = 28", "first dropped bit = 1  ->  +1")

y += 200
d.panel(y, 150, 3, "Why handle the sign separately?",
        "A plain >> on a negative number rounds toward minus infinity.")
d.box(60, y + 70, 450, 56, "plain:  -28,617,520 >> 20", ["= -28   (exact -27.292)"], col=PINK, mono_size=12.5)
d.box(590, y + 70, 450, 56, "round_shift: round |x|, then negate", ["= -27   (round half away from zero)"], col=EMERALD, mono_size=12.5)

y += 170
d.legend(30, y, [(BLUE, "integer part"), (AMBER, "first dropped bit"), (PINK, "plain shift"), (EMERALD, "round_shift")])
y = d.footnotes(30, y + 40, ["In hardware: one shift, one AND with a 1-bit mask, one add, and a sign flip. No divider."])
d.save(height=y + 10)
