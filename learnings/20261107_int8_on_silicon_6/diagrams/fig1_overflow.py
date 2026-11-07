"""Figure 1: an overflow that flips the sign, and a guard that cannot see it.

Values: lessons.py, section 1: alpha 2416, acc 1,000,000; true product 2,416,000,000;
int32 product -1,878,967,296; >> 20: -1792 vs 2304; clip: -128 vs 127.
"""

from svglib import AMBER, BLUE, EMERALD, PINK, Diagram

d = Diagram(width=1100)
d.header("An overflow guard that cannot fire",
         "Int8 on Silicon  ·  part 6  ·  alpha = 2416, acc = 1,000,000 (|acc * M| = 2305, past the 2048 limit)",
         params="the check ran on the product after it was computed in int32, so it only ever saw the wrapped value")

BW, BH = 290, 72
XS = [60, 410, 760]


def row(y, boxes, col_arrow):
    for i, (title, lines, col) in enumerate(boxes):
        d.box(XS[i], y, BW, BH, title, lines, col=col, mono_size=12)
        if i < 2:
            d.arrow(XS[i] + BW, y + BH / 2, XS[i + 1] - 2, y + BH / 2, col_arrow)


y = 116
d.panel(y, 170, 1, "What does the hardware compute?",
        "The int32 product wraps to a negative number, and the clip turns the largest output into the smallest.")
row(y + 74, [
    ("int32 product", ["2416 * 1,000,000", "-> -1,878,967,296"], PINK),
    (">> 20", ["-1792", "(int64 value: 2304)"], PINK),
    ("clip", ["-128", "(int64 path: 127)"], PINK),
], PINK)

y += 190
d.panel(y, 170, 2, "Why did the checks not catch it?",
        "Both checks compared the wrapped value with itself or with the int32 range it is already in.")
row(y + 74, [
    ("range check", ["is -1,878,967,296", "outside int32? no"], AMBER),
    ("reference check", ["int64(int32 * int32)", "== int32 * int32: yes"], AMBER),
    ("result", ["no warning,", "wrong sign"], PINK),
], AMBER)

y += 190
d.panel(y, 170, 3, "What would have caught it?",
        "Compute the product in 64 bits first, then check, or let the hardware saturate instead of wrap.")
row(y + 74, [
    ("int64 product", ["2,416,000,000", "outside int32: yes"], EMERALD),
    ("saturate", ["clamp to 2^31 - 1", "-> output 127"], EMERALD),
    ("test", ["one test with", "a huge accumulator"], BLUE),
], EMERALD)

y += 190
d.legend(30, y, [(PINK, "flipped result"), (AMBER, "the checks we had"), (EMERALD, "the fix"), (BLUE, "the test we needed")])
y = d.footnotes(30, y + 40, ["It needs |acc * M| > 2048, which is rare in a trained network, so the gap can stay hidden for a long time."])
d.save(height=y + 10)
