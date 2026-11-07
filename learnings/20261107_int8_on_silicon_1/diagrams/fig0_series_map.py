"""Figure 0: the map of the series. Six parts around one post-processing unit."""

from figkit import row
from svglib import AMBER, BLUE, EMERALD, NEUTRAL, VIOLET, Diagram

d = Diagram(width=1100)
d.header("Int8 on Silicon: the series",
         "Int8 on Silicon  ·  six notes on the integer math of an int8 accelerator (Roviero, 2020-2023)",
         params="prequel: Quantization Basics - the quantization formula, from number lines")

y = 116
d.panel(y, 170, 1, "How does one int8 layer compute its output?",
        "The core math: one multiply and one shift, the zero points, and every operation on one unit.")
row(d, y + 74, [
    ("part 1", ["one multiply, one shift", "alpha, shift 20, rounding"], BLUE),
    ("part 2", ["zero points disappear", "folded bias, padding"], BLUE),
    ("part 3", ["one unit, many ops", "add, multiply, ReLU family"], BLUE),
], NEUTRAL[0])

y += 190
d.panel(y, 170, 2, "What about everything else?",
        "Operations without a divider, the software around the math, and a look back.")
row(d, y + 74, [
    ("part 4", ["no divider", "pooling, tables, softmax"], AMBER),
    ("part 5", ["compiler frontend", "rewrites, partition, golden files"], VIOLET),
    ("part 6", ["what I would do differently", "the gaps, and what held up"], EMERALD),
], NEUTRAL[0])

y += 190
d.legend(30, y, [(BLUE, "integer math"), (AMBER, "operations without a divider"), (VIOLET, "software"), (EMERALD, "looking back")])
y = d.footnotes(30, y + 40, ["Every number in the series comes from a NumPy script next to each note."])
d.save(height=y + 10)
