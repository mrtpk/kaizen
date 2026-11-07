"""Figure 4: two accumulator dumps split a hardware mismatch in two.

The reference model has two named accumulator dumps for convolutions, before and after the
zero-point correction. In the end both held the same values, because the correction moved
into the post-processing step. The figure shows the idea.
"""

from figkit import row
from svglib import AMBER, BLUE, PINK, Diagram

d = Diagram(width=1100)
d.header("Before and after the correction",
         "Int8 on Silicon  ·  part 2  ·  a debugging idea for a convolution that differs on the hardware",
         params="dump the int32 accumulator twice: the raw MAC sum, and the sum after the zero-point correction")

y = 116
d.panel(y, 170, 1, "A layer differs on the hardware: where does it come from?",
        "Compare the raw MAC sums first. That one comparison separates the MAC path from the constants.")
row(d, y + 74, [
    ("compare 'before'", ["raw MAC sums,", "hardware vs golden"], BLUE),
    ("'before' differs", ["MAC array or", "data movement"], PINK),
    ("only 'after' differs", ["constants: bias, alpha,", "pad value, zero points"], AMBER),
])

y += 190
d.legend(30, y, [(BLUE, "the comparison"), (PINK, "MAC path"), (AMBER, "compile-time constants")])
y = d.footnotes(30, y + 40, ["A padding mismatch like the one above shows up only at the border outputs."])
d.save(height=y + 10)
