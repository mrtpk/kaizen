# LinkedIn post: Int8 on Silicon - Part 3: One unit, many operations

**Attach:** `fig1_add_three_passes.png`, `fig2_multiply_split.png`, `fig3_relu_family.png` as a carousel (in order).

---

Int8 on Silicon, part 3: one small post-processing unit (bias, multiply, shift, zero point, clip), and every operation has to fit in it.

Add: the bias slot removes each input's zero point, alpha rescales each input to the output scale, one rounding at the end. For our test scales it was identical to TFLite on 200,000 random pairs.

Constants: "add 0.75" goes into the bias, "multiply by -1.5" goes into alpha. One pass, no second tensor.

ReLU, ReLU6, LeakyReLU, PReLU and min(x, c) are one kernel: a slope for negative inputs and a clip. Only the constants change.

Multiply had to fit a 32-bit accumulator. We split M = s1·s2/s_out into two square roots, one per input, at 2^10 each, which was more accurate than the uneven split before it (46% vs 62% of outputs off exact rounding). Measuring it again this week, there is room to go further: for an SE block (x · sigmoid), one alpha is only 17, and putting all of M on one input would bring the error to 0.05%.

The models matched TFLite accuracy either way.

Code and figure scripts are in my kaizen repo (link in comments). Part 4: operations without a divider.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261107_int8_on_silicon_3
