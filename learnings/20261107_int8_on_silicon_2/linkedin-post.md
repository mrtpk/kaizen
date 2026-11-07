# LinkedIn post: Int8 on Silicon - Part 2: Making zero points disappear

**Attach:** `fig1_four_terms.png`, `fig2_padding.png` as a carousel (in order).

---

Int8 on Silicon, part 2: where do the zero points go?

A conv output is sum (q_in − z_in)(q_w − z_w). Multiply it out and you get four terms. Only one of them, sum q_in·q_w, depends on both the input and the weights. That is the MAC.

With symmetric int8 weights (z_w = 0), two terms are zero. The last one, z_in · sum q_w, depends only on the weights, so it's a constant per channel. We folded it into the bias at compile time.

Result: the MAC array multiplies raw int8 values and never knows that a zero point exists. No adder in front of every multiplier.

The catch: the folded bias assumes every point of the window holds z_in. At the border, some points are padding. Pad with 0 and the corner of a 5×5 image is off by about 4 int8 steps, and all 16 border outputs are wrong.

The fix is one rule: pad with the input zero point (real 0.0), never with integer 0.

Code and figure scripts are in my kaizen repo (link in comments). Part 3: one unit, many operations.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261107_int8_on_silicon_2
