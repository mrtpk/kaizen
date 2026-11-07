# LinkedIn post: Int8 on Silicon - Part 4: Operations without a divider

**Attach:** `fig1_avgpool_as_conv.png`, `fig2_lut_softmax.png`, `fig3_global_pool.png` as a carousel (in order).

---

Int8 on Silicon, part 4: how do you average, or compute a sigmoid, on hardware with no divider and no floating point?

Average pooling: the 1/9 goes into the integer multiplier, and the sum is a depthwise convolution with all-ones weights. The MAC array sums, the post-processing unit divides.

Sigmoid, tanh, hard-swish: an int8 input has only 256 values, so any one-input function is a 256-entry table, computed once at compile time.

Softmax needs the whole row, so it ran in float on the host.

The ideas held up. But re-running everything in NumPy this week found four small gaps:
→ SAME average pooling divides by 9 at the border, where TFLite divides by the valid count: border outputs off by up to 37.
→ Global average pooling rounded twice: 60% of outputs one or two steps off. One rounding is exact.
→ The tables used int() instead of round(): 113 of 256 sigmoid entries one step off, toward zero.
→ Resize weights were truncated one by one, so they summed to as low as 1021 instead of 1024.

All cheap to fix. Final model accuracy was still on par with TFLite.

Code and figure scripts are in my kaizen repo (link in comments). Part 5: the compiler frontend and verification.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261107_int8_on_silicon_4
