# LinkedIn post: Int8 on Silicon - Part 1: One multiply, one shift

**Attach:** `fig0_series_map.png`, `fig4_where_m_comes_from.png`, `fig1_post_processing_unit.png`, `fig5_round_shift_bits.png`, `fig2_two_ways.png`, `fig3_fixed_shift_cost.png` as a carousel (in this order, as in the post).

---

From 2020 to 2023, I wrote the reference model for an int8 neural network accelerator at Roviero. The hardware team gave us the constraints, we did the math in software, and then they built the same math in silicon. I'm writing it down before I forget it: "Int8 on Silicon", part 1. Thanks to my manager, Sambhu Surya Mohan, who guided me through much of it.

Every int8 layer ends with the same problem: an int32 accumulator has to be multiplied by a small real number M (say 0.0023050) and turned back into int8, with no floating point.

TFLite splits M into a 31-bit mantissa and a per-channel shift, and rounds twice. Our hardware could shift by only a few fixed amounts, because each extra shift amount costs area.

So we fixed the shift at 20 and put all the precision into one integer per channel: alpha = M × 2^20 = 2416. One multiply, one rounding shift.

Result: never more than 1 away from TFLite, and different in 3% of the values. Accuracy on full datasets was on par.

The part I only found while writing this: the 3% was not from the single rounding. We truncated alpha (2416.99 → 2416). With alpha rounded to 2417, our result matches exact rounding better than TFLite does (0.02% vs 0.18%). One rounding was right; rounding alpha too closes the gap.

And why 20? alpha × acc ≈ output × 2^20. An int8 output plus 20 fraction bits is 28 bits, so the product fits in int32 with 4 bits to spare.

Notes, code and figure scripts are in my kaizen repo (link in comments). Part 2: making zero points disappear.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261107_int8_on_silicon_1
