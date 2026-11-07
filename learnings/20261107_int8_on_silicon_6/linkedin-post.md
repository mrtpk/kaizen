# LinkedIn post: Int8 on Silicon - Part 6: What I would do differently

**Attach:** `fig4_what_held_up.png`, `fig1_overflow.png`, `fig2_bit_budget.png`, `fig3_scorecard.png` as a carousel (in this order, as in the post).

---

Int8 on Silicon, part 6, the last one: what held up, and what I would change.

What held up: one post-processing unit for every operation, and a fixed shift of 20 with one rounding. I only understood the 20 fully this week: alpha × acc is the output with 20 fraction bits, and an int8 output plus 4 spare bits leaves exactly 20 in a 32-bit product. With that headroom, it was the largest shift there is.

What I'd refine, found by re-running everything in NumPy:
→ Constants truncated with int() instead of rounded. Rounding alpha alone takes the mismatch from 2.83% to 0.02%, better than TFLite.
→ One extra rounding in global average pooling.
→ Border rules that nobody wrote down: average pooling and max pooling at SAME padding.
→ An overflow check that ran on the already-wrapped int32 value. When it overflows, 127 becomes -128, and no check fires.

None of these showed up in full-dataset accuracy, which stayed on par with TFLite. They showed up in tests a dataset never produces: a huge accumulator, all-negative inputs, every entry of a table.

So: round every constant, check in a wider type, write the border rules down, test the edges. And if you can, ask the hardware team for 8 more bits in the multiplier.

The whole series, with code and figure scripts, is in my kaizen repo (link in comments). Thanks for reading.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261107_int8_on_silicon_1
