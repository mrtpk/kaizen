# LinkedIn post: Int8 on Silicon - Part 6: What I would do differently

**Attach:** `fig4_what_held_up.png`, `fig1_overflow.png`, `fig2_bit_budget.png`, `fig3_scorecard.png` as a carousel (in this order, as in the post).

---

Our int8 accelerator matched TFLite accuracy on full datasets. It still had four bugs that no dataset showed.

I found them while writing this series, by re-running every piece in NumPy:
→ Constants truncated instead of rounded.
→ One extra rounding in global average pooling.
→ Border rules for pooling that nobody wrote down.
→ An overflow check that ran after the value had already wrapped. So 127 became -128, and no check fired.

What held up: one post-processing unit for every operation, and a fixed shift of 20 with one rounding.

If I built it again: round every constant, check overflow in a wider type, write the border rules down, and test the edges a dataset never produces.

Last part of Int8 on Silicon.

#Quantization #HardwareAcceleration #EdgeAI

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611int8-on-silicon-6-what-i-would-do-differently.html
