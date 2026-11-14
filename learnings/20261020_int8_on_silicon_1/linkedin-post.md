# LinkedIn post: Int8 on Silicon - Part 1: One multiply, one shift

**Attach:** `fig0_series_map.png`, `fig4_where_m_comes_from.png`, `fig1_post_processing_unit.png`, `fig5_round_shift_bits.png`, `fig2_two_ways.png`, `fig3_fixed_shift_cost.png` as a carousel (in this order, as in the post).

---

Our int8 chip had no floating point. Every layer still had to multiply by 0.0023050.

From 2020 to 2023, I wrote the reference model for an int8 neural network accelerator at Roviero. This series is the integer math behind it.

The problem: a layer ends with a 32-bit sum that must be scaled by a small real number and turned back into int8.

TFLite uses a 31-bit multiplier, a per-channel shift, and two roundings. Our hardware could shift by only a few fixed amounts. So we fixed the shift at 20 and put all the precision into one integer per channel: 0.0023050 × 2^20 ≈ 2416. One multiply, one shift.

The result was never more than 1 step away from TFLite.

While writing this, I found that our 3 % mismatch did not come from the single rounding. It came from truncating 2416.99 to 2416. With 2417, our math is closer to exact than TFLite (0.02 % vs 0.18 %).

Thanks to Sambhu Surya Mohan, who guided much of this work.

#Quantization #HardwareAcceleration #EdgeAI

---

**First comment:** https://www.thomaspaul.dev/2026/10/202610int8-on-silicon-1-one-multiply-one-shift.html
