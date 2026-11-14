# LinkedIn post: Int8 on Silicon - Part 2: Making zero points disappear

**Attach:** `fig1_four_terms.png`, `fig2_padding.png` as a carousel (in order).

---

Our int8 multipliers never knew that zero points exist. One padding rule kept it that way.

Int8 values carry an offset, the zero point. A convolution with offsets expands into four terms. Only one of them needs the input and the weights together. The others are constants per channel, so we folded them into the bias at compile time.

So the multiplier array multiplies raw int8 values. No adder in front of every multiplier.

The catch is at the border. The folded bias assumes every input in the window holds the zero point. If you pad with integer 0 instead, every border output is wrong. In a 5×5 example, all 16 border outputs were off, by up to 4 steps.

The rule: pad with the input zero point (real 0.0), never with integer 0.

Part 2 of Int8 on Silicon.

#Quantization #HardwareAcceleration #EdgeAI

---

**First comment:** https://www.thomaspaul.dev/2026/10/202610int8-on-silicon-2-zero-points.html
