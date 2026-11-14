# LinkedIn post: Int8 on Silicon - Part 3: One unit, many operations

**Attach:** `fig1_add_three_passes.png`, `fig2_multiply_split.png`, `fig3_relu_family.png`, `fig4_ops_as_constants.png` as a carousel (in this order, as in the post).

---

One small unit ran add, multiply, ReLU, LeakyReLU and PReLU on our int8 chip. It can only add a bias, multiply, shift and clip.

The trick is that every operation becomes constants for that unit:
→ Add: the bias removes each input's zero point, and the multiplier rescales. On 200,000 random pairs, it matched TFLite exactly.
→ "Add 0.75" goes into the bias. "Multiply by -1.5" goes into the multiplier.
→ The whole ReLU family is one kernel: a slope for negative inputs and a clip value.

Multiply was the hard one, because the product must fit in 32 bits. We split the scale evenly between the two inputs. Measured again now, putting the whole scale on one input would cut the error to 0.05 %.

Part 3 of Int8 on Silicon.

#Quantization #HardwareAcceleration #EdgeAI

---

**First comment:** https://www.thomaspaul.dev/2026/10/202610int8-on-silicon-3-one-unit-many-operations.html
