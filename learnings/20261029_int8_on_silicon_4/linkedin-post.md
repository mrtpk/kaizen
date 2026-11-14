# LinkedIn post: Int8 on Silicon - Part 4: Operations without a divider

**Attach:** `fig1_avgpool_as_conv.png`, `fig2_lut_softmax.png`, `fig3_global_pool.png` as a carousel (in order).

---

How do you compute an average on a chip with no divider?

You don't divide. You multiply.
→ Average pooling: the 1/9 goes into the integer multiplier, and the sum is a convolution with all-ones weights.
→ Sigmoid, tanh, hard-swish: an int8 input has only 256 values, so the function is a 256-entry table, built at compile time.
→ Softmax needs the whole row, so it ran on the host CPU.

The ideas held up. But re-checking everything in NumPy found four small gaps. One example: the tables used int() instead of round(), so 113 of 256 sigmoid entries were one step off.

Model accuracy was still on par with TFLite. Accuracy on a dataset does not show these gaps. Tests of the edge cases do.

Part 4 of Int8 on Silicon.

#Quantization #HardwareAcceleration #EdgeAI

---

**First comment:** https://www.thomaspaul.dev/2026/10/202610int8-on-silicon-4-operations-without-a-divider.html
