# LinkedIn post: Int8 on Silicon - Part 5: The compiler frontend, and how we verified it

**Attach:** `fig1_frontend.png`, `fig2_rewrites.png`, `fig3_partition_verify.png` as a carousel (in order).

---

Our hardware had to match the reference model bit for bit. TFLite only had to match in accuracy. We never mixed up the two targets.

This part is the software around the math. The compiler frontend read an int8 TFLite model and rewrote the graph for the hardware. Every rewrite had one rule: the same int8 output before and after.

It also split each model between the accelerator and the host CPU. The rule people forget: also cut every edge that jumps over a host operation. If you don't, an accelerator subgraph waits for the host in the middle.

For every layer, the reference model wrote golden files in plain text, one value per line. A hardware engineer could grep a channel.

Part 5 of Int8 on Silicon.

#Compilers #HardwareAcceleration #EdgeAI

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611int8-on-silicon-5-compiler-frontend-and-verification.html
