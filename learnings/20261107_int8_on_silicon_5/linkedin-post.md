# LinkedIn post: Int8 on Silicon - Part 5: The compiler frontend, and how we verified it

**Attach:** `fig1_frontend.png`, `fig2_rewrites.png`, `fig3_partition_verify.png` as a carousel (in order).

---

Int8 on Silicon, part 5: the software around the math.

The frontend read an int8 TFLite model into a graph and rewrote it for the hardware: SAME convolutions became an explicit Pad + VALID convolution, Splits became Slices, and a PixelShuffle that arrived as Reshape-Transpose-Reshape-DepthToSpace became one DepthToSpace. Every rewrite had one rule: same int8 output before and after.

It also split each model between the accelerator and the host. The rule people forget: cutting edges at the supported/unsupported boundary is not enough. You also have to cut every edge that jumps over a host op in execution order, or an accelerator subgraph ends up waiting for the host in the middle.

For every layer, the reference model wrote a config, the constants, and golden files: plain text, one value per line, so a hardware engineer can grep a channel.

Two targets, never mixed up: the hardware had to match the golden files bit for bit. TFLite had to match in accuracy, on full datasets.

Code and figure scripts are in my kaizen repo (link in comments). Part 6, the last one: what I would do differently.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261107_int8_on_silicon_5
