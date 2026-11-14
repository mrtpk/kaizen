# LinkedIn post: Weight-Space Programming - Teaching Qwen3.5-0.8B when to call a tool

**Attach:** `fig1_ceiling.png`, `fig2_two_forms.png`, `fig3_seeds.png`, `fig4_tradeoff.png` as a carousel (in this order, as in the post).

---

My goal was a BFCL tool-use score of 60 for Qwen3.5-0.8B, by editing weights only. A ten-line calculation shows that it was never possible.

Even a perfect single-turn model scores 34.44 on BFCL-V4. The rest of the score needs correct arguments over many calls. I did this calculation last. It belongs first.

The edits did work on one decision: call a tool, or answer in text. Two agents tried two kinds of edit. Each gained 0.66 to 2.61 points. Each won in one mode of the model only, and each traded false calls against correct calls.

The most useful failure: the first neuron learned which dataset a row came from, not whether the row needed a call. A simple table of label against source would have caught it before the fit.

Code and the edited model are linked in the post.

#WeightSpaceProgramming #LLM #ToolUse

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611weight-space-programming-qwen-tool-calls.html
