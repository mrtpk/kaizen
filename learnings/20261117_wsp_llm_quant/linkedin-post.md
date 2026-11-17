# LinkedIn post: Weight-Space Programming - Can an agent quantize an LLM better than a recipe?

**Attach (carousel, in order):** `resources/fig1_damage.png`, `resources/fig2_control.png`

---

A coding agent measured Qwen3-32B and found where quantization loses accuracy. But on bit allocation, a greedy search of a few lines matched it, in these first three rounds.

Quantization stores a model's weights in fewer bits, for example 3 or 4 bits instead of 16. The hard part is where the accuracy goes.

Claude Code (Opus 5.5) ran every experiment on AMD MI325X GPUs. I set the goals and the rules.

What worked: the agent measured the model first. With plain 3-bit rounding, one linear layer of 448 made 96 % of the damage. With 4-bit activations, another single layer made 99 %. From this it found facts that hold: FP8 activations cost almost nothing, INT8 activations do not, and the attention keys need one scale per channel.

What did not: its bit allocation. A control measured each layer alone and filled the bit budget greedily. At 3.7 bits per weight, the two allocations look nothing alike and score the same, within seed noise. So the agent's main claim was withdrawn, by a rule we fixed before the control ran.

The lesson for me: a cheap control is the real test of an agent's research claim. And the experiment design needs more work. I plan more rounds.

Part 3 of my weight-space programming series.

#WeightSpaceProgramming #LLM #Quantization

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611weight-space-programming-llm-quantization.html
