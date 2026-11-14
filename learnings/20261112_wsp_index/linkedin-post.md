# LinkedIn post: Weight-Space Programming - Index: the experiments and the lessons

**Attach:** none (text post), or `fig1_ceiling.png` of the Qwen post.

---

Two experiments in editing model weights directly. Neither fix came from where I looked first.

I tried weight-space programming on a YOLO eel detector and on Qwen3.5-0.8B for tool calls. In both, a coding agent read and wrote the weights. I set the goals and the rules.

What held in both:
→ Reading the weights gave fast, correct first guesses.
→ An edit moved the model along a trade-off. It never added what the model lacked.
→ The biggest win was not an edit. For the eel model, a rule that changes no weight. For Qwen, a ten-line calculation that showed the goal was out of reach.

So my checklist now starts before any edit: compute the ceiling of the goal, and look at 20 failures by eye.

The index lists every experiment, with its post and its code.

#WeightSpaceProgramming #LLM #AIAgents

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611weight-space-programming-index.html
