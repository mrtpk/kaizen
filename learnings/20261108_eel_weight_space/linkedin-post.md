# LinkedIn post: Weight-Space Programming - Editing an eel detector without new data

**Attach:** `fig0_overview.png`, `fig1_read_weights.png`, `fig2_edit_ceiling.png`, `fig3_false_positives.png`, `photo1_duplicates.png`, `fig4_two_heads.png`, `fig5_wise.png`, `fig6_ladder.png`, `fig7_masks.png`, `photo3_refiner.png`, `fig8_next_time.png` as a carousel (in this order, as in the post).

---

I tried to fix a vision model by editing its weights, with no new data. Reading the weights worked. Fixing the misses did not.

The model is a YOLO26n segmentation model that finds eels in infrared video. The idea comes from MIT CSAIL's "Sparks of Weight-Space Programming", which did this for an LLM.

What the weights showed: 21 of 22 classes were dead, and one of three detection scales never learned eels. No inference needed.

What failed: the missed eels are short ones. Every eel that an edit won back cost 11 to 29 false positives. The features do not carry the information, so no edit can add it.

What worked: 329 of 403 false positives were second boxes on eels already found. A local-maximum rule, with nothing to fit, removed all but 1. On the half of the video that no edit saw, recall went from 0.894 to 0.944.

An agent, Claude Code, did almost all of the work. It read the weights, wrote and ran the edits, and logged every number. I set the goals and checked the results.

The weak points are in the post: one video, no footage from outside it, no error bars.

#WeightSpaceProgramming #ComputerVision #AIAgents

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611weight-space-programming-an-eel-detector.html
