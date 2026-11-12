# LinkedIn post: Weight-Space Programming - Index: the experiments and the lessons

DRAFT: not approved by the author.

**Attach:** none (text post), or `fig1_ceiling.png` of the Qwen post.

---

I have now tried weight-space programming on two real models: a YOLO26n eel detector and Qwen3.5-0.8B on tool use. In both, a coding agent read and wrote the weights; I set the goals and the rules.

What held in both: reading the weights gave fast, correct first guesses, and exact surgery was safe. An edit of one decision moved the model along a trade-off: recall against false positives, false calls against correct calls. No edit added what the model lacked.

The biggest wins were not where I looked first. For the eel model, a rule that changes no weight gave most of the gain. For Qwen, a ten-line ceiling calculation was the most useful result, and it came last.

The checklist I now run before the first edit: the ceiling of the goal, a split by time or source, a look at 20 failures by eye, a census of every error type, a cross-table of label against source, and a held-out probe ceiling.

The index page lists every experiment with its post and its code. I add a row for each new one.

First comment: https://github.com/mrtpk/kaizen/tree/master/learnings/20261112_wsp_index
