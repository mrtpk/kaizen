---
layout: post
title:  "Weight-Space Programming - Index: the experiments and the lessons"
date:   2026-11-12 00:00:00 +0530
categories: deep-learning
mathjax: true
---

Each experiment in this series edits the weights of a real model, and this page ties them together. It lists every experiment, its post and its code, and the lessons that hold across them. I add a row for each new experiment.

## Before diving in, what is weight-space programming?
Weight-space programming means changing a trained network by reading and writing its weights directly, from an understanding of what each part computes, mostly instead of training it with gradient descent. One exception in these notes: an interpolation after a fine-tune. The term comes from [Sparks of Weight-Space Programming][mit] by Takehi, Teoh, Zhang and Khattab at MIT CSAIL. In their headline result, an LLM edited 131,072 weights of Qwen3.5-0.8B to remove a looping bug, and MATH-500 rose from 7.9 % to 48.1 %.

My question in every experiment is the same: can I understand a model from its weights, and can I fix a real error with an edit that I can explain?

In every experiment, a coding agent wrote and ran the code. I set the goal and the rules, and I read the results. Each post names the agent.

## The experiments

| # | Post | Model and task | Agent | Hardware | Result | Code |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [Editing an eel detector without new data][eel] | YOLO26n-seg, 3.06M parameters, eels in infrared video | Claude Code | CPU only | held-out fitness 1.3346 to 1.4004, recall 0.894 to 0.944; most of the gain from a rule that changes no weight | [wsp-yolo-eel][repo-eel] (no checkpoint, frames or labels; crop figures only) |
| 2 | [Teaching Qwen3.5-0.8B when to call a tool][qwen] | Qwen3.5-0.8B on BFCL-V4 tool use | Cursor (v1), Claude (v2) | one RTX 5090 | +0.66 to +2.61 overall; two edits, each better in one mode only | [wsp-qwen-tools][repo-qwen]; v7 model on [Hugging Face][hf-qwen] |

Side notes are short notes next to the experiments. They use gradient training, so they are not weight-space programming experiments, and they have no row in the table and no repository.

- [Side note: an agent drives the training of the eel detector][side-eel]: Claude Code (Opus 5.5) ran a 5-fold training study of the eel detector on one AMD MI325X GPU and read the weights of the runs. Averages of independent runs collapse; averages of children of one trunk are safe but not better. The candidate was never compared against the production model.

Both repositories hold the code, the logs and the reports. The eel checkpoint, frames and labels are not public, so the first study does not run from its repository as it is. The second one runs from the public base model up to a full BFCL run (levels 1 to 4 of its README); the edit cells need new forward passes on a GPU.

## What held in both experiments

### Reading the weights works
The weights gave fast, correct first guesses. In the eel detector, the class biases showed 21 dead classes and a dead detection scale before any frame ran. In Qwen, in non-thinking mode, a linear read at the input of the last MLP knew better than the model itself when to call a tool. Exact surgery was safe: cutting the eel head from 22 classes to 1 changed no metric.

### An edit did not add what the model lacks
In both models, the errors that remained after the edits were not in one editable place. The missed eels were short or cut off, and the best linear read on their features did not separate them. Qwen's remaining failures were wrong argument values, multi-turn state and memory. No edit changed those. Two small models do not prove a general limit, but in both cases the fix probably needs new data, a new input or a larger model, not an edit.

### An edit of one decision moves along a trade-off
The eel row edit traded recall against false positives: each eel that it won back cost 11 to 29 false positives. The Qwen edits traded false calls against correct calls. In thinking mode, every edit that raised irrelevance lost AST. An edit can find a better point on the trade-off. None removed it.

### The biggest wins were not where I looked first
In the eel study, most false positives were second boxes on eels that were already found. A local-maximum rule in the head removed them, with nothing to fit, and gave most of the gain. In Qwen, the ten-line ceiling calculation of the score formula was the most useful result, and it came last.

### In-sample numbers were too optimistic
Neither study has a blind test. The eel settings were chosen on the half that I score. The first Qwen edit was fitted on the benchmark itself, and the benchmark chose the second one from four candidates.

## Do these before the first edit
This list comes from both studies. Each step can stop a project early, and an early stop is cheaper than a failed edit.

1. **Calculate the ceiling of the goal.** Use the metric formula and a census of the base model's errors. For each part of the metric, take its weight times the share of its failures that the edit can fix. If the sum is below the goal, change the goal or the method.
2. **Make the evaluation split first.** Split by time, recording or data source, not by random rows: neighbouring video frames and rows from one dataset leak. Keep a third split for the final test.
3. **Check the data setup.** Folder names, train/val/test lists, label classes, training logs. The eel training list held every frame, so no held-out data existed.
4. **Look at 20 or more failures by eye.** Many "missed short eels" were cut off or almost invisible, and the labels may be at fault (not checked). The Qwen "text" rows meant something else than I assumed.
5. **Count every error type**, not only the one you expect.
6. **Cross-tabulate the label against the data source.** When one source gives all positives and another all negatives, a read learns the source.
7. **Measure the probe ceiling on held-out data.** If the best linear read at a site cannot separate the cases, no linear edit at that site can fix them.
8. **Tune the regularization before you report a negative result.** With too little ridge, no Qwen read beat the model. With more, they did.

## What worked, and what did not

| Failure type | Can an edit fix it? | Seen in |
| --- | --- | --- |
| A decision that a linear read at one site separates well | yes, with a trade-off | Qwen: call or text |
| A dead or unused part | yes, exactly | eel: 21 dead classes, a dead scale |
| A missing post-processing step | yes, often with no new weights | eel: duplicate boxes |
| Two good parts that the model does not combine | yes, with an exact closed-form merge | eel: the two heads, merged by widening |
| Missing information in the input or the features | not in these studies | eel: short and cut-off eels |
| A resolution limit of the architecture | not in these studies | eel: masks from a 1/4-resolution grid |
| A capability limit | not in these studies | Qwen: argument values, multi-step state, memory |

Techniques that worked: reading biases and output rows, exact surgery with an exact check, an output merge by widening, a parameter-free rule in the graph, interpolation after a fine-tune, a rank-1 edit of the last MLP, and programming a single neuron with a ridge gate.

Techniques that failed: weight averages of networks that do not share a basis (the side note shows why: the activations shrink, and on one fold a shared trunk prevents it), threshold shifts on scores that the features cannot separate, "no call" targets taken from refusal rows, a push at the decision token when the reasoning already planned the action, and post-processing that the errors make useless.

## How I evaluate an edit now
1. Count both sides of the trade-off before I accept an edit.
2. Accept a gain only when every seed of the edit is above every seed of the base, and use four or more seeds. With no real effect, the rule passes by chance once in 6 tries with 2 seeds against 2, once in 20 with 3 against 3, and once in 70 with 4 against 4.
3. Do not choose between candidates on the final test. If I must, report the result as optimistic.
4. Measure the side effects, the speed and the memory, together with the accuracy.
5. Write every evaluation as one line in one log, and make every table from that log.

## What comes next
More experiments follow. Each one gets a post, a wsp- repository, and a row in the table above. A side note gets a post and a line under the table.

So, in short: weights are a good place to read a model and a safe place for exact surgery, and an edit of one decision buys a better point on a trade-off, not a new capability. The cheapest step in both studies was the one I did last: the ceiling of the goal.

**Reference:**
+ [Sparks of Weight-Space Programming (Takehi, Teoh, Zhang and Khattab, MIT CSAIL)][mit]
+ [Weight-Space Programming - Editing an eel detector without new data][eel]
+ [Weight-Space Programming - Teaching Qwen3.5-0.8B when to call a tool][qwen]
+ [Weight-Space Programming - Side note: an agent drives the training of the eel detector][side-eel]
+ [wsp-yolo-eel][repo-eel], [wsp-qwen-tools][repo-qwen]

[mit]: https://rikiyat.github.io/blog/weight-space-programming
[eel]: https://www.thomaspaul.dev/2026/11/202611weight-space-programming-an-eel-detector.html
[qwen]: https://www.thomaspaul.dev/2026/11/202611weight-space-programming-qwen-tool-calls.html
[repo-eel]: https://github.com/mrtpk/wsp-yolo-eel
[repo-qwen]: https://github.com/mrtpk/wsp-qwen-tools
[side-eel]: https://www.thomaspaul.dev/2026/11/202611weight-space-programming-side-note-agent-drives-training.html
[hf-qwen]: https://huggingface.co/mrtpk/Qwen3.5-0.8B-wsp-tool-use
