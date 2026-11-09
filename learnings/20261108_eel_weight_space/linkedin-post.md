# LinkedIn post: Weight-Space Programming - Editing an eel detector without new data

**Attach:** `fig0_overview.png`, `fig1_read_weights.png`, `fig2_edit_ceiling.png`, `fig3_false_positives.png`, `photo1_duplicates.png`, `fig4_two_heads.png`, `fig5_wise.png`, `fig6_ladder.png`, `fig7_masks.png`, `photo3_refiner.png`, `fig8_next_time.png` as a carousel (in this order, as in the post).

---

Can we fix a model by editing its weights, with no new data? @MIT CSAIL's "Sparks of Weight-Space Programming" did it for an LLM. I tried it on a YOLO26n segmentation model from @Ultralytics that finds eels in infrared video.

Reading the weights worked. The biases showed 21 of 22 classes dead, and one of three detection scales that never learned eels. Cutting the head to one class changed no metric, to 16 decimal places.

Fixing the misses by editing did not. The missed eels are short ones, and every eel that a weight edit won back cost 11 to 29 false positives. The features do not carry the information, so no edit can add it.

The false positives were a different story. 329 of 403 were second boxes on eels that were already found. A local-maximum rule in the head, with nothing to fit, removed all but 1.

Two more closed-form moves helped. YOLO26 ships a second head that inference throws away. Averaging the two heads' weights destroyed the model (box mAP50 0.145). But a head of double width averages their outputs exactly, inside one normal checkpoint. And after a fine-tune that lost box accuracy, moving only a quarter of the way from the old weights to the new ones kept part of the gain and none of the loss.

On the second half of the video, which no edit was fitted on: fitness 1.3346 -> 1.4004, recall 0.894 -> 0.944.

Masks were the hard part. An eel is 9 px wide, and the mask grid cell is 5 px. A small learned refiner fixed the oversized masks (area 1.14x -> 0.99x), but not the IoU.

The weak points are in the note too: no footage from outside the training video, settings chosen on the half I score, no error bars. Notes, code and figure scripts are in my kaizen repo (link in comments).

What excites me most: this is not only for LLMs. YOLO26 is a CNN, and its weights showed the defects directly: 21 dead classes, a detection scale that never learned eels, a second head that inference discards. We can take CNNs that are already trained, read their weights, and find their defects, if any, before we collect new data or retrain.

The MIT work used an LLM to edit an LLM. Here an LLM agent (Claude Code by @Anthropic) did weight-space programming on a CNN almost fully autonomously: it read the weights, wrote and ran the edits and evaluations, and logged every number. I set the goals and checked the results. I am fascinated by this, and honestly a bit mind-blown.

I also see this as an observability tool. Today we watch a model through its outputs: metrics, drift, error logs. The weights are a second signal. A check of the weights needs no data and no inference: dead classes, dead scales, unused heads. It can run on every checkpoint, before deployment, as a health check. More experiments to follow.

#WeightSpaceProgramming #DeepLearning #ComputerVision #CNN #MLObservability #YOLO #AIAgents #ClaudeCode

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_eel_weight_space
