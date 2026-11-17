---
layout: post
title:  "Weight-Space Programming - Can an agent quantize an LLM better than a recipe?"
date:   2026-11-17 00:00:00 +0530
categories: deep-learning
mathjax: true
---

Can a coding agent find a better way to store an LLM in fewer bits than the published methods? This note is about three rounds in which an agent tried, on Llama-3-8B and Qwen3-32B. The short answer is no, on the part I cared about most: a cheap automatic method matched the agent's bit allocation. The agent's measurements of the model were still useful.

A recent essay by Dwarkesh Patel, [The avalanche that hit math is coming for AI development][patel], argues that AI will soon automate AI research. Quantization research is a small, well-scored piece of AI research, so it is a fair place to test that claim at home.

The agent was Claude Code (CLI) with Opus 5.5, on AMD MI325X GPUs in Docker. It wrote and ran every experiment, and kept a lab log. I set the goals and the rules, and I approved each change to them. The code, the plans and the lab notes are in [wsp-llm-quant][repo]. The [index][index] lists all the notes of this series.

## Why is quantization in this series?
Weight-space programming means changing a trained network by reading and writing its weights directly, from an understanding of what each part computes. Quantization writes every weight again, in fewer bits. In these rounds the agent first measured which linears and which activations lose accuracy, and then chose a format and a bit width for each linear from those measurements. No round used gradient training. So the question here is the same as in the other notes: does an understanding of the model give a better edit than a recipe?

Before diving in, two facts about the setup. **The quantization is simulated**: the weights and activations are rounded to the low-bit grid and stored in bf16, so the accuracy does not depend on a kernel. And **the result files are not public**: they stayed on the GPU machine, so the numbers in this note are transcribed from the study's report and lab notes.

## Round 1: a better SpinQuant is not a win
The goal: 4-bit weights, 4-bit activations and a 4-bit KV cache (W4A4KV4) on Llama-3-8B, better than SpinQuant. No gradient training, and at most 20 numbers tuned on scores. To pass, every one of 4 seeds of the agent had to beat every seed of every baseline by a margin, on perplexity **and** on the zero-shot average.

The agent stayed inside SpinQuant's pipeline and tuned it: random Hadamard rotations, GPTQ with an asymmetric target, activation clipping. Its WikiText-2 gap to FP16 was 12 % smaller than SpinQuant's (7.220 against 7.370, FP16 6.138), and it passed the perplexity margin on all three test sets.

It failed on zero-shot. Every agent seed (worst 64.33) beat every baseline seed (best 63.33), but by 1.00 points, and the margin was 2.20 points. One bad SpinQuant seed (61.06) made that margin large. Under the protocol, the round is a FAIL.

My verdict was the same for another reason. About 2 % better perplexity is a result I could sell, but not a real win for the compute the agent had. It tuned the knobs of one published method and never looked at the model.

## Round 2: measure the model first
For Qwen3-32B, I changed the brief. The method and the format were free. The goal was the essence of quantization: the same accuracy for less memory or less compute. Before any test data was read, we fixed two cost measures (M: stored bits per weight; B: weight bits × activation bits, which is not a speed measure), the seeds, and a bar. The first bar was +1 % perplexity; on selection data only 8-bit weights passed it, so I approved +4 % before the test.

This time the agent profiled the model first: it quantized each of the 448 linears alone and measured the damage.

![Two panels of c4dev perplexity increase, each linear of Qwen3-32B quantized alone with RTN. 1: 3-bit weights. L63.down_proj +28.67, the next linear L62.gate_proj +0.05, the other 446 linears together +1.22. 2: 4-bit activations per token. L6.gate_proj +1835, the next linear L63.gate_proj +6.8, the other 446 together +9.](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261117_wsp_llm_quant/resources/fig1_damage.png)
*Figure: with plain rounding (RTN), one linear makes almost all of the weight damage, and one makes almost all of the activation damage.*

It means that with 3-bit RTN weights, the last MLP's down projection makes 95.8 % of the summed damage, and only 14 of 448 linears cost more than +0.01. Its inputs reach 8,768, so the problem is in what the weights multiply, not in the weights. For 4-bit activations, layer 6's gate projection makes 99.1 % of it: it creates the massive activation. The agent found two more facts this way. FP8 activations are almost free (+0.014), while INT8 activations are not (+0.144). And the K cache needs one scale per channel (+0.018 at 4 bits), not per token (+3.27).

Why do small groups fix 4-bit activations? Let's say a few channels are 100 times larger than the rest. With one scale per token, the step of the 4-bit grid follows the outliers, and every ordinary value rounds to almost the same level. With one scale per 32 values, only the groups that hold an outlier pay. This is a synthetic illustration, not a study result; the full script is available [here][script].

{% highlight python %}
pt, g32 = outlier_demo_ordinary()   # 256 tokens x 1024 channels, 8 channels x100
print(f"ratio on ordinary channels: {pt / g32:.1f}x")
print("Group-32 error on ordinary channels is at least 5x smaller than per-token. Is pass?", g32 * 5 < pt)
# relative error, ordinary channels: per-token 5.996   group-32 0.804
# ratio on ordinary channels: 7.5x
# Group-32 error on ordinary channels is at least 5x smaller than per-token. Is pass? True
{% endhighlight %}

From these measurements the agent built its finals. The test sets were used once, with 4 seeds per final and up to 8 per baseline:

| Config | W bits | A bits | WikiText-2 | fw25test | Zero-shot | Result |
| --- | --- | --- | --- | --- | --- | --- |
| GPTQ-asym W4, L63.down INT8, FP8 activations | 4.266 | 8.20 | +1.75 % | +1.60 % | 69.94 | B win |
| W3 in layers 0-33, W4 in 34-63, L63.down INT8 | 3.743 | 16 | +2.48 % | +2.37 % | 70.23 | M win |
| baseline: GPTQ W4g128 | 4.250 | 16 | +2.12 % | +1.92 % | 70.10 | pass |
| baseline: FP8 W8A8 | 8.002 | 8.20 | +0.55 % | +0.48 % | 70.93 | pass |

The W4A8 finals are more accurate than GPTQ W4 with 16-bit activations, at 0.52 times the B cost. The mixed 3/4-bit final passes the bar 0.507 bits below GPTQ W4. Two more mixed finals passed under the frozen rule, but one zero-shot seed of each was below the guard (68.92 and 68.87 against 68.94), so I do not count them. In the W4A4KV4 format, the agent's design was 5.8 times closer to bf16 than a QuaRot-style baseline (+4.39 % against +25.65 % on WikiText-2), but both failed the bar.

The agent also found a bug in its own baseline. The first QuaRot-style run lost 73 %, far from the published range. The cause: the K cache was quantized per token, without the q/k rotation after RoPE.

### A speed check
B counts bits; it does not measure speed. So I asked for a real decode test. FP8 W8A8 with the vendor kernel ran at 82.8 tokens/s at batch 1, against 53.4 for bf16. No vendor W4A8 kernel exists on this ROCm stack, so the agent wrote its own Triton kernels. They reached 51.7 tokens/s, slower than bf16. The agent had predicted 125 to 135. This is a result about our kernels, not about the format, and the report makes no speed claim for W4.

## Round 3: does the agent's allocation beat a cheap search?
Round 2 had one gap. The M baselines use one bit width everywhere. A standard automatic allocation, given the same formats, could perhaps find an equal result without any analysis. Round 3 tested exactly that.

The control is cheap. It quantizes each linear alone with the same GPTQ pipeline, in each format of the menu (W3g128, W3g64, W4g128, INT8), and measures the KL divergence of the model's output. That took 4.6 GPU-hours. Then it fills the bit budget greedily: it always takes the upgrade with the largest KL drop per added bit. Let's look at the core of it.

{% highlight python %}
while True:
    best, best_rate = None, 0.0
    for k in menus:
        if pos[k] + 1 < len(hulls[k]):
            (b0, kl0), (b1, kl1) = hulls[k][pos[k]], hulls[k][pos[k] + 1]
            extra = (b1 - b0) * sizes[k] / total
            if mean_bits() + extra <= budget + 1e-12 and (kl0 - kl1) / extra > best_rate:
                best, best_rate = k, (kl0 - kl1) / extra
    if best is None:
        return {k: hulls[k][pos[k]][0] for k in menus}
    pos[best] += 1
{% endhighlight %}

On a toy menu, it gives INT8 first to a small, sensitive k_proj, because each bit there is cheap. The real control did the same: it put INT8 into 40 of the 128 k/v projections, upgraded 70 linears in layers 0-33, where the agent upgraded none, and left L63.down at W4. Its allocation looks nothing like the agent's.

A measurement defect came first. In bf16, every format of every linear gave a KL of about 1e-3, even INT8: any change moved the output by the same rounding noise through 64 layers. The agent found it after 2 layers, before any control result, and moved the sensitivity step to fp32.

The verdict rule was fixed before the control ran. With d the gap in mean perplexity and s the pooled seed std:

$$ s = \sqrt{\frac{s_{agent}^2 + s_{control}^2}{2}}, \quad \text{agent wins if } d > 2s \text{ on both test sets} $$

A tie withdraws the claim that the agent's allocation is better.

![Two panels of perplexity increase against bf16, mean of 4 seeds with a one-std bar, agent in blue and control in amber. 1: at 3.7432 bits, WikiText-2 agent +2.479 and control +2.491, fw25test agent +2.367 and control +2.390. 2: at 3.6891 bits, WikiText-2 agent +2.769 and control +2.693, fw25test agent +2.446 and control +2.570.](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261117_wsp_llm_quant/resources/fig2_control.png)
*Figure: at both bit budgets, the greedy control and the agent are within seed noise.*

{% highlight python %}
v34, _ = verdict(pa["late34"]["agent"], pa["late34"]["control"])
v44, _ = verdict(pa["late44"]["agent"], pa["late44"]["control"])
print("Both targets are a tie, so claim A is withdrawn. Is pass?", v34 == v44 == "TIE")
# late34 (W 3.7432): wt2: d +0.012, 2s 0.274  fw25: d +0.023, 2s 0.227 -> TIE
# late44 (W 3.6891): wt2: d -0.076, 2s 0.173  fw25: d +0.124, 2s 0.299 -> TIE
# Both targets are a tie, so claim A is withdrawn. Is pass? True
{% endhighlight %}

At 3.74 bits the gaps are 0.012 and 0.023 points; a win needed more than 0.27 and 0.23. At 3.69 bits the gaps go in opposite directions, and the control is the more robust config: its worst zero-shot seed is 69.99, while the agent's (68.92) fails the guard. So the claim of round 2 that the agent's configs give more accuracy per bit is withdrawn. Phase A took 12.4 GPU-hours.

Two very different allocations give the same accuracy. At about 3.7 bits per weight on this model, where the extra bits go matters little, within reasonable allocations. And the L63.down INT8 exception, the agent's best-known finding, is not necessary under GPTQ with 16-bit activations at this budget. It was necessary for RTN.

## What I take from it
- **What survives is the measurement, not the allocation.** FP8 against INT8 activations, small groups for 4-bit activations, per-channel K, the q/k rotation after RoPE, the source of the massive activation: these are format and structure findings, and the control does not touch them. No automatic control was run for them either.
- **A cheap control is the test of an agent's claim.** Without phase A, round 2 would read as "the agent beats the baselines at 3.7 bits". The baselines were simply too weak.
- **Proxies are not results.** B is not speed, and the agent's speed prediction was about 2.5 times too high. Perplexity is not zero-shot: below 4 bits, zero-shot moved with the calibration seed before perplexity showed it (selection data, not confirmed on test).
- **The agent was a careful researcher.** It found the protocol defects of round 1, the bug in its own QuaRot baseline, and the bf16 noise floor in its own control, each before the result that depended on it. It did not find a better idea than a greedy fill.

Does this say anything about the essay's claim? Not much on its own: one agent, one task, three rounds. But the useful work here was measurement and honest controls, and the agent did that well. The new idea did not come.

So, in short: the agent's measurements of Qwen3-32B found real format and structure facts, but a greedy KL allocation matched its bit allocation, so its main claim was withdrawn.

**Reference:**
- [Sparks of Weight-Space Programming][mit], Takehi, Teoh, Zhang and Khattab, MIT CSAIL.
- [The avalanche that hit math is coming for AI development][patel], Dwarkesh Patel.
- [SpinQuant][spinquant], [QuaRot][quarot].
- [wsp-llm-quant][repo]: code, plans, lab notes and reports of all three rounds.
- [Weight-Space Programming - Index][index]

[mit]: https://rikiyat.github.io/blog/weight-space-programming
[patel]: https://substack.com/home/post/p-224624765
[spinquant]: https://github.com/facebookresearch/SpinQuant
[quarot]: https://github.com/spcl/QuaRot
[repo]: https://github.com/mrtpk/wsp-llm-quant
[index]: https://www.thomaspaul.dev/2026/11/202611weight-space-programming-index.html
[script]: https://github.com/mrtpk/kaizen/blob/master/learnings/20261117_wsp_llm_quant/llm_quant.py
