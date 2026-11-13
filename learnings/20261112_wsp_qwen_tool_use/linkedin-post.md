# LinkedIn post: Weight-Space Programming - Teaching Qwen3.5-0.8B when to call a tool

DRAFT: not approved by the author.

**Attach:** `fig1_ceiling.png`, `fig2_two_forms.png`, `fig3_seeds.png`, `fig4_tradeoff.png` as a carousel (in this order, as in the post).

---

Can a weight edit make a small LLM better at tool use? I tried it on Qwen3.5-0.8B with BFCL-V4. The goal was 60, from a base of about 20.

The goal was out of reach, and the score formula says so. Even a perfect single-turn model scores 34.44. The rest of the score needs correct arguments and state over many calls. I did this calculation last. It belongs first.

The edits did work on one decision: call a tool or answer in text. Two agents tried two forms. Cursor fitted a rank-1 edit of the last MLP: 3,670,016 values, fitted on the benchmark. Claude programmed two single neurons under the OpenWSP rules: 6,144 values, fitted on practice data only.

Each gained +0.66 to +2.61 points. Each won in one mode only: the two neurons in thinking mode (22.54 against 21.49), the rank-1 edit in non-thinking mode (20.29 against 19.55). And every edit traded false calls against correct calls.

The most useful failure: the first neuron learned which dataset a row came from, not whether it needed a call. Every "no call" row came from one dataset, every "call" row from another. A cross-table of label against source would have caught it before the fit.

Code, edits, ledgers and score tables: wsp-qwen-tools. The v7 model is on Hugging Face: mrtpk/Qwen3.5-0.8B-wsp-tool-use. The index of the series lists both experiments and the lessons across them.

First comment: https://github.com/mrtpk/kaizen/tree/master/learnings/20261112_wsp_qwen_tool_use
