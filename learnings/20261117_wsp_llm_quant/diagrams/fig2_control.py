"""Figure 2: the agent's bit allocation against the greedy KL control (round 3 phase A).

Values: llm_quant.py (phase_a in study_results.json, from round2-qwen3-32b/NOTES.md, PHASE A VERDICT).
"""

from _llm_quant import SERIES_NAME, post
from svglib import AMBER, BLUE, INK, LINE, MUTE, Diagram

p = post()
pa = p.load()["phase_a"]
X0, WD, LO, HI = 300, 560, 2.0, 3.0


def sx(v: float) -> float:
    return X0 + WD * (v - LO) / (HI - LO)


def rows(d: Diagram, y: float, tgt: str) -> float:
    t = pa[tgt]
    yy = y
    for key, name in (("wt2", "WikiText-2"), ("fw25", "fw25test")):
        for who, col in (("agent", BLUE), ("control", AMBER)):
            m, s = t[who][key]
            d.text(X0 - 14, yy + 5, f"{name}, {who}", 12.5, INK, anchor="end")
            d.line(X0, yy, X0 + WD, yy, LINE, 1)
            d.line(sx(m - s), yy, sx(m + s), yy, col[0], 3)
            d.circle(sx(m), yy, 7, fill=col[0])
            d.text(X0 + WD + 20, yy + 5, f"+{m:.3f} ± {s:.3f}", 12.5, INK, mono=True)
            yy += 28
        yy += 8
    for tick in (2.0, 2.5, 3.0):
        d.text(sx(tick), yy + 6, f"+{tick:.1f} %", 12, MUTE, anchor="middle", mono=True)
    return yy


d = Diagram(width=1100)
d.header(
    "A greedy KL allocation ties the agent's allocation",
    f"{SERIES_NAME}  ·  LLM quantization  ·  Qwen3-32B, round 3 phase A, test sets used once",
    params="perplexity increase against bf16  ·  dot: mean of 4 seeds  ·  bar: ± 1 seed std",
)
y1, h1 = 120, 250
v34, ds34 = p.verdict(pa["late34"]["agent"], pa["late34"]["control"])
d.panel(y1, h1, 1, f"At {pa['late34']['W']} bits per weight: is the agent's allocation better?",
        f"No. {v34}: gaps {ds34['wt2'][0]:+.3f} and {ds34['fw25'][0]:+.3f}; "
        f"a win needs more than {2 * ds34['wt2'][1]:.3f} and {2 * ds34['fw25'][1]:.3f}.")
rows(d, y1 + 84, "late34")
y2, h2 = y1 + h1 + 20, 250
v44, _ = p.verdict(pa["late44"]["agent"], pa["late44"]["control"])
d.panel(y2, h2, 2, f"At {pa['late44']['W']} bits per weight: is the agent's allocation better?",
        f"No. {v44}, in opposite directions. "
        f"The agent's worst zero-shot seed ({pa['late44']['agent']['zs_worst']}) fails the guard.")
rows(d, y2 + 84, "late44")
y = y2 + h2 + 20
d.legend(30, y, [(BLUE, "agent: W3 in layers 0-33, W4 in 34-63, L63.down INT8"),
                 (AMBER, "control: KL of each linear alone, greedy fill")])
end = d.footnotes(30, y + 40, [
    "Same formats, GPTQ pipeline, calibration and seeds. "
    "The control puts INT8 into 40 of 128 k/v linears and keeps L63.down at W4.",
    "Rule fixed before any control result: a win needs a gap above 2 pooled seed std on both sets.",
])
d.save(height=end + 10)
