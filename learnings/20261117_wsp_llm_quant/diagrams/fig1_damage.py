"""Figure 1: where the quantization damage of Qwen3-32B comes from (round 2, P1 profiling).

Each of the 448 linears is quantized alone; the bar is the c4dev perplexity increase.
Values: llm_quant.py (round2.p1_* in study_results.json, from round2-qwen3-32b/NOTES.md, P1.3c).
"""

from _llm_quant import SERIES_NAME, post
from svglib import BLUE, INK, NEUTRAL, PINK, Diagram

p = post()
r2 = p.load()["round2"]
w, a = r2["p1_w3g128_rtn_c4dev"], r2["p1_a4_per_token_c4dev"]
X, LW, BW = 30, 330, 520


def bars(d: Diagram, y: float, rows: list[tuple[str, float, str, str]], vmax: float) -> float:
    for i, (label, v, txt, col) in enumerate(rows):
        yy = y + i * 34
        d.text(X + LW - 12, yy + 14, label, 12.5, INK, anchor="end")
        d.rect(X + LW, yy, max(2.0, BW * v / vmax), 22, fill=col, stroke=None, rx=3)
        d.text(X + LW + BW + 16, yy + 15, txt, 12.5, INK, mono=True)
    return y + len(rows) * 34


d = Diagram(width=1100)
d.header(
    "One linear makes almost all of the damage",
    f"{SERIES_NAME}  ·  LLM quantization  ·  Qwen3-32B, 448 linears, each quantized alone",
    params="metric: c4dev perplexity increase (bf16 7.383, 16 windows)  ·  RTN, no GPTQ",
)
y1, h1 = 120, 210
d.panel(y1, h1, 1, "3-bit weights (W3g128) on one linear at a time: which linear hurts?",
        f"L63.down_proj: {w['L63.down_proj']} of {w['all_448']}. "
        f"Only {w['linears_above_0.01']} of 448 linears cost more than +0.01.")
rest_w = w["all_448"] - w["L63.down_proj"] - w["L62.gate_proj"]
bars(d, y1 + 80, [
    ("L63.down_proj", w["L63.down_proj"], f"+{w['L63.down_proj']}", BLUE[0]),
    ("L62.gate_proj (next largest)", w["L62.gate_proj"], f"+{w['L62.gate_proj']}", NEUTRAL[0]),
    ("the other 446 linears, sum", rest_w, f"+{rest_w:.2f}", NEUTRAL[0]),
], w["all_448"])
y2, h2 = y1 + h1 + 20, 210
d.panel(y2, h2, 2, "4-bit activations per token on one linear at a time: which linear hurts?",
        f"L6.gate_proj: +{a['L6.gate_proj']} of +{a['all_448']}. It makes the massive activation.")
rest_a = a["all_448"] - a["L6.gate_proj"] - a["L63.gate_proj"]
bars(d, y2 + 80, [
    ("L6.gate_proj", a["L6.gate_proj"], f"+{a['L6.gate_proj']}", PINK[0]),
    ("L63.gate_proj (next largest)", a["L63.gate_proj"], f"+{a['L63.gate_proj']}", NEUTRAL[0]),
    ("the other 446 linears, sum", rest_a, f"+{rest_a:.0f}", NEUTRAL[0]),
], a["all_448"])
y = y2 + h2 + 20
d.legend(30, y, [(BLUE, "weight problem"), (PINK, "activation problem"), (NEUTRAL, "everything else")])
end = d.footnotes(30, y + 40, [
    "Sums of one-at-a-time costs. Errors do not add exactly when many linears are quantized together.",
    "With GPTQ instead of RTN, L63.down_proj costs much less (round 3). Simulated quantization in bf16.",
])
d.save(height=end + 10)
