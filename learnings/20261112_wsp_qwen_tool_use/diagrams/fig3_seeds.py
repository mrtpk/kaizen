"""Figure 3: every seed of base, E4b, v4 and v7 in both modes, against the acceptance rule.

Values: qwen_tool_edits.py, part B (seeds from study_results.json).
"""

from _wsp import SERIES_NAME, post
from svglib import BLUE, EMERALD, INK, LINE, MUTE, NEUTRAL, VIOLET, Diagram

p = post()
dat = p.load()
MODELS = [("base", "base", NEUTRAL), ("E4b (v1)", "e4b", BLUE), ("v4 (v2)", "v4", VIOLET), ("v7 (v2)", "v7", EMERALD)]


def strip(d: Diagram, y: float, mode: str, lo: float, hi: float) -> None:
    x0, w = 220, 760
    base_max = max(p.seeds(dat, "base", mode))
    for i, (lab, key, col) in enumerate(MODELS):
        yy = y + i * 34
        d.text(x0 - 14, yy + 5, lab, 12.5, INK, anchor="end")
        d.line(x0, yy, x0 + w, yy, LINE, 1)
        for s in p.seeds(dat, key, mode):
            d.circle(x0 + w * (s - lo) / (hi - lo), yy, 7, fill=col[0])
        res = "" if key == "base" else ("pass" if p.passes(dat, key, mode) else "fail")
        d.text(x0 + w + 16, yy + 5, f"{p.m(dat, key, mode):.2f}  {res}", 12.5, INK, mono=True)
    bx = x0 + w * (base_max - lo) / (hi - lo)
    d.line(bx, y - 16, bx, y + 34 * 3 + 14, MUTE, 1.2, dash="4 3")
    d.text(bx + 6, y - 6 + 34 * 4, f"highest base seed {base_max:.2f}", 12, MUTE, mono=True)
    for t in range(int(lo), int(hi) + 1):
        tx = x0 + w * (t - lo) / (hi - lo)
        d.text(tx, y + 34 * 4 + 16, f"{t}", 12, MUTE, anchor="middle", mono=True)


d = Diagram(width=1100)
d.header(
    "Only v7 clears every base seed in thinking mode",
    f"{SERIES_NAME}  ·  BFCL-V4 overall, one dot per seed, web search not run",
    params="rule: every edit seed above every base seed  ·  2 vs 2 passes by chance 1 in 6, 3 vs 3 1 in 20",
)
y1, h1 = 120, 245
d.panel(
    y1,
    h1,
    1,
    "Thinking mode: which edit passes the rule?",
    "v7, with 3 seeds against 3. E4b ties the highest base seed; v4 is below it.",
)
strip(d, y1 + 90, "think", 20, 23)
y2, h2 = y1 + h1 + 20, 245
d.panel(
    y2,
    h2,
    2,
    "Non-thinking mode: which edit passes the rule?",
    "All three, with 2 seeds against 2 (provisional). E4b is highest.",
)
strip(d, y2 + 90, "nothink", 17, 21)
y = y2 + h2 + 20
d.legend(30, y, [(NEUTRAL, "base"), (BLUE, "v1 rank-1 edit"), (VIOLET, "v2 one neuron"), (EMERALD, "v2 two neurons")])
end = d.footnotes(
    30,
    y + 40,
    ["Numbers after each row: the mean of the seeds and the result of the rule. Dashed line: the highest base seed."],
)
d.save(height=end + 10)
