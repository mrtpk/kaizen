"""Figure 1: where the base score comes from, and the ceiling of a perfect single-turn edit.

Values: qwen_tool_edits.py, part A, section 2 (base thinking, mean of 3 seeds).
"""

from _wsp import SERIES_NAME, hbars, post
from svglib import AMBER, BLUE, EMERALD, LINE, NEUTRAL, PINK, Diagram

p = post()
dat = p.load()
runs = dat["models"]["base"]["think"]
avg = {k: p.mean([s[k] for s in runs]) for k in runs[0]}
base, perfect = p.overall(avg), p.perfect_single_turn(avg)

d = Diagram(width=1100)
d.header(
    "A perfect call decision cannot reach 60",
    f"{SERIES_NAME}  ·  Qwen3.5-0.8B on BFCL-V4, thinking mode, base model, mean of 3 seeds",
    params="overall = 0.1 non-live + 0.1 live + 0.1 irrelevance + 0.3 multi-turn + 0.4 agentic (memory, web search)",
)
y1, h1 = 120, 80 + 26 * 6
d.panel(
    y1,
    h1,
    1,
    "Where do the base model's points come from?",
    f"{0.1 * (avg['non_live_ast'] + avg['live'] + avg['irrelevance']):.2f} of {base:.2f} points come from the three "
    f"single-turn parts.",
)
rows = [
    ("non-live AST  (max 10)", 0.1 * avg["non_live_ast"], f"{0.1 * avg['non_live_ast']:.2f}", BLUE[0]),
    ("live  (max 10)", 0.1 * avg["live"], f"{0.1 * avg['live']:.2f}", BLUE[0]),
    ("irrelevance  (max 10)", 0.1 * avg["irrelevance"], f"{0.1 * avg['irrelevance']:.2f}", BLUE[0]),
    ("multi-turn  (max 30)", 0.3 * avg["multi_turn"], f"{0.3 * avg['multi_turn']:.2f}", AMBER[0]),
    ("memory  (max 20)", 0.2 * avg["memory"], f"{0.2 * avg['memory']:.2f}", AMBER[0]),
    ("web search  (max 20, not run)", 0.0, "0", NEUTRAL[0]),
]
hbars(d, 40, y1 + 70, rows, 30, width=500, label_w=280)
for i, mx in enumerate((10, 10, 10, 30, 20, 20)):
    d.line(320 + 500 * mx / 30, y1 + 70 + 26 * i, 320 + 500 * mx / 30, y1 + 88 + 26 * i, LINE, 2)
y2 = y1 + h1 + 20
h2 = 80 + 26 * 3
d.panel(
    y2,
    h2,
    2,
    "What does a perfect single-turn edit give?",
    f"Non-live, live and irrelevance at 100: {perfect:.2f}. The goal needs +{60 - base:.2f}; single-turn gives "
    f"+{perfect - base:.2f}.",
)
rows2 = [
    ("base", base, f"{base:.2f}", NEUTRAL[0]),
    ("perfect single-turn", perfect, f"{perfect:.2f}", EMERALD[0]),
    ("goal", 60.0, "60", PINK[0]),
]
hbars(d, 40, y2 + 70, rows2, 60, width=600, label_w=280)
y = y2 + h2 + 20
d.legend(30, y, [(BLUE, "single-turn parts"), (AMBER, "multi-step parts"), (EMERALD, "ceiling"), (PINK, "goal")])
end = d.footnotes(
    30,
    y + 40,
    [
        "Gray tick = the most that a part can give. Agentic = mean of memory and web search; web search was not run "
        "and counts as 0.",
        "The ceiling assumes that a decision edit could make every single-turn entry right. No real edit came close.",
    ],
)
d.save(height=end + 10)
