"""Figure 4: thinking-mode edits by irrelevance and AST (mean of non-live and live): the trade-off.

Values: qwen_tool_edits.py, part B (means of the seeds from study_results.json).
"""

from _wsp import SERIES_NAME, post
from svglib import AMBER, BLUE, EMERALD, INK, LINE, MUTE, NEUTRAL, VIOLET, Diagram

p = post()
dat = p.load()
PTS = [
    ("base", "base", NEUTRAL, (10, -10)),
    ("MIT edit", "rikiyat", AMBER, (10, 16)),
    ("E1b", "e1b", BLUE, (10, -8)),
    ("E4b", "e4b", BLUE, (-12, -10)),
    ("run 1", "run1", VIOLET, (10, 5)),
    ("v4", "v4", VIOLET, (10, 14)),
    ("v6", "v6", VIOLET, (10, 16)),
    ("v7", "v7", EMERALD, (10, 5)),
    ("v9", "v9", VIOLET, (10, 5)),
    ("v10", "v10", VIOLET, (10, 14)),
]
X0, Y0, W, H = 120, 190, 860, 380
XLO, XHI, YLO, YHI = 45, 80, 44, 60


def px(v: float) -> float:
    return X0 + W * (v - XLO) / (XHI - XLO)


def py(v: float) -> float:
    return Y0 + H - H * (v - YLO) / (YHI - YLO)


d = Diagram(width=1100)
d.header(
    "Fewer false calls cost correct calls",
    f"{SERIES_NAME}  ·  BFCL-V4, thinking mode, mean of the seeds of each edit",
    params="x: irrelevance (no call where no tool fits)  ·  y: AST, mean of non-live and live (a correct call where "
           "one fits)",
)
d.panel(
    120,
    H + 130,
    1,
    "Does any edit raise both?",
    "No. Every edit that raises irrelevance loses AST. E4b comes closest; v9 gains AST and loses irrelevance.",
)
d.rect(X0, Y0, W, H, fill="#ffffff", stroke=LINE, rx=2, sw=1)
for t in range(XLO, XHI + 1, 5):
    d.line(px(t), Y0 + H, px(t), Y0 + H + 5, MUTE, 1)
    d.text(px(t), Y0 + H + 20, f"{t}", 12, MUTE, anchor="middle", mono=True)
for t in range(YLO, YHI + 1, 4):
    d.line(X0 - 5, py(t), X0, py(t), MUTE, 1)
    d.text(X0 - 10, py(t) + 4, f"{t}", 12, MUTE, anchor="end", mono=True)
for lab, key, col, (dx, dy) in PTS:
    ast = (p.m(dat, key, "think", "non_live_ast") + p.m(dat, key, "think", "live")) / 2
    x, y = px(p.m(dat, key, "think", "irrelevance")), py(ast)
    d.circle(x, y, 7, fill=col[0])
    d.text(x + dx, y + dy, lab, 12.5, INK, anchor="end" if dx < 0 else "start")
y = 120 + H + 130 + 20
d.legend(
    30,
    y,
    [
        (NEUTRAL, "base"),
        (AMBER, "MIT math edit"),
        (BLUE, "v1 rank-1"),
        (VIOLET, "v2 candidates"),
        (EMERALD, "v7 (kept)"),
    ],
)
end = d.footnotes(
    30,
    y + 40,
    [
        "Base, v7: 3 seeds. MIT edit: 1 seed. The others: 2 seeds. Axes do not start at 0.",
        "E4b raises non-live AST but loses more live AST; it also loses 5 points of memory.",
    ],
)
d.save(height=end + 10)
