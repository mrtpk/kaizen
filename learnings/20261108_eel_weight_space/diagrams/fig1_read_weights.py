"""Figure 1: what the class biases say, and where the misses are.

Values: eel_weight_space.py, part B: class biases; recall by eel length (1,003 eels, every fifth frame).
"""

from _eel import SERIES_NAME, data, hbars
from figkit import table
from svglib import BLUE, MUTE, NEUTRAL, PINK, Diagram

d0 = data()
lg, rec, dg = d0["logged"], d0["recall_by_length"], d0["diag"]
dead, eel = lg["class_bias"]["dead"], lg["class_bias"]["eel"]

d = Diagram(width=1100)
d.header(
    "The weights show a one-class model with a dead scale",
    f"{SERIES_NAME}  ·  class biases of the deployed head, and recall by eel length",
    params=f"found {dg['found']} of {dg['eels']} eels  ·  "
           f"detections by stride: 8 -> {dg['by_stride']['8']}, 16 -> {dg['by_stride']['16']}, "
           f"32 -> {dg['by_stride']['32']}",
)
y1 = 120
d.panel(
    y1,
    175,
    1,
    "What do the 22 class biases say?",
    "21 rows are identical and pinned low. The eel row at stride 32 equals them: that scale never learned eels.",
)


def fmt(v: float) -> str:
    return f"{v:+.2f}"


table(
    d,
    y1 + 90,
    [
        ["Classifier row", "stride 8", "stride 16", "stride 32"],
        ["21 unused classes", *map(fmt, dead)],
        ["American_Eel (class 21)", *map(fmt, eel)],
    ],
    [70, 420, 620, 820],
    color_fn=lambda i, j, v: PINK[0] if (i == 2 and j == 3) else ("#111827" if i == 2 else MUTE),
)
y2 = y1 + 195
d.panel(
    y2,
    75 + 26 * len(rec) + 20,
    2,
    "Which eels does the model miss?",
    "Short ones. Under 60 px, it finds fewer than 4 in 10. Above 150 px, nearly all.",
)
rows = []
for r in rec:
    name = f"{r['lo']}+ px" if r["hi"] >= 10**6 else f"{r['lo']}-{r['hi']} px"
    col = PINK[0] if r["recall"] < 0.8 else BLUE[0]
    rows.append((f"{name} ({r['eels']} eels)", 100 * r["recall"], f"{100 * r['recall']:.1f} %", col))
hbars(d, 40, y2 + 70, rows, 0, 100, width=560, label_w=220)
y = y2 + 75 + 26 * len(rec) + 40
d.legend(
    30,
    y,
    [(BLUE, "recall at 80 % or more"), (PINK, "recall under 80 % / dead stride-32 row"), (NEUTRAL, "dead classes")],
)
end = d.footnotes(
    30,
    y + 40,
    ["Eel length = longest side of the label polygon, in 640 x 360 frame pixels. Confidence 0.25, input 512 px."],
)
d.save(height=end + 10)
