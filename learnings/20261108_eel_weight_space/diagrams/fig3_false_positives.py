"""Figure 3: the false-positive census, and the peak-suppression rule.

Values: eel_weight_space.py, part B: census on all 4,986 frames at confidence 0.25; counts after peak suppression.
"""

from _eel import SERIES_NAME, data, hbars
from svglib import AMBER, BLUE, EMERALD, INK, MUTE, NEUTRAL, PINK, Diagram

lg = data()["logged"]
c, a = lg["census"], lg["after_peak"]
fps = c["dup"] + c["misloc"] + c["background"]
d = Diagram(width=1100)
d.header(
    "Most false positives were second boxes on found eels",
    f"{SERIES_NAME}  ·  every detection at confidence 0.25 on all {lg['frames']} frames",
    params=f"true positives {c['tp']}  ·  false positives {fps}  ·  "
           f"duplicate score ~ {c['dup_score_ratio']} x the true positive's",
)
y1 = 120
d.panel(
    y1,
    200,
    1,
    "What are the false positives?",
    f"{c['dup']} of {fps} are duplicates. Background false positives: {c['background']}.",
)
rows = [
    ("duplicate, same scale", c["dup_same"], f"{c['dup_same']}  ({c['dup_adjacent']} on the next cell)", PINK[0]),
    ("duplicate, other scale", c["dup_cross"], f"{c['dup_cross']}  (stride 8 and 16 both fire)", PINK[0]),
    ("mislocalized", c["misloc"], f"{c['misloc']}  ({c['misloc_on_missed']} on missed eels)", AMBER[0]),
    ("background", c["background"], f"{c['background']}", NEUTRAL[0]),
]
hbars(d, 40, y1 + 74, rows, 0, 200, width=420, label_w=220)
y2 = y1 + 220
d.panel(
    y2,
    250,
    2,
    "How does peak suppression remove them without NMS?",
    "A cell keeps its eel logit only if no neighbour on its own scale or the next scales beats it.",
)
# a 5 x 5 stride-16 grid with a 3 x 3 window
gx, gy, s = 90, y2 + 80, 30
vals = {(2, 2): "3.0", (2, 3): "2.0"}
for i in range(5):
    for j in range(5):
        inwin = 1 <= i <= 3 and 1 <= j <= 3
        fill = BLUE[1] if inwin else "#ffffff"
        d.rect(gx + j * s, gy + i * s, s, s, fill=fill, stroke=NEUTRAL[0], rx=0, sw=1)
        if (i, j) in vals:
            col = EMERALD[0] if (i, j) == (2, 2) else PINK[0]
            d.text(gx + j * s + s / 2, gy + i * s + 19, vals[(i, j)], 11.5, col, "bold", "middle", mono=True)
d.text(gx + 2.5 * s, gy - 10, "stride 16, 3 x 3 window", 12, MUTE, anchor="middle")
d.arrow(gx + 5 * s + 20, gy + 2.5 * s, gx + 5 * s + 80, gy + 2.5 * s, INK)
tx = gx + 5 * s + 100
lines = [
    "keep = logit >= max of its 3 x 3 window",
    "  and >= the 3 x 3 max of the coarser scale (upsampled)",
    "  and >= the 3 x 3 max of the finer scale (pooled 2 x 2)",
    "others: logit - 20  (score ~ 0)",
]
for k, s_ in enumerate(lines):
    d.text(tx, gy + 20 + 22 * k, s_, 12.5, INK, mono=True)
d.text(
    tx,
    gy + 120,
    f"duplicates {c['dup']} -> {a['dup']}   mislocalized {c['misloc']} -> {a['misloc']}   "
    f"true positives {c['tp']} -> {a['tp']}",
    12.5,
    EMERALD[0],
    "bold",
    mono=True,
)
y = y2 + 270
d.legend(30, y, [(PINK, "duplicate"), (AMBER, "mislocalized"), (EMERALD, "kept peak"), (BLUE, "window")])
end = d.footnotes(
    30,
    y + 40,
    [
        "A duplicate scores about 2/3 of its true positive, so a higher confidence threshold removes true eels first.",
        "The rule is CenterNet's local-maximum trick applied to the YOLO26 one-to-one head, plus the cross-scale "
        "check.",
    ],
)
d.save(height=end + 10)
