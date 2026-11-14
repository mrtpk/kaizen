"""Figure 6: the held-out ladder from the deployed model to eel_v2.

Values: eel_weight_space.py, part B: held-out half, the v2 ladder.
"""

from _eel import SERIES_NAME, data, hbars
from svglib import BLUE, EMERALD, NEUTRAL, Diagram

d0 = data()
r = d0["runs"]
steps = [
    ("deployed, as shipped", "late_base_o2o_e2e"),
    ("+ NMS 0.5", "late_base_o2o_post50"),
    ("peak suppression instead", "late_base_peak2"),
    ("+ class mean of both heads + NMS 0.5", "late_base_mixmean_peak2_nms50"),
    ("+ WiSE-FT alpha 0.25 = eel_v2", "late_eelv2_peak2_post50"),
]
rows = []
for i, (lab, t) in enumerate(steps):
    v = r[t]
    col = NEUTRAL[0] if i == 0 else (EMERALD[0] if i == len(steps) - 1 else BLUE[0])
    rows.append(
        (
            lab,
            v["fitness"],
            f"{v['fitness']:.4f}   R {v['box']['R']:.3f}   box {v['box']['mAP']:.3f}   mask {v['mask']['mAP']:.3f}",
            col,
        )
    )
dep, v2 = r[steps[0][1]], r[steps[-1][1]]
d = Diagram(width=1100)
d.header(
    "Each step adds a little, and all of it fits in one checkpoint",
    f"{SERIES_NAME}  ·  held-out half, {d0['logged']['heldout_frames']} frames",
    params=f"recall {dep['box']['R']:.3f} -> {v2['box']['R']:.3f}  ·  "
    f"box mAP50-95 {dep['box']['mAP']:.3f} -> {v2['box']['mAP']:.3f}  ·  "
    f"mask mAP50-95 {dep['mask']['mAP']:.3f} -> {v2['mask']['mAP']:.3f}",
)
y1 = 120
h1 = 80 + 26 * len(rows)
d.panel(
    y1,
    h1,
    1,
    "How much does each step add?",
    "Duplicate removal gives most of it. The hidden head and interpolation add the rest.",
)
hbars(d, 40, y1 + 70, rows, 1.32, 1.41, width=300, label_w=290)
y = y1 + h1 + 20
d.legend(30, y, [(NEUTRAL, "deployed"), (BLUE, "decoding and merge steps"), (EMERALD, "eel_v2.pt")])
end = d.footnotes(
    30,
    y + 40,
    [
        "Bars start at 1.32. R = box recall; box and mask = mAP50-95. eel_v2.pt loads in stock Ultralytics;",
        "peak suppression is a 20-line patch of the head's forward pass.",
    ],
)
d.save(height=end + 10)
