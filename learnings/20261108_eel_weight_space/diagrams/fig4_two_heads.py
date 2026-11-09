"""Figure 4: two heads in one checkpoint. Averaging weights breaks the model; a widened head averages outputs exactly.

Values: eel_weight_space.py, part A check 3 (synthetic) and part B (head blend, merge error, subset fitness).
"""

from _eel import SERIES_NAME, data, post
from figkit import row
from svglib import BLUE, EMERALD, INK, PINK, VIOLET, Diagram

d0 = data()
lg, r = d0["logged"], d0["runs"]
hb = lg["head_blend"]
err_w, err_b, _ = post().merge_check()

d = Diagram(width=1100)
d.header(
    "Average the heads' outputs, not their weights",
    f"{SERIES_NAME}  ·  YOLO26 keeps a one-to-one head (deployed) and a one-to-many head (discarded)",
    params=f"subset fitness: one-to-one {r['sub_o2o_e2e']['fitness']:.4f}, "
           f"one-to-many + NMS {r['sub_o2m_nms']['fitness']:.4f}, "
    f"class mean + NMS 0.5 {r['sub_mixmean_nms50']['fitness']:.4f}",
)
y1 = 120
d.panel(
    y1,
    175,
    1,
    "Can I average the two heads' weights?",
    f"No. Box mAP50 falls to {hb['t0.5_mAP50']} at t = 0.5. The heads share no basis.",
)
row(
    d,
    y1 + 72,
    [
        ("one-to-one head", ["theta_1"], BLUE),
        (
            "(1 - t) theta_1 + t theta_2",
            [f"t 0.25: mAP50 {hb['t0.25_mAP50']}", f"t 0.5: mAP50 {hb['t0.5_mAP50']}"],
            PINK,
        ),
        ("one-to-many head", ["theta_2"], VIOLET),
    ],
    bh=80,
)
y2 = y1 + 195
d.panel(
    y2,
    175,
    2,
    "Then how do I put an ensemble into the weights?",
    "Stack the hidden layers side by side, and give the last layer half of each head's weights.",
)
row(
    d,
    y2 + 72,
    [
        ("first layers, stacked", ["W1 = [W1_a ; W1_b]", "middle 1x1: block-diagonal"], VIOLET),
        ("last 1x1 conv", ["W2 = [0.5 W2_a | 0.5 W2_b]", "b2 = 0.5 (b2_a + b2_b)"], VIOLET),
        ("output", ["= 0.5 head_a + 0.5 head_b", f"max error {lg['merge_max_err']:.0e} (fp16)"], EMERALD),
    ],
    arrow_col=INK,
    bh=80,
)
y = y2 + 195
d.legend(
    30, y, [(BLUE, "deployed head"), (VIOLET, "discarded head / merged parts"), (PINK, "broken"), (EMERALD, "exact")]
)
end = d.footnotes(
    30,
    y + 40,
    [
        f"Synthetic check in the script: widened head error {err_w:.1e}, weight average error {err_b:.2f}.",
        "Real head: BatchNorm folded into each conv, depthwise first conv interleaved. Class branch only; 3.18M "
        "parameters.",
    ],
)
d.save(height=end + 10)
