"""Figure 5: fine-tuning on the first half, then interpolating back toward the base weights.

Values: eel_weight_space.py, part B: held-out half, peak suppression.
"""

from _eel import SERIES_NAME, data, hbars
from svglib import BLUE, EMERALD, PINK, Diagram

r = data()["runs"]
alphas = [
    ("0 (base)", "late_base_peak2"),
    ("0.15", "late_wise_a0.15_peak2"),
    ("0.25", "late_wise_a0.25_peak2"),
    ("0.35", "late_wise_a0.35_peak2"),
    ("0.5", "late_wise_a0.5_peak2"),
    ("0.75", "late_wise_a0.75_peak2"),
    ("1 (fine-tuned)", "late_ft6_peak2"),
]
best = max(alphas, key=lambda t: r[t[1]]["fitness"])[1]
base = r["late_base_peak2"]["fitness"]


def bar(label, tag):
    v = r[tag]
    col = EMERALD[0] if tag == best else (BLUE[0] if v["fitness"] >= base else PINK[0])
    return (
        label,
        v["fitness"],
        f"{v['fitness']:.4f}   R {v['box']['R']:.3f}   box mAP50-95 {v['box']['mAP']:.3f}",
        col,
    )


d = Diagram(width=1100)
d.header(
    "A quarter of the way to the fine-tune is the best point",
    f"{SERIES_NAME}  ·  theta = theta_base + alpha (theta_ft - theta_base), held-out half",
    params="fine-tune: 6 epochs on the first half only  ·  every row with peak suppression",
)
y1 = 120
h1 = 80 + 26 * len(alphas)
d.panel(
    y1,
    h1,
    1,
    "Which alpha scores best on the second half?",
    "alpha 0.15-0.35. The raw fine-tune raises recall but loses box accuracy.",
)
hbars(d, 40, y1 + 70, [bar(f"alpha {lab}", t) for lab, t in alphas], 1.34, 1.40, width=330, label_w=170)
parts = [
    ("backbone only", "late_wise_bb_a1_peak2"),
    ("box branches only", "late_wise_box_a1_peak2"),
    ("class branches only", "late_wise_cls_a1_peak2"),
    ("head-only fine-tune", "late_fthead6_peak2"),
]
y2 = y1 + h1 + 20
h2 = 80 + 26 * len(parts)
d.panel(
    y2,
    h2,
    2,
    "Which part of the fine-tune does the damage?",
    "Mostly the backbone. But a head-only fine-tune loses box accuracy too.",
)
hbars(d, 40, y2 + 70, [bar(lab, t) for lab, t in parts], 1.34, 1.40, width=330, label_w=170)
y = y2 + h2 + 20
d.legend(30, y, [(EMERALD, "best"), (BLUE, "at or above the base"), (PINK, "below the base")])
end = d.footnotes(
    30,
    y + 40,
    [
        "Bars start at 1.34. Fitness = box and mask, 0.1 x mAP50 + 0.9 x mAP50-95. alpha was chosen on the same half "
        "it is scored on."
    ],
)
d.save(height=end + 10)
