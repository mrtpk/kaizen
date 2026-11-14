"""Figure 0: the map of the study, and what worked.

Values: eel_weight_space.py, part B (study_results.json).
"""

from _eel import SERIES_NAME, data
from figkit import row
from svglib import AMBER, EMERALD, PINK, Diagram

d0 = data()
r, lg = d0["runs"], d0["logged"]
dep, v2 = r["late_base_o2o_e2e"], r["late_eelv2_peak2_post50"]
e = lg["stride8_edits"]
fp_per = f"{min(f / n for _, n, f in e):.0f}-{max(f / n for _, n, f in e):.0f} FP per eel won"
c = lg["census"]
bd, rf = lg["boundary"], lg["refined"]

d = Diagram(width=1100)
d.header(
    "Six things I tried on an eel detector's weights",
    f"{SERIES_NAME}  ·  one YOLO26n-seg model  ·  {lg['frames']} infrared frames, one video",
    params=f"held-out half: fitness {dep['fitness']:.4f} -> {v2['fitness']:.4f}  ·  "
    f"recall {dep['box']['R']:.3f} -> {v2['box']['R']:.3f}",
)
y1 = 120
d.panel(
    y1,
    170,
    1,
    "Part 1: can the weights explain and fix the model?",
    "They explain it well. They do not fix its misses.",
)
row(
    d,
    y1 + 70,
    [
        ("Read the biases", ["21 of 22 classes dead", "stride 32 dead"], EMERALD),
        ("Exact surgery", ["22 classes -> 1", "every metric identical"], EMERALD),
        ("Edit to fix recall", ["stride-8 eel row", fp_per], PINK),
    ],
    bh=80,
)
y2 = y1 + 190
d.panel(
    y2,
    170,
    2,
    "Part 2: false positives, the hidden head, interpolation, masks",
    "Three closed-form changes help. Mask boundaries are only partly fixed.",
)
row(
    d,
    y2 + 70,
    [
        ("Peak suppression", [f"duplicates {c['dup']} -> {lg['after_peak']['dup']}", "no NMS, no fit"], EMERALD),
        ("Merge heads + WiSE-FT", ["exact widened head", "alpha 0.25 interpolation"], EMERALD),
        (
            "Mask boundaries",
            [f"area {bd['area']:.2f}x -> {rf['area']:.2f}x", f"IoU {bd['iou']:.3f} -> {rf['iou']:.3f}"],
            AMBER,
        ),
    ],
    bh=80,
)
y = y2 + 190
d.legend(30, y, [(EMERALD, "worked"), (AMBER, "partly"), (PINK, "did not work")])
end = d.footnotes(
    30,
    y + 40,
    [
        "Held-out half = the second half of the video (frames 470754-529830). Fitness = sum over box and mask",
        "of 0.1 x mAP50 + 0.9 x mAP50-95. The original model saw every frame in training.",
    ],
)
d.save(height=end + 10)
