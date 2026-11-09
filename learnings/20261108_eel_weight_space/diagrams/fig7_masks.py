"""Figure 7: why mask boundaries are off, and what the refiner changes.

Values: eel_weight_space.py, part A check 4 (grid cell, band IoU) and part B (mask boundaries, 2,319 held-out eels).
"""

from _eel import SERIES_NAME, data, post
from figkit import table
from svglib import BLUE, EMERALD, INK, MUTE, NEUTRAL, PINK, Diagram

d0 = data()
lg = d0["logged"]
bd, rf = lg["boundary"], lg["refined"]
m = post()
cell = 640 / 512 * 4
iou1, iou2 = m.band_iou(9, 1), m.band_iou(9, 2)

d = Diagram(width=1100)
d.header(
    "An eel is less than two mask cells wide",
    f"{SERIES_NAME}  ·  mask boundaries at full frame resolution, {bd['eels']} held-out eels",
    params=f"mask grid cell = {cell:.1f} frame px  ·  "
           f"eel width ~ {bd['eel_width']} px = {bd['eel_width'] / cell:.1f} cells",
)
y1 = 120
d.panel(
    y1,
    215,
    1,
    "Why can't the mask place the edge to one pixel?",
    f"The mask is drawn on a grid at 1/4 of the input. One pixel of error on a 9 px eel costs {1 - iou1:.0%} IoU.",
)
px, gx, gy = 14, 90, y1 + 100
for k in range(30):
    inside = 10 <= k < 19
    d.rect(gx + k * px, gy, px, 24, fill=BLUE[1] if inside else "#ffffff", stroke=NEUTRAL[0], rx=0, sw=0.8)
for k in range(0, 31, 5):
    d.line(gx + k * px, gy - 14, gx + k * px, gy + 38, PINK[0], sw=1.6)
d.text(gx, gy - 22, "frame pixels (blue = eel, 9 px)  ·  red lines = mask grid cells, 5 px", 12, MUTE)
d.text(gx, gy + 62, f"shift 1 px: IoU {iou1:.2f}    shift 2 px: IoU {iou2:.2f}", 12.5, INK, mono=True)
tx = 600
for k, s in enumerate(
    [
        f"mean boundary error   {bd['absd']:.2f} px",
        f"signed error          +{bd['sd']:.2f} px (too fat)",
        f"area / label area     {bd['area']:.2f}",
        f"missed pixels outside box  {bd['fn_outside_box']:.0%}",
    ]
):
    k0, k1 = s.split("  ", 1)
    d.text(tx, gy - 6 + 22 * k, k0, 12.5, INK)
    d.text(tx + 230, gy - 6 + 22 * k, k1.strip(), 12.5, INK, mono=True)
y2 = y1 + 235
rows = [["Measure", "YOLO mask", "refined", "change"]]
for k, name, better_low in [
    ("bf1", "boundary F at 1 px", False),
    ("bf2", "boundary F at 2 px", False),
    ("absd", "mean boundary error, px", True),
    ("sd", "signed boundary error, px", True),
    ("area", "area / label area", None),
    ("iou", "mean IoU", False),
]:
    rows.append([name, f"{bd[k]:.3f}", f"{rf[k]:.3f}", f"{rf[k] - bd[k]:+.3f}"])
h2 = 100 + 26 * len(rows)
d.panel(
    y2,
    h2,
    2,
    "What does a small learned refiner change?",
    f"It fixes the fat bias and the edge pixels. It does not change IoU. {rf['params']:,} parameters.",
)
table(
    d,
    y2 + 90,
    rows,
    [70, 420, 620, 820],
    color_fn=lambda i, j, v: (MUTE if v == "+0.000" else EMERALD[0]) if j == 3 else INK,
)
y = y2 + h2 + 20
d.legend(30, y, [(BLUE, "eel pixels"), (PINK, "mask grid"), (EMERALD, "refiner gain")])
end = d.footnotes(
    30,
    y + 40,
    [
        "Refiner inputs: the infrared crop, the upsampled YOLO logits and the box. It predicts a correction to the "
        "logits.",
        f"Trained on {rf['fit_eels']} eels of the first half for {rf['epochs']} epochs; scored on the second half. "
        f"About {rf['ms_per_eel']} ms per eel on one CPU thread.",
    ],
)
d.save(height=end + 10)
