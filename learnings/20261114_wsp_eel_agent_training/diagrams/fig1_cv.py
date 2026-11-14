"""Figure 1: recipe A vs recipe B on the 5 folds, held-out mask mAP50-95 and mask mAP50.

Values: eel_agent_training.py (cv in study_results.json).
"""

from statistics import mean

from _eel import SERIES_NAME, post
from svglib import AMBER, BLUE, INK, LINE, MUTE, Diagram

p = post()
cv = p.load()["cv"]
X0, WD = 200, 700


def strip(d: Diagram, y: float, metric: str, lo: float, hi: float, ticks: list[float]) -> None:
    for f in range(5):
        yy = y + f * 30
        d.text(X0 - 14, yy + 5, f"fold {f}", 12.5, INK, anchor="end")
        d.line(X0, yy, X0 + WD, yy, LINE, 1)
        for r, col in (("A", BLUE), ("B", AMBER)):
            v = cv[r][metric][f]
            d.circle(X0 + WD * (v - lo) / (hi - lo), yy, 7, fill=col[0])
    yy = y + 5 * 30
    d.text(X0 - 14, yy + 5, "mean", 12.5, INK, "bold", anchor="end")
    d.line(X0, yy, X0 + WD, yy, LINE, 1)
    for r, col in (("A", BLUE), ("B", AMBER)):
        d.circle(X0 + WD * (mean(cv[r][metric]) - lo) / (hi - lo), yy, 7, fill=col[0])
    d.text(X0 + WD + 16, yy + 5,
           f"A {mean(cv['A'][metric]):.3f}  B {mean(cv['B'][metric]):.3f}", 12.5, INK, mono=True)
    for t in ticks:
        tx = X0 + WD * (t - lo) / (hi - lo)
        d.text(tx, yy + 28, f"{t:.2f}", 12, MUTE, anchor="middle", mono=True)


d = Diagram(width=1100)
d.header(
    "Copy-paste wins on reliability, not on the main score",
    f"{SERIES_NAME}  ·  side note  ·  YOLO26n-seg, 5 folds, about 1,000 held-out frames per fold",
    params="A: batch 64 from COCO  ·  B: A + copy_paste 0.5  ·  one run per fold and recipe",
)
y1, h1 = 120, 290
d.panel(y1, h1, 1, "Mask mAP50-95: which recipe is better?", "Neither. B leads on 2 of 5 folds; the gap is inside the fold spread.")
strip(d, y1 + 80, "mask50-95", 0.46, 0.53, [0.46, 0.48, 0.50, 0.52])
y2, h2 = y1 + h1 + 20, 290
d.panel(y2, h2, 2, "Mask mAP50: which recipe is better?", "B, on 4 of 5 folds, with half the spread. On fold 3, A drops to 0.920.")
strip(d, y2 + 80, "mask50", 0.91, 0.975, [0.92, 0.94, 0.96])
y = y2 + h2 + 20
d.legend(30, y, [(BLUE, "recipe A"), (AMBER, "recipe B (shipped as the candidate)")])
end = d.footnotes(30, y + 40, ["Each dot: the best checkpoint of one run, scored on its fold's held-out frames. Masks at 1/4 resolution."])
d.save(height=end + 10)
