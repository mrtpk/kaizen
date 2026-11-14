"""Figure 2: weight averages on fold 0, independent runs vs children of one shared trunk.

Values: eel_agent_training.py (merge_f0 and trunk_f0 in study_results.json).
"""

from _eel_side import SERIES_NAME, hbars, post
from svglib import EMERALD, MUTE, NEUTRAL, PINK, Diagram

p = post()
dat = p.load()
m, t = dat["merge_f0"], dat["trunk_f0"]
LO, HI = 0.28, 0.50
ref = p.mm(m, "A_f0")


def bars(d: Diagram, y: float, rows: list[tuple[str, float, tuple]]) -> float:
    end = hbars(d, 30, y, [(lab, v, f"{v:.3f}", col[0]) for lab, v, col in rows], LO, HI, width=560, value_x=960)
    rx = 30 + 300 + 560 * (ref - LO) / (HI - LO)
    d.line(rx, y - 8, rx, end, MUTE, 1.2, dash="4 3")
    d.text(rx - 6, end + 14, f"best single run A_f0 {ref:.3f}", 12, MUTE, anchor="end", mono=True)
    return end


d = Diagram(width=1100)
d.header(
    "Weight averages need a shared trunk, and even then they do not add accuracy",
    f"{SERIES_NAME}  ·  side note  ·  fold 0 held-out mask mAP50-95, one pair of runs, one trunk",
    params="axis starts at 0.28  ·  BN re-est.: batch-norm statistics measured again on 512 training frames",
)
y1, h1 = 120, 250
d.panel(y1, h1, 1, "Independent runs from COCO: can I average them?",
        "No. The midpoint loses 0.10; a third run makes it worse. BN re-estimation repairs only part.")
bars(d, y1 + 74, [
    ("A_f0", p.mm(m, "A_f0"), NEUTRAL),
    ("B_f0", p.mm(m, "B_f0"), NEUTRAL),
    ("½ A + ½ B", p.mm(m, "interp_A_f0_B_f0_0.5"), PINK),
    ("½ A + ½ B, BN re-est.", p.mm(m, "interp_A_f0_B_f0_0.5_recal"), PINK),
    ("soup of A, B, C", p.mm(m, "soup"), PINK),
    ("soup of A, B, C, BN re-est.", p.mm(m, "soup_recal"), PINK),
])
y2, h2 = y1 + h1 + 20, 250
d.panel(y2, h2, 2, "Children of one 50-epoch trunk: can I average them?",
        "Blends lose nothing; the soup stays below the best single run.")
bars(d, y2 + 74, [
    ("child A1 (last)", p.mm(t, "child_A1:last"), NEUTRAL),
    ("child A2 (last)", p.mm(t, "child_A2:last"), NEUTRAL),
    ("child B1 (last)", p.mm(t, "child_B1:last"), NEUTRAL),
    ("½ A2 + ½ B1", p.mm(t, "interp0.5:child_A2+child_B1"), EMERALD),
    ("soup of 3 children", p.mm(t, "soup3:last"), EMERALD),
    ("soup of 3, BN re-est.", p.mm(t, "soup3:last+recal"), EMERALD),
])
y = y2 + h2 + 20
d.legend(30, y, [(NEUTRAL, "single run"), (PINK, "average, no shared trunk"), (EMERALD, "average, shared trunk")])
end = d.footnotes(30, y + 40, [
    "Seed noise on fold 0 is 0.001 to 0.005. The trunk runs used momentum 0.937, the CV runs 0.9.",
    "CPU re-evaluation of fold 0: it differs from the CV table by up to 0.002.",
])
d.save(height=end + 10)
