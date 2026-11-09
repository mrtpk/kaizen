"""Figure 2: editing the stride-8 eel row, and the best any linear row could do.

Values: eel_weight_space.py, part B: stride-8 edits and probe ceiling (first half fit, second half scored).
"""

from _eel import SERIES_NAME, data
from figkit import table
from svglib import AMBER, INK, MUTE, PINK, Diagram

lg = data()["logged"]
per = [fp / n for _, n, fp in lg["stride8_edits"]]
ratio = [held / own for _, _, held, own in lg["probe"]]
d = Diagram(width=1100)
d.header(
    f"Every short eel won by an edit costs {min(per):.0f} to {max(per):.0f} false positives",
    f"{SERIES_NAME}  ·  edits of the stride-8 eel row, fitted on the first half, scored on the second",
    params=f"{lg['stride8_targets']} short eels missed in the second half",
)
y1 = 120
rows = [["Edit", "eels won", "extra false positives", "per eel"]]
for name, n, fp in lg["stride8_edits"]:
    rows.append([name, f"+{n}", f"+{fp}", f"{fp / n:.0f}"])
h1 = 100 + 26 * len(rows)
d.panel(y1, h1, 1, "What do bias and LDA edits buy?", "A few eels, at a high price in false positives.")
table(d, y1 + 90, rows, [70, 420, 620, 870], color_fn=lambda i, j, v: PINK[0] if j == 3 else INK)
y2 = y1 + h1 + 20
rows = [["Short-eel recall", "current row", "best row, other half", "best row, own half"]]
for rc, cur, held, own in lg["probe"]:
    rows.append([f"{rc} %", f"{cur} FP", f"{held} FP", f"{own} FP"])
h2 = 100 + 26 * len(rows)
d.panel(
    y2,
    h2,
    2,
    "Could any linear row do better?",
    f"Not much. On the half it was fitted on, the probe looks "
    f"{min(ratio):.1f}-{max(ratio):.1f}x better: the leak to avoid.",
)
table(d, y2 + 90, rows, [70, 380, 600, 850], color_fn=lambda i, j, v: AMBER[0] if j == 3 else (INK if j else MUTE))
y = y2 + h2 + 20
d.legend(30, y, [(PINK, "false positives per eel won"), (AMBER, "in-sample score (optimistic)")])
end = d.footnotes(
    30,
    y + 40,
    [
        "The probe is a class-weighted logistic regression on the 64 stride-8 features: a measurement, not an edit to "
        "keep.",
        "LDA edit: a rank-1 change along the direction that separates missed eels from background.",
    ],
)
d.save(height=end + 10)
