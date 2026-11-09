"""Figure 8: the defects of this study and what I would do next time.

Values: none measured; the defects are those listed in the post's "What I would do next time" section.
"""

from _eel import SERIES_NAME
from svglib import AMBER, BLUE, INK, MUTE, PINK, Diagram

EVAL = [
    ("No footage outside the training video", "Score on another day, site or camera before any claim"),
    ("Settings chosen on the half I score", "Split by time into fit, tune and test; score test once"),
    ("No error bars", "Bootstrap over clips; repeat each fine-tune with 3 seeds"),
]
DATA = [
    ("No rule for partly visible eels", "Write one; relabel the eels at the opening and the frame edge"),
    ("Some labels sit off the eel's body", "Measure label offset on a sample before trusting 1-px metrics"),
]
MODEL = [
    ("Mask grid 1/4 of the input", "Train at 640-768 px, or use stride-2 prototypes"),
    ("Short boxes cut the masks", "Train masks uncropped, or grow the crop box"),
    ("Brittle to low contrast and noise", "Add both as augmentation; keep the perturbation test"),
    ("Deployed head leaves duplicates", "Ship peak suppression, or train with it in the loop"),
]

d = Diagram(width=1100)
d.header(
    "What I would do differently next time",
    f"{SERIES_NAME}  ·  the defects of this study, and the fix for each",
    params="left: what was wrong  ·  right: what to do",
)
y = 120
for num, (title, sub, rows, col) in enumerate(
    [
        (
            "How would I make the numbers trustworthy?",
            "Hold out data the model never saw, and measure the noise.",
            EVAL,
            PINK,
        ),
        ("What would I fix in the labels?", "Decide what counts as an eel before measuring misses.", DATA, AMBER),
        ("What would I change in the model?", "The mask grid and the training recipe, not the weights.", MODEL, BLUE),
    ],
    start=1,
):
    h = 80 + 26 * len(rows)
    d.panel(y, h, num, title, sub)
    for k, (defect, fix) in enumerate(rows):
        ry = y + 84 + 26 * k
        d.rect(60, ry - 13, 8, 16, fill=col[0], stroke=None, rx=2)
        d.text(80, ry, defect, 12.5, INK)
        d.text(470, ry, "->", 12.5, MUTE, mono=True)
        d.text(510, ry, fix, 12.5, INK)
    y += h + 20
d.legend(30, y, [(PINK, "evaluation"), (AMBER, "labels"), (BLUE, "model")])
end = d.footnotes(
    30,
    y + 40,
    ["A clip = about 13 consecutive frames; neighbouring frames are near-duplicates, so bootstrap whole clips."],
)
d.save(height=end + 10)
