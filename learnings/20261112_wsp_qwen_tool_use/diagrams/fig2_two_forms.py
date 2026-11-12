"""Figure 2: the two edit forms, a rank-1 change (v1 E4b) and two programmed neurons (v2 v7).

Values: qwen_tool_edits.py, part A, section 3 (sizes), and part B (E4b changed values; v7 neuron slots).
"""

from _wsp import SERIES_NAME, post
from svglib import BLUE, BORDER, EMERALD, INK, VIOLET, WHITE, Diagram

p = post()
dat = p.load()
hd, mlp = p.LOGGED["hidden"], p.LOGGED["mlp"]
n1, n2 = p.LOGGED["neuron_slots"]
e4 = dat["edits"]["e4b"]["model.language_model.layers.23.mlp.down_proj.weight"]

d = Diagram(width=1100)
d.header(
    "Two ways to write one decision into the last MLP",
    f"{SERIES_NAME}  ·  Qwen3.5-0.8B, layer 23 MLP (hidden {hd}, width {mlp}, SwiGLU)",
    params=f"v1 E4b: {hd * mlp:,} values   ·   v2 v7: {6 * hd:,} values   ·   ratio {hd * mlp / (6 * hd):.0f}",
)

# panel 1: rank-1
y1, h1 = 120, 230
d.panel(
    y1,
    h1,
    1,
    "v1 E4b: what does a rank-1 edit change?",
    "Every value of the down projection. It acts at every position, with a strength set by the read.",
)
mx, my, mw, mh = 60, y1 + 75, 420, 120
d.rect(mx, my, mw, mh, fill=BLUE[1], stroke=BLUE[0], rx=4, sw=1.5)
d.text(mx + mw / 2, my + mh / 2 - 6, f"down_proj  W  ({hd} x {mlp})", 13, INK, "bold", "middle", mono=True)
d.text(mx + mw / 2, my + mh / 2 + 14, "W' = W - k u rho^T,  k = 5", 12.5, INK, anchor="middle", mono=True)
d.box(
    540,
    my,
    500,
    mh,
    "What changes",
    [
        f"all {hd * mlp:,} values (rank 1)",
        f"{e4['changed_values']:,} differ after bf16 rounding",
        f"relative norm of the change {e4['relative_norm']}",
        "fit: BFCL prompts and activations",
    ],
    col=BLUE[0],
    align="start",
)

# panel 2: two neurons
y2, h2 = y1 + h1 + 20, 270
d.panel(
    y2,
    h2,
    2,
    "v2 v7: what does a programmed neuron change?",
    "Three vectors per neuron. The gate keeps it silent everywhere except at decision positions.",
)
gx, gy = 60, y2 + 80
d.rect(gx, gy, 420, 150, fill=WHITE, stroke=BORDER, rx=4, sw=1)
for i, (lab, col) in enumerate((("gate_proj rows", VIOLET), ("up_proj rows", VIOLET), ("down_proj columns", EMERALD))):
    yy = gy + 18 + i * 44
    d.text(gx + 14, yy + 14, lab, 12.5, INK, mono=True)
    for j, nn in enumerate((n1, n2)):
        d.rect(gx + 200 + j * 110, yy, 90, 20, fill=col[1], stroke=col[0], rx=3, sw=1.4)
        d.text(gx + 245 + j * 110, yy + 14.5, f"{nn}", 12, INK, anchor="middle", mono=True)
d.box(
    540,
    gy,
    500,
    150,
    "What changes",
    [
        f"neuron {n1}: non-thinking decisions",
        f"neuron {n2}: thinking decisions",
        f"3 x {hd} = {3 * hd:,} values per neuron, {6 * hd:,} in total",
        "out = silu(g . x) (a . x) d,  a = r - theta q / c,  d = K u",
        "fit: practice data only (no BFCL rows)",
    ],
    col=VIOLET[0],
    align="start",
)
y = y2 + h2 + 20
d.legend(
    30, y, [(BLUE, "rank-1 change"), (VIOLET, "read side: gate and up rows"), (EMERALD, "write side: down column")]
)
end = d.footnotes(
    30,
    y + 40,
    [
        "u raises the <tool_call> logit and moves the common tokens least; rho reads the 3584 MLP activations,"
        " r reads the MLP input.",
        "Layer 23 is the last layer: no later layer reads the MLP output. Only the final norm sits before the output "
        "layer.",
    ],
)
d.save(height=end + 10)
