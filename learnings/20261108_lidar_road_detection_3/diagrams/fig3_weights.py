"""Figure 3: where the weights of the network are.

Values: context_module.py, section 4: 947458 weights and biases; the six 128 -> 128
context layers hold 885504 (93.5 %).
"""

from _lrd3 import hbars, script
from svglib import BLUE, EMERALD, VIOLET, Diagram

c = script("20261108_lidar_road_detection_3", "context_module")
layers = [
    ("encoder 6 -> 32", c.conv_params(3, 6, 32), BLUE[0]),
    ("encoder 32 -> 32", c.conv_params(3, 32, 32), BLUE[0]),
    ("context 32 -> 128", c.conv_params(3, 32, 128), VIOLET[0]),
    ("context 128 -> 128, x6", 6 * c.conv_params(3, 128, 128), VIOLET[0]),
    ("context 1x1 128 -> 32", c.conv_params(1, 128, 32), VIOLET[0]),
    ("decoder 32 -> 32", c.conv_params(3, 32, 32), EMERALD[0]),
    ("decoder 32 -> 2", c.conv_params(3, 32, 2), EMERALD[0]),
]
total = sum(n for _, n, _ in layers)
rows = [(name, n, f"{n}  ({100 * n / total:.1f} %)", col) for name, n, col in layers]

d = Diagram(width=1100)
d.header("Under a million weights, almost all in the context module",
         "LiDAR Road Detection  ·  part 3  ·  weights and biases per layer, as I read the paper's figure",
         params=f"total: {total} weights and biases")
y1 = 116
d.panel(y1, 270, 1, "Which layers hold the weights?",
        f"The six 128 -> 128 layers hold {100 * layers[3][1] / total:.1f} % of them.")
hbars(d, 60, y1 + 74, rows, vmax=900000, width=520, label_w=220)
y = y1 + 290
d.legend(30, y, [(BLUE, "encoder"), (VIOLET, "context module"), (EMERALD, "decoder")])
end = d.footnotes(30, y + 40, ["A 3x3 convolution from a to b maps has 9ab weights and b biases. The paper does not "
                               "give a total."])
d.save(height=end + 10)
