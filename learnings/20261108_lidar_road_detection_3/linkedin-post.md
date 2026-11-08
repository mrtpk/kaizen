# LinkedIn post: LiDAR Road Detection - Part 3: Seeing far with dilated convolutions

**Attach:** `fig1_network.png`, `fig2_receptive_field.png`, `fig3_weights.png` as a carousel (in this order, as in the post).

---

LiDAR Road Detection, part 3: the top-view grid is 400 cells tall and mostly empty far away. How does a network judge a far cell from what is around it?

Plain 3x3 layers grow the view by 2 cells per layer; a view 255 cells tall would need 127 of them. Pooling helps, but it throws away the resolution we need for a label per 0.1 m cell.

LoDNN pools once, then uses seven dilated 3x3 convolutions. From layer 2 on, the height dilation is twice the width, the same 1:2 shape as the grid; from layer 3 on, each layer doubles it. The receptive field grows to 129 x 255 cells on a 100 x 200 input. A short script checks the paper's table by sending one impulse through the layers.

One detail matters: a receptive field is centred on its output. So the field is larger than the grid, but only outputs near the middle (9.7 % of the cells) see all of it; an output at the far edge sees 64.8 % of the distance ahead.

By my reading of the paper's figure, that is 947458 weights, 93.5 % of them in six layers of the context module.

Notes, code and figure scripts are in my kaizen repo (link in comments). Part 4: from 64 beams to 16.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_3
