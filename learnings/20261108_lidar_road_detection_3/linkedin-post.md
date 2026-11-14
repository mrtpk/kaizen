# LinkedIn post: LiDAR Road Detection - Part 3: Seeing far with dilated convolutions

**Attach:** `fig1_network.png`, `fig2_receptive_field.png`, `fig3_weights.png` as a carousel (in this order, as in the post).

---

To label a cell 40 m away, a network must see a lot of the area around it. Plain 3×3 layers would need 127 layers for that.

LoDNN pools once, then uses seven dilated convolutions. Each layer skips more cells than the one before, so the view doubles per layer instead of growing by 2 cells. The receptive field becomes larger than the whole grid, with under a million weights.

One detail people miss: the receptive field is centred on its output. So only outputs near the middle of the grid see all of it. An output at the far edge sees only 64.8 % of the distance ahead.

Part 3 of LiDAR Road Detection.

#LiDAR #AutonomousVehicles #DeepLearning

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611lidar-road-detection-3-seeing-far-with-dilated-convolutions.html
