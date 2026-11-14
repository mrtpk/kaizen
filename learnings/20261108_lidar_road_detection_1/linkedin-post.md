# LinkedIn post: LiDAR Road Detection - Part 1: From points to a top-view grid

**Attach:** `fig0_series_map.png`, `fig1_grid_geometry.png`, `fig3_six_channels.png`, `fig2_scatter_to_grid.png`, `fig4_sparsity.png` as a carousel (in this order, as in the post).

---

A LiDAR point cloud has no order that a convolution can use. So we drew it from above.

In 2018, I built the road detection module of an autonomous golf cart at Infosys. We used LoDNN, a method by Caltagirone et al. On the cart it ran at 83 FPS with a 94 % F1 score. This series rebuilds it step by step on one public KITTI scan.

Step one: a grid 40 m ahead and 20 m wide, in 0.1 m cells. Each cell gets six simple values from its points: count, mean reflectance, and four height statistics. Six images, stacked like colour channels.

One NumPy trap: grid[row, col] += 1 counts a cell only once, even when many points fall in it. On the sample scan it counted 7536 points. np.add.at counted all 15515.

Part 1 of LiDAR Road Detection.

#LiDAR #AutonomousVehicles #DeepLearning

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611lidar-road-detection-1-from-points-to-a-top-view-grid.html
