# LinkedIn post: LiDAR Road Detection - Part 1: From points to a top-view grid

**Attach:** `fig0_series_map.png`, `fig1_grid_geometry.png`, `fig3_six_channels.png`, `fig2_scatter_to_grid.png`, `fig4_sparsity.png` as a carousel (in this order, as in the post).

---

In 2018, I worked on the road detection module of an autonomous golf cart at Infosys. We used LoDNN, a LiDAR-only method by Caltagirone et al., and ran it on a 16-beam Velodyne VLP-16. On the cart it reached a 94 % F1 score and 95 % average accuracy, at 83 FPS (12 ms per frame). I'm writing it down: "LiDAR Road Detection", part 1.

The data from the cart and our TensorFlow port belong to Infosys, so the series rebuilds each step with new scripts on one public KITTI scan.

Step one: a point cloud has no order a convolution can use. So LoDNN draws it from above, on a grid 6 m to 46 m ahead and 10 m to each side, in cells of 0.1 m. That is 400 x 200 cells.

Each cell gets six simple statistics of its points: the count, the mean reflectance, and the max, min, mean and standard deviation of the height. Six images, stacked like colour channels.

The trap: grid[row, col] += 1 does not count when two points share a cell. On the sample scan it counts 7536 points; np.add.at counts all 15515.

And the grid is sparse: 9.4 % of the cells hold a point, and only 1.8 % between 36 m and 46 m.

Notes, code and figure scripts are in my kaizen repo (link in comments). Part 2: labels for a top view.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_1
