# LinkedIn post: LiDAR Road Detection - Part 2: Labels for a top view

**Attach:** `fig1_projection_chain.png`, `fig2_sparse_dense.png`, `fig3_road_by_distance.png` as a carousel (in this order, as in the post).

---

The road label in KITTI is drawn on the camera image. Our network needs it on a top-view grid. How do you move it?

Warping the image to a top view assumes a flat road, so cars and curbs get stretched onto the ground. LoDNN goes the other way: project each LiDAR point into the camera image, read the label under it, and put the point back on the grid.

That gives only 3319 road cells, in rings with gaps between them.

The fix: around each road point, add small pseudo points, and keep one only if it also lands on the road label. The road cells grow to 28401, and the label no longer looks like rings.

Part 2 of LiDAR Road Detection.

#LiDAR #AutonomousVehicles #DeepLearning

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611lidar-road-detection-2-labels-for-a-top-view.html
