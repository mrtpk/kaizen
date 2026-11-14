# LinkedIn post: LiDAR Road Detection - Part 4: From 64 beams to 16, and back to points

**Attach:** `fig2_occupancy.png`, `fig1_rings_on_road.png`, `fig3_mask_to_points.png` as a carousel (in this order, as in the post).

---

KITTI uses a 64-beam LiDAR. Our golf cart had a 16-beam one. What does losing three quarters of the beams do?

I emulated it: keep every 4th ring of a KITTI scan. The occupied cells drop from 9.4 % to 2.6 %. On the road ahead, the largest gap between two rings grows from 5.8 m to 11.2 m.

The method did not change. Same grid, same six input images. Empty cells are simply 0.

The last step goes back to points: each point takes the road value of its cell. With a perfect road map, this agrees with the camera label for 95.4 % of the points.

Part 4 of LiDAR Road Detection.

#LiDAR #AutonomousVehicles #DeepLearning

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611lidar-road-detection-4-from-64-beams-to-16-and-back-to-points.html
