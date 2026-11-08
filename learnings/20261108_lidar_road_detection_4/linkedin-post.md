# LinkedIn post: LiDAR Road Detection - Part 4: From 64 beams to 16, and back to points

**Attach:** `fig2_occupancy.png`, `fig1_rings_on_road.png`, `fig3_mask_to_points.png` as a carousel (in this order, as in the post).

---

LiDAR Road Detection, part 4: KITTI uses a 64-beam LiDAR. On our golf cart at Infosys, we had a 16-beam VLP-16. What does a quarter of the beams do to the grid?

I don't have the cart data, so I emulated it: split a KITTI scan into its 64 rings and keep every 4th one. The occupied cells drop from 9.4 % to 2.6 %. On the road straight ahead, 9 rings cross the 40 m grid instead of 37, and the largest gap between two rings grows from 5.8 m to 11.2 m.

The six input images do not change with the sensor; empty cells are simply 0. On the cart, we kept the same grid and the same six images.

The last step is back to points: each point takes the road value of its cell. With a perfect road map, the lookup agrees with the camera label for 95.4 % of the points, and all 152 disagreements go one way: the grid label is a little wider than the road.

On the cart, the module reached a 94 % F1 score and 95 % average accuracy at 83 FPS.

Notes, code and figure scripts are in my kaizen repo (link in comments). Part 5: what held up.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_4
