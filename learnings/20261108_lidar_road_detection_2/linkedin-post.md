# LinkedIn post: LiDAR Road Detection - Part 2: Labels for a top view

**Attach:** `fig1_projection_chain.png`, `fig2_sparse_dense.png`, `fig3_road_by_distance.png` as a carousel (in this order, as in the post).

---

LiDAR Road Detection, part 2: the KITTI road label is drawn on the camera image, but LoDNN needs it on a top-view grid. How do you move it?

Warping the image to a top view assumes a flat road, so cars and curbs get stretched onto the ground. LoDNN goes the other way: project every LiDAR point into the camera image, read the label under it, and put the point on the grid with that label.

On one KITTI scan, 18969 of 116044 points land inside the image, and 7501 of them on the road. On the grid that gives only 3319 road cells, in rings with gaps between them.

So each road point gets pseudo points on small circles around it, 144 of them, and a pseudo point is kept only when it also lands on the road label. The road cells grow from 3319 to 28401. Far away, where the rings are far apart, they grow 25.7 times.

The label can only be as good as the camera label, but it no longer looks like rings.

Notes, code and figure scripts are in my kaizen repo (link in comments). Part 3: seeing far with dilated convolutions.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_2
