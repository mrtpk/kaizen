# LinkedIn post: LiDAR Road Detection - Part 5: What held up

**Attach:** `fig1_scorecard.png`.

---

LiDAR Road Detection, part 5, the last one: one script re-runs the checks of the series on the same KITTI scan, and all four pass.

What held up: the top view. Six statistics per cell on a fixed 400 x 200 grid, fast with a few NumPy group-by calls, and they do not care how many beams the sensor has. Labels from the camera, made dense with pseudo points (3319 road cells to 28401). And one pooling plus dilated convolutions: a receptive field larger than the grid, with under a million weights.

What was lost: the data from the cart and our TensorFlow port stayed with Infosys. So I cannot re-run the results from 2018 (94 % F1, 95 % average accuracy, 83 FPS); every other number in the series comes from a script and one public scan.

Two refinements showed up while writing: empty cells and cells near zero height look alike in the elevation images, and the grid road label is a little wider than the road.

The method is LoDNN by Caltagirone et al.; the series is my notes on building and running it.

Notes, code and figure scripts are in my kaizen repo (link in comments).

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_5
