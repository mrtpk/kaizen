---
layout: post
title:  "LiDAR Road Detection - Part 5: What held up"
date:   2026-11-08 00:00:00 +0530
categories: lidar deep-learning
mathjax: true
---

Lately, I have been going through my notes from a LiDAR road detection project at Infosys, and this last note looks back at the series: what the scripts show, what held up, and what was lost.

In [part 4][prev], we cut the KITTI scan from 64 rings to 16 and took the road map back to the points. Over four parts, we rebuilt the steps of LoDNN, the method of Caltagirone et al. that we used on an autonomous golf cart in 2018: the top-view grid, the labels, the network and the sensor. This part runs all the checks again in one script.

## The scorecard
`lessons.py` imports the scripts of parts 1 to 4 and re-runs their main checks on the same KITTI scan:

1. **Top-view grid.** 400 x 200 cells of 0.1 m; 15515 of the 116044 points fall in it, and 9.4 % of the cells hold a point. The vectorised images equal a plain loop.
2. **Labels.** From the points alone, 3319 cells are road; with pseudo points, 28401. Every sparse road cell stays road.
3. **Network.** The context module has a receptive field of 129 x 255 cells on its 100 x 200 input, larger than the input in both directions.
4. **16 beams.** With every 4th ring, 2.6 % of the cells hold a point instead of 9.4 %, and only 0.7 % from 36 m to 46 m. Every cell of the 16-ring grid is also in the 64-ring grid.

All four checks print `True`.

![One panel with four rows. 1: top-view grid, 400 x 200 cells, 9.4 % occupied, vectorised grid equals the loop, True. 2: labels, road cells 3319 to 28401, sparse road stays road, True. 3: network, receptive field 129 x 255 on 100 x 200, sees past the input, True. 4: 16 beams, occupied 2.6 % instead of 9.4 %, 16-ring cells within the 64-ring cells, True.](https://raw.githubusercontent.com/mrtpk/kaizen/master/learnings/20261108_lidar_road_detection_5/resources/fig1_scorecard.png)
*Figure: one line per part, with the number it showed and its check.*

Let's look at the code. The two facts that the whole design rests on, the grid size and the reach of the context module, fit in a few lines:

{% highlight python %}
rows, cols = round((46 - 6) / 0.1), round((10 - (-10)) / 0.1)   # the top-view grid
w = h = 1
for dw, dh in [(1, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]:
    w, h = w + 2 * dw, h + 2 * dh                                # receptive field of the context module
print("Grid:", rows, "x", cols, " receptive field:", w, "x", h)  # 400 x 200, 129 x 255
print("Receptive field larger than the pooled grid. Is pass?", w > cols // 2 and h > rows // 2)  # True
{% endhighlight %}

## What held up
**The top view.** Six statistics per cell, on a fixed grid in front of the car. They are fast to compute with a few NumPy group-by calls, and they do not care how many beams the sensor has: with 16 beams, there are simply more empty cells. On the cart, we kept the same grid and the same six images.

**Labels from the camera.** The road label of KITTI is drawn on the camera image, and projecting the points into it gives a top-view label without assuming a flat road. Pseudo points on small circles fill the space between the rings, most of all far away, where the rings are far apart.

**One pooling and dilated convolutions.** Seven dilated layers give a receptive field larger than the grid, with under a million weights. On the cart, our module ran at 12 ms per frame, 83 FPS, and reached a 94 % F1 score and 95 % average accuracy with the VLP-16.

## What was lost
The data that we recorded on the cart, and our TensorFlow port of the network, belong to Infosys, and I do not have them. So I cannot re-run the results above, and no script in this series reproduces them. Every other number in the series comes from the scripts and one public KITTI scan.

## Refinements I found while writing
Two small things showed up in the scripts:

- An empty cell is 0 in every image, and a cell of points near $z = 0$ has a mean elevation near 0 too. In the sample scan, 5 occupied cells have a mean elevation within 0.05 m of 0. The number-of-points image tells them apart, but a value that no real cell can have would make empty cells explicit.
- The road label on the grid is a little wider than the road. When it labels the points of the 16-ring scan, all 152 disagreements with the camera label are points that the grid calls road and the camera does not.

So, in short: the checks of the series pass on one KITTI scan, and the input encoding does not depend on the number of beams; the labels and the network were checked here with 64 rings only. What is gone is the data from the cart; what is left is the method, and these notes.

The full script is available [here][script]. The series starts with [part 1][part1].

**Reference:**
+ [Fast LIDAR-based Road Detection Using Fully Convolutional Neural Networks (Caltagirone et al.)][lodnn]
+ [lidar-buster: Python snippets for LiDAR point clouds (the version used here)][lidar-buster]
+ [Infosys: Autonomous technologies][infosys]
+ [Infosys' Driverless Golf Cart (video)][yt]
+ [Why you should be interested in LiDAR technologies?][kaizen-tech]
+ [LiDAR Basics - The Coordinate System][kaizen-coords]

[part1]: https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_1
[prev]: https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_4
[script]: https://github.com/mrtpk/kaizen/tree/master/learnings/20261108_lidar_road_detection_5
[lodnn]: https://arxiv.org/abs/1703.03613v2
[lidar-buster]: https://github.com/mrtpk/lidar-buster/tree/1bb1d59d2d470af4c08350cbd912081728cd01eb
[infosys]: https://www.infosys.com/insights/ai-automation/Pages/autonomous-technologies.aspx
[yt]: https://www.youtube.com/watch?v=X6wEztubV_4
[kaizen-tech]: https://github.com/mrtpk/kaizen/tree/master/learnings/20180827_lidar_tech
[kaizen-coords]: https://github.com/mrtpk/kaizen/tree/master/learnings/20180929_lidar-coordinate-basics
