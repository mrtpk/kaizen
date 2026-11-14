# LinkedIn post: LiDAR Road Detection - Part 5: What held up

**Attach:** `fig1_scorecard.png`.

---

Eight years later, I rebuilt our LiDAR road detector from public data. Here is what held up.

→ The top view: six statistics per cell on a fixed grid. Fast, and it does not care how many beams the sensor has.
→ Camera labels made dense with pseudo points.
→ One pooling step plus dilated convolutions: a view larger than the grid, with under a million weights.

What was lost: the cart data and our code stayed with Infosys. So I cannot re-run the 2018 results (94 % F1, 83 FPS). Every other number in the series comes from a script.

The method is LoDNN by Caltagirone et al. The series is my notes on building and running it.

Last part of LiDAR Road Detection.

#LiDAR #AutonomousVehicles #DeepLearning

---

**First comment:** https://www.thomaspaul.dev/2026/11/202611lidar-road-detection-5-what-held-up.html
