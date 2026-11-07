# LinkedIn post: Quantization Basics - the quantization formula

**Attach:** `fig1_move_and_stretch.png` … `fig5_round_trip.png` as a carousel (in order).

---

A few years ago, I wrote a note on Stack Overflow about the quantization formula. I went back to it this week, found a few gaps, and rewrote it.

The idea that makes it click: quantization is just moving a point from one number line (floats) to another (the 256 integers of int8).

Moving a point needs only two operations: translate (add) and scale (multiply). Percentage is two scales. Celsius to Fahrenheit is translate, scale, scale, translate. Quantization is the same four steps, folded into two numbers: scale and zero_point.

For a float range of [-1, 3]: scale = 4 / 255 = 0.015686 and zero_point = -64. So -1.0 → -128, 0.0 → -64, 1.0 → 0 and 3.0 → 127.

Two details that I refined this time: divide by 255 (the steps), not 256 (the levels), and make sure 0.0 is inside the range, so that it maps to an integer exactly.

The round trip to int8 and back costs at most half a step inside the range (0.0078 here). Outside the range, clipping costs anything: 3.5 comes back as 2.9961.

Every number is checked in NumPy. Notes, code and the figure scripts are in my kaizen repo (link in comments).

Next: Int8 on Silicon, a series on doing all of this with integers only, on real hardware.

---

**First comment:** https://github.com/mrtpk/kaizen/tree/master/learnings/20261107_quantization_formula
