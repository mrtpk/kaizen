# LinkedIn post: Quantization Basics - the quantization formula

**Attach:** `fig1_move_and_stretch.png` … `fig5_round_trip.png` as a carousel (in order).

---

Quantization is just moving a point from one number line to another.

Floats are one line. The 256 integers of int8 are another. To move a point between two lines, you need only two operations: add and multiply. Celsius to Fahrenheit is the same idea.

So the whole formula folds into two numbers: a scale and a zero point. For the float range [-1, 3], scale = 4/255 and zero_point = -64. Then -1.0 becomes -128, 0.0 becomes -64, and 3.0 becomes 127.

Two details people get wrong:
→ Divide by 255 (the steps), not 256 (the levels).
→ Keep 0.0 inside the range, so that zero maps to an exact integer.

I first wrote this as a Stack Overflow answer years ago. This is the cleaned-up version, with every number checked in NumPy.

Next: a series on doing all of this with integers only, on real hardware.

#Quantization #DeepLearning #EdgeAI

---

**First comment:** https://www.thomaspaul.dev/2026/10/202610quantization-formula.html
