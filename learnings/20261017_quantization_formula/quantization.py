"""Quantization basics: from number lines to the int8 quantization formula.

Every number in the post comes from this script.
Run:  uv run --with numpy python quantization.py
"""

import numpy as np

np.random.seed(0)


# ---- moving and stretching a point --------------------------------------
def translate(value, offset):
    return value + offset


def scale(value, factor):
    return value * factor


print("translate(4, 5) =", translate(4, 5))     # 9
print("translate(4, -5) =", translate(4, -5))   # -1
print("scale(4, 3) =", scale(4, 3))             # 12
print("scale(4, 0.5) =", scale(4, 0.5))         # 2.0, halve


# ---- percentage is two stretches -----------------------------------------
def percentage(value, max_value):
    normed = scale(value, 1 / max_value)   # 0 .. max_value  ->  0 .. 1
    return scale(normed, 100)              # 0 .. 1          ->  0 .. 100


print("percentage(25, 500) =", percentage(25, 500))  # 5.0


# ---- map a point from one number line to another -------------------------
def map_point(value, src_min, src_max, dst_min, dst_max):
    src_length = src_max - src_min
    dst_length = dst_max - dst_min
    value = translate(value, -src_min)        # move the source start to 0
    value = scale(value, 1 / src_length)      # 0 .. src_length -> 0 .. 1
    value = scale(value, dst_length)          # 0 .. 1          -> 0 .. dst_length
    value = translate(value, dst_min)         # move 0 to the destination start
    return value


# 25 on 0..500 is 5 on 0..100 (the percentage again)
print("map_point(25, 0, 500, 0, 100) =", map_point(25, 0, 500, 0, 100))
# Celsius to Fahrenheit is a map between two number lines: 0..100 C -> 32..212 F
print("map_point(37, 0, 100, 32, 212) =", map_point(37, 0, 100, 32, 212))  # 98.6
# the ends map to the ends
print("map_point ends. Is pass?",
      np.isclose(map_point(-1.0, -1.0, 3.0, -128, 127), -128)
      and np.isclose(map_point(3.0, -1.0, 3.0, -128, 127), 127))


# ---- the int8 quantization formula ---------------------------------------
Q_MIN, Q_MAX = -128, 127


def quantization_params(r_min, r_max):
    r_min, r_max = min(r_min, 0.0), max(r_max, 0.0)   # 0.0 must be on the line
    if r_max == r_min:                                 # constant tensor (all zeros)
        return 1.0, 0
    s = (r_max - r_min) / (Q_MAX - Q_MIN)              # 255 steps between 256 levels
    z = int(np.clip(np.round(Q_MIN - r_min / s), Q_MIN, Q_MAX))
    return s, z


def quantize(r, s, z):
    q = np.round(r / s) + z
    return np.clip(q, Q_MIN, Q_MAX).astype(np.int8)


def dequantize(q, s, z):
    return (q.astype(np.int32) - z) * s       # widen first: int8 - z can overflow int8


r_min, r_max = -1.0, 3.0
s, z = quantization_params(r_min, r_max)
print(f"range [{r_min}, {r_max}] -> scale = {s:.6f}, zero_point = {z}")   # 0.015686, -64

for r in [-1.0, 0.0, 1.0, 3.0, 3.5]:
    q = quantize(np.array([r]), s, z)[0]
    back = dequantize(np.array([q], dtype=np.int8), s, z)[0]
    print(f"  r = {r:5.2f} -> q = {int(q):4d} -> back = {back:.4f}  (error {back - r:+.4f})")

print("0.0 maps to the zero point. Is pass?", quantize(np.array([0.0]), s, z)[0] == z)

# round trip on random values inside the range: error is at most scale / 2
x = np.random.uniform(r_min, r_max, 100_000)
err = np.abs(dequantize(quantize(x, s, z), s, z) - x)
print(f"max round-trip error = {err.max():.6f}, scale / 2 = {s / 2:.6f}")
print("Round-trip error <= scale / 2. Is pass?", err.max() <= s / 2 + 1e-9)


# ---- symmetric quantization: zero_point = 0 ------------------------------
def symmetric_params(r_abs_max):
    return r_abs_max / 127, 0           # -127 .. 127, so that -r and r are mirror images


s_sym, z_sym = symmetric_params(3.0)
print(f"symmetric, |r| <= 3.0 -> scale = {s_sym:.6f}, zero_point = {z_sym}")
print("  1.0 ->", int(quantize(np.array([1.0]), s_sym, z_sym)[0]),
      " -1.0 ->", int(quantize(np.array([-1.0]), s_sym, z_sym)[0]))
