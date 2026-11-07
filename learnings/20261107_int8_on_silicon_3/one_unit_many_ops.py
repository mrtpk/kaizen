"""Int8 on Silicon, part 3: one unit, many operations.

Add, multiply, operations with a constant, and the ReLU family, all as
"(x + bias) * alpha, rounding shift, + zero point, clip".
Every number in the post comes from this script.
Run:  uv run --with numpy python one_unit_many_ops.py
"""

import math

import numpy as np

rng = np.random.default_rng(0)
N = 200_000


def round_shift(x, shift):
    """Round half away from zero (works on numpy int64 arrays)."""
    a = np.abs(x)
    r = (a >> shift) + ((a >> (shift - 1)) & 1)
    return np.where(x < 0, -r, r)


def clip8(x):
    return np.clip(x, -128, 127)


def exact(r, s, z):
    """Float reference: quantize the real result with round half away from zero."""
    v = r / s
    return clip8(np.sign(v) * np.floor(np.abs(v) + 0.5) + z).astype(np.int64)


# TFLite helpers (default build: two roundings), on int64 numpy arrays
def quantize_multiplier(m):
    q, shift = math.frexp(m)
    q31 = round(q * (1 << 31))
    if q31 == 1 << 31:
        q31, shift = q31 // 2, shift + 1
    return q31, shift


def high_mul(a, b):
    ab = a * b
    v = ab + np.where(ab >= 0, 1 << 30, 1 - (1 << 30))
    return np.where(v >= 0, v // (1 << 31), -((-v) // (1 << 31)))


def rounding_divide(x, e):
    if e == 0:
        return x
    mask = (1 << e) - 1
    threshold = (mask >> 1) + (x < 0)
    return (x >> e) + ((x & mask) > threshold)


def mbqm(x, m):
    q31, shift = quantize_multiplier(m)
    left, right = max(shift, 0), max(-shift, 0)
    return rounding_divide(high_mul(x * (1 << left), q31), right)


# ---- add: three passes through the unit, one rounding -----------------------
s1, z1, s2, z2, so, zo = 0.05, -10, 0.03, 5, 0.07, -3
q1 = rng.integers(-128, 128, N)
q2 = rng.integers(-128, 128, N)
real_sum = s1 * (q1 - z1) + s2 * (q2 - z2)

a1 = int(s1 / so * (1 << 20))
a2 = int(s2 / so * (1 << 20))
ours_add = clip8(round_shift((q1 - z1) * a1 + (q2 - z2) * a2, 20) + zo)

# TFLite add (add.cc / integer_ops/add.h): left shift 20, three multipliers
twice_max = 2 * max(s1, s2)
x1 = mbqm((q1 - z1) * (1 << 20), s1 / twice_max)
x2 = mbqm((q2 - z2) * (1 << 20), s2 / twice_max)
tfl_add = clip8(mbqm(x1 + x2, twice_max / ((1 << 20) * so)) + zo)
ref_add = exact(real_sum, so, zo)

print("ADD  s1 = 0.05, z1 = -10, s2 = 0.03, z2 = 5, s_out = 0.07, z_out = -3")
print(f"  alpha1 = int(s1 / s_out * 2^20) = {a1}, alpha2 = {a2}, bias1 = {-z1}, bias2 = {-z2}")
print(f"  ours != TFLite: {(ours_add != tfl_add).mean() * 100:.3f} %, max {np.abs(ours_add - tfl_add).max()}")
print(f"  ours != exact:  {(ours_add != ref_add).mean() * 100:.3f} %;  TFLite != exact: {(tfl_add != ref_add).mean() * 100:.3f} %")
q1e, q2e = 40, -70
v = round_shift(np.array([(q1e - z1) * a1 + (q2e - z2) * a2]), 20)[0] + zo
print(f"  example q1 = {q1e}, q2 = {q2e}: ({q1e - z1})*{a1} + ({q2e - z2})*{a2} = {(q1e - z1) * a1 + (q2e - z2) * a2:,}"
      f" -> >>20 = {v - zo} -> + z_out = {v};  float {s1 * (q1e - z1) + s2 * (q2e - z2):.3f} / s_out + z_out = "
      f"{(s1 * (q1e - z1) + s2 * (q2e - z2)) / so + zo:.3f}")


# ---- multiply: the square-root split ------------------------------------------
s1, z1, s2, z2, so, zo = 0.05, -10, 0.03, 5, 0.12, -20
q1 = rng.integers(-128, 128, N)
q2 = rng.integers(-128, 128, N)
real_prod = s1 * (q1 - z1) * s2 * (q2 - z2)

m1 = int(s1 / math.sqrt(so) * (1 << 10))
m2 = int(s2 / math.sqrt(so) * (1 << 10))
ours_mul = clip8(round_shift(((q1 - z1) * m1) * ((q2 - z2) * m2), 20) + zo)
tfl_mul = clip8(mbqm((q1 - z1) * (q2 - z2), s1 * s2 / so) + zo)
ref_mul = exact(real_prod, so, zo)
print("\nMUL  s1 = 0.05, z1 = -10, s2 = 0.03, z2 = 5, s_out = 0.12, z_out = -20")
print(f"  alpha_i = int(s_i / sqrt(s_out) * 2^10): {m1}, {m2};  product shift 20")
print(f"  ours != TFLite: {(ours_mul != tfl_mul).mean() * 100:.3f} %, max {np.abs(ours_mul - tfl_mul).max()}")
print(f"  ours != exact:  {(ours_mul != ref_mul).mean() * 100:.3f} %;  TFLite != exact: {(tfl_mul != ref_mul).mean() * 100:.3f} %")
op_max = 255 * max(m1, m2)
print(f"  largest scaled operand |(q - z) * alpha| <= 255 * {max(m1, m2)} = {op_max} ({op_max.bit_length()} bits)")
print(f"  alpha error: {m1} vs {s1 / math.sqrt(so) * 1024:.2f}, {m2} vs {s2 / math.sqrt(so) * 1024:.2f}")


def mul_variants(s1, z1, s2, z2, so, zo, label):
    q1 = rng.integers(-128, 128, N)
    q2 = rng.integers(-128, 128, N)
    ref = exact(s1 * (q1 - z1) * s2 * (q2 - z2), so, zo)
    tfl = clip8(mbqm((q1 - z1) * (q2 - z2), s1 * s2 / so) + zo)
    a1t, a2t = int(s1 / math.sqrt(so) * 1024), int(s2 / math.sqrt(so) * 1024)
    a1r, a2r = round(s1 / math.sqrt(so) * 1024), round(s2 / math.sqrt(so) * 1024)
    a_one = round(s1 * s2 / so * (1 << 20))
    split_t = clip8(round_shift(((q1 - z1) * a1t) * ((q2 - z2) * a2t), 20) + zo)
    split_r = clip8(round_shift(((q1 - z1) * a1r) * ((q2 - z2) * a2r), 20) + zo)
    one = clip8(round_shift((q1 - z1) * (q2 - z2) * a_one, 20) + zo)
    print(f"  {label}: alphas {a1t}, {a2t} (truncated)")
    for name, out in [("split, truncated alphas (ours)", split_t), ("split, rounded alphas", split_r),
                      (f"all of M on one side (alpha {a_one})", one), ("TFLite", tfl)]:
        print(f"    {name:38s} != exact {(out != ref).mean() * 100:6.2f} %, max {np.abs(out - ref).max()}")


print("  where the multiply error comes from:")
mul_variants(0.05, -10, 0.03, 5, 0.12, -20, "example above")
# squeeze-and-excitation: activation times a sigmoid output (scale 1/256, zero point -128)
mul_variants(0.05, -10, 1 / 256, -128, 0.05, -10, "SE block, x * sigmoid")


# ---- an operand that is a constant ---------------------------------------------
s_in, z_in, so, zo = 0.02, 10, 0.03, -5
q = np.arange(-128, 128)
c = 0.75                                        # add 0.75
alpha = int(s_in / so * (1 << 16))
bias = int(c / s_in) - z_in                     # c in input units, folded with the zero point
ours_addc = clip8(round_shift((q + bias) * alpha, 16) + zo)
ref_addc = exact(s_in * (q - z_in) + c, so, zo)
print(f"\nADD CONST  x + {c}: alpha = {alpha}, bias = int({c} / {s_in}) - {z_in} = {bias}  ({c / s_in} truncated)")
print(f"  ours != exact: {(ours_addc != ref_addc).sum()} of 256 inputs, max {np.abs(ours_addc - ref_addc).max()}")
print("  (the constant 37.5 has no exact integer in input units; the bias can only hold 37 or 38)")

k = -1.5                                        # multiply by -1.5
alpha_k = int(k * s_in / so * (1 << 16))
ours_mulc = clip8(round_shift((q - z_in) * alpha_k, 16) + zo)
ref_mulc = exact(k * s_in * (q - z_in), so, zo)
print(f"MUL CONST  x * {k}: alpha = int({k} * s_in / s_out * 2^16) = {alpha_k}, bias = {-z_in}")
print(f"  ours != exact: {(ours_mulc != ref_mulc).sum()} of 256 inputs, max {np.abs(ours_mulc - ref_mulc).max()}")


# ---- the ReLU family as one PReLU kernel ----------------------------------------
s_in, z_in, so, zo = 1 / 16, 0, 1 / 16, 0      # exact in binary, so the float reference has no tie noise


def prelu(q, slope_q, slope_s, slope_z, lo=-128, hi=127):
    v = q.astype(np.int64) - z_in
    a_pos = int(s_in / so * (1 << 16))
    a_neg = int(s_in * slope_s / so * (1 << 16))
    out = np.where(v >= 0, round_shift(v * a_pos, 16), round_shift((slope_q - slope_z) * v * a_neg, 16))
    return np.clip(out + zo, lo, hi)


q = np.arange(-128, 128)
x = s_in * (q - z_in)
cases = {
    "ReLU (slope 0)": (prelu(q, 0, 1.0, 0), np.maximum(x, 0)),
    "LeakyReLU 0.1 (slope q = 1, scale 0.1)": (prelu(q, 1, 0.1, 0), np.where(x >= 0, x, 0.1 * x)),
    "PReLU slope 0.25 (q = 64, scale 1/256)": (prelu(q, 64, 1 / 256, 0), np.where(x >= 0, x, 0.25 * x)),
    "ReLU6 (slope 0, clip at 6.0 = q 96)": (prelu(q, 0, 1.0, 0, hi=int(6.0 / so) + zo), np.clip(x, 0, 6)),
    "min(x, 2.0) (slope 1, clip at q 32)": (prelu(q, 1, 1.0, 0, hi=int(2.0 / so) + zo), np.minimum(x, 2.0)),
}
print("\nRELU FAMILY  s_in = s_out = 1/16, zero points 0; all through one PReLU kernel (shift 16)")
for name, (ours, real) in cases.items():
    ref = exact(real, so, zo)
    print(f"  {name:42s} ours != exact: {(ours != ref).sum()} of 256, max {np.abs(ours - ref).max()}")
