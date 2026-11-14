"""Int8 on Silicon, part 1: one multiply, one shift.

Compares two ways to compute round(acc * M) with integers only:
  - TFLite (default build): Q31 multiplier + shift, two roundings
  - ours: alpha = int(M * 2^20), one multiply, one rounding shift by 20
Every number in the post comes from this script.
Run:  uv run --with numpy python one_multiply_one_shift.py
"""

import math

import numpy as np

rng = np.random.default_rng(0)


# ---- the multiplier M ----------------------------------------------------
s_in, s_w, s_out = 0.0235, 0.0041, 0.0418          # example scales of one conv layer
M = float(np.float32(s_in) * (np.float32(s_w) / np.float32(s_out)))
print(f"M = s_in * s_w / s_out = {M:.10f}")


# ---- TFLite: Q31 mantissa + shift, two roundings --------------------------
def quantize_multiplier(m):
    """frexp, then round the mantissa to Q31 (TFLite QuantizeMultiplier)."""
    q, shift = math.frexp(m)
    q_fixed = round(q * (1 << 31))
    if q_fixed == 1 << 31:
        q_fixed //= 2
        shift += 1
    return q_fixed, shift


def saturating_rounding_doubling_high_mul(a, b):
    """gemmlowp: high 32 bits of 2*a*b, rounded (rounding #1)."""
    if a == b == -(1 << 31):
        return (1 << 31) - 1
    ab = a * b
    nudge = (1 << 30) if ab >= 0 else 1 - (1 << 30)
    v = ab + nudge
    return v // (1 << 31) if v >= 0 else -((-v) // (1 << 31))   # C++ division truncates toward 0


def rounding_divide_by_pot(x, exponent):
    """gemmlowp: x / 2^exponent, round half away from zero (rounding #2)."""
    mask = (1 << exponent) - 1
    remainder = x & mask
    threshold = (mask >> 1) + (1 if x < 0 else 0)
    return (x >> exponent) + (1 if remainder > threshold else 0)


def tflite_requant(acc, q31, shift):
    left, right = max(shift, 0), max(-shift, 0)
    return rounding_divide_by_pot(saturating_rounding_doubling_high_mul(acc * (1 << left), q31), right)


# ---- ours: alpha = M * 2^20, one multiply, one rounding shift ------------
SHIFT = 20


def make_alpha(m, shift=SHIFT):
    """Computed once per channel, at compile time (float32, truncated, as in the reference model)."""
    return int(np.float32(m) * np.float32(1 << shift))


def round_shift(x, shift):
    """Round half away from zero with a shift, a mask and the sign."""
    a = abs(x)
    r = (a >> shift) + ((a >> (shift - 1)) & 1)
    return -r if x < 0 else r


def ours_requant(acc, alpha, shift=SHIFT):
    return round_shift(alpha * acc, shift)


q31, shift = quantize_multiplier(M)
alpha = make_alpha(M)
print(f"TFLite: q31 = {q31}, shift = {shift}  (M = q31 / 2^31 * 2^{shift})")
print(f"ours:   alpha = {alpha}, shift = {SHIFT}  (alpha / 2^20 = {alpha / 2**SHIFT:.10f}, "
      f"relative error {(alpha / 2**SHIFT - M) / M * 100:+.3f} %)")

for acc in [1000, 12345, -54321]:
    print(f"  acc = {acc:6d}: exact {acc * M:9.4f}  TFLite {tflite_requant(acc, q31, shift):5d}  "
          f"ours {ours_requant(acc, alpha):5d}")

# ---- compare on many accumulators ----------------------------------------
N = 200_000
accs = rng.integers(-60_000, 60_001, N)
ours = np.array([ours_requant(int(a), alpha) for a in accs])
tfl = np.array([tflite_requant(int(a), q31, shift) for a in accs])
exact = np.array([int(math.copysign(math.floor(abs(a * M) + 0.5), a * M)) for a in accs])   # round half away from zero
diff = ours != tfl
print(f"ours != TFLite: {diff.mean() * 100:.2f} % of {N}, max difference {np.abs(ours - tfl).max()}")
print(f"TFLite != exact rounding: {(tfl != exact).mean() * 100:.2f} %")
print(f"ours != exact rounding:   {(ours != exact).mean() * 100:.2f} %")
print("Never more than 1 apart. Is pass?", np.abs(ours - tfl).max() <= 1)

# where does the mismatch come from? round alpha instead of truncating it
alpha_r = int(round(M * (1 << SHIFT)))
ours_r = np.array([ours_requant(int(a), alpha_r) for a in accs])
print(f"with alpha rounded ({alpha_r}): ours != exact {(ours_r != exact).mean() * 100:.2f} %, "
      f"ours != TFLite {(ours_r != tfl).mean() * 100:.2f} %")

# overflow: alpha * acc ~ (acc * M) * 2^20 must fit in int32
print(f"int32 overflow only when |acc * M| > 2^31 / 2^20 = {2**31 // 2**SHIFT}")


# ---- one output through the post-processing unit -------------------------
acc, bias, zp_out, lo, hi = 12345, -500, -12, -128, 127
t = acc + bias
p = alpha * t
r = round_shift(p, SHIFT)
y = min(max(r + zp_out, lo), hi)
print(f"\nunit: acc {acc} + bias {bias} = {t};  x alpha {alpha} = {p:,};  "
      f">> {SHIFT} (rounded) = {r} ({p / 2**SHIFT:.3f});  + zero_point {zp_out} = {r + zp_out};  clip -> {y}")
print(f"float check: (acc + bias) * M + zero_point = {t * M + zp_out:.3f}")


# ---- what a fixed shift costs --------------------------------------------
print("\nM          alpha     bits  alpha error")
for m in [0.5, 0.1, 0.01, M, 0.001, 0.0001]:
    a = make_alpha(m)
    bits = a.bit_length()
    err = (a / 2**SHIFT - m) / m * 100
    print(f"{m:<10.6g} {a:<9d} {bits:<5d} {err:+8.3f} %")


# ---- rounding at the bit level (figure 5) --------------------------------
for x in (28_617_520, alpha * 12_420, -28_617_520):
    a = abs(x)
    print(f"x = {x:,}: integer part {a >> SHIFT}, first dropped bit {(a >> (SHIFT - 1)) & 1} -> round_shift = "
          f"{round_shift(x, SHIFT)}"
          f"  (plain >> gives {x >> SHIFT}, exact {x / 2**SHIFT:.3f})")
