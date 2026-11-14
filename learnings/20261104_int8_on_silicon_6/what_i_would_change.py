"""Int8 on Silicon, part 6: what I would do differently.

Demonstrates the overflow guard that cannot fire, max pooling and concat
semantics, and the 32-bit budget that alpha and the product share.
Every number in the post comes from this script.
Run:  uv run --with numpy python what_i_would_change.py
"""


import numpy as np

rng = np.random.default_rng(0)


# ---- 1. an overflow guard that cannot fire ----------------------------------
def guard(value):
    """Our check: warn if a value is outside int32. It receives the product AFTER it was computed in int32."""
    return value > 2**31 - 1 or value < -2**31


alpha = np.int32(2416)                       # part 1
acc = np.int32(1_000_000)                    # |acc * M| = 2305: past the 2048 limit from part 1
with np.errstate(over="ignore"):
    p32 = alpha * acc                        # int32 x int32 -> int32: wraps
p64 = np.int64(alpha) * np.int64(acc)
print(f"OVERFLOW  alpha {alpha} x acc {acc:,}")
print(f"  true product {p64:,}; int32 product {p32:,} (wrapped)")
print(f"  guard(int32 product) fires: {guard(int(p32))};  guard(int64 product) fires: {guard(int(p64))}")
shift = 20
r32 = int(p32) >> shift
r64 = int(p64) >> shift
c32, c64 = max(min(r32, 127), -128), max(min(r64, 127), -128)
print(f"  >> 20: wrapped gives {r32}, true gives {r64} -> after clip: {c32} vs {c64}")
# the "reference" check: int32 * int32 widened after the fact is the same wrapped value
with np.errstate(over="ignore"):
    out_ref = np.int64(np.int32(alpha) * np.int32(acc))
print(f"  assert(int64(int32 * int32) == int32 * int32): {out_ref == p32}  (always true, so it never catches anything)")


# ---- 2. max pooling: pad with the zero point vs skip padding -------------------
s, z = 0.05, 20
q = rng.integers(-128, 20, (8, 8))           # mostly negative real values (q < z), no ReLU before the pool
pad = np.pad(q, 1, constant_values=z)        # ours: padding = z = real 0.0
ours = np.array([[pad[y:y + 3, x:x + 3].max() for x in range(8)] for y in range(8)])
tfl = np.array([[q[max(0, y - 1):y + 2, max(0, x - 1):x + 2].max() for x in range(8)] for y in range(8)])
diff = ours != tfl
print(f"\nMAXPOOL 3x3 SAME, mostly negative input: {int(diff.sum())} of 64 differ "
      f"(inner {int(diff[1:-1, 1:-1].sum())}), max {np.abs(ours - tfl).max()} steps "
      f"= {np.abs(ours - tfl).max() * s:.2f} in real units")


# ---- 3. concat with different input scales -----------------------------------------
s1, z1, s2, z2 = 0.05, 0, 0.10, 0
s_out, z_out = 0.10, 0
a = rng.integers(-128, 128, 1000)
b = rng.integers(-128, 128, 1000)
raw = np.concatenate([a, b])                                      # ours: copy the bytes
# what a scale check would force: requantize input 1
requant = np.concatenate([np.clip(np.round(a * s1 / s_out), -128, 127), b])
real = np.concatenate([a * s1, b * s2])
print(f"\nCONCAT s1 = {s1}, s2 = {s2}, s_out = {s_out}:")
print(f"  copy:        max real error {np.abs(raw * s_out - real).max():.2f}")
print(f"  requantize:  max real error {np.abs(requant * s_out - real).max():.3f}")


# ---- 4. alpha and the product share 32 bits -----------------------------------------
print("\nBIT BUDGET  alpha * acc ~ output * 2^shift must fit in int32 (31 bits + sign)")
print("shift  overflow when |output| >   alpha bits for M = 0.0023   for M = 0.0001")
for sh in (16, 20, 23, 24, 28):
    limit = 2**31 // 2**sh
    b1 = int(0.0023050238 * 2**sh).bit_length()
    b2 = int(0.0001 * 2**sh).bit_length()
    flag = "  <- int8 range (128) no longer fits" if limit < 128 else ""
    print(f"{sh:5d}  {limit:>23,}   {b1:>21}   {b2:>13}{flag}")
for width in (32, 40, 48):
    sh = width - 1 - 7 - 4                    # sign, int8 magnitude (7 bits), 4 spare bits (part 1)
    print(f"  {width}-bit product: shift up to {sh}, alpha bits for M = 0.0001: {int(0.0001 * 2**sh).bit_length()}")
