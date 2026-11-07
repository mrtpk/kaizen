"""Int8 on Silicon, part 4: operations without a divider.

Average pooling as a convolution, global average pooling, lookup tables,
softmax on the host, and bilinear resize, each compared with TFLite's
integer reference or with exact rounding.
Every number in the post comes from this script.
Run:  uv run --with numpy python no_divider.py
"""

import math

import numpy as np

rng = np.random.default_rng(0)


def round_shift(x, shift):
    a = np.abs(x)
    r = (a >> shift) + ((a >> (shift - 1)) & 1)
    return np.where(x < 0, -r, r)


def round_half_away(v):
    return np.sign(v) * np.floor(np.abs(v) + 0.5)


# ---- 1. average pooling as a convolution with all-ones weights -----------
s, z = 0.05, -10                         # TFLite avg pool: input and output share scale and zero point
H = W = 8
K = 3
q = rng.integers(-128, 128, (H, W))
alpha = int(1 / (K * K) * (1 << 20))     # 1/N folded into alpha, at 2^20
print(f"AVG POOL 3x3 SAME on 8x8: alpha = int(2^20 / 9) = {alpha} ({(1 << 20) / 9:.2f})")

tfl = np.zeros((H, W), dtype=np.int64)
ours = np.zeros((H, W), dtype=np.int64)
p = np.pad(q, 1, constant_values=z)      # ours: pad with the zero point (real 0.0)
for y in range(H):
    for x in range(W):
        win = q[max(0, y - 1):y + 2, max(0, x - 1):x + 2]          # TFLite: only valid points
        acc, n = int(win.sum()), win.size
        tfl[y, x] = (acc + n // 2) // n if acc > 0 else -((-acc + n // 2) // n)
        acc9 = int((p[y:y + K, x:x + K] - z).sum())                  # ours: 9 points, minus z_in (folded bias)
        ours[y, x] = round_shift(np.array(acc9 * alpha), 20) + z
diff = ours != tfl
inner = diff[1:-1, 1:-1]
print(f"  inner outputs differ: {int(inner.sum())} of {inner.size}; border outputs differ: "
      f"{int(diff.sum() - inner.sum())} of {diff.size - inner.size}, max {np.abs(ours - tfl).max()}")
cy, cx = 0, 0
print(f"  corner: TFLite averages 4 points ({tfl[cy, cx]}), ours averages 9 points with 5 zeros ({ours[cy, cx]})")


# ---- 2. global average pooling: two roundings ------------------------------
s_in, z_in, s_out, z_out = 0.05, -10, 0.02, 5
N = 7 * 7
mean_mult = int(math.frexp(1 / N)[0] * (1 << (20 + math.frexp(1 / N)[1])))   # floor(2^20 / N)
a_req = int(s_in / s_out * (1 << 20))
trials = 100_000
x = rng.integers(-128, 128, (trials, N))
sums = x.sum(axis=1)
mean_q = round_shift(sums * mean_mult, 20)                                   # rounding 1: mean of raw q
two = np.clip(round_shift((mean_q - z_in) * a_req, 20) + z_out, -128, 127)    # rounding 2: requantize
one = np.clip(round_shift((sums - N * z_in) * int(round(s_in / (s_out * N) * (1 << 20))), 20) + z_out, -128, 127)
exact = np.clip(round_half_away(s_in * (sums / N - z_in) / s_out) + z_out, -128, 127)
print(f"\nGLOBAL AVG POOL 7x7, s_in {s_in}/z {z_in} -> s_out {s_out}/z {z_out}: mean multiplier {mean_mult} = floor(2^20/49)")
print(f"  two roundings (ours) != exact: {(two != exact).mean() * 100:.2f} %, max {np.abs(two - exact).max()}")
print(f"  one rounding           != exact: {(one != exact).mean() * 100:.2f} %, max {np.abs(one - exact).max()}")


# ---- 3. lookup tables ----------------------------------------------------------
s_in, z_in = 0.08, 0
s_out, z_out = 1 / 256, -128                    # TFLite's fixed output quantization for int8 sigmoid
qi = np.arange(-128, 128)
sig = 1 / (1 + np.exp(-s_in * (qi - z_in)))
lut_trunc = np.clip((sig / s_out + z_out).astype(int), -128, 127)            # int(): truncation, as in our code
lut_round = np.clip(round_half_away(sig / s_out) + z_out, -128, 127)
print(f"\nSIGMOID LUT (256 entries), s_in {s_in}: truncated table != rounded table in {(lut_trunc != lut_round).sum()} of 256 entries")
print(f"  examples q_in -> (truncated, rounded): "
      + ", ".join(f"{a} -> ({lut_trunc[a + 128]}, {int(lut_round[a + 128])})" for a in (-20, 0, 20)))


def hard_swish(x):
    return x * np.clip(x + 3, 0, 6) / 6


s_in, z_in, s_out, z_out = 0.05, 0, 0.04, -20
hs = hard_swish(s_in * (qi - z_in))
lut_hs_t = np.clip((hs / s_out + z_out).astype(int), -128, 127)
lut_hs_r = np.clip(round_half_away(hs / s_out) + z_out, -128, 127)
print(f"HARD-SWISH LUT: truncated != rounded in {(lut_hs_t != lut_hs_r).sum()} of 256 entries")


# ---- 4. softmax on the host ------------------------------------------------------
s_in, z_in = 0.1, 0
s_out, z_out = 1 / 256, -128
logits_q = rng.integers(-128, 128, (20_000, 10))
r = s_in * (logits_q - z_in)
e = np.exp(r - r.max(axis=1, keepdims=True))
prob = e / e.sum(axis=1, keepdims=True)
sm_trunc = np.clip((prob / s_out + z_out).astype(int), -128, 127)
sm_round = np.clip(round_half_away(prob / s_out) + z_out, -128, 127)
print(f"\nSOFTMAX (float on the host, 10 classes): truncated output != rounded in {(sm_trunc != sm_round).mean() * 100:.1f} % of values")
print("  argmax unchanged. Is pass?",
      bool((np.argmax(sm_trunc, axis=1) == np.argmax(prob, axis=1)).mean() > 0.99))


# ---- 5. bilinear resize x2 -------------------------------------------------------
def coords_float(o, scale, size):
    v = (o + 0.5) * scale - 0.5
    lo = max(int(math.floor(v)), 0)
    hi = min(int(math.ceil(v)), size - 1)
    return v, lo, hi


def resize_ours(img, out_h, out_w):
    """Float coordinates, each of the 4 weights truncated to 10 bits, round shift 10."""
    h, w = img.shape
    out = np.zeros((out_h, out_w), dtype=np.int64)
    wsum = []
    for y in range(out_h):
        iy, y0, y1 = coords_float(y, h / out_h, h)
        for x in range(out_w):
            ix, x0, x1 = coords_float(x, w / out_w, w)
            dy = 0.0 if y0 == y1 else iy - y0
            dx = 0.0 if x0 == x1 else ix - x0
            ws = [int((1 - dy) * (1 - dx) * 1024), int(dy * (1 - dx) * 1024),
                  int((1 - dy) * dx * 1024), int(dy * dx * 1024)]
            wsum.append(sum(ws))
            acc = (int(img[y0, x0]) * ws[0] + int(img[y1, x0]) * ws[1]
                   + int(img[y0, x1]) * ws[2] + int(img[y1, x1]) * ws[3])
            out[y, x] = round_shift(np.array(acc), 10)
    return out, np.array(wsum)


def resize_tflite(img, out_h, out_w):
    """TFLite ResizeBilinearInteger, half_pixel_centers, default build."""
    h, w = img.shape
    hs10 = ((1 << 10) * h + out_h // 2) // out_h
    ws10 = ((1 << 10) * w + out_w // 2) // out_w

    def ci(v, sc, size):
        sv = v * sc + sc // 2 - (1 << 9)
        lo = max(sv // (1 << 10) if sv >= 0 else -((-sv) // (1 << 10)), 0)
        hi = min((sv + (1 << 10) - 1) // (1 << 10) if sv + 1023 >= 0 else -((-(sv + 1023)) // 1024), size - 1)
        return sv, lo, hi

    out = np.zeros((out_h, out_w), dtype=np.int64)
    for y in range(out_h):
        iy, y0, y1 = ci(y, hs10, h)
        for x in range(out_w):
            ix, x0, x1 = ci(x, ws10, w)
            fy, fx = iy - 1024 * y0, ix - 1024 * x0
            acc = (int(img[y0, x0]) * (1024 - fy) * (1024 - fx) + int(img[y1, x0]) * fy * (1024 - fx)
                   + int(img[y0, x1]) * (1024 - fy) * fx + int(img[y1, x1]) * fy * fx)
            rnd = (1 << 19) if acc > 0 else -(1 << 19)
            v = acc + rnd
            out[y, x] = v // (1 << 20) if v >= 0 else -((-v) // (1 << 20))
    return out


img = rng.integers(-128, 128, (8, 8))
o_ours, wsum = resize_ours(img, 16, 16)
o_tfl = resize_tflite(img, 16, 16)
print(f"\nRESIZE BILINEAR 8x8 -> 16x16: our 4 weights sum to {sorted(set(wsum.tolist()))} (should be 1024)")
print(f"  ours != TFLite: {(o_ours != o_tfl).sum()} of {o_ours.size}, max {np.abs(o_ours - o_tfl).max()}")

img = rng.integers(-128, 128, (8, 8))
o_ours, wsum = resize_ours(img, 13, 13)
o_tfl = resize_tflite(img, 13, 13)
print(f"RESIZE BILINEAR 8x8 -> 13x13: our 4 weights sum to {min(wsum)}..{max(wsum)} "
      f"({(wsum < 1024).mean() * 100:.0f} % of outputs below 1024)")
print(f"  ours != TFLite: {(o_ours != o_tfl).sum()} of {o_ours.size}, max {np.abs(o_ours - o_tfl).max()}")
