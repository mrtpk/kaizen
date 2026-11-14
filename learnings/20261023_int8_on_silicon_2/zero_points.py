"""Int8 on Silicon, part 2: making zero points disappear.

Shows that the input zero point can be folded into the bias at compile time,
and that this is exact only when the padding uses the input zero point.
Every number in the post comes from this script.
Run:  uv run --with numpy python zero_points.py
"""

import numpy as np

rng = np.random.default_rng(0)

# ---- one 3x3 convolution, one output channel ------------------------------
s_in, z_in = 4 / 255, -64            # activations in [-1, 3] (from Quantization Basics)
s_w, z_w = 0.0041, 0                 # int8 weights are symmetric: zero point 0
K = 3
q_w = rng.integers(-127, 128, (K, K))
bias = 1500                          # int32 bias, in units of s_in * s_w
q_in = rng.integers(-128, 128, (5, 5))

print("weights:\n", q_w)
print("sum of weights =", q_w.sum())


# ---- the four terms --------------------------------------------------------
patch = q_in[1:4, 1:4]               # one 3x3 window fully inside the image
N = K * K
t1 = int((patch * q_w).sum())
t2 = -z_w * int(patch.sum())
t3 = -z_in * int(q_w.sum())
t4 = N * z_in * z_w
ref = int(((patch - z_in) * (q_w - z_w)).sum())
print(f"\nsum (q_in - z_in)(q_w - z_w) = {ref}")
print(f"  term 1  sum q_in*q_w        = {t1}")
print(f"  term 2 -z_w * sum q_in      = {t2}   (z_w = 0)")
print(f"  term 3 -z_in * sum q_w      = {t3}   (constant: fold into the bias)")
print(f"  term 4  N * z_in * z_w      = {t4}   (z_w = 0)")
print("Four terms add up. Is pass?", t1 + t2 + t3 + t4 == ref)

bias_folded = bias - z_in * int(q_w.sum())
print(f"\nbias = {bias}, folded bias = bias - z_in * sum(q_w) = {bias_folded}")
print(f"raw MAC {t1} + folded bias {bias_folded} = {t1 + bias_folded}; reference {ref} + bias {bias} = {ref + bias}")
print("Folding is exact inside the image. Is pass?", t1 + bias_folded == ref + bias)


# ---- the whole image with SAME padding ------------------------------------
def conv_reference(q, w, b):
    """TFLite style: skip padded points (they are real 0), add the input offset in the loop."""
    H, W = q.shape
    out = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            acc = 0
            for dy in range(K):
                for dx in range(K):
                    iy, ix = y + dy - 1, x + dx - 1
                    if 0 <= iy < H and 0 <= ix < W:
                        acc += int(w[dy, dx]) * (int(q[iy, ix]) - z_in)
            out[y, x] = acc + b
    return out


def conv_folded(q, w, b_folded, pad_value):
    """Ours: pad the image, raw int8 MAC, folded bias."""
    p = np.pad(q, 1, constant_values=pad_value)
    H, W = q.shape
    out = np.zeros((H, W), dtype=np.int64)
    for y in range(H):
        for x in range(W):
            out[y, x] = int((p[y:y + K, x:x + K] * w).sum()) + b_folded
    return out


ref_img = conv_reference(q_in, q_w, bias)
pad_zp = conv_folded(q_in, q_w, bias_folded, pad_value=z_in)
pad_0 = conv_folded(q_in, q_w, bias_folded, pad_value=0)
print(f"\npad with z_in = {z_in}: equal to reference. Is pass?", np.array_equal(pad_zp, ref_img))
wrong = pad_0 != ref_img
print(f"pad with 0: {int(wrong.sum())} of {wrong.size} outputs wrong (all on the border: {bool(wrong[1:-1, 1:-1].sum() == 0)})")
corner = (0, 0)
print(f"corner output: reference {ref_img[corner]}, pad 0 gives {pad_0[corner]} "
      f"(off by {pad_0[corner] - ref_img[corner]})")
# the corner window has 5 padded points; each one adds z_in * w instead of 0
padded_w = int(q_w[0, :].sum() + q_w[1:, 0].sum())
print(f"  the 5 padded weights sum to {padded_w}; pad 0 drops z_in * {padded_w} = {z_in * padded_w} from the accumulator")

# in float, what is the corner error after requantization?
s_out = 0.0418
M = s_in * s_w / s_out
print(f"corner error in output units: {(pad_0[corner] - ref_img[corner]) * M:.2f} int8 steps")


# ---- weights with a zero point ---------------------------------------------
z_w2 = 3
q_w2 = rng.integers(-128, 128, (K, K))
w_pre = q_w2 - z_w2
print(f"\nweights with z_w = {z_w2}: subtract it offline -> range of q_w - z_w is "
      f"[{-128 - z_w2}, {127 - z_w2}] for this zero point, [-255, 255] in general (9 bits)")
ref2 = int(((patch - z_in) * (q_w2 - z_w2)).sum())
ours2 = int((patch * w_pre).sum()) - z_in * int(w_pre.sum())
print("Pre-subtracted weights + folded bias. Is pass?", ref2 == ours2)
