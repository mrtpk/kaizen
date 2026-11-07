"""Int8 on Silicon, part 5: the compiler frontend.

Three graph rewrites, each checked with NumPy, and the partitioning rule that
splits a model between the accelerator and the host.
Every number in the post comes from this script.
Run:  uv run --with numpy python frontend.py
"""

import math

import numpy as np

rng = np.random.default_rng(0)


# ---- 1. SAME convolution -> explicit PAD + VALID convolution ---------------------
def same_pads(size, k, stride):
    """TensorFlow SAME padding: the extra pixel goes to the bottom / right."""
    out = math.ceil(size / stride)
    total = max((out - 1) * stride + k - size, 0)
    return total // 2, total - total // 2


def conv_same_reference(q, w, stride, z_in):
    """TFLite style: padded points are skipped (real 0.0), input offset in the loop."""
    H, W = q.shape
    k = w.shape[0]
    pt, _ = same_pads(H, k, stride)
    pl, _ = same_pads(W, k, stride)
    oh, ow = math.ceil(H / stride), math.ceil(W / stride)
    out = np.zeros((oh, ow), dtype=np.int64)
    for y in range(oh):
        for x in range(ow):
            for dy in range(k):
                for dx in range(k):
                    iy, ix = y * stride + dy - pt, x * stride + dx - pl
                    if 0 <= iy < H and 0 <= ix < W:
                        out[y, x] += int(w[dy, dx]) * (int(q[iy, ix]) - z_in)
    return out


def conv_valid(q, w, stride, z_in):
    k = w.shape[0]
    oh, ow = (q.shape[0] - k) // stride + 1, (q.shape[1] - k) // stride + 1
    out = np.zeros((oh, ow), dtype=np.int64)
    for y in range(oh):
        for x in range(ow):
            out[y, x] = int(((q[y * stride:y * stride + k, x * stride:x * stride + k] - z_in) * w).sum())
    return out


z_in = -64
for size, stride in [(7, 2), (8, 2), (8, 1)]:
    q = rng.integers(-128, 128, (size, size))
    w = rng.integers(-127, 128, (3, 3))
    pt, pb = same_pads(size, 3, stride)
    padded = np.pad(q, ((pt, pb), (pt, pb)), constant_values=z_in)
    ok = np.array_equal(conv_valid(padded, w, stride, z_in), conv_same_reference(q, w, stride, z_in))
    print(f"SAME 3x3 stride {stride} on {size}x{size}: pad top/bottom = {pt}/{pb} -> PAD + VALID. Is pass? {ok}")


# ---- 2. Reshape, Transpose, Reshape, DepthToSpace -> one DepthToSpace (CRD) ------
def depth_to_space_dcr(x, r):
    """TensorFlow / TFLite DEPTH_TO_SPACE (NHWC, depth-column-row order)."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, r, r, c // (r * r))
    x = x.transpose(0, 1, 3, 2, 4, 5)
    return x.reshape(n, h * r, w * r, c // (r * r))


def depth_to_space_crd(x, r):
    """PyTorch PixelShuffle order (column-row-depth), written for NHWC."""
    n, h, w, c = x.shape
    x = x.reshape(n, h, w, c // (r * r), r, r)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(n, h * r, w * r, c // (r * r))


r, C = 2, 3
x = rng.integers(-128, 128, (1, 4, 4, C * r * r))
# a PixelShuffle (CRD) written as Reshape-Transpose-Reshape in front of a DCR DepthToSpace:
pattern = x.reshape(1, 4, 4, C, r, r)                     # Reshape
pattern = pattern.transpose(0, 1, 2, 4, 5, 3)             # Transpose
pattern = pattern.reshape(1, 4, 4, C * r * r)             # Reshape
pattern = depth_to_space_dcr(pattern, r)                  # DepthToSpace (DCR)
print(f"\nReshape-Transpose-Reshape-DepthToSpace == DepthToSpace(CRD). Is pass? "
      f"{np.array_equal(pattern, depth_to_space_crd(x, r))}")
print("  4 layers become 1")


# ---- 3. Split -> Slices ---------------------------------------------------------
x = rng.integers(-128, 128, (1, 6, 8, 8))                  # NCHW
splits = np.split(x, 3, axis=1)
slices = [x[:, i * 2:(i + 1) * 2] for i in range(3)]
print(f"\nSplit into 3 == 3 Slices. Is pass? {all(np.array_equal(a, b) for a, b in zip(splits, slices))}")


# ---- 4. Partition a model between the accelerator and the host ------------------
# a small graph in execution order: (name, supported on the accelerator, inputs)
graph = [
    ("conv1", True, []),
    ("conv2", True, ["conv1"]),
    ("argmax", False, ["conv1"]),        # an op the accelerator does not support
    ("conv3", True, ["conv2"]),
    ("add", True, ["conv3", "argmax"]),
    ("softmax", False, ["add"]),
]
order = [n for n, _, _ in graph]
sup = {n: s for n, s, _ in graph}
edges = [(i, n) for n, _, ins in graph for i in ins]


def partition(edges, cut_span_rule):
    keep = []
    for a, b in edges:
        if sup[a] != sup[b]:
            continue                                         # rule 1: cut at the support boundary
        if cut_span_rule:
            span = order[order.index(a):order.index(b) + 1]
            if any(not sup[n] for n in span):
                continue                                     # rule 2: an unsupported op runs in between
        keep.append((a, b))
    parent = {n: n for n in order}

    def find(n):
        while parent[n] != n:
            n = parent[n]
        return n

    for a, b in keep:
        parent[find(b)] = find(a)
    groups = {}
    for n in order:
        groups.setdefault(find(n), []).append(n)
    return list(groups.values())


print("\nPARTITION  order:", " -> ".join(order))
for rule in (False, True):
    parts = partition(edges, rule)
    label = "with the span rule   " if rule else "boundary rule only   "
    print(f"  {label}: " + " | ".join(("ACC " if sup[p[0]] else "HOST ") + "{" + ", ".join(p) + "}" for p in parts))
print("  Without the span rule, {conv1, conv2, conv3, add} is one accelerator subgraph,"
      " but add needs argmax, which runs on the host after conv1.")
