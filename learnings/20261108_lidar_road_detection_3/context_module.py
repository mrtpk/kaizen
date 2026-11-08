"""LiDAR Road Detection, part 3: seeing far with dilated convolutions.

Re-computes the receptive field of the LoDNN context module from its dilation
table, checks it with an impulse sent through dilated 3x3 convolutions, and
counts the weights of the network as I read it from the paper's figure.
Every number in the post comes from this script.

Run:  uv run --with numpy python context_module.py
"""

import numpy as np

# Context module of LoDNN (dilation as (width, height)), then a 1x1 convolution.
DILATIONS = [(1, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 64)]
GRID_H, GRID_W = 400, 200          # input grid, cells (rows = distance ahead)
CTX_H, CTX_W = 200, 100            # after one 2x2 max pooling


def receptive_fields(dilations: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """(width, height) after each 3x3 layer: each layer adds 2 * dilation."""
    w = h = 1
    out = []
    for dw, dh in dilations:
        w += 2 * dw
        h += 2 * dh
        out.append((w, h))
    return out


def dilated_ones(x: np.ndarray, dw: int, dh: int) -> np.ndarray:
    """3x3 convolution with all-ones weights and the given dilation, zero padding."""
    out = np.zeros_like(x)
    H, W = x.shape
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            si, sj = i * dh, j * dw
            src = x[max(0, si):H + min(0, si), max(0, sj):W + min(0, sj)]
            out[max(0, -si):H + min(0, -si), max(0, -sj):W + min(0, -sj)] += src
    return out


def impulse_support(dilations: list[tuple[int, int]], size: tuple[int, int]) -> tuple[int, int]:
    """Send one impulse through the layers; measure the width and height it reaches."""
    x = np.zeros(size)
    x[size[0] // 2, size[1] // 2] = 1.0
    for dw, dh in dilations:
        x = dilated_ones(x, dw, dh)
    rows, cols = np.nonzero(x)
    return int(cols.max() - cols.min() + 1), int(rows.max() - rows.min() + 1)


def conv_params(k: int, c_in: int, c_out: int) -> int:
    return k * k * c_in * c_out + c_out


def main() -> None:
    print("== 0. Plain and dilated 3x3 layers ==")
    for n in (1, 2, 3):
        print(f"{n} plain 3x3 layer(s): receptive field {receptive_fields([(1, 1)] * n)[-1][0]} x {receptive_fields([(1, 1)] * n)[-1][1]}")
    print("one 3x3 layer with dilation 2: receptive field %d x %d" % receptive_fields([(2, 2)])[-1])

    print("\n== 1. Receptive field from the dilation table ==")
    rf = receptive_fields(DILATIONS)
    for n, ((dw, dh), (w, h)) in enumerate(zip(DILATIONS, rf), start=1):
        print(f"layer {n}: dilation (w={dw:2d}, h={dh:2d})  receptive field {w:3d} x {h:3d}")
    w7, h7 = rf[-1]
    print("after layer 8 (1x1): %d x %d" % (w7, h7))

    print("\n== 2. Check with an impulse ==")
    w_imp, h_imp = impulse_support(DILATIONS, (h7 + 41, w7 + 41))
    print("impulse reaches", w_imp, "x", h_imp, "cells")
    print("Impulse matches the table. Is pass?", (w_imp, h_imp) == (w7, h7))
    # Plain 3x3 layers (no dilation) for the same reach in height:
    plain = (h7 - 1) // 2
    print("plain 3x3 layers needed for a height of %d: %d" % (h7, plain))

    print("\n== 3. Does it see the whole grid? ==")
    print("context input: %d x %d (width x height) after pooling the %d x %d grid" % (CTX_W, CTX_H, GRID_W, GRID_H))
    print("receptive field / context input: width %.2f, height %.2f" % (w7 / CTX_W, h7 / CTX_H))
    print("dilation ratio height / width in every layer from 2:", sorted({dh / dw for dw, dh in DILATIONS[1:]}))
    # In input cells: two 3x3 convs (+2 each), pool (+1, then x2), context, unpool, two 3x3 convs (+2 each).
    enc = 1 + 2 + 2
    after_pool = enc + 1
    w_in = after_pool + 2 * (w7 - 1) + 2 + 2
    h_in = after_pool + 2 * (h7 - 1) + 2 + 2
    print("whole network, in input cells: %d x %d = %.1f m x %.1f m (grid is 20 m x 40 m)"
          % (w_in, h_in, w_in * 0.1, h_in * 0.1))
    # The field is centred on its output: it reaches (size - 1) / 2 cells to each side.
    reach_w, reach_h = (w_in - 1) // 2, (h_in - 1) // 2
    print("reach of one output: %d cells to each side across, %d cells ahead and behind" % (reach_w, reach_h))
    rows = np.arange(GRID_H)
    cols = np.arange(GRID_W)
    full_r = (rows - reach_h <= 0) & (rows + reach_h >= GRID_H - 1)
    full_c = (cols - reach_w <= 0) & (cols + reach_w >= GRID_W - 1)
    print("outputs that see the whole grid: rows %d-%d, cols %d-%d = %.1f %% of the cells"
          % (rows[full_r].min(), rows[full_r].max(), cols[full_c].min(), cols[full_c].max(),
             100 * full_r.mean() * full_c.mean()))
    edge_rows = min(GRID_H, reach_h + 1)
    print("an output at the far edge (row 0) sees rows 0-%d = %.1f %% of the distance ahead"
          % (edge_rows - 1, 100 * edge_rows / GRID_H))
    ctx_reach = (h7 - 1) // 2
    print("in the context module alone: an edge output reaches %d of the %d rows" % (ctx_reach + 1, CTX_H))

    print("\n== 4. How many weights? (my reading of the paper's figure) ==")
    layers = [
        ("encoder 3x3, 6 -> 32", conv_params(3, 6, 32)),
        ("encoder 3x3, 32 -> 32", conv_params(3, 32, 32)),
        ("context 3x3, 32 -> 128", conv_params(3, 32, 128)),
        ("context 3x3, 128 -> 128 (x6)", 6 * conv_params(3, 128, 128)),
        ("context 1x1, 128 -> 32", conv_params(1, 128, 32)),
        ("decoder 3x3, 32 -> 32", conv_params(3, 32, 32)),
        ("decoder 3x3, 32 -> 2", conv_params(3, 32, 2)),
    ]
    total = sum(n for _, n in layers)
    for name, n in layers:
        print(f"{name:30s} {n:8d}  ({100 * n / total:4.1f} %)")
    print("total weights and biases:", total)
    ends = layers[0][1] + layers[1][1] + layers[5][1] + layers[6][1]
    print("encoder + decoder: %d (%.1f %%)" % (ends, 100 * ends / total))
    grid_cells = GRID_H * GRID_W
    print("grid cells per image:", grid_cells)


if __name__ == "__main__":
    main()
