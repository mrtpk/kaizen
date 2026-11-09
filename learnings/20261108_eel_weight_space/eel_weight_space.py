"""Weight-Space Programming - an eel detector: the checks and the numbers of the post.

Part A runs the four weight-space tools of the study on small synthetic arrays (NumPy only):
  1. pruning a 22-class head to 1 class is exact
  2. peak suppression removes adjacent and cross-scale duplicates
  3. averaging two heads' weights is not averaging their outputs; a widened head is, exactly
  4. WiSE-FT interpolation, and why one pixel matters on a 9 px eel
Part B prints the study's results from study_results.json (exported from the experiment logs
by experiments/export_blog_numbers.py in the study repository).

Every number in the post comes from this script.
Run:  uv run --with numpy python eel_weight_space.py
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load() -> dict:
    return json.loads((HERE / "study_results.json").read_text())


def fitness(run: dict) -> float:
    return run["fitness"]


# ---- 1. prune 22 classes to 1 ---------------------------------------------
def prune_check() -> float:
    rng = np.random.default_rng(0)
    feats = rng.normal(size=(64, 100))  # 64 features at 100 anchors
    w, b = rng.normal(size=(22, 64)), rng.normal(size=22)
    full = w @ feats + b[:, None]  # 22 class logits per anchor
    eel = w[21:22] @ feats + b[21:22, None]  # the sliced 1-class head
    return float(np.abs(full[21:22] - eel).max())


# ---- 2. peak suppression ---------------------------------------------------
def maxpool3(m: np.ndarray) -> np.ndarray:
    p = np.pad(m, 1, constant_values=-np.inf)
    return np.max([p[i:i + m.shape[0], j:j + m.shape[1]] for i in range(3) for j in range(3)], axis=0)


def peak_suppress(maps: list[np.ndarray], penalty: float = 20.0) -> list[np.ndarray]:
    """maps: class logits per scale, finest first (stride 8, 16, 32); each map is half the size of the last.

    A cell keeps its logit when it is the max of its own 3x3 neighbourhood and is not beaten by the
    overlapping 3x3 neighbourhood of the next coarser and the next finer scale. Others get -penalty.
    """
    out = []
    for i, m in enumerate(maps):
        keep = m >= maxpool3(m)
        if i + 1 < len(maps):  # coarser scale, upsampled (nearest)
            up = np.kron(maxpool3(maps[i + 1]), np.ones((2, 2)))[:m.shape[0], :m.shape[1]]
            keep &= m >= up
        if i > 0:  # finer scale, 3x3 max then 2x2 max-pool
            f = maxpool3(maps[i - 1])
            f = f[:2 * m.shape[0], :2 * m.shape[1]].reshape(m.shape[0], 2, m.shape[1], 2).max(axis=(1, 3))
            keep &= m >= f
        out.append(np.where(keep, m, m - penalty))
    return out


def toy_maps() -> list[np.ndarray]:
    """Two eels. Eel A: stride-16 peak (logit 3.0), a duplicate on the next cell (2.0) and a cross-scale
    double at stride 8 (2.0). Eel B: a short eel at stride 8 only (1.5). Background: -6."""
    s8, s16, s32 = np.full((16, 16), -6.0), np.full((8, 8), -6.0), np.full((4, 4), -6.0)
    s16[3, 3], s16[3, 4] = 3.0, 2.0
    s8[6, 7] = 2.0
    s8[13, 2] = 1.5
    return [s8, s16, s32]


def count_boxes(maps: list[np.ndarray], conf: float = 0.25) -> int:
    thr = np.log(conf / (1 - conf))  # sigmoid(logit) >= conf
    return int(sum((m >= thr).sum() for m in maps))


# ---- 3. two heads: weight average vs output average ------------------------
def silu(x: np.ndarray) -> np.ndarray:
    return x / (1 + np.exp(-x))


def head(p: dict, x: np.ndarray) -> np.ndarray:
    return p["w2"] @ silu(p["w1"] @ x + p["b1"][:, None]) + p["b2"][:, None]


def make_head(rng: np.random.Generator, c: int = 16, h: int = 16) -> dict:
    return {
        "w1": rng.normal(size=(h, c)),
        "b1": rng.normal(size=h),
        "w2": rng.normal(size=(1, h)),
        "b2": rng.normal(size=1),
    }


def widen(a: dict, b: dict) -> dict:
    """One head of double width whose output is exactly 0.5 * head_a + 0.5 * head_b."""
    return {
        "w1": np.vstack([a["w1"], b["w1"]]),
        "b1": np.concatenate([a["b1"], b["b1"]]),
        "w2": np.hstack([0.5 * a["w2"], 0.5 * b["w2"]]),
        "b2": 0.5 * (a["b2"] + b["b2"]),
    }


def blend(a: dict, b: dict, t: float) -> dict:
    return {k: (1 - t) * a[k] + t * b[k] for k in a}


def merge_check() -> tuple[float, float, int]:
    """Max error vs the mean of two heads' outputs: widened head, weight average; and the widened hidden width."""
    rng = np.random.default_rng(1)
    a, b = make_head(rng), make_head(rng)
    x = rng.normal(size=(16, 500))
    mean_out = 0.5 * head(a, x) + 0.5 * head(b, x)
    wide = widen(a, b)
    return (
        float(np.abs(head(wide, x) - mean_out).max()),
        float(np.abs(head(blend(a, b, 0.5), x) - mean_out).max()),
        wide["w1"].shape[0],
    )


# ---- 4. WiSE-FT and the mask grid ------------------------------------------
def wise(base: dict, ft: dict, alpha: float) -> dict:
    return {k: base[k] + alpha * (ft[k] - base[k]) for k in base}


def band_iou(width: int, shift: int) -> float:
    a = np.zeros(100, bool)
    a[40:40 + width] = True
    return float((a & np.roll(a, shift)).sum() / (a | np.roll(a, shift)).sum())


def main() -> None:
    d = load()
    runs, lg = d["runs"], d["logged"]

    print("== Part A: the tools on synthetic arrays")
    err = prune_check()
    print(f"1. Sliced 1-class head vs row 21 of the 22-class head: max difference {err:.1e}")
    print("   Pruning is exact (to float rounding). Is pass?", err < 1e-12)

    maps = toy_maps()
    before, after = count_boxes(maps), count_boxes(peak_suppress(maps))
    print(f"2. Boxes at conf >= 0.25: before peak suppression {before}, after {after}")
    print("   One box per eel after peak suppression. Is pass?", after == 2)

    err_wide, err_blend, width = merge_check()
    print(f"3. Max error vs the output mean: widened head {err_wide:.1e}, weight average {err_blend:.2f}")
    print("   Widened head is exact. Is pass?", err_wide < 1e-9)
    print("   Weight average is exact. Is pass?", err_blend < 1e-9, "  # False on purpose")
    print(f"   Widened head: hidden width {width} (2 x {width // 2})")

    rng = np.random.default_rng(2)
    base, ft = make_head(rng), make_head(rng)
    w0, w1 = wise(base, ft, 0.0), wise(base, ft, 1.0)
    print(
        "4. WiSE alpha 0 = base and alpha 1 = fine-tuned. Is pass?",
        all(np.array_equal(w0[k], base[k]) and np.allclose(w1[k], ft[k]) for k in base),
    )
    cell = 640 / 512 * 4
    print(f"   Mask grid cell at 512 px input = {cell:.1f} frame px; a 9 px eel is {9 / cell:.1f} cells wide")
    print(f"   IoU of a 9 px band shifted by 1 px: {band_iou(9, 1):.2f}, by 2 px: {band_iou(9, 2):.2f}")

    print("\n== Part B: the study")
    print(
        f"frames {lg['frames']}, labels {lg['labels']}, held-out frames {lg['heldout_frames']}; "
        f"parameters {lg['params_original'] / 1e6:.2f}M (v2: {lg['params_v2'] / 1e6:.2f}M)"
    )
    print("class biases (stride 8, 16, 32): dead rows", lg["class_bias"]["dead"], "eel row", lg["class_bias"]["eel"])
    dg = d["diag"]
    print(
        f"diagnostic: {dg['found']} of {dg['eels']} eels found "
        f"({dg['found'] / dg['eels']:.1%}); by stride {dg['by_stride']}"
    )
    print(
        f"  misses {dg['misses']}: score failures {dg['score_failures']} "
        f"(median best score {dg['median_miss_score']}), "
        f"localization failures {dg['loc_failures']}"
    )
    for size, v in dg["imgsz"].items():
        print(
            f"  input {size} px: recall under 60 px {v['under60']:.0%}, 60-100 px {v['60to100']:.0%}, "
            f"all {v['all']:.1%}"
        )
    for r in d["recall_by_length"]:
        hi = "+" if r["hi"] >= 10**6 else f"-{r['hi']}"
        print(f"  length {r['lo']}{hi} px: {r['found']}/{r['eels']} = {r['recall']:.1%}")
    print("stride-8 edits (name, eels recovered of", lg["stride8_targets"], ", extra false positives):")
    for name, rec, fp in lg["stride8_edits"]:
        print(f"  {name:14s} +{rec:3d}  +{fp:5d}  ({fp / rec:.0f} false positives per eel)")
    print("probe ceiling (short-eel recall %, FP current row, FP best row held-out, FP best row own half):")
    for row in lg["probe"]:
        print("  ", row, f" own half looks {row[2] / row[3]:.1f}x better")
    print("robustness, box mAP50:", d["robustness"])
    print("mask ceiling (perfect shape, half a grid cell off):")
    for c in d["mask_ceiling"]:
        print(f"  {c['imgsz']} px, grid {c['grid']}: mean IoU {c['mean_iou']:.2f}, >=0.75: {c['ge75']}%")

    c = lg["census"]
    fps = c["dup"] + c["misloc"] + c["background"]
    print(
        f"\ntrue positives {c['tp']}; false positives {fps}: duplicates {c['dup']} "
        f"({c['dup'] / fps:.0%}; same scale {c['dup_same']}, "
        f"{c['dup_adjacent']} of them on the next cell; cross-scale {c['dup_cross']}), mislocalized {c['misloc']}, "
        f"background {c['background']}; duplicate score ~ {c['dup_score_ratio']} x its true positive"
    )
    print("after peak suppression:", lg["after_peak"])
    print("head weight blend:", lg["head_blend"], " widened merge max error:", lg["merge_max_err"])

    def show(title: str, tags: list[tuple[str, str]]) -> None:
        print(f"\n{title}")
        print(f"  {'':44s} {'boxP':>6} {'boxR':>6} {'bAP50':>6} {'bAP':>6} {'mAP50':>6} {'mAP':>6} {'fit':>7}")
        for name, t in tags:
            r = runs[t]
            print(
                f"  {name:44s} {r['box']['P']:6.3f} {r['box']['R']:6.3f} "
                f"{r['box']['mAP50']:6.3f} {r['box']['mAP']:6.3f} "
                f"{r['mask']['mAP50']:6.3f} {r['mask']['mAP']:6.3f} {r['fitness']:7.4f}"
            )

    show(
        "Full set (all 4,986 frames, training fit)",
        [
            ("deployed", "full_o2o_e2e"),
            ("peak suppression", "full_peak2"),
            ("peak + NMS 0.5", "full_peak2_post50"),
            ("class mean of both heads + peak + NMS 0.5", "full_mixmean_peak2_nms50"),
            ("eel_v2 + peak + NMS 0.5", "full_eelv2_peak2_post50"),
        ],
    )
    show(
        "Held-out half: the v2 ladder",
        [
            ("deployed", "late_base_o2o_e2e"),
            ("+ NMS 0.5", "late_base_o2o_post50"),
            ("+ peak suppression", "late_base_peak2"),
            ("+ class mean of both heads + NMS 0.5", "late_base_mixmean_peak2_nms50"),
            ("+ WiSE-FT alpha 0.25 (= eel_v2)", "late_eelv2_peak2_post50"),
        ],
    )
    show(
        "Held-out half: fine-tune and WiSE-FT (peak suppression)",
        [
            ("base (alpha 0)", "late_base_peak2"),
            ("alpha 0.15", "late_wise_a0.15_peak2"),
            ("alpha 0.25", "late_wise_a0.25_peak2"),
            ("alpha 0.35", "late_wise_a0.35_peak2"),
            ("alpha 0.5", "late_wise_a0.5_peak2"),
            ("alpha 0.75", "late_wise_a0.75_peak2"),
            ("fine-tuned (alpha 1)", "late_ft6_peak2"),
            ("fine-tuned backbone only", "late_wise_bb_a1_peak2"),
            ("fine-tuned box branches only", "late_wise_box_a1_peak2"),
            ("fine-tuned class branches only", "late_wise_cls_a1_peak2"),
            ("head-only fine-tune", "late_fthead6_peak2"),
        ],
    )
    show(
        "Held-out half: masks scored at full resolution",
        [
            ("deployed", "late_native_base_o2o_e2e"),
            ("eel_v2", "late_native_eelv2_peak2_post50"),
            ("mask fine-tune, raw", "late_native_ftmask1_peak2_post50"),
            ("mask fine-tune, WiSE 0.5, no merge", "late_native_wise_mask1_a0.5_peak2_post50"),
            ("mask fine-tune, WiSE 0.25 + merge", "late_native_mask1merged_a0.25_peak2_post50"),
            ("mask fine-tune, WiSE 0.5 + merge", "late_native_mask1merged_peak2_post50"),
            ("mask fine-tune, WiSE 0.75 + merge", "late_native_mask1merged_a0.75_peak2_post50"),
        ],
    )

    print("\nDecoding variants on the subset (every fifth frame), fitness:")
    for name, t in [
        ("one-to-one head, as deployed", "sub_o2o_e2e"),
        ("one-to-many head + NMS 0.7", "sub_o2m_nms"),
        ("one-to-many head + NMS 0.5", "sub_o2m_nms50"),
        ("class mean of both heads + NMS 0.5", "sub_mixmean_nms50"),
        ("one-to-one + NMS 0.5", "sub_o2o_post50"),
        ("peak suppression", "sub_peak2"),
        ("peak, 5x5 window", "sub_peak2_k5"),
        ("peak, penalty 8 / 4 / 2", ("peak2_pen8", "peak2_pen4", "peak2_pen2")),
        ("peak + input 576 px", "sub_peak2_imgsz576"),
        ("class mean + peak + NMS 0.5", "sub_mixmean_peak2_nms50"),
        ("flip TTA, one-to-many + NMS 0.5", "sub_o2m_nms50_flip"),
        ("flip TTA, class mean + peak + NMS", "sub_mixmean_peak2_nms50_flip"),
        ("box voting, one-to-many", "vote_o2m_v60_m"),
        ("box voting, best variant", "vote_clso2o_v60_m"),
    ]:
        tags = t if isinstance(t, tuple) else (t,)
        print(f"  {name:36s} " + " / ".join(f"{runs[x]['fitness']:.4f}" for x in tags))

    v2, dep = runs["late_eelv2_peak2_post50"], runs["late_base_o2o_e2e"]
    print(
        f"\nheld-out gain: recall {dep['box']['R']:.3f} -> {v2['box']['R']:.3f}, "
        f"box mAP50-95 {dep['box']['mAP']:.3f} -> "
        f"{v2['box']['mAP']:.3f}, mask mAP50-95 {dep['mask']['mAP']:.3f} -> {v2['mask']['mAP']:.3f}"
    )
    print("temporal re-scoring:", lg["temporal"])
    bd, rf = lg["boundary"], lg["refined"]
    print("\nmask boundaries, held-out eels", bd["eels"])
    for k, name in [
        ("iou", "mean IoU"),
        ("bf1", "boundary F at 1 px"),
        ("bf2", "boundary F at 2 px"),
        ("absd", "mean boundary error px"),
        ("sd", "signed boundary error px"),
        ("area", "area / label area"),
    ]:
        print(f"  {name:26s} YOLO {bd[k]:+.3f}  refined {rf[k]:+.3f}")
    print(f"  missed eel pixels outside the predicted box: {bd['fn_outside_box']:.0%}")
    print(f"  label polygons: median {bd['vertices']} vertices per eel, one every {bd['vertex_px']} px")
    print("  failed fixes (mean IoU):", lg["failed_mask_fixes"])
    print(
        f"  refiner: {rf['params']} parameters, {rf['fit_eels']} training eels, {rf['epochs']} epochs, "
        f"about {rf['ms_per_eel']} ms per eel on one CPU thread"
    )
    print("  wrapper, 29 eels:", lg["wrapper29"])


if __name__ == "__main__":
    main()
