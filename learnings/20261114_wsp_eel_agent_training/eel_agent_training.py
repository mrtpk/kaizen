"""Weight-Space Programming - Side note: an agent drives the training of the eel detector.

Part A shows, on small synthetic arrays (NumPy only), why an average of two networks that do not share
a channel basis shrinks the activations ("variance collapse"), and why a shared basis does not.
Part B prints the study's results from study_results.json (fold 0 merge and trunk tables, and the
5-fold CV table), and a few numbers that exist only in the study's notes (LOGGED).

Every number in the post comes from this script.
Run:  uv run --with numpy python eel_agent_training.py
"""

import json
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent

# Numbers that exist only in the study's notes and logs, copied exactly.
LOGGED: dict[str, Any] = {
    "frames": 4986,
    "val_per_fold": 1000,  # about 1,000 held-out frames per fold
    "fold_gap": 50,  # frames closer than 50 frame numbers stay in one fold
    "cv_jobs": 11,  # A and B on 5 folds, C on fold 0
    "crash_stopped": 10,  # a server crash stopped 10 of the 11 parallel CV jobs
    "seq_jobs": 10 + 5 + 1,  # resumes + experiments + final run, one at a time, no failure
    "final_hours": 2.17,
    "final_epoch": 150,
    "b_mean_ep146_154": 0.4956,  # held-out mask mAP50-95 of the 5 B folds, mean over epochs 146-154
    "channels": 10520,
    "cv_dead": 0,
    "cv_gamma_small_max": 1,  # BN gamma < 0.01, at most one per CV run
    "prod_dead": 66,
    "prod_gamma_small": 277,
    "fire_cv": (0.564, 0.572),
    "fire_prod": 0.468,
    "freeze_share": 34,  # % of weights in blocks 7-10
    "seed_noise": (0.001, 0.005),  # mask mAP50-95, child_A1 vs child_A2
    "momentum": (0.9, 0.937),  # CV auto vs explicit MuSGD in the trunk experiments
    "probe_frames": 80,
    # firing fingerprint, head drift (dmean) and mean eel confidence, model A_f0
    "fingerprint": {
        "clean": (0.0, 0.778),
        "dark": (0.04, 0.770),
        "bright": (0.04, 0.774),
        "blur": (0.13, 0.758),
        "turbid": (0.18, 0.668),
        "noise": (0.32, 0.640),
        "half scale": (0.43, 0.692),
    },
}


# ---- A. variance collapse ---------------------------------------------------
def collapse_demo(seed: int = 0) -> tuple[float, float, float]:
    """Two layers that compute the same features, one with its channels permuted.

    Returns the activation std of: one layer, the average of the aligned pair, the average of the
    unaligned pair. Inputs and weights are Gaussian, the layer is linear + ReLU.
    """
    rng = np.random.default_rng(seed)
    n, d, c = 4096, 64, 64
    x = rng.standard_normal((n, d))
    w = rng.standard_normal((d, c)) / np.sqrt(d)
    w_perm = w[:, rng.permutation(c)]  # same function, other channel order

    def act(wt: np.ndarray) -> np.ndarray:
        return np.maximum(x @ wt, 0.0)

    return float(act(w).std()), float(act((w + w) / 2).std()), float(act((w + w_perm) / 2).std())


def load() -> dict:
    return json.loads((HERE / "study_results.json").read_text())


def mm(d: dict, key: str) -> float:
    return float(d[key]["mask50-95"])


if __name__ == "__main__":
    one, aligned, unaligned = collapse_demo()
    print(f"std: one layer {one:.3f}, aligned average {aligned:.3f}, unaligned average {unaligned:.3f}")
    print("Aligned average keeps the scale. Is pass?", abs(aligned - one) < 1e-9)
    print("Unaligned average shrinks it by more than 20 %. Is pass?", unaligned < 0.8 * one)

    dat = load()
    cv = dat["cv"]
    for r in ("A", "B"):
        a, b = cv[r]["mask50-95"], cv[r]["mask50"]
        print(f"{r}: mask50-95 {mean(a):.4f} +- {stdev(a):.3f}  mask50 {mean(b):.4f} +- {stdev(b):.4f}")
    wins50 = sum(b > a for a, b in zip(cv["A"]["mask50"], cv["B"]["mask50"]))
    wins5095 = sum(b > a for a, b in zip(cv["A"]["mask50-95"], cv["B"]["mask50-95"]))
    print(f"B ahead on mask50 in {wins50} of 5 folds, on mask50-95 in {wins5095} of 5")
    print("C (batch 128) fold 0:", cv["C_f0"])

    m = dat["merge_f0"]
    print("fold 0 independent runs, mask50-95:")
    for k in ("A_f0", "B_f0", "C_f0", "interp_A_f0_B_f0_0.5", "interp_A_f0_B_f0_0.5_recal", "soup", "soup_recal"):
        print(f"  {k:28s} {mm(m, k):.3f}")
    barrier = mean([mm(m, "A_f0"), mm(m, "B_f0")]) - mm(m, "interp_A_f0_B_f0_0.5")
    print(f"barrier at the midpoint: {barrier:.3f}")

    t = dat["trunk_f0"]
    print("fold 0 children of one 50-epoch trunk, mask50-95:")
    for k in ("child_A1:last", "child_A2:last", "child_B1:last", "interp0.5:child_A2+child_B1",
              "soup3:last", "soup3:last+recal", "child_A2:best", "freezeP5:best"):
        print(f"  {k:28s} {mm(t, k):.3f}")
    ok = all(mm(t, f"interp0.5:{p}") >= min(mm(t, p.split('+')[0] + ':last'), mm(t, p.split('+')[1] + ':last'))
             for p in ("child_A1+child_A2", "child_A1+child_B1", "child_A2+child_B1"))
    print("Every child blend at or above its weaker endpoint. Is pass?", ok)
    print("Trunk soup beats the best single checkpoint A_f0. Is pass?", mm(t, "soup3:last") > mm(m, "A_f0"))
    # Is pass? False: the soup does not add accuracy, which is the finding.
    print("LOGGED:", json.dumps(LOGGED))
