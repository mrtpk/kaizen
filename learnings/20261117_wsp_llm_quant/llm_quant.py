"""Weight-Space Programming - Can an agent quantize an LLM better than a recipe?

Part A is a synthetic illustration (NumPy only, not a study result): 4-bit activations with a few
outlier channels, quantized per token and in groups of 32.
Part B is the greedy bit allocation of the round-3 control, on a toy menu of four linears.
Part C applies the binding verdict rule of round 3 phase A to the study's seed means and stds.
Part D prints the other numbers of the post from study_results.json.

study_results.json is transcribed by hand from the study's report and lab notes, because the result
JSONs stayed on the GPU machine (see its "_source" field).

Every number in the post comes from this script.
Run:  uv run --with numpy python llm_quant.py
"""

import json
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def load() -> dict:
    return json.loads((HERE / "study_results.json").read_text())


# ---- A. per-token vs grouped 4-bit activations (synthetic) ------------------------------------
def quant_int4(x: np.ndarray, group: int) -> np.ndarray:
    """Asymmetric 4-bit min/max fake quantization along the last axis, in groups of `group`."""
    t, c = x.shape
    g = x.reshape(t, c // group, group)
    lo, hi = g.min(-1, keepdims=True), g.max(-1, keepdims=True)
    scale = np.maximum(hi - lo, 1e-12) / 15
    q = np.clip(np.round((g - lo) / scale), 0, 15)
    return (q * scale + lo).reshape(t, c)


def outlier_demo(seed: int = 0) -> tuple[float, float]:
    """Activations: 256 tokens x 1024 channels, 8 channels 100x larger (like the outlier channels of Qwen3-32B).

    Returns the relative error of per-token 4-bit and of 4-bit in groups of 32.
    """
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((256, 1024))
    x[:, rng.choice(1024, 8, replace=False)] *= 100.0

    def rel(xq: np.ndarray) -> float:
        return float(np.linalg.norm(xq - x) / np.linalg.norm(x))

    return rel(quant_int4(x, 1024)), rel(quant_int4(x, 32))


def outlier_demo_ordinary(seed: int = 0) -> tuple[float, float]:
    """The same, but the error on the ordinary channels only (the outlier channels excluded)."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((256, 1024))
    out = rng.choice(1024, 8, replace=False)
    x[:, out] *= 100.0
    keep = np.setdiff1d(np.arange(1024), out)

    def rel(xq: np.ndarray) -> float:
        return float(np.linalg.norm((xq - x)[:, keep]) / np.linalg.norm(x[:, keep]))

    return rel(quant_int4(x, 1024)), rel(quant_int4(x, 32))


# ---- B. greedy allocation on convex hulls ---------------------------------------------------
def lower_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Lower convex hull of (bits, kl) points, sorted by bits. Dominated formats drop out."""
    pts = sorted(points)
    hull: list[tuple[float, float]] = []
    for p in pts:
        while hull and p[1] >= hull[-1][1]:
            hull.pop()  # more bits, no lower KL: never worth it
        while len(hull) >= 2:
            (x1, y1), (x2, y2) = hull[-2], hull[-1]
            if (y2 - y1) * (p[0] - x1) >= (p[1] - y1) * (x2 - x1):
                hull.pop()  # hull[-1] is above the line hull[-2] -> p
            else:
                break
        hull.append(p)
    return hull


def greedy(menus: dict[str, list[tuple[float, float]]], sizes: dict[str, int], budget: float) -> dict[str, float]:
    """Start every linear at its cheapest format. Take the upgrade with the largest KL drop per added
    bit (bits weighted by the size of the linear) while the mean bits per weight stays <= budget."""
    hulls = {k: lower_hull(v) for k, v in menus.items()}
    pos = {k: 0 for k in menus}
    total = sum(sizes.values())

    def mean_bits() -> float:
        return sum(hulls[k][pos[k]][0] * sizes[k] for k in menus) / total

    while True:
        best, best_rate = None, 0.0
        for k in menus:
            if pos[k] + 1 < len(hulls[k]):
                (b0, kl0), (b1, kl1) = hulls[k][pos[k]], hulls[k][pos[k] + 1]
                extra = (b1 - b0) * sizes[k] / total
                if mean_bits() + extra <= budget + 1e-12 and (kl0 - kl1) / extra > best_rate:
                    best, best_rate = k, (kl0 - kl1) / extra
        if best is None:
            return {k: hulls[k][pos[k]][0] for k in menus}
        pos[best] += 1


TOY_MENU = [3.148, 3.297, 4.156, 8.003]  # W3g128, W3g64, W4g128, INT8 (bits per weight incl. scales)
TOY = {  # name: (size in millions of weights, KL at each format) -- invented values, shape only
    "k_proj (small, sensitive)": (5, [8e-4, 6e-4, 1e-4, 1e-6]),
    "q_proj": (42, [1.3e-4, 1.0e-4, 4.1e-5, 8e-7]),
    "down_proj, late": (131, [6.7e-3, 5.0e-3, 1.2e-3, 1.1e-5]),
    "up_proj, early": (131, [2e-4, 1.6e-4, 5e-5, 1e-6]),
}


def toy_allocation(budget: float = 3.75) -> dict[str, float]:
    menus = {k: list(zip(TOY_MENU, kls)) for k, (_, kls) in TOY.items()}
    sizes = {k: s for k, (s, _) in TOY.items()}
    return greedy(menus, sizes, budget)


# ---- C. the binding verdict rule of phase A --------------------------------------------------
def verdict(agent: dict, control: dict) -> tuple[str, dict[str, tuple[float, float]]]:
    """d = control - agent (mean delta-%); s = sqrt((s_a^2 + s_c^2) / 2). Agent wins: d > 2s on both
    sets (and the zero-shot guard). Control wins: -d > 2s on both. Else a tie."""
    ds = {}
    for k in ("wt2", "fw25"):
        (ma, sa), (mc, sc) = agent[k], control[k]
        ds[k] = (mc - ma, sqrt((sa ** 2 + sc ** 2) / 2))
    if all(d > 2 * s for d, s in ds.values()) and agent["zs_worst"] >= 68.94:
        return "AGENT WINS", ds
    if all(-d > 2 * s for d, s in ds.values()):
        return "CONTROL WINS", ds
    return "TIE", ds


def main() -> None:
    r = load()
    r1, r2, pa = r["round1"], r["round2"], r["phase_a"]

    print("== A. 4-bit activations with outlier channels (synthetic illustration)")
    pt, g32 = outlier_demo()
    pt_o, g32_o = outlier_demo_ordinary()
    print(f"relative error, all channels:      per-token {pt:.3f}   group-32 {g32:.3f}")
    print(f"relative error, ordinary channels: per-token {pt_o:.3f}   group-32 {g32_o:.3f}")
    print(f"ratio on ordinary channels: {pt_o / g32_o:.1f}x")
    print("Group-32 error on ordinary channels is at least 5x smaller than per-token. Is pass?", g32_o * 5 < pt_o)

    print("\n== B. greedy allocation on a toy menu (invented KL values)")
    alloc = toy_allocation()
    for k, b in alloc.items():
        print(f"  {k:28s} {b:.3f} bits")
    sizes = {k: s for k, (s, _) in TOY.items()}
    mb = sum(alloc[k] * sizes[k] for k in alloc) / sum(sizes.values())
    print(f"  mean bits per weight {mb:.3f}")
    print("Small, sensitive k_proj gets INT8 first. Is pass?", alloc["k_proj (small, sensitive)"] == 8.003)

    print("\n== C. phase A verdict (rule fixed before any control result)")
    for tgt in ("late34", "late44"):
        v, ds = verdict(pa[tgt]["agent"], pa[tgt]["control"])
        parts = "  ".join(f"{k}: d {d:+.3f}, 2s {2 * s:.3f}" for k, (d, s) in ds.items())
        print(f"{tgt} (W {pa[tgt]['W']}): {parts} -> {v}")
    v34, _ = verdict(pa["late34"]["agent"], pa["late34"]["control"])
    v44, _ = verdict(pa["late44"]["agent"], pa["late44"]["control"])
    print("Both targets are a tie, so claim A is withdrawn. Is pass?", v34 == v44 == "TIE")

    print("\n== D. other numbers of the post")
    w = r2["p1_w3g128_rtn_c4dev"]
    a = r2["p1_a4_per_token_c4dev"]
    print(f"W3g128 RTN, one linear at a time: L63.down {w['L63.down_proj']} of {w['all_448']} "
          f"({100 * w['L63.down_proj'] / w['all_448']:.1f} %); linears above +0.01: {w['linears_above_0.01']} of 448")
    print(f"A4 per token, one linear at a time: L6.gate {a['L6.gate_proj']} of {a['all_448']} "
          f"({100 * a['L6.gate_proj'] / a['all_448']:.1f} %)")
    gap_agent = r1["means"]["agent"]["wt2"] - r1["fp16"]["wt2"]
    gap_spin = r1["means"]["spinquant"]["wt2"] - r1["fp16"]["wt2"]
    print(f"round 1 WikiText-2 gap to FP16: agent {gap_agent:.3f}, SpinQuant {gap_spin:.3f} "
          f"({100 * (1 - gap_agent / gap_spin):.0f} % smaller)")
    print(f"round 1 zero-shot: agent worst {r1['agent_range']['zs'][0]} vs best baseline {r1['best_baseline']['zs']}, "
          f"needed +{r1['margin']['zs']} -> margin {r1['agent_range']['zs'][0] - r1['best_baseline']['zs']:.2f}")
    m = r2["bf16"]
    print(f"round 2 +4% bar: WikiText-2 <= {m['wt2'] * 1.04:.4f}, fw25test <= {m['fw25'] * 1.04:.4f}")
    q = next(b for b in r2["baselines"] if b["name"].startswith("QuaRot"))
    o = next(f for f in r2["finals"] if f["name"] == "ours_W4A4KV4_g32")
    print(f"W4A4KV4 WikiText-2: agent +{o['wt2']} %, QuaRot-style +{q['wt2']} % ({q['wt2'] / o['wt2']:.1f}x)")
    s = r2["speed_b1_tok_s"]
    print(f"decode b1 tok/s: bf16 {s['bf16']}, FP8 {s['fp8_w8a8']}, own W4A8 {s['w4a8_own']} "
          f"(bound {s['w4a8_bound']}, "
          f"predicted {s['w4a8_predicted'][0]}-{s['w4a8_predicted'][1]})")
    c = pa["control_late34_counts"]
    kv8 = c["k"][3] + c["v"][3]
    print(f"control late34: INT8 in {kv8} of 128 k/v linears; "
          f"upgrades in layers 0-33: {pa['control_upgrades_layers_0_33']}; L63.down stays W4")


if __name__ == "__main__":
    main()
