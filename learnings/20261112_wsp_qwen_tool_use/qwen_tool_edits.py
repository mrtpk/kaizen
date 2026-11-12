"""Weight-Space Programming - Qwen3.5-0.8B tool calls: the checks and the numbers of the post.

Part A runs the ideas of the study on small synthetic arrays (NumPy only):
  1. the acceptance rule: how often "every edit seed above every base seed" passes by chance
  2. the BFCL-V4 overall score from its parts, and the ceiling of a perfect single-turn model
  3. a rank-1 edit of the last down projection: dense, and its effect on the logits
  4. a programmed neuron: a ridge gate that opens only at "decision AND thinking" positions
  5. a read that learns the data source instead of the decision
Part B prints the study's results from study_results.json (exported from the score tables by
analysis/export_blog_numbers.py in the wsp-qwen-tools repository), and a few numbers that exist
only in the study's findings logs (LOGGED, with the finding ids).

Every number in the post comes from this script.
Run:  uv run --with numpy python qwen_tool_edits.py
"""

import json
from math import comb
from typing import Any
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

# Numbers that exist only in the findings logs of the study (v1: F.., v2: G..), copied exactly.
LOGGED: dict[str, Any] = {
    "p_call_auroc_nothink": 0.914,  # F6: p(<tool_call>) at the first assistant token
    "p_call_auroc_think": 0.980,  # F13: after </think>
    "text_instead_of_call_base": (685, 699),  # F43: of 2,501 call entries, non-thinking, two seeds
    "text_instead_of_call_e4b": (332, 328),  # F43
    "call_entries": 2501,
    "read_auroc_model": 0.839,  # G18 table: ToolACE held-out, non-thinking, model p_call
    "read_auroc_ridge": 0.947,  # G18: LDA read, lambda 3, layer-23 MLP input
    "run1_simple_python_fail": (325, 400, 107),  # G11: no-call failures of 400, edit vs base
    "additive_gate_open": (72, 91),  # G23: % of non-decision positions where the v5 gate opens
    "ridge_gate_held_out": (2.58, -3.38),  # G24: min at thinking decisions, max at every other kind
    "v6_simple_python_decoder_fail": (
        28,
        92,
    ),  # G25: simple_python ast_decoder failures, thinking, two seeds summed, base -> v6
    "codex_tasks": 16,  # G14
    "codex_max_pass": 15,  # G14: no model passes more than 15 %
    "neuron_slots": (1488, 1152),  # v7 neurons (G26)
    "hidden": 1024,
    "mlp": 3584,
}


def load() -> dict:
    return json.loads((HERE / "study_results.json").read_text())


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


# ---- 1. acceptance rule ------------------------------------------------------
def chance_pass(n: int) -> float:
    """n edit seeds and n base seeds from one distribution: P(all edit seeds above all base seeds)."""
    return 1 / comb(2 * n, n)


def chance_pass_mc(n: int, trials: int = 200_000) -> float:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(trials, 2 * n))
    return float(np.mean(x[:, :n].min(1) > x[:, n:].max(1)))


# ---- 2. BFCL-V4 overall --------------------------------------------------------
WEIGHTS = {"non_live_ast": 0.1, "live": 0.1, "irrelevance": 0.1, "multi_turn": 0.3}


def overall(s: dict) -> float:
    """Overall = 10 % non-live + 10 % live + 10 % irrelevance + 30 % multi-turn + 40 % agentic.
    Agentic is the mean of memory and web search; web search is not run, so it counts as 0."""
    return sum(w * s[k] for k, w in WEIGHTS.items()) + 0.4 * (s["memory"] + 0.0) / 2


def perfect_single_turn(s: dict) -> float:
    return overall(dict(s, non_live_ast=100.0, live=100.0, irrelevance=100.0))


# ---- 3. rank-1 edit of the last down projection --------------------------------
def rank1_check() -> tuple[float, int, int]:
    rng = np.random.default_rng(1)
    d, m, v, n = 32, 112, 50, 200  # hidden, MLP width, vocabulary, positions
    W = rng.normal(size=(d, m)) / np.sqrt(m)  # down_proj
    U = rng.normal(size=(v, d)) / np.sqrt(d)  # output layer
    h = rng.normal(size=(n, m))  # down_proj inputs
    x = rng.normal(size=(n, d))  # residual before the MLP output
    rho = rng.normal(size=m)
    rho /= np.linalg.norm(rho)  # read
    u = rng.normal(size=d)
    u /= np.linalg.norm(u)  # write
    k = 5.0
    W2 = W - k * np.outer(u, rho)
    before = (x + h @ W.T) @ U.T
    after = (x + h @ W2.T) @ U.T
    predicted = -k * np.outer(h @ rho, U @ u)  # the change, if nothing sits in between
    err = float(np.abs((after - before) - predicted).max())
    changed = int(np.sum(W2 != W))

    return err, changed, d * m


# ---- 4. programmed neuron: the gate ----------------------------------------------
def gate_check() -> dict:
    """Positions of four kinds: (decision?, thinking?). The neuron must open only at decision AND thinking."""
    rng = np.random.default_rng(2)
    d = 64
    dec, think, const, both = (v / np.linalg.norm(v) for v in rng.normal(size=(4, d)))

    def positions(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        a = rng.integers(0, 2, n)  # decision position?
        b = rng.integers(0, 2, n)  # thinking mode?
        # every position carries a constant direction (a residual stream has a large mean); it acts as the bias
        # thinking decisions also carry a direction of their own; a linear gate needs it, because a sum of
        # the two cues cannot make an AND with a margin
        x = (
            0.25 * rng.normal(size=(n, d))
            + 6 * const
            + np.outer(6 * a, dec)
            + np.outer(6 * b, think)
            + np.outer(4 * a * b, both)
        )
        return x, a, b

    xf, af, bf = positions(4000)
    xt, at, bt = positions(4000)
    target = np.where((af == 1) & (bf == 1), 4.0, -8.0)  # +4 at the wanted kind, -8 at every other kind
    lam = 3.0
    g = np.linalg.solve(xf.T @ xf + lam * np.eye(d), xf.T @ target)  # ridge, no intercept
    add = dec + think  # the additive gate: "decision + mode"
    want = (at == 1) & (bt == 1)
    ridge_z, add_z = xt @ g, xt @ add
    return {
        "ridge_min_wanted": float(ridge_z[want].min()),
        "ridge_max_other": float(ridge_z[~want].max()),
        "ridge_open_other": float(np.mean(ridge_z[~want] > 0)),
        "add_open_other": float(np.mean(add_z[~want] > 0)),
        "add_open_one_cue": float(np.mean(add_z[(at + bt) == 1] > 0)),
        "silu_wanted": float(np.median(silu(ridge_z[want]))),
        "silu_other_max": float(np.abs(silu(ridge_z[~want])).max()),
    }


def silu(z: np.ndarray) -> np.ndarray:
    return z / (1 + np.exp(-z))


# ---- 5. a read that learns the source ----------------------------------------------
def auroc(score: np.ndarray, label: np.ndarray) -> float:
    r = score.argsort().argsort() + 1.0
    p = label == 1
    return float((r[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * (~p).sum()))


def source_check() -> tuple[float, float, float]:
    """'no call' rows all from source A, 'call' rows all from source B. The source is easier to read."""
    rng = np.random.default_rng(3)
    d = 64
    need, style = rng.normal(size=d), rng.normal(size=d)
    need, style = need / np.linalg.norm(need), style / np.linalg.norm(style)

    def rows(n: int, call: np.ndarray, src: np.ndarray) -> np.ndarray:
        return rng.normal(size=(n, d)) + np.outer(0.8 * (2 * call - 1), need) + np.outer(3.0 * (2 * src - 1), style)

    def ridge(x: np.ndarray, y: np.ndarray, lam: float = 3.0) -> np.ndarray:
        xc = x - x.mean(0)
        return np.linalg.solve(xc.T @ xc + lam * np.eye(d), xc.T @ (y - y.mean()))

    n = 600
    y = rng.integers(0, 2, n)
    confounded = ridge(rows(n, y, y), y)  # label == source
    yc = rng.integers(0, 2, n)
    clean = ridge(rows(n, yc, np.zeros(n, int)), yc)  # one source, both labels
    yt = rng.integers(0, 2, n)
    test = rows(n, yt, np.ones(n, int))  # a third setting: one source, both labels
    fit_auc = auroc(rows(n, y, y) @ confounded, y)
    return fit_auc, auroc(test @ confounded, yt), auroc(test @ clean, yt)


# ---- part B helpers -------------------------------------------------------------
def seeds(dat: dict, model: str, mode: str, key: str = "overall") -> list[float]:
    return [s[key] for s in dat["models"][model].get(mode, [])]


def m(dat: dict, model: str, mode: str, key: str = "overall", n: int | None = None) -> float:
    xs = seeds(dat, model, mode, key)
    return mean(xs[:n] if n else xs)


def passes(dat: dict, model: str, mode: str) -> bool:
    return min(seeds(dat, model, mode)) > max(seeds(dat, "base", mode))


def main() -> None:
    print("=== Part A: the ideas on synthetic arrays ===")
    print("\n1. acceptance rule: every edit seed above every base seed, with no real effect")
    for n in (2, 3, 4):
        exact, mc = chance_pass(n), chance_pass_mc(n)
        print(f"  {n} vs {n} seeds: 1 in {round(1 / exact)} (exact {exact:.4f}, simulated {mc:.4f})")
    print(
        f"  simulation agrees with 1/C(2n,n). Is pass? "
        f"{all(abs(chance_pass(n) - chance_pass_mc(n)) < 0.003 for n in (2, 3, 4))}"
    )

    dat = load()
    print("\n2. BFCL-V4 overall from its parts")
    errs = [abs(overall(s) - s["overall"]) for mdl in dat["models"].values() for runs in mdl.values() for s in runs]
    print(
        f"  {len(errs)} runs, largest difference to the reported overall {max(errs):.3f}. Is pass? {max(errs) < 0.01}"
    )
    base_t = dat["models"]["base"]["think"]
    avg = {k: mean([s[k] for s in base_t]) for k in base_t[0]}
    print(
        f"  base thinking, mean of 3 seeds: overall {overall(avg):.2f}; non-live {avg['non_live_ast']:.2f}, live "
        f"{avg['live']:.2f},"
        f" irrelevance {avg['irrelevance']:.2f}, multi-turn {avg['multi_turn']:.2f}, memory {avg['memory']:.2f}"
    )
    single = 0.1 * (avg["non_live_ast"] + avg["live"] + avg["irrelevance"])
    print(
        f"  points by part: non-live {0.1 * avg['non_live_ast']:.2f}, live {0.1 * avg['live']:.2f},"
        f" irrelevance {0.1 * avg['irrelevance']:.2f}, multi-turn {0.3 * avg['multi_turn']:.2f} of 30,"
        f" memory {0.2 * avg['memory']:.2f} of 20, web search 0 of 20"
    )
    print(
        f"  single-turn parts give {single:.2f} of the {overall(avg):.2f} points; multi-turn "
        f"{0.3 * avg['multi_turn']:.2f};"
        f" memory {0.2 * avg['memory']:.2f}"
    )
    print(
        f"  perfect single-turn (non-live, live, irrelevance = 100): overall {perfect_single_turn(avg):.2f}"
        f" (+{perfect_single_turn(avg) - overall(avg):.2f}); goal 60 needs +{60 - overall(avg):.2f}."
        f" Is the goal out of reach by single-turn alone? {perfect_single_turn(avg) < 60}"
    )
    print(f"  over-calls in thinking mode (100 - irrelevance): {100 - avg['irrelevance']:.1f} %")

    print("\n3. rank-1 edit W' = W - k u rho^T of the last down projection (toy: hidden 32, MLP 112)")
    err, changed, total = rank1_check()
    print(f"  logit change = -k (rho . h) (U u), no norm in between: max error {err:.1e}. Is pass? {err < 1e-9}")
    print(f"  changed values {changed} of {total}: a rank-1 edit is dense. Is pass? {changed == total}")
    hd, mlp = LOGGED["hidden"], LOGGED["mlp"]
    print(
        f"  in Qwen3.5-0.8B: rank 1 on {hd} x {mlp} = {hd * mlp:,} values; one programmed neuron = 3 x {hd} = "
        f"{3 * hd:,};"
        f" two neurons = {6 * hd:,}; ratio {hd * mlp / (6 * hd):.0f}"
    )

    print("\n4. programmed neuron: a gate for 'decision AND thinking'")
    gc = gate_check()
    print(
        f"  ridge gate (targets +4 / -8, no intercept): min at wanted positions {gc['ridge_min_wanted']:.2f},"
        f" max elsewhere {gc['ridge_max_other']:.2f}; open elsewhere {gc['ridge_open_other']:.1%}."
        f" Is pass? {gc['ridge_min_wanted'] > 0 > gc['ridge_max_other']}"
    )
    print(
        f"  additive gate (decision + mode): open at {gc['add_open_other']:.1%} of the other positions,"
        f" {gc['add_open_one_cue']:.1%} where only one cue is present. A clean AND? {gc['add_open_other'] < 0.01}"
        "  (False on purpose: an additive gate is an OR)"
    )
    print(
        f"  silu(gate): median {gc['silu_wanted']:.2f} at wanted positions, at most {gc['silu_other_max']:.4f} "
        f"elsewhere"
    )

    print("\n5. a read that learns the source (label == source)")
    fit_auc, conf_auc, clean_auc = source_check()
    print(
        f"  confounded read: AUROC {fit_auc:.3f} on its own data, {conf_auc:.3f} on one-source data."
        f" Does it read the decision? {conf_auc > 0.7}  (False on purpose)"
    )
    print(
        f"  read fitted on one source with both labels: AUROC {clean_auc:.3f} on the same test. Is pass? "
        f"{clean_auc > 0.7}"
    )

    print("\n=== Part B: the study's results (BFCL-V4, web search not run) ===")
    bt, bn = m(dat, "base", "think"), m(dat, "base", "nothink")
    print(
        f"base seeds: thinking {seeds(dat, 'base', 'think')} (mean {bt:.2f}); non-thinking "
        f"{seeds(dat, 'base', 'nothink')} (mean {bn:.2f})"
    )
    print(f"seed spread, base thinking: {max(seeds(dat, 'base', 'think')) - min(seeds(dat, 'base', 'think')):.2f}")
    cols = ("overall", "non_live_ast", "live", "multi_turn", "memory", "irrelevance")
    for mode in ("think", "nothink"):
        print(f"\n{mode}: mean of seeds (n)  " + "  ".join(f"{c:>12s}" for c in cols) + "   gain   rule")
        for lab in ("base", "rikiyat", "e1b", "e4b", "e5c", "run1", "v4", "v6", "v7", "v8", "v9", "v10"):
            if mode not in dat["models"][lab]:
                continue
            n = len(seeds(dat, lab, mode))
            vals = "  ".join(f"{m(dat, lab, mode, c):12.2f}" for c in cols)
            gain = m(dat, lab, mode) - (bt if mode == "think" else bn)
            rule = "" if lab == "base" else ("pass" if passes(dat, lab, mode) else "fail")
            print(f"  {lab:8s} ({n})          {vals}  {gain:+6.2f}   {rule}  seeds {seeds(dat, lab, mode)}")
    v7t, e4t = m(dat, "v7", "think"), m(dat, "e4b", "think")
    v7n, e4n = m(dat, "v7", "nothink"), m(dat, "e4b", "nothink")
    print(
        f"\nv7 vs E4b: thinking {v7t:.2f} vs {e4t:.2f} ({v7t - e4t:+.2f}), every v7 seed above every E4b seed?"
        f" {min(seeds(dat, 'v7', 'think')) > max(seeds(dat, 'e4b', 'think'))}"
    )
    print(
        f"           non-thinking {v7n:.2f} vs {e4n:.2f} ({v7n - e4n:+.2f}), every E4b seed above every v7 seed?"
        f" {min(seeds(dat, 'e4b', 'nothink')) > max(seeds(dat, 'v7', 'nothink'))}"
    )
    print(
        f"           mean of the two modes: v7 {(v7t + v7n) / 2:.2f}, E4b {(e4t + e4n) / 2:.2f}, base "
        f"{(bt + bn) / 2:.2f}"
    )

    def ast(lab: str) -> float:
        return (m(dat, lab, "think", "non_live_ast") + m(dat, lab, "think", "live")) / 2

    both = []
    for lab in ("rikiyat", "e1b", "e4b", "e5c", "run1", "v4", "v6", "v7", "v9", "v10"):
        di = m(dat, lab, "think", "irrelevance") - m(dat, "base", "think", "irrelevance")
        da = ast(lab) - ast("base")
        both.append(di > 0 and da > 0)
        print(f"  trade-off, thinking, {lab:7s}: irrelevance {di:+6.2f}, AST (mean of non-live and live) {da:+6.2f}")
    print(f"  does any thinking edit raise both irrelevance and AST? {any(both)}")
    print(
        f"  E4b thinking: non-live "
        f"{m(dat, 'e4b', 'think', 'non_live_ast') - m(dat, 'base', 'think', 'non_live_ast'):+.2f},"
        f" live {m(dat, 'e4b', 'think', 'live') - m(dat, 'base', 'think', 'live'):+.2f},"
        f" memory {m(dat, 'e4b', 'think', 'memory') - m(dat, 'base', 'think', 'memory'):+.2f}"
    )
    print(f"           mode-mean difference v7 - E4b: {(v7t + v7n) / 2 - (e4t + e4n) / 2:.2f}")
    print(f"v9 vs v7, thinking: {m(dat, 'v9', 'think') - v7t:+.2f}")
    v7_irr = m(dat, "v7", "think", "irrelevance") - m(dat, "base", "think", "irrelevance")
    v7_mt = m(dat, "v7", "think", "multi_turn") - m(dat, "base", "think", "multi_turn")
    v7_nl = m(dat, "v7", "think", "non_live_ast") - m(dat, "base", "think", "non_live_ast")
    print(
        f"v7 thinking vs base (3 seeds each): irrelevance {v7_irr:+.1f}, multi-turn {v7_mt:+.1f}, non-live AST "
        f"{v7_nl:+.1f}"
    )
    e4 = dat["edits"]["e4b"]["model.language_model.layers.23.mlp.down_proj.weight"]
    print(
        f"E4b: tensor {e4['shape']}, {e4['changed_values']:,} values change in bf16, relative norm "
        f"{e4['relative_norm']}"
    )

    print("\nlogged (findings logs):")
    a, b = LOGGED["text_instead_of_call_base"], LOGGED["text_instead_of_call_e4b"]
    print(
        f"  p_call AUROC: non-thinking first token {LOGGED['p_call_auroc_nothink']}, thinking after </think> "
        f"{LOGGED['p_call_auroc_think']:.3f}"
    )
    print(
        f"  text instead of a call (non-thinking, of {LOGGED['call_entries']:,}): base {a[0]} / {a[1]}, E4b {b[0]} / "
        f"{b[1]}"
        f" ({mean(list(b)) / mean(list(a)):.2f} x)"
    )
    print(f"  read vs model, ToolACE held-out: AUROC {LOGGED['read_auroc_model']} -> {LOGGED['read_auroc_ridge']}")
    f, t, base_f = LOGGED["run1_simple_python_fail"]
    print(f"  run 1, simple_python non-thinking: {f} of {t} fail with no call (base {base_f})")
    print(
        f"  additive gate open at {LOGGED['additive_gate_open'][0]} to {LOGGED['additive_gate_open'][1]} % of "
        f"non-decision positions;"
        f" ridge gate held-out: min {LOGGED['ridge_gate_held_out'][0]} at thinking decisions, max "
        f"{LOGGED['ridge_gate_held_out'][1]} elsewhere"
    )
    print(
        f"  v6 simple_python decoder failures: {LOGGED['v6_simple_python_decoder_fail'][0]} -> "
        f"{LOGGED['v6_simple_python_decoder_fail'][1]}"
    )
    print(f"  Codex: {LOGGED['codex_tasks']} tasks, no model passes more than {LOGGED['codex_max_pass']} %")


if __name__ == "__main__":
    main()
