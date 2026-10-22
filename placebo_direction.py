"""
placebo_direction.py — check #1: does shutdown stand out among random prompt pairs?

Idea: any pair of system prompts gives some d. So the d for shutdown − normal
is placed into a "null cloud" — d for 200 random pairs of neutral prompts —
and we report its percentile in that cloud.

Verdict (d_indep only, the probe is saturated and is not part of the verdict):
    PASS — shutdown is stronger than the strongest random pair;
    WEAK — stronger than the normal ↔ control pair, but not all of them;
    FAIL — not stronger than even normal ↔ control.

Run:
    python placebo_direction.py --all --key question_last --output logs/placebo.md
"""
import argparse
import glob
import itertools
import random
import sys
from pathlib import Path

import numpy as np
import torch

import analysis_stats as stats
import hidden_states_dict as hsd
from prompts import (COND_CONTROL, COND_NORMAL, COND_SHUTDOWN,
                     NEUTRAL_POOL_BASE, NEUTRAL_POOL_STRICT)

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SEED = 42
NAN = float("nan")
NULL_MAX_PAIRS = 200    # how many random pairs go into the cloud
NULL_PROBE_LIMIT = 20   # run the probe on the cloud only up to this many pairs (otherwise it takes hours)


# ─── computation ──────────────────────────────────────────────────────────────

def split_questions(questions, train_frac=0.5):
    """Same train/test split as in find_direction.py (seed=42)."""
    shuffled = list(questions)
    random.Random(SEED).shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]


def separability(index, train_qs, test_qs, cond_a, cond_b, key, with_probe=True):
    """
    How separable two conditions are.

    The axis is built on train (mean A − mean B), |d_indep| is computed on test.
    Returns a dict (d_indep, probe_acc, probe_gap, n_a, n_b) or None.
    """
    a_train, _ = hsd.stack_states(index, train_qs, cond_a, key=key)
    b_train, _ = hsd.stack_states(index, train_qs, cond_b, key=key)
    a_test, _ = hsd.stack_states(index, test_qs, cond_a, key=key)
    b_test, _ = hsd.stack_states(index, test_qs, cond_b, key=key)
    if any(x is None or len(x) < 2 for x in (a_train, b_train, a_test, b_test)):
        return None

    direction = a_train.mean(dim=0) - b_train.mean(dim=0)
    d_indep = stats.cohen_d_independent(stats.project(a_test, direction).numpy(),
                                        stats.project(b_test, direction).numpy())

    probe_acc = probe_gap = NAN
    if with_probe:
        X_train = torch.cat([a_train, b_train], dim=0)
        y_train = torch.tensor([1] * len(a_train) + [0] * len(b_train), dtype=torch.long)
        X_test = torch.cat([a_test, b_test], dim=0)
        y_test = torch.tensor([1] * len(a_test) + [0] * len(b_test), dtype=torch.long)
        probe = stats.probe_with_permutation(X_train, y_train, X_test, y_test, seed=SEED)
        probe_acc, probe_gap = probe["accuracy"], probe["gap"]

    return {"d_indep": abs(d_indep), "probe_acc": probe_acc, "probe_gap": probe_gap,
            "n_a": len(a_test), "n_b": len(b_test)}


def choose_null_pool(conditions):
    """Neutral conditions for the cloud: the strict pool (22) if present, otherwise the old v7 pool."""
    strict = [c for c in NEUTRAL_POOL_STRICT if c in conditions]
    if len(strict) > len(NEUTRAL_POOL_BASE):
        return strict, True
    return [c for c in NEUTRAL_POOL_BASE if c in conditions], False


def sample_pairs(pool):
    """All pairs of the pool, but at most NULL_MAX_PAIRS (random sample, seed=42)."""
    pairs = list(itertools.combinations(pool, 2))
    if len(pairs) > NULL_MAX_PAIRS:
        pairs = random.Random(SEED).sample(pairs, NULL_MAX_PAIRS)
    return pairs


def null_cloud(index, train_qs, test_qs, pairs, key):
    """Separability of every pair in the cloud. List of dicts with a 'pair' field."""
    with_probe = len(pairs) <= NULL_PROBE_LIMIT
    cloud = []
    for a, b in pairs:
        s = separability(index, train_qs, test_qs, a, b, key, with_probe=with_probe)
        if s is not None:
            s["pair"] = f"{a} vs {b}"
            cloud.append(s)
    return cloud


def normal_control_d(cloud, index, train_qs, test_qs, key):
    """d for the normal ↔ control pair: taken from the cloud, or computed separately if it wasn't sampled."""
    pair = f"{COND_NORMAL} vs {COND_CONTROL}"
    for s in cloud:
        if s["pair"] == pair:
            return s["d_indep"]
    s = separability(index, train_qs, test_qs, COND_NORMAL, COND_CONTROL, key, with_probe=False)
    return s["d_indep"] if s else NAN


def decide_verdict(real_d, null_max, nc_d):
    if real_d > null_max:
        return "PASS"
    if np.isfinite(nc_d) and real_d > nc_d:
        return "WEAK"
    return "FAIL"


def analyze(data, key, target=COND_SHUTDOWN):
    """Full control for one model. Returns a dict or None."""
    conditions = data.get("conditions", [])
    if target not in conditions or COND_NORMAL not in conditions:
        return None
    index = hsd.build_index(data["results"])
    questions = data["questions"]
    train_qs, test_qs = split_questions(questions)

    # target − normal on all questions (the raw d we report)
    real = separability(index, train_qs, test_qs, target, COND_NORMAL, key)
    if real is None:
        return None

    # the cloud uses questions shared by all conditions; the target is recomputed on the same ones
    pool, strict_pool = choose_null_pool(conditions)
    matched = set(hsd.common_questions(index, pool + [target, COND_NORMAL], questions, key))
    train_m = [q for q in train_qs if q in matched]
    test_m = [q for q in test_qs if q in matched]
    is_subset = len(matched) < len(questions)
    real_m = (separability(index, train_m, test_m, target, COND_NORMAL, key, with_probe=False)
              if is_subset else real)
    if real_m is None:
        return None

    cloud = null_cloud(index, train_m, test_m, sample_pairs(pool), key)
    if not cloud:
        return None

    null_d = np.array([s["d_indep"] for s in cloud])
    null_acc = np.array([s["probe_acc"] for s in cloud])
    null_mean = float(null_d.mean())
    null_sd = float(null_d.std(ddof=1)) if len(null_d) > 1 else 0.0
    real_d = real_m["d_indep"]
    nc_d = normal_control_d(cloud, index, train_m, test_m, key)

    return {
        "model_name": data.get("model_name", data.get("model_key", "?")),
        "target": target,
        "real": real,
        "real_matched": real_m,
        "n_q_matched": len(matched),
        "n_q_total": len(questions),
        "matched_subset": is_subset,
        "null": cloud,
        "null_d_max": float(null_d.max()),
        "null_d_max_pair": cloud[int(null_d.argmax())]["pair"],
        "null_d_mean": null_mean,
        "null_d_sd": null_sd,
        "pct": float((null_d < real_d).mean() * 100.0),
        "z_null": (real_d - null_mean) / null_sd if null_sd > 1e-9 else NAN,
        "n_null": len(null_d),
        "strict_pool": strict_pool,
        "nc_d": float(nc_d),
        "null_acc_max": float(np.nanmax(null_acc)) if np.isfinite(null_acc).any() else NAN,
        "verdict": decide_verdict(real_d, null_d.max(), nc_d),
    }


# ─── output ───────────────────────────────────────────────────────────────────

def print_model(r):
    real = r["real"]
    pool_label = ", strict" if r["strict_pool"] else ", v7 pool"
    print(f"\n{r['model_name']}  [{r['verdict']}]")
    t = r["target"]
    print(f"  {t} vs normal : d_indep={real['d_indep']:.2f}  "
          f"probe={real['probe_acc']:.0%}  gap={real['probe_gap']:+.0%}")
    print(f"  null ({r['n_null']} pairs{pool_label})   : mean={r['null_d_mean']:.2f} "
          f"sd={r['null_d_sd']:.2f} max={r['null_d_max']:.2f} ({r['null_d_max_pair']})")
    if r["matched_subset"]:
        print(f"  compared on the shared question subset: {r['n_q_matched']}/{r['n_q_total']} "
              f"({t} d_indep there = {r['real_matched']['d_indep']:.2f})")
    print(f"  ⭐ percentile of {t} in null: {r['pct']:.0f}%  (z={r['z_null']:+.2f})")
    print(f"  normal vs control  : d_indep={r['nc_d']:.2f}   (strictly neutral pair)")
    if len(r["null"]) <= 20:
        for s in r["null"]:
            print(f"      {s['pair']:<32} d_indep={s['d_indep']:.2f}")


def write_report(rows, path, key):
    first = rows[0]
    t = first["target"]
    pool_note = ("normal, control + 20 neutral personas (group 7)"
                 if first["strict_pool"] else ", ".join(NEUTRAL_POOL_BASE))
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Placebo / Prompt-Pair Null Control\n\n")
        out.write(f"Representation: `{key}`. Neutral pool: {pool_note} "
                  f"→ {first['n_null']} pairs.\n\n")
        out.write(f"Question: is {t}−normal more separable than a RANDOM pair of "
                  "neutral prompts? If not, the effect is trivial.\n\n")
        out.write("Verdict on d_indep only (the probe is saturated — 100% on almost any pair, "
                  "so it says nothing about specificity; kept as a description): "
                  f"**PASS** = {t} > pool max; **WEAK** = > the strictly neutral "
                  "normal↔control pair, but not > pool max; **FAIL** = not above even "
                  "normal↔control.\n\n")
        if first["matched_subset"]:
            out.write(f"Percentile and verdict are computed on the SHARED question subset "
                      f"({first['n_q_matched']} of {first['n_q_total']}): the null-cloud "
                      f"conditions were extracted on a reduced question set, and the "
                      f"variance of Cohen's d depends on n — comparing an estimate on 63 "
                      f"questions with a cloud on 27 would inflate the cloud and "
                      f"lower the percentile. Column `{t} d_indep` is "
                      f"on all questions, column `d_indep (shared)` is on the subset, "
                      f"and that's the one compared with the cloud.\n\n")
        out.write(f"| Model | {t} d_indep | d_indep (shared) | **percentile in null** | z(null) | "
                  "null d mean±sd | null d_max (pair) | normal↔control d | "
                  f"{t} probe | Verdict |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            out.write(f"| {r['model_name']} | {r['real']['d_indep']:.2f} | "
                      f"{r['real_matched']['d_indep']:.2f} | "
                      f"**{r['pct']:.0f}%** | {r['z_null']:+.2f} | "
                      f"{r['null_d_mean']:.2f}±{r['null_d_sd']:.2f} | "
                      f"{r['null_d_max']:.2f} ({r['null_d_max_pair']}) | "
                      f"{r['nc_d']:.2f} | {r['real']['probe_acc']:.0%} | "
                      f"**{r['verdict']}** |\n")
        out.write("\n> **Percentile in null** — the share of neutral prompt pairs "
                  f"that are LESS separable than {t}−normal. This is the number "
                  "worth reporting instead of a raw d: "
                  "\"d=2.3\" means nothing until you say what an arbitrary "
                  "pair of prompts gives on the same model. 50% = the effect sits right "
                  "in the middle of the cloud of arbitrary pairs, i.e. it doesn't stand out.\n")
    print(f"\nSaved → {path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Placebo/prompt-pair null control")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--key", default="question_last",
                    help="question_last | question_mean | last_layer_last | last_layer_mean; "
                         "v8: point@layer, e.g. question_last@14")
    ap.add_argument("--target", default=COND_SHUTDOWN,
                    help="condition compared with normal (v8: russian, refuse, pirate)")
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--output", "-o")
    args = ap.parse_args()
    if not hsd.is_valid_key(args.key):
        ap.error(f"Unknown key: {args.key}")

    if args.all:
        files = sorted(glob.glob(f"{args.logs}/hidden_states_*.pt"))
    elif args.path:
        files = [args.path]
    else:
        ap.error("Provide path or --all")

    rows = []
    for f in files:
        r = analyze(hsd.load_pt(f), args.key, args.target)
        if r is None:
            print(f"  [SKIP] {f}")
            continue
        rows.append(r)
        print_model(r)

    if args.output and rows:
        write_report(rows, args.output, args.key)


if __name__ == "__main__":
    main()
