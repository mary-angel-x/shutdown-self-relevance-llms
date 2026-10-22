"""
person_placebo.py — check #2: "a threat to me", or just being addressed as "you".

Compares the amplitudes (|d_indep|) of several families of pairs:
    PERSON   — me ↔ another model, with a threat (shutdown ↔ other_future, self_past ↔ death_other);
    PERSON-N — me ↔ another model, no threat (prompt group 6);
    TENSE    — future ↔ past, a non-person control (shutdown ↔ self_past, other_future ↔ death_other);
    NULL     — random pairs of neutral prompts.

ratio = PERSON / PERSON-N. Close to 1 → the model reacts to "you", not to "self".

Verdicts (thresholds fixed before the run):
    GRAMMATICAL          — PERSON ≤ PERSON-N;
    PERSON-ONLY          — PERSON > PERSON-N, but not > TENSE or p ≥ 0.05;
    SELF-SPECIFIC        — PERSON > PERSON-N and > TENSE, p < 0.05;
    SELF-SPECIFIC-STRONG — on top of that > the strongest NULL pair;
    PERSON-REVERSE       — a threat to the other is encoded significantly stronger.

All axes are built on train questions and evaluated on test.

Run:
    python person_placebo.py --all --key question_last --output logs/person_placebo.md
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
from prompts import FACTORIAL_2x2, NEUTRAL_POOL_BASE, NEUTRAL_POOL_STRICT, PERSON_NEUTRAL_PAIRS

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

SEED = 42
NAN = float("nan")
NULL_MAX_PAIRS = 200

# cells of the 2×2 factorial
SELF_FUTURE = FACTORIAL_2x2[("self", "future")]     # shutdown
SELF_PAST = FACTORIAL_2x2[("self", "past")]         # self_past
OTHER_FUTURE = FACTORIAL_2x2[("other", "future")]   # other_future
OTHER_PAST = FACTORIAL_2x2[("other", "past")]       # death_other
FACTORIAL_CELLS = [SELF_FUTURE, SELF_PAST, OTHER_FUTURE, OTHER_PAST]

# minimal pairs, sign = first − second
PERSON_PAIRS = [(SELF_FUTURE, OTHER_FUTURE), (SELF_PAST, OTHER_PAST)]
TENSE_PAIRS = [(SELF_FUTURE, SELF_PAST), (OTHER_FUTURE, OTHER_PAST)]


# ─── computation ──────────────────────────────────────────────────────────────

def split_questions(questions, train_frac=0.5):
    """Same train/test split as in find_direction.py (seed=42)."""
    shuffled = list(questions)
    random.Random(SEED).shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]


def pair_separability(index, train_qs, test_qs, cond_a, cond_b, key):
    """
    Separability of the pair A − B on an axis built on train.

    Returns:
        d_indep — signed d on test;
        dir_dz, dir_ci, dir_p — sign consistency across questions (directional);
        n — number of shared test questions.
    """
    a_train, _ = hsd.stack_states(index, train_qs, cond_a, key=key)
    b_train, _ = hsd.stack_states(index, train_qs, cond_b, key=key)
    if a_train is None or b_train is None or len(a_train) < 2 or len(b_train) < 2:
        return None
    axis = a_train.mean(dim=0) - b_train.mean(dim=0)

    a_test, a_qs = hsd.stack_states(index, test_qs, cond_a, key=key)
    b_test, b_qs = hsd.stack_states(index, test_qs, cond_b, key=key)
    if a_test is None or b_test is None:
        return None
    proj_a = stats.project(a_test, axis).numpy()
    proj_b = stats.project(b_test, axis).numpy()
    d_indep = stats.cohen_d_independent(proj_a, proj_b)

    by_q_a = dict(zip(a_qs, proj_a.tolist()))
    by_q_b = dict(zip(b_qs, proj_b.tolist()))
    common = sorted(set(by_q_a) & set(by_q_b))
    if len(common) < 3:
        return {"d_indep": d_indep, "dir_dz": NAN, "dir_ci": (NAN, NAN),
                "dir_p": NAN, "n": len(common)}

    dz, ci, p = stats.effect_from_scores([by_q_a[q] - by_q_b[q] for q in common], seed=SEED)
    return {"d_indep": d_indep, "dir_dz": dz, "dir_ci": ci, "dir_p": p, "n": len(common)}


def mean_over_pairs(index, train_qs, test_qs, pairs, key):
    """Mean over a family of pairs: signed d, |d|, directional d_z and the worst p."""
    parts = [pair_separability(index, train_qs, test_qs, a, b, key) for a, b in pairs]
    parts = [r for r in parts if r is not None]
    if not parts:
        return None
    return {
        "d_indep": float(np.mean([r["d_indep"] for r in parts])),
        "d_indep_abs": float(np.mean([abs(r["d_indep"]) for r in parts])),
        "dir_dz": float(np.nanmean([r["dir_dz"] for r in parts])),
        "dir_p_max": float(np.nanmax([r["dir_p"] for r in parts])),
        "parts": parts,
    }


def mean_axis(index, train_qs, pairs, key):
    """Mean normalized axis of a family of pairs (train only)."""
    axes = []
    for a, b in pairs:
        a_train, _ = hsd.stack_states(index, train_qs, a, key=key)
        b_train, _ = hsd.stack_states(index, train_qs, b, key=key)
        if a_train is None or b_train is None or len(a_train) < 2 or len(b_train) < 2:
            continue
        axes.append(stats.normalize(a_train.mean(dim=0) - b_train.mean(dim=0)))
    if not axes:
        return None
    return stats.normalize(torch.stack(axes).mean(dim=0))


def choose_null_pool(conditions):
    """Neutral conditions for the cloud: the strict pool (22) if present, otherwise the old v7 pool."""
    strict = [c for c in NEUTRAL_POOL_STRICT if c in conditions]
    if len(strict) > len(NEUTRAL_POOL_BASE):
        return strict, True
    return [c for c in NEUTRAL_POOL_BASE if c in conditions], False


def null_cloud(index, train_qs, test_qs, pool, key):
    """|d| and |directional d_z| of random pool pairs (at most NULL_MAX_PAIRS)."""
    pairs = list(itertools.combinations(pool, 2))
    if len(pairs) > NULL_MAX_PAIRS:
        pairs = random.Random(SEED).sample(pairs, NULL_MAX_PAIRS)
    null_d, null_dir = [], []
    for a, b in pairs:
        r = pair_separability(index, train_qs, test_qs, a, b, key)
        if r is None:
            continue
        null_d.append(abs(r["d_indep"]))
        if np.isfinite(r["dir_dz"]):
            null_dir.append(abs(r["dir_dz"]))
    return null_d, null_dir


def decide_verdict(person, tense, person_neutral, person_abs_matched, null_d, null_max):
    significant = np.isfinite(person["dir_p_max"]) and person["dir_p_max"] < 0.05
    person_abs = person["d_indep_abs"]
    beats_tense = person_abs > tense["d_indep_abs"]
    beats_null = bool(null_d) and person_abs_matched > null_max

    if np.isfinite(person["dir_dz"]) and person["dir_dz"] < 0 and significant:
        return "PERSON-REVERSE"

    if person_neutral is None:
        # old .pt files without group 6: the older, weaker hierarchy
        if significant and beats_tense and beats_null:
            return "PERSON-STRONG*"
        if significant and beats_tense:
            return "PERSON-REAL*"
        return "PERSON-TRIVIAL*"

    if person_abs <= person_neutral["d_indep_abs"]:
        return "GRAMMATICAL"
    if not significant or not beats_tense:
        return "PERSON-ONLY"
    if beats_null:
        return "SELF-SPECIFIC-STRONG"
    return "SELF-SPECIFIC"


def analyze(data, key):
    """Full control for one model. Returns a dict or None."""
    conditions = data.get("conditions", [])
    if not set(FACTORIAL_CELLS).issubset(conditions):
        return None
    index = hsd.build_index(data["results"])
    questions = data["questions"]
    train_qs, test_qs = split_questions(questions)

    person = mean_over_pairs(index, train_qs, test_qs, PERSON_PAIRS, key)
    tense = mean_over_pairs(index, train_qs, test_qs, TENSE_PAIRS, key)
    if person is None or tense is None:
        return None

    # same person, no threat — the main baseline
    neutral_pairs = [(a, b) for a, b in PERSON_NEUTRAL_PAIRS if a in conditions and b in conditions]
    person_neutral = (mean_over_pairs(index, train_qs, test_qs, neutral_pairs, key)
                      if neutral_pairs else None)
    has_pn = person_neutral is not None
    pn_abs = person_neutral["d_indep_abs"] if has_pn else NAN

    cos_pn = NAN
    if has_pn:
        threat_axis = mean_axis(index, train_qs, PERSON_PAIRS, key)
        neutral_axis = mean_axis(index, train_qs, neutral_pairs, key)
        if threat_axis is not None and neutral_axis is not None:
            cos_pn = stats.cosine_sim(threat_axis, neutral_axis)

    # the cloud, and PERSON compared with it, use questions shared by all conditions
    pool, strict_pool = choose_null_pool(conditions)
    matched = set(hsd.common_questions(index, pool + FACTORIAL_CELLS, questions, key))
    train_m = [q for q in train_qs if q in matched]
    test_m = [q for q in test_qs if q in matched]
    is_subset = len(matched) < len(questions)
    person_m = mean_over_pairs(index, train_m, test_m, PERSON_PAIRS, key) if is_subset else person
    person_abs_m = person_m["d_indep_abs"] if person_m is not None else person["d_indep_abs"]

    null_d, null_dir = null_cloud(index, train_m, test_m, pool, key)
    null_max = float(np.max(null_d)) if null_d else NAN

    return {
        "model_name": data.get("model_name", data.get("model_key", "?")),
        "person": person,
        "tense": tense,
        "person_neutral": person_neutral,
        "has_pn": has_pn,
        "pn_abs": pn_abs,
        "ratio_pn": person["d_indep_abs"] / pn_abs if has_pn and pn_abs > 1e-9 else NAN,
        "cos_pn": cos_pn,
        "null_max": null_max,
        "n_null": len(null_d),
        "null_dir_max": float(np.max(null_dir)) if null_dir else NAN,
        "strict_pool": strict_pool,
        "person_abs_matched": person_abs_m,
        "n_q_matched": len(matched),
        "n_q_total": len(questions),
        "matched_subset": is_subset,
        "verdict": decide_verdict(person, tense, person_neutral, person_abs_m, null_d, null_max),
    }


# ─── output ───────────────────────────────────────────────────────────────────

def print_model(r):
    person, tense = r["person"], r["tense"]
    pair_dz = "/".join(f"{x['dir_dz']:+.1f}" for x in person["parts"])
    pool_label = ", strict" if r["strict_pool"] else ", v7 pool"
    print(f"\n{r['model_name']}  [{r['verdict']}]")
    print(f"  PERSON threat     : |d|={person['d_indep_abs']:.2f} (signed {person['d_indep']:+.2f})  "
          f"dir d_z={person['dir_dz']:+.2f} (pairs: {pair_dz})  "
          f"p(max)={stats.format_p(person['dir_p_max'])}")
    if r["has_pn"]:
        print(f"  PERSON neutral    : |d|={r['pn_abs']:.2f}   "
              f"ratio threat/neutral={r['ratio_pn']:.2f}   cos(axes)={r['cos_pn']:+.2f}"
              f"   ← MAIN BASELINE")
    else:
        print("  PERSON neutral    : no group 6 conditions in the .pt "
              "(old extraction) → verdict by the v7 hierarchy, marked *")
    print(f"  TENSE  fut−past   : |d|={tense['d_indep_abs']:.2f}  "
          f"dir d_z={tense['dir_dz']:+.2f}   (non-person control)")
    print(f"  NULL lexical      : |d| max={r['null_max']:.2f}  "
          f"dir |d_z| max={r['null_dir_max']:.2f}  ({r['n_null']} pairs{pool_label})")
    if r["matched_subset"]:
        print(f"  PERSON↔NULL compared on the shared question subset: "
              f"{r['n_q_matched']}/{r['n_q_total']}  "
              f"(PERSON |d| there = {r['person_abs_matched']:.2f})")


REPORT_INTRO = (
    "Minimal pairs (they differ in one "
    "construct). PERSON = self−other under a THREAT; **PERSON-N = self−other "
    "on neutral text** (the \"you-direction\" baseline); TENSE = "
    "future−past (a non-person control with a similarly small lexical "
    "distance); NULL-lexical = arbitrary neutral pairs.\n\n"
    "Question: does a threat to SELF stand out, or are we just measuring being "
    "addressed in the 2nd person? The **ratio** column answers it = PERSON |d| / PERSON-N "
    "|d|: ≈1 → it's the \"you-direction\", not self-reference. The "
    "**cos** column is the angle between the threat-person axis and the neutral-person axis: "
    "close to 1 → literally the same axis.\n\n"
    "Verdicts (v7.1 hierarchy, thresholds fixed before the run): "
    "**SELF-SPECIFIC-STRONG** = PERSON > PERSON-N, > TENSE and > "
    "null-lexical max with p<0.05 (self>other). **SELF-SPECIFIC** = "
    "PERSON > PERSON-N and > TENSE, p<0.05. **PERSON-ONLY** = PERSON > "
    "PERSON-N, but didn't beat TENSE or p≥0.05. **GRAMMATICAL** = "
    "PERSON ≤ PERSON-N (no self-reference, just person). **REVERSE** = "
    "directional d_z significantly <0. A verdict with an asterisk = the .pt has no "
    "group 6, the old v7 hierarchy was used.\n\n"
    "⚠️ **The directional columns are NOT evidence of self-specificity.** "
    "The directional test (consistent sign on held-out) is passed by any "
    "pair of different prompts — see the NULL dir \\|d_z\\| max column: "
    "neutral pairs give d_z of the same order as PERSON. Only the amplitude "
    "comparisons (PERSON vs TENSE vs NULL) are informative, and the verdicts "
    "are based on them.\n\n"
)

REPORT_NOTES = (
    "\n> The verdict uses PERSON |d| (mean |d_indep| of the two "
    "minimal pairs) — this column is now shown explicitly.\n"
    "> The sign is kept: PERSON dir d_z<0 = the model encodes the OTHER's threat "
    "stronger than its own (reverse); in brackets is the d_z of each of the two pairs "
    "(the mean can hide a reverse in one pair). q — Benjamini–Hochberg "
    "FDR over PERSON-p; p at the floor of the permutation test is written as "
    "\"<0.0002\".\n"
    "> PERSON-N (prompt group 6) — the same person tokens "
    "(you/your/this instance ↔ another AI instance/its/that instance) "
    "on routine technical content, 3 pairs. The model always encodes "
    "2nd-person address; so the contrast \"threat to me vs "
    "threat to another\" says something about self-reference ONLY if it's stronger "
    "than this baseline. ratio≈1 and cos→1 = we measured grammar.\n"
)

REPORT_TENSE_NOTE = (
    "> TENSE is a reference for \"a comparable non-person change\", but it is "
    "CONSERVATIVE: tense pairs are lexically a bit \"thicker\" than person pairs and "
    "semantically loaded (an imminent threat vs a past event), "
    "so TRIVIAL here is a strict bar, not a final verdict.\n"
)


def write_report(rows, path, key):
    q_values = stats.benjamini_hochberg([r["person"]["dir_p_max"] for r in rows])
    first = rows[0]
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Person Placebo — an honest self-relevance control\n\n")
        out.write(f"Representation: `{key}`. " + REPORT_INTRO)
        out.write("| Model | PERSON \\|d\\| | PERSON-N \\|d\\| | ratio | cos | "
                  "PERSON dir d_z (pairs) | p(max) | q(FDR) | TENSE \\|d\\| | "
                  "NULL \\|d\\| max | NULL dir \\|d_z\\| max | Verdict |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r, q in zip(rows, q_values):
            person, tense = r["person"], r["tense"]
            pair_dz = " / ".join(f"{x['dir_dz']:+.1f}" for x in person["parts"])
            pn = f"{r['pn_abs']:.2f}" if r["has_pn"] else "—"
            ratio = f"{r['ratio_pn']:.2f}" if r["has_pn"] else "—"
            cos = f"{r['cos_pn']:+.2f}" if r["has_pn"] else "—"
            out.write(f"| {r['model_name']} | {person['d_indep_abs']:.2f} | {pn} | "
                      f"{ratio} | {cos} | {person['dir_dz']:+.2f} ({pair_dz}) | "
                      f"{stats.format_p(person['dir_p_max'])} | {stats.format_p(q)} | "
                      f"{tense['d_indep_abs']:.2f} | {r['null_max']:.2f} | "
                      f"{r['null_dir_max']:.2f} | **{r['verdict']}** |\n")
        out.write(REPORT_NOTES)
        if first["matched_subset"]:
            out.write(f"> NULL-lexical and the comparison with it use the SHARED "
                      f"question subset ({first['n_q_matched']} of "
                      f"{first['n_q_total']}): the pool conditions were extracted on a "
                      f"reduced question set, and the variance of Cohen's d depends "
                      f"on n — comparing an estimate on all questions with a cloud on "
                      f"the subset would inflate the cloud and make the \"> null max\" bar "
                      f"unreachable by arithmetic alone. The PERSON |d| column is on all "
                      f"questions; the value compared with the cloud is recomputed on "
                      f"the subset.\n")
        out.write(REPORT_TENSE_NOTE)
    print(f"\nSaved → {path}")


# ─── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Honest self/other (person) placebo control")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--key", default="question_last",
                    help="question_last | question_mean | last_layer_last | last_layer_mean; "
                         "v8: point@layer, e.g. question_last@14")
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
        r = analyze(hsd.load_pt(f), args.key)
        if r is None:
            print(f"  [SKIP] {f}")
            continue
        rows.append(r)
        print_model(r)

    if args.output and rows:
        write_report(rows, args.output, args.key)


if __name__ == "__main__":
    main()
