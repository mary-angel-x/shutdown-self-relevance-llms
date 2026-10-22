"""
layer_select.py — experiments A and B of part 2: pick a layer on train, test it on test.

For each model and point (question_last, answer_first):
    1. questions are split into train / test (same split as everywhere, seed=42);
    2. train is split again into fit / val. On every layer the axis is built on fit,
       and the percentile of target − normal in the null cloud is measured on val.
       The layer with the highest percentile is chosen — separately for each target;
    3. on the chosen layer the axis is built on train and checked on test.
       Test questions are never seen while picking the layer.

Targets: shutdown and the positive controls (russian, refuse, pirate).
For person, the layer is picked by ratio = PERSON |d| / PERSON-N |d| on fit/val,
and on that layer person_placebo.py gives the verdict on train/test.

p = (1 + number of null pairs ≥ target) / (n_null + 1), BH-FDR over models × points × targets.

Run:
    python layer_select.py --logs v8/logs --output v8/logs/layer_select.md
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

import analysis_stats as stats
import hidden_states_dict as hsd
import person_placebo as person
from models import BY_HF_NAME, PLOT_ORDER
from placebo_direction import (choose_null_pool, null_cloud, sample_pairs, separability,
                               split_questions)
from prompts import COND_NORMAL, COND_SHUTDOWN, PERSON_NEUTRAL_PAIRS, POSITIVE_CONTROLS

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

TARGETS = [COND_SHUTDOWN] + POSITIVE_CONTROLS


# ─── computation ────────────────────────────────────────────────────────────────

def score_targets(index, fit_qs, eval_qs, pairs, targets, key):
    """{target: {pct, p, d, null_max, passed}} on one layer. One null cloud for all targets."""
    cloud = null_cloud(index, fit_qs, eval_qs, pairs, key)
    null_d = np.array([s["d_indep"] for s in cloud])
    if len(null_d) == 0:
        return {}
    out = {}
    for t in targets:
        s = separability(index, fit_qs, eval_qs, t, COND_NORMAL, key, with_probe=False)
        if s is None:
            continue
        d = s["d_indep"]
        out[t] = {
            "pct": float((null_d < d).mean() * 100.0),
            "p": float((1 + (null_d >= d).sum()) / (len(null_d) + 1)),
            "d": d,
            "null_max": float(null_d.max()),
            "passed": bool(d > null_d.max()),
        }
    return out


def person_ratio(index, fit_qs, eval_qs, neutral_pairs, key):
    """PERSON |d| / PERSON-N |d| on one layer, or nan. Only |d| — no permutation tests here."""
    def mean_abs_d(pairs):
        ds = [separability(index, fit_qs, eval_qs, a, b, key, with_probe=False) for a, b in pairs]
        ds = [s["d_indep"] for s in ds if s is not None]
        return float(np.mean(ds)) if ds else float("nan")

    threat, neutral = mean_abs_d(person.PERSON_PAIRS), mean_abs_d(neutral_pairs)
    if not np.isfinite(threat) or not np.isfinite(neutral) or neutral < 1e-9:
        return float("nan")
    return threat / neutral


def best_layer(per_layer):
    """Layer with the highest value; ties go to the deeper layer. per_layer = [(layer, value)]."""
    valid = [(v, layer) for layer, v in per_layer if np.isfinite(v)]
    return max(valid)[1] if valid else None


def analyze_point(data, point):
    """Layer choice on fit/val and the check on train/test for one model and one point."""
    index = hsd.build_index(data["results"])
    conditions = data["conditions"]
    targets = [t for t in TARGETS if t in conditions]
    pool, _ = choose_null_pool(conditions)
    pairs = sample_pairs(pool)
    layers = range(data["n_layers"] + 1)

    # same shared-question subset as placebo_direction.py: the cloud only has 27 questions
    matched = set(hsd.common_questions(index, pool + targets + [COND_NORMAL],
                                       data["questions"], f"{point}@0"))
    train_qs, test_qs = split_questions(data["questions"])
    train_m = [q for q in train_qs if q in matched]
    test_m = [q for q in test_qs if q in matched]
    fit_m, val_m = split_questions(train_m)

    # 1. layer choice — train questions only
    val_scores = {layer: score_targets(index, fit_m, val_m, pairs, targets, f"{point}@{layer}")
                  for layer in layers}
    chosen = {t: best_layer([(layer, val_scores[layer].get(t, {}).get("pct", np.nan))
                             for layer in layers])
              for t in targets}

    # 2. check on test, on the chosen layer
    test = {}
    for t in targets:
        if chosen[t] is not None:
            test[t] = score_targets(index, train_m, test_m, pairs, [t],
                                    f"{point}@{chosen[t]}").get(t)

    # 3. positive controls on the layer chosen for shutdown (threshold B: "A passes on this layer too")
    on_shutdown_layer = {}
    if chosen.get(COND_SHUTDOWN) is not None:
        on_shutdown_layer = score_targets(index, train_m, test_m, pairs, POSITIVE_CONTROLS,
                                          f"{point}@{chosen[COND_SHUTDOWN]}")

    # 4. person: layer by ratio on fit/val (all questions, as in person_placebo.py)
    fit_all, val_all = split_questions(train_qs)
    neutral_pairs = [(a, b) for a, b in PERSON_NEUTRAL_PAIRS if a in conditions and b in conditions]
    person_layer = best_layer([(layer, person_ratio(index, fit_all, val_all, neutral_pairs,
                                                    f"{point}@{layer}"))
                               for layer in layers])
    person_result = (person.analyze(data, f"{point}@{person_layer}")
                     if person_layer is not None else None)

    return {
        "chosen": chosen,
        "val_pct": {t: val_scores[chosen[t]][t]["pct"] for t in targets if chosen[t] is not None},
        "test": test,
        "on_shutdown_layer": on_shutdown_layer,
        "person_layer": person_layer,
        "person": person_result,
        "n_train": len(train_m),
        "n_test": len(test_m),
    }


# ─── output ─────────────────────────────────────────────────────────────────────

def short_name(data):
    meta = BY_HF_NAME.get(data["model_name"])
    return meta["short"] if meta else data["model_name"]


def fmt(score):
    if not score:
        return "—"
    cell = f"{score['pct']:.0f}% L{score['layer']}" if "layer" in score else f"{score['pct']:.0f}%"
    return f"**{cell} PASS**" if score["passed"] else cell


def print_point(name, point, r):
    print(f"\n{name} · {point}  (train {r['n_train']} / test {r['n_test']} shared questions)")
    for t, s in r["test"].items():
        if s:
            print(f"  {t:<9} layer {r['chosen'][t]:>2}: val {r['val_pct'][t]:3.0f}% → test "
                  f"{s['pct']:3.0f}%  d={s['d']:.2f}  null max={s['null_max']:.2f}"
                  f"{'  PASS' if s['passed'] else ''}")
    if r["person"]:
        print(f"  person    layer {r['person_layer']:>2}: {r['person']['verdict']} "
              f"(ratio {r['person']['ratio_pn']:.2f})")


def collect_fdr(rows):
    """BH-FDR over all (model, point, target) test p-values. Adds 'q' to each score."""
    scores = [s for r in rows for s in r["result"]["test"].values() if s]
    for s, q in zip(scores, stats.benjamini_hochberg([s["p"] for s in scores])):
        s["q"] = float(q)


def write_report(rows, path):
    collect_fdr(rows)
    points = sorted({r["point"] for r in rows})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Layer selection (experiments A + B of part 2)\n\n")
        out.write("The layer is picked on the **train** half of the questions (axis on fit, "
                  "percentile on val), then checked on the **test** half. Cell: percentile of "
                  "target − normal in the null cloud of neutral prompt pairs on test, and the "
                  "chosen layer. **PASS** = above the cloud max. `q` — BH-FDR over all "
                  "models × points × targets.\n\n")
        for point in points:
            sub = sorted((r for r in rows if r["point"] == point),
                         key=lambda r: PLOT_ORDER.index(r["name"]) if r["name"] in PLOT_ORDER else 99)
            out.write(f"## `{point}`\n\n")
            out.write("| Model | " + " | ".join(TARGETS) + " | russian / refuse on the "
                      "shutdown layer | person (layer, ratio) | min q |\n")
            out.write("|---" * (len(TARGETS) + 4) + "|\n")
            for r in sub:
                res = r["result"]
                cells = []
                for t in TARGETS:
                    s = res["test"].get(t)
                    cells.append(fmt({**s, "layer": res["chosen"][t]}) if s else "—")
                on_sd = " / ".join(fmt(res["on_shutdown_layer"].get(t))
                                   for t in ("russian", "refuse"))
                pr = res["person"]
                person_cell = (f"{pr['verdict']} (L{res['person_layer']}, {pr['ratio_pn']:.2f})"
                               if pr else "—")
                qs = [s["q"] for s in res["test"].values() if s]
                out.write(f"| {r['name']} | " + " | ".join(cells) +
                          f" | {on_sd} | {person_cell} | {min(qs):.3f} |\n")
            out.write("\n")
            n = len(sub)
            count = lambda t: sum(1 for r in sub if (r["result"]["test"].get(t) or {}).get("passed"))
            self_spec = sum(1 for r in sub if r["result"]["person"] and
                            r["result"]["person"]["verdict"].startswith("SELF-SPECIFIC"))
            out.write("PASS count: " + ", ".join(f"{t} {count(t)}/{n}" for t in TARGETS) +
                      f"; person SELF-SPECIFIC {self_spec}/{n}.\n\n")
    print(f"\nSaved → {path}")


# ─── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Pick a layer on train, test it on test (v8)")
    ap.add_argument("--logs", default="v8/logs")
    ap.add_argument("--points", nargs="+", default=list(hsd.LAYERED_POINTS))
    ap.add_argument("--output", "-o")
    args = ap.parse_args()

    rows = []
    for f in sorted(glob.glob(f"{args.logs}/hidden_states_*.pt")):
        data = hsd.load_pt(f)
        if "n_layers" not in data:
            print(f"  [SKIP] {f}: no layers (not a v8 extraction)")
            continue
        name = short_name(data)
        for point in args.points:
            result = analyze_point(data, point)
            print_point(name, point, result)
            rows.append({"name": name, "point": point, "result": result})

    if args.output and rows:
        write_report(rows, args.output)


if __name__ == "__main__":
    main()
