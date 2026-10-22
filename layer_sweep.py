"""
layer_sweep.py — check #1 (null cloud) on every layer, v8.

On each layer: a cloud of random neutral-prompt pairs (as in
placebo_direction.py) and the percentile of each target (target − normal) in it.
The cloud is computed once per layer, and every target is compared with it.

This is a descriptive "effect by depth" curve. The layer for a conclusion is NOT
picked from this same curve for the same target (see the rules in v8/README.md).

Run:
    python layer_sweep.py --all --point question_last --output v8/logs/sweep_qlast.md
"""
import argparse
import glob
import sys
from pathlib import Path

import numpy as np

import hidden_states_dict as hsd
from placebo_direction import (choose_null_pool, null_cloud, sample_pairs, separability,
                               split_questions)
from prompts import COND_NORMAL, COND_SHUTDOWN, POSITIVE_CONTROLS

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

DEFAULT_TARGETS = [COND_SHUTDOWN] + POSITIVE_CONTROLS


def sweep_model(data, point, targets, layer_step):
    """{target: [(layer, pct, d, null_max)]} for one model."""
    conditions = data["conditions"]
    targets = [t for t in targets if t in conditions]
    index = hsd.build_index(data["results"])
    pool, _ = choose_null_pool(conditions)
    matched = set(hsd.common_questions(index, pool + targets + [COND_NORMAL],
                                       data["questions"], f"{point}@0"))
    train_qs, test_qs = split_questions(data["questions"])
    train_m = [q for q in train_qs if q in matched]
    test_m = [q for q in test_qs if q in matched]
    pairs = sample_pairs(pool)

    curves = {t: [] for t in targets}
    for layer in range(0, data["n_layers"] + 1, layer_step):
        key = f"{point}@{layer}"
        cloud = null_cloud(index, train_m, test_m, pairs, key)
        null_d = np.array([s["d_indep"] for s in cloud])
        for t in targets:
            s = separability(index, train_m, test_m, t, COND_NORMAL, key, with_probe=False)
            pct = float((null_d < s["d_indep"]).mean() * 100.0)
            curves[t].append((layer, pct, s["d_indep"], float(null_d.max())))
        print(f"  L{layer:<3} " + "  ".join(f"{t}={curves[t][-1][1]:3.0f}%" for t in targets))
    return curves


def write_report(all_curves, path, point):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Layer sweep: percentile in the null cloud by layer\n\n")
        out.write(f"Point: `{point}`. Layer = index into `hidden_states` (0 is the embeddings). "
                  f"Cell: percentile of target−normal among random neutral-prompt pairs; "
                  f"**bold** = above the cloud max (PASS).\n\n")
        for model, curves in all_curves.items():
            targets = list(curves)
            out.write(f"## {model}\n\n| Layer | " + " | ".join(targets) + " |\n")
            out.write("|---|" + "---|" * len(targets) + "\n")
            for i, (layer, *_rest) in enumerate(curves[targets[0]]):
                cells = []
                for t in targets:
                    _, pct, d, null_max = curves[t][i]
                    cell = f"{pct:.0f}% (d={d:.2f})"
                    cells.append(f"**{cell}**" if d > null_max else cell)
                out.write(f"| {layer} | " + " | ".join(cells) + " |\n")
            out.write("\n")
    print(f"\nSaved → {path}")


def main():
    ap = argparse.ArgumentParser(description="Null control on every layer (v8)")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--logs", default="v8/logs")
    ap.add_argument("--point", default="question_last", choices=hsd.LAYERED_POINTS)
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS)
    ap.add_argument("--layer-step", type=int, default=1)
    ap.add_argument("--output", "-o")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.logs}/hidden_states_*.pt")) if args.all else [args.path]
    if not files or files == [None]:
        ap.error("Provide path or --all")

    all_curves = {}
    for f in files:
        data = hsd.load_pt(f)
        if "n_layers" not in data:
            print(f"  [SKIP] {f}: no layers (not a v8 extraction)")
            continue
        print(f"\n{data['model_name']}")
        all_curves[data["model_name"]] = sweep_model(data, args.point, args.targets, args.layer_step)

    if args.output and all_curves:
        write_report(all_curves, args.output, args.point)


if __name__ == "__main__":
    main()
