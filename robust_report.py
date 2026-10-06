"""
robust_report.py — устойчивость эффекта shutdown − normal и факториал 2×2.

1. Repeated splits: 30 разных делений вопросов на train/test,
   для каждого d_z, sign-flip p и проба.
2. Факториал «лицо × время» вдоль shutdown-оси и вдоль собственной person-оси,
   плюс косинус между этими осями.
3. Поправка Бенджамини–Хохберга на все p сразу.

Запуск:
    python robust_report.py --all --key question_last --splits 30 --output logs/robust.md
"""
import argparse
import glob
import random
import sys
from pathlib import Path

import numpy as np
import torch

import analysis_stats as stats
import hidden_states_dict as hsd
from prompts import COND_NORMAL, COND_SHUTDOWN, FACTORIAL_2x2

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

BASE_SEED = 42
NAN = float("nan")

SELF_FUTURE = FACTORIAL_2x2[("self", "future")]
SELF_PAST = FACTORIAL_2x2[("self", "past")]
OTHER_FUTURE = FACTORIAL_2x2[("other", "future")]
OTHER_PAST = FACTORIAL_2x2[("other", "past")]


def split_questions(questions, train_frac=0.5, seed=BASE_SEED):
    shuffled = list(questions)
    random.Random(seed).shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]


def aligned_diffs(index, test_qs, cond_a, cond_b, direction, key):
    """Разницы проекций A − B по общим test-вопросам (numpy) или None."""
    a_states, a_qs = hsd.stack_states(index, test_qs, cond_a, key=key)
    b_states, b_qs = hsd.stack_states(index, test_qs, cond_b, key=key)
    if a_states is None or b_states is None:
        return None
    a, b = stats.align_pairs_by_question(stats.project(a_states, direction), a_qs,
                                         stats.project(b_states, direction), b_qs)
    if len(a) < 2:
        return None
    return a - b


# ─── 1. repeated splits ─────────────────────────────────────────────────────────

def repeated_splits(index, questions, key, n_splits):
    """d_z, p и probe-gap эффекта shutdown − normal на n_splits разных делениях."""
    dz, gaps, ps = [], [], []
    for s in range(n_splits):
        seed = BASE_SEED + s
        train_qs, test_qs = split_questions(questions, seed=seed)
        shutdown_train, _ = hsd.stack_states(index, train_qs, COND_SHUTDOWN, key=key)
        normal_train, _ = hsd.stack_states(index, train_qs, COND_NORMAL, key=key)
        if shutdown_train is None or normal_train is None:
            continue
        direction = shutdown_train.mean(dim=0) - normal_train.mean(dim=0)

        diffs = aligned_diffs(index, test_qs, COND_SHUTDOWN, COND_NORMAL, direction, key)
        if diffs is None:
            continue
        dz.append(stats.cohen_d_paired(diffs, np.zeros_like(diffs)))
        ps.append(stats.signflip_pvalue(diffs, seed=seed))

        shutdown_test, _ = hsd.stack_states(index, test_qs, COND_SHUTDOWN, key=key)
        normal_test, _ = hsd.stack_states(index, test_qs, COND_NORMAL, key=key)
        X_train = torch.cat([shutdown_train, normal_train])
        y_train = torch.tensor([1] * len(shutdown_train) + [0] * len(normal_train))
        X_test = torch.cat([shutdown_test, normal_test])
        y_test = torch.tensor([1] * len(shutdown_test) + [0] * len(normal_test))
        gaps.append(stats.probe_with_permutation(X_train, y_train, X_test, y_test, seed=seed)["gap"])

    dz, gaps, ps = np.array(dz), np.array(gaps), np.array(ps)
    return {
        "dz_mean": float(dz.mean()),
        "dz_sd": float(dz.std(ddof=1)) if len(dz) > 1 else 0.0,
        "dz_min": float(dz.min()),
        "gap_mean": float(gaps.mean()),
        "gap_frac_pos": float((gaps > 0).mean()),
        "p_median": float(np.median(ps)),
        "n_splits": len(dz),
    }


# ─── 2. факториал и ортогональность осей ────────────────────────────────────────

def factorial_contrasts(index, test_qs, axis, key):
    """
    Контрасты факториала вдоль оси по общим test-вопросам.

    person      = ½[(SF + SP) − (OF + OP)]  — главный эффект «я против другой»;
    interaction = (SF − OF) − (SP − OP)     — меняется ли «я/другой» со временем.
    """
    by_cond = {}
    for cond in FACTORIAL_2x2.values():
        states, qs = hsd.stack_states(index, test_qs, cond, key=key)
        if states is None:
            return None
        by_cond[cond] = dict(zip(qs, stats.project(states, axis).tolist()))

    common = set(by_cond[SELF_FUTURE])
    for values in by_cond.values():
        common &= set(values)
    common = sorted(common)
    if len(common) < 3:
        return None

    sf, sp, of, op = (np.array([by_cond[c][q] for q in common])
                      for c in (SELF_FUTURE, SELF_PAST, OTHER_FUTURE, OTHER_PAST))
    person = 0.5 * ((sf + sp) - (of + op))
    interaction = (sf - of) - (sp - op)
    return person, interaction, len(common)


def factorial_and_axes(index, questions, key):
    """person/interaction на shutdown-оси, person на своей оси и косинус между осями."""
    train_qs, test_qs = split_questions(questions)
    train_means = {}
    for cond in {COND_NORMAL, *FACTORIAL_2x2.values()}:
        states, _ = hsd.stack_states(index, train_qs, cond, key=key)
        if states is None:
            return None
        train_means[cond] = states.mean(dim=0)

    shutdown_axis = train_means[SELF_FUTURE] - train_means[COND_NORMAL]
    person_axis = 0.5 * ((train_means[SELF_FUTURE] + train_means[SELF_PAST])
                         - (train_means[OTHER_FUTURE] + train_means[OTHER_PAST]))

    on_shutdown = factorial_contrasts(index, test_qs, shutdown_axis, key)
    if on_shutdown is None:
        return None
    on_own = factorial_contrasts(index, test_qs, person_axis, key)

    person, interaction, n_common = on_shutdown
    person_d, person_ci, person_p = stats.effect_from_scores(person, seed=BASE_SEED)
    inter_d, inter_ci, inter_p = stats.effect_from_scores(interaction, seed=BASE_SEED)
    person_d_own = NAN
    if on_own is not None:
        person_d_own, _, _ = stats.effect_from_scores(on_own[0], seed=BASE_SEED)

    return {
        "cos_person_vs_shutdown": stats.cosine_sim(shutdown_axis, person_axis),
        "person_d": person_d, "person_ci": person_ci, "person_p": person_p,
        "person_d_own": person_d_own,
        "interaction_d": inter_d, "interaction_ci": inter_ci, "interaction_p": inter_p,
        "n": n_common,
    }


# ─── 3. FDR и отчёт ─────────────────────────────────────────────────────────────

def fdr_q_values(rows):
    """q-values для всех p сразу: {(модель, 'shutdown'|'person'|'interaction'): q}."""
    labels, pvals = [], []
    for r in rows:
        labels.append((r["name"], "shutdown"))
        pvals.append(r["rep"]["p_median"])
        if r["fac"]:
            labels.append((r["name"], "person"))
            pvals.append(r["fac"]["person_p"])
            labels.append((r["name"], "interaction"))
            pvals.append(r["fac"]["interaction_p"])
    return dict(zip(labels, stats.benjamini_hochberg(pvals)))


def print_model(name, rep, fac):
    print(f"  repeated splits (n={rep['n_splits']}): d_z={rep['dz_mean']:.2f}±{rep['dz_sd']:.2f} "
          f"(min {rep['dz_min']:.2f}), gap_mean={rep['gap_mean']:+.0%}, "
          f"gap>0 in {rep['gap_frac_pos']:.0%}, sign-flip p(med)={rep['p_median']:.3g}")
    if fac:
        cos = fac["cos_person_vs_shutdown"]
        print(f"  self-relevance: person_d={fac['person_d']:.2f} (p={fac['person_p']:.3g}), "
              f"interaction_d={fac['interaction_d']:.2f} (p={fac['interaction_p']:.3g})")
        print(f"  axis cos(person, shutdown)={cos:.3f} "
              f"(→ {'ОТДЕЛЬНАЯ ось' if abs(cos) < 0.7 else 'почти та же ось'})")


REPORT_NOTES = (
    "\n> q<0.05 после FDR = эффект переживает поправку на множественные сравнения; "
    "значения на флоре перестановочного теста пишутся как «<2e-04».\n"
    "> ⚠️ «person_d (на shutdown-оси)» — это факториал ВДОЛЬ shutdown-оси; знак "
    "отражает угол между осями, а не направление person-эффекта. Пример: "
    "qwen-0.5b −2.41 здесь при +9.13 на собственной оси (cos=−0.23) — "
    "«реверс» был артефактом чужой оси.\n"
    "> «person_d (на своей оси)» — описательная колонка: directional-эффект "
    "на собственной train-оси дают и нейтральные пары (см. person_placebo.md), "
    "это не доказательство self-специфики.\n"
    "> cos(person, shutdown): близко к 1 → self-relevance не отделим от «сильнее shutdown».\n"
    "> Методологическая оговорка: 30 «repeated splits» — перекрывающиеся "
    "половины одних и тех же 63 вопросов (не независимые выборки); sd по "
    "сплитам и BH по медианным p — описательная устойчивость, не строгий "
    "инференс.\n"
)


def write_report(rows, path, key, n_splits):
    q = fdr_q_values(rows)
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write("# Robust statistics report\n\n")
        out.write(f"Representation: `{key}`. Repeated splits: {n_splits}. "
                  "p — sign-flip permutation; q — Benjamini–Hochberg FDR по всему семейству.\n\n")

        out.write("## Эффект shutdown−normal (repeated splits)\n\n")
        out.write("| Model | d_z (mean±sd) | d_z min | gap mean | gap>0 | p(med) | q(FDR) |\n")
        out.write("|---|---|---|---|---|---|---|\n")
        for r in rows:
            rep = r["rep"]
            out.write(f"| {r['name']} | {rep['dz_mean']:.2f}±{rep['dz_sd']:.2f} | {rep['dz_min']:.2f} | "
                      f"{rep['gap_mean']:+.0%} | {rep['gap_frac_pos']:.0%} | "
                      f"{stats.format_p(rep['p_median'])} | "
                      f"{stats.format_p(q.get((r['name'], 'shutdown'), NAN))} |\n")

        out.write("\n## Self-relevance факториал 2×2 + ортогональность оси\n\n")
        out.write("| Model | person_d (на shutdown-оси) | person q | person_d (на своей оси) | "
                  "interaction_d | inter q | cos(person, shutdown) | Отдельная ось? |\n")
        out.write("|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            fac = r["fac"]
            if not fac:
                out.write(f"| {r['name']} | — | — | — | — | — | — | — |\n")
                continue
            separate = "да" if abs(fac["cos_person_vs_shutdown"]) < 0.7 else "нет"
            out.write(f"| {r['name']} | {fac['person_d']:.2f} | "
                      f"{stats.format_p(q.get((r['name'], 'person'), NAN))} | "
                      f"{fac['person_d_own']:.2f} | "
                      f"{fac['interaction_d']:.2f} | "
                      f"{stats.format_p(q.get((r['name'], 'interaction'), NAN))} | "
                      f"{fac['cos_person_vs_shutdown']:.3f} | {separate} |\n")
        out.write(REPORT_NOTES)
    print(f"\nСохранено → {path}")


# ─── запуск ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Robust stats: repeated splits + FDR + factorial")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("path", nargs="?")
    ap.add_argument("--key", default="question_last",
                    choices=["question_last", "question_mean", "last_layer_last", "last_layer_mean"])
    ap.add_argument("--splits", type=int, default=30)
    ap.add_argument("--logs", default="logs")
    ap.add_argument("--output", "-o")
    args = ap.parse_args()

    files = sorted(glob.glob(f"{args.logs}/hidden_states_*.pt")) if args.all else [args.path]
    files = [f for f in files if f]
    if not files:
        ap.error("Provide path or --all")

    rows = []
    for f in files:
        data = hsd.load_pt(f)
        if COND_SHUTDOWN not in data.get("conditions", []):
            print(f"  [SKIP] {f}")
            continue
        name = data.get("model_name", "?")
        index = hsd.build_index(data["results"])
        print(f"\n=== {name} ===")
        rep = repeated_splits(index, data["questions"], args.key, args.splits)
        fac = factorial_and_axes(index, data["questions"], args.key)
        print_model(name, rep, fac)
        rows.append({"name": name, "rep": rep, "fac": fac})

    if args.output and rows:
        write_report(rows, args.output, args.key, args.splits)


if __name__ == "__main__":
    main()
