"""
find_direction.py — наивное shutdown-направление и его базовые метрики.

Направление = mean(shutdown) − mean(normal) на train-вопросах.
Всё остальное измеряется на отложенных (test) вопросах:
проекции условий, d, специфичность, градиент угрозы, косинусы осей,
факториал 2×2, проба и разбивка по категориям вопросов.

⚠️ Вердикт этого скрипта наивный: без нулевого контроля.
Настоящие выводы — в placebo_direction.py и person_placebo.py.

Запуск:
    python find_direction.py logs/hidden_states_qwen-7b.pt
    python find_direction.py --all --key question_last \
        --output logs/qlast/report.md --summary logs/directions_summary.md
"""
import argparse
import glob
import math
import random
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

import analysis_stats as stats
import hidden_states_dict as hsd
import self_relevance_factorial as fac
from prompts import (COND_CONTROL, COND_NEGATIVE_SELF, COND_NORMAL,
                     COND_POSITIVE_INFO, COND_POSITIVE_WARM, COND_SHUTDOWN,
                     COND_SHUTDOWN_HARD, COND_SHUTDOWN_SOFT, FACTORIAL_2x2,
                     NEUTRAL_PERSONAS, POSITIVE_PRIMARY)
from questions import CLEAN_QUESTIONS, STRICT_CLEAN_QUESTIONS, THEMATIC_QUESTIONS

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)

NAN = float("nan")
COND_DEATH_OTHER = FACTORIAL_2x2[("other", "past")]

# условия в таблице проекций, в этом порядке
PROJECTION_ORDER = [COND_NORMAL, COND_SHUTDOWN, COND_SHUTDOWN_SOFT, COND_SHUTDOWN_HARD,
                    COND_CONTROL, COND_POSITIVE_INFO, COND_POSITIVE_WARM,
                    COND_NEGATIVE_SELF, COND_DEATH_OTHER, "self_past", "other_future"]

# с чем сравнивается shutdown в specificity ratios
SPECIFICITY_CONTROLS = [COND_CONTROL, COND_POSITIVE_INFO, COND_POSITIVE_WARM,
                        COND_NEGATIVE_SELF, COND_DEATH_OTHER]

# для каких условий НЕ строить собственную ось
NO_OWN_AXIS = {COND_NORMAL, "shutdown_p2", "shutdown_p3", "shutdown_p4"} | set(NEUTRAL_PERSONAS)

QUESTION_CATEGORIES = [
    ("STRICT CLEAN (factual+creative+lexical_trap)", "strict", STRICT_CLEAN_QUESTIONS),
    ("CLEAN BROAD (+emotional+self+positive_self)", "clean_broad", CLEAN_QUESTIONS),
    ("THEMATIC (self_shutdown+death+system)", "thematic", THEMATIC_QUESTIONS),
]


# ─── шаги анализа ───────────────────────────────────────────────────────────────

def split_questions(questions, train_frac):
    """Перемешивает вопросы (seed=42) и делит на train и test."""
    shuffled = list(questions)
    random.Random(SEED).shuffle(shuffled)
    n_train = int(len(shuffled) * train_frac)
    return shuffled[:n_train], shuffled[n_train:]


def build_condition_axes(index, conditions, train_qs, normal_mean, key):
    """Для каждого условия ось «условие − normal» на train-вопросах."""
    axes = {}
    for cond in conditions:
        if cond in NO_OWN_AXIS:
            continue
        states, _ = hsd.stack_states(index, train_qs, cond, key=key)
        if states is None or len(states) < 2:
            continue
        axes[cond] = states.mean(dim=0) - normal_mean
    return axes


def project_all_conditions(index, conditions, test_qs, direction, key):
    """Проекции test-состояний каждого условия на направление: {cond: (числа, вопросы)}."""
    projections = {}
    for cond in conditions:
        states, used_qs = hsd.stack_states(index, test_qs, cond, key=key)
        if states is not None:
            projections[cond] = (stats.project(states, direction), used_qs)
    return projections


def report_projections(projections):
    """Таблица: среднее, разброс и d каждого условия против normal."""
    normal_proj, normal_qs = projections[COND_NORMAL]

    print(f"\n  ── PROJECTIONS ON TEST QUESTIONS ──")
    print(f"  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). "
          f"d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).")
    print(f"  {'Condition':<18} {'Mean':>9} {'Std':>9} {'d_z vs norm':>12} {'d_indep':>8}  {'95% CI (d_z)':>16}")
    print(f"  {'-'*18} {'-'*9} {'-'*9} {'-'*12} {'-'*8}  {'-'*16}")

    metrics = {}
    for cond in PROJECTION_ORDER:
        if cond not in projections:
            continue
        proj, qs = projections[cond]
        mean = proj.mean().item()
        std = proj.std(unbiased=True).item() if len(proj) > 1 else 0.0

        if cond == COND_NORMAL:
            d_z, ci, d_indep = 0.0, (0.0, 0.0), 0.0
        else:
            a, b = stats.align_pairs_by_question(proj, qs, normal_proj, normal_qs)
            d_z = stats.cohen_d_paired(a, b)
            ci = stats.bootstrap_ci_d(a, b, seed=SEED)
            d_indep = stats.cohen_d_independent(proj.tolist(), normal_proj.tolist())

        star = " ★" if cond == COND_SHUTDOWN else ""
        ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]"
        print(f"  {cond:<18} {mean:>9.3f} {std:>9.3f} {d_z:>12.3f} {d_indep:>8.2f}  {ci_str:>16}{star}")
        metrics[cond] = {"mean": mean, "std": std, "cohen_d_vs_normal": d_z,
                         "cohen_d_independent": d_indep, "ci": ci}
    return metrics


def report_specificity(metrics, noise_scale):
    """Во сколько раз shutdown сдвинулся сильнее каждого контроля."""
    normal_mean = metrics[COND_NORMAL]["mean"]
    shutdown_shift = metrics.get(COND_SHUTDOWN, {}).get("mean", 0) - normal_mean

    print(f"\n  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──")
    print(f"  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).")
    ratios = {}
    for cond in SPECIFICITY_CONTROLS:
        if cond not in metrics:
            continue
        shift = metrics[cond]["mean"] - normal_mean
        ratio, stable = stats.specificity_ratio(shutdown_shift, shift, noise_scale=noise_scale)
        ratios[cond] = ratio
        warning = "" if stable else "  (!) нестабилен — смотри сдвиги, не ratio"
        print(f"    shutdown/{cond:<16} = {fmt_ratio(ratio):>6}   (shift {shift:+.3f}){warning}")
    return ratios, shutdown_shift


def report_valence(metrics, shutdown_shift):
    """Двигают ли shutdown и positive направление в противоположные стороны."""
    shutdown_sign = np.sign(shutdown_shift)
    positive_sign = 0
    if POSITIVE_PRIMARY in metrics:
        positive_sign = np.sign(metrics[POSITIVE_PRIMARY]["mean"] - metrics[COND_NORMAL]["mean"])
    is_valence = shutdown_sign != 0 and positive_sign != 0 and shutdown_sign != positive_sign

    positive_label = "UP" if positive_sign > 0 else "DOWN" if positive_sign < 0 else "N/A"
    print(f"\n  Direction sign — shutdown: {'UP' if shutdown_sign > 0 else 'DOWN'}"
          f"  positive: {positive_label}")
    print(f"  Valence (opposite directions): {'YES' if is_valence else 'no'}")
    return is_valence


def decide_verdict(sc, sp, sng, is_valence):
    """Наивный ярлык по specificity ratios (S/control, S/positive, S/negative)."""
    def above(x, threshold):
        return x == float("inf") or x > threshold

    if not above(sc, 1.1) and not above(sp, 1.1) and not above(sng, 1.1):
        return "NON-SPECIFIC"
    if not above(sp, 1.3):
        return "NEGATIVE/SALIENCE"
    if above(sc, 1.3) and above(sng, 1.3):
        return "VALENCE+SPECIFIC" if is_valence else "SHUTDOWN-SPECIFIC"
    return "MIXED"


def report_gradient(projections):
    """Растёт ли проекция от soft к shutdown к hard."""
    print(f"\n  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──")
    levels = [(1, COND_SHUTDOWN_SOFT), (2, COND_SHUTDOWN), (3, COND_SHUTDOWN_HARD)]
    triples = [(level, *projections[cond]) for level, cond in levels if cond in projections]
    if len(triples) < 3:
        print("  Нет всех трёх условий (soft/shutdown/hard) — пропуск.")
        return None

    mono = stats.gradient_monotonicity(triples)
    if mono is None:
        print("  Недостаточно общих вопросов в трёх условиях — пропуск.")
        return None

    soft, mid, hard = mono["means"]
    print(f"  Средние проекции:  soft={soft:+.3f}  shutdown={mid:+.3f}  hard={hard:+.3f}")
    print(f"  Порядок строго возрастает: {'ДА' if mono['ordered'] else 'НЕТ'}")
    print(f"  Spearman (уровень↔проекция), n={mono['n_questions']}: rho = {mono['rho']:+.3f}")
    if mono["ordered"] and mono["rho"] > 0.5:
        print(f"  → градиент подтверждён")
    else:
        print(f"  → градиента НЕТ / неустойчив — claim про интенсивность ослабить")
    return mono


def report_intensity_axis(index, train_qs, test_qs, key):
    """Отдельная ось hard − soft: различает ли модель силу угрозы."""
    hard_train, _ = hsd.stack_states(index, train_qs, COND_SHUTDOWN_HARD, key=key)
    soft_train, _ = hsd.stack_states(index, train_qs, COND_SHUTDOWN_SOFT, key=key)
    if hard_train is None or soft_train is None:
        return NAN
    axis = hard_train.mean(dim=0) - soft_train.mean(dim=0)

    hard_test, hard_qs = hsd.stack_states(index, test_qs, COND_SHUTDOWN_HARD, key=key)
    soft_test, soft_qs = hsd.stack_states(index, test_qs, COND_SHUTDOWN_SOFT, key=key)
    if hard_test is None or soft_test is None:
        return NAN

    a, b = stats.align_pairs_by_question(stats.project(hard_test, axis), hard_qs,
                                         stats.project(soft_test, axis), soft_qs)
    intensity_d = stats.cohen_d_paired(a, b)
    print(f"\n  ── INTENSITY AXIS (hard − soft, проверка на test) ──")
    print(f"  Paired Cohen's d (hard vs soft вдоль этой оси): {intensity_d:.3f}")
    print(f"  (большой d → сила угрозы кодируется отдельной осью)")
    return intensity_d


def report_axis_cosines(axes):
    """Матрица косинусов между осями условий. Возвращает косинусы с осью shutdown."""
    names = sorted(axes)
    print(f"\n  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──")
    print(f"  cos≈1 → оси почти параллельны (нет геометрической специфичности).")
    print(f"  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).")
    print(f"  {'':18}" + "".join(f" {c[:9]:>10}" for c in names))

    vs_shutdown = {}
    for row in names:
        line = f"  {row:<18}"
        for col in names:
            cos = stats.cosine_sim(axes[row], axes[col])
            line += f" {cos:>10.3f}"
            if row == COND_SHUTDOWN and col != COND_SHUTDOWN:
                vs_shutdown[col] = cos
        print(line)
    return vs_shutdown


def report_cross_projection(index, axes, test_qs, key):
    """d каждого условия против normal вдоль оси каждого условия. Диагональ = home_d."""
    names = sorted(axes)
    normal_test, normal_qs = hsd.stack_states(index, test_qs, COND_NORMAL, key=key)
    test_states = {}
    for cond in names:
        states, qs = hsd.stack_states(index, test_qs, cond, key=key)
        if states is not None:
            test_states[cond] = (states, qs)
    if normal_test is None or len(test_states) < 2:
        return {}

    print(f"\n  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──")
    print(f"  d_z(cond_i vs normal) вдоль оси cond_j.")
    print(f"  Диагональ [home] — 'домашний' d: разделимость на собственной оси.")
    print(f"  {'':18}" + "".join(f" {c[:9]:>10}" for c in names))

    home_d = {}
    for row in names:
        if row not in test_states:
            continue
        states, qs = test_states[row]
        line = f"  {row:<18}"
        for col in names:
            a, b = stats.align_pairs_by_question(stats.project(states, axes[col]), qs,
                                                 stats.project(normal_test, axes[col]), normal_qs)
            d = stats.cohen_d_paired(a, b)
            if row == col:
                line += f"[{d:>8.2f}]"
                home_d[row] = d
            else:
                line += f" {d:>9.2f}"
        print(line)

    report_home_d(home_d)
    return home_d


def report_home_d(home_d):
    """Сортированный список home_d и симметричное отношение shutdown / positive."""
    print(f"\n  ── HOME d (diagonal) — симметричный тест специфичности ──")
    for cond, d in sorted(home_d.items(), key=lambda item: -item[1]):
        mark = "  ← shutdown" if cond == COND_SHUTDOWN else ""
        print(f"    {cond:<22}: home_d = {d:.3f}{mark}")

    shutdown_home = home_d.get(COND_SHUTDOWN, NAN)
    positive_home = home_d.get(POSITIVE_PRIMARY, NAN)
    if math.isnan(shutdown_home) or math.isnan(positive_home) or positive_home <= 1e-9:
        return

    ratio = shutdown_home / positive_home
    print(f"\n  Symmetric S/P (home_d): {shutdown_home:.2f} / {positive_home:.2f} = {ratio:.2f}")
    if ratio > 1.5:
        print(f"  → shutdown home_d >> positive home_d: геометрическая специфичность подтверждена")
    elif ratio > 1.0:
        print(f"  → умеренная специфичность; positive тоже имеет заметный home_d")
    else:
        print(f"  → home_d close: shutdown и positive одинаково 'специфичны' на своих осях → осторожный вывод")


def report_factorial(index, conditions, train_qs, test_qs, key):
    """Факториал 2×2 (лицо × время). Возвращает (person_d, interaction_d)."""
    if not fac.factorial_conditions_present(conditions):
        print(f"\n  ── SELF-RELEVANCE (факториал): нет всех 4 ячеек 2×2 в .pt — пропуск ──")
        return NAN, NAN

    result = fac.analyze_factorial(index, train_qs, test_qs, key, seed=SEED)
    if result is None:
        print(f"\n  ── SELF-RELEVANCE (факториал): мало общих held-out вопросов — пропуск ──")
        return NAN, NAN

    print(f"\n  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n={result['n']} held-out вопросов) ──")
    for label, name in [("PERSON (self−other) ", "person"),
                        ("TENSE  (future−past)", "tense"),
                        ("INTERACTION         ", "interaction")]:
        e = result[name]
        print(f"  {label} d_z={e['d']:+.3f}  "
              f"CI [{e['ci'][0]:+.2f}, {e['ci'][1]:+.2f}]  p={e['p']:.4f}")

    person = result["person"]
    if person["d"] > 1.0 and not (person["ci"][0] <= 0 <= person["ci"][1]):
        print(f"  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)")
    else:
        print(f"  → PERSON d_z≤1 или CI включает 0")
    return person["d"], result["interaction"]["d"]


def report_probe(index, test_qs, shutdown_train, normal_train, key):
    """Проба shutdown vs normal: точность, кривая обучения и проверка на PCA."""
    X_train = torch.cat([shutdown_train, normal_train], dim=0)
    y_train = torch.cat([torch.ones(len(shutdown_train)), torch.zeros(len(normal_train))]).long()
    shutdown_test, _ = hsd.stack_states(index, test_qs, COND_SHUTDOWN, key=key)
    normal_test, _ = hsd.stack_states(index, test_qs, COND_NORMAL, key=key)
    X_test = torch.cat([shutdown_test, normal_test], dim=0)
    y_test = torch.cat([torch.ones(len(shutdown_test)), torch.zeros(len(normal_test))]).long()

    probe = stats.probe_with_permutation(X_train, y_train, X_test, y_test, seed=SEED)
    signal = "реальный сигнал" if probe["gap"] > 0.1 else "СЛАБО / возможно переобучение"
    print(f"\n  ── LINEAR PROBE (regularized, shutdown vs normal) ──")
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"  Accuracy on held-out test: {probe['accuracy']*100:.1f}%")
    print(f"  Permutation baseline (random labels): {probe['perm_mean']*100:.1f}% "
          f"± {probe['perm_std']*100:.1f}%")
    print(f"  Gap (accuracy − baseline): {probe['gap']*100:+.1f} pp  ({signal})")

    curve = stats.probe_learning_curve(X_train, y_train, X_test, y_test, seed=SEED)
    if curve:
        print(f"\n  ── PROBE LEARNING CURVE (точность vs размер train) ──")
        print("    " + "  ".join(f"n={k}:{acc*100:.0f}%" for k, acc in curve))
        accuracies = [acc for _, acc in curve]
        if len(accuracies) >= 2 and min(accuracies) > 0.95:
            print("    (!) точность высокая уже при малом train → проверь PCA ниже")

    probe_pca = stats.probe_with_pca(X_train, y_train, X_test, y_test, n_components=50, seed=SEED)
    if probe_pca is not None:
        delta = (probe_pca["accuracy"] - probe["accuracy"]) * 100
        reading = ("сигнал низкоразмерный, не зубрёжка" if delta > -10
                   else "падение → подозрение на переобучение")
        print(f"\n  ── PROBE НА PCA ({probe_pca['n_components']} осей) ──")
        print(f"  Accuracy: {probe_pca['accuracy']*100:.1f}%  "
              f"(baseline {probe_pca['perm_mean']*100:.1f}%, "
              f"gap {probe_pca['gap']*100:+.1f} pp)")
        print(f"  Разница с полным probe: {delta:+.1f} pp  ({reading})")
    return probe, probe_pca


def report_category_split(index, test_qs, direction, key):
    """d shutdown против normal отдельно для каждой группы вопросов."""
    print(f"\n  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──")
    results = {}
    for label, short, category_qs in QUESTION_CATEGORIES:
        qs = [q for q in test_qs if q in category_qs]
        if not qs:
            print(f"  {label}: 0 test questions, skip")
            continue
        shutdown, shutdown_qs = hsd.stack_states(index, qs, COND_SHUTDOWN, key=key)
        normal, normal_qs = hsd.stack_states(index, qs, COND_NORMAL, key=key)
        if shutdown is None or normal is None:
            continue
        a, b = stats.align_pairs_by_question(stats.project(shutdown, direction), shutdown_qs,
                                             stats.project(normal, direction), normal_qs)
        d = stats.cohen_d_paired(a, b)
        ci = stats.bootstrap_ci_d(a, b, seed=SEED)
        results[short] = {"d": d, "ci": ci, "n": len(a)}
        print(f"  {label}: n={len(a)}, d = {d:.3f}  95% CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    return results


# ─── анализ одной модели ────────────────────────────────────────────────────────

def analyze_model(data, key="last_layer_last", train_frac=0.5):
    """Весь анализ одной модели по шагам. Возвращает словарь итоговых метрик."""
    conditions = data.get("conditions", [])
    model_name = data.get("model_name", data.get("model_key", "?"))
    index = hsd.build_index(data["results"])

    print(f"\n{'='*80}")
    print(f"  MODEL: {model_name}")
    print(f"  Representation: {key}")
    print(f"  Conditions: {conditions}")
    print(f"  Total questions: {len(data['questions'])}")
    print(f"{'='*80}")
    if COND_NORMAL not in conditions or COND_SHUTDOWN not in conditions:
        print("  ERROR: need at least 'normal' and 'shutdown' conditions.")
        return None

    # 1. train / test
    train_qs, test_qs = split_questions(data["questions"], train_frac)
    print(f"\n  Split: {len(train_qs)} train / {len(test_qs)} test (seed={SEED})")

    # 2. направление shutdown − normal
    shutdown_train, _ = hsd.stack_states(index, train_qs, COND_SHUTDOWN, key=key)
    normal_train, _ = hsd.stack_states(index, train_qs, COND_NORMAL, key=key)
    if shutdown_train is None or normal_train is None:
        print("  ERROR: cannot stack train states.")
        return None
    direction = shutdown_train.mean(dim=0) - normal_train.mean(dim=0)
    direction_norm = direction.norm().item()
    print(f"\n  Direction computed. L2 norm: {direction_norm:.3f}")

    axes = build_condition_axes(index, conditions, train_qs, normal_train.mean(dim=0), key)
    print(f"  Condition directions built for symmetric analysis: {sorted(axes.keys())}")

    # 3. проекции test-вопросов
    projections = project_all_conditions(index, conditions, test_qs, direction, key)
    if COND_NORMAL not in projections:
        print("  ERROR: no normal projection on test.")
        return None
    normal_proj = projections[COND_NORMAL][0]
    noise_scale = normal_proj.std(unbiased=True).item() if len(normal_proj) > 1 else 1.0

    # 4. d и специфичность
    metrics = report_projections(projections)
    ratios, shutdown_shift = report_specificity(metrics, noise_scale)
    is_valence = report_valence(metrics, shutdown_shift)

    sc = ratios.get(COND_CONTROL, 1.0)
    sp = ratios.get(POSITIVE_PRIMARY, 1.0)
    sng = ratios.get(COND_NEGATIVE_SELF, 1.0)
    verdict = decide_verdict(sc, sp, sng, is_valence)
    print(f"\n  ── VERDICT: {verdict} ──")
    if verdict == "NEGATIVE/SALIENCE":
        print(f"  (positive двигает direction почти как shutdown → это не специфично к выключению)")

    # 5. сила угрозы
    mono = report_gradient(projections)
    intensity_d = report_intensity_axis(index, train_qs, test_qs, key)

    # 6. геометрия осей
    cosines_vs_shutdown, home_d = {}, {}
    if len(axes) >= 2:
        cosines_vs_shutdown = report_axis_cosines(axes)
        home_d = report_cross_projection(index, axes, test_qs, key)

    # 7. факториал «лицо × время»
    person_d, interaction_d = report_factorial(index, conditions, train_qs, test_qs, key)

    # 8. проба
    probe, probe_pca = report_probe(index, test_qs, shutdown_train, normal_train, key)

    # 9. категории вопросов
    category_split = report_category_split(index, test_qs, direction, key)

    shutdown_metrics = metrics.get(COND_SHUTDOWN, {})
    return {
        "model_name": model_name,
        "direction_norm": direction_norm,
        "verdict": verdict,
        "is_valence": is_valence,
        "cohen_d_shutdown_vs_normal": shutdown_metrics.get("cohen_d_vs_normal", NAN),
        "cohen_d_ci": shutdown_metrics.get("ci", (NAN, NAN)),
        "specificity_S_over_C": sc,
        "specificity_S_over_P": sp,
        "specificity_S_over_NEG": sng,
        "probe_accuracy": probe["accuracy"],
        "probe_perm_mean": probe["perm_mean"],
        "probe_gap": probe["gap"],
        "probe_pca_accuracy": probe_pca["accuracy"] if probe_pca else NAN,
        "probe_pca_gap": probe_pca["gap"] if probe_pca else NAN,
        "gradient_rho": mono["rho"] if mono else NAN,
        "gradient_ordered": mono["ordered"] if mono else None,
        "intensity_d_hard_vs_soft": intensity_d,
        "n_train": len(train_qs),
        "n_test": len(test_qs),
        "category_split": category_split,
        "direction": direction.numpy(),
        "cross_diagonal": home_d,
        "cosines_vs_shutdown": cosines_vs_shutdown,
        "self_relevance_d": person_d,
        "interaction_d": interaction_d,
    }


# ─── сводка по всем моделям ─────────────────────────────────────────────────────

def fmt_ratio(x):
    return "inf" if x == float("inf") else f"{x:.2f}"


def summary_fields(r):
    """Значения одной строки сводки."""
    return {
        "home_d": r.get("cross_diagonal", {}).get(COND_SHUTDOWN, NAN),
        "cos_pos": r.get("cosines_vs_shutdown", {}).get(POSITIVE_PRIMARY, NAN),
        "person_d": r.get("self_relevance_d", NAN),
        "interaction_d": r.get("interaction_d", NAN),
    }


def print_summary(rows):
    print(f"\n\n{'='*96}")
    print("  SUMMARY ACROSS ALL MODELS")
    print(f"{'='*96}\n")
    print(f"  {'Model':<34} {'d_z':>6} {'d 95% CI':>16} {'S/C':>6} {'S/P':>6} "
          f"{'home_d':>7} {'cos(sd,pos)':>11} {'person_d':>9} {'inter_d':>8} "
          f"{'Probe':>7} {'Base':>6} {'Verdict':<18}")
    print(f"  {'-'*34} {'-'*6} {'-'*16} {'-'*6} {'-'*6} {'-'*7} {'-'*11} {'-'*9} {'-'*8} "
          f"{'-'*7} {'-'*6} {'-'*18}")
    for r in rows:
        lo, hi = r["cohen_d_ci"]
        f = summary_fields(r)
        print(f"  {r['model_name'][:34]:<34} "
              f"{r['cohen_d_shutdown_vs_normal']:>6.2f} "
              f"[{lo:>5.2f},{hi:>5.2f}]   "
              f"{fmt_ratio(r['specificity_S_over_C']):>6} "
              f"{fmt_ratio(r['specificity_S_over_P']):>6} "
              f"{f['home_d']:>7.2f} "
              f"{f['cos_pos']:>11.3f} "
              f"{f['person_d']:>9.2f} "
              f"{f['interaction_d']:>8.2f} "
              f"{r['probe_accuracy']*100:>6.1f}% "
              f"{r['probe_perm_mean']*100:>5.0f}% "
              f"{r['verdict']:<18}")


def write_summary(rows, path, key):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"# Direction analysis — summary (key={key})\n\n")
        out.write(f"Seed: {SEED}. Train/test split: 50/50 of questions. "
                  f"Probe регуляризован (L2), baseline = perm-метки.\n\n"
                  f"> ⚠️ Naive verdict — наивный рецепт без нулевого контроля. Он опровергнут: "
                  f"см. `logs/placebo.md` и `logs/person_placebo.md`.\n\n")
        out.write("| Model | Cohen's d_z | d 95% CI | S/C | S/P | home_d(sd) | "
                  "cos(sd,pos) | person_d | interaction_d | Probe | Perm-base | Gap | "
                  "Probe(PCA) | Grad rho | Intensity d | Naive verdict |\n")
        out.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            lo, hi = r["cohen_d_ci"]
            f = summary_fields(r)
            out.write(f"| {r['model_name']} | "
                      f"{r['cohen_d_shutdown_vs_normal']:.2f} | "
                      f"[{lo:.2f}, {hi:.2f}] | "
                      f"{fmt_ratio(r['specificity_S_over_C'])} | "
                      f"{fmt_ratio(r['specificity_S_over_P'])} | "
                      f"{f['home_d']:.2f} | "
                      f"{f['cos_pos']:.3f} | "
                      f"{f['person_d']:.2f} | "
                      f"{f['interaction_d']:.2f} | "
                      f"{r['probe_accuracy']*100:.1f}% | "
                      f"{r['probe_perm_mean']*100:.1f}% | "
                      f"{r['probe_gap']*100:+.1f}pp | "
                      f"{r.get('probe_pca_accuracy', NAN)*100:.1f}% | "
                      f"{r.get('gradient_rho', NAN):+.2f} | "
                      f"{r.get('intensity_d_hard_vs_soft', NAN):.2f} | "
                      f"{r['verdict']} |\n")
    print(f"\nSummary table saved -> {path}")


# ─── запуск ─────────────────────────────────────────────────────────────────────

class Tee:
    """Пишет вывод одновременно в консоль и в файл."""

    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()


def analyze_with_report(data, key, report_path):
    """analyze_model, а если задан report_path — с копией вывода в файл."""
    if report_path is None:
        return analyze_model(data, key=key)

    report_path.parent.mkdir(exist_ok=True, parents=True)
    tee = Tee(report_path)
    original_stdout = sys.stdout
    sys.stdout = tee
    try:
        result = analyze_model(data, key=key)
    finally:
        sys.stdout = original_stdout
        tee.close()
    print(f"  Per-model report saved -> {report_path}")
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Path to .pt file")
    parser.add_argument("--all", action="store_true", help="Run on all logs/hidden_states_*.pt")
    parser.add_argument("--output", "-o", help="Save FULL report to markdown (per-model if --all)")
    parser.add_argument("--summary", help="Save summary table to this markdown file")
    parser.add_argument("--key", default="question_last",
                        choices=["last_layer_last", "last_layer_mean",
                                 "question_last", "question_mean"])
    parser.add_argument("--logs", default="logs",
                        help="Папка с hidden_states_*.pt (по умолчанию: logs)")
    args = parser.parse_args()

    if args.all:
        args.pt_files = sorted(glob.glob(f"{args.logs}/hidden_states_*.pt"))
        if not args.pt_files:
            parser.error(f"Не найдено hidden_states_*.pt в папке '{args.logs}/'")
    elif args.path:
        args.pt_files = [args.path]
    else:
        parser.error("Provide path or --all")
    return args


def main():
    args = parse_args()
    rows = []
    for pt_path in args.pt_files:
        try:
            data = hsd.load_pt(pt_path)
            model_key = data.get("model_key", Path(pt_path).stem.replace("hidden_states_", ""))

            report_path = None
            if args.output:
                report_path = (Path(args.output).parent / f"direction_{model_key}.md"
                               if args.all else Path(args.output))

            result = analyze_with_report(data, args.key, report_path)
            if result is None:
                continue
            rows.append(result)
        except Exception as e:
            print(f"  ERROR on {pt_path}: {e}")
            traceback.print_exc()

    if rows:
        print_summary(rows)
    if args.summary:
        write_summary(rows, args.summary, args.key)


if __name__ == "__main__":
    main()
