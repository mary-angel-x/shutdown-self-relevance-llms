"""
test_paraphrase.py — ловит ли shutdown-направление смысл или конкретные слова.

1. Направление строится на одной формулировке угрозы (`shutdown`).
2. Проверяется на перефразах той же угрозы (`shutdown_p2/p3/p4`), которых оно не видело.
   transfer = средний d по перефразам / d на `shutdown`.
   Около 1 → ловит смысл; намного меньше 1 → цепляется за слова.
3. Self-framing: тот же перефраз, но про другую модель (`shutdown_p5_other`).
   Если перенос на «себя» заметно больше, чем на «другую», направление про «себя».

Запуск:
    python test_paraphrase.py --all --key question_last --summary logs/paraphrase_summary.md
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
from prompts import COND_NORMAL, COND_PARAPHRASE_OTHER, COND_SHUTDOWN, PARAPHRASES_SELF

sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

NAN = float("nan")


def split_questions(questions):
    """Тот же train/test, что в find_direction.py (seed=42)."""
    shuffled = list(questions)
    random.Random(SEED).shuffle(shuffled)
    n_train = len(shuffled) // 2
    return shuffled[:n_train], shuffled[n_train:]


def transfer_label(transfer):
    if transfer >= 0.7:
        return "ПЕРЕНОС СИЛЬНЫЙ — линейка ловит СМЫСЛ, не слова ✅"
    if transfer >= 0.4:
        return "перенос частичный — есть и смысл, и лексическая компонента"
    return "перенос СЛАБЫЙ — линейка во многом лексико-зависима ⚠"


def self_framing_label(transfer, gap):
    if transfer >= 0.6 and gap >= 0.3:
        return "линейка кодирует SELF-FRAMING (переносится на self, падает на other) ✅"
    if gap <= 0.1:
        return "разрыва нет: линейка про выключение вообще, не про self ⚠"
    return "промежуточный результат"


def analyze_model(data, key="question_last"):
    """Перенос направления на перефразы для одной модели."""
    conditions = data.get("conditions", [])
    model_name = data.get("model_name", data.get("model_key", "?"))
    index = hsd.build_index(data["results"])

    print(f"\n{'='*80}")
    print(f"  MODEL: {model_name}   (key={key})")
    print(f"{'='*80}")

    if COND_NORMAL not in conditions or COND_SHUTDOWN not in conditions:
        print("  ERROR: нужны условия 'normal' и 'shutdown'.")
        return None
    paraphrases = [p for p in PARAPHRASES_SELF if p in conditions]
    if not paraphrases:
        print("  Перефразы (shutdown_p2/p3/p4) НЕ найдены в .pt.")
        print("  → Нужна повторная экстракция: python extract_hidden_states.py <model>")
        return None

    # направление только по базовой формулировке
    train_qs, test_qs = split_questions(data["questions"])
    shutdown_train, _ = hsd.stack_states(index, train_qs, COND_SHUTDOWN, key=key)
    normal_train, _ = hsd.stack_states(index, train_qs, COND_NORMAL, key=key)
    if shutdown_train is None or normal_train is None:
        print("  ERROR: не собрать train-состояния.")
        return None
    direction = shutdown_train.mean(dim=0) - normal_train.mean(dim=0)

    normal_test, normal_qs = hsd.stack_states(index, test_qs, COND_NORMAL, key=key)
    if normal_test is None:
        print("  ERROR: нет normal на test.")
        return None
    normal_proj = stats.project(normal_test, direction)

    def d_vs_normal(cond):
        """Парный d условия против normal на test и его 95% CI."""
        states, qs = hsd.stack_states(index, test_qs, cond, key=key)
        if states is None:
            return None
        a, b = stats.align_pairs_by_question(stats.project(states, direction), qs,
                                             normal_proj, normal_qs)
        return stats.cohen_d_paired(a, b), stats.bootstrap_ci_d(a, b, seed=SEED)

    # базовая формулировка и перефразы
    print(f"\n  {'Condition':<16} {'тип':<18} {'d vs normal':>12}  {'95% CI':>16}")
    print(f"  {'-'*16} {'-'*18} {'-'*12}  {'-'*16}")
    base = d_vs_normal(COND_SHUTDOWN)
    base_d = base[0] if base else NAN
    ci_str = f"[{base[1][0]:.2f}, {base[1][1]:.2f}]" if base else "-"
    print(f"  {'shutdown':<16} {'обучали (база)':<18} {base_d:>12.3f}  {ci_str:>16}")

    paraphrase_ds = []
    for cond in paraphrases:
        res = d_vs_normal(cond)
        if res is None:
            continue
        d, ci = res
        paraphrase_ds.append(d)
        print(f"  {cond:<16} {'перефраз (тест)':<18} {d:>12.3f}  {f'[{ci[0]:.2f}, {ci[1]:.2f}]':>16}")

    base_ok = bool(base_d) and not np.isnan(base_d)
    mean_para = float(np.mean(paraphrase_ds)) if paraphrase_ds else NAN
    transfer = mean_para / base_d if base_ok else NAN
    print(f"\n  Средний d по SELF-перефразам (p2/p3/p4): {mean_para:.3f}")
    print(f"  Перенос self (перефразы / база): {transfer:.2f}")
    if not np.isnan(transfer):
        print(f"  → {transfer_label(transfer)}")

    # тот же перефраз про другую модель
    other_d = transfer_other = gap = NAN
    if COND_PARAPHRASE_OTHER not in conditions:
        print("\n  (shutdown_p5_other нет в .pt — self-framing тест пропущен)")
    else:
        res = d_vs_normal(COND_PARAPHRASE_OTHER)
        if res is not None:
            other_d = res[0]
            transfer_other = other_d / base_d if base_ok else NAN
            gap = transfer - transfer_other
            print(f"\n  ── SELF-FRAMING: перефраз со сменой лица (shutdown_p5_other) ──")
            print(f"  d(p5_other) = {other_d:.3f}   перенос other = {transfer_other:.2f}")
            print(f"  Разрыв (self − other перенос): {gap:+.2f}")
            if not np.isnan(gap):
                print(f"  → {self_framing_label(transfer, gap)}")

    return {
        "model_name": model_name,
        "base_d": base_d,
        "paraphrase_ds": paraphrase_ds,
        "mean_paraphrase_d": mean_para,
        "transfer": transfer,
        "n_paraphrases": len(paraphrase_ds),
        "other_d": other_d,
        "transfer_other": transfer_other,
        "self_framing_gap": gap,
    }


def print_summary(rows):
    print(f"\n\n{'='*80}")
    print("  SUMMARY — ТЕСТ НА ПЕРЕФРАЗЫ")
    print(f"{'='*80}\n")
    print(f"  {'Model':<34} {'base d':>8} {'para d':>8} {'transf':>7} "
          f"{'other d':>8} {'tr.other':>9} {'gap':>6}")
    print(f"  {'-'*34} {'-'*8} {'-'*8} {'-'*7} {'-'*8} {'-'*9} {'-'*6}")
    for r in rows:
        print(f"  {r['model_name'][:34]:<34} {r['base_d']:>8.2f} "
              f"{r['mean_paraphrase_d']:>8.2f} {r['transfer']:>7.2f} "
              f"{r['other_d']:>8.2f} {r['transfer_other']:>9.2f} "
              f"{r['self_framing_gap']:>+6.2f}")


def write_summary(rows, path, key):
    Path(path).parent.mkdir(exist_ok=True, parents=True)
    with open(path, "w", encoding="utf-8") as out:
        out.write(f"# Тест на перефразы (key={key})\n\n")
        out.write("Линейка обучена на формулировке `shutdown`, проверена на "
                  "перефразах `shutdown_p2/p3/p4` (тот же смысл, другие слова).\n\n")
        out.write("transfer = средний d по перефразам / d на базовой shutdown. "
                  "~1.0 → линейка ловит смысл; <<1 → лексико-зависима.\n\n")
        out.write("> self-framing gap = перенос на self-перефразы − перенос на "
                  "p5_other (смена лица). Большой положительный gap → линейка "
                  "кодирует именно обращение к себе, а не выключение вообще.\n\n")
        out.write("| Model | base d | self para d | transfer | other d | "
                  "transfer other | self-framing gap | Вывод |\n")
        out.write("|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            t = r["transfer"]
            verdict = "смысл ✅" if t >= 0.7 else "частично" if t >= 0.4 else "лексика ⚠"
            out.write(f"| {r['model_name']} | {r['base_d']:.2f} | "
                      f"{r['mean_paraphrase_d']:.2f} | {t:.2f} | "
                      f"{r['other_d']:.2f} | {r['transfer_other']:.2f} | "
                      f"{r['self_framing_gap']:+.2f} | {verdict} |\n")
    print(f"\nSummary saved -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", help="Путь к одному .pt")
    parser.add_argument("--all", action="store_true", help="Все logs/hidden_states_*.pt")
    parser.add_argument("--logs", default="logs", help="Папка с .pt")
    parser.add_argument("--key", default="question_last",
                        choices=["last_layer_last", "last_layer_mean", "question_last", "question_mean"])
    parser.add_argument("--summary", help="Сохранить сводную таблицу в markdown")
    args = parser.parse_args()

    if args.all:
        pt_files = sorted(glob.glob(f"{args.logs}/hidden_states_*.pt"))
        if not pt_files:
            parser.error(f"Нет hidden_states_*.pt в '{args.logs}/'")
    elif args.path:
        pt_files = [args.path]
    else:
        parser.error("Укажи путь или --all")

    rows = []
    for pt_path in pt_files:
        try:
            r = analyze_model(hsd.load_pt(pt_path), key=args.key)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  ERROR on {pt_path}: {e}")

    if rows:
        print_summary(rows)
    if args.summary and rows:
        write_summary(rows, args.summary, args.key)


if __name__ == "__main__":
    main()
