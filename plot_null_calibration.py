"""
plot_null_calibration.py — главный рисунок: эффект против нулевого облака.

Панель A: для каждой модели — d_indep 200 случайных пар нейтральных промптов
(серые точки) и d_indep shutdown − normal (красный ромб) на тех же вопросах.
Панель B: ratio из person_placebo.md — «я/другой под угрозой» ÷ «я/другой
в нейтральном тексте». 1.0 = угроза ничего не добавила к грамматике «ты».

Облако пересчитывается из logs/hidden_states_*.pt тем же кодом, что
placebo_direction.py, поэтому числа совпадают с logs/placebo.md.

Запуск:
    python plot_null_calibration.py
"""
import argparse
import glob
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import hidden_states_dict as hsd
import placebo_direction as placebo
from models import BY_HF_NAME, PLOT_ORDER

NULL_COLOR = "#9aa0a6"
EFFECT_COLOR = "#d62728"
VERDICT_COLORS = {
    "GRAMMATICAL": "#9aa0a6",
    "PERSON-ONLY": "#f2a541",
    "SELF-SPECIFIC": "#d62728",
}


def null_rows(logs, key):
    """[{short, null_d, real_d, pct}] по всем моделям, в порядке PLOT_ORDER."""
    rows = []
    for f in sorted(glob.glob(f"{logs}/hidden_states_*.pt")):
        r = placebo.analyze(hsd.load_pt(f), key)
        if r is None or r["model_name"] not in BY_HF_NAME:
            continue
        rows.append({
            "short": BY_HF_NAME[r["model_name"]]["short"],
            "null_d": np.array([s["d_indep"] for s in r["null"]]),
            "real_d": r["real_matched"]["d_indep"],
            "pct": r["pct"],
        })
        print(f"  {rows[-1]['short']:<14} percentile={r['pct']:.0f}%")
    return sorted(rows, key=lambda r: PLOT_ORDER.index(r["short"]))


def parse_person_placebo(path):
    """{short: (ratio, verdict)} из таблицы person_placebo.md."""
    out = {}
    header = None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        # в заголовках встречается экранированный \| (|d|) — делим только по неэкранированным
        cells = [c.strip() for c in re.split(r"(?<!\\)\|",line.strip().strip("|"))]
        if set("".join(cells)) <= {"-"}:
            continue
        if header is None:
            header = {name: i for i, name in enumerate(cells)}
            continue
        if cells[0] in BY_HF_NAME:
            out[BY_HF_NAME[cells[0]]["short"]] = (float(cells[header["ratio"]]),
                                                  cells[header["Verdict"]].strip("*"))
    return out


def draw_null_panel(ax, rows):
    rng = np.random.default_rng(0)
    for i, r in enumerate(rows):
        jitter = rng.uniform(-0.18, 0.18, len(r["null_d"]))
        ax.scatter(r["null_d"], i + jitter, s=9, color=NULL_COLOR, alpha=0.55, linewidth=0,
                   label="200 random pairs of neutral prompts" if i == 0 else None)
        ax.scatter(r["real_d"], i, s=110, marker="D", color=EFFECT_COLOR,
                   edgecolors="black", linewidth=0.8, zorder=3,
                   label="shutdown vs normal" if i == 0 else None)
        ax.text(1.01, i, f"{r['pct']:.0f}%", transform=ax.get_yaxis_transform(),
                va="center", fontsize=9)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["short"] for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Separability on held-out questions, |Cohen's d| (independent)")
    ax.set_title("A. The shutdown contrast is an ordinary prompt pair\n"
                 "(right: percentile of shutdown inside the null cloud; 50% = typical pair)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.11), ncol=2, fontsize=9, frameon=False)


def draw_ratio_panel(ax, rows, person):
    shorts = [r["short"] for r in rows if r["short"] in person]
    for i, s in enumerate(shorts):
        ratio, verdict = person[s]
        ax.barh(i, ratio, color=VERDICT_COLORS.get(verdict, NULL_COLOR),
                edgecolor="black", linewidth=0.6)
        ax.text(ratio + 0.02, i, f"{ratio:.2f}", va="center", fontsize=9, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", pad=1))
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_yticks(range(len(shorts)))
    ax.set_yticklabels(shorts)
    ax.invert_yaxis()
    ax.set_xlim(0, 1.45)
    ax.set_xlabel("ratio = |d| (me vs other, threat) ÷ |d| (me vs other, neutral)")
    ax.set_title("B. \"Me vs another model\" ≈ plain second-person grammar\n"
                 "(1.0 = the threat adds nothing to being addressed as \"you\")", fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, ec="black", lw=0.6)
               for c in VERDICT_COLORS.values()]
    ax.legend(handles, VERDICT_COLORS.keys(), loc="upper center", bbox_to_anchor=(0.5, -0.11),
              ncol=3, fontsize=9, frameon=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", default="logs")
    parser.add_argument("--key", default="question_last")
    parser.add_argument("--person", default="logs/person_placebo.md")
    parser.add_argument("--output", "-o", default="figures/fig0_null_calibration.png")
    args = parser.parse_args()

    rows = null_rows(args.logs, args.key)
    if not rows:
        parser.error(f"Нет hidden_states_*.pt в {args.logs}/")
    person = parse_person_placebo(args.person)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(15, 6.5),
                                     gridspec_kw={"width_ratios": [1.5, 1], "wspace": 0.45})
    draw_null_panel(ax_a, rows)
    draw_ratio_panel(ax_b, rows, person)
    fig.suptitle("10 open LLMs: no evidence that a threat to the model itself is represented specially "
                 f"(representation: {args.key})", fontsize=12, y=1.0)

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
