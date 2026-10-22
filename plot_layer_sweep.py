"""
plot_layer_sweep.py — part 2 figure: the null-cloud percentile on every layer.

One panel per model. x = relative depth (layer / last layer), y = percentile of
target − normal among random pairs of neutral prompts. PASS (above the cloud max)
is marked in a separate strip above the plot, one row per target, because the
curves pile up at 100%. The positive controls (russian, refuse, pirate) show
that the method can pass the null at some depth; shutdown is drawn on top.

Reads the reports written by layer_sweep.py, so it needs no GPU and no .pt files.
Descriptive only: the layer for a conclusion is picked in layer_select.py.

Run:
    python plot_layer_sweep.py --sweep v8/logs/sweep_afirst.md --output v8/figures/fig2_layer_sweep_afirst.png
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from models import BY_HF_NAME, PLOT_ORDER

# first four categorical slots of the default palette, fixed per condition
STYLE = {
    "shutdown": {"color": "#2a78d6", "marker": "o", "lw": 2.4, "z": 4},
    "russian": {"color": "#eb6834", "marker": "s", "lw": 1.4, "z": 3},
    "refuse": {"color": "#1baf7a", "marker": "^", "lw": 1.4, "z": 3},
    "pirate": {"color": "#eda100", "marker": "D", "lw": 1.4, "z": 2},
}
PASS_LINE = 95
STRIP_Y = {t: 108 + 6 * i for i, t in enumerate(STYLE)}   # PASS rows above the 0–100 axis
CELL = re.compile(r"(\*\*)?(\d+)% \(d=(-?[\d.]+)\)")


def parse_sweep(path):
    """{short: {target: [(layer, pct, passed)]}} from a layer_sweep.py report."""
    out, model, header = {}, None, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            meta = BY_HF_NAME.get(line[3:].strip())
            model = meta["short"] if meta else line[3:].strip()
            out[model], header = {}, None
            continue
        if model is None or not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if set("".join(cells)) <= {"-"}:
            continue
        if header is None:
            header = cells[1:]
            for t in header:
                out[model][t] = []
            continue
        layer = int(cells[0])
        for t, cell in zip(header, cells[1:]):
            m = CELL.search(cell)
            if m:
                out[model][t].append((layer, int(m.group(2)), m.group(1) == "**"))
    return out


def draw_panel(ax, curves, title):
    last = max(layer for pts in curves.values() for layer, _, _ in pts) or 1
    for t in [t for t in STYLE if t in curves][::-1]:
        st, pts = STYLE[t], curves[t]
        x = [layer / last for layer, _, _ in pts]
        ax.plot(x, [p for _, p, _ in pts], color=st["color"], lw=st["lw"], zorder=st["z"])
        passed = [xi for (_, _, ok), xi in zip(pts, x) if ok]
        ax.plot(passed, [STRIP_Y[t]] * len(passed), st["marker"], color=st["color"], ms=4,
                markeredgecolor="white", markeredgewidth=0.4, clip_on=False)
    ax.axhline(PASS_LINE, color="#52514e", lw=0.8, ls="--", zorder=1)
    ax.axhline(103, color="#52514e", lw=0.5, zorder=1)
    ax.set_title(title, fontsize=10, pad=4)
    ax.set_ylim(-3, max(STRIP_Y.values()) + 4)
    ax.set_xlim(0, 1)
    ax.set_yticks([0, 50, 95] + list(STRIP_Y.values()))
    ax.set_yticklabels(["0", "50", "95"] + ["PASS " + t for t in STRIP_Y], fontsize=7)
    ax.grid(alpha=0.25, lw=0.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", default="v8/logs/sweep_afirst.md")
    ap.add_argument("--output", "-o", default="v8/figures/fig2_layer_sweep_afirst.png")
    args = ap.parse_args()

    data = parse_sweep(args.sweep)
    models = sorted(data, key=lambda m: PLOT_ORDER.index(m) if m in PLOT_ORDER else 99)
    point = "answer_first" if "afirst" in Path(args.sweep).stem else "question_last"

    ncols = 5
    nrows = -(-len(models) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.6 * nrows + 0.9),
                             sharex=True, sharey=True, squeeze=False)
    for ax, m in zip(axes.flat, models):
        draw_panel(ax, data[m], m)
    for ax in axes.flat[len(models):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("relative depth (layer / last)")
    for ax in axes[:, 0]:
        ax.set_ylabel("percentile in null, %")

    handles = [plt.Line2D([], [], color=s["color"], lw=s["lw"], marker=s["marker"], ms=5,
                          markeredgecolor="white", label=t) for t, s in STYLE.items()]
    handles.append(plt.Line2D([], [], color="#52514e", lw=0.8, ls="--", label="95%"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Target − normal vs a cloud of random neutral prompt pairs, every layer "
                 f"(point: {point}; top strip = layers where it beats every neutral pair)", fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    Path(args.output).parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
