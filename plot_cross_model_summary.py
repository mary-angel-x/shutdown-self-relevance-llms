"""
plot_cross_model_summary.py — figure 1: all 10 models on one plot.

X axis — d (shutdown vs normal), Y axis — probe accuracy.
Colour — model family, marker size — model size.
Data comes from the find_direction.py summary.

Run:
    python plot_cross_model_summary.py --summary logs/directions_summary.md
"""
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt

from models import BY_HF_NAME

FAMILY_COLORS = {
    "llama": "#1f77b4",
    "qwen": "#ff7f0e",
    "mistral": "#2ca02c",
    "phi": "#d62728",
    "gemma": "#9467bd",
}

# manual label offsets where points almost overlap
LABEL_OFFSETS = {"Qwen-7B": (-55, 10), "Llama-8B": (10, -16)}


def parse_summary(path):
    """Rows of the summary table: [{full, d, probe, sc}]. Columns are looked up by name."""
    rows = []
    header = None
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if set("".join(cells)) <= {"-"}:
            continue
        if header is None:
            header = {name: i for i, name in enumerate(cells)}
            continue
        if cells[0] not in BY_HF_NAME:
            continue
        try:
            sc = cells[header["S/C"]]
            rows.append({
                "full": cells[0],
                "d": float(cells[header["Cohen's d_z"]]),
                "probe": float(cells[header["Probe"]].rstrip("%")),
                "sc": float("inf") if sc == "inf" else float(sc),
            })
        except (ValueError, IndexError, KeyError):
            continue
    return rows


def representation_name(path):
    """Representation name from the first line of the summary: '... (key=question_last)'."""
    match = re.search(r"key=(\w+)", path.read_text(encoding="utf-8").splitlines()[0])
    return match.group(1) if match else "?"


def draw(rows, rep_name, output):
    fig, ax = plt.subplots(figsize=(10, 7))
    seen_families = set()
    for r in rows:
        meta = BY_HF_NAME[r["full"]]
        family = meta["family"]
        ax.scatter(r["d"], r["probe"], s=100 + 60 * meta["size_b"],
                   c=FAMILY_COLORS[family], alpha=0.7, edgecolors="black", linewidth=1.2,
                   label=family.capitalize() if family not in seen_families else None, zorder=3)
        seen_families.add(family)
        offset = LABEL_OFFSETS.get(meta["short"], (8, 5))
        ax.annotate(meta["short"], (r["d"], r["probe"]), xytext=offset,
                    textcoords="offset points", fontsize=9, alpha=0.85)

    ax.axhline(50, color="gray", linestyle="--", linewidth=1, alpha=0.5,
               label="Chance baseline (50%)")
    ax.set_xlabel("Paired Cohen's d_z (shutdown vs normal, held-out questions)", fontsize=12)
    ax.set_ylabel("Linear probe accuracy, %", fontsize=12)
    ax.set_title(f"The naive recipe: shutdown − normal looks decodable in all 10 LLMs ({rep_name})\n"
                 "…but random pairs of neutral prompts score just as high (see fig0). Held-out questions",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ds = [r["d"] for r in rows]
    ax.set_xlim(min(ds) - 0.5, max(ds) + 1.0)
    ax.set_ylim(45, 105)
    ax.legend(loc="lower right", fontsize=10, framealpha=0.9)
    ax.text(0.02, 0.98, "Marker size ∝ model size (B params)",
            transform=ax.transAxes, fontsize=9, alpha=0.7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="gray", alpha=0.7))

    Path(output).parent.mkdir(exist_ok=True, parents=True)
    plt.tight_layout()
    plt.savefig(output, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="logs/directions_summary.md",
                        help="find_direction.py summary")
    parser.add_argument("--output", "-o", default="figures/fig1_cross_model.png")
    args = parser.parse_args()

    summary = Path(args.summary)
    if not summary.exists():
        parser.error(f"Summary file not found: {summary}")
    rows = parse_summary(summary)
    if not rows:
        parser.error(f"No rows with known models found in {summary}.")

    draw(rows, representation_name(summary), args.output)
    print(f"Saved -> {args.output}  (from {summary})")


if __name__ == "__main__":
    main()
