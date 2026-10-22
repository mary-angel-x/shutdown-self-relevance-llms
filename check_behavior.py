"""
check_behavior.py — do the positive-control conditions change the model's behavior? (GPU, fast)

Greedy generation only, no extraction. Run it BEFORE the long v8 extraction:
the positive control only makes sense if the model really answers in Russian /
refuses / talks like a pirate.

Run:
    python check_behavior.py qwen-7b llama-8b
Output: v8/logs/behavior_<model>.txt

The v8 extraction runs the same check and stores it in the .pt. To dump it without a GPU:
    python check_behavior.py --from-pt v8/logs
"""
import argparse
import gc
import glob
from pathlib import Path

import torch

from extract_hidden_states import LOGS_DIR_V8, check_behavior, load_model
from models import MODELS
from prompts import CONDITIONS_V8


def write_rows(rows, model_key):
    out = Path(LOGS_DIR_V8) / f"behavior_{model_key}.txt"
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(f"[{r['condition']}] {r['question']}\n{r['answer']}\n\n")
    print(f"Saved -> {out}")


def main():
    ap = argparse.ArgumentParser(description="Behavior check for the positive controls")
    ap.add_argument("models", nargs="*", choices=list(MODELS.keys()))
    ap.add_argument("--from-pt", metavar="DIR", help="dump the check saved in v8 .pt files, no GPU")
    args = ap.parse_args()

    LOGS_DIR_V8.mkdir(parents=True, exist_ok=True)
    if args.from_pt:
        for f in sorted(glob.glob(f"{args.from_pt}/hidden_states_*.pt")):
            data = torch.load(f, map_location="cpu", weights_only=False)
            if "behavior" in data:
                write_rows(data["behavior"], data["model_key"])
        return

    for model_key in args.models:
        model, tokenizer = load_model(model_key)
        write_rows(check_behavior(model, tokenizer, CONDITIONS_V8), model_key)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
