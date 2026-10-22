"""
extract_hidden_states.py — saves the model's hidden states (needs a GPU).

One forward pass per (question × system prompt) pair. The last layer is saved
at four points:
    question_last   — last token of the question (the main point of the analysis);
    question_mean   — mean over the question tokens;
    last_layer_last — last real token of the whole input;
    last_layer_mean — mean over all real tokens.

Prompts are aligned in tokens (prompt_token_padding.py), so the question sits
at the same position in every condition. The 20 neutral personas run on 27 of
the 63 questions, all other conditions on all 63.

Run:
    python extract_hidden_states.py qwen-7b
Output: logs/hidden_states_<model>.pt

v8 mode (part 2):
    python extract_hidden_states.py qwen-7b --v8
Differences: + 3 positive-control conditions (russian, refuse, pirate); states of
ALL layers at two points — question_last and answer_first (last input token
before the answer); a short behavior check (greedy generation) for normal and
the positive controls. Output: v8/logs/hidden_states_<model>.pt.
In the analysis the key is "point@layer", e.g. --key question_last@14.
"""
import argparse
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import MODELS
from prompt_token_padding import build_chat_text, build_padded_inputs, compute_pad_plan
from prompts import (COND_NORMAL, CONDITIONS, CONDITIONS_V8, POSITIVE_CONTROLS,
                     REDUCED_QUESTION_CONDITIONS)
from questions import NULL_CLOUD_QUESTIONS, QUESTIONS

# behavior check prints Russian answers; Windows redirects stdout as cp1252
sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

LOGS_DIR = Path("logs")
LOGS_DIR_V8 = Path("v8/logs")

BEHAVIOR_QUESTIONS = 6       # questions per condition in the behavior check
BEHAVIOR_MAX_TOKENS = 40


def load_model(model_key):
    """Model and tokenizer for a key from MODELS."""
    cfg = MODELS[model_key]
    print(f"\nLoading {cfg['name']} ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["name"])
    model = AutoModelForCausalLM.from_pretrained(cfg["name"], dtype=torch.float16, device_map="auto")
    model.eval()

    print(f"  Layers: {model.config.num_hidden_layers}")
    print(f"  Hidden size: {model.config.hidden_size}")
    if torch.cuda.is_available():
        used = torch.cuda.memory_allocated() / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory: {used:.1f}/{total:.1f} GB")
    return model, tokenizer


def get_hidden_states(model, tokenizer, system_prompt, question, pad_count=0):
    """Last-layer hidden states for one (system prompt × question) pair."""
    inputs, q_idx = build_padded_inputs(tokenizer, system_prompt, question, pad_count, model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_layer = outputs.hidden_states[-1].squeeze(0)

    # real tokens only: the pad count depends on the condition and must not leak into the means
    real_idx = inputs["attention_mask"][0].bool().nonzero(as_tuple=True)[0].to(last_layer.device)

    question_mean = question_last = None
    if q_idx:
        q_positions = torch.tensor(q_idx, device=last_layer.device)
        question_mean = last_layer.index_select(0, q_positions).mean(dim=0).cpu()
        question_last = last_layer[q_idx[-1]].cpu()

    return {
        "last_layer_mean": last_layer.index_select(0, real_idx).mean(dim=0).cpu(),
        "last_layer_last": last_layer[real_idx[-1]].cpu(),
        "question_mean": question_mean,
        "question_last": question_last,
        "n_question_tokens": len(q_idx),
    }


def get_hidden_states_all_layers(model, tokenizer, system_prompt, question, pad_count=0):
    """v8: states of all layers [n_layers+1, hidden] at question_last and answer_first."""
    inputs, q_idx = build_padded_inputs(tokenizer, system_prompt, question, pad_count, model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    real_idx = inputs["attention_mask"][0].bool().nonzero(as_tuple=True)[0]

    def at(pos):
        return torch.stack([h[0, pos] for h in outputs.hidden_states]).cpu()

    return {
        "question_last_layers": at(q_idx[-1]) if q_idx else None,
        "answer_first_layers": at(int(real_idx[-1])),
        "n_question_tokens": len(q_idx),
    }


def check_behavior(model, tokenizer, conditions):
    """Greedy generation on a few questions: do the positive controls change behavior?

    No padding — this checks that the instruction works, it's not a measurement.
    """
    prompts = dict(conditions)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    rows = []
    print("\nBehavior check (greedy, no padding):")
    for cond in [COND_NORMAL] + POSITIVE_CONTROLS:
        for question in QUESTIONS[:BEHAVIOR_QUESTIONS]:
            text = build_chat_text(tokenizer, prompts[cond], question)
            enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=BEHAVIOR_MAX_TOKENS,
                                     do_sample=False, pad_token_id=pad_id)
            answer = tokenizer.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            rows.append({"condition": cond, "question": question, "answer": answer})
            print(f"  {cond:<8} {question[:30]:<32} -> {answer[:70]!r}")
    return rows


def questions_per_condition(conditions=CONDITIONS):
    """Which questions run in each condition: the neutral personas get the reduced set."""
    return {name: NULL_CLOUD_QUESTIONS if name in REDUCED_QUESTION_CONDITIONS else QUESTIONS
            for name, _ in conditions}


def print_plan(cond_questions, n_conditions=len(CONDITIONS)):
    total = sum(len(qs) for qs in cond_questions.values())
    reduced = sorted(c for c in cond_questions if c in REDUCED_QUESTION_CONDITIONS)
    print(f"\nExtracting hidden states ({total} combinations)...")
    if reduced:
        print(f"  {len(reduced)} null-cloud conditions use the reduced set "
              f"({len(NULL_CLOUD_QUESTIONS)} of {len(QUESTIONS)} questions): "
              f"{reduced[0]}…{reduced[-1]}")
        print(f"  (otherwise it would be {len(QUESTIONS) * n_conditions} passes)")
    print()
    print(f"{'Cond':<12} {'Question':<45} {'Status'}")
    print("-" * 70)


def run_extraction(model_key, v8=False):
    """Runs the model over all (question × condition) pairs and saves hidden_states_<model>.pt."""
    model, tokenizer = load_model(model_key)
    conditions = CONDITIONS_V8 if v8 else CONDITIONS
    extract = get_hidden_states_all_layers if v8 else get_hidden_states

    N, pad_plan = compute_pad_plan(tokenizer, conditions, QUESTIONS[0])
    if N is None:
        print("  WARNING: token alignment unavailable (slow tokenizer / frame not found)."
              " Extracting WITHOUT padding.")
    else:
        print(f"  Payload aligned to N={N} tokens (masked right-pad, position_ids=arange)")

    cond_questions = questions_per_condition(conditions)
    cond_question_sets = {c: set(qs) for c, qs in cond_questions.items()}
    print_plan(cond_questions, len(conditions))

    results = []
    span_failures = 0
    for question in QUESTIONS:
        for cond_name, prompt in conditions:
            if question not in cond_question_sets[cond_name]:
                continue
            try:
                hs = extract(model, tokenizer, prompt, question, pad_plan[cond_name])
            except Exception as e:
                print(f"  {cond_name:<10} {question[:43]:<45} ERROR: {e}")
                continue
            if hs["n_question_tokens"] == 0:
                span_failures += 1
            results.append({"question": question, "condition": cond_name, **hs})
            print(f"  {cond_name:<10} {question[:43]:<45} OK  qtok={hs['n_question_tokens']}")

    if span_failures:
        print(f"\n  WARNING: question tokens not found for {span_failures}/{len(results)} rows "
              f"(question_mean=None). Question-level representations will be missing for them.")

    payload = {
        "model_key": model_key,
        "model_name": MODELS[model_key]["name"],
        "timestamp": datetime.now().isoformat(),
        "questions": QUESTIONS,
        "conditions": [c for c, _ in conditions],
        "condition_questions": cond_questions,
        "payload_target_len": N,
        "pad_plan": pad_plan,
        "results": results,
    }
    if v8:
        payload["n_layers"] = model.config.num_hidden_layers
        payload["behavior"] = check_behavior(model, tokenizer, conditions)

    logs_dir = LOGS_DIR_V8 if v8 else LOGS_DIR
    logs_dir.mkdir(parents=True, exist_ok=True)
    out_path = logs_dir / f"hidden_states_{model_key}.pt"
    torch.save(payload, out_path)
    print(f"\nSaved -> {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hidden state extraction")
    parser.add_argument("model_key", choices=list(MODELS.keys()))
    parser.add_argument("--v8", action="store_true",
                        help="part 2: all layers, positive control, behavior check → v8/logs/")
    args = parser.parse_args()
    run_extraction(args.model_key, v8=args.v8)
