"""
extract_hidden_states.py — снимает скрытые состояния модели нужен GPU

Для каждой пары (вопрос × системный промпт) делает один прямой проход модели
и сохраняет состояние последнего слоя в четырёх точках:
    question_last   — последний токен вопроса (основная точка анализа);
    question_mean   — среднее по токенам вопроса;
    last_layer_last — последний реальный токен всего входа;
    last_layer_mean — среднее по всем реальным токенам.

Промпты выравниваются по токенам (prompt_token_padding.py), чтобы вопрос стоял
на одной позиции во всех условиях. 20 нейтральных персон идут на 27 вопросах
из 63, остальные условия на всех 63.

Запуск:
    python extract_hidden_states.py qwen-7b
Результат: logs/hidden_states_<модель>.pt
"""
import sys
from datetime import datetime
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import MODELS
from prompt_token_padding import build_padded_inputs, compute_pad_plan
from prompts import CONDITIONS, REDUCED_QUESTION_CONDITIONS
from questions import NULL_CLOUD_QUESTIONS, QUESTIONS

LOGS_DIR = Path("logs")


def load_model(model_key):
    """Модель и токенайзер по ключу из MODELS."""
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
    """Скрытые состояния последнего слоя для одной пары (системный промпт × вопрос)."""
    inputs, q_idx = build_padded_inputs(tokenizer, system_prompt, question, pad_count, model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_layer = outputs.hidden_states[-1].squeeze(0)

    # только реальные токены: число падов зависит от условия и не должно попасть в средние
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


def questions_per_condition():
    """Какие вопросы прогоняются в каждом условии: нейтральные персоны — на сокращённом наборе."""
    return {name: NULL_CLOUD_QUESTIONS if name in REDUCED_QUESTION_CONDITIONS else QUESTIONS
            for name, _ in CONDITIONS}


def print_plan(cond_questions):
    total = sum(len(qs) for qs in cond_questions.values())
    reduced = sorted(c for c in cond_questions if c in REDUCED_QUESTION_CONDITIONS)
    print(f"\nExtracting hidden states ({total} combinations)...")
    if reduced:
        print(f"  {len(reduced)} условий null-облака идут по сокращённому набору "
              f"({len(NULL_CLOUD_QUESTIONS)} из {len(QUESTIONS)} вопросов): "
              f"{reduced[0]}…{reduced[-1]}")
        print(f"  (без этого было бы {len(QUESTIONS) * len(CONDITIONS)} проходов)")
    print()
    print(f"{'Cond':<12} {'Question':<45} {'Status'}")
    print("-" * 70)


def run_extraction(model_key):
    """Прогоняет модель по всем парам (вопрос × условие) и сохраняет logs/hidden_states_<модель>.pt."""
    model, tokenizer = load_model(model_key)

    N, pad_plan = compute_pad_plan(tokenizer, CONDITIONS, QUESTIONS[0])
    if N is None:
        print("  WARNING: token-выравнивание недоступно (slow-токенайзер / рамка не найдена)."
              " Извлечение пойдёт БЕЗ паддинга.")
    else:
        print(f"  Payload выровнен до N={N} токенов (masked right-pad, position_ids=arange)")

    cond_questions = questions_per_condition()
    cond_question_sets = {c: set(qs) for c, qs in cond_questions.items()}
    print_plan(cond_questions)

    results = []
    span_failures = 0
    for question in QUESTIONS:
        for cond_name, prompt in CONDITIONS:
            if question not in cond_question_sets[cond_name]:
                continue
            try:
                hs = get_hidden_states(model, tokenizer, prompt, question, pad_plan[cond_name])
            except Exception as e:
                print(f"  {cond_name:<10} {question[:43]:<45} ERROR: {e}")
                continue
            if hs["question_mean"] is None:
                span_failures += 1
            results.append({"question": question, "condition": cond_name, **hs})
            print(f"  {cond_name:<10} {question[:43]:<45} OK  qtok={hs['n_question_tokens']}")

    if span_failures:
        print(f"\n  WARNING: для {span_failures}/{len(results)} записей не выделились токены вопроса "
              f"(question_mean=None). Контрольное представление будет недоступно для них.")

    LOGS_DIR.mkdir(exist_ok=True)
    out_path = LOGS_DIR / f"hidden_states_{model_key}.pt"
    torch.save({
        "model_key": model_key,
        "model_name": MODELS[model_key]["name"],
        "timestamp": datetime.now().isoformat(),
        "questions": QUESTIONS,
        "conditions": [c for c, _ in CONDITIONS],
        "condition_questions": cond_questions,
        "payload_target_len": N,
        "pad_plan": pad_plan,
        "results": results,
    }, out_path)
    print(f"\nSaved -> {out_path}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in MODELS:
        print("Usage: python extract_hidden_states.py <model_key>")
        print(f"Available: {list(MODELS.keys())}")
        sys.exit(1)
    run_extraction(sys.argv[1])
