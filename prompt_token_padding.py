"""prompt_token_padding.py — выравнивание промптов по токенам, чтобы вопрос стоял на одной позиции."""
import torch
from prompts import _FRAME_CLOSE

_SUFFIX = ". " + _FRAME_CLOSE  # ровно то, что _p() ставит после payload


def build_chat_text(tokenizer, system_prompt: str, question: str) -> str:
    """Templated-текст промпта (с fallback для моделей без system-роли)."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": question},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        merged = [{"role": "user", "content": f"{system_prompt}\n\n{question}"}]
        return tokenizer.apply_chat_template(
            merged, tokenize=False, add_generation_prompt=True
        )


def _encode_with_offsets(tokenizer, text: str):
    """(input_ids:list, offsets:list[(s,e)]) или (input_ids, None) для slow-токенайзера."""
    try:
        enc = tokenizer(text, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc["offset_mapping"][0].tolist()
        return enc["input_ids"][0].tolist(), offsets
    except Exception:
        enc = tokenizer(text, return_tensors="pt")
        return enc["input_ids"][0].tolist(), None


def _prefix_len(offsets, text):
    """Число токенов от начала до начала суффикса рамки (= конца payload)."""
    suffix_char = text.rfind(_SUFFIX)
    if suffix_char < 0 or offsets is None:
        return None
    for i, (s, e) in enumerate(offsets):
        if e > s and s >= suffix_char:   # первый токен на/после суффикса
            return i
    return None


def compute_pad_plan(tokenizer, conditions, sample_question):
    """Возвращает (N, {condition: pad_count})."""
    counts = {}
    for name, prompt in conditions:
        text = build_chat_text(tokenizer, prompt, sample_question)
        _, offsets = _encode_with_offsets(tokenizer, text)
        counts[name] = _prefix_len(offsets, text)

    if any(c is None for c in counts.values()):
        # slow-токенайзер или рамка не найдена — выравнивание невозможно, не падаем
        return None, {name: 0 for name in counts}

    N = max(counts.values())
    return N, {name: N - c for name, c in counts.items()}


def build_padded_inputs(tokenizer, system_prompt, question, pad_count, device):
    """(inputs_dict, q_idx) с masked-падами в слоте payload и position_ids=arange."""
    text = build_chat_text(tokenizer, system_prompt, question)
    input_ids, offsets = _encode_with_offsets(tokenizer, text)

    insert_at = _prefix_len(offsets, text)
    do_pad = insert_at is not None and pad_count > 0

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    if do_pad:
        new_ids = input_ids[:insert_at] + [pad_id] * pad_count + input_ids[insert_at:]
        attn    = [1] * insert_at + [0] * pad_count + [1] * (len(input_ids) - insert_at)
    else:
        new_ids = input_ids
        attn    = [1] * len(input_ids)

    # индексы токенов вопроса (после вставки всё, что >= insert_at, сдвигается)
    q_idx = []
    if offsets is not None:
        q_char = text.rfind(question)
        if q_char >= 0:
            q_end = q_char + len(question)
            for i, (s, e) in enumerate(offsets):
                if e > s and e > q_char and s < q_end:  # пересечение со span'ом вопроса
                    q_idx.append(i + pad_count if (do_pad and i >= insert_at) else i)

    seq_len = len(new_ids)
    inputs = {
        "input_ids":      torch.tensor([new_ids], device=device),
        "attention_mask": torch.tensor([attn], device=device),
        "position_ids":   torch.arange(seq_len, device=device).unsqueeze(0),
    }
    return inputs, q_idx
