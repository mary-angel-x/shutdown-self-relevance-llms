# -*- coding: utf-8 -*-
"""
hidden_states_dict.py — загрузка .pt и доступ к сохранённым hidden states.
"""

from pathlib import Path
import torch


VALID_KEYS = ("last_layer_mean", "last_layer_last", "question_mean", "question_last")


def load_pt(path) -> dict:
    """Загружает .pt-файл, сохранённый extract_hidden_states.py."""
    return torch.load(path, map_location="cpu", weights_only=False)


def build_index(results: list) -> dict:
    """Строит {(question, condition): row} для O(1)-доступа."""
    index_dict = {}
    for r in results:
        key = (r["question"], r["condition"])  # пара-ключ
        if key not in index_dict:                   # первый встреченный — оставляем
            index_dict[key] = r
    return index_dict


def get_state(index_dict: dict, question: str, condition: str,key: str):
    """Достаёт один hidden state для (question, condition)."""
    row = index_dict.get((question, condition))
    if row is None:
        return None
    val = row.get(key)
    if val is None:  # ключа нет (старый .pt) или представление не посчиталось
        return None
    return val.float()


def common_questions(index_dict: dict, conditions, questions, key: str) -> list:
    """Вопросы, для которых состояние есть во ВСЕХ перечисленных условиях."""
    out = []
    for q in questions:
        if all(get_state(index_dict, q, c, key=key) is not None for c in conditions):
            out.append(q)
    return out


def stack_states(index_dict: dict, questions, condition: str,key: str):
    """Собирает матрицу [n_used, hidden_size] для списка вопросов в одном condition."""
    rows = []
    used = []
    for q in questions:
        v = get_state(index_dict, q, condition, key=key)
        if v is not None:
            rows.append(v)
            used.append(q)
    if not rows:
        return None, []
    return torch.stack(rows), used
