# -*- coding: utf-8 -*-
"""
hidden_states_dict.py — loading .pt files and access to the saved hidden states.
"""

import torch


VALID_KEYS = ("last_layer_mean", "last_layer_last", "question_mean", "question_last")

# v8: states for every layer, key is "point@layer", e.g. "question_last@14".
# Layer = index into outputs.hidden_states (0 is the embeddings, -1 the last layer).
LAYERED_POINTS = ("question_last", "answer_first")


def parse_key(key: str):
    """'question_last@14' → ('question_last_layers', 14); 'question_last' → ('question_last', None)."""
    if "@" not in key:
        return key, None
    point, layer = key.split("@", 1)
    if point not in LAYERED_POINTS:
        raise ValueError(f"Unknown point '{point}', available: {LAYERED_POINTS}")
    return f"{point}_layers", int(layer)


def is_valid_key(key: str) -> bool:
    try:
        field, _ = parse_key(key)
    except ValueError:
        return False
    return field in VALID_KEYS or field.endswith("_layers")


def load_pt(path) -> dict:
    """Loads a .pt file saved by extract_hidden_states.py."""
    return torch.load(path, map_location="cpu", weights_only=False)


def build_index(results: list) -> dict:
    """Builds {(question, condition): row} for O(1) lookup."""
    index_dict = {}
    for r in results:
        key = (r["question"], r["condition"])
        if key not in index_dict:                   # keep the first one we see
            index_dict[key] = r
    return index_dict


def get_state(index_dict: dict, question: str, condition: str,key: str):
    """One hidden state for (question, condition)."""
    row = index_dict.get((question, condition))
    if row is None:
        return None
    field, layer = parse_key(key)
    val = row.get(field)
    if val is None:  # no such key (old .pt) or the representation wasn't computed
        return None
    if layer is not None:
        val = val[layer]
    return val.float()


def common_questions(index_dict: dict, conditions, questions, key: str) -> list:
    """Questions that have a state in ALL of the given conditions."""
    out = []
    for q in questions:
        if all(get_state(index_dict, q, c, key=key) is not None for c in conditions):
            out.append(q)
    return out


def stack_states(index_dict: dict, questions, condition: str,key: str):
    """Stacks a [n_used, hidden_size] matrix for a list of questions in one condition."""
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
