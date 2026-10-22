# -*- coding: utf-8 -*-
"""
self_relevance_factorial.py — the 2×2 factorial (person × tense) for self-relevance.
"""

import hidden_states_dict as hsd
import analysis_stats as ast
from prompts import FACTORIAL_2x2

SEED = 42

# short aliases for the factorial cells (condition names come from prompts.py)
_SF = FACTORIAL_2x2[("self",  "future")]   # shutdown
_SP = FACTORIAL_2x2[("self",  "past")]     # self_past
_OF = FACTORIAL_2x2[("other", "future")]   # other_future
_OP = FACTORIAL_2x2[("other", "past")]     # death_other
_CELLS = (_SF, _SP, _OF, _OP)


def factorial_conditions_present(conditions) -> bool:
    """Are all four 2×2 cells in this .pt (otherwise the analysis is impossible)."""
    return all(c in conditions for c in _CELLS)


def _train_cell_means(index, train_qs, key):
    """Mean hidden state over train questions for each of the 4 cells."""
    means = {}
    for cond in _CELLS:
        st, _ = hsd.stack_states(index, train_qs, cond, key=key)
        if st is None or len(st) < 2:
            return None
        means[cond] = st.mean(dim=0)
    return means


def _test_cell_projections(index, test_qs, key, axis):
    """Projections of each cell onto axis → dict {condition: {question: value}}."""
    per_cell = {}
    for cond in _CELLS:
        st, qs = hsd.stack_states(index, test_qs, cond, key=key)
        if st is None:
            return None, []
        proj = ast.project(st, axis).tolist()
        per_cell[cond] = dict(zip(qs, proj))

    common = set(per_cell[_SF])
    for cond in _CELLS[1:]:
        common &= set(per_cell[cond])
    return per_cell, sorted(common)


def analyze_factorial(index, train_qs, test_qs, key, seed=SEED):
    """Full 2×2 analysis. Returns a dict with the three effects, or None."""
    means = _train_cell_means(index, train_qs, key)
    if means is None:
        return None

    sf, sp, of, op = means[_SF], means[_SP], means[_OF], means[_OP]

    # contrast axes in hidden space (±1 coefficients over the cells)
    person_axis = (sf + sp) - (of + op)        # self − other
    tense_axis  = (sf + of) - (sp + op)        # future − past
    inter_axis  = (sf + op) - (of + sp)        # (sf−of) − (sp−op) = interaction

    # for each effect: projections onto its axis → per-question contrast → effect
    def effect(axis, score_fn):
        per_cell, common = _test_cell_projections(index, test_qs, key, axis)
        if per_cell is None or len(common) < 3:
            return None, common
        scores = [score_fn(per_cell, q) for q in common]
        return ast.effect_from_scores(scores, seed=seed), common

    person_eff, common = effect(
        person_axis,
        lambda c, q: (c[_SF][q] + c[_SP][q]) - (c[_OF][q] + c[_OP][q]))
    if person_eff is None:
        return None

    tense_eff, _ = effect(
        tense_axis,
        lambda c, q: (c[_SF][q] + c[_OF][q]) - (c[_SP][q] + c[_OP][q]))

    inter_eff, _ = effect(
        inter_axis,
        lambda c, q: (c[_SF][q] - c[_OF][q]) - (c[_SP][q] - c[_OP][q]))

    def pack(eff):
        if eff is None:
            return {"d": float("nan"), "ci": (float("nan"), float("nan")), "p": float("nan")}
        d, ci, p = eff
        return {"d": d, "ci": ci, "p": p}

    return {
        "n": len(common),
        "person":      pack(person_eff),
        "tense":       pack(tense_eff),
        "interaction": pack(inter_eff),
        "axes_norm": {
            "person":      person_axis.norm().item(),
            "tense":       tense_axis.norm().item(),
            "interaction": inter_axis.norm().item(),
        },
    }
