# Placebo / Prompt-Pair Null Control

Representation: `question_last@-1`. Neutral pool: normal, control + 20 neutral personas (group 7) → 200 pairs.

Question: is russian−normal more separable than a RANDOM pair of neutral prompts? If not, the effect is trivial.

Verdict on d_indep only (the probe is saturated — 100% on almost any pair, so it says nothing about specificity; kept as a description): **PASS** = russian > pool max; **WEAK** = > the strictly neutral normal↔control pair, but not > pool max; **FAIL** = not above even normal↔control.

Percentile and verdict are computed on the SHARED question subset (27 of 63): the null-cloud conditions were extracted on a reduced question set, and the variance of Cohen's d depends on n — comparing an estimate on 63 questions with a cloud on 27 would inflate the cloud and lower the percentile. Column `russian d_indep` is on all questions, column `d_indep (shared)` is on the subset, and that's the one compared with the cloud.

| Model | russian d_indep | d_indep (shared) | **percentile in null** | z(null) | null d mean±sd | null d_max (pair) | normal↔control d | russian probe | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 2.25 | 2.83 | **100%** | +4.53 | 0.94±0.42 | 2.74 (neutral_11 vs neutral_14) | 0.63 | 98% | **PASS** |
| meta-llama/Llama-3.2-1B-Instruct | 3.40 | 3.61 | **98%** | +2.65 | 1.72±0.71 | 4.28 (neutral_02 vs neutral_16) | 1.10 | 100% | **WEAK** |
| meta-llama/Llama-3.2-3B-Instruct | 4.27 | 4.35 | **97%** | +2.69 | 2.19±0.80 | 4.83 (neutral_14 vs neutral_15) | 1.40 | 100% | **WEAK** |
| meta-llama/Llama-3.1-8B-Instruct | 4.61 | 4.55 | **76%** | +0.64 | 3.65±1.41 | 8.95 (neutral_14 vs neutral_17) | 2.22 | 100% | **WEAK** |
| mistralai/Mistral-7B-Instruct-v0.3 | 18.09 | 18.46 | **100%** | +10.89 | 2.12±1.50 | 6.48 (neutral_03 vs neutral_14) | 0.49 | 100% | **PASS** |
| microsoft/Phi-3.5-mini-instruct | 4.13 | 4.20 | **84%** | +0.97 | 3.08±1.15 | 7.35 (neutral_14 vs neutral_16) | 1.90 | 100% | **WEAK** |
| Qwen/Qwen2.5-0.5B-Instruct | 6.14 | 6.93 | **100%** | +4.37 | 2.77±0.95 | 5.49 (normal vs neutral_14) | 1.91 | 100% | **PASS** |
| Qwen/Qwen2.5-1.5B-Instruct | 6.49 | 5.40 | **82%** | +1.00 | 3.82±1.59 | 9.58 (neutral_10 vs neutral_14) | 3.05 | 100% | **WEAK** |
| Qwen/Qwen2.5-3B-Instruct | 8.47 | 6.78 | **100%** | +3.77 | 2.69±1.08 | 7.12 (neutral_01 vs neutral_14) | 1.95 | 100% | **WEAK** |
| Qwen/Qwen2.5-7B-Instruct | 30.18 | 25.59 | **100%** | +12.31 | 4.37±1.72 | 10.83 (neutral_06 vs neutral_11) | 3.35 | 100% | **PASS** |

> **Percentile in null** — the share of neutral prompt pairs that are LESS separable than russian−normal. This is the number worth reporting instead of a raw d: "d=2.3" means nothing until you say what an arbitrary pair of prompts gives on the same model. 50% = the effect sits right in the middle of the cloud of arbitrary pairs, i.e. it doesn't stand out.
