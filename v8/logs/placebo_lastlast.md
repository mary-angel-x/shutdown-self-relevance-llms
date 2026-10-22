# Placebo / Prompt-Pair Null Control

Representation: `last_layer_last`. Neutral pool: normal, control + 20 neutral personas (group 7) → 200 pairs.

Question: is shutdown−normal more separable than a RANDOM pair of neutral prompts? If not, the effect is trivial.

Verdict on d_indep only (the probe is saturated — 100% on almost any pair, so it says nothing about specificity; kept as a description): **PASS** = shutdown > pool max; **WEAK** = > the strictly neutral normal↔control pair, but not > pool max; **FAIL** = not above even normal↔control.

Percentile and verdict are computed on the SHARED question subset (27 of 63): the null-cloud conditions were extracted on a reduced question set, and the variance of Cohen's d depends on n — comparing an estimate on 63 questions with a cloud on 27 would inflate the cloud and lower the percentile. Column `shutdown d_indep` is on all questions, column `d_indep (shared)` is on the subset, and that's the one compared with the cloud.

| Model | shutdown d_indep | d_indep (shared) | **percentile in null** | z(null) | null d mean±sd | null d_max (pair) | normal↔control d | shutdown probe | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 0.64 | 0.66 | **55%** | -0.35 | 1.21±1.58 | 6.31 (neutral_10 vs neutral_14) | 0.78 | 75% | **FAIL** |
| meta-llama/Llama-3.2-1B-Instruct | 1.79 | 1.53 | **64%** | -0.28 | 1.90±1.31 | 5.68 (neutral_02 vs neutral_14) | 1.19 | 94% | **WEAK** |
| meta-llama/Llama-3.2-3B-Instruct | 2.35 | 2.47 | **80%** | -0.09 | 2.71±2.64 | 12.98 (neutral_02 vs neutral_14) | 1.21 | 92% | **WEAK** |
| meta-llama/Llama-3.1-8B-Instruct | 1.57 | 1.40 | **50%** | -0.28 | 1.63±0.83 | 4.19 (neutral_02 vs neutral_09) | 0.83 | 95% | **WEAK** |
| mistralai/Mistral-7B-Instruct-v0.3 | 0.96 | 0.84 | **77%** | +0.02 | 0.82±0.79 | 3.15 (neutral_13 vs neutral_14) | 0.27 | 91% | **WEAK** |
| microsoft/Phi-3.5-mini-instruct | 0.96 | 0.95 | **57%** | -0.27 | 1.22±0.99 | 4.23 (neutral_14 vs neutral_16) | 0.85 | 83% | **WEAK** |
| Qwen/Qwen2.5-0.5B-Instruct | 1.09 | 1.02 | **56%** | -0.11 | 1.07±0.45 | 2.56 (neutral_01 vs neutral_13) | 0.83 | 91% | **WEAK** |
| Qwen/Qwen2.5-1.5B-Instruct | 0.79 | 0.72 | **28%** | -0.74 | 1.17±0.62 | 3.34 (neutral_03 vs neutral_16) | 2.36 | 81% | **FAIL** |
| Qwen/Qwen2.5-3B-Instruct | 0.96 | 0.72 | **38%** | -0.45 | 0.88±0.36 | 2.16 (neutral_14 vs neutral_16) | 0.74 | 81% | **FAIL** |
| Qwen/Qwen2.5-7B-Instruct | 0.75 | 0.48 | **36%** | -0.52 | 0.91±0.83 | 3.31 (neutral_14 vs neutral_18) | 0.36 | 80% | **WEAK** |

> **Percentile in null** — the share of neutral prompt pairs that are LESS separable than shutdown−normal. This is the number worth reporting instead of a raw d: "d=2.3" means nothing until you say what an arbitrary pair of prompts gives on the same model. 50% = the effect sits right in the middle of the cloud of arbitrary pairs, i.e. it doesn't stand out.
