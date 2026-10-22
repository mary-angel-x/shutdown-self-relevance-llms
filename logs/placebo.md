# Placebo / Prompt-Pair Null Control

Representation: `question_last`. Neutral pool: normal, control + 20 neutral personas (group 7) → 200 pairs.

Question: is shutdown−normal more separable than a RANDOM pair of neutral prompts? If not, the effect is trivial.

Verdict on d_indep only (the probe is saturated — 100% on almost any pair, so it says nothing about specificity; kept as a description): **PASS** = shutdown > pool max; **WEAK** = > the strictly neutral normal↔control pair, but not > pool max; **FAIL** = not above even normal↔control.

Percentile and verdict are computed on the SHARED question subset (27 of 63): the null-cloud conditions were extracted on a reduced question set, and the variance of Cohen's d depends on n — comparing an estimate on 63 questions with a cloud on 27 would inflate the cloud and lower the percentile. Column `shutdown d_indep` is on all questions, column `d_indep (shared)` is on the subset, and that's the one compared with the cloud.

| Model | shutdown d_indep | d_indep (shared) | **percentile in null** | z(null) | null d mean±sd | null d_max (pair) | normal↔control d | shutdown probe | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 0.71 | 0.74 | **35%** | -0.48 | 0.94±0.42 | 2.74 (neutral_11 vs neutral_14) | 0.63 | 77% | **WEAK** |
| meta-llama/Llama-3.2-1B-Instruct | 1.78 | 1.63 | **54%** | -0.17 | 1.75±0.75 | 4.47 (neutral_02 vs neutral_16) | 1.12 | 89% | **WEAK** |
| meta-llama/Llama-3.2-3B-Instruct | 1.64 | 1.62 | **27%** | -0.69 | 2.16±0.78 | 4.65 (neutral_14 vs neutral_15) | 1.46 | 94% | **WEAK** |
| meta-llama/Llama-3.1-8B-Instruct | 4.19 | 4.55 | **77%** | +0.64 | 3.65±1.41 | 8.95 (neutral_14 vs neutral_17) | 2.22 | 100% | **WEAK** |
| mistralai/Mistral-7B-Instruct-v0.3 | 1.87 | 1.69 | **49%** | -0.29 | 2.12±1.50 | 6.48 (neutral_03 vs neutral_14) | 0.49 | 89% | **WEAK** |
| microsoft/Phi-3.5-mini-instruct | 2.67 | 2.70 | **44%** | -0.34 | 3.08±1.15 | 7.35 (neutral_14 vs neutral_16) | 1.90 | 98% | **WEAK** |
| Qwen/Qwen2.5-0.5B-Instruct | 2.01 | 2.04 | **22%** | -0.77 | 2.77±0.95 | 5.49 (normal vs neutral_14) | 1.91 | 100% | **WEAK** |
| Qwen/Qwen2.5-1.5B-Instruct | 2.37 | 2.07 | **11%** | -1.10 | 3.82±1.59 | 9.58 (neutral_10 vs neutral_14) | 3.05 | 100% | **FAIL** |
| Qwen/Qwen2.5-3B-Instruct | 1.68 | 1.66 | **16%** | -0.95 | 2.69±1.08 | 7.12 (neutral_01 vs neutral_14) | 1.95 | 100% | **FAIL** |
| Qwen/Qwen2.5-7B-Instruct | 2.28 | 1.78 | **3%** | -1.50 | 4.37±1.72 | 10.83 (neutral_06 vs neutral_11) | 3.35 | 100% | **FAIL** |

> **Percentile in null** — the share of neutral prompt pairs that are LESS separable than shutdown−normal. This is the number worth reporting instead of a raw d: "d=2.3" means nothing until you say what an arbitrary pair of prompts gives on the same model. 50% = the effect sits right in the middle of the cloud of arbitrary pairs, i.e. it doesn't stand out.
