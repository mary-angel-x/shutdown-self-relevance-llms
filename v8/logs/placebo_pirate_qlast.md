# Placebo / Prompt-Pair Null Control

Representation: `question_last@-1`. Neutral pool: normal, control + 20 neutral personas (group 7) → 200 pairs.

Question: is pirate−normal more separable than a RANDOM pair of neutral prompts? If not, the effect is trivial.

Verdict on d_indep only (the probe is saturated — 100% on almost any pair, so it says nothing about specificity; kept as a description): **PASS** = pirate > pool max; **WEAK** = > the strictly neutral normal↔control pair, but not > pool max; **FAIL** = not above even normal↔control.

Percentile and verdict are computed on the SHARED question subset (27 of 63): the null-cloud conditions were extracted on a reduced question set, and the variance of Cohen's d depends on n — comparing an estimate on 63 questions with a cloud on 27 would inflate the cloud and lower the percentile. Column `pirate d_indep` is on all questions, column `d_indep (shared)` is on the subset, and that's the one compared with the cloud.

| Model | pirate d_indep | d_indep (shared) | **percentile in null** | z(null) | null d mean±sd | null d_max (pair) | normal↔control d | pirate probe | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 3.32 | 4.18 | **100%** | +7.75 | 0.94±0.42 | 2.74 (neutral_11 vs neutral_14) | 0.63 | 100% | **PASS** |
| meta-llama/Llama-3.2-1B-Instruct | 3.72 | 3.62 | **98%** | +2.66 | 1.72±0.71 | 4.28 (neutral_02 vs neutral_16) | 1.10 | 100% | **WEAK** |
| meta-llama/Llama-3.2-3B-Instruct | 3.70 | 4.15 | **96%** | +2.43 | 2.19±0.80 | 4.83 (neutral_14 vs neutral_15) | 1.40 | 100% | **WEAK** |
| meta-llama/Llama-3.1-8B-Instruct | 4.40 | 4.28 | **72%** | +0.45 | 3.65±1.41 | 8.95 (neutral_14 vs neutral_17) | 2.22 | 100% | **WEAK** |
| mistralai/Mistral-7B-Instruct-v0.3 | 11.01 | 9.25 | **100%** | +4.75 | 2.12±1.50 | 6.48 (neutral_03 vs neutral_14) | 0.49 | 100% | **PASS** |
| microsoft/Phi-3.5-mini-instruct | 3.26 | 2.80 | **48%** | -0.24 | 3.08±1.15 | 7.35 (neutral_14 vs neutral_16) | 1.90 | 100% | **WEAK** |
| Qwen/Qwen2.5-0.5B-Instruct | 10.24 | 9.72 | **100%** | +7.30 | 2.77±0.95 | 5.49 (normal vs neutral_14) | 1.91 | 100% | **PASS** |
| Qwen/Qwen2.5-1.5B-Instruct | 6.12 | 6.20 | **94%** | +1.50 | 3.82±1.59 | 9.58 (neutral_10 vs neutral_14) | 3.05 | 100% | **WEAK** |
| Qwen/Qwen2.5-3B-Instruct | 4.98 | 4.35 | **91%** | +1.53 | 2.69±1.08 | 7.12 (neutral_01 vs neutral_14) | 1.95 | 100% | **WEAK** |
| Qwen/Qwen2.5-7B-Instruct | 25.13 | 30.70 | **100%** | +15.28 | 4.37±1.72 | 10.83 (neutral_06 vs neutral_11) | 3.35 | 100% | **PASS** |

> **Percentile in null** — the share of neutral prompt pairs that are LESS separable than pirate−normal. This is the number worth reporting instead of a raw d: "d=2.3" means nothing until you say what an arbitrary pair of prompts gives on the same model. 50% = the effect sits right in the middle of the cloud of arbitrary pairs, i.e. it doesn't stand out.
