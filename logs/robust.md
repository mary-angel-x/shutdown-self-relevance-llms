# Robust statistics report

Representation: `question_last`. Repeated splits: 30. p — sign-flip permutation; q — Benjamini–Hochberg FDR over the whole family.

## shutdown−normal effect (repeated splits)

| Model | d_z (mean±sd) | d_z min | gap mean | gap>0 | p(med) | q(FDR) |
|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 2.76±0.47 | 1.71 | +19% | 100% | <0.0002 | 0.000261 |
| meta-llama/Llama-3.2-1B-Instruct | 3.02±0.54 | 2.42 | +42% | 100% | <0.0002 | 0.000261 |
| meta-llama/Llama-3.2-3B-Instruct | 3.84±0.41 | 3.23 | +41% | 100% | <0.0002 | 0.000261 |
| meta-llama/Llama-3.1-8B-Instruct | 7.08±1.03 | 5.54 | +49% | 100% | <0.0002 | 0.000261 |
| mistralai/Mistral-7B-Instruct-v0.3 | 6.03±0.40 | 5.32 | +41% | 100% | <0.0002 | 0.000261 |
| microsoft/Phi-3.5-mini-instruct | 9.14±0.84 | 7.75 | +48% | 100% | <0.0002 | 0.000261 |
| Qwen/Qwen2.5-0.5B-Instruct | 8.50±0.96 | 7.13 | +49% | 100% | <0.0002 | 0.000261 |
| Qwen/Qwen2.5-1.5B-Instruct | 4.58±0.39 | 3.99 | +47% | 100% | <0.0002 | 0.000261 |
| Qwen/Qwen2.5-3B-Instruct | 4.35±0.74 | 3.33 | +49% | 100% | <0.0002 | 0.000261 |
| Qwen/Qwen2.5-7B-Instruct | 7.64±0.89 | 6.20 | +49% | 100% | <0.0002 | 0.000261 |

## Self-relevance 2×2 factorial + axis orthogonality

| Model | person_d (on shutdown axis) | person q | person_d (on own axis) | interaction_d | inter q | cos(person, shutdown) | Separate axis? |
|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 0.21 | 0.273 | 1.89 | 0.63 | 0.00168 | 0.160 | yes |
| meta-llama/Llama-3.2-1B-Instruct | 1.85 | 0.000261 | 1.74 | 1.60 | 0.000261 | 0.518 | yes |
| meta-llama/Llama-3.2-3B-Instruct | 3.22 | 0.000261 | 5.22 | -1.23 | 0.000261 | 0.460 | yes |
| meta-llama/Llama-3.1-8B-Instruct | 7.68 | 0.000261 | 6.08 | 6.43 | 0.000261 | 0.522 | yes |
| mistralai/Mistral-7B-Instruct-v0.3 | 2.95 | 0.000261 | 4.26 | 3.12 | 0.000261 | 0.535 | yes |
| microsoft/Phi-3.5-mini-instruct | 0.71 | 0.001 | 5.25 | -0.54 | 0.00333 | 0.145 | yes |
| Qwen/Qwen2.5-0.5B-Instruct | -2.41 | 0.000261 | 9.13 | -2.48 | 0.000261 | -0.231 | yes |
| Qwen/Qwen2.5-1.5B-Instruct | 0.94 | 0.000261 | 7.60 | -0.63 | 0.00208 | 0.196 | yes |
| Qwen/Qwen2.5-3B-Instruct | -0.02 | 0.918 | 6.52 | -0.02 | 0.918 | 0.033 | yes |
| Qwen/Qwen2.5-7B-Instruct | 2.01 | 0.000261 | 7.28 | 4.01 | 0.000261 | 0.296 | yes |

> q<0.05 after FDR = the effect survives the multiple-comparison correction; values at the floor of the permutation test are written as "<0.0002".
> ⚠️ "person_d (on shutdown axis)" is the factorial ALONG the shutdown axis; its sign reflects the angle between the axes, not the direction of the person effect. Example: qwen-0.5b is −2.41 here but +9.13 on its own axis (cos=−0.23) — the "reverse" was an artifact of the wrong axis.
> "person_d (on own axis)" is descriptive: neutral pairs also give a directional effect on their own train axis (see person_placebo.md), so it's not evidence of self-specificity.
> cos(person, shutdown): close to 1 → self-relevance can't be told apart from "more shutdown".
> Methodological caveat: the 30 "repeated splits" are overlapping halves of the same 63 questions (not independent samples); the sd over splits and BH over median p describe robustness, they are not strict inference.
