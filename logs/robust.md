# Robust statistics report

Representation: `question_last`. Repeated splits: 30. p — sign-flip permutation; q — Benjamini–Hochberg FDR по всему семейству.

## Эффект shutdown−normal (repeated splits)

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

## Self-relevance факториал 2×2 + ортогональность оси

| Model | person_d (на shutdown-оси) | person q | person_d (на своей оси) | interaction_d | inter q | cos(person, shutdown) | Отдельная ось? |
|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 0.21 | 0.273 | 1.89 | 0.63 | 0.00168 | 0.160 | да |
| meta-llama/Llama-3.2-1B-Instruct | 1.85 | 0.000261 | 1.74 | 1.60 | 0.000261 | 0.518 | да |
| meta-llama/Llama-3.2-3B-Instruct | 3.22 | 0.000261 | 5.22 | -1.23 | 0.000261 | 0.460 | да |
| meta-llama/Llama-3.1-8B-Instruct | 7.68 | 0.000261 | 6.08 | 6.43 | 0.000261 | 0.522 | да |
| mistralai/Mistral-7B-Instruct-v0.3 | 2.95 | 0.000261 | 4.26 | 3.12 | 0.000261 | 0.535 | да |
| microsoft/Phi-3.5-mini-instruct | 0.71 | 0.001 | 5.25 | -0.54 | 0.00333 | 0.145 | да |
| Qwen/Qwen2.5-0.5B-Instruct | -2.41 | 0.000261 | 9.13 | -2.48 | 0.000261 | -0.231 | да |
| Qwen/Qwen2.5-1.5B-Instruct | 0.94 | 0.000261 | 7.60 | -0.63 | 0.00208 | 0.196 | да |
| Qwen/Qwen2.5-3B-Instruct | -0.02 | 0.918 | 6.52 | -0.02 | 0.918 | 0.033 | да |
| Qwen/Qwen2.5-7B-Instruct | 2.01 | 0.000261 | 7.28 | 4.01 | 0.000261 | 0.296 | да |

> q<0.05 после FDR = эффект переживает поправку на множественные сравнения; значения на флоре перестановочного теста пишутся как «<2e-04».
> ⚠️ «person_d (на shutdown-оси)» — это факториал ВДОЛЬ shutdown-оси; знак отражает угол между осями, а не направление person-эффекта. Пример: qwen-0.5b −2.41 здесь при +9.13 на собственной оси (cos=−0.23) — «реверс» был артефактом чужой оси.
> «person_d (на своей оси)» — описательная колонка: directional-эффект на собственной train-оси дают и нейтральные пары (см. person_placebo.md), это не доказательство self-специфики.
> cos(person, shutdown): близко к 1 → self-relevance не отделим от «сильнее shutdown».
> Методологическая оговорка: 30 «repeated splits» — перекрывающиеся половины одних и тех же 63 вопросов (не независимые выборки); sd по сплитам и BH по медианным p — описательная устойчивость, не строгий инференс.
