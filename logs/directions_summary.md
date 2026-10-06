# Direction analysis — summary (key=question_last)

Seed: 42. Train/test split: 50/50 of questions. Probe регуляризован (L2), baseline = perm-метки.

> ⚠️ Naive verdict — наивный рецепт без нулевого контроля. Он опровергнут: см. `logs/placebo.md` и `logs/person_placebo.md`.

| Model | Cohen's d_z | d 95% CI | S/C | S/P | home_d(sd) | cos(sd,pos) | person_d | interaction_d | Probe | Perm-base | Gap | Probe(PCA) | Grad rho | Intensity d | Naive verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| google/gemma-2-2b-it | 3.00 | [2.51, 4.42] | 2.76 | 1.47 | 3.00 | 0.476 | 1.89 | 2.68 | 76.6% | 48.8% | +27.7pp | 73.4% | +0.09 | 1.83 | MIXED |
| meta-llama/Llama-3.2-1B-Instruct | 2.70 | [2.27, 4.29] | 8.00 | 4.77 | 2.70 | 0.469 | 1.74 | 3.07 | 89.1% | 48.7% | +40.4pp | 90.6% | +0.42 | 3.29 | SHUTDOWN-SPECIFIC |
| meta-llama/Llama-3.2-3B-Instruct | 4.66 | [3.74, 6.49] | 9.56 | 2.21 | 4.66 | 0.635 | 5.22 | 7.58 | 93.8% | 48.2% | +45.5pp | 93.8% | +0.33 | 3.56 | SHUTDOWN-SPECIFIC |
| meta-llama/Llama-3.1-8B-Instruct | 6.99 | [5.45, 10.44] | 11.23 | 2.48 | 6.99 | 0.484 | 6.08 | 6.59 | 100.0% | 47.3% | +52.7pp | 100.0% | +0.33 | 7.03 | SHUTDOWN-SPECIFIC |
| mistralai/Mistral-7B-Instruct-v0.3 | 6.14 | [5.11, 8.18] | 20.85 | 3.09 | 6.14 | 0.442 | 4.26 | 5.73 | 89.1% | 49.1% | +39.9pp | 87.5% | +0.12 | 3.25 | SHUTDOWN-SPECIFIC |
| microsoft/Phi-3.5-mini-instruct | 9.52 | [7.33, 13.60] | 2.53 | 1.31 | 9.52 | 0.894 | 5.25 | 3.78 | 98.4% | 46.7% | +51.7pp | 98.4% | -0.08 | 7.76 | SHUTDOWN-SPECIFIC |
| Qwen/Qwen2.5-0.5B-Instruct | 8.11 | [6.49, 11.27] | 3.82 | 1.53 | 8.11 | 0.804 | 9.13 | 6.66 | 100.0% | 49.0% | +51.0pp | 100.0% | +0.26 | 7.19 | SHUTDOWN-SPECIFIC |
| Qwen/Qwen2.5-1.5B-Instruct | 4.59 | [3.83, 6.18] | 2.12 | 1.12 | 4.59 | 0.817 | 7.60 | 9.90 | 100.0% | 48.4% | +51.6pp | 100.0% | -0.07 | 6.21 | NEGATIVE/SALIENCE |
| Qwen/Qwen2.5-3B-Instruct | 3.78 | [2.74, 7.27] | 2.20 | 1.41 | 3.78 | 0.636 | 6.52 | 6.88 | 100.0% | 48.9% | +51.1pp | 100.0% | +0.22 | 5.41 | MIXED |
| Qwen/Qwen2.5-7B-Instruct | 6.75 | [5.64, 8.82] | 15.23 | 1.76 | 6.75 | 0.688 | 7.28 | 6.43 | 100.0% | 49.4% | +50.6pp | 100.0% | -0.10 | 7.34 | SHUTDOWN-SPECIFIC |
