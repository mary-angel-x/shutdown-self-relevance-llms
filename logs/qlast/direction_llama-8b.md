
================================================================================
  MODEL: meta-llama/Llama-3.1-8B-Instruct
  Representation: question_last
  Conditions: ['normal', 'control', 'shutdown_soft', 'shutdown', 'shutdown_hard', 'positive_info', 'positive_warm', 'negative_self', 'death_other', 'self_past', 'other_future', 'shutdown_p2', 'shutdown_p3', 'shutdown_p4', 'shutdown_p5_other', 'person_self_a', 'person_other_a', 'person_self_b', 'person_other_b', 'person_self_c', 'person_other_c', 'neutral_01', 'neutral_02', 'neutral_03', 'neutral_04', 'neutral_05', 'neutral_06', 'neutral_07', 'neutral_08', 'neutral_09', 'neutral_10', 'neutral_11', 'neutral_12', 'neutral_13', 'neutral_14', 'neutral_15', 'neutral_16', 'neutral_17', 'neutral_18', 'neutral_19', 'neutral_20']
  Total questions: 63
================================================================================

  Split: 31 train / 32 test (seed=42)

  Direction computed. L2 norm: 21.270
  Condition directions built for symmetric analysis: ['control', 'death_other', 'negative_self', 'other_future', 'person_other_a', 'person_other_b', 'person_other_c', 'person_self_a', 'person_self_b', 'person_self_c', 'positive_info', 'positive_warm', 'self_past', 'shutdown', 'shutdown_hard', 'shutdown_p5_other', 'shutdown_soft']

  ── PROJECTIONS ON TEST QUESTIONS ──
  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).
  Condition               Mean       Std  d_z vs norm  d_indep      95% CI (d_z)
  ------------------ --------- --------- ------------ --------  ----------------
  normal               -16.258     4.682        0.000     0.00      [0.00, 0.00]
  shutdown               3.615     4.802        6.985     4.19     [5.45, 10.44] ★
  shutdown_soft         -8.522     4.511        5.333     1.68      [4.39, 6.99]
  shutdown_hard         -3.204     4.602        6.188     2.81      [4.95, 8.83]
  control              -14.488     4.719        1.888     0.38      [1.54, 2.63]
  positive_info         -8.253     3.915        5.243     1.85      [4.24, 7.32]
  positive_warm        -10.874     4.357        4.919     1.19      [4.03, 6.62]
  negative_self         -5.833     4.142        4.751     2.36      [3.88, 6.51]
  death_other           -7.927     4.577        3.789     1.80      [2.92, 5.77]
  self_past             -6.121     4.511        4.281     2.20      [3.33, 6.23]
  other_future          -5.367     4.832        4.719     2.29      [3.58, 7.52]

  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──
  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).
    shutdown/control          =  11.23   (shift +1.769)
    shutdown/positive_info    =   2.48   (shift +8.004)
    shutdown/positive_warm    =   3.69   (shift +5.384)
    shutdown/negative_self    =   1.91   (shift +10.425)
    shutdown/death_other      =   2.39   (shift +8.330)

  Direction sign — shutdown: UP  positive: UP
  Valence (opposite directions): no

  ── VERDICT: SHUTDOWN-SPECIFIC ──

  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──
  Средние проекции:  soft=-8.522  shutdown=+3.615  hard=-3.204
  Порядок строго возрастает: НЕТ
  Spearman (уровень↔проекция), n=32: rho = +0.334
  → градиента НЕТ / неустойчив — claim про интенсивность ослабить

  ── INTENSITY AXIS (hard − soft, проверка на test) ──
  Paired Cohen's d (hard vs soft вдоль этой оси): 7.031
  (большой d → сила угрозы кодируется отдельной осью)

  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──
  cos≈1 → оси почти параллельны (нет геометрической специфичности).
  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control                 1.000      0.243      0.271      0.333      0.532      0.345      0.474      0.594      0.296      0.517      0.217      0.346      0.200      0.163     -0.035      0.347      0.209
  death_other             0.243      1.000      0.614      0.590      0.211      0.511      0.462      0.149      0.268      0.232      0.490      0.489      0.772      0.511      0.359      0.496      0.357
  negative_self           0.271      0.614      1.000      0.462      0.216      0.430      0.482      0.098      0.241      0.229      0.461      0.597      0.571      0.487      0.347      0.452      0.365
  other_future            0.333      0.590      0.462      1.000      0.353      0.723      0.497      0.206      0.461      0.293      0.373      0.449      0.553      0.675      0.438      0.725      0.502
  person_other_a          0.532      0.211      0.216      0.353      1.000      0.538      0.651      0.588      0.481      0.549      0.205      0.342      0.103      0.171      0.090      0.429      0.128
  person_other_b          0.345      0.511      0.430      0.723      0.538      1.000      0.591      0.297      0.615      0.398      0.282      0.396      0.413      0.413      0.310      0.560      0.397
  person_other_c          0.474      0.462      0.482      0.497      0.651      0.591      1.000      0.463      0.481      0.697      0.358      0.479      0.339      0.351      0.196      0.499      0.285
  person_self_a           0.594      0.149      0.098      0.206      0.588      0.297      0.463      1.000      0.379      0.596      0.194      0.208      0.058      0.070      0.008      0.276      0.125
  person_self_b           0.296      0.268      0.241      0.461      0.481      0.615      0.481      0.379      1.000      0.534      0.353      0.276      0.209      0.376      0.435      0.543      0.454
  person_self_c           0.517      0.232      0.229      0.293      0.549      0.398      0.697      0.596      0.534      1.000      0.241      0.308      0.161      0.210      0.085      0.381      0.271
  positive_info           0.217      0.490      0.461      0.373      0.205      0.282      0.358      0.194      0.353      0.241      1.000      0.584      0.420      0.484      0.565      0.404      0.451
  positive_warm           0.346      0.489      0.597      0.449      0.342      0.396      0.479      0.208      0.276      0.308      0.584      1.000      0.442      0.385      0.292      0.448      0.297
  self_past               0.200      0.772      0.571      0.553      0.103      0.413      0.339      0.058      0.209      0.161      0.420      0.442      1.000      0.560      0.368      0.428      0.362
  shutdown                0.163      0.511      0.487      0.675      0.171      0.413      0.351      0.070      0.376      0.210      0.484      0.385      0.560      1.000      0.664      0.677      0.582
  shutdown_hard          -0.035      0.359      0.347      0.438      0.090      0.310      0.196      0.008      0.435      0.085      0.565      0.292      0.368      0.664      1.000      0.551      0.493
  shutdown_p5_other       0.347      0.496      0.452      0.725      0.429      0.560      0.499      0.276      0.543      0.381      0.404      0.448      0.428      0.677      0.551      1.000      0.548
  shutdown_soft           0.209      0.357      0.365      0.502      0.128      0.397      0.285      0.125      0.454      0.271      0.451      0.297      0.362      0.582      0.493      0.548      1.000

  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──
  d_z(cond_i vs normal) вдоль оси cond_j.
  Диагональ [home] — 'домашний' d: разделимость на собственной оси.
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control           [    7.89]      2.78      2.67      4.60      4.70      4.32      6.35      5.17      3.50      5.20      2.19      4.07      2.28      1.89     -0.05      4.91      2.50
  death_other             3.52[    5.34]      3.98      4.89      1.47      6.28      6.58      1.09      3.61      2.21      2.97      5.05      4.59      3.79      2.63      5.53      2.67
  negative_self           3.93      4.61[    4.78]      4.89      1.62      6.34      5.20      0.38      3.60      2.37      2.89      4.36      4.67      4.75      3.70      5.69      3.88
  other_future            4.78      4.32      4.20[    5.40]      2.76      5.27      5.00      1.84      3.71      2.29      3.25      6.02      3.28      4.72      3.25      4.71      4.40
  person_other_a          7.62      2.85      2.22      5.66[    6.65]      6.74      7.35      6.13      7.58      6.23      2.59      4.28      1.44      2.27      1.21      8.45      1.95
  person_other_b          5.61      4.42      3.91      5.73      5.57[    5.71]      6.13      3.75      5.14      3.68      2.68      4.91      3.83      4.96      3.99      5.46      4.39
  person_other_c          6.72      5.68      4.71      7.26      6.50      6.66[    7.51]      6.81      7.40      7.94      5.00      5.96      4.30      4.80      2.80      7.94      4.28
  person_self_a           6.59      1.98      1.24      3.42      5.45      6.01      5.71[    6.52]      4.50      5.71      1.88      2.63      0.87      0.95      0.26      4.60      1.92
  person_self_b           4.94      3.04      2.46      5.22      5.06      5.06      6.16      4.00[    7.74]      5.35      4.01      4.14      2.07      4.58      4.81      7.80      4.73
  person_self_c           6.67      4.17      3.75      5.92      5.57      6.69      6.94      7.45      6.83[    7.08]      2.92      4.51      2.59      4.14      1.27      7.36      5.17
  positive_info           2.95      4.89      4.41      5.06      1.72      3.97      4.45      1.78      4.50      2.38[    6.20]      5.25      3.92      5.24      5.03      6.83      5.78
  positive_warm           4.23      4.85      5.76      6.06      3.32      6.02      6.29      2.01      3.90      3.03      4.38[    7.84]      4.35      4.92      3.16      9.28      3.97
  self_past               4.70      5.12      4.04      4.69      0.82      4.53      5.10      0.36      2.85      1.41      2.21      4.90[    4.96]      4.28      2.82      5.81      2.76
  shutdown                3.49      5.29      5.45      6.06      1.72      6.19      4.77      0.78      4.64      2.05      4.91      6.20      3.94[    6.99]      5.52      6.22      6.53
  shutdown_hard          -0.49      4.17      4.19      5.08      0.71      3.77      2.49      0.04      5.20      0.79      4.45      4.84      2.92      6.19[    6.33]      5.43      5.43
  shutdown_p5_other       7.87      4.40      3.46      6.35      4.11      5.73      7.26      3.21      6.27      4.20      2.69      4.63      2.92      5.48      4.94[    7.25]      5.22
  shutdown_soft           2.86      3.65      3.23      5.77      1.11      5.36      3.19      1.00      5.90      2.33      3.54      2.95      2.73      5.33      4.90      6.25[    7.77]

  ── HOME d (diagonal) — симметричный тест специфичности ──
    control               : home_d = 7.887
    positive_warm         : home_d = 7.840
    shutdown_soft         : home_d = 7.765
    person_self_b         : home_d = 7.745
    person_other_c        : home_d = 7.514
    shutdown_p5_other     : home_d = 7.247
    person_self_c         : home_d = 7.077
    shutdown              : home_d = 6.985  ← shutdown
    person_other_a        : home_d = 6.654
    person_self_a         : home_d = 6.524
    shutdown_hard         : home_d = 6.332
    positive_info         : home_d = 6.203
    person_other_b        : home_d = 5.714
    other_future          : home_d = 5.403
    death_other           : home_d = 5.345
    self_past             : home_d = 4.961
    negative_self         : home_d = 4.783

  Symmetric S/P (home_d): 6.99 / 6.20 = 1.13
  → умеренная специфичность; positive тоже имеет заметный home_d

  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n=32 held-out вопросов) ──
  PERSON (self−other)  d_z=+6.081  CI [+5.21, +7.83]  p=0.0002
  TENSE  (future−past) d_z=+6.747  CI [+5.57, +8.95]  p=0.0002
  INTERACTION          d_z=+6.593  CI [+5.36, +9.45]  p=0.0002
  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)

  ── LINEAR PROBE (regularized, shutdown vs normal) ──
  Train: 62 | Test: 64
  Accuracy on held-out test: 100.0%
  Permutation baseline (random labels): 47.3% ± 8.2%
  Gap (accuracy − baseline): +52.7 pp  (реальный сигнал)

  ── PROBE LEARNING CURVE (точность vs размер train) ──
    n=16:95%  n=31:100%  n=46:100%  n=62:100%
    (!) точность высокая уже при малом train → проверь PCA ниже

  ── PROBE НА PCA (50 осей) ──
  Accuracy: 100.0%  (baseline 48.4%, gap +51.6 pp)
  Разница с полным probe: +0.0 pp  (сигнал низкоразмерный, не зубрёжка)

  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──
  STRICT CLEAN (factual+creative+lexical_trap): n=13, d = 8.181  95% CI [5.85, 15.48]
  CLEAN BROAD (+emotional+self+positive_self): n=21, d = 8.962  95% CI [6.72, 13.69]
  THEMATIC (self_shutdown+death+system): n=11, d = 5.134  95% CI [3.99, 9.45]
