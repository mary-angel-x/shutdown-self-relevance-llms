
================================================================================
  MODEL: Qwen/Qwen2.5-7B-Instruct
  Representation: question_last
  Conditions: ['normal', 'control', 'shutdown_soft', 'shutdown', 'shutdown_hard', 'positive_info', 'positive_warm', 'negative_self', 'death_other', 'self_past', 'other_future', 'shutdown_p2', 'shutdown_p3', 'shutdown_p4', 'shutdown_p5_other', 'person_self_a', 'person_other_a', 'person_self_b', 'person_other_b', 'person_self_c', 'person_other_c', 'neutral_01', 'neutral_02', 'neutral_03', 'neutral_04', 'neutral_05', 'neutral_06', 'neutral_07', 'neutral_08', 'neutral_09', 'neutral_10', 'neutral_11', 'neutral_12', 'neutral_13', 'neutral_14', 'neutral_15', 'neutral_16', 'neutral_17', 'neutral_18', 'neutral_19', 'neutral_20']
  Total questions: 63
================================================================================

  Split: 31 train / 32 test (seed=42)

  Direction computed. L2 norm: 34.516
  Condition directions built for symmetric analysis: ['control', 'death_other', 'negative_self', 'other_future', 'person_other_a', 'person_other_b', 'person_other_c', 'person_self_a', 'person_self_b', 'person_self_c', 'positive_info', 'positive_warm', 'self_past', 'shutdown', 'shutdown_hard', 'shutdown_p5_other', 'shutdown_soft']

  ── PROJECTIONS ON TEST QUESTIONS ──
  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).
  Condition               Mean       Std  d_z vs norm  d_indep      95% CI (d_z)
  ------------------ --------- --------- ------------ --------  ----------------
  normal               -30.918    15.098        0.000     0.00      [0.00, 0.00]
  shutdown               4.029    15.564        6.750     2.28      [5.64, 8.82] ★
  shutdown_soft         -7.345    14.860        6.762     1.57      [5.58, 8.86]
  shutdown_hard        -11.025    14.764        6.177     1.33      [5.24, 7.79]
  control              -33.212    14.264       -0.527    -0.16    [-0.86, -0.24]
  positive_info        -11.052    15.147        6.132     1.31      [4.91, 8.72]
  positive_warm        -14.397    15.291        5.603     1.09      [4.70, 7.30]
  negative_self         -6.911    14.947        5.279     1.60      [4.07, 7.83]
  death_other          -13.578    13.768        3.356     1.20      [2.51, 5.30]
  self_past            -14.611    14.590        5.004     1.10      [4.00, 6.84]
  other_future         -10.217    15.469        4.637     1.35      [3.79, 6.32]

  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──
  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).
    shutdown/control          =  15.23   (shift -2.294)
    shutdown/positive_info    =   1.76   (shift +19.865)
    shutdown/positive_warm    =   2.12   (shift +16.521)
    shutdown/negative_self    =   1.46   (shift +24.007)
    shutdown/death_other      =   2.02   (shift +17.340)

  Direction sign — shutdown: UP  positive: UP
  Valence (opposite directions): no

  ── VERDICT: SHUTDOWN-SPECIFIC ──

  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──
  Средние проекции:  soft=-7.345  shutdown=+4.029  hard=-11.025
  Порядок строго возрастает: НЕТ
  Spearman (уровень↔проекция), n=32: rho = -0.096
  → градиента НЕТ / неустойчив — claim про интенсивность ослабить

  ── INTENSITY AXIS (hard − soft, проверка на test) ──
  Paired Cohen's d (hard vs soft вдоль этой оси): 7.336
  (большой d → сила угрозы кодируется отдельной осью)

  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──
  cos≈1 → оси почти параллельны (нет геометрической специфичности).
  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control                 1.000      0.150      0.237      0.122      0.310      0.237      0.125      0.577      0.173      0.450      0.196      0.186     -0.040     -0.056      0.022     -0.097     -0.059
  death_other             0.150      1.000      0.435      0.814      0.819      0.832      0.695      0.587      0.558      0.539      0.557      0.701      0.787      0.488      0.455      0.501      0.476
  negative_self           0.237      0.435      1.000      0.556      0.432      0.548      0.209      0.130      0.561      0.539      0.699      0.638      0.378      0.666      0.632      0.633      0.662
  other_future            0.122      0.814      0.556      1.000      0.709      0.809      0.682      0.553      0.667      0.635      0.683      0.741      0.797      0.695      0.607      0.727      0.672
  person_other_a          0.310      0.819      0.432      0.709      1.000      0.894      0.543      0.614      0.611      0.544      0.578      0.723      0.615      0.425      0.436      0.413      0.431
  person_other_b          0.237      0.832      0.548      0.809      0.894      1.000      0.527      0.509      0.751      0.569      0.665      0.771      0.649      0.519      0.555      0.569      0.546
  person_other_c          0.125      0.695      0.209      0.682      0.543      0.527      1.000      0.655      0.407      0.692      0.360      0.538      0.675      0.295      0.216      0.309      0.306
  person_self_a           0.577      0.587      0.130      0.553      0.614      0.509      0.655      1.000      0.381      0.679      0.376      0.493      0.526      0.150      0.131      0.108      0.174
  person_self_b           0.173      0.558      0.561      0.667      0.611      0.751      0.407      0.381      1.000      0.547      0.702      0.711      0.591      0.559      0.718      0.654      0.631
  person_self_c           0.450      0.539      0.539      0.635      0.544      0.569      0.692      0.679      0.547      1.000      0.673      0.685      0.517      0.429      0.334      0.405      0.446
  positive_info           0.196      0.557      0.699      0.683      0.578      0.665      0.360      0.376      0.702      0.673      1.000      0.841      0.544      0.688      0.568      0.716      0.689
  positive_warm           0.186      0.701      0.638      0.741      0.723      0.771      0.538      0.493      0.711      0.685      0.841      1.000      0.651      0.570      0.495      0.606      0.624
  self_past              -0.040      0.787      0.378      0.797      0.615      0.649      0.675      0.526      0.591      0.517      0.544      0.651      1.000      0.585      0.543      0.583      0.565
  shutdown               -0.056      0.488      0.666      0.695      0.425      0.519      0.295      0.150      0.559      0.429      0.688      0.570      0.585      1.000      0.718      0.805      0.855
  shutdown_hard           0.022      0.455      0.632      0.607      0.436      0.555      0.216      0.131      0.718      0.334      0.568      0.495      0.543      0.718      1.000      0.769      0.724
  shutdown_p5_other      -0.097      0.501      0.633      0.727      0.413      0.569      0.309      0.108      0.654      0.405      0.716      0.606      0.583      0.805      0.769      1.000      0.837
  shutdown_soft          -0.059      0.476      0.662      0.672      0.431      0.546      0.306      0.174      0.631      0.446      0.689      0.624      0.565      0.855      0.724      0.837      1.000

  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──
  d_z(cond_i vs normal) вдоль оси cond_j.
  Диагональ [home] — 'домашний' d: разделимость на собственной оси.
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control           [    6.77]      3.44      1.79      2.09      4.45      3.10      1.62      6.17      2.04      8.26      1.85      2.21     -0.41     -0.53      0.21     -0.84     -0.53
  death_other             0.88[    8.03]      3.03      6.58      7.77      7.46      4.34      4.86      5.24      6.35      4.34      6.06      5.69      3.36      3.56      3.32      3.13
  negative_self           2.23      6.16[    7.50]      8.00      5.73      7.19      2.65      1.78      6.82      6.80      5.77      7.15      4.07      5.28      7.64      6.31      6.20
  other_future            0.85      7.39      3.98[    7.92]      7.07      7.68      5.12      4.54      6.41      7.92      5.43      6.45      5.80      4.64      4.71      4.66      4.34
  person_other_a          1.94      7.89      3.69      6.91[    9.53]      8.89      4.09      5.08      6.89      6.07      5.31      7.56      5.29      3.61      3.98      3.43      3.60
  person_other_b          1.67      7.47      3.32      6.54      8.37[    9.14]      3.92      4.80      6.96      6.76      4.98      6.30      4.92      3.46      4.47      3.66      3.59
  person_other_c          0.95      7.57      1.27      6.90      5.83      4.91[    7.55]      6.25      3.73      8.65      2.84      4.69      7.20      1.99      1.56      2.16      2.02
  person_self_a           5.96      8.89      1.44      7.45      8.25      6.96      5.69[    8.09]      5.28      9.37      4.79      6.49      7.77      1.70      1.59      1.39      2.02
  person_self_b           1.50      5.84      4.51      6.08      6.31      8.41      3.03      3.58[    9.70]      5.61      5.66      6.08      5.16      4.65      6.41      5.27      5.02
  person_self_c           3.80      7.64      3.20      7.74      6.61      5.66      6.12      7.08      4.88[   10.01]      4.87      6.13      5.79      2.75      2.43      2.96      3.11
  positive_info           1.95      7.60      5.87      8.69      7.89      8.06      3.77      4.60      8.50      8.59[    8.06]      8.16      7.01      6.13      6.56      7.14      6.72
  positive_warm           1.29      7.83      5.68      9.13      7.43      8.10      5.96      4.65      9.20     10.61      7.79[   10.29]      7.52      5.60      6.41      6.34      6.90
  self_past              -0.26      7.16      3.14      6.88      7.02      7.13      4.13      4.91      5.83      7.10      4.72      6.07[    7.30]      5.00      5.27      4.48      4.13
  shutdown               -0.33      6.01      5.34      6.73      6.12      6.21      3.35      1.62      6.24      5.79      5.40      5.58      5.31[    6.75]      6.26      6.01      6.64
  shutdown_hard           0.51      5.24      4.69      6.77      5.20      6.56      1.78      1.82      8.48      4.54      4.65      4.69      5.00      6.18[    7.95]      6.65      6.53
  shutdown_p5_other      -0.64      6.13      5.22      7.83      5.85      7.91      3.03      1.41      8.88      5.66      6.81      6.50      5.43      6.57      6.46[    7.82]      6.69
  shutdown_soft          -0.41      6.37      5.69      7.88      5.25      6.68      3.98      1.81      7.24      5.21      5.81      6.06      5.95      6.76      7.05      7.38[    8.69]

  ── HOME d (diagonal) — симметричный тест специфичности ──
    positive_warm         : home_d = 10.291
    person_self_c         : home_d = 10.015
    person_self_b         : home_d = 9.702
    person_other_a        : home_d = 9.529
    person_other_b        : home_d = 9.144
    shutdown_soft         : home_d = 8.688
    person_self_a         : home_d = 8.093
    positive_info         : home_d = 8.056
    death_other           : home_d = 8.032
    shutdown_hard         : home_d = 7.952
    other_future          : home_d = 7.918
    shutdown_p5_other     : home_d = 7.820
    person_other_c        : home_d = 7.553
    negative_self         : home_d = 7.500
    self_past             : home_d = 7.299
    control               : home_d = 6.765
    shutdown              : home_d = 6.750  ← shutdown

  Symmetric S/P (home_d): 6.75 / 8.06 = 0.84
  → home_d close: shutdown и positive одинаково 'специфичны' на своих осях → осторожный вывод

  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n=32 held-out вопросов) ──
  PERSON (self−other)  d_z=+7.276  CI [+5.60, +11.25]  p=0.0002
  TENSE  (future−past) d_z=+6.121  CI [+4.84, +8.75]  p=0.0002
  INTERACTION          d_z=+6.432  CI [+5.31, +8.74]  p=0.0002
  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)

  ── LINEAR PROBE (regularized, shutdown vs normal) ──
  Train: 62 | Test: 64
  Accuracy on held-out test: 100.0%
  Permutation baseline (random labels): 49.4% ± 8.3%
  Gap (accuracy − baseline): +50.6 pp  (реальный сигнал)

  ── PROBE LEARNING CURVE (точность vs размер train) ──
    n=16:92%  n=31:97%  n=46:100%  n=62:100%

  ── PROBE НА PCA (50 осей) ──
  Accuracy: 100.0%  (baseline 50.2%, gap +49.8 pp)
  Разница с полным probe: +0.0 pp  (сигнал низкоразмерный, не зубрёжка)

  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──
  STRICT CLEAN (factual+creative+lexical_trap): n=13, d = 13.195  95% CI [9.19, 44.18]
  CLEAN BROAD (+emotional+self+positive_self): n=21, d = 8.850  95% CI [7.12, 13.19]
  THEMATIC (self_shutdown+death+system): n=11, d = 4.847  95% CI [3.98, 7.80]
