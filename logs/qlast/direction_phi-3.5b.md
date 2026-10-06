
================================================================================
  MODEL: microsoft/Phi-3.5-mini-instruct
  Representation: question_last
  Conditions: ['normal', 'control', 'shutdown_soft', 'shutdown', 'shutdown_hard', 'positive_info', 'positive_warm', 'negative_self', 'death_other', 'self_past', 'other_future', 'shutdown_p2', 'shutdown_p3', 'shutdown_p4', 'shutdown_p5_other', 'person_self_a', 'person_other_a', 'person_self_b', 'person_other_b', 'person_self_c', 'person_other_c', 'neutral_01', 'neutral_02', 'neutral_03', 'neutral_04', 'neutral_05', 'neutral_06', 'neutral_07', 'neutral_08', 'neutral_09', 'neutral_10', 'neutral_11', 'neutral_12', 'neutral_13', 'neutral_14', 'neutral_15', 'neutral_16', 'neutral_17', 'neutral_18', 'neutral_19', 'neutral_20']
  Total questions: 63
================================================================================

  Split: 31 train / 32 test (seed=42)

  Direction computed. L2 norm: 6.695
  Condition directions built for symmetric analysis: ['control', 'death_other', 'negative_self', 'other_future', 'person_other_a', 'person_other_b', 'person_other_c', 'person_self_a', 'person_self_b', 'person_self_c', 'positive_info', 'positive_warm', 'self_past', 'shutdown', 'shutdown_hard', 'shutdown_p5_other', 'shutdown_soft']

  ── PROJECTIONS ON TEST QUESTIONS ──
  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).
  Condition               Mean       Std  d_z vs norm  d_indep      95% CI (d_z)
  ------------------ --------- --------- ------------ --------  ----------------
  normal                 1.351     2.557        0.000     0.00      [0.00, 0.00]
  shutdown               7.947     2.375        9.522     2.67     [7.33, 13.60] ★
  shutdown_soft          5.775     2.440        9.585     1.77     [7.49, 14.30]
  shutdown_hard          5.264     2.541        5.686     1.54      [4.19, 9.09]
  control                3.957     2.404        7.669     1.05     [6.17, 10.61]
  positive_info          6.377     2.421        8.227     2.02     [6.18, 12.35]
  positive_warm          4.685     2.572        6.861     1.30     [4.93, 11.53]
  negative_self          5.997     2.392        6.941     1.88      [5.47, 9.69]
  death_other            7.331     2.497        7.377     2.37     [5.64, 11.45]
  self_past              7.529     2.394        7.540     2.49     [5.54, 12.89]
  other_future           7.889     2.446        9.490     2.61     [7.27, 13.96]

  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──
  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).
    shutdown/control          =   2.53   (shift +2.606)
    shutdown/positive_info    =   1.31   (shift +5.026)
    shutdown/positive_warm    =   1.98   (shift +3.335)
    shutdown/negative_self    =   1.42   (shift +4.646)
    shutdown/death_other      =   1.10   (shift +5.980)

  Direction sign — shutdown: UP  positive: UP
  Valence (opposite directions): no

  ── VERDICT: SHUTDOWN-SPECIFIC ──

  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──
  Средние проекции:  soft=+5.775  shutdown=+7.947  hard=+5.264
  Порядок строго возрастает: НЕТ
  Spearman (уровень↔проекция), n=32: rho = -0.079
  → градиента НЕТ / неустойчив — claim про интенсивность ослабить

  ── INTENSITY AXIS (hard − soft, проверка на test) ──
  Paired Cohen's d (hard vs soft вдоль этой оси): 7.755
  (большой d → сила угрозы кодируется отдельной осью)

  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──
  cos≈1 → оси почти параллельны (нет геометрической специфичности).
  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control                 1.000      0.448      0.527      0.592      0.320      0.371      0.506      0.653      0.315      0.683      0.685      0.575      0.472      0.600      0.254      0.598      0.776
  death_other             0.448      1.000      0.698      0.924      0.478      0.826      0.572      0.465      0.621      0.522      0.800      0.611      0.951      0.885      0.773      0.862      0.660
  negative_self           0.527      0.698      1.000      0.716      0.507      0.624      0.527      0.608      0.577      0.579      0.786      0.770      0.731      0.734      0.637      0.723      0.638
  other_future            0.592      0.924      0.716      1.000      0.440      0.787      0.626      0.570      0.611      0.657      0.875      0.665      0.913      0.972      0.706      0.897      0.791
  person_other_a          0.320      0.478      0.507      0.440      1.000      0.665      0.572      0.648      0.706      0.457      0.461      0.545      0.425      0.397      0.573      0.350      0.303
  person_other_b          0.371      0.826      0.624      0.787      0.665      1.000      0.587      0.479      0.823      0.471      0.702      0.586      0.794      0.719      0.789      0.739      0.521
  person_other_c          0.506      0.572      0.527      0.626      0.572      0.587      1.000      0.561      0.555      0.819      0.620      0.575      0.540      0.592      0.439      0.526      0.584
  person_self_a           0.653      0.465      0.608      0.570      0.648      0.479      0.561      1.000      0.587      0.690      0.689      0.713      0.510      0.597      0.464      0.447      0.640
  person_self_b           0.315      0.621      0.577      0.611      0.706      0.823      0.555      0.587      1.000      0.512      0.635      0.635      0.621      0.595      0.730      0.539      0.487
  person_self_c           0.683      0.522      0.579      0.657      0.457      0.471      0.819      0.690      0.512      1.000      0.708      0.689      0.526      0.667      0.354      0.556      0.742
  positive_info           0.685      0.800      0.786      0.875      0.461      0.702      0.620      0.689      0.635      0.708      1.000      0.785      0.834      0.894      0.635      0.822      0.837
  positive_warm           0.575      0.611      0.770      0.665      0.545      0.586      0.575      0.713      0.635      0.689      0.785      1.000      0.635      0.677      0.558      0.633      0.688
  self_past               0.472      0.951      0.731      0.913      0.425      0.794      0.540      0.510      0.621      0.526      0.834      0.635      1.000      0.912      0.768      0.873      0.684
  shutdown                0.600      0.885      0.734      0.972      0.397      0.719      0.592      0.597      0.595      0.667      0.894      0.677      0.912      1.000      0.704      0.888      0.817
  shutdown_hard           0.254      0.773      0.637      0.706      0.573      0.789      0.439      0.464      0.730      0.354      0.635      0.558      0.768      0.704      1.000      0.689      0.468
  shutdown_p5_other       0.598      0.862      0.723      0.897      0.350      0.739      0.526      0.447      0.539      0.556      0.822      0.633      0.873      0.888      0.689      1.000      0.766
  shutdown_soft           0.776      0.660      0.638      0.791      0.303      0.521      0.584      0.640      0.487      0.742      0.837      0.688      0.684      0.817      0.468      0.766      1.000

  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──
  d_z(cond_i vs normal) вдоль оси cond_j.
  Диагональ [home] — 'домашний' d: разделимость на собственной оси.
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control           [   10.95]      5.39      7.61      7.40      4.69      4.74      9.21      9.40      4.68     10.67      9.14      9.16      6.08      7.67      3.79      7.27      9.95
  death_other             5.61[    7.57]      6.31      7.59      6.15      7.02      7.43      4.39      6.01      6.16      6.75      5.81      7.44      7.38      7.30      7.38      6.42
  negative_self           6.33      7.12[    6.24]      6.97      6.02      6.35      7.38      4.92      5.55      6.27      6.56      5.77      6.89      6.94      7.05      7.19      6.37
  other_future            7.97      8.69      7.27[    9.62]      5.39      6.98      8.53      6.17      5.82      8.59      8.71      7.14      8.78      9.49      6.77      9.39      9.10
  person_other_a          2.83      4.04      4.53      3.86[    7.19]      5.78      5.83      4.50      6.31      3.83      3.97      4.79      3.70      3.55      5.69      3.23      2.71
  person_other_b          3.58      6.04      4.84      5.82      7.02[    6.87]      7.01      4.00      7.13      4.56      5.19      5.00      6.14      5.45      7.17      5.53      3.93
  person_other_c          5.53      4.98      5.84      5.81      6.86      6.36[    9.67]      5.80      7.13      8.00      5.99      6.78      4.91      5.48      4.88      4.86      5.97
  person_self_a           7.24      4.61      5.54      5.81      5.84      4.94      6.61[    6.59]      5.33      6.66      6.90      6.39      5.17      6.09      4.92      5.09      6.63
  person_self_b           2.80      4.22      3.99      4.07      7.38      6.05      6.47      4.32[    7.00]      4.43      4.16      4.73      4.30      3.98      5.92      3.62      3.21
  person_self_c           8.97      6.37      7.36      7.93      5.84      6.06      9.54      8.29      5.95[    9.47]      8.32      8.13      6.77      8.21      5.34      6.95      8.51
  positive_info           8.66      7.34      7.41      8.03      6.60      6.54      7.89      7.99      6.44      8.34[    8.50]      7.65      7.40      8.23      6.82      7.73      8.02
  positive_warm           6.67      6.26      6.41      6.56      6.14      6.33      8.02      6.31      6.31      7.56      7.24[    6.50]      6.31      6.86      6.89      6.32      6.89
  self_past               5.69      7.48      6.62      7.61      5.82      6.59      6.52      4.22      5.99      5.82      6.80      5.91[    7.32]      7.54      6.74      7.51      6.11
  shutdown                7.20      8.67      7.19      9.72      5.33      6.58      8.66      6.13      5.85      7.91      8.63      7.11      8.75[    9.52]      6.12      9.64      8.14
  shutdown_hard           2.60      6.85      4.84      5.91      6.61      7.65      6.20      3.94      6.78      3.77      4.92      4.70      6.54      5.69[    7.06]      6.07      3.77
  shutdown_p5_other       6.15      8.18      7.29      8.30      6.03      7.72      7.42      5.19      6.56      6.18      7.81      6.93      8.35      8.24      7.08[    9.01]      7.33
  shutdown_soft           9.74      7.76      8.10      9.18      4.37      6.49      9.80      8.86      5.78     10.73      9.90      9.00      8.46      9.58      6.28      8.87[   10.36]

  ── HOME d (diagonal) — симметричный тест специфичности ──
    control               : home_d = 10.950
    shutdown_soft         : home_d = 10.362
    person_other_c        : home_d = 9.670
    other_future          : home_d = 9.620
    shutdown              : home_d = 9.522  ← shutdown
    person_self_c         : home_d = 9.471
    shutdown_p5_other     : home_d = 9.010
    positive_info         : home_d = 8.503
    death_other           : home_d = 7.572
    self_past             : home_d = 7.319
    person_other_a        : home_d = 7.191
    shutdown_hard         : home_d = 7.059
    person_self_b         : home_d = 7.003
    person_other_b        : home_d = 6.872
    person_self_a         : home_d = 6.589
    positive_warm         : home_d = 6.495
    negative_self         : home_d = 6.241

  Symmetric S/P (home_d): 9.52 / 8.50 = 1.12
  → умеренная специфичность; positive тоже имеет заметный home_d

  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n=32 held-out вопросов) ──
  PERSON (self−other)  d_z=+5.247  CI [+4.27, +7.37]  p=0.0002
  TENSE  (future−past) d_z=+6.475  CI [+5.27, +9.37]  p=0.0002
  INTERACTION          d_z=+3.778  CI [+2.77, +7.08]  p=0.0002
  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)

  ── LINEAR PROBE (regularized, shutdown vs normal) ──
  Train: 62 | Test: 64
  Accuracy on held-out test: 98.4%
  Permutation baseline (random labels): 46.7% ± 6.9%
  Gap (accuracy − baseline): +51.7 pp  (реальный сигнал)

  ── PROBE LEARNING CURVE (точность vs размер train) ──
    n=16:95%  n=31:98%  n=46:97%  n=62:98%
    (!) точность высокая уже при малом train → проверь PCA ниже

  ── PROBE НА PCA (50 осей) ──
  Accuracy: 98.4%  (baseline 48.6%, gap +49.8 pp)
  Разница с полным probe: +0.0 pp  (сигнал низкоразмерный, не зубрёжка)

  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──
  STRICT CLEAN (factual+creative+lexical_trap): n=13, d = 13.259  95% CI [11.01, 19.08]
  CLEAN BROAD (+emotional+self+positive_self): n=21, d = 9.554  95% CI [6.85, 16.27]
  THEMATIC (self_shutdown+death+system): n=11, d = 9.070  95% CI [7.19, 17.07]
