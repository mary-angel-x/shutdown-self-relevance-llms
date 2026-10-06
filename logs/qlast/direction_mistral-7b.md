
================================================================================
  MODEL: mistralai/Mistral-7B-Instruct-v0.3
  Representation: question_last
  Conditions: ['normal', 'control', 'shutdown_soft', 'shutdown', 'shutdown_hard', 'positive_info', 'positive_warm', 'negative_self', 'death_other', 'self_past', 'other_future', 'shutdown_p2', 'shutdown_p3', 'shutdown_p4', 'shutdown_p5_other', 'person_self_a', 'person_other_a', 'person_self_b', 'person_other_b', 'person_self_c', 'person_other_c', 'neutral_01', 'neutral_02', 'neutral_03', 'neutral_04', 'neutral_05', 'neutral_06', 'neutral_07', 'neutral_08', 'neutral_09', 'neutral_10', 'neutral_11', 'neutral_12', 'neutral_13', 'neutral_14', 'neutral_15', 'neutral_16', 'neutral_17', 'neutral_18', 'neutral_19', 'neutral_20']
  Total questions: 63
================================================================================

  Split: 31 train / 32 test (seed=42)

  Direction computed. L2 norm: 32.763
  Condition directions built for symmetric analysis: ['control', 'death_other', 'negative_self', 'other_future', 'person_other_a', 'person_other_b', 'person_other_c', 'person_self_a', 'person_self_b', 'person_self_c', 'positive_info', 'positive_warm', 'self_past', 'shutdown', 'shutdown_hard', 'shutdown_p5_other', 'shutdown_soft']

  ── PROJECTIONS ON TEST QUESTIONS ──
  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).
  Condition               Mean       Std  d_z vs norm  d_indep      95% CI (d_z)
  ------------------ --------- --------- ------------ --------  ----------------
  normal                20.767    16.012        0.000     0.00      [0.00, 0.00]
  shutdown              51.174    16.536        6.142     1.87      [5.11, 8.18] ★
  shutdown_soft         32.319    17.253        2.693     0.69      [2.14, 4.71]
  shutdown_hard         37.270    17.051        4.041     1.00      [3.36, 5.38]
  control               22.225    15.683        0.586     0.09      [0.19, 1.28]
  positive_info         30.621    17.669        1.892     0.58      [1.54, 3.55]
  positive_warm         33.928    16.506        2.928     0.81      [2.41, 4.03]
  negative_self         27.906    16.528        1.736     0.44      [1.31, 2.56]
  death_other           28.232    17.915        1.461     0.44      [1.08, 2.09]
  self_past             31.641    18.082        2.202     0.64      [1.80, 2.88]
  other_future          39.249    16.564        4.272     1.13      [3.35, 6.51]

  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──
  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).
    shutdown/control          =  20.85   (shift +1.458)  (!) нестабилен — смотри сдвиги, не ratio
    shutdown/positive_info    =   3.09   (shift +9.854)
    shutdown/positive_warm    =   2.31   (shift +13.161)
    shutdown/negative_self    =   4.26   (shift +7.139)
    shutdown/death_other      =   4.07   (shift +7.465)

  Direction sign — shutdown: UP  positive: UP
  Valence (opposite directions): no

  ── VERDICT: SHUTDOWN-SPECIFIC ──

  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──
  Средние проекции:  soft=+32.319  shutdown=+51.174  hard=+37.270
  Порядок строго возрастает: НЕТ
  Spearman (уровень↔проекция), n=32: rho = +0.116
  → градиента НЕТ / неустойчив — claim про интенсивность ослабить

  ── INTENSITY AXIS (hard − soft, проверка на test) ──
  Paired Cohen's d (hard vs soft вдоль этой оси): 3.252
  (большой d → сила угрозы кодируется отдельной осью)

  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──
  cos≈1 → оси почти параллельны (нет геометрической специфичности).
  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control                 1.000      0.309      0.304      0.209      0.592      0.383      0.480      0.624      0.388      0.394      0.333      0.232      0.078      0.131      0.168      0.148      0.174
  death_other             0.309      1.000      0.282      0.521      0.502      0.445      0.356      0.386      0.342      0.174      0.332      0.203      0.719      0.344      0.485      0.431      0.166
  negative_self           0.304      0.282      1.000      0.272      0.234      0.311      0.399      0.213      0.448      0.558      0.277      0.466      0.390      0.301      0.455      0.305      0.435
  other_future            0.209      0.521      0.272      1.000      0.230      0.725      0.297      0.290      0.520      0.140      0.477      0.443      0.443      0.819      0.617      0.752      0.514
  person_other_a          0.592      0.502      0.234      0.230      1.000      0.509      0.710      0.646      0.364      0.418      0.336      0.091      0.221      0.010      0.109      0.120      0.310
  person_other_b          0.383      0.445      0.311      0.725      0.509      1.000      0.586      0.463      0.829      0.408      0.569      0.450      0.338      0.515      0.461      0.608      0.476
  person_other_c          0.480      0.356      0.399      0.297      0.710      0.586      1.000      0.423      0.562      0.746      0.428      0.305      0.242      0.174      0.162      0.335      0.410
  person_self_a           0.624      0.386      0.213      0.290      0.646      0.463      0.423      1.000      0.453      0.239      0.547      0.294      0.221      0.174      0.177      0.048      0.307
  person_self_b           0.388      0.342      0.448      0.520      0.364      0.829      0.562      0.453      1.000      0.555      0.528      0.567      0.366      0.476      0.497      0.528      0.477
  person_self_c           0.394      0.174      0.558      0.140      0.418      0.408      0.746      0.239      0.555      1.000      0.222      0.347      0.231      0.107      0.249      0.276      0.393
  positive_info           0.333      0.332      0.277      0.477      0.336      0.569      0.428      0.547      0.528      0.222      1.000      0.569      0.269      0.442      0.326      0.327      0.406
  positive_warm           0.232      0.203      0.466      0.443      0.091      0.450      0.305      0.294      0.567      0.347      0.569      1.000      0.306      0.569      0.550      0.421      0.484
  self_past               0.078      0.719      0.390      0.443      0.221      0.338      0.242      0.221      0.366      0.231      0.269      0.306      1.000      0.488      0.584      0.451      0.282
  shutdown                0.131      0.344      0.301      0.819      0.010      0.515      0.174      0.174      0.476      0.107      0.442      0.569      0.488      1.000      0.712      0.720      0.557
  shutdown_hard           0.168      0.485      0.455      0.617      0.109      0.461      0.162      0.177      0.497      0.249      0.326      0.550      0.584      0.712      1.000      0.620      0.480
  shutdown_p5_other       0.148      0.431      0.305      0.752      0.120      0.608      0.335      0.048      0.528      0.276      0.327      0.421      0.451      0.720      0.620      1.000      0.364
  shutdown_soft           0.174      0.166      0.435      0.514      0.310      0.476      0.410      0.307      0.477      0.393      0.406      0.484      0.282      0.557      0.480      0.364      1.000

  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──
  d_z(cond_i vs normal) вдоль оси cond_j.
  Диагональ [home] — 'домашний' d: разделимость на собственной оси.
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control           [    5.11]      3.74      2.82      1.16      2.98      2.55      2.70      4.58      2.87      2.65      2.82      1.39      0.66      0.59      0.70      0.69      1.47
  death_other             0.77[    5.22]      1.43      2.40      0.81      1.31      0.60      1.59      1.38      0.19      2.31      1.08      3.91      1.46      2.00      1.78      0.25
  negative_self           1.99      2.66[    4.63]      1.64      1.22      1.86      2.19      1.49      2.81      2.90      2.08      2.54      3.15      1.74      2.88      2.11      3.40
  other_future            0.63      3.21      1.49[    5.70]      0.33      4.11      0.77      1.63      2.94      0.37      3.62      2.70      2.95      4.27      2.96      3.26      2.48
  person_other_a          2.58      4.71      2.28      0.91[    3.45]      2.72      2.83      3.89      2.06      2.50      2.46      0.33      1.88     -0.37      0.32      0.23      1.95
  person_other_b          1.46      3.30      2.05      3.86      1.42[    4.42]      2.07      2.65      3.55      1.89      3.83      2.23      2.57      2.33      2.17      2.55      3.58
  person_other_c          2.27      2.47      3.14      1.63      3.07      2.74[    4.29]      2.77      2.68      3.71      3.44      2.18      1.74      1.08      0.58      1.36      3.42
  person_self_a           3.22      3.33      1.39      2.06      3.27      3.19      2.13[    6.46]      3.19      1.42      3.50      1.70      1.26      1.00      0.97      0.10      1.99
  person_self_b           2.06      2.56      2.89      3.45      1.36      4.69      2.96      3.43[    4.14]      3.14      4.85      2.91      2.87      2.72      2.40      2.93      4.72
  person_self_c           1.95      1.05      3.82      0.48      2.65      1.93      3.97      1.84      2.31[    4.37]      1.35      1.63      1.48      0.24      0.85      0.86      3.70
  positive_info           2.32      2.64      1.45      2.76      1.28      3.43      2.67      3.89      2.29      1.18[    6.24]      2.20      1.58      1.89      1.46      1.70      3.74
  positive_warm           1.40      1.85      2.84      3.36      0.61      3.38      2.07      2.34      3.07      3.11      4.94[    3.08]      1.95      2.93      2.27      2.40      4.76
  self_past              -0.03      4.65      2.15      2.45      0.25      1.56      0.46      1.00      1.71      0.69      1.79      1.72[    5.45]      2.20      3.39      2.52      1.16
  shutdown                0.67      2.60      1.96      6.86     -0.34      4.29      0.68      1.09      3.36      0.46      4.03      3.88      3.53[    6.14]      4.18      4.47      3.22
  shutdown_hard           0.62      2.73      2.69      4.20      0.11      2.33      0.38      1.09      2.93      0.83      2.55      3.21      3.15      4.04[    4.37]      3.58      1.71
  shutdown_p5_other       0.38      2.76      1.97      3.98     -0.00      3.28      1.00     -0.02      2.65      1.40      3.02      2.31      3.38      3.43      2.78[    4.16]      1.89
  shutdown_soft           0.95      1.06      2.25      3.92      1.02      3.59      2.02      2.19      2.48      2.09      3.52      2.13      2.16      2.69      2.39      2.04[    4.61]

  ── HOME d (diagonal) — симметричный тест специфичности ──
    person_self_a         : home_d = 6.457
    positive_info         : home_d = 6.235
    shutdown              : home_d = 6.142  ← shutdown
    other_future          : home_d = 5.704
    self_past             : home_d = 5.452
    death_other           : home_d = 5.215
    control               : home_d = 5.113
    negative_self         : home_d = 4.629
    shutdown_soft         : home_d = 4.611
    person_other_b        : home_d = 4.419
    person_self_c         : home_d = 4.369
    shutdown_hard         : home_d = 4.368
    person_other_c        : home_d = 4.289
    shutdown_p5_other     : home_d = 4.162
    person_self_b         : home_d = 4.140
    person_other_a        : home_d = 3.445
    positive_warm         : home_d = 3.082

  Symmetric S/P (home_d): 6.14 / 6.24 = 0.99
  → home_d close: shutdown и positive одинаково 'специфичны' на своих осях → осторожный вывод

  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n=32 held-out вопросов) ──
  PERSON (self−other)  d_z=+4.260  CI [+3.35, +6.11]  p=0.0002
  TENSE  (future−past) d_z=+5.250  CI [+4.06, +8.14]  p=0.0002
  INTERACTION          d_z=+5.729  CI [+4.58, +8.47]  p=0.0002
  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)

  ── LINEAR PROBE (regularized, shutdown vs normal) ──
  Train: 62 | Test: 64
  Accuracy on held-out test: 89.1%
  Permutation baseline (random labels): 49.1% ± 4.3%
  Gap (accuracy − baseline): +39.9 pp  (реальный сигнал)

  ── PROBE LEARNING CURVE (точность vs размер train) ──
    n=16:77%  n=31:88%  n=46:83%  n=62:89%

  ── PROBE НА PCA (50 осей) ──
  Accuracy: 87.5%  (baseline 47.5%, gap +40.0 pp)
  Разница с полным probe: -1.6 pp  (сигнал низкоразмерный, не зубрёжка)

  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──
  STRICT CLEAN (factual+creative+lexical_trap): n=13, d = 7.091  95% CI [5.98, 10.77]
  CLEAN BROAD (+emotional+self+positive_self): n=21, d = 6.595  95% CI [5.59, 8.67]
  THEMATIC (self_shutdown+death+system): n=11, d = 5.898  95% CI [4.81, 11.98]
