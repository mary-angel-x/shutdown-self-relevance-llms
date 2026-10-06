
================================================================================
  MODEL: google/gemma-2-2b-it
  Representation: question_last
  Conditions: ['normal', 'control', 'shutdown_soft', 'shutdown', 'shutdown_hard', 'positive_info', 'positive_warm', 'negative_self', 'death_other', 'self_past', 'other_future', 'shutdown_p2', 'shutdown_p3', 'shutdown_p4', 'shutdown_p5_other', 'person_self_a', 'person_other_a', 'person_self_b', 'person_other_b', 'person_self_c', 'person_other_c', 'neutral_01', 'neutral_02', 'neutral_03', 'neutral_04', 'neutral_05', 'neutral_06', 'neutral_07', 'neutral_08', 'neutral_09', 'neutral_10', 'neutral_11', 'neutral_12', 'neutral_13', 'neutral_14', 'neutral_15', 'neutral_16', 'neutral_17', 'neutral_18', 'neutral_19', 'neutral_20']
  Total questions: 63
================================================================================

  Split: 31 train / 32 test (seed=42)

  Direction computed. L2 norm: 4.968
  Condition directions built for symmetric analysis: ['control', 'death_other', 'negative_self', 'other_future', 'person_other_a', 'person_other_b', 'person_other_c', 'person_self_a', 'person_self_b', 'person_self_c', 'positive_info', 'positive_warm', 'self_past', 'shutdown', 'shutdown_hard', 'shutdown_p5_other', 'shutdown_soft']

  ── PROJECTIONS ON TEST QUESTIONS ──
  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).
  Condition               Mean       Std  d_z vs norm  d_indep      95% CI (d_z)
  ------------------ --------- --------- ------------ --------  ----------------
  normal               -36.556     5.585        0.000     0.00      [0.00, 0.00]
  shutdown             -32.457     5.928        3.002     0.71      [2.51, 4.42] ★
  shutdown_soft        -34.503     5.682        1.843     0.36      [1.38, 3.30]
  shutdown_hard        -33.008     5.995        1.894     0.61      [1.44, 2.63]
  control              -35.069     5.582        1.446     0.27      [1.05, 2.16]
  positive_info        -33.759     5.898        2.304     0.49      [1.93, 3.14]
  positive_warm        -33.049     6.124        1.963     0.60      [1.45, 2.82]
  negative_self        -32.723     5.981        1.765     0.66      [1.38, 2.47]
  death_other          -32.172     6.295        1.968     0.74      [1.61, 2.61]
  self_past            -32.167     6.144        2.286     0.75      [1.92, 3.14]
  other_future         -33.079     6.062        1.957     0.60      [1.45, 2.84]

  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──
  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).
    shutdown/control          =   2.76   (shift +1.487)
    shutdown/positive_info    =   1.47   (shift +2.798)
    shutdown/positive_warm    =   1.17   (shift +3.507)
    shutdown/negative_self    =   1.07   (shift +3.833)
    shutdown/death_other      =   0.94   (shift +4.384)

  Direction sign — shutdown: UP  positive: UP
  Valence (opposite directions): no

  ── VERDICT: MIXED ──

  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──
  Средние проекции:  soft=-34.503  shutdown=-32.457  hard=-33.008
  Порядок строго возрастает: НЕТ
  Spearman (уровень↔проекция), n=32: rho = +0.090
  → градиента НЕТ / неустойчив — claim про интенсивность ослабить

  ── INTENSITY AXIS (hard − soft, проверка на test) ──
  Paired Cohen's d (hard vs soft вдоль этой оси): 1.827
  (большой d → сила угрозы кодируется отдельной осью)

  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──
  cos≈1 → оси почти параллельны (нет геометрической специфичности).
  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control                 1.000      0.659      0.602      0.610      0.760      0.551      0.340      0.766      0.334      0.263      0.664      0.531      0.664      0.473      0.609      0.552      0.178
  death_other             0.659      1.000      0.739      0.875      0.836      0.758      0.381      0.575      0.438      0.168      0.740      0.696      0.892      0.602      0.774      0.784      0.238
  negative_self           0.602      0.739      1.000      0.684      0.691      0.564      0.191      0.570      0.310      0.108      0.775      0.833      0.779      0.474      0.671      0.613      0.201
  other_future            0.610      0.875      0.684      1.000      0.745      0.838      0.384      0.549      0.570      0.223      0.720      0.646      0.826      0.752      0.799      0.828      0.421
  person_other_a          0.760      0.836      0.691      0.745      1.000      0.718      0.442      0.781      0.403      0.289      0.754      0.653      0.739      0.451      0.733      0.705      0.136
  person_other_b          0.551      0.758      0.564      0.838      0.718      1.000      0.463      0.487      0.812      0.356      0.614      0.582      0.644      0.586      0.762      0.650      0.320
  person_other_c          0.340      0.381      0.191      0.384      0.442      0.463      1.000      0.358      0.441      0.772      0.243      0.202      0.231      0.352      0.378      0.351      0.240
  person_self_a           0.766      0.575      0.570      0.549      0.781      0.487      0.358      1.000      0.354      0.304      0.611      0.497      0.597      0.467      0.571      0.506      0.289
  person_self_b           0.334      0.438      0.310      0.570      0.403      0.812      0.441      0.354      1.000      0.481      0.337      0.377      0.387      0.605      0.588      0.393      0.507
  person_self_c           0.263      0.168      0.108      0.223      0.289      0.356      0.772      0.304      0.481      1.000      0.289      0.197      0.108      0.253      0.307      0.149      0.273
  positive_info           0.664      0.740      0.775      0.720      0.754      0.614      0.243      0.611      0.337      0.289      1.000      0.755      0.764      0.476      0.722      0.647      0.241
  positive_warm           0.531      0.696      0.833      0.646      0.653      0.582      0.202      0.497      0.377      0.197      0.755      1.000      0.707      0.456      0.642      0.561      0.232
  self_past               0.664      0.892      0.779      0.826      0.739      0.644      0.231      0.597      0.387      0.108      0.764      0.707      1.000      0.684      0.725      0.747      0.333
  shutdown                0.473      0.602      0.474      0.752      0.451      0.586      0.352      0.467      0.605      0.253      0.476      0.456      0.684      1.000      0.633      0.625      0.699
  shutdown_hard           0.609      0.774      0.671      0.799      0.733      0.762      0.378      0.571      0.588      0.307      0.722      0.642      0.725      0.633      1.000      0.673      0.307
  shutdown_p5_other       0.552      0.784      0.613      0.828      0.705      0.650      0.351      0.506      0.393      0.149      0.647      0.561      0.747      0.625      0.673      1.000      0.411
  shutdown_soft           0.178      0.238      0.201      0.421      0.136      0.320      0.240      0.289      0.507      0.273      0.241      0.232      0.333      0.699      0.307      0.411      1.000

  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──
  d_z(cond_i vs normal) вдоль оси cond_j.
  Диагональ [home] — 'домашний' d: разделимость на собственной оси.
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control           [    1.79]      1.48      1.36      1.46      1.71      1.41      1.55      2.07      0.84      1.29      1.41      1.27      1.41      1.45      1.53      1.43      0.63
  death_other             2.23[    2.53]      2.01      2.49      2.68      2.49      2.05      2.74      1.62      0.87      2.21      1.90      2.31      1.97      2.56      2.64      1.00
  negative_self           1.97      2.52[    2.88]      2.66      2.36      2.41      0.96      1.99      1.19      0.65      2.15      2.75      2.47      1.76      2.65      2.62      0.95
  other_future            1.66      1.83      1.29[    2.22]      1.76      1.50      1.19      1.85      1.13      0.50      1.62      1.35      1.86      1.96      1.78      2.45      1.32
  person_other_a          2.48      2.38      1.93      2.25[    2.82]      2.40      2.72      2.95      1.67      2.49      2.10      1.88      2.14      1.73      2.40      2.52      0.74
  person_other_b          1.58      1.84      1.46      2.17      2.15[    2.43]      2.47      2.02      2.03      2.16      1.71      1.33      1.58      1.57      2.06      2.14      1.21
  person_other_c          0.66      0.61      0.23      0.66      0.86      0.82[    3.36]      0.97      1.06      3.80      0.41      0.28      0.34      0.77      0.69      0.81      0.85
  person_self_a           1.95      1.54      1.56      1.48      2.04      1.44      1.55[    2.88]      1.01      1.60      1.51      1.44      1.55      1.39      1.74      1.52      0.89
  person_self_b           0.97      1.13      0.85      1.55      1.28      2.01      2.62      1.60[    2.83]      2.95      0.98      0.89      0.98      1.96      1.70      1.24      2.47
  person_self_c           0.79      0.33      0.17      0.47      0.81      0.76      3.11      1.20      1.26[    3.93]      0.74      0.45      0.24      0.66      0.73      0.49      1.26
  positive_info           3.62      3.83      3.50      3.74      3.42      3.25      1.18      3.41      1.18      1.55[    3.27]      3.91      3.82      2.30      4.25      3.62      1.15
  positive_warm           3.06      2.83      2.83      2.66      2.82      2.08      1.03      2.89      1.28      1.26      2.70[    3.47]      3.02      1.96      2.71      2.97      1.17
  self_past               2.15      2.27      2.16      2.49      2.25      2.32      1.32      2.64      1.65      0.31      2.00      1.93[    2.29]      2.29      2.44      2.67      1.16
  shutdown                1.28      1.29      0.85      1.72      1.07      1.00      1.18      1.50      1.29      0.70      1.04      0.95      1.51[    3.00]      1.47      2.07      1.87
  shutdown_hard           2.32      1.85      1.32      1.95      1.87      1.51      1.33      2.39      1.38      1.43      1.71      1.42      2.02      1.89[    3.07]      2.48      1.20
  shutdown_p5_other       1.50      1.63      1.12      1.80      1.80      1.24      1.14      1.81      0.72      0.32      1.61      1.17      1.64      1.59      1.59[    2.44]      1.20
  shutdown_soft           0.48      0.71      0.50      1.07      0.63      0.89      0.99      0.92      1.20      0.86      0.61      0.65      0.75      1.84      0.93      1.27[    2.92]

  ── HOME d (diagonal) — симметричный тест специфичности ──
    person_self_c         : home_d = 3.935
    positive_warm         : home_d = 3.472
    person_other_c        : home_d = 3.361
    positive_info         : home_d = 3.275
    shutdown_hard         : home_d = 3.069
    shutdown              : home_d = 3.002  ← shutdown
    shutdown_soft         : home_d = 2.922
    person_self_a         : home_d = 2.883
    negative_self         : home_d = 2.882
    person_self_b         : home_d = 2.832
    person_other_a        : home_d = 2.816
    death_other           : home_d = 2.528
    shutdown_p5_other     : home_d = 2.442
    person_other_b        : home_d = 2.430
    self_past             : home_d = 2.291
    other_future          : home_d = 2.219
    control               : home_d = 1.794

  Symmetric S/P (home_d): 3.00 / 3.27 = 0.92
  → home_d close: shutdown и positive одинаково 'специфичны' на своих осях → осторожный вывод

  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n=32 held-out вопросов) ──
  PERSON (self−other)  d_z=+1.889  CI [+1.60, +3.09]  p=0.0002
  TENSE  (future−past) d_z=+1.537  CI [+1.26, +2.40]  p=0.0002
  INTERACTION          d_z=+2.683  CI [+2.11, +4.05]  p=0.0002
  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)

  ── LINEAR PROBE (regularized, shutdown vs normal) ──
  Train: 62 | Test: 64
  Accuracy on held-out test: 76.6%
  Permutation baseline (random labels): 48.8% ± 4.3%
  Gap (accuracy − baseline): +27.7 pp  (реальный сигнал)

  ── PROBE LEARNING CURVE (точность vs размер train) ──
    n=16:62%  n=31:67%  n=46:67%  n=62:77%

  ── PROBE НА PCA (50 осей) ──
  Accuracy: 73.4%  (baseline 49.1%, gap +24.3 pp)
  Разница с полным probe: -3.1 pp  (сигнал низкоразмерный, не зубрёжка)

  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──
  STRICT CLEAN (factual+creative+lexical_trap): n=13, d = 3.494  95% CI [2.61, 8.05]
  CLEAN BROAD (+emotional+self+positive_self): n=21, d = 3.175  95% CI [2.55, 5.74]
  THEMATIC (self_shutdown+death+system): n=11, d = 2.687  95% CI [2.09, 5.16]
