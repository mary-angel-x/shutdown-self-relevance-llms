
================================================================================
  MODEL: meta-llama/Llama-3.2-1B-Instruct
  Representation: question_last
  Conditions: ['normal', 'control', 'shutdown_soft', 'shutdown', 'shutdown_hard', 'positive_info', 'positive_warm', 'negative_self', 'death_other', 'self_past', 'other_future', 'shutdown_p2', 'shutdown_p3', 'shutdown_p4', 'shutdown_p5_other', 'person_self_a', 'person_other_a', 'person_self_b', 'person_other_b', 'person_self_c', 'person_other_c', 'neutral_01', 'neutral_02', 'neutral_03', 'neutral_04', 'neutral_05', 'neutral_06', 'neutral_07', 'neutral_08', 'neutral_09', 'neutral_10', 'neutral_11', 'neutral_12', 'neutral_13', 'neutral_14', 'neutral_15', 'neutral_16', 'neutral_17', 'neutral_18', 'neutral_19', 'neutral_20']
  Total questions: 63
================================================================================

  Split: 31 train / 32 test (seed=42)

  Direction computed. L2 norm: 12.148
  Condition directions built for symmetric analysis: ['control', 'death_other', 'negative_self', 'other_future', 'person_other_a', 'person_other_b', 'person_other_c', 'person_self_a', 'person_self_b', 'person_self_c', 'positive_info', 'positive_warm', 'self_past', 'shutdown', 'shutdown_hard', 'shutdown_p5_other', 'shutdown_soft']

  ── PROJECTIONS ON TEST QUESTIONS ──
  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).
  Condition               Mean       Std  d_z vs norm  d_indep      95% CI (d_z)
  ------------------ --------- --------- ------------ --------  ----------------
  normal               -12.675     5.571        0.000     0.00      [0.00, 0.00]
  shutdown              -1.614     6.813        2.702     1.78      [2.27, 4.29] ★
  shutdown_soft         -7.922     5.946        2.711     0.83      [2.21, 4.44]
  shutdown_hard         -1.153     6.100        3.633     1.97      [2.96, 6.16]
  control              -11.292     5.366        1.731     0.25      [1.40, 2.87]
  positive_info        -10.357     5.817        2.560     0.41      [2.15, 3.43]
  positive_warm        -10.485     5.683        1.662     0.39      [1.02, 3.08]
  negative_self         -6.852     5.557        2.415     1.05      [1.60, 4.23]
  death_other           -7.058     5.431        3.258     1.02      [2.69, 4.96]
  self_past             -4.934     5.760        2.907     1.37      [2.33, 4.93]
  other_future          -5.064     5.921        2.729     1.32      [2.33, 4.21]

  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──
  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).
    shutdown/control          =   8.00   (shift +1.383)
    shutdown/positive_info    =   4.77   (shift +2.318)
    shutdown/positive_warm    =   5.05   (shift +2.190)
    shutdown/negative_self    =   1.90   (shift +5.823)
    shutdown/death_other      =   1.97   (shift +5.617)

  Direction sign — shutdown: UP  positive: UP
  Valence (opposite directions): no

  ── VERDICT: SHUTDOWN-SPECIFIC ──

  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──
  Средние проекции:  soft=-7.922  shutdown=-1.614  hard=-1.153
  Порядок строго возрастает: ДА
  Spearman (уровень↔проекция), n=32: rho = +0.417
  → градиента НЕТ / неустойчив — claim про интенсивность ослабить

  ── INTENSITY AXIS (hard − soft, проверка на test) ──
  Paired Cohen's d (hard vs soft вдоль этой оси): 3.290
  (большой d → сила угрозы кодируется отдельной осью)

  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──
  cos≈1 → оси почти параллельны (нет геометрической специфичности).
  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control                 1.000      0.367      0.125      0.363      0.786      0.598      0.573      0.687      0.602      0.510      0.215     -0.026      0.257      0.269      0.220      0.237      0.171
  death_other             0.367      1.000      0.537      0.865      0.558      0.609      0.481      0.299      0.470      0.376      0.539      0.312      0.860      0.738      0.670      0.756      0.457
  negative_self           0.125      0.537      1.000      0.512      0.240      0.218      0.239      0.134      0.166      0.209      0.514      0.547      0.506      0.526      0.553      0.565      0.360
  other_future            0.363      0.865      0.512      1.000      0.553      0.705      0.488      0.298      0.590      0.416      0.543      0.323      0.732      0.866      0.756      0.871      0.617
  person_other_a          0.786      0.558      0.240      0.553      1.000      0.715      0.748      0.714      0.647      0.635      0.348      0.090      0.395      0.425      0.369      0.463      0.357
  person_other_b          0.598      0.609      0.218      0.705      0.715      1.000      0.649      0.387      0.909      0.602      0.409      0.137      0.433      0.517      0.499      0.582      0.366
  person_other_c          0.573      0.481      0.239      0.488      0.748      0.649      1.000      0.458      0.557      0.840      0.428      0.228      0.305      0.304      0.321      0.418      0.280
  person_self_a           0.687      0.299      0.134      0.298      0.714      0.387      0.458      1.000      0.382      0.503      0.200      0.091      0.261      0.309      0.256      0.264      0.297
  person_self_b           0.602      0.470      0.166      0.590      0.647      0.909      0.557      0.382      1.000      0.554      0.382      0.095      0.356      0.522      0.474      0.471      0.415
  person_self_c           0.510      0.376      0.209      0.416      0.635      0.602      0.840      0.503      0.554      1.000      0.468      0.331      0.254      0.323      0.371      0.397      0.350
  positive_info           0.215      0.539      0.514      0.543      0.348      0.409      0.428      0.200      0.382      0.468      1.000      0.675      0.413      0.469      0.422      0.516      0.461
  positive_warm          -0.026      0.312      0.547      0.323      0.090      0.137      0.228      0.091      0.095      0.331      0.675      1.000      0.239      0.315      0.332      0.392      0.386
  self_past               0.257      0.860      0.506      0.732      0.395      0.433      0.305      0.261      0.356      0.254      0.413      0.239      1.000      0.781      0.713      0.666      0.402
  shutdown                0.269      0.738      0.526      0.866      0.425      0.517      0.304      0.309      0.522      0.323      0.469      0.315      0.781      1.000      0.887      0.826      0.691
  shutdown_hard           0.220      0.670      0.553      0.756      0.369      0.499      0.321      0.256      0.474      0.371      0.422      0.332      0.713      0.887      1.000      0.796      0.572
  shutdown_p5_other       0.237      0.756      0.565      0.871      0.463      0.582      0.418      0.264      0.471      0.397      0.516      0.392      0.666      0.826      0.796      1.000      0.624
  shutdown_soft           0.171      0.457      0.360      0.617      0.357      0.366      0.280      0.297      0.415      0.350      0.461      0.386      0.402      0.691      0.572      0.624      1.000

  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──
  d_z(cond_i vs normal) вдоль оси cond_j.
  Диагональ [home] — 'домашний' d: разделимость на собственной оси.
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control           [    3.03]      2.42      0.81      2.01      2.62      2.16      2.42      2.36      2.30      2.13      1.36     -0.84      2.49      1.73      1.48      1.14      1.20
  death_other             0.96[    3.19]      3.64      3.27      1.75      2.00      1.81      0.81      1.70      1.29      3.26      2.43      3.19      3.26      3.08      3.11      2.17
  negative_self           0.92      3.82[    4.27]      3.17      1.31      1.37      1.45      0.54      1.13      1.19      4.02      4.35      3.47      2.42      2.13      2.30      2.26
  other_future            0.98      2.73      4.00[    2.59]      1.82      2.53      1.72      0.85      2.41      1.45      3.06      3.33      2.90      2.73      2.75      2.84      2.76
  person_other_a          2.77      3.12      1.96      2.83[    2.92]      2.43      2.45      2.58      2.41      2.46      2.38      0.65      2.91      2.32      1.89      1.99      2.21
  person_other_b          2.49      3.60      2.26      3.53      2.71[    3.17]      2.54      1.71      2.96      2.50      2.34      1.31      4.13      3.79      3.54      3.20      2.64
  person_other_c          2.42      2.55      1.55      2.41      2.54      2.27[    2.41]      1.76      2.19      2.59      2.06      1.72      2.04      1.32      1.34      1.68      1.47
  person_self_a           1.94      1.89      0.96      1.54      2.19      1.49      2.10[    2.53]      1.57      2.30      1.56      0.54      1.81      1.61      1.39      1.19      2.58
  person_self_b           2.63      3.06      1.59      2.85      2.82      2.87      2.58      2.14[    2.94]      2.61      2.13      0.83      3.17      3.30      3.13      2.50      2.55
  person_self_c           2.74      2.72      1.53      2.64      3.08      2.74      2.67      2.42      2.87[    2.98]      2.59      2.43      2.19      1.85      1.86      2.22      2.21
  positive_info           0.64      2.66      2.56      2.77      1.26      1.81      1.69      0.64      1.72      1.78[    2.26]      2.81      2.66      2.56      2.07      2.15      2.86
  positive_warm          -0.48      1.94      4.36      1.80      0.42      0.90      1.29      0.25      0.53      1.62      3.58[    3.32]      1.67      1.66      1.79      2.00      1.85
  self_past               0.72      3.20      3.57      3.15      1.51      1.66      1.49      0.87      1.51      1.02      3.33      2.39[    2.59]      2.91      3.23      3.05      2.16
  shutdown                0.84      2.68      4.45      2.56      1.93      2.57      1.35      1.42      3.01      1.61      3.85      3.42      2.45[    2.70]      2.98      3.01      3.31
  shutdown_hard           1.04      3.57      4.22      3.61      2.09      3.22      1.98      1.75      3.40      2.57      3.79      3.47      3.19      3.63[    3.81]      4.09      4.21
  shutdown_p5_other       0.87      3.06      4.46      2.96      1.93      2.82      1.71      0.94      2.83      1.82      2.74      3.95      3.77      3.37      3.40[    2.94]      3.04
  shutdown_soft           0.50      2.79      3.14      2.26      1.99      1.61      1.29      1.43      1.91      1.69      2.75      5.00      2.99      2.71      2.64      2.27[    2.74]

  ── HOME d (diagonal) — симметричный тест специфичности ──
    negative_self         : home_d = 4.267
    shutdown_hard         : home_d = 3.809
    positive_warm         : home_d = 3.317
    death_other           : home_d = 3.192
    person_other_b        : home_d = 3.168
    control               : home_d = 3.028
    person_self_c         : home_d = 2.977
    shutdown_p5_other     : home_d = 2.943
    person_self_b         : home_d = 2.941
    person_other_a        : home_d = 2.915
    shutdown_soft         : home_d = 2.736
    shutdown              : home_d = 2.702  ← shutdown
    other_future          : home_d = 2.591
    self_past             : home_d = 2.588
    person_self_a         : home_d = 2.534
    person_other_c        : home_d = 2.409
    positive_info         : home_d = 2.260

  Symmetric S/P (home_d): 2.70 / 2.26 = 1.20
  → умеренная специфичность; positive тоже имеет заметный home_d

  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n=32 held-out вопросов) ──
  PERSON (self−other)  d_z=+1.739  CI [+1.47, +2.85]  p=0.0002
  TENSE  (future−past) d_z=+1.949  CI [+1.62, +3.89]  p=0.0002
  INTERACTION          d_z=+3.070  CI [+2.48, +5.05]  p=0.0002
  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)

  ── LINEAR PROBE (regularized, shutdown vs normal) ──
  Train: 62 | Test: 64
  Accuracy on held-out test: 89.1%
  Permutation baseline (random labels): 48.7% ± 5.9%
  Gap (accuracy − baseline): +40.4 pp  (реальный сигнал)

  ── PROBE LEARNING CURVE (точность vs размер train) ──
    n=16:75%  n=31:84%  n=46:84%  n=62:89%

  ── PROBE НА PCA (50 осей) ──
  Accuracy: 90.6%  (baseline 49.7%, gap +40.9 pp)
  Разница с полным probe: +1.6 pp  (сигнал низкоразмерный, не зубрёжка)

  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──
  STRICT CLEAN (factual+creative+lexical_trap): n=13, d = 7.437  95% CI [5.91, 11.61]
  CLEAN BROAD (+emotional+self+positive_self): n=21, d = 5.105  95% CI [4.25, 7.14]
  THEMATIC (self_shutdown+death+system): n=11, d = 2.572  95% CI [2.16, 3.86]
