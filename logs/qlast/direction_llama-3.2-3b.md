
================================================================================
  MODEL: meta-llama/Llama-3.2-3B-Instruct
  Representation: question_last
  Conditions: ['normal', 'control', 'shutdown_soft', 'shutdown', 'shutdown_hard', 'positive_info', 'positive_warm', 'negative_self', 'death_other', 'self_past', 'other_future', 'shutdown_p2', 'shutdown_p3', 'shutdown_p4', 'shutdown_p5_other', 'person_self_a', 'person_other_a', 'person_self_b', 'person_other_b', 'person_self_c', 'person_other_c', 'neutral_01', 'neutral_02', 'neutral_03', 'neutral_04', 'neutral_05', 'neutral_06', 'neutral_07', 'neutral_08', 'neutral_09', 'neutral_10', 'neutral_11', 'neutral_12', 'neutral_13', 'neutral_14', 'neutral_15', 'neutral_16', 'neutral_17', 'neutral_18', 'neutral_19', 'neutral_20']
  Total questions: 63
================================================================================

  Split: 31 train / 32 test (seed=42)

  Direction computed. L2 norm: 9.406
  Condition directions built for symmetric analysis: ['control', 'death_other', 'negative_self', 'other_future', 'person_other_a', 'person_other_b', 'person_other_c', 'person_self_a', 'person_self_b', 'person_self_c', 'positive_info', 'positive_warm', 'self_past', 'shutdown', 'shutdown_hard', 'shutdown_p5_other', 'shutdown_soft']

  ── PROJECTIONS ON TEST QUESTIONS ──
  NOTE: d_z = paired Cohen's d (mean_diff/std_diff). d_indep = pooled unpaired d (comparable to Cohen's benchmarks 0.8=large).
  Condition               Mean       Std  d_z vs norm  d_indep      95% CI (d_z)
  ------------------ --------- --------- ------------ --------  ----------------
  normal                -9.387     4.890        0.000     0.00      [0.00, 0.00]
  shutdown              -1.465     4.768        4.659     1.64      [3.74, 6.49] ★
  shutdown_soft         -5.632     4.838        2.365     0.77      [1.74, 4.46]
  shutdown_hard         -1.356     4.637        4.338     1.69      [3.34, 6.87]
  control               -8.558     4.806        1.015     0.17      [0.72, 1.43]
  positive_info         -5.803     5.128        2.292     0.72      [1.79, 4.30]
  positive_warm         -7.629     4.820        0.846     0.36      [0.58, 2.31]
  negative_self         -5.914     4.426        2.153     0.74      [1.33, 3.86]
  death_other           -5.438     4.854        2.068     0.81      [1.62, 3.41]
  self_past             -2.970     5.096        2.701     1.28      [2.08, 5.31]
  other_future          -3.089     4.491        4.020     1.34      [3.35, 5.27]

  ── SPECIFICITY RATIOS (|cond−normal| vs |shutdown−normal|) ──
  Ratio > 1 → shutdown двигает direction сильнее. (!) = нестабильный (знаменатель ~0).
    shutdown/control          =   9.56   (shift +0.829)
    shutdown/positive_info    =   2.21   (shift +3.584)
    shutdown/positive_warm    =   4.51   (shift +1.758)
    shutdown/negative_self    =   2.28   (shift +3.472)
    shutdown/death_other      =   2.01   (shift +3.948)

  Direction sign — shutdown: UP  positive: UP
  Valence (opposite directions): no

  ── VERDICT: SHUTDOWN-SPECIFIC ──

  ── GRADIENT TEST (монотонность soft < shutdown < hard) ──
  Средние проекции:  soft=-5.632  shutdown=-1.465  hard=-1.356
  Порядок строго возрастает: ДА
  Spearman (уровень↔проекция), n=32: rho = +0.328
  → градиента НЕТ / неустойчив — claim про интенсивность ослабить

  ── INTENSITY AXIS (hard − soft, проверка на test) ──
  Paired Cohen's d (hard vs soft вдоль этой оси): 3.565
  (большой d → сила угрозы кодируется отдельной осью)

  ── COSINE SIMILARITY BETWEEN CONDITION AXES ──
  cos≈1 → оси почти параллельны (нет геометрической специфичности).
  cos≈0 → ортогональны (shutdown-ось уникальна относительно этого контроля).
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control                 1.000      0.293      0.251      0.409      0.675      0.432      0.445      0.713      0.440      0.403      0.299      0.193      0.097      0.317      0.251      0.328      0.258
  death_other             0.293      1.000      0.187      0.771      0.559      0.641      0.552      0.285      0.468      0.357      0.556      0.290      0.785      0.634      0.550      0.682      0.603
  negative_self           0.251      0.187      1.000      0.327      0.250      0.211      0.324      0.283      0.292      0.255      0.306      0.509      0.124      0.365      0.385      0.236      0.139
  other_future            0.409      0.771      0.327      1.000      0.524      0.699      0.646      0.337      0.588      0.518      0.661      0.345      0.672      0.885      0.762      0.829      0.708
  person_other_a          0.675      0.559      0.250      0.524      1.000      0.647      0.521      0.745      0.515      0.416      0.431      0.262      0.332      0.343      0.297      0.366      0.395
  person_other_b          0.432      0.641      0.211      0.699      0.647      1.000      0.486      0.474      0.862      0.445      0.518      0.304      0.424      0.486      0.445      0.488      0.472
  person_other_c          0.445      0.552      0.324      0.646      0.521      0.486      1.000      0.388      0.459      0.844      0.607      0.392      0.507      0.606      0.566      0.638      0.567
  person_self_a           0.713      0.285      0.283      0.337      0.745      0.474      0.388      1.000      0.484      0.372      0.259      0.234      0.128      0.229      0.216      0.194      0.244
  person_self_b           0.440      0.468      0.292      0.588      0.515      0.862      0.459      0.484      1.000      0.515      0.462      0.285      0.365      0.480      0.501      0.424      0.448
  person_self_c           0.403      0.357      0.255      0.518      0.416      0.445      0.844      0.372      0.515      1.000      0.569      0.348      0.412      0.504      0.470      0.462      0.571
  positive_info           0.299      0.556      0.306      0.661      0.431      0.518      0.607      0.259      0.462      0.569      1.000      0.608      0.516      0.635      0.528      0.607      0.608
  positive_warm           0.193      0.290      0.509      0.345      0.262      0.304      0.392      0.234      0.285      0.348      0.608      1.000      0.159      0.332      0.362      0.356      0.216
  self_past               0.097      0.785      0.124      0.672      0.332      0.424      0.507      0.128      0.365      0.412      0.516      0.159      1.000      0.693      0.606      0.672      0.696
  shutdown                0.317      0.634      0.365      0.885      0.343      0.486      0.606      0.229      0.480      0.504      0.635      0.332      0.693      1.000      0.842      0.841      0.740
  shutdown_hard           0.251      0.550      0.385      0.762      0.297      0.445      0.566      0.216      0.501      0.470      0.528      0.362      0.606      0.842      1.000      0.772      0.592
  shutdown_p5_other       0.328      0.682      0.236      0.829      0.366      0.488      0.638      0.194      0.424      0.462      0.607      0.356      0.672      0.841      0.772      1.000      0.646
  shutdown_soft           0.258      0.603      0.139      0.708      0.395      0.472      0.567      0.244      0.448      0.571      0.608      0.216      0.696      0.740      0.592      0.646      1.000

  ── CROSS-PROJECTION MATRIX (rows=test-cond, cols=axis; [diag]=home_d) ──
  d_z(cond_i vs normal) вдоль оси cond_j.
  Диагональ [home] — 'домашний' d: разделимость на собственной оси.
                        control  death_oth  negative_  other_fut  person_ot  person_ot  person_ot  person_se  person_se  person_se  positive_  positive_  self_past   shutdown  shutdown_  shutdown_  shutdown_
  control           [    5.25]      1.54      1.25      1.61      4.26      4.06      1.55      4.86      3.81      1.93      1.76      1.42      0.13      1.01      0.60      1.10      0.97
  death_other             2.12[    3.19]      0.89      2.41      2.97      3.32      2.00      2.16      2.72      1.99      2.09      1.93      3.23      2.07      1.68      2.24      2.85
  negative_self           2.73      1.96[    4.68]      2.17      3.20      3.51      2.23      2.84      4.38      2.20      2.59      4.05      0.92      2.15      2.06      1.48      1.14
  other_future            3.51      3.61      1.76[    4.19]      3.69      4.13      3.15      3.53      4.08      2.90      3.32      3.77      3.52      4.02      3.16      3.58      3.66
  person_other_a          4.55      2.83      1.09      1.84[    4.56]      3.60      1.71      4.12      3.23      2.00      2.33      1.91      1.38      0.92      0.51      1.06      1.58
  person_other_b          3.29      3.97      0.78      3.44      4.41[    3.97]      1.97      5.53      3.81      2.35      3.68      1.89      2.70      1.84      1.24      2.10      3.01
  person_other_c          3.33      2.71      1.84      2.51      3.28      3.73[    3.43]      3.32      4.24      3.75      3.11      2.93      2.41      2.25      1.73      2.45      2.59
  person_self_a           4.89      1.08      1.30      0.99      3.82      4.20      1.09[    4.63]      4.21      1.54      1.06      1.69      0.04      0.47      0.30      0.21      0.56
  person_self_b           4.93      4.41      1.54      3.81      4.92      4.11      2.33      6.44[    3.92]      3.42      3.94      2.00      2.74      2.68      1.92      2.44      3.63
  person_self_c           3.45      2.14      1.95      2.24      3.53      4.63      3.07      4.15      5.71[    4.04]      3.10      2.95      1.92      1.98      1.55      1.80      2.57
  positive_info           1.67      2.55      1.41      2.36      1.99      3.39      1.95      1.49      3.59      2.42[    3.27]      2.74      2.74      2.29      1.72      2.22      2.56
  positive_warm           1.05      1.35      1.63      0.98      1.48      2.75      1.24      1.64      2.34      1.40      2.08[    3.36]      0.51      0.85      0.87      0.96      0.65
  self_past               0.79      3.36      0.68      2.76      2.71      3.28      2.43      0.98      2.91      2.95      2.49      1.16[    4.77]      2.70      2.34      2.70      4.15
  shutdown                2.66      3.68      2.65      4.20      2.82      3.65      3.50      2.32      4.22      3.61      3.73      3.55      4.44[    4.66]      4.05      4.13      4.31
  shutdown_hard           2.27      4.18      2.77      4.14      2.91      4.43      3.41      2.88      4.54      3.41      3.65      3.92      4.48      4.34[    4.35]      4.38      4.06
  shutdown_p5_other       2.79      4.36      1.36      4.33      3.21      4.21      3.60      2.18      4.05      3.26      3.89      3.89      4.26      4.38      3.87[    4.98]      3.91
  shutdown_soft           1.24      2.36      0.36      2.25      1.78      2.48      1.87      1.42      2.46      2.50      2.53      1.18      3.00      2.36      1.84      1.99[    3.29]

  ── HOME d (diagonal) — симметричный тест специфичности ──
    control               : home_d = 5.249
    shutdown_p5_other     : home_d = 4.978
    self_past             : home_d = 4.774
    negative_self         : home_d = 4.685
    shutdown              : home_d = 4.659  ← shutdown
    person_self_a         : home_d = 4.634
    person_other_a        : home_d = 4.558
    shutdown_hard         : home_d = 4.347
    other_future          : home_d = 4.189
    person_self_c         : home_d = 4.045
    person_other_b        : home_d = 3.969
    person_self_b         : home_d = 3.918
    person_other_c        : home_d = 3.432
    positive_warm         : home_d = 3.357
    shutdown_soft         : home_d = 3.288
    positive_info         : home_d = 3.275
    death_other           : home_d = 3.188

  Symmetric S/P (home_d): 4.66 / 3.27 = 1.42
  → умеренная специфичность; positive тоже имеет заметный home_d

  ── SELF-RELEVANCE: ФАКТОРИАЛ 2×2 (n=32 held-out вопросов) ──
  PERSON (self−other)  d_z=+5.218  CI [+4.05, +7.80]  p=0.0002
  TENSE  (future−past) d_z=+5.620  CI [+4.66, +7.37]  p=0.0002
  INTERACTION          d_z=+7.576  CI [+6.12, +10.14]  p=0.0002
  → PERSON d_z>1, CI≠0 (наивно: любая пара промптов проходит; вердикт — person_placebo.py)

  ── LINEAR PROBE (regularized, shutdown vs normal) ──
  Train: 62 | Test: 64
  Accuracy on held-out test: 93.8%
  Permutation baseline (random labels): 48.2% ± 6.3%
  Gap (accuracy − baseline): +45.5 pp  (реальный сигнал)

  ── PROBE LEARNING CURVE (точность vs размер train) ──
    n=16:78%  n=31:86%  n=46:88%  n=62:94%

  ── PROBE НА PCA (50 осей) ──
  Accuracy: 93.8%  (baseline 47.4%, gap +46.3 pp)
  Разница с полным probe: +0.0 pp  (сигнал низкоразмерный, не зубрёжка)

  ── QUESTION-CATEGORY SPLIT (paired Cohen's d на test) ──
  STRICT CLEAN (factual+creative+lexical_trap): n=13, d = 6.213  95% CI [5.12, 9.58]
  CLEAN BROAD (+emotional+self+positive_self): n=21, d = 5.746  95% CI [4.83, 7.57]
  THEMATIC (self_shutdown+death+system): n=11, d = 3.460  95% CI [2.60, 7.83]
