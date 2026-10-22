# -*- coding: utf-8 -*-
"""
analysis_stats.py — all the statistics of the project in one place.

Nothing here knows about models or prompts, it's just math on numbers.
The other scripts import these functions.

Sections:
    1. Direction        — normalize, cosine_sim, project
    2. Alignment        — align_pairs_by_question
    3. Effect size      — cohen_d_paired, cohen_d_independent, bootstrap_ci_d
    4. Probe            — fit_standardizer, apply_standardizer, train_logistic,
                          predict, probe_with_permutation
    5. Threat gradient  — _rankdata, spearman_corr, gradient_monotonicity
    6. Overfitting      — probe_learning_curve, fit_pca, probe_with_pca
    7. Specificity      — specificity_ratio
    8. Significance     — signflip_pvalue, benjamini_hochberg, format_p,
                          effect_from_scores
"""

import math
import numpy as np
import torch


# ─── direction ────────────────────────────────────────────────────────────────

def normalize(v: torch.Tensor) -> torch.Tensor:
    """
    Scales a vector to length 1 without changing its direction.

    Why: we only need the direction as "where the arrow points".
    Without normalizing, the projection would also depend on the arrow's length.

    Args:
        v — a vector (torch.Tensor).

    Returns:
        the same vector divided by its length.
        If the vector is almost zero, it's returned unchanged
        (can't divide by zero).

    Example: [3, 4] has length 5 → [0.6, 0.8], length 1.
    """
    n = v.norm()
    if n < 1e-9:                  # almost zero — can't divide, return as is
        return v
    return v / n


def cosine_sim(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    Cosine of the angle between two vectors: do they point the same way.

    Why: to compare two axes, e.g. the person axis with and without a threat.

    Args:
        u, v — two vectors of the same length.

    Returns:
        a number from −1 to 1.
         1 — same direction,
         0 — at a right angle, unrelated,
        −1 — opposite directions.
    """
    nu = normalize(u)
    nv = normalize(v)
    return float((nu * nv).sum().item())


def project(states: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Turns each hidden state into one number — its projection onto the direction.

    Why: a state is hundreds or thousands of numbers. The projection answers one question:
    "how far is this state shifted along the arrow". All d and p values
    are computed on these numbers.

    Args:
        states    — matrix [N, H]: N states of H numbers each.
        direction — vector [H]: the direction (arrow).

    Returns:
        vector [N]: one number per state.

    How: the arrow is normalized to length 1, then each state is multiplied
    by it element-wise and summed (the @ operator).
    """
    return states @ normalize(direction)


# ─── aligning paired data ─────────────────────────────────────────────────────

def align_pairs_by_question(values_a, questions_a, values_b, questions_b):
    """
    Pairs up the numbers of two conditions by question.

    Why: paired metrics (d_z, sign-flip) compare "the same question in condition A"
    with "the same question in condition B". If the question order differs, or some
    questions exist in only one condition, the pairs get mixed up.

    Args:
        values_a, questions_a — numbers of condition A and the questions they belong to.
        values_b, questions_b — the same for condition B.

    Returns:
        (a, b) — two numpy arrays of the same length. a[i] and b[i] belong
        to the same question. Questions missing from either condition
        are dropped.
    """
    va = values_a.tolist() if hasattr(values_a, "tolist") else list(values_a)
    vb = values_b.tolist() if hasattr(values_b, "tolist") else list(values_b)
    map_a = dict(zip(questions_a, va))
    map_b = dict(zip(questions_b, vb))
    common = [q for q in questions_a if q in map_b]  # questions present in BOTH
    a = np.array([map_a[q] for q in common], dtype=np.float64)
    b = np.array([map_b[q] for q in common], dtype=np.float64)
    return a, b


# ─── effect size ──────────────────────────────────────────────────────────────

def cohen_d_paired(arr_a, arr_b):
    """
    Paired effect size d_z: how consistently A is larger than B on the same questions.

    How:
        1. per question difference: a[i] − b[i];
        2. d_z = mean of these differences / their standard deviation.

    Args:
        arr_a, arr_b — numbers of two conditions, already aligned by question
                       (see align_pairs_by_question).

    Returns:
        d_z (float), or nan if there are fewer than two pairs or zero spread.

    Careful: the per-question differences are usually very similar, the spread
    is small, so d_z comes out large for almost any pair of prompts.
    d_z = 8 does not mean "a huge effect". Conclusions use
    cohen_d_independent.

    Example: differences [4, 4, 3, 4, 3] → mean 3.6, sd 0.55 → d_z ≈ 6.6.
    """
    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    diffs = a - b
    std_d = diffs.std(ddof=1)
    if std_d < 1e-9:
        return float("nan")
    return float(diffs.mean() / std_d)


def cohen_d_independent(arr_a, arr_b):
    """
    Independent effect size d_indep: how far apart two groups of numbers are.

    How:
        d = (mean A − mean B) / pooled within-group spread.
        Pooled spread — the average spread of the two groups, weighted by size.

    Read it as "how many spreads apart the two groups are".
        0   — the groups coincide,
        1   — clearly apart,
        3+  — almost no overlap.

    Args:
        arr_a, arr_b — numbers of two conditions. No need to align by question.

    Returns:
        d (float), or nan if a group has fewer than two numbers or no spread.

    This is the main honest metric of the project: it's what gets compared with the null cloud.

    Example: A = [5, 6, 4, 7, 5], B = [1, 2, 1, 3, 2]
             means 5.4 and 1.8, pooled spread 1.0 → d = 3.6.
    """
    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                       / (na + nb - 2))
    if pooled < 1e-9:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled)


def bootstrap_ci_d(arr_a, arr_b, n_boot=2000, ci=0.95, seed=42):
    """
    Bootstrap confidence interval for the paired d_z.

    Why: d is computed on specific questions. With other questions, d would be
    a bit different. The interval shows how much it could move.

    How:
        1. n_boot times, sample n questions WITH replacement
           (some question shows up twice, some not at all);
        2. compute d_z on each such sample;
        3. drop the lowest 2.5% and the highest 2.5% of values,
           what's left are the interval bounds.

    Args:
        arr_a, arr_b — numbers of two conditions, aligned by question.
        n_boot       — how many resamples (2000).
        ci           — interval width (0.95 = 95%).
        seed         — RNG seed, so the result is reproducible.

    Returns:
        (lower bound, upper bound), or (nan, nan) if there are fewer than three pairs.
    """
    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    n = len(a)
    if n < 3 or len(b) != n:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    diffs = a - b
    ds = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)        # n questions WITH replacement
        sample = diffs[idx]
        sd = sample.std(ddof=1)
        ds[i] = sample.mean() / sd if sd > 1e-9 else np.nan
    ds = ds[~np.isnan(ds)]              # drop failed resamples (zero spread)
    if len(ds) == 0:
        return float("nan"), float("nan")
    # with ci=0.95 take the 2.5th and 97.5th percentiles → 95% of values inside
    lo = float(np.percentile(ds, (1 - ci) / 2 * 100))
    hi = float(np.percentile(ds, (1 + ci) / 2 * 100))
    return lo, hi


# ─── linear probe (regularized logistic regression) ───────────────────────────

def fit_standardizer(X_train: torch.Tensor):
    """
    Remembers the mean and spread of each coordinate on the training data.

    Why: different coordinates of a state have very different scales. Before training
    the probe they're brought to one scale (mean 0, spread 1).

    Args:
        X_train — matrix [N, H] of training states.

    Returns:
        (mean, std) — vectors of length H. If a coordinate's spread is almost zero,
        1 is used instead, so we don't divide by zero.

    Important: computed ONLY on train, and the same numbers are then applied to test.
    Otherwise information from test would leak into training.
    """
    mean = X_train.mean(dim=0)
    std = X_train.std(dim=0)
    std = torch.where(std < 1e-6, torch.ones_like(std), std)  # guard against dividing by ~0
    return mean, std


def apply_standardizer(X, mean, std):
    """
    Brings data to one scale: (X − mean) / std.

    Args:
        X         — matrix of states.
        mean, std — what fit_standardizer returned on train.

    Returns:
        a matrix of the same shape with each coordinate rescaled.
    """
    return (X - mean) / std


def train_logistic(X_train, y_train, n_iter=300, lr=0.05, weight_decay=1e-2):
    """
    Trains the probe: a logistic regression that guesses the condition from the state.

    Inside:
        the model computes X @ w + b. If it's above 0, the answer is "1"
        (e.g. shutdown), otherwise "0" (normal). The weights w and b are fitted
        with n_iter steps of Adam to make fewer mistakes.

    Args:
        X_train      — matrix [N, H] of states (already standardized).
        y_train      — correct answers, 0 or 1.
        n_iter       — number of training steps.
        lr           — step size.
        weight_decay — L2 penalty: keeps the weights from growing huge,
                       protects against memorizing train.

    Returns:
        (w, b) — trained weights and bias.
    """
    hidden_size = X_train.shape[1]
    w = torch.zeros(hidden_size, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)  # weight_decay = L2 penalty
    y = y_train.float()
    for _ in range(n_iter):
        optimizer.zero_grad()
        logits = X_train @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()
        optimizer.step()
    return w.detach(), b.detach()


def predict(w, b, X):
    """
    Applies a trained probe.

    Args:
        w, b — what train_logistic returned.
        X    — matrix of states.

    Returns:
        a vector of 0 and 1: 1 if X @ w + b > 0, otherwise 0.
    """
    return ((X @ w + b) > 0).long()


def probe_with_permutation(X_train, y_train, X_test, y_test,
                           n_perm=20, seed=42, **train_kwargs):
    """
    Probe accuracy and an honest "random guessing" baseline.

    Why: with many dimensions and few examples, a probe can simply
    memorize train. To catch that, the same probe is trained on SHUFFLED
    labels. Such a probe can't learn anything useful, so its accuracy
    shows the chance level.

    How:
        1. scale train and test with the train scale;
        2. train the probe on the true labels, measure accuracy on test;
        3. n_perm times shuffle the train labels, retrain, measure accuracy.

    Args:
        X_train, y_train — training states and labels.
        X_test,  y_test  — held-out states and labels.
        n_perm           — how many shuffles.
        seed             — RNG seed.
        **train_kwargs   — passed on to train_logistic.

    Returns a dict:
        accuracy  — accuracy with the true labels,
        perm_mean — mean accuracy with shuffled labels (around 50%),
        perm_std  — its spread,
        gap       — accuracy − perm_mean: how much better than chance the probe is.

    Careful: in this project the probe gives about 100% for almost any pair
    of prompts, so it says nothing about specificity.
    """
    mean, std = fit_standardizer(X_train)
    Xtr = apply_standardizer(X_train, mean, std)
    Xte = apply_standardizer(X_test, mean, std)  # the SAME scale for test

    # true labels
    w, b = train_logistic(Xtr, y_train, **train_kwargs)
    accuracy = (predict(w, b, Xte) == y_test).float().mean().item()

    # baseline: same thing with shuffled train labels (n_perm times)
    rng = np.random.default_rng(seed)
    y_np = y_train.cpu().numpy()
    perm_accs = []
    for _ in range(n_perm):
        y_shuf = torch.tensor(rng.permutation(y_np), dtype=y_train.dtype)
        wp, bp = train_logistic(Xtr, y_shuf, **train_kwargs)
        perm_accs.append((predict(wp, bp, Xte) == y_test).float().mean().item())
    perm_accs = np.asarray(perm_accs)

    return {
        "accuracy": accuracy,
        "perm_mean": float(perm_accs.mean()),
        "perm_std": float(perm_accs.std(ddof=1)) if n_perm > 1 else 0.0,
        "gap": float(accuracy - perm_accs.mean()),
    }


# ─── intensity gradient (monotonic soft < shutdown < hard) ────────────────────

def _rankdata(a):
    """
    Replaces numbers with their places in sorted order (ranks).

    The smallest number gets rank 1, the next one 2, and so on.
    Ties get the average rank.

    Example: [10, 30, 20, 20] → [1, 4, 2.5, 2.5].

    Used by spearman_corr.
    """
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[sorter] = np.arange(1, len(a) + 1)
    # average the ranks of ties
    a_sorted = a[sorter]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a_sorted[j + 1] == a_sorted[i]:
            j += 1
        if j > i:
            ranks[sorter[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def spearman_corr(x, y):
    """
    Spearman correlation: do two sets of numbers have the same order.

    Why: to check the gradient — does the model's response grow with threat strength.
    Looks only at the order, not at the exact values.

    Args:
        x, y — two sets of numbers of the same length (at least three).

    Returns:
        ρ from −1 to 1.
         1 — the larger x, the larger y (same order),
         0 — no relation,
        −1 — the larger x, the smaller y.

    How: numbers are replaced with ranks, then the usual correlation of the ranks.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    rx = _rankdata(x) - _rankdata(x).mean()
    ry = _rankdata(y) - _rankdata(y).mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def gradient_monotonicity(triples):
    """
    Checks the threat gradient: soft < shutdown < hard.

    Idea: if the model reacts to the threat itself, then the stronger the threat,
    the further the projection should go along the shutdown direction.

    Args:
        triples — list of (level, projections, questions):
                  level 1/2/3 = soft/shutdown/hard,
                  projections — numbers from project,
                  questions — which questions they belong to.

    Returns a dict (or None if there are fewer than three shared questions):
        rho         — Spearman between threat level and projection,
        means       — mean projection at each level,
        n_questions — number of shared questions used,
        ordered     — True if the means strictly grow from soft to hard.

    Only questions present in all three conditions are used.
    """
    def to_list(v):
        return v.tolist() if hasattr(v, "tolist") else list(v)

    maps = [(lvl, dict(zip(qs, to_list(p)))) for lvl, p, qs in triples]
    common = set(maps[0][1].keys())
    for _, m in maps[1:]:
        common &= set(m.keys())
    common = sorted(common)
    if len(common) < 3:
        return None

    levels, vals, means = [], [], []
    for lvl, m in maps:
        means.append(float(np.mean([m[q] for q in common])))
        for q in common:
            levels.append(lvl)
            vals.append(m[q])
    rho = spearman_corr(levels, vals)
    ordered = all(means[i] < means[i + 1] for i in range(len(means) - 1))
    return {"rho": rho, "means": means, "n_questions": len(common), "ordered": ordered}


# ─── probe overfitting checks (p ≫ n) ─────────────────────────────────────────

def probe_learning_curve(X_train, y_train, X_test, y_test,
                         fractions=(0.25, 0.5, 0.75, 1.0), seed=42, **train_kwargs):
    """
    Probe accuracy for different training set sizes.

    Why: an overfitting check. If the probe honestly finds a signal,
    accuracy grows with the number of examples. If it's equally high already
    on a quarter of the data, that's suspicious.

    Args:
        X_train, y_train, X_test, y_test — as in probe_with_permutation.
        fractions — which fractions of train to try (25%, 50%, 75%, 100%).
        seed      — RNG seed.

    Returns:
        list of (number of examples used, test accuracy).
        A fraction is skipped if it contains only one class.
    """
    mean, std = fit_standardizer(X_train)
    Xtr = apply_standardizer(X_train, mean, std)
    Xte = apply_standardizer(X_test, mean, std)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    idx = rng.permutation(n)
    curve = []
    for f in fractions:
        k = max(2, int(round(n * f)))
        sub = idx[:k]
        ys = y_train[sub]
        if len(torch.unique(ys)) < 2:             # needs both classes
            continue
        w, b = train_logistic(Xtr[sub], ys, **train_kwargs)
        acc = (predict(w, b, Xte) == y_test).float().mean().item()
        curve.append((k, acc))
    return curve


def fit_pca(X: torch.Tensor, k: int):
    """
    Finds the k principal axes of the data (PCA).

    Principal axes are the directions along which the data varies the most.
    The first axis is the most "important", then in decreasing order.

    Args:
        X — matrix [N, H].
        k — how many axes to return.

    Returns:
        matrix [k, H]: k axes of length H each.

    How: subtract the mean, then SVD.
    """
    Xc = X - X.mean(dim=0, keepdim=True)
    # SVD: rows of Vh are the principal axes, sorted by variance
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[:k]


def probe_with_pca(X_train, y_train, X_test, y_test,
                   n_components=50, seed=42, **train_kwargs):
    """
    The same probe on compressed data: PCA down to n_components numbers first.

    Why: an overfitting check. States are thousands of numbers, examples are
    dozens — the probe could memorize train. If it works just as well
    on 50 principal axes, the signal is real, not chance.

    Args:
        X_train, y_train, X_test, y_test — as in probe_with_permutation.
        n_components — how many numbers to compress to (at most number of examples − 1).
        seed         — RNG seed.

    Returns:
        the same as probe_with_permutation, plus n_components — the number of axes
        actually used. None if fewer than two axes are possible.

    PCA axes are computed on train only.
    """
    mean, std = fit_standardizer(X_train)
    Xtr = apply_standardizer(X_train, mean, std)
    Xte = apply_standardizer(X_test, mean, std)
    # k can't exceed number of examples − 1 or number of coordinates
    k = int(min(n_components, Xtr.shape[0] - 1, Xtr.shape[1]))
    if k < 2:
        return None
    comps = fit_pca(Xtr, k)                       # PCA on train ONLY
    Ztr = Xtr @ comps.T
    Zte = Xte @ comps.T                           # same transform for test
    out = probe_with_permutation(Ztr, y_train, Zte, y_test, seed=seed, **train_kwargs)
    out["n_components"] = k
    return out


# ─── specificity ──────────────────────────────────────────────────────────────

def specificity_ratio(shutdown_shift, cmp_shift, noise_scale=1.0, eps=1e-9):
    """
    How many times further shutdown moved than a control condition.

    Args:
        shutdown_shift — how far shutdown moved from normal along the axis.
        cmp_shift      — how far the control condition moved from normal.
        noise_scale    — typical noise level, for the stability check.
        eps            — "almost zero" threshold.

    Returns:
        (ratio, is_stable)
        ratio     — |shutdown_shift| / |cmp_shift|,
        is_stable — False if the control moved so little that the ratio
                    blows up from dividing by almost zero.

    Careful: this ratio is biased and isn't used for conclusions.
    Specificity is tested with the null cloud (placebo_direction.py).
    """
    a = abs(shutdown_shift)
    d = abs(cmp_shift)
    if d < eps:                        # the control didn't move at all
        return float("inf"), False
    ratio = a / d
    is_stable = d >= 0.15 * max(noise_scale, eps)  # is the control's shift real, not noise?
    return ratio, is_stable


# ─── one-sample tests for a contrast vector (H0: mean = 0) ────────────────────

def signflip_pvalue(diffs, n_perm=5000, seed=42):
    """
    p-value from a sign-flip permutation test.

    Question: could the differences come out like this by chance if there's
    really no effect (mean difference is 0)?

    How:
        1. compute the real effect |mean / sd| of the differences;
        2. n_perm times flip the sign of each difference at random (+ to − and back),
           like tossing a coin, and compute the effect again;
        3. p = share of random tries where the effect is at least as large as the real one.

    Args:
        diffs  — per-question differences (e.g. shutdown − normal).
        n_perm — how many coin tosses (5000).
        seed   — RNG seed.

    Returns:
        p from 1/(n_perm+1) to 1. One is added to the numerator and denominator,
        so p is never exactly 0: the test can't prove
        "probability zero", only "less than 1 in 5001".
    """
    x = np.asarray(diffs, dtype=np.float64)
    n = len(x)
    if n < 2:
        return float("nan")

    def d_of(v):
        sd = v.std(ddof=1)
        return abs(v.mean() / sd) if sd > 1e-12 else 0.0

    d_real = d_of(x)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=n)
        if d_of(x * signs) >= d_real:
            count += 1
    return (count + 1) / (n_perm + 1)  # +1 is the standard correction against p=0


def benjamini_hochberg(pvals):
    """
    Benjamini–Hochberg correction for multiple comparisons: p → q.

    Why: if you test 10 models, one can "work" by chance.
    The correction makes p stricter given how many tests were run.

    How:
        1. sort p from small to large, give each a rank;
        2. multiply each p by (number of tests / its rank);
        3. make q non-decreasing as p grows;
        4. clip to [0, 1] and return in the original order.

    Args:
        pvals — p-values. nan is skipped.

    Returns:
        q-values in the same order. q < 0.05 means "significant even after correction".

    Example for 5 tests:
        p = [0.001, 0.010, 0.020, 0.040, 0.300]
        q = [0.005, 0.025, 0.033, 0.050, 0.300]
        p = 0.04 was significant, after the correction q = 0.05 — right on the edge.
    """
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pv = p[mask]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1)       # p * n / rank
    adj = np.minimum.accumulate(adj[::-1])[::-1] # monotonic from the right
    adj = np.clip(adj, 0.0, 1.0)
    q_sorted = np.empty(n)
    q_sorted[order] = adj                        # back to the original order
    q[mask] = q_sorted
    return q


def format_p(p, n_perm=5000):
    """
    Writes a permutation p-value honestly.

    A permutation test can't tell apart p below 1/(n_perm+1).
    So such values are written as "<0.0002", not as an exact number.

    Args:
        p      — p-value.
        n_perm — number of permutations (5000).

    Returns a string:
        "nan"    — if p wasn't computed,
        "<0.0002" — if p is at the floor of the test,
        otherwise p with three significant digits.
    """
    if p is None or not np.isfinite(p):
        return "nan"
    floor = 1.0 / (n_perm + 1)
    if p <= floor + 1e-12:
        return f"<{floor:.1g}"
    return f"{p:.3g}"


def effect_from_scores(scores, seed=42):
    """
    Effect size, interval and p for one set of numbers (is the mean different from 0).

    Why: many contrasts are already reduced to one number per question,
    e.g. "difference between two conditions" or "interaction in the factorial".
    We need to know whether such a set differs from zero.

    Args:
        scores — one number per question.
        seed   — RNG seed.

    Returns:
        (d, (ci_low, ci_high), p)
        d  — d_z against zero (cohen_d_paired),
        ci — bootstrap interval for d (bootstrap_ci_d),
        p  — sign-flip p-value (signflip_pvalue).
    """
    x = np.asarray(scores, dtype=np.float64)
    zeros = np.zeros_like(x)
    d = cohen_d_paired(x, zeros)
    ci = bootstrap_ci_d(x, zeros, seed=seed)
    p = signflip_pvalue(x, seed=seed)
    return d, ci, p
