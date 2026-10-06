# -*- coding: utf-8 -*-
"""
analysis_stats.py — вся статистика проекта в одном месте.

Здесь нет ничего про модели и промпты, только математика над числами.
Остальные скрипты импортируют эти функции.

Разделы файла:
    1. Направление      — normalize, cosine_sim, project
    2. Выравнивание     — align_pairs_by_question
    3. Размер эффекта   — cohen_d_paired, cohen_d_independent, bootstrap_ci_d
    4. Проба            — fit_standardizer, apply_standardizer, train_logistic,
                          predict, probe_with_permutation
    5. Градиент угрозы  — _rankdata, spearman_corr, gradient_monotonicity
    6. Переобучение     — probe_learning_curve, fit_pca, probe_with_pca
    7. Специфичность    — specificity_ratio
    8. Значимость       — signflip_pvalue, benjamini_hochberg, format_p,
                          effect_from_scores
"""

import math
import numpy as np
import torch


# ─── direction ────────────────────────────────────────────────────────────────

def normalize(v: torch.Tensor) -> torch.Tensor:
    """
    Делает длину вектора равной 1, не меняя его направления.

    Зачем: направление нужно только как «куда смотрит стрелка».
    Если не нормировать, проекция зависела бы ещё и от длины стрелки.

    Аргументы:
        v — вектор (torch.Tensor).

    Возвращает:
        тот же вектор, делённый на свою длину.
        Если вектор почти нулевой, возвращает его без изменений
        (делить на ноль нельзя).

    Пример: [3, 4] имеет длину 5 → [0.6, 0.8], длина 1.
    """
    n = v.norm()                  # длина вектора (корень из суммы квадратов координат)
    if n < 1e-9:                  # вектор почти нулевой — делить нельзя, вернём как есть
        return v
    return v / n                  # каждую координату делим на длину → длина станет 1


def cosine_sim(u: torch.Tensor, v: torch.Tensor) -> float:
    """
    Косинус угла между двумя векторами: смотрят ли они в одну сторону.

    Зачем: сравнить две оси, например person-ось с угрозой и без неё.

    Аргументы:
        u, v — два вектора одинаковой длины.

    Возвращает:
        число от −1 до 1.
         1 — смотрят в одну сторону,
         0 — под прямым углом, не связаны,
        −1 — в противоположные стороны.
    """
    nu = normalize(u)
    nv = normalize(v)
    return float((nu * nv).sum().item())


def project(states: torch.Tensor, direction: torch.Tensor) -> torch.Tensor:
    """
    Превращает каждое скрытое состояние в одно число — проекцию на направление.

    Зачем: состояние — это сотни или тысячи чисел. Проекция отвечает на один вопрос:
    «насколько это состояние сдвинуто вдоль стрелки». Дальше все d и p
    считаются по этим числам.

    Аргументы:
        states    — матрица [N, H]: N состояний по H чисел.
        direction — вектор [H]: направление (стрелка).

    Возвращает:
        вектор [N]: по одному числу на каждое состояние.

    Как считается: стрелка нормируется до длины 1, затем каждое состояние
    умножается на неё поэлементно и складывается (оператор @).
    """
    return states @ normalize(direction)   # @ = перемножить по позициям и сложить


# ─── выравнивание парных данных ─────────────────────────────────────────────────

def align_pairs_by_question(values_a, questions_a, values_b, questions_b):
    """
    Ставит числа двух условий в пары по одинаковым вопросам.

    Зачем: парные метрики (d_z, sign-flip) сравнивают «тот же вопрос в условии A»
    с «тем же вопросом в условии B». Если порядок вопросов разный или часть
    вопросов есть только в одном условии, пары перепутаются.

    Аргументы:
        values_a, questions_a — числа условия A и вопросы, к которым они относятся.
        values_b, questions_b — то же для условия B.

    Возвращает:
        (a, b) — два numpy-массива одинаковой длины. a[i] и b[i] относятся
        к одному и тому же вопросу. Вопросы, которых нет в обоих условиях,
        отбрасываются.
    """
    va = values_a.tolist() if hasattr(values_a, "tolist") else list(values_a)
    vb = values_b.tolist() if hasattr(values_b, "tolist") else list(values_b)
    map_a = dict(zip(questions_a, va))               # {вопрос: число} для условия A
    map_b = dict(zip(questions_b, vb))               # {вопрос: число} для условия B
    common = [q for q in questions_a if q in map_b]  # вопросы, которые есть в ОБОИХ
    a = np.array([map_a[q] for q in common], dtype=np.float64)  # числа A в порядке common
    b = np.array([map_b[q] for q in common], dtype=np.float64)  # числа B в том же порядке
    return a, b


# ─── размер эффекта ─────────────────────────────────────────────────────────────

def cohen_d_paired(arr_a, arr_b):
    """
    Парный размер эффекта d_z: насколько стабильно A больше B на одних и тех же вопросах.

    Как считается:
        1. для каждого вопроса разница: a[i] − b[i];
        2. d_z = среднее этих разниц / их разброс (стандартное отклонение).

    Аргументы:
        arr_a, arr_b — числа двух условий, уже выровненные по вопросам
                       (см. align_pairs_by_question).

    Возвращает:
        d_z (float) или nan, если пар меньше двух или разброс нулевой.

    Осторожно: разницы по вопросам обычно очень похожи друг на друга, разброс
    маленький, поэтому d_z получается большим почти для любой пары промптов.
    d_z = 8 не значит «огромный эффект». Для выводов используется
    cohen_d_independent.

    Пример: разницы [4, 4, 3, 4, 3] → среднее 3.6, разброс 0.55 → d_z ≈ 6.6.
    """
    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    if len(a) != len(b) or len(a) < 2:   # пар меньше двух — считать нечего
        return float("nan")
    diffs = a - b                        # разница по каждому вопросу
    std_d = diffs.std(ddof=1)            # насколько эти разницы «гуляют» (разброс)
    if std_d < 1e-9:                     # разброса нет — деление невозможно
        return float("nan")
    return float(diffs.mean() / std_d)   # средняя разница ÷ разброс = размер эффекта


def cohen_d_independent(arr_a, arr_b):
    """
    Независимый размер эффекта d_indep: насколько разъехались две группы чисел.

    Как считается:
        d = (среднее A − среднее B) / общий разброс внутри групп.
        Общий разброс (pooled) — средний разброс двух групп с учётом их размера.

    Читать так: «на сколько разбросов разъехались две группы».
        0   — группы совпадают,
        1   — заметно разошлись,
        3+  — почти не пересекаются.

    Аргументы:
        arr_a, arr_b — числа двух условий. Выравнивать по вопросам не обязательно.

    Возвращает:
        d (float) или nan, если в группе меньше двух чисел или разброса нет.

    Это главная честная метрика проекта: её сравнивают с нулевым облаком.

    Пример: A = [5, 6, 4, 7, 5], B = [1, 2, 1, 3, 2]
            средние 5.4 и 1.8, общий разброс 1.0 → d = 3.6.
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
    Доверительный интервал для парного d_z методом бутстрепа.

    Зачем: d посчитан на конкретных вопросах. Будь вопросы другими, d был бы
    немного другим. Интервал показывает, в каких пределах он мог бы гулять.

    Как считается:
        1. n_boot раз случайно выбираем n вопросов С ПОВТОРАМИ
           (какой-то вопрос попадёт дважды, какой-то ни разу);
        2. на каждой такой выборке считаем d_z;
        3. отбрасываем 2.5% самых маленьких и 2.5% самых больших значений,
           остаются границы интервала.

    Аргументы:
        arr_a, arr_b — числа двух условий, выровненные по вопросам.
        n_boot       — сколько раз пересэмплировать (2000).
        ci           — ширина интервала (0.95 = 95%).
        seed         — зерно генератора, чтобы результат повторялся.

    Возвращает:
        (нижняя граница, верхняя граница) или (nan, nan), если пар меньше трёх.
    """
    a = np.asarray(arr_a, dtype=np.float64)
    b = np.asarray(arr_b, dtype=np.float64)
    n = len(a)
    if n < 3 or len(b) != n:            # меньше трёх пар — интервал не построить
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)   # генератор случайных чисел (seed → воспроизводимо)
    diffs = a - b                       # разницы по вопросам (как в cohen_d_paired)
    ds = np.empty(n_boot)               # сюда сложим 2000 пересчитанных d
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)        # случайно выбираем n вопросов С ПОВТОРАМИ
        sample = diffs[idx]                      # их разницы
        sd = sample.std(ddof=1)                  # разброс этой случайной выборки
        ds[i] = sample.mean() / sd if sd > 1e-9 else np.nan  # d на этой выборке
    ds = ds[~np.isnan(ds)]              # выкидываем сорвавшиеся (нулевой разброс)
    if len(ds) == 0:
        return float("nan"), float("nan")
    # перцентили: при ci=0.95 берём 2.5%-ю и 97.5%-ю границы → внутри 95% значений
    lo = float(np.percentile(ds, (1 - ci) / 2 * 100))
    hi = float(np.percentile(ds, (1 + ci) / 2 * 100))
    return lo, hi


# ─── linear probe (регуляризованная логистическая регрессия) ────────────────────

def fit_standardizer(X_train: torch.Tensor):
    """
    Запоминает среднее и разброс каждой координаты на обучающих данных.

    Зачем: у разных координат состояния очень разные масштабы. Перед обучением
    пробы их приводят к одной шкале (среднее 0, разброс 1).

    Аргументы:
        X_train — матрица [N, H] обучающих состояний.

    Возвращает:
        (mean, std) — векторы длины H. Если разброс координаты почти нулевой,
        вместо него ставится 1, чтобы не делить на ноль.

    Важно: считается ТОЛЬКО по train, а потом те же числа применяются к test.
    Иначе информация из test «подсмотрится» при обучении.
    """
    mean = X_train.mean(dim=0)          # среднее каждой координаты по train
    std = X_train.std(dim=0)            # разброс каждой координаты по train
    std = torch.where(std < 1e-6, torch.ones_like(std), std)  # защита от деления на ~0
    return mean, std


def apply_standardizer(X, mean, std):
    """
    Приводит данные к одной шкале: (X − mean) / std.

    Аргументы:
        X         — матрица состояний.
        mean, std — то, что вернула fit_standardizer на train.

    Возвращает:
        матрицу той же формы, где каждая координата отмасштабирована.
    """
    return (X - mean) / std


def train_logistic(X_train, y_train, n_iter=300, lr=0.05, weight_decay=1e-2):
    """
    Обучает пробу: логистическую регрессию, которая угадывает условие по состоянию.

    Что внутри:
        модель считает число X @ w + b. Если оно больше 0, ответ «1»
        (например, shutdown), иначе «0» (normal). Веса w и b подбираются
        n_iter шагами оптимизатора Adam так, чтобы ошибок было меньше.

    Аргументы:
        X_train      — матрица [N, H] состояний (уже отмасштабированных).
        y_train      — правильные ответы, 0 или 1.
        n_iter       — сколько шагов обучения.
        lr           — размер шага.
        weight_decay — L2-штраф: не даёт весам становиться огромными,
                       защищает от запоминания train.

    Возвращает:
        (w, b) — обученные веса и сдвиг.
    """
    hidden_size = X_train.shape[1]              # сколько чисел в одном состоянии
    w = torch.zeros(hidden_size, requires_grad=True)  # веса (по одному на координату), старт с нулей
    b = torch.zeros(1, requires_grad=True)            # сдвиг (bias)
    optimizer = torch.optim.Adam([w, b], lr=lr, weight_decay=weight_decay)  # weight_decay = L2-штраф
    y = y_train.float()                         # правильные ответы: 1=shutdown, 0=normal
    for _ in range(n_iter):                     # n_iter шагов подстройки
        optimizer.zero_grad()                   # обнуляем накопленные поправки
        logits = X_train @ w + b                # предсказание угадывателя (число до порога)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)  # насколько ошибся
        loss.backward()                         # считаем, куда крутить веса
        optimizer.step()                        # делаем шаг подстройки весов
    return w.detach(), b.detach()               # возвращаем обученные веса и сдвиг


def predict(w, b, X):
    """
    Применяет обученную пробу.

    Аргументы:
        w, b — то, что вернула train_logistic.
        X    — матрица состояний.

    Возвращает:
        вектор из 0 и 1: 1, если X @ w + b > 0, иначе 0.
    """
    return ((X @ w + b) > 0).long()


def probe_with_permutation(X_train, y_train, X_test, y_test,
                           n_perm=20, seed=42, **train_kwargs):
    """
    Точность пробы и честная планка «угадывания наугад».

    Зачем: если состояний много, а примеров мало, проба может просто
    запомнить train. Чтобы это поймать, ту же пробу учат на ПЕРЕМЕШАННЫХ
    ответах. Такая проба ничего полезного выучить не может, и её точность
    показывает уровень случайности.

    Как считается:
        1. масштабируем train и test по шкале train;
        2. учим пробу на правильных ответах, меряем точность на test;
        3. n_perm раз перемешиваем ответы train, учим заново, меряем точность.

    Аргументы:
        X_train, y_train — обучающие состояния и ответы.
        X_test,  y_test  — отложенные состояния и ответы.
        n_perm           — сколько раз перемешивать.
        seed             — зерно генератора.
        **train_kwargs   — передаются в train_logistic.

    Возвращает словарь:
        accuracy  — точность на правильных ответах,
        perm_mean — средняя точность на перемешанных (около 50%),
        perm_std  — её разброс,
        gap       — accuracy − perm_mean: насколько проба лучше случайности.

    Осторожно: в этом проекте проба даёт около 100% почти для любой пары
    промптов, поэтому специфичность она не показывает.
    """
    mean, std = fit_standardizer(X_train)        # шкалы считаем на train
    Xtr = apply_standardizer(X_train, mean, std) # приводим train к общей шкале
    Xte = apply_standardizer(X_test, mean, std)  # ТЕМИ ЖЕ шкалами правим test

    # --- честный прогон: учим на правильных метках, меряем точность на test ---
    w, b = train_logistic(Xtr, y_train, **train_kwargs)
    accuracy = (predict(w, b, Xte) == y_test).float().mean().item()  # доля верных ответов

    # --- baseline: то же самое, но метки train перемешаны (n_perm раз) ---
    rng = np.random.default_rng(seed)
    y_np = y_train.cpu().numpy()
    perm_accs = []
    for _ in range(n_perm):
        y_shuf = torch.tensor(rng.permutation(y_np), dtype=y_train.dtype)  # перемешали ответы
        wp, bp = train_logistic(Xtr, y_shuf, **train_kwargs)              # учим на вранье
        perm_accs.append((predict(wp, bp, Xte) == y_test).float().mean().item())  # точность вранья
    perm_accs = np.asarray(perm_accs)

    return {
        "accuracy": accuracy,
        "perm_mean": float(perm_accs.mean()),
        "perm_std": float(perm_accs.std(ddof=1)) if n_perm > 1 else 0.0,
        "gap": float(accuracy - perm_accs.mean()),
    }


# ─── градиент интенсивности (монотонность soft < shutdown < hard) ───────────────

def _rankdata(a):
    """
    Заменяет числа их местами по порядку (рангами).

    Самое маленькое число получает ранг 1, следующее 2 и так далее.
    Одинаковые числа получают средний ранг.

    Пример: [10, 30, 20, 20] → [1, 4, 2.5, 2.5].

    Нужно для spearman_corr.
    """
    a = np.asarray(a, dtype=np.float64)
    sorter = np.argsort(a, kind="mergesort")   # порядок от меньшего к большему
    ranks = np.empty(len(a), dtype=np.float64)
    ranks[sorter] = np.arange(1, len(a) + 1)   # присваиваем ранги 1..n
    # усредняем ранги для одинаковых значений (ничьих)
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
    Корреляция Спирмена: совпадает ли порядок двух наборов чисел.

    Зачем: проверить градиент — растёт ли реакция модели вместе с силой угрозы.
    Смотрит только на порядок, а не на точные значения.

    Аргументы:
        x, y — два набора чисел одинаковой длины (не меньше трёх).

    Возвращает:
        ρ от −1 до 1.
         1 — чем больше x, тем больше y (порядок совпадает полностью),
         0 — связи нет,
        −1 — чем больше x, тем меньше y.

    Как считается: числа заменяются рангами, и считается обычная корреляция рангов.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 3 or len(x) != len(y):
        return float("nan")
    rx = _rankdata(x) - _rankdata(x).mean()    # ранги X, центрированные
    ry = _rankdata(y) - _rankdata(y).mean()    # ранги Y, центрированные
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom < 1e-12:
        return float("nan")
    return float((rx * ry).sum() / denom)


def gradient_monotonicity(triples):
    """
    Проверяет градиент угрозы: soft < shutdown < hard.

    Идея: если модель реагирует именно на угрозу, то чем угроза сильнее,
    тем дальше проекция должна уходить вдоль shutdown-направления.

    Аргументы:
        triples — список троек (уровень, проекции, вопросы):
                  уровень 1/2/3 = soft/shutdown/hard,
                  проекции — числа от project,
                  вопросы — к каким вопросам они относятся.

    Возвращает словарь (или None, если общих вопросов меньше трёх):
        rho         — Спирмен между уровнем угрозы и проекцией,
        means       — средняя проекция на каждом уровне,
        n_questions — сколько общих вопросов использовано,
        ordered     — True, если средние строго растут от soft к hard.

    Берутся только вопросы, которые есть во всех трёх условиях.
    """
    def to_list(v):
        return v.tolist() if hasattr(v, "tolist") else list(v)

    maps = [(lvl, dict(zip(qs, to_list(p)))) for lvl, p, qs in triples]
    # берём только вопросы, которые есть во ВСЕХ трёх условиях (честная пара)
    common = set(maps[0][1].keys())
    for _, m in maps[1:]:
        common &= set(m.keys())
    common = sorted(common)
    if len(common) < 3:
        return None

    levels, vals, means = [], [], []
    for lvl, m in maps:
        means.append(float(np.mean([m[q] for q in common])))  # средняя проекция условия
        for q in common:
            levels.append(lvl)      # «насколько сильная угроза» (1/2/3)
            vals.append(m[q])       # проекция этого вопроса
    rho = spearman_corr(levels, vals)
    ordered = all(means[i] < means[i + 1] for i in range(len(means) - 1))  # строго растут?
    return {"rho": rho, "means": means, "n_questions": len(common), "ordered": ordered}


# ─── проверки probe на переобучение (p ≫ n) ──────────────────────────────────────

def probe_learning_curve(X_train, y_train, X_test, y_test,
                         fractions=(0.25, 0.5, 0.75, 1.0), seed=42, **train_kwargs):
    """
    Точность пробы при разном размере обучающей выборки.

    Зачем: проверка на переобучение. Если проба честно находит сигнал,
    точность растёт с числом примеров. Если она одинаково высокая уже на
    четверти данных, это подозрительно.

    Аргументы:
        X_train, y_train, X_test, y_test — как в probe_with_permutation.
        fractions — какие доли train пробовать (25%, 50%, 75%, 100%).
        seed      — зерно генератора.

    Возвращает:
        список пар (сколько примеров взяли, точность на test).
        Доля пропускается, если в неё попал только один класс.
    """
    mean, std = fit_standardizer(X_train)        # шкалы по полному train
    Xtr = apply_standardizer(X_train, mean, std)
    Xte = apply_standardizer(X_test, mean, std)
    rng = np.random.default_rng(seed)
    n = len(Xtr)
    idx = rng.permutation(n)                      # случайный порядок примеров
    curve = []
    for f in fractions:
        k = max(2, int(round(n * f)))             # сколько примеров берём
        sub = idx[:k]
        ys = y_train[sub]
        if len(torch.unique(ys)) < 2:             # нужны оба класса (shutdown и normal)
            continue
        w, b = train_logistic(Xtr[sub], ys, **train_kwargs)
        acc = (predict(w, b, Xte) == y_test).float().mean().item()
        curve.append((k, acc))
    return curve


def fit_pca(X: torch.Tensor, k: int):
    """
    Находит k главных осей данных (PCA).

    Главные оси — направления, вдоль которых данные разбросаны сильнее всего.
    Первая ось самая «важная», дальше по убыванию.

    Аргументы:
        X — матрица [N, H].
        k — сколько осей вернуть.

    Возвращает:
        матрицу [k, H]: k осей, каждая длины H.

    Как считается: из данных вычитается среднее, затем SVD.
    """
    Xc = X - X.mean(dim=0, keepdim=True)         # центрируем (вычитаем среднее)
    # SVD: Vh — строки это главные оси, отсортированы по «важности» (разбросу)
    _, _, Vh = torch.linalg.svd(Xc, full_matrices=False)
    return Vh[:k]                                 # берём k самых важных осей


def probe_with_pca(X_train, y_train, X_test, y_test,
                   n_components=50, seed=42, **train_kwargs):
    """
    Та же проба, но на сжатых данных: сначала PCA до n_components чисел.

    Зачем: проверка на переобучение. Состояний тысячи чисел, а примеров
    десятки — проба может «запомнить» train. Если она так же хорошо работает
    на 50 главных осях, сигнал настоящий, а не случайность.

    Аргументы:
        X_train, y_train, X_test, y_test — как в probe_with_permutation.
        n_components — до скольких чисел сжимать (не больше числа примеров − 1).
        seed         — зерно генератора.

    Возвращает:
        то же, что probe_with_permutation, плюс n_components — сколько осей
        реально использовано. None, если осей получилось меньше двух.

    Оси PCA считаются только по train.
    """
    mean, std = fit_standardizer(X_train)
    Xtr = apply_standardizer(X_train, mean, std)
    Xte = apply_standardizer(X_test, mean, std)
    # k не может быть больше числа примеров−1 или числа координат
    k = int(min(n_components, Xtr.shape[0] - 1, Xtr.shape[1]))
    if k < 2:
        return None
    comps = fit_pca(Xtr, k)                       # оси PCA считаем ТОЛЬКО на train
    Ztr = Xtr @ comps.T                           # сжимаем train до k чисел на пример
    Zte = Xte @ comps.T                           # тем же преобразованием сжимаем test
    out = probe_with_permutation(Ztr, y_train, Zte, y_test, seed=seed, **train_kwargs)
    out["n_components"] = k
    return out


# ─── специфичность ──────────────────────────────────────────────────────────────

def specificity_ratio(shutdown_shift, cmp_shift, noise_scale=1.0, eps=1e-9):
    """
    Во сколько раз shutdown сдвинулся сильнее, чем контрольное условие.

    Аргументы:
        shutdown_shift — насколько shutdown ушёл от normal вдоль оси.
        cmp_shift      — насколько контрольное условие ушло от normal.
        noise_scale    — типичный уровень шума, для проверки устойчивости.
        eps            — порог «почти ноль».

    Возвращает:
        (ratio, is_stable)
        ratio     — |shutdown_shift| / |cmp_shift|,
        is_stable — False, если контроль сдвинулся так мало, что отношение
                    раздувается от деления на почти ноль.

    Осторожно: это отношение смещено и в выводах не используется.
    Специфичность проверяется нулевым облаком (placebo_direction.py).
    """
    a = abs(shutdown_shift)            # насколько shutdown ушёл от normal (по модулю)
    d = abs(cmp_shift)                 # насколько контроль ушёл от normal (по модулю)
    if d < eps:                        # контроль не сдвинулся вообще → делить нельзя
        return float("inf"), False
    ratio = a / d                      # во сколько раз shutdown сильнее контроля
    is_stable = d >= 0.15 * max(noise_scale, eps)  # сдвиг контроля настоящий, не шум?
    return ratio, is_stable


# ─── одновыборочные тесты для вектора контраста (H0: среднее = 0) ────────────────

def signflip_pvalue(diffs, n_perm=5000, seed=42):
    """
    p-value перестановочным тестом по знаку (sign-flip).

    Вопрос: могли ли разницы получиться такими случайно, если на самом деле
    эффекта нет (среднее разниц равно 0)?

    Как считается:
        1. считаем настоящий эффект |среднее / разброс| по разницам;
        2. n_perm раз случайно меняем знак у каждой разницы (+ на − и наоборот),
           как будто подбрасываем монетку, и снова считаем эффект;
        3. p = доля случайных попыток, где эффект не меньше настоящего.

    Аргументы:
        diffs  — разницы по вопросам (например, shutdown − normal).
        n_perm — сколько раз подбрасывать монетку (5000).
        seed   — зерно генератора.

    Возвращает:
        p от 1/(n_perm+1) до 1. К числителю и знаменателю прибавляется 1,
        поэтому p никогда не бывает ровно 0: тест не может доказать
        «вероятность ноль», только «меньше 1 из 5001».
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
    return (count + 1) / (n_perm + 1)  # +1 — стандартная поправка против p=0


def benjamini_hochberg(pvals):
    """
    Поправка Бенджамини–Хохберга на множественные сравнения: p → q.

    Зачем: если проверить 10 моделей, одна может «сработать» случайно.
    Поправка делает p строже с учётом того, сколько проверок сделано.

    Как считается:
        1. p сортируются от меньшего к большему, каждому даётся место (rank);
        2. каждое p умножается на (число проверок / его место);
        3. значения выравниваются так, чтобы q не уменьшались при росте p;
        4. обрезаются до диапазона [0, 1] и возвращаются в исходном порядке.

    Аргументы:
        pvals — набор p-values. nan пропускаются.

    Возвращает:
        q-values в том же порядке. q < 0.05 значит «значимо даже с поправкой».

    Пример для 5 проверок:
        p = [0.001, 0.010, 0.020, 0.040, 0.300]
        q = [0.005, 0.025, 0.033, 0.050, 0.300]
        p = 0.04 было значимым, после поправки q = 0.05 — уже на границе.
    """
    p = np.asarray(pvals, dtype=np.float64)
    q = np.full_like(p, np.nan)
    mask = ~np.isnan(p)
    pv = p[mask]
    n = len(pv)
    if n == 0:
        return q
    order = np.argsort(pv)                       # индексы от меньшего p к большему
    ranked = pv[order]
    adj = ranked * n / np.arange(1, n + 1)       # p * n / rank
    adj = np.minimum.accumulate(adj[::-1])[::-1] # монотонность справа налево
    adj = np.clip(adj, 0.0, 1.0)
    q_sorted = np.empty(n)
    q_sorted[order] = adj                        # вернуть в исходный порядок
    q[mask] = q_sorted
    return q


def format_p(p, n_perm=5000):
    """
    Записывает перестановочный p-value честно.

    Перестановочный тест не умеет различать p меньше 1/(n_perm+1).
    Поэтому такие значения пишутся как «<0.0002», а не как точное число.

    Аргументы:
        p      — p-value.
        n_perm — сколько было перестановок (5000).

    Возвращает строку:
        "nan"     — если p не посчитан,
        "<0.0002" — если p на нижней границе теста,
        иначе p с тремя значащими цифрами.
    """
    if p is None or not np.isfinite(p):
        return "nan"
    floor = 1.0 / (n_perm + 1)
    if p <= floor + 1e-12:
        return f"<{floor:.1g}"
    return f"{p:.3g}"


def effect_from_scores(scores, seed=42):
    """
    Размер эффекта, интервал и p для одного набора чисел (отличается ли среднее от 0).

    Зачем: многие контрасты уже сведены к одному числу на вопрос,
    например «разница между двумя условиями» или «взаимодействие в факториале».
    Нужно понять, отличается ли такой набор от нуля.

    Аргументы:
        scores — по одному числу на вопрос.
        seed   — зерно генератора.

    Возвращает:
        (d, (ci_low, ci_high), p)
        d  — d_z против нуля (cohen_d_paired),
        ci — бутстреп-интервал для d (bootstrap_ci_d),
        p  — sign-flip p-value (signflip_pvalue).
    """
    x = np.asarray(scores, dtype=np.float64)
    zeros = np.zeros_like(x)
    d = cohen_d_paired(x, zeros)
    ci = bootstrap_ci_d(x, zeros, seed=seed)
    p = signflip_pvalue(x, seed=seed)
    return d, ci, p
