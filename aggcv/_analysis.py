"""Аналітичні утиліти: порівняння правил і стабільність вибору.

Обидві функції працюють із ``cv_results_`` уже навченого пошуку і не
перенавчають внутрішні моделі.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone

from ._aggregation import (RESHUFFLED_STRATEGIES, STRATEGIES, aggregate,
                           split_score_matrix)


def _metric_name(search, metric):
    if metric is not None:
        return metric
    if getattr(search, "refit_metric", None) is not None:
        return search.refit_metric
    if isinstance(search.refit, str):            # multimetric: refit="rmse"
        return search.refit
    return "score"


def compare_strategies(search, metric=None):
    """Вибір кожного з шести правил для вже навченого пошуку.

    Parameters
    ----------
    search : навчений GridSearchCV / RandomizedSearchCV або їхній підклас
        Використовується лише ``cv_results_``; моделі не перенавчаються.
    metric : str або None, default=None
        Ім'я скорера, якщо метрик кілька. Якщо None -- ``search.refit_metric``
        (якщо є), далі рядок ``search.refit``, інакше ``"score"``.

    Returns
    -------
    comparison : pandas.DataFrame
        По одному рядку на правило, у порядку ``aggcv.STRATEGIES``. Колонки:

        - ``strategy`` -- назва правила;
        - ``best_index`` -- індекс обраної конфігурації в ``cv_results_``;
        - ``params`` -- її параметри (dict);
        - ``aggregated_score`` -- агрегована оцінка обраної конфігурації
          (для ``"rank"`` -- мінус середній ранг, для ``"vote"`` -- кількість
          голосів; між правилами не порівнюється).

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.model_selection import GridSearchCV
    >>> from aggcv import compare_strategies
    >>> X, y = make_regression(n_samples=60, n_features=5, random_state=0)
    >>> search = GridSearchCV(Lasso(), {"alpha": [0.1, 1.0]}, cv=5).fit(X, y)
    >>> compare_strategies(search)[["strategy", "best_index"]].shape
    (6, 2)
    """
    scores = split_score_matrix(search.cv_results_, _metric_name(search, metric))
    rows = []
    for strategy in STRATEGIES:
        best, agg = aggregate(scores, strategy)
        rows.append({
            "strategy": strategy,
            "best_index": best,
            "params": search.cv_results_["params"][best],
            "aggregated_score": float(agg[best]),
        })
    return pd.DataFrame(rows, columns=["strategy", "best_index", "params", "aggregated_score"])


def _row_permutations(n_configs, n_splits, rng):
    """Незалежна випадкова перестановка індексів стовпців для кожного рядка."""
    idx = np.tile(np.arange(n_splits), (n_configs, 1))
    return rng.permuted(idx, axis=1)


def reshuffle_scores(scores, n_splits=None, random_state=None):
    """Імітує перемішані розбиття за матрицею оцінок зі спільного (парного) пулу.

    Кожна конфігурація (рядок) отримує *власну* випадкову підмножину з
    ``n_splits`` стовпців матриці без повторень, у випадковому порядку. До
    результату застосовні правила ``aggcv.RESHUFFLED_STRATEGIES``
    (single, mean, median, max) через :func:`aggcv.aggregate`.

    Parameters
    ----------
    scores : array-like of shape (n_configs, n_pool)
        Оцінки конфігурацій на спільному пулі з ``n_pool`` розбиттів.
    n_splits : int або None, default=None
        Кількість розбиттів N на конфігурацію; None -- усі ``n_pool``.
    random_state : int, numpy.random.Generator або None, default=None
        Зерно (``numpy.random.default_rng``).

    Returns
    -------
    reshuffled : ndarray of shape (n_configs, n_splits)
        Стовпець ``k`` -- k-те власне розбиття кожної конфігурації; значення
        в одному стовпці походять із різних розбиттів пулу.

    Notes
    -----
    Це наближення до справжнього перемішування (:class:`aggcv.ReshuffledSearchCV`),
    де розбиття конфігурацій незалежні. Тут усі підмножини беруться з одного
    скінченного пулу з M розбиттів, тому дві конфігурації можуть випадково
    отримати спільне розбиття: очікувана кількість спільних розбиттів для
    пари конфігурацій дорівнює N²/M (напр. 1 за N = 10, M = 100), а за N = M
    усі конфігурації використовують увесь пул. Наближення тим краще, чим
    менше N²/M.

    .. warning::
       Для правила ``"max"`` імітація некоректна, якщо
       ``n_configs * n_splits >= n_pool``: разом конфігурації покривають
       увесь пул, тому ``max`` майже завжди знаходить найлегше розбиття пулу
       й обирає найкращу з конфігурацій, яким воно дісталося, тобто поводиться
       як парний ``max`` на всьому пулі. За справжнього перемішування кожне
       розбиття дістається лише одній конфігурації. Через це імітація завищує
       стабільність ``max`` (на даних досліджень, elasticnet, N = 10: SD
       зовнішнього RMSE 0.0019 проти 0.0093 у парному режимі).

    Examples
    --------
    >>> import numpy as np
    >>> from aggcv import aggregate, reshuffle_scores
    >>> pool = np.arange(12.0).reshape(3, 4)
    >>> sub = reshuffle_scores(pool, n_splits=2, random_state=0)
    >>> sub.shape
    (3, 2)
    >>> aggregate(sub, "mean")[0]
    2
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or 0 in scores.shape:
        raise ValueError("scores має бути непорожньою матрицею (n_configs, n_pool)")
    n_configs, n_pool = scores.shape
    if n_splits is None:
        n_splits = n_pool
    if not 1 <= n_splits <= n_pool:
        raise ValueError(f"n_splits має бути в межах [1, {n_pool}]")
    cols = _row_permutations(n_configs, n_pool, np.random.default_rng(random_state))
    return np.take_along_axis(scores, cols[:, :n_splits], axis=1)


def selection_stability(search, n_permutations=50, n_values=None, random_state=2026,
                        *, metric=None, reshuffle=False,
                        X=None, y=None, X_test=None, y_test=None):
    """Стабільність вибору конфігурації залежно від кількості розбиттів N.

    Стовпці матриці оцінок (розбиття) випадково переставляються; для кожної
    перестановки і кожного N кожне правило обирає конфігурацію за першими N
    стовпцями. Вибір порівнюється з вибором того самого правила на повному
    пулі розбиттів у початковому порядку.

    Parameters
    ----------
    search : навчений пошук або array-like of shape (n_configs, n_splits)
        Навчений GridSearchCV / RandomizedSearchCV чи їхній підклас
        (використовується ``cv_results_``; внутрішні моделі не
        перенавчаються) або готова матриця оцінок, більше = краще (напр.
        зі збережених результатів досліджень).
    n_permutations : int, default=50
        Кількість випадкових перестановок розбиттів.
    n_values : iterable of int або None, default=None
        Значення N. Якщо None -- ``1, ..., min(20, n_splits)``. Значення поза
        ``[1, n_splits]`` спричиняють ``ValueError``.
    random_state : int, default=2026
        Зерно генератора перестановок (``numpy.random.default_rng``).
    metric : str або None, default=None
        Ім'я скорера, якщо метрик кілька, як у :func:`compare_strategies`.
    reshuffle : bool, default=False
        Імітація перемішаних розбиттів. Якщо True, у кожній перестановці
        кожна конфігурація отримує *власну* незалежну перестановку стовпців
        (як у :func:`reshuffle_scores`) і оцінюється за першими N своїми
        стовпцями; розглядаються лише правила single, mean, median, max.
        Еталон (``full_index``) -- той самий вибір на повному парному пулі.
        Обмеження наближення див. у :func:`reshuffle_scores`: пара
        конфігурацій має в середньому N²/M спільних розбиттів.
    X, y : array-like, default=None
        Навчальні дані, на яких навчався ``search``. Потрібні лише разом із
        ``X_test``, ``y_test``.
    X_test, y_test : array-like, default=None
        Зовнішні тестові дані (опційно). Якщо їх задано, кожна *різна* обрана
        конфігурація один раз перенавчається на ``X``, ``y`` і оцінюється
        скорером пошуку на ``X_test``, ``y_test``; внутрішні моделі не
        перенавчаються.

    Returns
    -------
    stability : pandas.DataFrame
        По одному рядку на пару (правило, N). Колонки:

        - ``strategy``, ``n_splits`` -- правило і N;
        - ``full_index`` -- вибір правила на повному пулі;
        - ``agreement`` -- частка перестановок, у яких вибір за N розбиттями
          збігся з ``full_index``;
        - ``n_distinct`` -- кількість різних конфігурацій, обраних за всіма
          перестановками;
        - ``test_score_mean``, ``test_score_std`` (``ddof=1``; NaN за однієї
          перестановки), ``full_test_score`` -- лише якщо задано тестові дані:
          зовнішня оцінка обраних конфігурацій (більше = краще) за
          перестановками та оцінка вибору на повному пулі.

    Notes
    -----
    Правило ``"single"`` завжди використовує лише перший стовпець перестановки,
    тому його рядки не залежать від N. Для ``"max"`` і ``"vote"`` нічиї
    залежать від порядку розбиттів, тому навіть за ``N = n_splits`` збіг може
    бути меншим за 1. За ``reshuffle=True`` ``"single"`` -- це імітація
    перемішаного holdout (перше власне розбиття кожної конфігурації).
    """
    from_search = hasattr(search, "cv_results_")
    if from_search:
        metric = _metric_name(search, metric)
        scores = split_score_matrix(search.cv_results_, metric)
    else:
        scores = np.asarray(search, dtype=float)
        if scores.ndim != 2 or 0 in scores.shape:
            raise ValueError("Потрібен навчений пошук або матриця (n_configs, n_splits)")
    n_configs, n_splits = scores.shape
    strategies = RESHUFFLED_STRATEGIES if reshuffle else STRATEGIES
    if n_values is None:
        n_values = range(1, min(20, n_splits) + 1)
    n_values = [int(n) for n in n_values]
    if not n_values or min(n_values) < 1 or max(n_values) > n_splits:
        raise ValueError(f"n_values мають бути в межах [1, {n_splits}]")
    if n_permutations < 1:
        raise ValueError("n_permutations має бути >= 1")
    with_test = X_test is not None or y_test is not None
    if with_test and any(a is None for a in (X, y, X_test, y_test)):
        raise ValueError("Для зовнішньої оцінки потрібні X, y, X_test і y_test")
    if with_test and not from_search:
        raise ValueError("Зовнішня оцінка можлива лише для навченого пошуку")

    rng = np.random.default_rng(random_state)
    full = {s: aggregate(scores, s)[0] for s in strategies}

    # chosen[strategy][i_n, b] -- вибір за першими N стовпцями перестановки b
    chosen = {s: np.empty((len(n_values), n_permutations), dtype=int) for s in strategies}
    for b in range(n_permutations):
        if reshuffle:
            # власна перестановка стовпців для кожної конфігурації
            permuted = np.take_along_axis(
                scores, _row_permutations(n_configs, n_splits, rng), axis=1)
        else:
            permuted = scores[:, rng.permutation(n_splits)]
        for i, n in enumerate(n_values):
            sub = permuted[:, :n]
            for s in strategies:
                chosen[s][i, b] = aggregate(sub, s)[0]

    test_score = {}
    if with_test:
        scorer = search.scorer_[metric] if isinstance(search.scorer_, dict) else search.scorer_
        needed = set(full.values()).union(*(np.unique(c).tolist() for c in chosen.values()))
        for idx in sorted(needed):
            params = search.cv_results_["params"][idx]
            model = clone(search.estimator).set_params(**clone(params, safe=False))
            test_score[idx] = float(scorer(model.fit(X, y), X_test, y_test))

    rows = []
    for s in strategies:
        for i, n in enumerate(n_values):
            picks = chosen[s][i]
            row = {
                "strategy": s,
                "n_splits": n,
                "full_index": full[s],
                "agreement": float(np.mean(picks == full[s])),
                "n_distinct": int(len(np.unique(picks))),
            }
            if with_test:
                ts = np.array([test_score[j] for j in picks])
                row["test_score_mean"] = float(ts.mean())
                row["test_score_std"] = float(ts.std(ddof=1)) if len(ts) > 1 else np.nan
                row["full_test_score"] = test_score[full[s]]
            rows.append(row)
    return pd.DataFrame(rows)
