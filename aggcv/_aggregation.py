"""Правила агрегування матриці оцінок (конфігурація x розбиття).

Усі скорери scikit-learn мають конвенцію "більше = краще"
(напр. ``neg_root_mean_squared_error``), тому скрізь використовується argmax.

Правила відтворюють код досліджень (``run_experiments.py``) разом із
тай-брейками; відповідність перевіряє ``tests/test_aggcv_research_regression.py``.
"""

import numpy as np
from scipy.stats import rankdata

STRATEGIES = ("single", "mean", "median", "rank", "max", "vote")
# правила, що не потребують спільних розбиттів (кожна конфігурація -- на власних)
RESHUFFLED_STRATEGIES = ("single", "mean", "median", "max")


def split_score_matrix(cv_results, metric="score"):
    """Матриця оцінок (конфігурація x розбиття) з ``cv_results_``.

    Parameters
    ----------
    cv_results : dict
        Атрибут ``cv_results_`` навченого ``GridSearchCV`` /
        ``RandomizedSearchCV`` (або словник, який отримує callable ``refit``).
    metric : str, default="score"
        ``"score"`` для однієї метрики або ім'я скорера за multimetric scoring.

    Returns
    -------
    scores : ndarray of shape (n_configs, n_splits)
        Оцінки ``split{k}_test_{metric}``; більше = краще. Невдалі навчання
        (NaN) замінено на ``-inf``, щоб така конфігурація ніколи не перемагала.

    Raises
    ------
    KeyError
        Якщо в ``cv_results`` немає ключів ``split*_test_{metric}``.
    """
    n_splits = 0
    while f"split{n_splits}_test_{metric}" in cv_results:
        n_splits += 1
    if n_splits == 0:
        raise KeyError(f"У cv_results_ немає split*_test_{metric}")
    scores = np.column_stack(
        [np.asarray(cv_results[f"split{k}_test_{metric}"], dtype=float)
         for k in range(n_splits)]
    )
    return np.where(np.isnan(scores), -np.inf, scores)


def aggregate(scores, strategy):
    """Обирає конфігурацію за матрицею оцінок заданим правилом.

    Parameters
    ----------
    scores : array-like of shape (n_configs, n_splits)
        Оцінки конфігурацій на розбиттях; більше = краще. NaN трактується
        як ``-inf``.
    strategy : {"single", "mean", "median", "rank", "max", "vote"}
        Правило вибору:

        - ``"single"`` -- argmax першого стовпця (одне розбиття);
        - ``"mean"`` / ``"median"`` -- argmax середнього / медіани за рядком;
        - ``"max"`` -- конфігурація з глобальним максимумом матриці;
        - ``"vote"`` -- найбільше перемог на окремих розбиттях;
        - ``"rank"`` -- найменший середній ранг (ранги в межах розбиття,
          1 = найкраща, однакові оцінки отримують середній ранг).

    Returns
    -------
    best_index : int
        Індекс обраної конфігурації (рядок матриці).
    aggregated : ndarray of shape (n_configs,)
        Агреговані оцінки, більше = краще: значення першого розбиття
        (single), середнє (mean), медіана (median), максимум за рядком (max),
        кількість голосів (vote), мінус середній ранг (rank). Для
        конфігурацій з невдалим навчанням -- ``-inf``.

    Notes
    -----
    Нічиї розв'язуються так само, як у коді досліджень:

    - single, mean, median -- перша конфігурація в порядку рядків;
    - max -- конфігурація-переможець найранішого розбиття, на якому досягнуто
      глобальний максимум (усередині розбиття -- перша в порядку рядків);
    - vote -- конфігурація, що першою стала переможцем у порядку розбиттів;
    - rank -- менша дисперсія рангів (``ddof=0``), далі краще середнє
      значення метрики, далі перша в порядку рядків.

    Конфігурація, що має хоча б одну невдалу оцінку (NaN / ``-inf``),
    вилучається з вибору за всіма правилами (як і в стандартному
    ``GridSearchCV``, де її середнє дорівнює NaN).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy має бути одним із {STRATEGIES}")
    scores = np.array(scores, dtype=float)
    if scores.ndim != 2 or 0 in scores.shape:
        raise ValueError("scores має бути непорожньою матрицею (n_configs, n_splits)")
    n_configs, n_splits = scores.shape

    failed = np.isnan(scores).any(axis=1) | np.isneginf(scores).any(axis=1)
    scores[failed] = -np.inf

    if strategy == "single":
        agg = scores[:, 0].copy()
        return int(np.argmax(agg)), agg

    if strategy == "mean":
        agg = scores.mean(axis=1)
        return int(np.argmax(agg)), agg

    if strategy == "median":
        agg = np.median(scores, axis=1)
        return int(np.argmax(agg)), agg

    if strategy == "max":
        # локальний переможець кожного розбиття, далі найкращий із них;
        # за рівних значень -- переможець найранішого розбиття
        winners = scores.argmax(axis=0)
        local_best = scores[winners, np.arange(n_splits)]
        return int(winners[np.argmax(local_best)]), scores.max(axis=1)

    if strategy == "vote":
        winners = scores.argmax(axis=0)            # переможець кожного розбиття
        votes = np.bincount(winners, minlength=n_configs).astype(float)
        top = np.flatnonzero(votes == votes.max())
        # нічия: конфігурація, що першою стала переможцем у порядку розбиттів
        best = min(top, key=lambda c: int(np.flatnonzero(winners == c)[0]))
        votes[failed] = -np.inf
        return int(best), votes

    # rank: розкладка [розбиття, конфігурація] і зведення за axis=0 -- як у
    # коді досліджень, щоб порядок підсумовування (і тай-брейки) збігалися
    by_split = np.ascontiguousarray(scores.T)
    ranks = np.empty_like(by_split)
    for k in range(n_splits):
        # ранг 1 = найкраща конфігурація в межах розбиття
        ranks[k] = rankdata(-by_split[k], method="average")
    mean_rank = ranks.mean(axis=0)
    var_rank = ranks.var(axis=0)
    mean_score = by_split.mean(axis=0)
    best = min(
        range(n_configs),
        key=lambda i: (mean_rank[i], var_rank[i], -mean_score[i]),
    )
    agg = -mean_rank
    agg[failed] = -np.inf
    return int(best), agg
