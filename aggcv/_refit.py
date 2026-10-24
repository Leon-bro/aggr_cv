"""Callable ``refit`` для стандартних GridSearchCV / RandomizedSearchCV."""

from ._aggregation import STRATEGIES, aggregate, split_score_matrix


class _AggregatedRefit:
    """Callable ``refit(cv_results) -> best_index``.

    Клас, а не замикання, щоб навчений пошук можна було серіалізувати pickle.
    """

    def __init__(self, strategy, metric):
        self.strategy = strategy
        self.metric = metric
        self.__name__ = f"refit_{strategy}_{metric}"

    def __call__(self, cv_results):
        best, _ = aggregate(split_score_matrix(cv_results, self.metric), self.strategy)
        return best

    def __repr__(self):
        return f"make_refit(strategy={self.strategy!r}, metric={self.metric!r})"


def make_refit(strategy="mean", metric="score"):
    """Callable для ``GridSearchCV(refit=...)`` / ``RandomizedSearchCV(refit=...)``.

    Використовує лише публічний API scikit-learn: ``refit`` офіційно приймає
    callable, який отримує ``cv_results_`` і повертає ``best_index_``.

    Parameters
    ----------
    strategy : {"single", "mean", "median", "rank", "max", "vote"}, default="mean"
        Правило вибору конфігурації, див. :func:`aggcv.aggregate`.
    metric : str, default="score"
        ``"score"`` для однієї метрики або ім'я скорера за multimetric scoring.

    Returns
    -------
    refit : callable
        Функція ``refit(cv_results) -> int``.

    Notes
    -----
    За callable ``refit`` scikit-learn не заповнює ``best_score_``. Якщо
    потрібна агрегована оцінка обраної конфігурації, використовуйте
    :class:`aggcv.AggregatedGridSearchCV`.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.model_selection import GridSearchCV
    >>> from aggcv import make_refit
    >>> X, y = make_regression(n_samples=60, n_features=5, random_state=0)
    >>> search = GridSearchCV(Lasso(), {"alpha": [0.1, 1.0]}, cv=5,
    ...                       refit=make_refit("rank")).fit(X, y)
    >>> search.best_params_
    {'alpha': 0.1}
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy має бути одним із {STRATEGIES}")
    return _AggregatedRefit(strategy, metric)
