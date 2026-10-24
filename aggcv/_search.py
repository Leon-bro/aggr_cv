"""Підкласи GridSearchCV / RandomizedSearchCV з правилом агрегування."""

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from ._aggregation import STRATEGIES, aggregate, split_score_matrix

_FITTED_ATTRS = ("split_scores_", "aggregated_scores_")


class _AggregationMixin:
    """Спільна логіка вибору конфігурації для обох підкласів."""

    def _metric_name(self):
        if self.refit_metric is not None:
            return self.refit_metric
        if isinstance(self.refit, str):          # multimetric: refit="rmse"
            return self.refit
        return "score"

    def _aggregate_results(self, results):
        scores = split_score_matrix(results, self._metric_name())
        best, agg = aggregate(scores, self.aggregation)
        self.split_scores_ = scores
        self.aggregated_scores_ = agg
        return best

    # BaseSearchCV.fit викликає self._select_best_index(self.refit, refit_metric,
    # results) рівно один раз, після чого (якщо refit не callable) записує
    # best_score_ = results["mean_test_<metric>"][best_index_].
    # Це перевизначення ПРИВАТНОГО методу. Перевірено на scikit-learn 1.8.0
    # (Python 3.14.3, numpy 2.4.4, scipy 1.17.1). Зміну сигнатури або поведінки
    # в іншій версії ловить tests/test_aggcv_sklearn_private.py.
    def _select_best_index(self, refit, refit_metric, results):
        if callable(refit):
            return super()._select_best_index(refit, refit_metric, results)
        return self._aggregate_results(results)

    def fit(self, X, y=None, **params):
        """Запускає пошук і обирає конфігурацію правилом ``aggregation``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Навчальні дані.
        y : array-like of shape (n_samples,) або (n_samples, n_outputs), default=None
            Цільова змінна.
        **params : dict
            Параметри для ``fit`` оцінювача, скорера і розбивача CV
            (напр. ``groups``), як у ``GridSearchCV.fit``.

        Returns
        -------
        self : object
            Навчений пошук.
        """
        if self.aggregation not in STRATEGIES:
            raise ValueError(f"aggregation має бути одним із {STRATEGIES}")
        for attr in _FITTED_ATTRS:
            if hasattr(self, attr):
                delattr(self, attr)
        super().fit(X, y, **params)
        if callable(self.refit):
            return self
        if not hasattr(self, "aggregated_scores_"):
            # multimetric з refit=False: sklearn не обирає конфігурацію;
            # обираємо самі, якщо явно задано refit_metric
            if self.refit_metric is None:
                return self
            self.best_index_ = self._aggregate_results(self.cv_results_)
            self.best_params_ = self.cv_results_["params"][self.best_index_]
        self.best_score_ = float(self.aggregated_scores_[self.best_index_])
        return self


_AGGREGATION_DOC = """
    Працює як батьківський клас scikit-learn, але конфігурацію обирає за
    матрицею оцінок (конфігурація x розбиття) правилом ``aggregation``.
    Решта параметрів і атрибутів -- як у батьківського класу.

    Parameters
    ----------
    aggregation : {{"single", "mean", "median", "rank", "max", "vote"}}, default="mean"
        Правило вибору конфігурації, див. :func:`aggcv.aggregate`.
        ``"mean"`` збігається зі стандартною поведінкою scikit-learn.
        Ігнорується, якщо ``refit`` -- callable (тоді вибір робить він).
    refit_metric : str або None, default=None
        Ім'я скорера, за оцінками якого обирається конфігурація за multimetric
        scoring. Якщо None і ``refit`` -- рядок, використовується він;
        інакше ``"score"``. За multimetric scoring і ``refit=False``
        конфігурація обирається лише тоді, коли ``refit_metric`` задано явно;
        інакше, як у scikit-learn, ``best_index_`` / ``best_params_`` /
        ``best_score_`` не створюються.

    Attributes
    ----------
    split_scores_ : ndarray of shape (n_configs, n_splits)
        Оцінки конфігурацій на розбиттях (NaN замінено на ``-inf``).
    aggregated_scores_ : ndarray of shape (n_configs,)
        Агреговані оцінки конфігурацій, більше = краще.
    best_index_ : int
        Індекс обраної конфігурації в ``cv_results_``.
    best_params_ : dict
        Параметри обраної конфігурації.
    best_score_ : float
        Агрегована оцінка обраної конфігурації, тобто
        ``aggregated_scores_[best_index_]``. УВАГА: це не завжди значення
        метрики. Для ``"rank"`` це МІНУС СЕРЕДНІЙ РАНГ (напр. ``-1.4``), для
        ``"vote"`` -- КІЛЬКІСТЬ ГОЛОСІВ (напр. ``6.0``), для ``"single"`` --
        оцінка на першому розбитті, для ``"max"`` -- найкраща оцінка на
        одному розбитті, для ``"mean"`` / ``"median"`` -- середнє / медіана.
        Середнє значення метрики обраної конфігурації завжди доступне як
        ``cv_results_["mean_test_<metric>"][best_index_]``.
    best_estimator_ : estimator
        Обрана конфігурація, перенавчена на всіх даних (якщо ``refit``).

    Notes
    -----
    ``split_scores_``, ``aggregated_scores_`` і ``best_score_`` не
    заповнюються, якщо ``refit`` -- callable. ``rank_test_<metric>`` у
    ``cv_results_`` завжди відповідає стандартному рангу за середнім і може
    не збігатися з ``best_index_``.
"""


class AggregatedGridSearchCV(_AggregationMixin, GridSearchCV):
    __doc__ = (
        "GridSearchCV з вибором правила агрегування результатів розбиттів.\n"
        + _AGGREGATION_DOC.format()
        + """
    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Lasso
    >>> from aggcv import AggregatedGridSearchCV
    >>> X, y = make_regression(n_samples=60, n_features=5, random_state=0)
    >>> search = AggregatedGridSearchCV(Lasso(), {"alpha": [0.1, 1.0]}, cv=5,
    ...                                 aggregation="vote").fit(X, y)
    >>> search.best_params_, search.best_score_
    ({'alpha': 0.1}, 5.0)
    """
    )

    _parameter_constraints = {
        **GridSearchCV._parameter_constraints,
        "aggregation": [str],
        "refit_metric": [str, None],
    }

    def __init__(self, estimator, param_grid, *, aggregation="mean",
                 refit_metric=None, scoring=None, n_jobs=None, refit=True,
                 cv=None, verbose=0, pre_dispatch="2*n_jobs",
                 error_score=np.nan, return_train_score=False):
        super().__init__(estimator, param_grid, scoring=scoring, n_jobs=n_jobs,
                         refit=refit, cv=cv, verbose=verbose,
                         pre_dispatch=pre_dispatch, error_score=error_score,
                         return_train_score=return_train_score)
        self.aggregation = aggregation
        self.refit_metric = refit_metric


class AggregatedRandomizedSearchCV(_AggregationMixin, RandomizedSearchCV):
    __doc__ = (
        "RandomizedSearchCV з вибором правила агрегування результатів розбиттів.\n"
        + _AGGREGATION_DOC.format()
    )

    _parameter_constraints = {
        **RandomizedSearchCV._parameter_constraints,
        "aggregation": [str],
        "refit_metric": [str, None],
    }

    def __init__(self, estimator, param_distributions, *, aggregation="mean",
                 refit_metric=None, n_iter=10, scoring=None, n_jobs=None,
                 refit=True, cv=None, verbose=0, pre_dispatch="2*n_jobs",
                 random_state=None, error_score=np.nan,
                 return_train_score=False):
        super().__init__(estimator, param_distributions, n_iter=n_iter,
                         scoring=scoring, n_jobs=n_jobs, refit=refit, cv=cv,
                         verbose=verbose, pre_dispatch=pre_dispatch,
                         random_state=random_state, error_score=error_score,
                         return_train_score=return_train_score)
        self.aggregation = aggregation
        self.refit_metric = refit_metric
