"""Пошук із перемішаними розбиттями (Nagler et al., NeurIPS 2024).

Власний цикл оцінки, без ``BaseSearchCV`` / ``evaluate_candidates``. З
публічного API scikit-learn: ``clone``, ``check_scoring`` / ``get_scorer``,
``indexable``, ``ParameterGrid``, metadata routing (``MetadataRouter``,
``MethodMapping``, ``process_routing``), ``sklearn.utils.parallel``
(обгортки joblib, що передають конфігурацію sklearn у процеси-виконавці).

Приватні помічники scikit-learn (усі дрібні, сторож --
``tests/test_aggcv_sklearn_private.py``):

- ``BaseEstimator._validate_params`` -- перевірка ``_parameter_constraints``;
- ``sklearn.utils.metaestimators._safe_split`` -- розрізання ``X``, ``y`` за
  індексами (з урахуванням попередньо обчислених ядер);
- ``sklearn.utils.validation._check_method_params`` -- розрізання
  fit- / score-параметрів довжини ``n_samples`` за індексами.
"""

import numbers
import time
import traceback
import warnings
from collections import Counter, defaultdict
from copy import deepcopy

import numpy as np
from scipy.stats import rankdata
from sklearn import get_config
from sklearn.base import BaseEstimator, MetaEstimatorMixin, clone
from sklearn.exceptions import FitFailedWarning
from sklearn.metrics import check_scoring, get_scorer
from sklearn.model_selection import ParameterGrid
from sklearn.utils import Bunch, check_random_state, get_tags, indexable
from sklearn.utils.metadata_routing import (MetadataRouter, MethodMapping,
                                            process_routing)
from sklearn.utils.metaestimators import _safe_split, available_if
from sklearn.utils.parallel import Parallel, delayed
from sklearn.utils.validation import _check_method_params, check_is_fitted

from ._aggregation import RESHUFFLED_STRATEGIES, STRATEGIES, aggregate


def _routing_enabled():
    return get_config().get("enable_metadata_routing", False)


def _check_refit(search, attr):
    if not search.refit:
        raise AttributeError(
            f"Атрибут {attr!r} доступний лише після перенавчання найкращої "
            f"конфігурації, а {type(search).__name__} створено з refit=False."
        )


def _estimator_has(attr):
    """Чи можна делегувати ``attr`` до ``best_estimator_`` (як у BaseSearchCV)."""

    def check(self):
        _check_refit(self, attr)
        if hasattr(self, "best_estimator_"):
            getattr(self.best_estimator_, attr)
        else:
            getattr(self.estimator, attr)
        return True

    return check


def _score(estimator, X, y, scorer, score_params, metrics, multimetric, error_score):
    """Оцінки для однієї пари; повертає (dict метрика -> float, невдалі метрики)."""
    try:
        if y is None:
            scores = scorer(estimator, X, **score_params)
        else:
            scores = scorer(estimator, X, y, **score_params)
    except Exception:
        if error_score == "raise":
            raise
        warnings.warn(
            "Оцінювання завершилося помилкою; оцінку для цієї пари "
            f"(конфігурація, розбиття) встановлено в {error_score}. "
            f"Подробиці:\n{traceback.format_exc()}",
            UserWarning,
        )
        return {m: error_score for m in metrics}, set(metrics)

    if not multimetric:
        if not isinstance(scores, numbers.Number):
            raise ValueError(
                f"Скорер має повертати число, отримано {scores!r} "
                f"({type(scores).__name__}). Callable, що повертає dict, не "
                "підтримується: передайте scoring як dict скорерів."
            )
        scores = {"score": scores}
    failed = set()
    for name, value in scores.items():
        if isinstance(value, str):       # check_scoring(raise_exc=False): текст винятку
            warnings.warn(
                f"Оцінювання метрики {name!r} завершилося помилкою; оцінку "
                f"встановлено в {error_score}. Подробиці:\n{value}",
                UserWarning,
            )
            scores[name] = error_score
            failed.add(name)
    return {m: float(scores[m]) for m in metrics}, failed


def _fit_and_score_pair(estimator, X, y, *, params, train, test, fit_params,
                        score_params, scorer, metrics, multimetric,
                        return_train_score, error_score):
    """Навчання й оцінка однієї пари (конфігурація, розбиття)."""
    estimator = clone(estimator).set_params(**clone(params, safe=False))
    fit_params = _check_method_params(X, params=fit_params, indices=train)
    X_train, y_train = _safe_split(estimator, X, y, train)
    X_test, y_test = _safe_split(estimator, X, y, test, train)

    out = {"fit_error": None, "failed_metrics": set(), "score_time": 0.0,
           "train_scores": None}
    start = time.perf_counter()
    try:
        if y_train is None:
            estimator.fit(X_train, **fit_params)
        else:
            estimator.fit(X_train, y_train, **fit_params)
    except Exception:
        if error_score == "raise":
            raise
        out["fit_time"] = time.perf_counter() - start
        out["fit_error"] = traceback.format_exc()
        out["failed_metrics"] = set(metrics)
        out["test_scores"] = {m: error_score for m in metrics}
        if return_train_score:
            out["train_scores"] = dict(out["test_scores"])
        return out
    out["fit_time"] = time.perf_counter() - start

    start = time.perf_counter()
    test_params = _check_method_params(X, params=score_params, indices=test)
    out["test_scores"], out["failed_metrics"] = _score(
        estimator, X_test, y_test, scorer, test_params, metrics, multimetric, error_score)
    out["score_time"] = time.perf_counter() - start
    if return_train_score:
        train_params = _check_method_params(X, params=score_params, indices=train)
        out["train_scores"], _ = _score(
            estimator, X_train, y_train, scorer, train_params, metrics, multimetric,
            error_score)
    return out


def _masked_param_arrays(candidates):
    """Ключі ``param_*`` у ``cv_results_`` -- як у scikit-learn 1.8."""
    n = len(candidates)
    by_name = defaultdict(dict)
    for i, params in enumerate(candidates):
        for name, value in params.items():
            by_name[f"param_{name}"][i] = value
    for key, values in by_name.items():
        try:
            arr = np.array(list(values.values()))
        except ValueError:
            dtype = np.dtype(object)
        else:
            dtype = arr.dtype if arr.dtype.kind != "U" and arr.ndim == 1 else object
        ma = np.ma.MaskedArray(np.empty(n, dtype=dtype), mask=True)
        for i, value in values.items():
            ma[i] = value
        yield key, ma


def _failure_summary(fit_errors, candidates, n_splits):
    """Текст про невдалі навчання: кількість, повністю невдалі конфігурації, помилки."""
    n_failed = sum(e is not None for e in fit_errors)
    errors = np.array([e is not None for e in fit_errors]).reshape(len(candidates), n_splits)
    fully = [candidates[i] for i in np.flatnonzero(errors.all(axis=1))]
    lines = [f"{n_failed} з {len(fit_errors)} навчань завершилися помилкою."]
    if fully:
        lines.append(f"Конфігурації, у яких невдалі ВСІ {n_splits} навчань "
                     f"({len(fully)}): {fully}.")
    lines.append("Конфігурація з хоча б одним невдалим навчанням не може бути обрана.")
    lines.append("Помилки (кількість: останній рядок трасування):")
    last_lines = Counter(e.strip().splitlines()[-1] for e in fit_errors if e is not None)
    lines += [f"  {count}: {msg}" for msg, count in last_lines.most_common()]
    first = next(e for e in fit_errors if e is not None)
    lines.append(f"Перше повне трасування:\n{first}")
    return "\n".join(lines)


class ReshuffledSearchCV(MetaEstimatorMixin, BaseEstimator):
    """Пошук по сітці, де кожна конфігурація оцінюється на ВЛАСНИХ розбиттях.

    У стандартному ``GridSearchCV`` усі конфігурації оцінюються на спільних
    розбиттях. Тут для кожної конфігурації створюється копія розбивача ``cv``
    із власним зерном, отриманим детерміновано з ``random_state``
    (перемішування розбиттів за Nagler et al., 2024). Кількість розбиттів
    однакова для всіх конфігурацій. За ``cv`` з одним розбиттям
    (напр. ``ShuffleSplit(n_splits=1)``) це перемішаний holdout.

    Клас має власний цикл оцінки: усі пари (конфігурація, розбиття)
    навчаються одним ``joblib.Parallel``, тому ``n_jobs`` паралелить і між
    конфігураціями, зокрема за одного розбиття. Публічні атрибути -- ті самі,
    що в ``GridSearchCV``.

    Parameters
    ----------
    estimator : estimator object
        Оцінювач або ``Pipeline``; клонується для кожної пари.
    param_grid : dict або list of dict
        Сітка параметрів, як у ``GridSearchCV``.
    aggregation : {"single", "mean", "median", "max"}, default="mean"
        Правило вибору конфігурації за матрицею оцінок, див.
        :func:`aggcv.aggregate`. ``"single"`` використовує перше власне
        розбиття кожної конфігурації. ``"rank"`` і ``"vote"`` заборонені
        (``ValueError``): вони порівнюють конфігурації в межах одного
        розбиття, а тут стовпець ``split{k}`` у кожної конфігурації свій.
    scoring : str, callable, list, tuple, dict або None, default=None
        Як у ``GridSearchCV``. Callable має повертати число; для кількох
        метрик передайте list / tuple / dict.
    cv : розбивач із параметром ``random_state``
        Напр. ``ShuffleSplit``, ``GroupShuffleSplit``,
        ``StratifiedShuffleSplit``, ``KFold(shuffle=True)``,
        ``RepeatedKFold``. Його власний ``random_state`` ігнорується (об'єкт
        не змінюється). Ціле число, ``None``, ітератор розбиттів або розбивач
        без перемішування спричиняють помилку.
    random_state : int, RandomState instance або None, default=None
        Джерело зерен розбивачів. За цілого значення результат відтворюваний;
        зерно конфігурації ``i`` не залежить від розміру сітки і від
        ``n_jobs``. ``None`` -- невідтворювані розбиття.
    n_jobs : int або None, default=None
        Кількість паралельних процесів для всіх C x N пар
        (конфігурація, розбиття). Результати не залежать від ``n_jobs``.
    refit : bool або str, default=True
        Перенавчити обрану конфігурацію на всіх ``X``, ``y``. За кількох
        метрик -- ім'я метрики, за якою обирається конфігурація (або
        ``False``). Callable не підтримується.
    error_score : "raise" або float, default=np.nan
        Значення оцінки для пари, де навчання або оцінювання завершилося
        помилкою; ``"raise"`` -- пропустити виняток.
    verbose : int, default=0
        Рівень повідомлень (передається в ``joblib.Parallel``).
    return_train_score : bool, default=False
        Додати оцінки на навчальних частинах (``split{k}_train_*`` тощо).

    Attributes
    ----------
    cv_results_ : dict of numpy (masked) ndarrays
        Ті самі ключі, що в ``GridSearchCV``: ``params``, ``param_*``,
        ``split{k}_test_{metric}``, ``mean/std/rank_test_{metric}``,
        ``mean/std_fit_time``, ``mean/std_score_time`` (і ``*_train_*`` за
        ``return_train_score``). ``rank_test_*`` -- ранг за середнім.
        ``split{k}`` -- це k-те ВЛАСНЕ розбиття кожної конфігурації.
    best_index_ : int
        Індекс обраної конфігурації.
    best_params_ : dict
        Параметри обраної конфігурації.
    best_score_ : float
        Агрегована оцінка обраної конфігурації,
        ``aggregated_scores_[best_index_]``.
    best_estimator_ : estimator
        Обрана конфігурація, перенавчена на всіх даних (лише за ``refit``).
    refit_time_ : float
        Час перенавчання в секундах (лише за ``refit``).
    scorer_ : callable або dict
        Скорер (за кількох метрик -- dict ім'я -> скорер).
    multimetric_ : bool
        Чи задано кілька метрик.
    n_splits_ : int
        Кількість розбиттів на конфігурацію.
    split_seeds_ : ndarray of shape (n_configs,)
        Зерно розбивача кожної конфігурації.
    split_scores_ : ndarray of shape (n_configs, n_splits)
        Оцінки метрики вибору на власних розбиттях; пари з невдалим
        навчанням або оцінюванням -- ``-inf``. Значення в одному стовпці
        отримані на різних розбиттях і попарно не порівнюються.
    aggregated_scores_ : ndarray of shape (n_configs,)
        Агреговані оцінки всіх конфігурацій, більше = краще.

    ``split_scores_``, ``aggregated_scores_``, ``best_*`` не створюються за
    кількох метрик і ``refit=False`` (як ``best_*`` у ``GridSearchCV``).

    Notes
    -----
    Невдалі навчання. Пара, де ``fit`` завершився помилкою, отримує
    ``error_score`` у ``cv_results_`` і ``-inf`` у ``split_scores_``
    незалежно від значення ``error_score``; конфігурація з хоча б одним
    невдалим навчанням не може бути обрана. Повністю невдала конфігурація
    НЕ зупиняє пошук: ``fit`` видає одне ``FitFailedWarning`` з підсумком.
    ``ValueError`` -- лише якщо невдалі всі навчання всіх конфігурацій.
    Якщо невдалі навчання є в кожній конфігурації, обирається перша в
    порядку сітки (як у ``GridSearchCV``, де тоді всі середні -- NaN).

    Групи і metadata routing. ``fit(X, y, groups=g)`` передає групи
    розбивачу. За ``sklearn.set_config(enable_metadata_routing=True)``
    групи, fit- і score-параметри маршрутизуються через
    ``get_metadata_routing``, як у ``GridSearchCV``; зокрема у вкладеній CV
    ``cross_validate(search, ..., params={"groups": g})`` групи доходять до
    внутрішнього розбивача.

    References
    ----------
    Nagler T., Schneider L., Bischl B., Feurer M. Reshuffling Resampling
    Splits Can Improve Generalization of Hyperparameter Optimization.
    NeurIPS 2024.

    Examples
    --------
    >>> from sklearn.datasets import make_regression
    >>> from sklearn.linear_model import Lasso
    >>> from sklearn.model_selection import ShuffleSplit
    >>> from aggcv import ReshuffledSearchCV
    >>> X, y = make_regression(n_samples=60, n_features=5, random_state=0)
    >>> holdout = ShuffleSplit(n_splits=1, test_size=0.25)
    >>> search = ReshuffledSearchCV(Lasso(), {"alpha": [0.1, 1.0, 10.0]},
    ...                             cv=holdout, random_state=0).fit(X, y)
    >>> search.split_scores_.shape, len(set(search.split_seeds_))
    ((3, 1), 3)
    """

    _parameter_constraints = {
        "estimator": "no_validation",
        "param_grid": [dict, list],
        "aggregation": [str],
        "scoring": [str, callable, list, tuple, dict, None],
        "cv": ["cv_object"],
        "random_state": ["random_state"],
        "n_jobs": [numbers.Integral, None],
        "refit": ["boolean", str],
        "error_score": [str, numbers.Real],
        "verbose": ["verbose"],
        "return_train_score": ["boolean"],
    }

    def __init__(self, estimator, param_grid, *, aggregation="mean", scoring=None,
                 cv=None, random_state=None, n_jobs=None, refit=True,
                 error_score=np.nan, verbose=0, return_train_score=False):
        self.estimator = estimator
        self.param_grid = param_grid
        self.aggregation = aggregation
        self.scoring = scoring
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.refit = refit
        self.error_score = error_score
        self.verbose = verbose
        self.return_train_score = return_train_score

    def __sklearn_tags__(self):
        # як BaseSearchCV.__sklearn_tags__ у scikit-learn 1.8
        tags = super().__sklearn_tags__()
        sub = get_tags(self.estimator)
        tags.estimator_type = sub.estimator_type
        tags.classifier_tags = deepcopy(sub.classifier_tags)
        tags.regressor_tags = deepcopy(sub.regressor_tags)
        tags.input_tags.pairwise = sub.input_tags.pairwise
        tags.input_tags.sparse = sub.input_tags.sparse
        tags.array_api_support = sub.array_api_support
        return tags

    # --- перевірки параметрів -------------------------------------------------

    def _check_params(self):
        self._validate_params()
        if self.aggregation in STRATEGIES and self.aggregation not in RESHUFFLED_STRATEGIES:
            raise ValueError(
                f"aggregation={self.aggregation!r} не підтримується в "
                "ReshuffledSearchCV: правила 'rank' і 'vote' порівнюють "
                "конфігурації в межах одного розбиття, а тут кожна конфігурація "
                "оцінюється на власних розбиттях. Доступні правила: "
                f"{RESHUFFLED_STRATEGIES}; для 'rank' / 'vote' використовуйте "
                "AggregatedGridSearchCV зі спільними розбиттями."
            )
        if self.aggregation not in RESHUFFLED_STRATEGIES:
            raise ValueError(f"aggregation має бути одним із {RESHUFFLED_STRATEGIES}")
        if isinstance(self.error_score, str) and self.error_score != "raise":
            raise ValueError("error_score має бути числом або 'raise'")
        cv = self.cv
        if not hasattr(cv, "split") or not hasattr(cv, "random_state"):
            raise TypeError(
                "ReshuffledSearchCV потребує розбивач cv із параметром "
                "random_state (напр. ShuffleSplit, GroupShuffleSplit, "
                f"KFold(shuffle=True)); отримано {cv!r}."
            )
        if getattr(cv, "shuffle", True) is False:
            raise ValueError(
                f"Розбивач {cv!r} не перемішує дані (shuffle=False), тому всі "
                "конфігурації отримали б однакові розбиття."
            )

    def _get_scorers(self):
        """(callable для оцінювання, ``scorer_``, імена метрик, multimetric)."""
        scoring = self.scoring
        if isinstance(scoring, (list, tuple, dict)):
            scorer = check_scoring(self.estimator, scoring,
                                   raise_exc=self.error_score == "raise")
            items = scoring.items() if isinstance(scoring, dict) else ((s, s) for s in scoring)
            scorers = {name: get_scorer(s) if isinstance(s, str) else s for name, s in items}
            return scorer, scorers, list(scorers), True
        scorer = check_scoring(self.estimator, scoring)
        return scorer, scorer, ["score"], False

    def _selection_metric(self, multimetric, metrics):
        if not multimetric:
            return "score"
        if isinstance(self.refit, str):
            if self.refit not in metrics:
                raise ValueError(
                    f"refit={self.refit!r} має бути однією з метрик {metrics}.")
            return self.refit
        if self.refit:
            raise ValueError(
                "За кількох метрик refit має бути іменем метрики, за якою "
                f"обирається конфігурація (одна з {metrics}), або False.")
        return None

    # --- metadata routing -----------------------------------------------------

    def get_metadata_routing(self):
        """Маршрутизація метаданих (як у ``GridSearchCV``).

        ``fit`` -> ``estimator.fit``, ``splitter.split`` (напр. ``groups``),
        ``scorer.score``; ``score`` -> ``scorer.score``.

        Returns
        -------
        routing : MetadataRouter
            Об'єкт маршрутизації.
        """
        router = MetadataRouter(owner=self)
        router.add(estimator=self.estimator,
                   method_mapping=MethodMapping().add(caller="fit", callee="fit"))
        router.add(scorer=self._get_scorers()[0],
                   method_mapping=MethodMapping()
                   .add(caller="score", callee="score")
                   .add(caller="fit", callee="score"))
        router.add(splitter=self.cv,
                   method_mapping=MethodMapping().add(caller="fit", callee="split"))
        return router

    # --- навчання -------------------------------------------------------------

    def fit(self, X, y=None, *, groups=None, **fit_params):
        """Оцінює кожну конфігурацію на власних розбиттях і обирає найкращу.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Навчальні дані.
        y : array-like of shape (n_samples,) або (n_samples, n_outputs), default=None
            Цільова змінна.
        groups : array-like of shape (n_samples,), default=None
            Мітки груп для розбивача (напр. ``GroupShuffleSplit``).
        **fit_params : dict
            Параметри ``fit`` оцінювача; масиви довжини ``n_samples``
            розрізаються за індексами розбиття. За увімкненого metadata
            routing -- маршрутизуються (оцінювачу, скореру, розбивачу).

        Returns
        -------
        self : object
            Навчений пошук.
        """
        self._check_params()
        for attr in [a for a in vars(self) if a.endswith("_") and not a.startswith("_")]:
            delattr(self, attr)

        scorer, scorers, metrics, multimetric = self._get_scorers()
        metric = self._selection_metric(multimetric, metrics)
        X, y = indexable(X, y)

        if _routing_enabled():
            if groups is not None:
                fit_params["groups"] = groups
            routed = process_routing(self, "fit", **fit_params)
        else:
            routed = Bunch(estimator=Bunch(fit=fit_params),
                           splitter=Bunch(split={"groups": groups}),
                           scorer=Bunch(score={}))

        candidates = list(ParameterGrid(self.param_grid))
        n_candidates = len(candidates)
        rng = check_random_state(self.random_state)
        seeds = rng.randint(np.iinfo(np.int32).max, size=n_candidates)
        splits = []
        for seed in seeds:
            cv = clone(self.cv, safe=False)             # копія; self.cv не змінюється
            cv.random_state = int(seed)
            splits.append(list(cv.split(X, y, **routed.splitter.split)))
        n_splits = len(splits[0])
        if any(len(s) != n_splits for s in splits):
            raise ValueError("Розбивач дав різну кількість розбиттів для різних зерен.")

        if self.verbose > 0:
            print(f"Fitting {n_splits} folds for each of {n_candidates} candidates, "
                  f"totalling {n_candidates * n_splits} fits")
        out = Parallel(n_jobs=self.n_jobs, verbose=self.verbose)(
            delayed(_fit_and_score_pair)(
                self.estimator, X, y, params=params, train=train, test=test,
                fit_params=routed.estimator.fit, score_params=routed.scorer.score,
                scorer=scorer, metrics=metrics, multimetric=multimetric,
                return_train_score=self.return_train_score,
                error_score=self.error_score)
            for params, config_splits in zip(candidates, splits)
            for train, test in config_splits
        )

        fit_errors = [o["fit_error"] for o in out]
        if all(e is not None for e in fit_errors):
            raise ValueError(
                "Усі навчання всіх конфігурацій завершилися помилкою; імовірно, "
                "оцінювач або дані налаштовано неправильно (для налагодження "
                "задайте error_score='raise').\n"
                + _failure_summary(fit_errors, candidates, n_splits))
        if any(e is not None for e in fit_errors):
            warnings.warn(_failure_summary(fit_errors, candidates, n_splits),
                          FitFailedWarning)

        self.cv_results_ = self._format_results(candidates, n_splits, metrics, out)
        self.split_seeds_ = seeds
        self.scorer_ = scorers
        self.multimetric_ = multimetric
        self.n_splits_ = n_splits

        if metric is not None:
            shape = (n_candidates, n_splits)
            matrix = np.array([o["test_scores"][metric] for o in out]).reshape(shape)
            invalid = np.array([metric in o["failed_metrics"] for o in out]).reshape(shape)
            matrix[invalid | np.isnan(matrix)] = -np.inf
            best, agg = aggregate(matrix, self.aggregation)
            self.split_scores_ = matrix
            self.aggregated_scores_ = agg
            self.best_index_ = best
            self.best_params_ = candidates[best]
            self.best_score_ = float(agg[best])

        if self.refit:
            self.best_estimator_ = clone(self.estimator).set_params(
                **clone(self.best_params_, safe=False))
            start = time.perf_counter()
            if y is None:
                self.best_estimator_.fit(X, **routed.estimator.fit)
            else:
                self.best_estimator_.fit(X, y, **routed.estimator.fit)
            self.refit_time_ = time.perf_counter() - start
            if hasattr(self.best_estimator_, "feature_names_in_"):
                self.feature_names_in_ = self.best_estimator_.feature_names_in_
        return self

    def _format_results(self, candidates, n_splits, metrics, out):
        """``cv_results_`` з тими самими ключами й формулами, що в scikit-learn 1.8."""
        n_candidates = len(candidates)
        results = {}

        def store(key, values, splits=False, rank=False):
            array = np.asarray(values, dtype=np.float64).reshape(n_candidates, n_splits)
            if splits:
                for k in range(n_splits):
                    results[f"split{k}_{key}"] = array[:, k]
            means = np.average(array, axis=1)
            results[f"mean_{key}"] = means
            results[f"std_{key}"] = np.sqrt(
                np.average((array - means[:, np.newaxis]) ** 2, axis=1))
            if rank:
                if np.isnan(means).all():
                    ranks = np.ones_like(means, dtype=np.int32)
                else:
                    filled = np.nan_to_num(means, nan=np.nanmin(means) - 1)
                    ranks = rankdata(-filled, method="min").astype(np.int32, copy=False)
                results[f"rank_{key}"] = ranks

        store("fit_time", [o["fit_time"] for o in out])
        store("score_time", [o["score_time"] for o in out])
        results.update(_masked_param_arrays(candidates))
        results["params"] = candidates
        for m in metrics:
            store(f"test_{m}", [o["test_scores"][m] for o in out], splits=True, rank=True)
            if self.return_train_score:
                store(f"train_{m}", [o["train_scores"][m] for o in out], splits=True)
        return results

    # --- делегування best_estimator_ -----------------------------------------

    def score(self, X, y=None, **params):
        """Оцінка ``best_estimator_`` скорером пошуку (метрикою ``refit``).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Дані.
        y : array-like of shape (n_samples,) або (n_samples, n_outputs), default=None
            Цільова змінна.
        **params : dict
            Параметри скорера; лише за увімкненого metadata routing.

        Returns
        -------
        score : float
            Оцінка; більше = краще.
        """
        _check_refit(self, "score")
        check_is_fitted(self)
        if _routing_enabled():
            score_params = process_routing(self, "score", **params).scorer["score"]
        elif params:
            raise ValueError(
                f"Параметри {sorted(params)} для score підтримуються лише за "
                "sklearn.set_config(enable_metadata_routing=True).")
        else:
            score_params = {}
        scorer = self.scorer_[self.refit] if self.multimetric_ else self.scorer_
        return scorer(self.best_estimator_, X, y, **score_params)

    @available_if(_estimator_has("score_samples"))
    def score_samples(self, X):
        """``best_estimator_.score_samples(X)``."""
        check_is_fitted(self)
        return self.best_estimator_.score_samples(X)

    @available_if(_estimator_has("predict"))
    def predict(self, X):
        """``best_estimator_.predict(X)``."""
        check_is_fitted(self)
        return self.best_estimator_.predict(X)

    @available_if(_estimator_has("predict_proba"))
    def predict_proba(self, X):
        """``best_estimator_.predict_proba(X)``."""
        check_is_fitted(self)
        return self.best_estimator_.predict_proba(X)

    @available_if(_estimator_has("predict_log_proba"))
    def predict_log_proba(self, X):
        """``best_estimator_.predict_log_proba(X)``."""
        check_is_fitted(self)
        return self.best_estimator_.predict_log_proba(X)

    @available_if(_estimator_has("decision_function"))
    def decision_function(self, X):
        """``best_estimator_.decision_function(X)``."""
        check_is_fitted(self)
        return self.best_estimator_.decision_function(X)

    @available_if(_estimator_has("transform"))
    def transform(self, X):
        """``best_estimator_.transform(X)``."""
        check_is_fitted(self)
        return self.best_estimator_.transform(X)

    @available_if(_estimator_has("inverse_transform"))
    def inverse_transform(self, X):
        """``best_estimator_.inverse_transform(X)``."""
        check_is_fitted(self)
        return self.best_estimator_.inverse_transform(X)

    @property
    def n_features_in_(self):
        """Кількість ознак, на яких навчено ``best_estimator_``."""
        try:
            check_is_fitted(self)
        except Exception as exc:
            raise AttributeError(
                f"{type(self).__name__} ще не навчено: немає n_features_in_.") from exc
        return self.best_estimator_.n_features_in_

    @property
    def classes_(self):
        """Мітки класів ``best_estimator_`` (для класифікаторів)."""
        _estimator_has("classes_")(self)
        return self.best_estimator_.classes_
