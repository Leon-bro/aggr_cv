"""Сторож приватного API scikit-learn, на який спирається aggcv.

1. ``_AggregationMixin`` (AggregatedGridSearchCV / AggregatedRandomizedSearchCV)
   перевизначає ``BaseSearchCV._select_best_index``. Тести мають упасти, якщо в
   новій версії scikit-learn зміниться сигнатура методу, його поведінка або
   спосіб, у який ``fit`` його викликає й заповнює ``best_score_``.
2. ``ReshuffledSearchCV`` використовує три дрібні приватні помічники:
   ``BaseEstimator._validate_params``, ``sklearn.utils.metaestimators._safe_split``
   і ``sklearn.utils.validation._check_method_params``.

Перевірено на scikit-learn 1.8.0.
"""

import inspect

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection._search import BaseSearchCV
from sklearn.svm import SVR
from sklearn.utils.metaestimators import _safe_split
from sklearn.utils.validation import _check_method_params

from aggcv import AggregatedGridSearchCV, ReshuffledSearchCV

RESULTS = {
    "params": [{"alpha": 1}, {"alpha": 2}, {"alpha": 3}],
    "rank_test_score": np.array([2, 1, 3]),
    "rank_test_mae": np.array([3, 2, 1]),
}


def test_signature_unchanged():
    assert isinstance(inspect.getattr_static(BaseSearchCV, "_select_best_index"), staticmethod)
    sig = inspect.signature(BaseSearchCV._select_best_index)
    assert list(sig.parameters) == ["refit", "refit_metric", "results"]
    for cls in (GridSearchCV, RandomizedSearchCV):
        assert "_select_best_index" not in vars(cls)


def test_behaviour_unchanged():
    select = BaseSearchCV._select_best_index
    assert select(True, "score", RESULTS) == 1          # argmin рангу за середнім
    assert select("mae", "mae", RESULTS) == 2
    seen = []
    assert select(lambda res: seen.append(res) or 0, "score", RESULTS) == 0
    assert seen == [RESULTS]                            # callable отримує results
    with pytest.raises(TypeError):
        select(lambda res: 1.0, "score", RESULTS)
    with pytest.raises(IndexError):
        select(lambda res: 3, "score", RESULTS)


@pytest.mark.parametrize(
    "scoring, refit, expected_metric",
    [(None, True, "score"),
     ({"r2": "r2", "mae": "neg_mean_absolute_error"}, "mae", "mae")],
)
def test_fit_calls_hook_once_and_sets_best_score(scoring, refit, expected_metric):
    calls = []

    class Spy(GridSearchCV):
        def _select_best_index(self, refit, refit_metric, results):
            calls.append((refit, refit_metric, results))
            return 2                                    # свідомо не найкраща

    X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    s = Spy(Ridge(), {"alpha": [0.1, 1.0, 1e6]}, cv=3, scoring=scoring, refit=refit).fit(X, y)
    assert len(calls) == 1
    assert calls[0][0] is refit and calls[0][1] == expected_metric
    assert f"split2_test_{expected_metric}" in calls[0][2]
    assert calls[0][2] is s.cv_results_
    assert s.best_index_ == 2
    assert s.best_params_ == {"alpha": 1e6}
    assert s.best_estimator_.alpha == 1e6               # refit за повернутим індексом
    assert s.best_score_ == s.cv_results_[f"mean_test_{expected_metric}"][2]


def test_subclass_refits_once():
    fits = []

    class CountingRidge(Ridge):
        def fit(self, X, y, sample_weight=None):
            fits.append(self.alpha)
            return super().fit(X, y, sample_weight=sample_weight)

    X, y = make_regression(n_samples=40, n_features=4, random_state=0)
    AggregatedGridSearchCV(CountingRidge(), {"alpha": [0.1, 1.0]}, cv=3,
                           aggregation="rank").fit(X, y)
    assert len(fits) == 2 * 3 + 1                       # без подвійного refit


# --- ReshuffledSearchCV: дрібні приватні помічники --------------------------------

def test_validate_params_hook():
    """``_validate_params()`` перевіряє ``_parameter_constraints`` і кидає
    InvalidParameterError (підклас ValueError і TypeError)."""
    from sklearn.base import BaseEstimator
    from sklearn.utils._param_validation import InvalidParameterError

    assert list(inspect.signature(BaseEstimator._validate_params).parameters) == ["self"]
    s = ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, n_jobs="two")
    with pytest.raises(InvalidParameterError, match="n_jobs"):
        s._validate_params()
    assert issubclass(InvalidParameterError, ValueError)
    assert issubclass(InvalidParameterError, TypeError)
    ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, cv=3)._validate_params()   # cv_object


def test_safe_split_contract():
    assert list(inspect.signature(_safe_split).parameters) == [
        "estimator", "X", "y", "indices", "train_indices"]
    X = np.arange(20.0).reshape(10, 2)
    y = np.arange(10.0)
    train, test = np.array([0, 2, 4, 6]), np.array([1, 3])
    Xt, yt = _safe_split(Ridge(), X, y, train)
    np.testing.assert_array_equal(Xt, X[train])
    np.testing.assert_array_equal(yt, y[train])
    assert _safe_split(Ridge(), X, None, train)[1] is None
    # попередньо обчислене ядро: рядки -- indices, стовпці -- train_indices
    K = np.arange(100.0).reshape(10, 10)
    pre = SVR(kernel="precomputed")
    np.testing.assert_array_equal(_safe_split(pre, K, y, train)[0], K[np.ix_(train, train)])
    np.testing.assert_array_equal(_safe_split(pre, K, y, test, train)[0], K[np.ix_(test, train)])


def test_check_method_params_contract():
    assert list(inspect.signature(_check_method_params).parameters) == [
        "X", "params", "indices"]
    X = np.zeros((6, 2))
    w = np.arange(6.0)
    idx = np.array([1, 4])
    out = _check_method_params(X, params={"sample_weight": w, "flag": 3,
                                          "short": np.arange(4.0)}, indices=idx)
    np.testing.assert_array_equal(out["sample_weight"], w[idx])   # довжина n_samples -- розрізано
    assert out["flag"] == 3                                       # не масив -- без змін
    np.testing.assert_array_equal(out["short"], np.arange(4.0))   # інша довжина -- без змін
    assert _check_method_params(X, params={}, indices=idx) == {}
