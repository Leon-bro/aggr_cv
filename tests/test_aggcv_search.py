"""AggregatedGridSearchCV / AggregatedRandomizedSearchCV і make_refit на синтетичних даних."""

import pickle

import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     RandomizedSearchCV, cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aggcv import (STRATEGIES, AggregatedGridSearchCV,
                   AggregatedRandomizedSearchCV, aggregate, make_refit)

RMSE = "neg_root_mean_squared_error"
MULTI = {"rmse": RMSE, "mae": "neg_mean_absolute_error"}


@pytest.fixture(scope="module")
def data():
    X, y = make_regression(n_samples=120, n_features=20, noise=15, random_state=0)
    groups = np.repeat(np.arange(60), 2)        # два "ока" на "пацієнта"
    return X, y, groups


def pipe():
    return make_pipeline(StandardScaler(), Lasso(max_iter=5000))


GRID = {"lasso__alpha": np.linspace(0.05, 5, 12)}
CV = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)


def test_mean_matches_sklearn(data):
    X, y, g = data
    ref = GridSearchCV(pipe(), GRID, cv=CV, scoring=RMSE).fit(X, y, groups=g)
    ours = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="mean",
                                  scoring=RMSE).fit(X, y, groups=g)
    assert ours.best_index_ == ref.best_index_
    assert ours.best_score_ == ref.best_score_          # точна рівність
    assert ours.best_params_ == ref.best_params_
    np.testing.assert_array_equal(ours.aggregated_scores_,
                                  ref.cv_results_["mean_test_score"])


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_refit_callable_equals_subclass(data, strategy):
    X, y, g = data
    a = GridSearchCV(pipe(), GRID, cv=CV, scoring=RMSE,
                     refit=make_refit(strategy)).fit(X, y, groups=g)
    b = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation=strategy,
                               scoring=RMSE).fit(X, y, groups=g)
    assert a.best_index_ == b.best_index_
    assert a.best_params_ == b.best_params_
    np.testing.assert_array_equal(a.predict(X), b.predict(X))
    assert b.split_scores_.shape == (12, 10)
    assert b.best_score_ == b.aggregated_scores_[b.best_index_]
    assert not hasattr(a, "best_score_")        # sklearn не заповнює за callable


def test_best_score_meaning_for_rank_and_vote(data):
    X, y, g = data
    rank = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="rank",
                                  scoring=RMSE).fit(X, y, groups=g)
    assert -12 <= rank.best_score_ <= -1                # мінус середній ранг
    vote = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="vote",
                                  scoring=RMSE).fit(X, y, groups=g)
    assert 1 <= vote.best_score_ <= 10                  # кількість голосів
    assert vote.best_score_ == int(vote.best_score_)
    assert vote.aggregated_scores_.sum() == 10


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_randomized_search(data, strategy):
    X, y, g = data
    kw = dict(n_iter=6, random_state=3, cv=CV, scoring=RMSE)
    a = RandomizedSearchCV(pipe(), GRID, refit=make_refit(strategy), **kw).fit(X, y, groups=g)
    b = AggregatedRandomizedSearchCV(pipe(), GRID, aggregation=strategy, **kw).fit(X, y, groups=g)
    assert a.best_index_ == b.best_index_
    assert b.split_scores_.shape == (6, 10)
    assert b.best_score_ == b.aggregated_scores_[b.best_index_]


def test_clone_and_params():
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="rank", refit_metric="mae")
    c = clone(s)
    assert c.get_params()["aggregation"] == "rank"
    assert c.get_params()["refit_metric"] == "mae"
    c.set_params(aggregation="vote", estimator__lasso__alpha=2.0)
    assert c.aggregation == "vote"
    assert c.estimator.named_steps["lasso"].alpha == 2.0
    assert s.aggregation == "rank"

    r = clone(AggregatedRandomizedSearchCV(pipe(), GRID, aggregation="max", n_iter=3))
    assert (r.aggregation, r.n_iter) == ("max", 3)


def test_invalid_aggregation(data):
    X, y, g = data
    with pytest.raises(ValueError):
        AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="best").fit(X, y, groups=g)


def test_multimetric_refit_string(data):
    X, y, g = data
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="median",
                               scoring=MULTI, refit="mae").fit(X, y, groups=g)
    assert s.split_scores_.shape == (len(GRID["lasso__alpha"]), 10)
    np.testing.assert_array_equal(s.split_scores_[:, 3], s.cv_results_["split3_test_mae"])
    ref = GridSearchCV(pipe(), GRID, cv=CV, scoring=MULTI,
                       refit=make_refit("median", "mae")).fit(X, y, groups=g)
    assert s.best_index_ == ref.best_index_
    assert s.best_score_ == np.median(s.split_scores_[s.best_index_])


def test_multimetric_refit_metric_overrides(data):
    X, y, g = data
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="max", scoring=MULTI,
                               refit="mae", refit_metric="rmse").fit(X, y, groups=g)
    np.testing.assert_array_equal(s.split_scores_[:, 0], s.cv_results_["split0_test_rmse"])
    assert s.best_index_ == aggregate(s.split_scores_, "max")[0]


def test_multimetric_without_refit(data):
    X, y, g = data
    # як у sklearn: без refit і без refit_metric конфігурація не обирається
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, scoring=MULTI, refit=False).fit(X, y, groups=g)
    assert not hasattr(s, "best_index_")
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="rank", scoring=MULTI,
                               refit=False, refit_metric="mae").fit(X, y, groups=g)
    assert s.best_index_ == aggregate(s.split_scores_, "rank")[0]
    assert s.best_params_ == s.cv_results_["params"][s.best_index_]
    assert not hasattr(s, "best_estimator_")


def test_refit_false_single_metric(data):
    X, y, g = data
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="vote",
                               scoring=RMSE, refit=False).fit(X, y, groups=g)
    assert s.best_index_ == aggregate(s.split_scores_, "vote")[0]
    assert not hasattr(s, "best_estimator_")


def test_user_callable_refit_takes_precedence(data):
    X, y, g = data
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="mean",
                               scoring=RMSE, refit=lambda res: 5).fit(X, y, groups=g)
    assert s.best_index_ == 5
    assert not hasattr(s, "aggregated_scores_")
    assert not hasattr(s, "best_score_")


def test_failed_fit_never_wins(data):
    X, y, g = data
    grid = {"lasso__alpha": [-1.0, 0.5, 1.0]}           # alpha < 0 -> помилка fit
    for strategy in STRATEGIES:
        with pytest.warns(Warning):
            s = AggregatedGridSearchCV(pipe(), grid, cv=CV, aggregation=strategy,
                                       scoring=RMSE).fit(X, y, groups=g)
        assert s.best_index_ != 0
        assert s.aggregated_scores_[0] == -np.inf


def test_pickle_fitted(data):
    X, y, g = data
    s = AggregatedGridSearchCV(pipe(), GRID, cv=CV, aggregation="rank",
                               scoring=RMSE).fit(X, y, groups=g)
    s2 = pickle.loads(pickle.dumps(s))
    assert s2.best_index_ == s.best_index_
    np.testing.assert_array_equal(s2.predict(X), s.predict(X))


# --- вкладена групова CV -----------------------------------------------------

_INNER_SPLITS = []


class RecordingGroupShuffleSplit(GroupShuffleSplit):
    """GroupShuffleSplit, що запам'ятовує групи кожного внутрішнього розбиття."""

    def split(self, X, y=None, groups=None):
        for train, val in super().split(X, y, groups):
            _INNER_SPLITS.append((set(groups[train]), set(groups[val])))
            yield train, val


@pytest.mark.parametrize("strategy", ["rank", "vote"])
def test_nested_grouped_cv(data, strategy):
    """Зовнішній GroupKFold(5), внутрішній GroupShuffleSplit(10); groups через routing."""
    X, y, g = data
    _INNER_SPLITS.clear()
    inner_cv = RecordingGroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)
    outer_cv = GroupKFold(5)
    with sklearn.config_context(enable_metadata_routing=True):
        inner = AggregatedGridSearchCV(pipe(), GRID, cv=inner_cv,
                                       aggregation=strategy, scoring=RMSE)
        res = cross_validate(inner, X, y, cv=outer_cv, params={"groups": g},
                             scoring=RMSE, return_estimator=True)
    assert len(res["test_score"]) == 5
    assert len(_INNER_SPLITS) == 5 * 10
    outer = list(outer_cv.split(X, y, groups=g))
    for i, (train_groups, val_groups) in enumerate(_INNER_SPLITS):
        # жодна група не потрапляє одночасно в train і validation
        assert train_groups and val_groups
        assert not train_groups & val_groups
        # внутрішні розбиття бачать лише групи зовнішнього train
        outer_train, outer_test = outer[i // 10]
        assert train_groups | val_groups == set(g[outer_train])
        assert not (train_groups | val_groups) & set(g[outer_test])
    for est in res["estimator"]:
        assert est.split_scores_.shape == (12, 10)
        assert est.best_index_ == aggregate(est.split_scores_, strategy)[0]
