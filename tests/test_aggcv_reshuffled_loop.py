"""Власний цикл оцінки ReshuffledSearchCV: паралелізм, невдачі, refit, routing."""

import os
import pickle
import time
import warnings

import numpy as np
import pytest
import sklearn
from sklearn.datasets import make_classification, make_regression
from sklearn.exceptions import FitFailedWarning
from sklearn.linear_model import Lasso, LogisticRegression, Ridge
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     KFold, ShuffleSplit, cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from aggcv import ReshuffledSearchCV, compare_strategies, selection_stability

RMSE = "neg_root_mean_squared_error"
HOLDOUT = ShuffleSplit(n_splits=1, test_size=0.25)
TIME_KEYS = {"mean_fit_time", "std_fit_time", "mean_score_time", "std_score_time"}


@pytest.fixture(scope="module")
def data():
    X, y = make_regression(n_samples=120, n_features=15, noise=30, random_state=0)
    return X, y, np.repeat(np.arange(60), 2)


class FailingRidge(Ridge):
    """Ridge, що падає за alpha < 0 (для перевірки політики невдалих навчань)."""

    def fit(self, X, y, sample_weight=None):
        if self.alpha < 0:
            raise RuntimeError("alpha < 0")
        return super().fit(X, y, sample_weight=sample_weight)


class FlakyRidge(Ridge):
    """Ridge, що падає, якщо в train потрапив рядок ``bad_row``."""

    def __init__(self, alpha=1.0, bad_row=-1):
        super().__init__(alpha=alpha)
        self.bad_row = bad_row

    def fit(self, X, y, sample_weight=None):
        if self.bad_row >= 0 and np.any(X[:, -1] == self.bad_row):
            raise RuntimeError("bad row in train")
        return super().fit(X, y, sample_weight=sample_weight)


_FIT_LOG = []


class RecordingRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        _FIT_LOG.append((len(X), None if sample_weight is None else np.asarray(sample_weight)))
        return super().fit(X, y, sample_weight=sample_weight)


class PidRidge(Ridge):
    """Записує PID процесу, де відбулося навчання (для перевірки паралелізму)."""

    def __init__(self, alpha=1.0, log_dir=None):
        super().__init__(alpha=alpha)
        self.log_dir = log_dir

    def fit(self, X, y, sample_weight=None):
        if self.log_dir is not None:
            time.sleep(0.3)
            with open(os.path.join(self.log_dir, f"{os.getpid()}_{time.perf_counter_ns()}"), "w"):
                pass
        return super().fit(X, y, sample_weight=sample_weight)


def _assert_same_results(a, b):
    assert set(a.cv_results_) == set(b.cv_results_)
    for key in set(a.cv_results_) - TIME_KEYS:
        if key == "params":
            assert a.cv_results_[key] == b.cv_results_[key]
        else:
            np.testing.assert_array_equal(a.cv_results_[key], b.cv_results_[key], err_msg=key)
    np.testing.assert_array_equal(a.split_scores_, b.split_scores_)
    assert a.best_index_ == b.best_index_
    assert a.best_score_ == b.best_score_


# --- n_jobs -------------------------------------------------------------------

@pytest.mark.parametrize("cv, aggregation", [(HOLDOUT, "single"),
                                             (ShuffleSplit(n_splits=3, test_size=0.25), "median")])
def test_n_jobs_does_not_change_results(data, cv, aggregation):
    X, y, _ = data
    kw = dict(cv=cv, scoring=RMSE, aggregation=aggregation, random_state=11)
    grid = {"alpha": np.logspace(-2, 3, 6)}
    one = ReshuffledSearchCV(Ridge(), grid, n_jobs=1, **kw).fit(X, y)
    two = ReshuffledSearchCV(Ridge(), grid, n_jobs=2, **kw).fit(X, y)
    _assert_same_results(one, two)
    np.testing.assert_array_equal(one.split_seeds_, two.split_seeds_)


def test_holdout_is_parallel_across_configs(data, tmp_path):
    # за N = 1 паралелізм -- між конфігураціями: навчання йдуть у кількох процесах
    X, y, _ = data
    s = ReshuffledSearchCV(PidRidge(log_dir=str(tmp_path)), {"alpha": [0.1, 1.0, 10.0, 100.0]},
                           cv=HOLDOUT, aggregation="single", random_state=0, n_jobs=2,
                           refit=False).fit(X, y)
    pids = {name.split("_")[0] for name in os.listdir(tmp_path)}
    assert len(os.listdir(tmp_path)) == 4
    assert len(pids) >= 2
    assert str(os.getpid()) not in pids
    assert s.split_scores_.shape == (4, 1)


# --- невдалі навчання ---------------------------------------------------------

def test_fully_failed_config_does_not_stop_search(data):
    X, y, _ = data
    s = ReshuffledSearchCV(FailingRidge(), {"alpha": [1.0, -1.0, 10.0]}, cv=HOLDOUT,
                           aggregation="single", random_state=0, scoring=RMSE)
    with pytest.warns(FitFailedWarning, match=r"1 з 3 навчань") as record:
        s.fit(X, y)
    assert len([w for w in record if w.category is FitFailedWarning]) == 1
    msg = str(next(w.message for w in record if w.category is FitFailedWarning))
    assert "{'alpha': -1.0}" in msg and "RuntimeError: alpha < 0" in msg
    assert np.isnan(s.cv_results_["split0_test_score"][1])     # error_score у cv_results_
    assert s.split_scores_[1, 0] == -np.inf
    assert s.aggregated_scores_[1] == -np.inf
    assert s.best_params_["alpha"] != -1.0
    assert s.cv_results_["rank_test_score"][1] == 3            # останній, як у GridSearchCV


@pytest.mark.parametrize("aggregation", ["mean", "median", "max"])
def test_partially_failed_config_cannot_win(data, aggregation):
    # bad_row у train лише частини розбиттів конфігурації -> часткова невдача
    X, y, _ = data
    Xi = np.column_stack([X, np.arange(len(X))])               # остання колонка -- номер рядка
    grid = [{"alpha": [0.1], "bad_row": [-1]}, {"alpha": [0.1], "bad_row": [5]}]
    s = ReshuffledSearchCV(FlakyRidge(), grid, cv=ShuffleSplit(n_splits=6, test_size=0.5),
                           scoring=RMSE, aggregation=aggregation, random_state=3)
    with pytest.warns(FitFailedWarning):
        s.fit(Xi, y)
    failed = np.isnan(np.column_stack(
        [s.cv_results_[f"split{k}_test_score"] for k in range(6)]))[1]
    assert 0 < failed.sum() < 6
    # без невдалих розбиттів конфігурація 1 тотожна конфігурації 0 на своїх розбиттях,
    # і на деяких розбиттях вона краща; проте обрана все одно 0
    assert s.best_index_ == 0
    assert s.aggregated_scores_[1] == -np.inf


def test_numeric_error_score_still_excluded(data):
    # error_score=0 краще за будь-яке -RMSE, але невдала конфігурація не перемагає
    X, y, _ = data
    s = ReshuffledSearchCV(FailingRidge(), {"alpha": [1.0, -1.0]}, cv=HOLDOUT, scoring=RMSE,
                           aggregation="single", random_state=0, error_score=0.0)
    with pytest.warns(FitFailedWarning):
        s.fit(X, y)
    assert s.cv_results_["split0_test_score"][1] == 0.0
    assert s.best_index_ == 0


def test_all_fits_failed_raises(data):
    X, y, _ = data
    s = ReshuffledSearchCV(FailingRidge(), {"alpha": [-1.0, -2.0]},
                           cv=ShuffleSplit(n_splits=2, test_size=0.25), random_state=0)
    with pytest.raises(ValueError, match=r"(?s)Усі навчання всіх конфігурацій.*4 з 4"):
        s.fit(X, y)


def test_error_score_raise(data):
    X, y, _ = data
    s = ReshuffledSearchCV(FailingRidge(), {"alpha": [1.0, -1.0]}, cv=HOLDOUT,
                           random_state=0, error_score="raise")
    with pytest.raises(RuntimeError, match="alpha < 0"):
        s.fit(X, y)


def test_invalid_error_score(data):
    X, y, _ = data
    with pytest.raises(ValueError):
        ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, cv=HOLDOUT, error_score="nan").fit(X, y)


# --- атрибути, refit, multimetric -----------------------------------------------

def test_cv_results_keys_match_gridsearchcv(data):
    X, y, _ = data
    grid = [{"alpha": [0.1, 1.0]}, {"alpha": [10.0], "fit_intercept": [False]}]
    cv = ShuffleSplit(n_splits=3, test_size=0.25, random_state=0)
    scoring = {"rmse": RMSE, "r2": "r2"}
    for kw in (dict(scoring=RMSE), dict(scoring=scoring, refit="rmse"),
               dict(scoring=RMSE, return_train_score=True)):
        ours = ReshuffledSearchCV(Ridge(), grid, cv=cv, random_state=0, **kw).fit(X, y)
        ref = GridSearchCV(Ridge(), grid, cv=cv, **kw).fit(X, y)
        assert list(ours.cv_results_) == list(ref.cv_results_)
        for key, value in ref.cv_results_.items():
            if key.startswith("param_"):
                assert ours.cv_results_[key].dtype == value.dtype
                np.testing.assert_array_equal(ours.cv_results_[key].mask, value.mask)
        for attr in ("best_index_", "best_params_", "best_score_", "best_estimator_",
                     "refit_time_", "n_splits_", "scorer_", "multimetric_"):
            assert hasattr(ours, attr), attr
        assert type(ours.scorer_) is type(ref.scorer_)


def test_refit_false_single_metric_still_selects(data):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 10.0]}, cv=HOLDOUT, refit=False,
                           random_state=0).fit(X, y)
    assert hasattr(s, "best_index_") and hasattr(s, "best_score_")
    assert not hasattr(s, "best_estimator_") and not hasattr(s, "refit_time_")
    with pytest.raises(AttributeError) as exc:
        s.predict(X)
    assert "refit=False" in str(exc.value.__cause__)        # як у GridSearchCV (available_if)
    assert not hasattr(s, "predict")


def test_multimetric_refit(data):
    X, y, _ = data
    scoring = {"rmse": RMSE, "r2": "r2"}
    grid = {"alpha": np.logspace(-2, 3, 6)}
    kw = dict(cv=ShuffleSplit(n_splits=3, test_size=0.25), scoring=scoring,
              aggregation="median", random_state=0)
    s = ReshuffledSearchCV(Ridge(), grid, refit="r2", **kw).fit(X, y)
    r2 = np.column_stack([s.cv_results_[f"split{k}_test_r2"] for k in range(3)])
    np.testing.assert_array_equal(s.split_scores_, r2)
    assert s.best_index_ == int(np.argmax(np.median(r2, axis=1)))
    assert s.multimetric_ and set(s.scorer_) == {"rmse", "r2"}
    assert s.score(X, y) == pytest.approx(s.scorer_["r2"](s.best_estimator_, X, y))

    none = ReshuffledSearchCV(Ridge(), grid, refit=False, **kw).fit(X, y)
    for attr in ("best_index_", "best_params_", "best_score_", "best_estimator_",
                 "split_scores_", "aggregated_scores_"):
        assert not hasattr(none, attr)
    _ = none.cv_results_["mean_test_rmse"]

    with pytest.raises(ValueError, match="іменем метрики"):
        ReshuffledSearchCV(Ridge(), grid, refit=True, **kw).fit(X, y)
    with pytest.raises(ValueError, match="однією з метрик"):
        ReshuffledSearchCV(Ridge(), grid, refit="mae", **kw).fit(X, y)


def test_refit_on_all_data_and_delegation(data):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 10.0]}, cv=HOLDOUT, scoring=RMSE,
                           random_state=0).fit(X, y)
    ref = Ridge(**s.best_params_).fit(X, y)
    np.testing.assert_array_equal(s.predict(X), ref.predict(X))
    assert s.refit_time_ >= 0
    assert s.n_features_in_ == X.shape[1]
    assert s.score(X, y) == pytest.approx(-np.sqrt(np.mean((ref.predict(X) - y) ** 2)))
    restored = pickle.loads(pickle.dumps(s))
    np.testing.assert_array_equal(restored.predict(X), s.predict(X))


def test_classifier_delegation():
    X, y = make_classification(n_samples=80, random_state=0)
    s = ReshuffledSearchCV(LogisticRegression(), {"C": [0.1, 1.0]}, random_state=0,
                           cv=KFold(n_splits=3, shuffle=True)).fit(X, y)
    assert s.predict_proba(X).shape == (80, 2)
    assert s.decision_function(X).shape == (80,)
    np.testing.assert_array_equal(s.classes_, [0, 1])
    assert sklearn.base.is_classifier(s)


def test_transform_delegation(data):
    X, y, _ = data
    pipe = make_pipeline(StandardScaler(), Ridge())
    s = ReshuffledSearchCV(StandardScaler(), {"with_mean": [True, False]}, cv=HOLDOUT,
                           scoring=lambda est, X_, y_=None: 0.0, random_state=0).fit(X)
    assert s.transform(X).shape == X.shape
    assert not hasattr(ReshuffledSearchCV(pipe, {}, cv=HOLDOUT), "transform")


def test_scoring_callable_returning_dict_rejected(data):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, cv=HOLDOUT, random_state=0,
                           scoring=lambda est, X_, y_: {"a": 1.0}, error_score="raise")
    with pytest.raises(ValueError, match="Callable, що повертає dict"):
        s.fit(X, y)


def test_analysis_utilities_work_unchanged(data):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), {"alpha": np.logspace(-2, 3, 5)}, scoring=RMSE,
                           cv=ShuffleSplit(n_splits=4, test_size=0.25), random_state=0).fit(X, y)
    cmp = compare_strategies(s).set_index("strategy")
    assert cmp.loc["mean", "best_index"] == s.best_index_
    stab = selection_stability(s, n_permutations=5, n_values=(1, 4), reshuffle=True,
                               X=X, y=y, X_test=X, y_test=y)
    assert {"agreement", "test_score_mean"} <= set(stab.columns)


def test_refit_is_not_repeated_and_fits_counted(data):
    X, y, _ = data
    _FIT_LOG.clear()
    ReshuffledSearchCV(RecordingRidge(), {"alpha": [0.1, 1.0, 10.0]},
                       cv=ShuffleSplit(n_splits=2, test_size=0.25), random_state=0).fit(X, y)
    assert len(_FIT_LOG) == 3 * 2 + 1
    assert _FIT_LOG[-1][0] == len(X)                           # refit на всіх даних


# --- fit-параметри, групи, routing ---------------------------------------------

def test_fit_params_sliced_without_routing(data):
    X, y, _ = data
    _FIT_LOG.clear()
    w = np.arange(1.0, len(X) + 1)
    s = ReshuffledSearchCV(RecordingRidge(), {"alpha": [0.1, 1.0]}, cv=HOLDOUT,
                           random_state=0).fit(X, y, sample_weight=w)
    for (n, sw), seed in zip(_FIT_LOG[:2], s.split_seeds_):
        train, _ = next(ShuffleSplit(n_splits=1, test_size=0.25,
                                     random_state=int(seed)).split(X))
        assert n == len(train)
        np.testing.assert_array_equal(sw, w[train])
    np.testing.assert_array_equal(_FIT_LOG[-1][1], w)          # refit -- усі ваги


def test_groups_never_split_between_train_and_val(data):
    X, y, g = data
    Xg = np.column_stack([X, g])
    seen = []

    class GroupCheckRidge(Ridge):
        def fit(self, X_, y_, sample_weight=None):
            seen.append(set(X_[:, -1]))
            return super().fit(X_, y_)

    s = ReshuffledSearchCV(GroupCheckRidge(), {"alpha": [0.1, 1.0, 10.0]}, random_state=0,
                           cv=GroupShuffleSplit(n_splits=3, test_size=0.25)).fit(Xg, y, groups=g)
    for i, seed in enumerate(s.split_seeds_):
        splits = GroupShuffleSplit(n_splits=3, test_size=0.25,
                                   random_state=int(seed)).split(Xg, y, g)
        for k, (train, test) in enumerate(splits):
            assert not set(g[train]) & set(g[test])
            assert seen[i * 3 + k] == set(g[train])


def test_groups_required_by_group_splitter(data):
    X, y, _ = data
    with pytest.raises(ValueError, match="groups"):
        ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, random_state=0,
                           cv=GroupShuffleSplit(n_splits=1)).fit(X, y)


def test_routing_groups_and_fit_params_in_nested_cv(data):
    X, y, g = data
    _FIT_LOG.clear()
    w = np.arange(1.0, len(X) + 1)
    outer_cv = GroupKFold(4)
    with sklearn.config_context(enable_metadata_routing=True):
        est = RecordingRidge().set_fit_request(sample_weight=True)
        inner = ReshuffledSearchCV(est, {"alpha": [0.1, 1.0]}, scoring=RMSE, random_state=0,
                                   cv=GroupShuffleSplit(n_splits=2, test_size=0.25))
        res = cross_validate(inner, X, y, cv=outer_cv, params={"groups": g, "sample_weight": w},
                             return_estimator=True)
    assert len(res["test_score"]) == 4
    assert len(_FIT_LOG) == 4 * (2 * 2 + 1)
    for i, (outer_train, _) in enumerate(outer_cv.split(X, y, groups=g)):
        search = res["estimator"][i]
        log = _FIT_LOG[i * 5:(i + 1) * 5]
        g_tr, w_tr = g[outer_train], w[outer_train]
        for j, seed in enumerate(search.split_seeds_):
            cv = GroupShuffleSplit(n_splits=2, test_size=0.25, random_state=int(seed))
            for k, (train, test) in enumerate(cv.split(outer_train, groups=g_tr)):
                assert not set(g_tr[train]) & set(g_tr[test])
                n, sw = log[j * 2 + k]
                assert n == len(train)
                np.testing.assert_array_equal(sw, w_tr[train])   # ваги дійшли й розрізані
        np.testing.assert_array_equal(log[-1][1], w_tr)          # refit -- усі ваги


def test_routing_without_request_raises(data):
    X, y, g = data
    w = np.ones(len(X))
    with sklearn.config_context(enable_metadata_routing=True):
        s = ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, random_state=0,
                               cv=GroupShuffleSplit(n_splits=1))
        with pytest.raises(Exception, match="sample_weight"):
            s.fit(X, y, groups=g, sample_weight=w)
        s.fit(X, y, groups=g)                                    # groups -- розбивачу
        assert s.n_splits_ == 1


def test_score_params_require_routing(data):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, cv=HOLDOUT, random_state=0).fit(X, y)
    with pytest.raises(ValueError, match="enable_metadata_routing"):
        s.score(X, y, sample_weight=np.ones(len(X)))


def test_get_metadata_routing_structure():
    s = ReshuffledSearchCV(Ridge(), {"alpha": [1.0]}, cv=GroupShuffleSplit(n_splits=1))
    with sklearn.config_context(enable_metadata_routing=True):
        routing = sklearn.utils.metadata_routing.get_routing_for_object(s)
        assert "groups" in routing.consumes("fit", ["groups", "foo"])


def test_verbose_prints_summary(data, capsys):
    X, y, _ = data
    ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=ShuffleSplit(n_splits=2),
                       random_state=0, verbose=1).fit(X, y)
    assert "Fitting 2 folds for each of 2 candidates, totalling 4 fits" in capsys.readouterr().out


def test_no_warnings_without_failures(data):
    X, y, _ = data
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, cv=HOLDOUT, random_state=0).fit(X, y)


def test_pipeline_with_groups(data):
    X, y, g = data
    pipe = make_pipeline(StandardScaler(), Lasso(max_iter=5000))
    s = ReshuffledSearchCV(pipe, {"lasso__alpha": [0.1, 1.0]}, scoring=RMSE, random_state=0,
                           cv=GroupShuffleSplit(n_splits=2, test_size=0.25)).fit(X, y, groups=g)
    assert s.best_estimator_.named_steps["lasso"].alpha == s.best_params_["lasso__alpha"]
    assert s.predict(X).shape == y.shape
