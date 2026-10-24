"""ReshuffledSearchCV і емуляція перемішування за парною матрицею."""

import numpy as np
import pytest
import sklearn
from sklearn.base import clone
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import (GridSearchCV, GroupKFold, GroupShuffleSplit,
                                     KFold, LeaveOneGroupOut, ShuffleSplit,
                                     cross_validate)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.estimator_checks import parametrize_with_checks

from aggcv import (RESHUFFLED_STRATEGIES, ReshuffledSearchCV, aggregate,
                   reshuffle_scores, selection_stability)

RMSE = "neg_root_mean_squared_error"
GRID = {"alpha": np.logspace(-2, 3, 8)}
HOLDOUT = ShuffleSplit(n_splits=1, test_size=0.25, random_state=123)

_SPLITS = []


class RecordingShuffleSplit(ShuffleSplit):
    def split(self, X, y=None, groups=None):
        for train, test in super().split(X, y, groups):
            _SPLITS.append((self.random_state, tuple(train), tuple(test)))
            yield train, test


class RecordingGroupShuffleSplit(GroupShuffleSplit):
    def split(self, X, y=None, groups=None):
        for train, test in super().split(X, y, groups):
            _SPLITS.append((self.random_state, set(groups[train]), set(groups[test])))
            yield train, test


@pytest.fixture(scope="module")
def data():
    X, y = make_regression(n_samples=120, n_features=15, noise=30, random_state=0)
    return X, y, np.repeat(np.arange(60), 2)


# --- перемішаний holdout (N = 1): основний випадок -----------------------------

def test_reshuffled_holdout_matches_manual(data):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), GRID, cv=HOLDOUT, scoring=RMSE,
                           aggregation="single", random_state=0).fit(X, y)
    assert s.split_scores_.shape == (8, 1)
    assert "split1_test_score" not in s.cv_results_
    manual = []
    for params, seed in zip(s.cv_results_["params"], s.split_seeds_):
        train, test = next(ShuffleSplit(n_splits=1, test_size=0.25,
                                        random_state=int(seed)).split(X))
        model = Ridge(**params).fit(X[train], y[train])
        manual.append(-root_mean_squared_error(y[test], model.predict(X[test])))
    np.testing.assert_allclose(s.split_scores_[:, 0], manual, rtol=1e-12)
    assert s.best_index_ == int(np.argmax(manual))
    assert s.best_score_ == s.split_scores_[s.best_index_, 0]
    assert s.best_estimator_.alpha == s.best_params_["alpha"]


def test_reshuffled_holdout_every_config_gets_own_split(data):
    X, y, _ = data
    _SPLITS.clear()
    cv = RecordingShuffleSplit(n_splits=1, test_size=0.25, random_state=123)
    s = ReshuffledSearchCV(Ridge(), GRID, cv=cv, scoring=RMSE, random_state=5).fit(X, y)
    assert len(_SPLITS) == 8                            # по одному розбиттю на конфігурацію
    assert [seed for seed, _, _ in _SPLITS] == s.split_seeds_.tolist()
    assert len({test for _, _, test in _SPLITS}) == 8   # усі validation-множини різні
    assert len(set(s.split_seeds_.tolist())) == 8
    assert cv.random_state == 123                       # розбивач користувача не змінено


def test_reshuffled_holdout_rules_coincide(data):
    X, y, _ = data
    picks = {a: ReshuffledSearchCV(Ridge(), GRID, cv=HOLDOUT, scoring=RMSE, refit=False,
                                   aggregation=a, random_state=0).fit(X, y).best_index_
             for a in RESHUFFLED_STRATEGIES}
    assert len(set(picks.values())) == 1                # за N = 1 правила тотожні


def test_reshuffled_holdout_differs_from_shared_holdout(data):
    X, y, _ = data
    shared = GridSearchCV(Ridge(), GRID, cv=HOLDOUT, scoring=RMSE).fit(X, y)
    own = ReshuffledSearchCV(Ridge(), GRID, cv=HOLDOUT, scoring=RMSE, random_state=0).fit(X, y)
    assert not np.allclose(shared.cv_results_["split0_test_score"], own.split_scores_[:, 0])


def test_reshuffled_holdout_with_pipeline_and_groups(data):
    X, y, g = data
    _SPLITS.clear()
    pipe = make_pipeline(StandardScaler(), Lasso(max_iter=5000))
    cv = RecordingGroupShuffleSplit(n_splits=1, test_size=0.25)
    s = ReshuffledSearchCV(pipe, {"lasso__alpha": [0.1, 1.0, 10.0]}, cv=cv,
                           scoring=RMSE, random_state=1).fit(X, y, groups=g)
    assert len(_SPLITS) == 3
    for _, train_groups, val_groups in _SPLITS:
        assert not train_groups & val_groups
    assert s.predict(X).shape == y.shape


# --- відтворюваність і кількість розбиттів --------------------------------------

def test_reproducible_from_random_state(data):
    X, y, _ = data
    cv = ShuffleSplit(n_splits=4, test_size=0.25)
    kw = dict(cv=cv, scoring=RMSE, refit=False)
    a = ReshuffledSearchCV(Ridge(), GRID, random_state=7, **kw).fit(X, y)
    b = ReshuffledSearchCV(Ridge(), GRID, random_state=7, **kw).fit(X, y)
    c = ReshuffledSearchCV(Ridge(), GRID, random_state=8, **kw).fit(X, y)
    np.testing.assert_array_equal(a.split_seeds_, b.split_seeds_)
    np.testing.assert_array_equal(a.split_scores_, b.split_scores_)
    assert not np.array_equal(a.split_seeds_, c.split_seeds_)
    assert not np.array_equal(a.split_scores_, c.split_scores_)
    # власний random_state розбивача ігнорується
    other_cv = ShuffleSplit(n_splits=4, test_size=0.25, random_state=999)
    d = ReshuffledSearchCV(Ridge(), GRID, random_state=7, scoring=RMSE, refit=False,
                           cv=other_cv).fit(X, y)
    np.testing.assert_array_equal(a.split_scores_, d.split_scores_)


def test_seed_of_config_does_not_depend_on_grid_size(data):
    X, y, _ = data
    kw = dict(cv=HOLDOUT, scoring=RMSE, refit=False, random_state=3)
    small = ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, **kw).fit(X, y)
    large = ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 1.0, 10.0, 100.0]}, **kw).fit(X, y)
    np.testing.assert_array_equal(small.split_seeds_, large.split_seeds_[:2])
    np.testing.assert_array_equal(small.split_scores_, large.split_scores_[:2])


@pytest.mark.parametrize("cv, n_splits", [
    (ShuffleSplit(n_splits=5, test_size=0.25), 5),
    (KFold(n_splits=3, shuffle=True), 3),
])
def test_same_number_of_splits_for_all_configs(data, cv, n_splits):
    X, y, _ = data
    _SPLITS.clear()
    s = ReshuffledSearchCV(Ridge(), GRID, cv=cv, scoring=RMSE, random_state=0).fit(X, y)
    assert s.split_scores_.shape == (8, n_splits)
    assert np.isfinite(s.split_scores_).all()
    assert len(s.cv_results_["params"]) == 8
    assert s.n_splits_ == n_splits


@pytest.mark.parametrize("aggregation", RESHUFFLED_STRATEGIES)
def test_rules_on_own_splits(data, aggregation):
    X, y, _ = data
    _SPLITS.clear()
    cv = RecordingShuffleSplit(n_splits=4, test_size=0.25)
    s = ReshuffledSearchCV(Ridge(), GRID, cv=cv, scoring=RMSE,
                           aggregation=aggregation, random_state=2).fit(X, y)
    assert len(_SPLITS) == 8 * 4
    per_config = [frozenset(t for _, _, t in _SPLITS[i * 4:(i + 1) * 4]) for i in range(8)]
    assert len(set(per_config)) == 8                    # набори розбиттів різні
    best, agg = aggregate(s.split_scores_, aggregation)
    assert s.best_index_ == best
    assert s.best_score_ == agg[best]


def test_mean_matches_sklearn_ranking(data):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), GRID, cv=ShuffleSplit(n_splits=4, test_size=0.25),
                           scoring=RMSE, random_state=2).fit(X, y)
    assert s.best_index_ == int(np.argmin(s.cv_results_["rank_test_score"]))
    assert s.best_score_ == s.cv_results_["mean_test_score"][s.best_index_]


# --- заборони й помилки ---------------------------------------------------------

@pytest.mark.parametrize("aggregation", ["rank", "vote"])
def test_rank_and_vote_forbidden(data, aggregation):
    X, y, _ = data
    s = ReshuffledSearchCV(Ridge(), GRID, cv=HOLDOUT, aggregation=aggregation)
    with pytest.raises(ValueError, match="власних розбиттях"):
        s.fit(X, y)
    assert not hasattr(s, "cv_results_")                # помилка до навчання


@pytest.mark.parametrize("cv, error", [
    (5, TypeError),
    (None, TypeError),
    ([(np.arange(60), np.arange(60, 120))], TypeError),
    (LeaveOneGroupOut(), TypeError),                    # немає random_state
    (GroupKFold(3), ValueError),                        # shuffle=False
    (KFold(n_splits=3), ValueError),                    # shuffle=False
])
def test_invalid_cv(data, cv, error):
    X, y, _ = data
    with pytest.raises(error):
        ReshuffledSearchCV(Ridge(), GRID, cv=cv).fit(X, y)


def test_invalid_aggregation(data):
    X, y, _ = data
    with pytest.raises(ValueError):
        ReshuffledSearchCV(Ridge(), GRID, cv=HOLDOUT, aggregation="best").fit(X, y)


# --- сумісність зі scikit-learn -------------------------------------------------

def test_clone_and_params():
    s = ReshuffledSearchCV(Ridge(), GRID, cv=HOLDOUT, aggregation="median", random_state=4)
    c = clone(s)
    assert (c.aggregation, c.random_state) == ("median", 4)
    c.set_params(aggregation="max", random_state=5, estimator__alpha=3.0)
    assert (c.aggregation, c.random_state, c.estimator.alpha) == ("max", 5, 3.0)
    assert s.random_state == 4


def test_nested_grouped_cv_with_routing(data):
    X, y, g = data
    _SPLITS.clear()
    outer_cv = GroupKFold(5)
    with sklearn.config_context(enable_metadata_routing=True):
        inner = ReshuffledSearchCV(Ridge(), GRID, scoring=RMSE, random_state=0,
                                   cv=RecordingGroupShuffleSplit(n_splits=1, test_size=0.25))
        res = cross_validate(inner, X, y, cv=outer_cv, params={"groups": g},
                             scoring=RMSE, return_estimator=True)
    assert len(res["test_score"]) == 5
    assert len(_SPLITS) == 5 * 8
    outer = list(outer_cv.split(X, y, groups=g))
    for i, (_, train_groups, val_groups) in enumerate(_SPLITS):
        assert not train_groups & val_groups
        assert train_groups | val_groups == set(g[outer[i // 8][0]])


def _expected_failed_checks(est):
    failed = {"check_supervised_y_2d": "не проходить і для стандартного GridSearchCV"}
    if isinstance(est.cv, ShuffleSplit):
        # Перевірка псує рядок 0 (dict у X) і чекає TypeError від fit. Якщо рядок 0
        # потрапив у train КОЖНОГО внутрішнього навчання, невдалі всі навчання всіх
        # конфігурацій -> ValueError (так само падає GridSearchCV з таким розбиттям).
        # За цих зерен це так; з KFold(shuffle=True) кожен рядок раз потрапляє у
        # validation, частина навчань вдається, і TypeError дає фінальний refit.
        failed["check_dtype_object"] = "рядок 0 у train усіх навчань -> усі невдалі"
    return failed


@parametrize_with_checks(
    [ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, random_state=0,
                        cv=KFold(n_splits=3, shuffle=True)),
     ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, random_state=0,
                        cv=ShuffleSplit(n_splits=2, test_size=0.25)),
     ReshuffledSearchCV(Ridge(), {"alpha": [0.1, 1.0]}, random_state=0, aggregation="single",
                        cv=ShuffleSplit(n_splits=1, test_size=0.25))],
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


# --- емуляція перемішування за парною матрицею ----------------------------------

def test_reshuffle_scores_rows_are_own_subsets():
    pool = np.arange(5 * 40, dtype=float).reshape(5, 40)        # рядок i: 40*i .. 40*i+39
    sub = reshuffle_scores(pool, n_splits=10, random_state=0)
    assert sub.shape == (5, 10)
    cols = sub - 40 * np.arange(5)[:, None]                     # індекси стовпців пулу
    for i in range(5):
        assert len(set(cols[i])) == 10                          # без повторень
        assert set(cols[i]) <= set(range(40))
    assert len({tuple(row) for row in cols}) == 5               # у кожного рядка своя підмножина
    np.testing.assert_array_equal(sub, reshuffle_scores(pool, 10, random_state=0))
    assert not np.array_equal(sub, reshuffle_scores(pool, 10, random_state=1))


def test_reshuffle_scores_full_pool_and_errors():
    pool = np.random.default_rng(0).normal(size=(4, 6))
    full = reshuffle_scores(pool, random_state=0)
    np.testing.assert_array_equal(np.sort(full, axis=1), np.sort(pool, axis=1))
    with pytest.raises(ValueError):
        reshuffle_scores(pool, n_splits=7)
    with pytest.raises(ValueError):
        reshuffle_scores(pool, n_splits=0)
    with pytest.raises(ValueError):
        reshuffle_scores(np.ones(3))


def test_reshuffle_scores_expected_overlap():
    # очікувана кількість спільних розбиттів пари конфігурацій = N^2 / M
    n, m = 10, 100
    pool = np.tile(np.arange(m, dtype=float), (2, 1))
    rng = np.random.default_rng(0)
    overlap = [len(set(sub[0]) & set(sub[1]))
               for sub in (reshuffle_scores(pool, n, rng) for _ in range(3000))]
    assert np.mean(overlap) == pytest.approx(n * n / m, abs=0.1)


def test_stability_reshuffle_columns_and_rules():
    pool = np.random.default_rng(1).normal(size=(6, 30))
    df = selection_stability(pool, n_permutations=20, n_values=(1, 5, 30), reshuffle=True)
    assert sorted(df["strategy"].unique()) == sorted(RESHUFFLED_STRATEGIES)
    assert len(df) == 4 * 3
    assert df["agreement"].between(0, 1).all()
    again = selection_stability(pool, n_permutations=20, n_values=(1, 5, 30), reshuffle=True)
    assert df.equals(again)
    # еталон -- вибір на повному парному пулі
    for s in RESHUFFLED_STRATEGIES:
        assert (df.loc[df["strategy"] == s, "full_index"] == aggregate(pool, s)[0]).all()
    # за N = M кожна конфігурація бачить увесь пул: mean / median / max збігаються з еталоном
    at_full = df[df["n_splits"] == 30].set_index("strategy")
    for s in ("mean", "median", "max"):
        assert at_full.loc[s, "agreement"] == 1.0


def test_stability_reshuffle_single_is_holdout_emulation():
    # c0 краща за c1 на КОЖНОМУ розбитті, але розкид між розбиттями великий:
    # парний holdout завжди обирає c0, перемішаний -- лише коли пощастить
    base = np.linspace(0, 10, 50)
    pool = np.vstack([base + 0.1, base])
    paired = selection_stability(pool, n_permutations=300, n_values=(1,), random_state=0)
    own = selection_stability(pool, n_permutations=300, n_values=(1,), random_state=0,
                              reshuffle=True)
    p = paired.set_index("strategy").loc["mean"]
    o = own.set_index("strategy")
    assert p["agreement"] == 1.0 and p["n_distinct"] == 1
    assert 0.4 < o.loc["single", "agreement"] < 0.65
    assert o.loc["single", "agreement"] == o.loc["mean", "agreement"]    # N = 1
    assert o.loc["single", "n_distinct"] == 2


def test_stability_accepts_matrix_like_search(data):
    X, y, _ = data
    s = GridSearchCV(Ridge(), GRID, cv=ShuffleSplit(n_splits=6, random_state=0),
                     scoring=RMSE).fit(X, y)
    matrix = np.column_stack([s.cv_results_[f"split{k}_test_score"] for k in range(6)])
    for reshuffle in (False, True):
        a = selection_stability(s, n_permutations=10, reshuffle=reshuffle)
        b = selection_stability(matrix, n_permutations=10, reshuffle=reshuffle)
        assert a.equals(b)
    with pytest.raises(ValueError):
        selection_stability(matrix, X=X, y=y, X_test=X, y_test=y)
