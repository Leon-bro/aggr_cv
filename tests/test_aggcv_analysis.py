"""compare_strategies і selection_stability."""

from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.datasets import make_regression
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split

from aggcv import (STRATEGIES, AggregatedGridSearchCV, aggregate,
                   compare_strategies, selection_stability)

RMSE = "neg_root_mean_squared_error"
MULTI = {"rmse": RMSE, "mae": "neg_mean_absolute_error"}
GRID = {"alpha": np.logspace(-2, 3, 8)}
CV = ShuffleSplit(n_splits=12, test_size=0.25, random_state=1)

FITS = []


class CountingRidge(Ridge):
    def fit(self, X, y, sample_weight=None):
        FITS.append(self.alpha)
        return super().fit(X, y, sample_weight=sample_weight)


@pytest.fixture(scope="module")
def split_data():
    X, y = make_regression(n_samples=120, n_features=15, noise=30, random_state=0)
    return train_test_split(X, y, test_size=0.3, random_state=0)


@pytest.fixture(scope="module")
def search(split_data):
    X, _, y, _ = split_data
    return GridSearchCV(CountingRidge(), GRID, cv=CV, scoring=RMSE).fit(X, y)


def fake_search(scores):
    """Мінімальний "навчений пошук" із заданою матрицею оцінок."""
    scores = np.asarray(scores, dtype=float)
    res = {f"split{k}_test_score": scores[:, k] for k in range(scores.shape[1])}
    res["params"] = [{"i": i} for i in range(scores.shape[0])]
    return SimpleNamespace(cv_results_=res, refit=True)


# --- compare_strategies ------------------------------------------------------

def test_compare_strategies_matches_separate_fits(split_data, search):
    X, _, y, _ = split_data
    n_fits = len(FITS)
    df = compare_strategies(search)
    assert len(FITS) == n_fits                          # без перенавчання
    assert list(df.columns) == ["strategy", "best_index", "params", "aggregated_score"]
    assert list(df["strategy"]) == list(STRATEGIES)
    for row in df.itertuples():
        own = AggregatedGridSearchCV(Ridge(), GRID, cv=CV, scoring=RMSE,
                                     aggregation=row.strategy).fit(X, y)
        assert row.best_index == own.best_index_
        assert row.params == own.best_params_
        assert row.aggregated_score == own.best_score_


def test_compare_strategies_toy():
    S = [[0.9, 0.1, 0.1, 0.1],
         [0.5, 0.5, 0.5, 0.5],
         [0.45, 0.6, 0.6, 0.45]]
    df = compare_strategies(fake_search(S)).set_index("strategy")
    assert df["best_index"].to_dict() == {"single": 0, "mean": 2, "median": 2,
                                          "rank": 1, "max": 0, "vote": 2}
    assert df.loc["rank", "aggregated_score"] == -1.75
    assert df.loc["vote", "aggregated_score"] == 2
    assert df.loc["max", "params"] == {"i": 0}


def test_compare_strategies_multimetric_and_refit_false(split_data):
    X, _, y, _ = split_data
    s = GridSearchCV(Ridge(), GRID, cv=CV, scoring=MULTI, refit=False).fit(X, y)
    with pytest.raises(KeyError):
        compare_strategies(s)                           # метрику не вказано
    df = compare_strategies(s, metric="mae")
    m = np.column_stack([s.cv_results_[f"split{k}_test_mae"] for k in range(12)])
    assert list(df["best_index"]) == [aggregate(m, st)[0] for st in STRATEGIES]

    by_refit = GridSearchCV(Ridge(), GRID, cv=CV, scoring=MULTI, refit="mae").fit(X, y)
    assert compare_strategies(by_refit).equals(df)
    by_param = AggregatedGridSearchCV(Ridge(), GRID, cv=CV, scoring=MULTI,
                                      refit="rmse", refit_metric="mae").fit(X, y)
    assert compare_strategies(by_param).equals(df)


# --- selection_stability -----------------------------------------------------

def test_stability_shape_and_defaults(search):
    df = selection_stability(search)
    assert list(df.columns) == ["strategy", "n_splits", "full_index", "agreement", "n_distinct"]
    assert len(df) == 6 * 12                            # N = 1..min(20, n_splits)
    assert sorted(df["n_splits"].unique()) == list(range(1, 13))
    assert df["agreement"].between(0, 1).all()
    assert (df["n_distinct"] >= 1).all() and (df["n_distinct"] <= len(GRID["alpha"])).all()
    full = compare_strategies(search).set_index("strategy")["best_index"]
    assert (df["full_index"] == df["strategy"].map(full)).all()


def test_stability_reproducible(search):
    a = selection_stability(search, n_permutations=10, n_values=(1, 3, 6))
    b = selection_stability(search, n_permutations=10, n_values=(1, 3, 6))
    c = selection_stability(search, n_permutations=10, n_values=(1, 3, 6), random_state=7)
    assert a.equals(b)
    assert not a.equals(c)


def test_stability_full_pool_is_order_invariant(search):
    df = selection_stability(search, n_permutations=20, n_values=(12,))
    df = df.set_index("strategy")
    for s in ("mean", "median", "rank"):
        assert df.loc[s, "agreement"] == 1.0
        assert df.loc[s, "n_distinct"] == 1


def test_stability_single_does_not_depend_on_n(search):
    df = selection_stability(search, n_permutations=30, n_values=(1, 4, 12))
    single = df[df["strategy"] == "single"]
    assert single["agreement"].nunique() == 1
    assert single["n_distinct"].nunique() == 1


def test_stability_manual_count():
    # c1 найкраща лише на розбитті 0, c0 -- на решті; повний пул: усі правила,
    # крім single і max, обирають c0
    S = np.array([[0.1, 0.6, 0.6, 0.6, 0.6],
                  [0.9, 0.2, 0.2, 0.2, 0.2]])
    df = selection_stability(fake_search(S), n_permutations=200, n_values=(1, 5),
                             random_state=0)
    rng = np.random.default_rng(0)
    first = np.array([rng.permutation(5)[0] for _ in range(200)])
    share_c0_first = float(np.mean(first != 0))
    df = df.set_index(["strategy", "n_splits"])
    assert df.loc[("mean", 1), "full_index"] == 0
    assert df.loc[("mean", 1), "agreement"] == share_c0_first
    assert df.loc[("mean", 1), "n_distinct"] == 2
    assert df.loc[("mean", 5), "agreement"] == 1.0
    assert df.loc[("mean", 5), "n_distinct"] == 1
    assert df.loc[("single", 5), "full_index"] == 1
    assert df.loc[("single", 5), "agreement"] == 1 - share_c0_first
    assert df.loc[("max", 5), "agreement"] == 1.0       # глобальний максимум -- c1


def test_stability_invalid_arguments(search, split_data):
    X, X_te, y, y_te = split_data
    with pytest.raises(ValueError):
        selection_stability(search, n_values=(1, 13))   # більше, ніж розбиттів
    with pytest.raises(ValueError):
        selection_stability(search, n_values=(0, 2))
    with pytest.raises(ValueError):
        selection_stability(search, n_permutations=0)
    with pytest.raises(ValueError):
        selection_stability(search, X_test=X_te, y_test=y_te)   # немає X, y


def test_stability_external_test_refits_only_final_models(search, split_data):
    X, X_te, y, y_te = split_data
    base = selection_stability(search, n_permutations=15, n_values=(1, 2, 6))
    FITS.clear()
    df = selection_stability(search, n_permutations=15, n_values=(1, 2, 6),
                             X=X, y=y, X_test=X_te, y_test=y_te)
    assert df[base.columns].equals(base)
    assert list(df.columns[-3:]) == ["test_score_mean", "test_score_std", "full_test_score"]
    # кожна різна обрана конфігурація навчається рівно один раз
    assert len(FITS) == len(set(FITS)) <= len(GRID["alpha"])
    assert set(search.cv_results_["params"][i]["alpha"] for i in df["full_index"]) <= set(FITS)

    row = df[(df["strategy"] == "mean") & (df["n_splits"] == 6)].iloc[0]
    alpha = search.cv_results_["params"][row.full_index]["alpha"]
    expected = -root_mean_squared_error(y_te, Ridge(alpha=alpha).fit(X, y).predict(X_te))
    assert row.full_test_score == pytest.approx(expected)
    assert (df["test_score_mean"] <= 0).all()           # neg RMSE: більше = краще
    stable = df[df["n_distinct"] == 1]
    assert (stable["test_score_std"] == 0).all()


def test_stability_external_test_multimetric(split_data):
    X, X_te, y, y_te = split_data
    s = GridSearchCV(Ridge(), GRID, cv=CV, scoring=MULTI, refit="mae").fit(X, y)
    df = selection_stability(s, n_permutations=5, n_values=(2,),
                             X=X, y=y, X_test=X_te, y_test=y_te)
    row = df[df["strategy"] == "mean"].iloc[0]
    alpha = s.cv_results_["params"][row.full_index]["alpha"]
    pred = Ridge(alpha=alpha).fit(X, y).predict(X_te)
    assert row.full_test_score == pytest.approx(-np.mean(np.abs(y_te - pred)))
