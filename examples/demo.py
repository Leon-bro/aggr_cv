"""Порівняння шести правил агрегування на синтетичних даних.

Запуск із кореня репозиторію::

    python examples/demo.py

Дані: ``make_regression``, по 2 спостереження на групу (як два ока пацієнта).
Зовнішній цикл -- GroupKFold(5), внутрішній -- GroupShuffleSplit(10).
"""

import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aggcv import (STRATEGIES, AggregatedGridSearchCV,  # noqa: E402
                   compare_strategies, selection_stability)

RMSE = "neg_root_mean_squared_error"


def main():
    X, y = make_regression(n_samples=160, n_features=30, n_informative=6,
                           noise=25, random_state=0)
    groups = np.repeat(np.arange(80), 2)

    pipe = make_pipeline(StandardScaler(), Lasso(max_iter=10_000))
    grid = {"lasso__alpha": np.logspace(-2, 1.5, 15)}
    inner_cv = GroupShuffleSplit(n_splits=10, test_size=0.25, random_state=42)

    # 1. Один пошук -- шість правил на тій самій матриці оцінок
    search = AggregatedGridSearchCV(pipe, grid, cv=inner_cv, scoring=RMSE)
    search.fit(X, y, groups=groups)
    print("Вибір на всіх даних (одна матриця 15 конфігурацій x 10 розбиттів)")
    print(compare_strategies(search).to_string(index=False))

    # 2. Стабільність вибору за перестановками розбиттів
    stab = selection_stability(search, n_permutations=50, n_values=(1, 2, 5, 10))
    print("\nЧастка збігів із вибором на повному пулі (50 перестановок)")
    print(stab.pivot(index="strategy", columns="n_splits", values="agreement")
              .loc[list(STRATEGIES)].to_string())
    print("\nКількість різних обраних конфігурацій")
    print(stab.pivot(index="strategy", columns="n_splits", values="n_distinct")
              .loc[list(STRATEGIES)].to_string())

    # 3. Вкладена групова CV: чесна оцінка кожного правила
    print("\nВкладена CV: зовнішній RMSE (середнє ± ст. відхилення за 5 фолдами)")
    with sklearn.config_context(enable_metadata_routing=True):
        for strategy in STRATEGIES:
            inner = AggregatedGridSearchCV(pipe, grid, cv=inner_cv,
                                           aggregation=strategy, scoring=RMSE)
            res = cross_validate(inner, X, y, cv=GroupKFold(5),
                                 params={"groups": groups}, scoring=RMSE,
                                 return_estimator=True)
            alphas = [e.best_params_["lasso__alpha"] for e in res["estimator"]]
            rmse = -res["test_score"]
            print(f"{strategy:<8} {rmse.mean():7.2f} ± {rmse.std():5.2f}   "
                  f"alpha за фолдами: {np.round(alphas, 3).tolist()}")


if __name__ == "__main__":
    main()
