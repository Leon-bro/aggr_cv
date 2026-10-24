"""aggcv -- GridSearchCV з правилами агрегування результатів розбиттів.

Два рівні використання:

1. ``make_refit(strategy)`` -> callable для стандартного
   ``GridSearchCV(refit=...)``; лише публічний API scikit-learn.
2. ``AggregatedGridSearchCV`` / ``AggregatedRandomizedSearchCV`` -- підкласи з
   параметром ``aggregation``; додатково заповнюють ``best_score_``,
   ``split_scores_`` і ``aggregated_scores_``.

Аналітичні утиліти для вже навченого пошуку (без перенавчання внутрішніх
моделей): ``compare_strategies``, ``selection_stability``, ``reshuffle_scores``.

``ReshuffledSearchCV`` -- пошук, де кожна конфігурація оцінюється на власних
розбиттях (Nagler et al., 2024).
"""

from ._aggregation import (RESHUFFLED_STRATEGIES, STRATEGIES, aggregate,
                           split_score_matrix)
from ._analysis import (compare_strategies, reshuffle_scores,
                        selection_stability)
from ._refit import make_refit
from ._reshuffled import ReshuffledSearchCV
from ._search import AggregatedGridSearchCV, AggregatedRandomizedSearchCV

__version__ = "0.4.0"

__all__ = [
    "RESHUFFLED_STRATEGIES",
    "STRATEGIES",
    "AggregatedGridSearchCV",
    "AggregatedRandomizedSearchCV",
    "ReshuffledSearchCV",
    "aggregate",
    "compare_strategies",
    "make_refit",
    "reshuffle_scores",
    "selection_stability",
    "split_score_matrix",
]
