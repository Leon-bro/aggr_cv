"""Правила агрегування на ручних матрицях (без навчання моделей)."""

import pickle

import numpy as np
import pytest

from aggcv import STRATEGIES, aggregate, make_refit, split_score_matrix


def test_toy_rules_disagree():
    # 3 конфігурації x 4 розбиття, більше = краще
    S = np.array([[0.9, 0.1, 0.1, 0.1],     # один дуже вдалий спліт
                  [0.5, 0.5, 0.5, 0.5],     # стабільна
                  [0.45, 0.6, 0.6, 0.45]])
    assert aggregate(S, "single")[0] == 0
    assert aggregate(S, "max")[0] == 0
    assert aggregate(S, "mean")[0] == 2
    assert aggregate(S, "median")[0] == 2
    assert aggregate(S, "vote")[0] == 2      # 2 перемоги проти 1 і 1
    # середні ранги конфігурацій 1 і 2 однакові (1.75): нічия розв'язується
    # меншою дисперсією рангів -> стабільна конфігурація 1
    assert aggregate(S, "rank")[0] == 1


def test_aggregated_values():
    S = np.array([[0.9, 0.1, 0.1, 0.1],
                  [0.5, 0.5, 0.5, 0.5],
                  [0.45, 0.6, 0.6, 0.45]])
    np.testing.assert_array_equal(aggregate(S, "single")[1], S[:, 0])
    np.testing.assert_array_equal(aggregate(S, "mean")[1], S.mean(axis=1))
    np.testing.assert_array_equal(aggregate(S, "median")[1], np.median(S, axis=1))
    np.testing.assert_array_equal(aggregate(S, "max")[1], S.max(axis=1))
    np.testing.assert_array_equal(aggregate(S, "vote")[1], [1, 1, 2])
    # ранги за розбиттями: c0 = 1,3,3,3; c1 = 2,2,2,1; c2 = 3,1,1,2
    np.testing.assert_array_equal(aggregate(S, "rank")[1], [-2.5, -1.75, -1.75])


def test_input_not_modified():
    S = np.array([[np.nan, 1.0], [0.5, 0.5]])
    before = S.copy()
    for s in STRATEGIES:
        aggregate(S, s)
    np.testing.assert_array_equal(S, before)


@pytest.mark.parametrize("strategy", ["single", "mean", "median"])
def test_tie_first_index(strategy):
    S = np.array([[0.1, 0.1, 0.1],
                  [0.7, 0.7, 0.7],
                  [0.7, 0.7, 0.7]])
    assert aggregate(S, strategy)[0] == 1


def test_vote_tie_first_winner_in_split_order():
    # c2 перемагає на розбиттях 0 і 3, c0 -- на 1 і 2: по 2 голоси;
    # c2 стала переможцем раніше (розбиття 0), хоча її індекс більший
    S = np.array([[0.1, 0.9, 0.9, 0.1],
                  [0.0, 0.0, 0.0, 0.0],
                  [0.9, 0.1, 0.1, 0.9]])
    best, votes = aggregate(S, "vote")
    assert best == 2
    np.testing.assert_array_equal(votes, [2, 0, 2])


def test_vote_within_split_tie_first_index():
    # на кожному розбитті c0 і c1 рівні -> голос отримує перша (c0)
    S = np.array([[0.5, 0.5], [0.5, 0.5], [0.1, 0.1]])
    best, votes = aggregate(S, "vote")
    assert best == 0
    np.testing.assert_array_equal(votes, [2, 0, 0])


def test_max_tie_earliest_split_not_lowest_index():
    # глобальний максимум 0.9 досягають c2 (розбиття 0) і c0 (розбиття 1);
    # як у Flow1_BaselineStrategy, перемагає переможець найранішого розбиття
    S = np.array([[0.1, 0.9],
                  [0.2, 0.2],
                  [0.9, 0.1]])
    assert aggregate(S, "max")[0] == 2


def test_rank_tie_lower_variance():
    # c0: ранги 1,3 (середній 2, дисперсія 1); c1: ранги 2,2 (дисперсія 0)
    S = np.array([[0.9, 0.1],
                  [0.5, 0.5],
                  [0.1, 0.9]])
    assert aggregate(S, "rank")[0] == 1


def test_rank_tie_better_mean_score():
    # c0 і c1: ранги {1,2} і {2,1} -> однакові середній ранг і дисперсія;
    # середнє значення метрики краще в c1 (0.75 проти 0.70)
    S = np.array([[0.8, 0.6],
                  [0.5, 1.0],
                  [0.0, 0.0]])
    assert aggregate(S, "rank")[0] == 1


def test_rank_uses_average_ranks_for_equal_scores():
    # розбиття 0: c0 і c1 рівні -> ранги 1.5, 1.5, 3
    S = np.array([[0.5, 0.1], [0.5, 0.9], [0.0, 0.5]])
    np.testing.assert_array_equal(aggregate(S, "rank")[1], [-2.25, -1.25, -2.5])


def test_rank_full_tie_first_index():
    S = np.array([[0.5, 0.5], [0.5, 0.5]])
    assert aggregate(S, "rank")[0] == 0


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_nan_never_wins(strategy):
    res = {"split0_test_score": np.array([np.nan, 0.1]),
           "split1_test_score": np.array([np.nan, 0.2])}
    M = split_score_matrix(res)
    assert np.isneginf(M[0]).all()
    best, agg = aggregate(M, strategy)
    assert best == 1
    assert agg[0] == -np.inf


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_partially_failed_config_never_wins(strategy):
    # c0 найкраща всюди, де навчилася, але одне навчання невдале
    S = np.array([[9.0, 9.0, np.nan, 9.0, 9.0],
                  [0.1, 0.2, 0.3, 0.2, 0.1],
                  [0.0, 0.1, 0.2, 0.1, 0.0]])
    best, agg = aggregate(S, strategy)
    assert best == 1
    assert agg[0] == -np.inf


def test_split_score_matrix_metric_and_errors():
    res = {"split0_test_mae": [1.0, 2.0], "split1_test_mae": [3.0, 4.0],
           "split0_test_rmse": [5.0, 6.0], "split1_test_rmse": [7.0, 8.0]}
    np.testing.assert_array_equal(split_score_matrix(res, "mae"), [[1, 3], [2, 4]])
    with pytest.raises(KeyError):
        split_score_matrix(res)                 # немає split*_test_score


def test_invalid_inputs():
    with pytest.raises(ValueError):
        aggregate(np.ones((2, 2)), "best")
    with pytest.raises(ValueError):
        aggregate(np.ones(3), "mean")
    with pytest.raises(ValueError):
        make_refit("best")


def test_make_refit_callable_is_picklable():
    refit = pickle.loads(pickle.dumps(make_refit("rank", "mae")))
    res = {"split0_test_mae": [0.1, 0.9], "split1_test_mae": [0.2, 0.8]}
    assert refit(res) == 1
