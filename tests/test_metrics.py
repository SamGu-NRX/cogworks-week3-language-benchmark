"""Hand-computed cases for the controller math."""

import numpy as np
import pytest

from language_search_benchmark.metrics import (
    chance_mrr,
    chance_mrr_first_relevant,
    mrr,
    mrr_with_misses,
    rank_matrix,
    ranks_of_gold,
    recall_at,
    search_ranks,
    text_first_relevant_ranks,
)


def test_rank_matrix_orders_by_score():
    scores = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.6]])
    order = rank_matrix(scores, seed=3)
    assert order[0].tolist() == [1, 2, 0]
    assert order[1].tolist() == [0, 2, 1]


def test_ranks_of_gold_hand_case():
    scores = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.6]])
    order = rank_matrix(scores, seed=3)
    ranks = ranks_of_gold(order, [0, 0])
    assert ranks.tolist() == [3, 1]
    assert mrr(ranks) == pytest.approx((1 / 3 + 1) / 2)
    assert recall_at(ranks, 1) == pytest.approx(0.5)


def test_tie_break_is_seeded_not_index_biased():
    scores = np.zeros((1, 100))
    order_a = rank_matrix(scores, seed=1)
    order_b = rank_matrix(scores, seed=1)
    order_c = rank_matrix(scores, seed=2)
    assert order_a.tolist() == order_b.tolist()
    assert order_a.tolist() != order_c.tolist()
    assert order_a[0].tolist() != list(range(100))


def test_chance_mrr_small_pool():
    # Pool of 3: E[1/rank] = (1 + 1/2 + 1/3)/3.
    assert chance_mrr(3) == pytest.approx((1 + 0.5 + 1 / 3) / 3)


def test_chance_mrr_first_relevant_matches_simulation():
    pool, relevant = 20, 4
    analytic = chance_mrr_first_relevant(pool, relevant)
    rng = np.random.RandomState(0)
    simulated = []
    for _ in range(20000):
        order = rng.permutation(pool)
        first = np.min(np.where(order < relevant)[0]) + 1
        simulated.append(1.0 / first)
    assert analytic == pytest.approx(np.mean(simulated), rel=0.02)


def test_text_first_relevant_ranks_hand_case():
    # Two groups of two; identical embeddings within a group.
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    ranks = text_first_relevant_ranks(embeddings, [0, 0, 1, 1], seed=5)
    assert ranks.tolist() == [1, 1, 1, 1]


def test_search_ranks_and_misses():
    rankings = [[7, 8, 9], [1, 2, 3], []]
    ranks, _ = search_ranks(rankings, gold_image_ids=[9, 5, 4], k=3, pool_size=10)
    assert ranks.tolist() == [3, 0, 0]
    assert mrr_with_misses(ranks) == pytest.approx((1 / 3) / 3)
