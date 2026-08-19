"""Hand-computed cases for the controller math."""

import numpy as np
import pytest

from language_search_benchmark.datasets import SEARCH_K, TextCase
from language_search_benchmark.metrics import (
    chance_mrr,
    chance_mrr_first_relevant,
    component_scores,
    mrr,
    mrr_with_misses,
    rank_matrix,
    ranks_of_gold,
    recall_at,
    search_ranks,
    text_chance_floor,
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


def test_text_chance_floor_weights_uneven_groups_by_query():
    """Two uneven groups: the floor is the mean over queries, not the floor
    of the rounded mean group size.

    Groups of 3 and 2 (N=5): each of the 3 queries in the big group faces a
    pool of 4 with 2 relevant items, each of the 2 in the small group faces
    the same pool with 1 relevant. Hand value: (3*(13/18) + 2*(25/48))/5 =
    77/120. The old collapsed-mean floor rounded the mean group size 2.5 to
    2 and reported chance_mrr_first_relevant(4, 1), understating it.
    """

    group_rows = [0, 0, 0, 1, 1]
    expected = (
        3 * chance_mrr_first_relevant(4, 2) + 2 * chance_mrr_first_relevant(4, 1)
    ) / 5
    assert text_chance_floor(group_rows) == pytest.approx(expected)
    assert text_chance_floor(group_rows) == pytest.approx(77 / 120)
    assert text_chance_floor(group_rows) > chance_mrr_first_relevant(4, 1)


def test_text_chance_floor_is_bit_identical_on_uniform_groups():
    """Both shipped manifests are uniform size-5 groups, and published
    text_chance values must not move: on uniform groups the weighted floor
    must reproduce the single-size value exactly, not approximately."""

    for n_groups, size in ((15, 5), (100, 5), (12, 3)):
        group_rows = [group for group in range(n_groups) for _ in range(size)]
        pool = n_groups * size - 1
        assert repr(text_chance_floor(group_rows)) == repr(
            chance_mrr_first_relevant(pool, size - 1)
        )


def test_text_chance_floor_skips_singletons_like_the_rank_function():
    # A singleton has no relevant item; it enlarges the pool every query
    # faces but contributes no query of its own.
    assert text_chance_floor([0, 0, 1]) == chance_mrr_first_relevant(2, 1)
    assert text_chance_floor([0, 1, 2]) == 0.0
    assert text_chance_floor([]) == 0.0


def test_component_scores_reports_the_query_weighted_text_floor():
    """text_chance in the metrics dict comes from the weighted floor."""

    case = TextCase(
        kind="text",
        captions=["a", "b", "c", "d", "e"],
        group_rows=[0, 0, 0, 1, 1],
        tie_break_seed=1,
    )
    outputs = [{"ok": False, "kind": "text", "error": "unused"}]
    metrics, _ = component_scores(outputs, [case], SEARCH_K)
    assert metrics["text_chance"] == text_chance_floor([0, 0, 0, 1, 1])


def test_text_first_relevant_ranks_hand_case():
    # Two groups of two; identical embeddings within a group.
    embeddings = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    ranks = text_first_relevant_ranks(embeddings, [0, 0, 1, 1], seed=5)
    assert ranks.tolist() == [1, 1, 1, 1]


def test_search_ranks_and_misses():
    rankings = [[7, 8, 9], [1, 2, 3], []]
    ranks, foreign = search_ranks(
        rankings, gold_image_ids=[9, 5, 4], k=3, pool=[1, 2, 3, 4, 5, 7, 8, 9]
    )
    assert ranks.tolist() == [3, 0, 0]
    assert mrr_with_misses(ranks) == pytest.approx((1 / 3) / 3)
    # Every returned id is in the pool here, so nothing is foreign. The old
    # version of this test discarded the second value with `_`, which is how
    # it went unnoticed that the function returned the query count under a
    # docstring promising a foreign-id count.
    assert foreign == 0


def test_search_ranks_counts_ids_outside_the_pool():
    """The second return value is what the docstring says it is."""

    rankings = [[99, 1], [2, 100]]
    ranks, foreign = search_ranks(
        rankings, gold_image_ids=[1, 2], k=2, pool=[1, 2, 3]
    )
    assert ranks.tolist() == [2, 1]
    assert foreign == 2  # 99 and 100


def test_search_ranks_without_a_pool_reports_no_foreign_ids():
    """Omitting the pool means "do not check", not "everything is foreign"."""

    ranks, foreign = search_ranks([[5]], gold_image_ids=[5], k=1)
    assert ranks.tolist() == [1]
    assert foreign == 0
