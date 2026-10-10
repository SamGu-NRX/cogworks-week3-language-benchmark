"""Validation-module tests: reference implementations on hand cases.

The reference side re-derives every published metric from its definition
(pure stdlib+numpy); these tests pin it with hand-computed values and check
agreement with the published implementations on small inputs. The heavy
sweeps live in ``python -m study.validate_metrics``; the smoke-scale run is
exercised end-to-end on ``tmp_path`` so nothing in this suite writes into
``study/``.
"""

from __future__ import annotations

import json
from fractions import Fraction

import numpy as np
import pytest

from language_search_benchmark import metrics as published
from study import validate_metrics as vm


def _bits(value: float) -> bytes:
    return np.float64(value).tobytes()


def test_pairwise_replica_matches_numpy_bit_for_bit():
    """The replica reproduces np.sum's exact operation order, adversarial
    magnitudes included: the algorithm is value-independent."""
    rng = np.random.RandomState(0)
    for n in [1, 2, 7, 8, 9, 15, 128, 129, 200, 344, 700, 1024]:
        values = [1.0 / int(v) for v in rng.randint(1, 50, size=n)]
        assert _bits(vm.reference_pairwise_sum(values)) == _bits(float(np.sum(values)))
    spikes = [1e16 if i % 2 == 0 else 1e-16 for i in range(129)]
    assert _bits(vm.reference_pairwise_sum(spikes)) == _bits(float(np.sum(spikes)))


def test_reference_mrr_hand_case():
    ranks = [1, 2, 3]
    hand = (1.0 + 0.5 + 1.0 / 3.0) / 3.0
    assert vm.reference_mrr(ranks) == hand
    assert _bits(vm.reference_mrr(ranks)) == _bits(published.mrr(np.asarray(ranks)))
    assert vm.reference_mrr([]) == 0.0


def test_reference_mrr_bit_equals_published_on_random_ranks():
    rng = np.random.RandomState(1)
    for n in [1, 7, 8, 20, 150, 700]:
        ranks = [int(v) for v in rng.randint(1, 700, size=n)]
        assert _bits(vm.reference_mrr(ranks)) == _bits(
            published.mrr(np.asarray(ranks, dtype=np.int64))
        )


def test_reference_recall_and_median_hand_cases():
    assert vm.reference_recall_at([1, 3], 1) == 0.5
    assert vm.reference_median_rank([1, 3]) == 2.0
    assert vm.reference_median_rank([1, 2, 700]) == 2.0
    ranks = np.asarray([1, 3, 7, 700], dtype=np.int64)
    assert _bits(vm.reference_recall_at([1, 3, 7, 700], 10)) == _bits(
        published.recall_at(ranks, 10)
    )
    assert _bits(vm.reference_median_rank([1, 3, 7, 700])) == _bits(
        published.median_rank(ranks)
    )


def test_reference_mrr_with_misses_hand_case():
    ranks = [3, 1, 0, 3]
    hand = (1.0 / 3.0 + 1.0 + 0.0 + 1.0 / 3.0) / 4.0
    assert vm.reference_mrr_with_misses(ranks) == hand
    assert _bits(vm.reference_mrr_with_misses(ranks)) == _bits(
        published.mrr_with_misses(np.asarray(ranks, dtype=np.int64))
    )


def test_reference_rank_matrix_hand_case():
    scores = np.array([[0.1, 0.9, 0.5], [0.7, 0.2, 0.6]])
    order = vm.reference_rank_matrix(scores, seed=3)
    assert order[0].tolist() == [1, 2, 0]
    assert order[1].tolist() == [0, 2, 1]
    assert np.array_equal(order, published.rank_matrix(scores, seed=3))


def test_reference_rank_matrix_all_ties_follow_seeded_permutation():
    """On an all-tie row the order IS the tie-break permutation's order --
    ties resolve to the seeded shuffle, not to index favoritism."""
    scores = np.zeros((2, 12))
    order = vm.reference_rank_matrix(scores, seed=9)
    perm = vm.reference_tie_break_permutation(12, 9)
    expected = np.argsort(perm)
    assert order[0].tolist() == expected.tolist()
    assert order[1].tolist() == expected.tolist()


def test_reference_rank_matrix_matches_published_on_ties_neg_zero_and_inf():
    rng = np.random.RandomState(2)
    mixed = np.zeros((6, 15))
    mixed[::2, ::3] = -0.0
    mixed[1::3, ::5] = 0.0
    finite = rng.rand(7, 7)
    np.fill_diagonal(finite, -np.inf)
    tied = np.zeros((6, 6))
    np.fill_diagonal(tied, -np.inf)
    for scores in (
        rng.randint(0, 3, size=(10, 20)).astype(np.float64),
        mixed,
        finite,
        tied,
    ):
        for seed in (0, 1, 20261009):
            assert np.array_equal(
                published.rank_matrix(scores, seed), vm.reference_rank_matrix(scores, seed)
            )


def test_reference_ranks_of_gold_hand_case():
    order = np.array([[1, 2, 0], [0, 2, 1]])
    assert vm.reference_ranks_of_gold(order, [0, 0]).tolist() == [3, 1]
    assert np.array_equal(
        vm.reference_ranks_of_gold(order, [0, 0]),
        published.ranks_of_gold(order, [0, 0]),
    )


def test_exact_floor_fractions_hand_cases():
    assert vm.reference_chance_mrr_at_k_exact(3, 3) == Fraction(11, 18)
    assert vm.reference_chance_mrr_first_relevant_exact(4, 2) == Fraction(13, 18)
    assert vm.reference_chance_mrr_first_relevant_exact(4, 1) == Fraction(25, 48)
    # The weighting the published docstring derives: (3*(13/18) + 2*(25/48))/5.
    assert vm.reference_text_chance_floor_exact([0, 0, 0, 1, 1]) == Fraction(77, 120)
    assert vm.reference_text_chance_floor_exact([0, 1, 2]) == Fraction(0)
    assert vm.reference_text_chance_floor_exact([]) == Fraction(0)


def test_text_chance_floor_uneven_6_2_2_2_matches_exact_to_noise():
    """The locked case: published iterative product vs the exact rational."""
    groups = [0] * 6 + [1, 1] + [2, 2] + [3, 3]
    exact = vm.reference_text_chance_floor_exact(groups)
    assert float(exact) == pytest.approx(0.4717471561627406)
    got = published.text_chance_floor(groups)
    assert vm._relative_error(got, float(exact)) <= vm.NOISE_TOL


def test_reference_search_ranks_hand_case():
    rankings = [[3, 7, 1], [5], [], [9, 8, 7, 6]]
    gold = [1, 5, 4, 7]
    ranks, foreign = vm.reference_search_ranks(rankings, gold, 3, pool=list(range(10)))
    assert ranks.tolist() == [3, 1, 0, 3]
    assert foreign == 0
    # Gold past depth k is a miss; the foreign count covers whole ranking
    # rows, not just the first k returns.
    ranks_shallow, foreign_shallow = vm.reference_search_ranks(
        rankings, gold, 2, pool=[0, 1, 2, 9]
    )
    assert ranks_shallow.tolist() == [0, 1, 0, 0]
    assert foreign_shallow == 6
    pub_ranks, pub_foreign = published.search_ranks(rankings, gold, 3, list(range(10)))
    assert np.array_equal(pub_ranks, ranks)
    assert pub_foreign == foreign
    pub_ranks_shallow, pub_foreign_shallow = published.search_ranks(
        rankings, gold, 2, [0, 1, 2, 9]
    )
    assert np.array_equal(pub_ranks_shallow, ranks_shallow)
    assert pub_foreign_shallow == foreign_shallow


def test_reference_text_first_relevant_ranks_matches_published():
    rng = np.random.RandomState(3)
    for trial in range(5):
        groups = [int(v) for v in rng.randint(0, 4, size=15)]
        embeddings = vm.l2_rows(rng.randn(15, 20))
        seed = 9 + trial
        assert np.array_equal(
            published.text_first_relevant_ranks(embeddings, groups, seed),
            vm.reference_text_first_relevant_ranks(embeddings, groups, seed),
        )


def test_reference_l2_normalize_rows_definition():
    matrix = np.array([[3.0, 4.0], [0.0, 0.0]])
    normalized = vm.reference_l2_normalize_rows(matrix)
    assert normalized[0].tolist() == pytest.approx([0.6, 0.8])
    assert normalized[1].tolist() == [0.0, 0.0]  # zero row stays zero
    assert vm.l2_rows(matrix)[0].tolist() == pytest.approx([0.6, 0.8])


def test_rounding_flip_counterexample_flips_rank():
    """The constructible near-tie: raw order correct, 5-dp pipeline inverted,
    verified through the driver's own normalization path."""
    counter = vm.build_rounding_flip_counterexample()
    assert counter["flips"]
    assert counter["raw_cosines"][0] > counter["raw_cosines"][1]
    assert counter["rounded_scores"][0] < counter["rounded_scores"][1]
    assert counter["gold_rank_raw"] == 1
    assert counter["gold_rank_rounded"] == 2
    assert counter["raw_gap"] < 1e-8  # a near-tie, not a real margin

    # Same flip through the published pipeline verbatim (checks.l2_normalize_rows
    # + np.round), so the counterexample does not depend on this module's
    # pipeline replica being bit-faithful.
    from language_search_benchmark.checks import l2_normalize_rows

    query, images = counter["query"], counter["images"]
    driver_scores = np.round(l2_normalize_rows(query), 5) @ np.round(
        l2_normalize_rows(images), 5
    ).T
    raw_scores = l2_normalize_rows(query) @ l2_normalize_rows(images).T
    assert raw_scores[0, 0] > raw_scores[0, 1]
    assert driver_scores[0, 0] < driver_scores[0, 1]


def test_run_validation_smoke_all_families_pass(tmp_path):
    output = tmp_path / "validation_results.json"
    results = vm.run_validation(scale="smoke", output_path=str(output))
    assert results["summary"]["overall"] == "PASS"
    assert results["summary"]["families_total"] == 7
    assert set(results["families"]) == {
        "mrr_summation",
        "permutation",
        "ties",
        "floors",
        "rounding",
        "truncation",
        "precision",
    }
    assert all(
        family["status"] == "PASS" and family["gate"] and family["measurements"]
        for family in results["families"].values()
    )
    with open(output, encoding="utf-8") as handle:
        persisted = json.load(handle)
    assert persisted["summary"]["overall"] == "PASS"
    assert persisted["discrepancies"]  # findings recorded even on a clean pass


def test_pairwise_replica_bit_equals_published_mean_at_pool_sizes():
    """End-to-end for the scope-1 finding: the reference MRR replica is
    bit-identical to the published np.mean-based mrr at realistic sizes."""
    rng = np.random.RandomState(20261009)
    for n in [20, 150, 700]:
        ranks = [int(v) for v in rng.randint(1, 700, size=n)]
        assert _bits(vm.reference_mrr(ranks)) == _bits(
            published.mrr(np.asarray(ranks, dtype=np.int64))
        )
