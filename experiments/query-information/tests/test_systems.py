"""System-level expectations derived by hand from the decoding semantics.

The exact-grammar system ranks candidates by multiset overlap with the
decoded (stopword-dropped, typo-corrected) query. Candidates compatible
under the frozen compat semantics carry the full required multiset, so
they achieve the maximum possible overlap score and every other candidate
scores strictly less. Therefore, for the exact-grammar system:

* the returned top-1 is always a compatible candidate (justified answers
  only), and
* the gold's rank never exceeds its compatible-set size.

The lexical system has no such guarantee -- idf weighting can promote an
incompatible candidate over a compatible one through norm differences --
so its coverage is measured and reported, not asserted here.
"""

import json
import os

from grammar import build_pool, rewrites_for_pool
import decoding
import numpy as np
import run as runner

ALL_RUNGS = runner.ALL_RUNGS


def test_exact_grammar_top1_always_compatible():
    pool = build_pool()
    scene_ids = [scene_id for scene_id, _ in pool]
    system = decoding.ExactGrammarSystem(pool, decoding.GrammarDecoder())
    for rung in ALL_RUNGS:
        for index, rewrite in enumerate(rewrites_for_pool(pool, rung)):
            rank_of_gold, _ = system.rank(rewrite)
            top1 = int(np.nonzero(rank_of_gold == 1)[0][0])
            compat = set(
                decoding.compatible_indices(
                    decoding.GrammarDecoder().decoded_tokens(rewrite), pool
                )
            )
            assert top1 in compat, (rung, index)


def test_exact_grammar_gold_rank_within_compat_size():
    pool = build_pool()
    system = decoding.ExactGrammarSystem(pool, decoding.GrammarDecoder())
    for rung in ALL_RUNGS:
        for index, rewrite in enumerate(rewrites_for_pool(pool, rung)):
            rank_of_gold, _ = system.rank(rewrite)
            size = len(
                decoding.compatible_indices(
                    decoding.GrammarDecoder().decoded_tokens(rewrite), pool
                )
            )
            assert int(rank_of_gold[index]) <= size, (rung, index)


def test_exact_grammar_reaches_the_typo_ceiling():
    # Every grammar-decoded typo query has a compatible set of size 1
    # (each corruption has a unique in-vocabulary correction), so the
    # identifiability ceiling is perfect MRR and the system must attain it.
    pool = build_pool()
    system = decoding.ExactGrammarSystem(pool, decoding.GrammarDecoder())
    ranks = [
        int(system.rank(rewrite)[0][index])
        for index, rewrite in enumerate(rewrites_for_pool(pool, "typo"))
    ]
    assert sum(1.0 / rank for rank in ranks) / len(ranks) == 1.0


def test_lexical_idf_is_deterministic():
    # Tie-breaking is seeded, not environmental: two fresh systems must
    # produce identical rankings for every rewrite on every rung.
    pool = build_pool()
    first = decoding.LexicalIdfSystem(pool)
    second = decoding.LexicalIdfSystem(pool)
    for rung in ALL_RUNGS:
        for rewrite in rewrites_for_pool(pool, rung):
            assert list(first.rank(rewrite)[0]) == list(second.rank(rewrite)[0])


def test_adapter_discovery_reports_empty_without_entry_points():
    discovered = decoding.discover_adapters()
    assert discovered["registered"] == []
