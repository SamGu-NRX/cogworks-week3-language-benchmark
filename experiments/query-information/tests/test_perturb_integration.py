"""The real perturb transforms, applied to the frozen pool, against strings
computed by hand from perturb.py's documented rules.

Every expected string below was derived by hand: stopwords from
``perturb.STOPWORDS`` (which contains ``near`` -- the study's central
discovery), truncation to the first three content words, and the typo
target from ``max(content, key=len)`` with the corruption character chosen
by ``index % len(usable)`` over QWERTY neighbours. If perturb.py changes,
these fail -- which is the point of freezing them.
"""

from grammar import POOL_SEED, build_pool, queries_for_pool, rewrites_for_pool


def pool():
    return build_pool()


def test_pool_seed_is_the_frozen_one():
    assert POOL_SEED == 20261010


def test_pool_order_is_frozen():
    # Hand-recorded from the seeded permutation; a change here moves every
    # typo, so it must be a decision, never an accident.
    expected = [
        "scene-009", "scene-013", "scene-011", "scene-017", "scene-038",
        "scene-003", "scene-014", "scene-029", "scene-025", "scene-015",
        "scene-033", "scene-024", "scene-020", "scene-016", "scene-036",
        "scene-021", "scene-023", "scene-007", "scene-030", "scene-006",
        "scene-018", "scene-010", "scene-000", "scene-039", "scene-019",
        "scene-001", "scene-035", "scene-005", "scene-032", "scene-004",
        "scene-008", "scene-002", "scene-027", "scene-012", "scene-022",
        "scene-034", "scene-037", "scene-026", "scene-028", "scene-031",
    ]
    assert [scene_id for scene_id, _ in pool()] == expected


def test_candidate_descriptions_are_unique():
    descriptions = [scene.description for _, scene in pool()]
    assert len(descriptions) == len(set(descriptions)) == 40


def test_verbatim_queries_use_the_frozen_template():
    queries = queries_for_pool(pool())
    assert queries[22] == "the red circle is left of the blue square"
    assert queries[0] == "the red circle is above the green diamond"
    assert queries[4] == "the blue triangle is near the blue square"
    assert queries[12] == "the blue circle is below the blue square"
    assert queries[35] == "the red square is near the blue square"


def test_keywords_rung_strings():
    # Hand-derived: drop STOPWORDS, keep order. Query 4 and 35 lose "near":
    # the real stopword list classifies the relation word itself as a
    # stopword, which no reading of the grammar would predict.
    rewrites = rewrites_for_pool(pool(), "keywords")
    assert rewrites[22] == "red circle left blue square"
    assert rewrites[0] == "red circle above green diamond"
    assert rewrites[4] == "blue triangle blue square"
    assert rewrites[12] == "blue circle below blue square"
    assert rewrites[35] == "red square blue square"


def test_truncated_rung_strings():
    rewrites = rewrites_for_pool(pool(), "truncated")
    assert rewrites[22] == "red circle left"
    assert rewrites[0] == "red circle above"
    assert rewrites[4] == "blue triangle blue"
    assert rewrites[12] == "blue circle below"
    assert rewrites[35] == "red square blue"


def test_typo_rung_strings():
    # Hand-derived corruption targets: the longest content word (first of
    # ties), interior characters only, character chosen by
    # index % len(usable). Query 0 corrupts "diamond" (7 > 6); query 4
    # corrupts "triangle" (8 > 6); queries 22 and 35 corrupt the s1 word,
    # which ties "square" and wins as the earlier list entry.
    rewrites = rewrites_for_pool(pool(), "typo")
    assert rewrites[0] == "the red circle is above the green duamond"
    assert rewrites[4] == "the blue trianfle is near the blue square"
    assert rewrites[12] == "the blue curcle is below the blue square"
    assert rewrites[22] == "the red cirxle is left of the blue square"
    assert rewrites[35] == "the red squaee is near the blue square"


def test_unknown_rung_raises():
    import pytest

    with pytest.raises(ValueError):
        rewrites_for_pool(pool(), "shuffled")
