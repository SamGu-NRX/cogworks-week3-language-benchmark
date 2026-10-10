"""Hand-computed compatible-target sets (compat version qi-compat-1).

Every expected index set below was derived by hand from the frozen
candidate list, not produced by the code under test. The derivation for
each case is written next to it so a reviewer can redo it on paper.
"""

from grammar import build_pool, queries_for_pool, rewrites_for_pool
import decoding


def compat(rewrite, decoder):
    return set(decoding.compatible_indices(decoder.decoded_tokens(rewrite), build_pool()))


def test_verbatim_identifies_every_candidate():
    # Full descriptions are unique as multisets (no scene's token multiset
    # equals another's), so the verbatim query leaves exactly one candidate
    # compatible, for all 40 queries. Hand-checked on the mirror cases:
    # no (c1,s1)/(c2,s2) swap pair exists in FIRST/SECOND_SCENES.
    lexical = decoding.LexicalDecoder()
    queries = queries_for_pool(build_pool())
    for index, query in enumerate(queries):
        assert compat(query, lexical) == {index}


def test_query22_truncated_pair():
    # "red circle left" (scene-000, pool index 22): candidates carrying
    # red AND circle AND left -- scene-000 itself and scene-001 (pool 25,
    # "red circle left of green diamond"). Nothing else has both circle
    # and left with red.
    assert compat("red circle left", decoding.LexicalDecoder()) == {22, 25}


def test_query22_keywords_unique():
    # Full five-token multiset: only scene-000 carries
    # {red, circle, left, blue, square}; scene-002 lacks circle, scene-001
    # lacks blue and square.
    assert compat("red circle left blue square", decoding.LexicalDecoder()) == {22}


def test_query22_typo_lexical_pair_and_grammar_unique():
    # "cirxle" is out of vocabulary, dropped by the lexical decoder: the
    # constraint is {red, left, blue, square}, which scene-000 (pool 22)
    # and scene-002 (pool 31, "red square left of blue square") both
    # satisfy. The grammar decoder corrects cirxle -> circle (the unique
    # one-substitution vocabulary neighbor) and recovers uniqueness.
    assert compat("the red cirxle is left of the blue square", decoding.LexicalDecoder()) == {22, 31}
    assert compat("the red cirxle is left of the blue square", decoding.GrammarDecoder()) == {22}


def test_query0_typo_stays_unique_even_lexically():
    # "duamond" dropped leaves {red, circle, above, green}: only scene-009
    # (pool 0). The corrupted word was the second scene's shape, and no
    # other red-circle-above candidate exists -- corruption of a
    # non-distinguishing attribute removes nothing here.
    assert compat("the red circle is above the green duamond", decoding.LexicalDecoder()) == {0}
    assert compat("the red circle is above the green duamond", decoding.GrammarDecoder()) == {0}


def test_query4_near_is_deleted_by_the_stopword_list():
    # "near" is in perturb.STOPWORDS, so the keywords rewrite of query 4
    # (scene-038, pool 4) is "blue triangle blue square" with multiset
    # {blue, blue, triangle, square}: the five scenes holding two blues
    # and a triangle -- scene-038 (pool 4), scene-014 (pool 6), scene-030
    # (pool 18), scene-006 (pool 19), scene-022 (pool 34). The relation is
    # gone before truncation ever runs: information was removed by the
    # stopword list, not by length.
    assert compat("blue triangle blue square", decoding.LexicalDecoder()) == {4, 6, 18, 19, 34}
    # Truncated keeps the same first three content words ("near" never
    # survives content_words), so it inherits the same five-candidate set.
    assert compat("blue triangle blue", decoding.LexicalDecoder()) == {4, 6, 18, 19, 34}


def test_query4_typo_lexical_pair_grammar_unique():
    # "trianfle" dropped leaves {blue, blue, near, square}: scene-038
    # (pool 4) and scene-036 (pool 14, "blue circle near blue square") --
    # both hold two blues, near, and a square. Red-circle-near-blue-square
    # (pool 28) holds ONE blue and is excluded: multiplicity is the frozen
    # semantics. Grammar decoding corrects trianfle -> triangle.
    assert compat("the blue trianfle is near the blue square", decoding.LexicalDecoder()) == {4, 14}
    assert compat("the blue trianfle is near the blue square", decoding.GrammarDecoder()) == {4}


def test_query12_multiset_multiplicity_excludes_the_single_blue_twin():
    # Truncated "blue circle below" ({blue, circle, below}) matches
    # scene-020 (pool 12), scene-021 (pool 15, green diamond second), AND
    # scene-016 (pool 13, "red circle below blue square" -- one blue is
    # enough for a single-blue constraint).
    assert compat("blue circle below", decoding.LexicalDecoder()) == {12, 13, 15}
    # But the typo rewrite keeps both blues ("the blue curcle is below the
    # blue square" minus "curcle" = {blue, blue, below, square}). Two
    # scenes carry that multiset: scene-020 and scene-022 (pool 34, "blue
    # triangle below blue square") -- the corrupted word was the shape
    # token that told them apart, and scene-016 (one blue) is still
    # excluded by multiplicity.
    assert compat("the blue curcle is below the blue square", decoding.LexicalDecoder()) == {12, 34}
    assert compat("the blue curcle is below the blue square", decoding.GrammarDecoder()) == {12}
    # Verbatim and keywords carry the same multiset and are also unique.
    assert compat("blue circle below blue square", decoding.LexicalDecoder()) == {12}
    assert compat(queries_for_pool(build_pool())[12], decoding.LexicalDecoder()) == {12}


def test_query35_truncated_collapses_ten_candidates():
    # "red square blue" ({red, square, blue}): "near" was stopword-deleted
    # AND "square" no longer constrains the first scene, because every
    # red scene with a blue square second scene carries the token square.
    # All ten: pools 11, 13, 20, 21, 22, 28, 30, 31, 35, 37.
    assert compat("red square blue", decoding.LexicalDecoder()) == {
        11, 13, 20, 21, 22, 28, 30, 31, 35, 37,
    }
    # The typo rewrite drops "squaee" instead and keeps "near": constraint
    # {red, near, blue, square} -> scene-032 (pool 28) and the gold
    # scene-034 (pool 35). Grammar correction restores uniqueness.
    assert compat("the red squaee is near the blue square", decoding.LexicalDecoder()) == {28, 35}
    assert compat("the red squaee is near the blue square", decoding.GrammarDecoder()) == {35}


def test_typo_corrections_are_unique_in_vocabulary():
    # Every corrupted token the real transform produces on this pool has
    # exactly one one-substitution neighbor in the attribute vocabulary.
    # Template words (the/is/of) are not attribute vocabulary and are
    # correctly uncorrectable.
    grammar_decoder = decoding.GrammarDecoder()
    pool = build_pool()
    for rewrite in rewrites_for_pool(pool, "typo"):
        grammar_decoder.decoded_tokens(rewrite)
    log = grammar_decoder.correction_log()
    for token, count in log.items():
        if token in ("the", "is", "of"):
            assert count == 0
        else:
            assert count == 1, token
