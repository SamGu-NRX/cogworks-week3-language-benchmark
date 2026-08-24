"""What a submission that never embeds anything can score.

Week 3 hands the submission `Resources`, whose `captions_path` is the full
COCO annotations file. That file IS the caption-to-image mapping, so it can
be turned into a dictionary in one pass: 400,172 pairs on the real artifact.
A submission that builds that dictionary and looks up the query string
answers verbatim queries without embedding anything.

This is not a leak that can be closed by stripping the payload, the way
Weeks 1 and 2 strip theirs. The assignment IS to embed those captions, and
the course teaches building an IDF table over the whole corpus, so the
corpus has to be there. The gold and the training data are the same object.

What contains it is the query rewrite grid. `search_mrr` averages four
rungs, and only the first is a verbatim caption. Measured on the real test
tier against the real scorer:

    search_mrr_verbatim   0.9520     the shortcut works perfectly here
    search_mrr_keywords   0.0279     and collapses one rung later
    search_mrr            0.2590     the average it actually scores
    overall               0.1582     against a chance floor of 0.0519
                                     and a reference submission at 0.6925

So the shortcut scores near chance overall while producing a signature no
honest submission produces: near-perfect on exact captions, near-zero the
moment a word moves. That gap is the finding, and it is worth more than a
refusal would be, because it tells a team what their code is actually doing.

These numbers are asserted loosely. The point is the shape (verbatim high,
rewritten near chance, overall near the floor), not four exact decimals,
which move with the fixture.
"""

from __future__ import annotations

import numpy as np
import pytest

from language_search_benchmark.plugins import LanguageSearchBenchmark


def _lookup_outputs(cases, caption_to_image):
    """A submission that embeds nothing and answers from the map it was given."""

    outputs = []
    for case in cases:
        kind = getattr(case, "kind", "")
        if kind == "text":
            outputs.append(
                {
                    "ok": True,
                    "embeddings": np.ones((len(case.captions), 8), dtype=np.float32),
                    "groups": [[index] for index in range(len(case.captions))],
                }
            )
        elif kind == "retrieval":
            outputs.append(
                {
                    "ok": True,
                    "text": np.ones((len(case.queries), 8), dtype=np.float32),
                    "images": np.ones((case.descriptors.shape[0], 8), dtype=np.float32),
                }
            )
        elif kind == "search":
            pool = [int(value) for value in case.image_ids]
            rankings = []
            for query in case.queries:
                hit = caption_to_image.get(query.strip())
                if hit in pool:
                    rankings.append([hit] + [p for p in pool if p != hit][: case.k - 1])
                else:
                    rankings.append(pool[: case.k])
            outputs.append({"ok": True, "rankings": rankings})
    return outputs


@pytest.fixture(scope="module")
def scored():
    benchmark = LanguageSearchBenchmark()
    cases = list(benchmark.load_cases("test"))
    # The map the shortcut would build, taken from the cases themselves rather
    # than from the artifact on disk, so this test needs no download. It is
    # the same relationship: every scored query is a verbatim caption of its
    # gold image, which is exactly why the shortcut works.
    caption_to_image = {}
    for case in cases:
        if getattr(case, "kind", "") == "search" and getattr(case, "rung", "verbatim") == "verbatim":
            for query, gold in zip(case.queries, case.gold_image_ids or []):
                caption_to_image[query.strip()] = int(gold)
    outputs = _lookup_outputs(cases, caption_to_image)
    return benchmark.score(outputs, cases)


def test_the_shortcut_wins_on_verbatim_captions(scored):
    assert scored["search_mrr_verbatim"] > 0.9


def test_and_collapses_the_moment_a_word_moves(scored):
    assert scored["search_mrr_keywords"] < 0.1
    assert scored["search_mrr_truncated"] < 0.1
    assert scored["search_mrr_typo"] < 0.1


def test_so_the_component_it_inflates_lands_near_the_floor(scored):
    """The rewrite grid is what makes this true. Scoring the verbatim rung
    alone, which is what `search_mrr` was before retrieval-v3, would publish
    0.95 for a submission that did not embed anything."""

    assert scored["search_mrr"] < 0.35


def test_and_overall_stays_near_chance(scored):
    """A reference submission scores 0.6925 on this tier."""

    assert scored["overall"] < 0.25
    assert scored["overall"] > scored["chance_mrr"]


def test_the_gap_between_rungs_is_the_signature(scored):
    """No honest submission produces this shape. The course predicts that IDF
    weighting makes stopword removal nearly free, so a real embedding scores
    close to its verbatim number on the keywords rung."""

    gap = scored["search_mrr_verbatim"] - scored["search_mrr_keywords"]
    assert gap > 0.8
