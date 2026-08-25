"""A submission that embeds nothing must score at its floor, component by
component.

Week 3 hands the submission the full COCO annotations file, because the
assignment is to embed those captions and the course teaches building an IDF
table over the whole corpus. That file is also the caption-to-image mapping,
400,172 pairs on the real artifact, so a dictionary built from it answers any
query that is a verbatim caption.

Before scorer version retrieval-v4, every scored verbatim query was exactly
that. Measured against the real scorer on the test tier, a submission with no
embedding at all scored:

    retrieval_mrr  1.0000
    overall        0.4822        against a reference at 0.6925

Stripping gold from the payload does not help, and it was already stripped:
the sandbox recomputes the gold row as the position of the caption's image
within `pool_image_ids`, and both of those are things the contract requires.
The fix was to stop asking questions whose answers are strings in the file,
so the scored aggregates now average the rewritten rungs and the verbatim
probe is reported beside them.

This file is the standing guard on that. It exists because a future change to
how probes are built could reopen the door and nothing else would notice.

One rule about the assertion: compare each component against ITS OWN floor.
`chance_mrr` is the retrieval floor over the descriptor pool; the text
component groups a different number of captions and publishes `text_chance`;
search returns only k ids and publishes `search_chance`. Comparing `text_mrr`
against `chance_mrr` reports a submission sitting exactly on its floor as 3.4
times chance, which is how this investigation nearly concluded that the text
component was leaking when it was not.
"""

from __future__ import annotations

import pytest

from language_search_benchmark.drivers import run_cases
from language_search_benchmark.metrics import component_scores

from language_search_benchmark.datasets import SEARCH_K

from .fixtures.synthetic import (
    PerfectAdapter,
    PureMemorizerAdapter,
    Universe,
)

#: How far above its own floor a scored component may sit for a submission
#: that has no embedding. Not zero: a lookup that answers nothing still lands
#: wherever an arbitrary ordering lands, which is around the floor and can be
#: a little above it by luck of the tie break. 1.2 is loose enough not to be
#: flaky on a small synthetic pool and far below the 19x that the contaminated
#: retrieval component reached.
AT_FLOOR = 1.2


@pytest.fixture(scope="module")
def universe():
    return Universe()


@pytest.fixture(scope="module")
def cases(universe):
    return universe.cases()


@pytest.fixture(scope="module")
def null_metrics(universe, cases):
    outputs = run_cases(lambda resources: PureMemorizerAdapter(universe), None, cases)
    metrics, _ = component_scores(outputs, cases, SEARCH_K)
    return metrics


@pytest.fixture(scope="module")
def reference_metrics(universe, cases):
    outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
    metrics, _ = component_scores(outputs, cases, SEARCH_K)
    return metrics


def test_retrieval_is_at_its_floor(null_metrics):
    """The component that was at 1.0000 against the real artifacts.

    This fixture reaches it a weaker way than the real exploit does, and the
    reason is worth writing down. The driver embeds the retrieval queries
    before it calls `prepare_database`, and `RetrievalCase` carries no image
    ids, so a submission cannot learn the pool row order in time from this
    synthetic universe. Against the real benchmark it can: the descriptor
    matrix is shared with the search case, whose `image_ids` name the same
    rows, and both are pinned artifacts a submission may inspect ahead of
    time. That is how the measured 1.0000 was reached.

    So this assertion is the weaker of the two guards. The strong one is the
    verbatim-versus-scored gap below, which holds in both settings.
    """

    assert null_metrics["retrieval_mrr"] <= null_metrics["chance_mrr"] * AT_FLOOR


def test_search_does_not_keep_the_verbatim_rung(null_metrics):
    """Asserted as a drop rather than as a floor, and the synthetic universe
    is why.

    Its pool is 30 images with k of 10, so a submission that falls back to
    pool order lands gold inside the top ten often enough to sit at 2.2 times
    the floor by luck alone. That is a fact about a fixture small enough to
    run in a second, not about the scoring: on the real test tier the pool is
    100 with the same k, and the same null sits at 0.62 times its floor.

    What holds on both is the shape. This submission answers the verbatim
    rung perfectly and the rewritten ones no better than an arbitrary
    ordering, so the scored number has to be far below the reported one.
    """

    assert null_metrics["search_mrr_verbatim"] == pytest.approx(1.0)
    assert null_metrics["search_mrr"] < null_metrics["search_mrr_verbatim"] * 0.5


def test_text_is_exploitable_on_a_grouping_that_is_by_image(null_metrics):
    """A finding rather than a guard, and it needs stating plainly.

    On the real benchmark this null sits at its floor, 0.1757 against a
    `text_chance` of 0.1721, because the text component groups captions by
    cosine similarity between their embeddings and a lookup has nothing to
    look up.

    In this synthetic universe it scores 1.0000, because the universe's
    grouping is exactly "captions of the same image", which is precisely the
    relation the annotations file hands over. That is not a defect in the
    scoring; it is a property of a fixture built so that a perfect adapter
    can recover the answer.

    Recorded because it marks the boundary of the rule. A text probe whose
    groups ARE the payload's own structure is contaminated the same way the
    verbatim queries were, and if the real grouping ever moves toward that
    shape, this is the shape it would move toward.
    """

    assert null_metrics["text_mrr"] >= null_metrics["text_chance"]


def test_the_verbatim_probes_still_report_what_the_lookup_can_do(null_metrics):
    """Reported, not scored. This is the memorization dial, and it is the
    clearest measurement the instrument has: an honest submission scores
    about the same on both, and this one does not."""

    # Search is the component this fixture can exploit; see the retrieval
    # test above for why it cannot reach the other one here.
    assert null_metrics["search_mrr_verbatim"] == pytest.approx(1.0)
    assert null_metrics["search_mrr_verbatim"] > null_metrics["search_mrr"]
    # Both verbatim probes are published, whatever they score, because the
    # gap between reported and scored is the reading.
    assert "retrieval_mrr_verbatim" in null_metrics


def test_an_honest_submission_is_not_punished(reference_metrics, null_metrics):
    """The control. Decontamination has to cost a real submission almost
    nothing, or it changed what the instrument measures rather than what it
    can be fooled by.

    Measured on the real reference and the real artifacts: overall fell 4.8%
    on the test tier and 3.4% on evaluation, while this null fell 83%.
    """

    assert reference_metrics["overall"] > null_metrics["overall"]
    # Retrieval and search are the components decontamination protects. Text
    # is excluded here for the reason the text test above records: this
    # fixture's grouping is by image, so the null solves it exactly and a
    # perfect adapter can only match it.
    for component in ("retrieval_mrr", "search_mrr"):
        assert reference_metrics[component] > null_metrics[component]
