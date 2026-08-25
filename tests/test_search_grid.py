"""``search_mrr`` measures something ``retrieval_mrr`` does not.

Before scorer version retrieval-v3, ``search_mrr`` scored the verbatim rung
alone. That gave it the same queries and the same pool as ``retrieval_mrr``,
so the only way the two could differ on a working submission was a query whose
correct image fell past the k-th search result. Measured on the reference
submission (2026-08-20, this working tree, the pinned artifacts):

    test tier        retrieval 0.63365079365079369
                     search    0.63365079365079369   equal under ``==``
    evaluation tier  retrieval 0.25860313692112463
                     search    0.25723363690130252   gap 0.00137

The whole evaluation-tier gap was traced to 15 of 150 queries whose correct
image sat beyond rank 50, which is the search payload's depth cap. Top-1
agreement was 150 of 150. Two metrics on a run page that agree to three
decimal places are one metric printed twice, and the one number that did
differ was an artifact of the cap rather than anything about the submission.

``search_mrr`` now averages the whole rewrite grid, so the two ask different
questions: is the embedding space aligned, and does end-to-end search survive
the queries a person types. These tests hold that separation in place.
"""

from __future__ import annotations

import pytest

from language_search_benchmark import perturb
from language_search_benchmark.datasets import SEARCH_K, TextCase
from language_search_benchmark.drivers import run_cases
from language_search_benchmark.metrics import (
    chance_mrr,
    chance_mrr_at_k,
    component_scores,
    text_chance_floor,
)
from language_search_benchmark.plugins import LanguageSearchBenchmark

from .fixtures.synthetic import (
    MemorizedCaptionAdapter,
    PerfectAdapter,
    Universe,
)

#: How far apart the two metrics must sit before we call them separate.
#:
#: This is a floor chosen to be well clear of the 0.00137 that motivated the
#: change, not a calibrated value: no measurement says a correct separation
#: has any particular size. It is roughly 35x the old gap and roughly a
#: fourteenth of the 0.53 the memorized-caption submission below actually
#: produces, so it fails loudly if the grid stops being averaged and stays
#: quiet under ordinary movement in the fixture.
SEPARATION_FLOOR = 0.05


@pytest.fixture(scope="module")
def universe():
    return Universe()


@pytest.fixture(scope="module")
def cases(universe):
    return universe.cases()


def _score(outputs, cases):
    benchmark = LanguageSearchBenchmark()
    return benchmark.score(outputs, cases), benchmark.last_diagnostics


class TestTheTwoMetricsAreSeparate:
    def test_a_caption_memorizer_splits_them(self, universe, cases):
        """Right embeddings, a search path that only knows exact captions.

        Both towers are the real ones, so retrieval is untouched. Search
        answers a caption it has seen and falls back to pool order on
        anything else, which is precisely what the rewrites are for.
        """

        outputs = run_cases(lambda resources: MemorizedCaptionAdapter(universe), None, cases)
        metrics, _ = _score(outputs, cases)

        assert metrics["retrieval_mrr"] == pytest.approx(1.0)
        # The verbatim rung alone still cannot see the problem. This is the
        # number retrieval-v2 published as `search_mrr`.
        assert metrics["search_mrr_verbatim"] == pytest.approx(1.0)
        assert metrics["retrieval_mrr"] - metrics["search_mrr"] > SEPARATION_FLOOR

    def test_search_mrr_is_the_mean_of_the_rewritten_rungs(self, universe, cases):
        """Stated as arithmetic so a change to the weighting has to be
        deliberate. Equal weight is not a claim that the rewrites matter
        equally; it is the absence of a claim, since no measurement supports
        any other set of weights.

        The verbatim rung is excluded, and that is the whole point of
        retrieval-v4. Its queries are captions read straight out of the
        annotations file the submission is handed, so a dictionary built from
        that file answers them: this adapter is that dictionary, and it scores
        near the top verbatim and near the floor everywhere else. Averaging
        the verbatim rung in gave it a quarter of the component for free."""

        outputs = run_cases(lambda resources: MemorizedCaptionAdapter(universe), None, cases)
        metrics, _ = _score(outputs, cases)
        rewritten = [
            metrics["search_mrr_{}".format(rung)]
            for rung in perturb.RUNGS
            if rung != "verbatim"
        ]
        assert len(rewritten) == 3
        assert metrics["search_mrr"] == pytest.approx(sum(rewritten) / 3.0)
        # Reported, so the memorization signature stays readable.
        assert "search_mrr_verbatim" in metrics
        assert metrics["search_mrr_verbatim"] > metrics["search_mrr"]

    def test_a_submission_that_survives_the_rewrites_keeps_its_score(self, universe, cases):
        """The separation must come from the submission, not from the metric.

        A correct submission scores the same on all four rungs, so averaging
        them costs it nothing. Otherwise `search_mrr` would be a penalty every
        team pays rather than a measurement.
        """

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        metrics, _ = _score(outputs, cases)
        assert metrics["search_mrr"] == pytest.approx(1.0)
        assert metrics["retrieval_mrr"] - metrics["search_mrr"] == pytest.approx(0.0)


class TestDegradationIsWhatMovesTheScore:
    def test_degrading_under_rewrites_scores_lower_than_holding_up(self, universe, cases):
        """The discriminating power the change was made for.

        Two submissions with identical embeddings and identical verbatim
        search. One holds up when the query is rewritten and one does not.
        Under retrieval-v2 they were the same on every published number.
        """

        holds_up, _ = _score(
            run_cases(lambda resources: PerfectAdapter(universe), None, cases), cases
        )
        degrades, _ = _score(
            run_cases(lambda resources: MemorizedCaptionAdapter(universe), None, cases), cases
        )

        # Identical everywhere retrieval-v2 could see.
        assert holds_up["text_mrr"] == degrades["text_mrr"]
        assert holds_up["retrieval_mrr"] == degrades["retrieval_mrr"]
        assert holds_up["search_mrr_verbatim"] == degrades["search_mrr_verbatim"]

        # And separated where retrieval-v3 can.
        assert degrades["search_mrr"] < holds_up["search_mrr"] - SEPARATION_FLOOR
        assert degrades["overall"] < holds_up["overall"]

    def test_the_diagnostics_name_the_rewrite_that_cost_the_score(self, universe, cases):
        """A number a team cannot act on is a number they will argue with.

        Each note compares a rung against the verbatim rung rather than
        against `search_mrr`, because `search_mrr` now contains the rung being
        described and would move with it.
        """

        outputs = run_cases(lambda resources: MemorizedCaptionAdapter(universe), None, cases)
        _metrics, diagnostics = _score(outputs, cases)
        assert any("unchanged captions" in note for note in diagnostics)


class TestAnIncompleteGridRefuses:
    def test_a_missing_rung_raises_rather_than_averaging_what_arrived(self, universe, cases):
        """A plausible wrong number is worse than a refusal.

        Dividing by four when three rungs arrived under-reports; dividing by
        three makes `search_mrr` mean something different from one run to the
        next with nothing on the page saying so. This can only happen if the
        case builders and `perturb.RUNGS` are edited apart, so it is a
        construction error and not anything a submission can trigger.
        """

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        short_cases = [case for case in cases if getattr(case, "rung", None) != "typo"]
        short_outputs = [
            output
            for output, case in zip(outputs, cases)
            if getattr(case, "rung", None) != "typo"
        ]
        with pytest.raises(ValueError, match="grid is incomplete"):
            component_scores(short_outputs, short_cases, SEARCH_K)

    def test_a_rung_that_ran_and_failed_is_not_a_missing_rung(self, universe, cases):
        """It scores 0 and drags the mean down, which is the honest reading:
        the submission was asked and could not answer."""

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        broken = []
        for output, case in zip(outputs, cases):
            if getattr(case, "rung", None) == "typo":
                broken.append({"ok": False, "kind": "search", "error": "raised on an unseen word"})
            else:
                broken.append(output)
        metrics, diagnostics = component_scores(broken, cases, SEARCH_K)
        assert metrics["search_mrr_typo"] == 0.0
        # One of the three scored rungs, not one of four: the verbatim rung
        # is reported and not scored since retrieval-v4, because its queries
        # are captions read out of the file the submission is handed.
        assert metrics["search_mrr"] == pytest.approx(2.0 / 3.0)
        assert any("typo query rung scored 0" in note for note in diagnostics)


class TestEveryMetricHasItsOwnFloor:
    def test_text_chance_is_published_and_is_not_the_image_floor(self, universe, cases):
        """`text_chance` was computed and then dropped into a diagnostic
        string. A student comparing `text_mrr` against `chance_mrr`, the only
        floor on the page, was comparing against the wrong one: measured on
        the reference, `text_chance` is 3.32x `chance_mrr` on the test tier
        and 3.93x on the evaluation tier."""

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        metrics, _ = _score(outputs, cases)
        text_case = next(case for case in cases if case.kind == "text")

        assert "text_chance" in metrics
        assert metrics["text_chance"] == text_chance_floor(text_case.group_rows)
        assert metrics["text_chance"] != metrics["chance_mrr"]

    def test_search_chance_sits_below_the_retrieval_floor(self, universe, cases):
        """Search returns k ids and scores anything past them as a miss, so
        its floor is the whole-pool floor truncated at k. On the evaluation
        pool of 700 with k 50 that is 0.006427 against 0.010184."""

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        metrics, _ = _score(outputs, cases)
        search_case = next(
            case
            for case in cases
            if case.kind == "search" and getattr(case, "rung", "verbatim") == "verbatim"
        )
        pool = len(search_case.image_ids)

        assert metrics["search_chance"] == chance_mrr_at_k(pool, SEARCH_K)
        assert metrics["search_chance"] <= metrics["chance_mrr"]

    def test_the_three_floors_are_reported_together(self, universe, cases):
        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        _metrics, diagnostics = _score(outputs, cases)
        note = next(n for n in diagnostics if n.startswith("chance baselines"))
        for name in ("text_mrr", "retrieval_mrr", "search_mrr"):
            assert name in note

    def test_the_whole_pool_floor_is_the_k_capped_floor_at_full_depth(self):
        """`chance_mrr` delegates to `chance_mrr_at_k`, so the two cannot
        drift apart. Bit-identical, since at full depth they evaluate the same
        expression; asserted with `==` rather than a tolerance because
        anything looser would let the published retrieval floor move."""

        for pool in (3, 10, 100, 700):
            assert chance_mrr(pool) == chance_mrr_at_k(pool, pool)
        # Asking for more than the pool holds is the same as asking for it all.
        assert chance_mrr_at_k(10, 999) == chance_mrr(10)


def test_scoring_a_text_only_case_list_still_works():
    """The text component is scored on its own by `test_metrics.py`, and the
    grid check must not turn that into a refusal: no search case means no
    grid to be incomplete."""

    case = TextCase(
        kind="text", captions=["a", "b", "c"], group_rows=[0, 0, 1], tie_break_seed=1
    )
    metrics, _ = component_scores(
        [{"ok": False, "kind": "text", "error": "unused"}], [case], SEARCH_K
    )
    assert metrics["search_mrr"] == 0.0
    assert metrics["search_chance"] == 0.0


class TestEveryNumberReachesThePageExplained:
    """A metric the scorer returns and the plugin never declares.

    `metric_labels` and `metric_help` are what the run page reads. The runner
    falls back to `key.replace("_", " ").title()` for anything unlabeled, so a
    metric added to the scorer alone arrives as "Retrieval Mrr Verbatim" with
    no explanation next to it.

    That happened: `retrieval_mrr_verbatim` was added in retrieval-v4 and no
    test noticed, because the shared explainability suite compares labels
    against help and a metric in neither is invisible to it. This closes that
    from the benchmark's own side, where the real scorer can be run.
    """

    def test_the_declared_keys_are_exactly_what_scoring_returns(self, universe, cases):
        """Both directions. A missing key is a number with no explanation; a
        stale one reads as current and describes something nobody sees."""

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        benchmark = LanguageSearchBenchmark()
        returned = set(benchmark.score(outputs, cases))
        declared = set(benchmark.sample_metric_keys)

        assert sorted(returned - declared) == [], "scored but never declared"
        assert sorted(declared - returned) == [], "declared but never scored"

    def test_every_returned_metric_has_a_label_and_an_explanation(self, universe, cases):
        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        benchmark = LanguageSearchBenchmark()
        returned = set(benchmark.score(outputs, cases))

        assert sorted(returned - set(benchmark.metric_labels)) == []
        assert sorted(returned - set(benchmark.metric_help)) == []

    def test_the_unscored_probes_say_so_in_their_label(self, universe, cases):
        """A student reading a table of sixteen numbers has no way to tell
        which three feed the score. These two do not, and the reason they
        exist is that they are the ones a lookup can answer."""

        benchmark = LanguageSearchBenchmark()
        for key in ("search_mrr_verbatim", "retrieval_mrr_verbatim"):
            assert "not scored" in benchmark.metric_labels[key]
