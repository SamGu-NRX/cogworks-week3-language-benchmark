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

    def test_a_retrieval_grid_missing_its_rewrites_refuses_too(self, universe, cases):
        """The half of this that was missing, and what it cost.

        `retrieval_mrr` is the mean of the three rewrites. With none of them
        present the mean was simply not taken and the verbatim score stayed,
        which is the memorization probe: a submission that embedded nothing
        scored 1.0000 there. The hosted sandbox rebuilt only the search
        rewrites for exactly this reason, so an official run would have
        published the probe under a scorer version claiming a three-rewrite
        average. Nothing raised.
        """

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        keep = [
            (output, case)
            for output, case in zip(outputs, cases)
            if not (case.kind == "retrieval" and getattr(case, "rung", "verbatim") != "verbatim")
        ]
        with pytest.raises(ValueError, match="retrieval grid is incomplete"):
            component_scores([o for o, _ in keep], [c for _, c in keep], SEARCH_K)

    @pytest.mark.parametrize("extra_output", [False, True])
    def test_output_count_mismatch_refuses(self, universe, cases, extra_output):
        # A text-only case has no rung guard to mask zip's silent truncation.
        text_cases = [case for case in cases if case.kind == "text"]
        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, text_cases)
        mismatched = outputs + outputs if extra_output else []
        with pytest.raises(ValueError, match="outputs for"):
            component_scores(mismatched, text_cases, SEARCH_K)

    def test_a_retrieval_grid_with_no_verbatim_case_refuses(self, universe, cases):
        """The guard used to read a dict seeded with a verbatim entry.

        With the verbatim case absent but its rewrites present, the seed made
        the grid look complete, `retrieval_mrr` averaged the rewrites, and
        `retrieval_mrr_verbatim` was published as the seeded 0.0, a number no
        component produced.
        """

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        keep = [
            (output, case)
            for output, case in zip(outputs, cases)
            if not (case.kind == "retrieval" and getattr(case, "rung", "verbatim") == "verbatim")
        ]
        with pytest.raises(ValueError, match="retrieval grid is incomplete"):
            component_scores([o for o, _ in keep], [c for _, c in keep], SEARCH_K)

    @pytest.mark.parametrize("kind", ["retrieval", "search"])
    @pytest.mark.parametrize("rung", perturb.RUNGS)
    def test_a_repeated_rung_refuses(self, universe, cases, kind, rung):
        """Neither the score dict nor the by-kind dict preserves duplicates."""
        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        paired = list(zip(outputs, cases))
        extra = next(
            (o, c) for o, c in paired
            if c.kind == kind and getattr(c, "rung", "verbatim") == rung
        )
        with pytest.raises(ValueError, match=kind + " grid is incomplete"):
            component_scores(
                [o for o, _ in paired] + [extra[0]],
                [c for _, c in paired] + [extra[1]],
                SEARCH_K,
            )

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

    def test_each_floor_is_published_beside_the_number_it_scales(
        self, universe, cases
    ):
        """Three floors, three distinct values, each typed and related.

        They were also written into a diagnostic, which printed the same
        three numbers a second time and made a list of floors the headline
        of a successful run.
        """

        benchmark = LanguageSearchBenchmark()
        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        metrics = benchmark.score(outputs, cases)

        # Which floor belongs to which number is the part a reader cannot
        # guess; the two tests above own the claim that the values differ.
        floors = {
            "text_chance": "text_mrr",
            "chance_mrr": "retrieval_mrr",
            "search_chance": "search_mrr",
        }
        for floor, scales in floors.items():
            assert floor in metrics
            assert benchmark.metric_roles[floor] == "floor"
            assert benchmark.metric_relations[floor] == scales
        assert not any("chance baselines" in note for note in benchmark.last_diagnostics)

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

    def test_every_metric_declares_what_kind_of_number_it_is(self, universe, cases):
        """The run page reads `metric_roles` and never a metric's name.

        A metric with no role renders as a scored row with an arrow saying
        which direction is better, which is right for a score and wrong for
        the other three kinds this benchmark publishes.
        """

        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        benchmark = LanguageSearchBenchmark()
        returned = set(benchmark.score(outputs, cases))

        assert sorted(returned - set(benchmark.metric_roles)) == []
        assert sorted(set(benchmark.metric_roles) - returned) == []

    def test_a_floor_and_a_probe_name_what_they_belong_beside(self):
        """A floor is the scale of one metric and a probe shadows one score.
        Neither means anything alone, so both must name their partner or the
        page has nowhere to put them."""

        benchmark = LanguageSearchBenchmark()
        for key, role in benchmark.metric_roles.items():
            if role in ("floor", "reported"):
                assert key in benchmark.metric_relations, key
                # And the thing it points at has to be a real metric.
                assert benchmark.metric_relations[key] in benchmark.metric_roles

    def test_the_curve_declares_the_metric_it_plots(self, universe, cases):
        """The points are `search_mrr_{rung}`. A runner with no way to ask
        fell back to `primary_metric` and labelled four search scores
        "overall" (run 1772)."""

        benchmark = LanguageSearchBenchmark()
        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        metrics = benchmark.score(outputs, cases)

        assert benchmark.sweep_metric == "search_mrr"
        assert benchmark.sweep_metric in benchmark.metric_labels
        plotted = {
            point[benchmark.sweep_label_key]: point[benchmark.sweep_y_key]
            for point in benchmark.last_sweep
        }
        assert plotted == {
            rung: metrics["{}_{}".format(benchmark.sweep_metric, rung)]
            for rung in perturb.RUNGS
        }

    def test_nothing_is_both_plotted_and_tabled(self, universe, cases):
        """A rung is read off the curve, where its exact value is printed
        beside its point. A row for it would be the same number twice."""

        benchmark = LanguageSearchBenchmark()
        outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
        benchmark.score(outputs, cases)
        plotted = {k for k, role in benchmark.metric_roles.items() if role == "plotted"}
        on_the_curve = {point["rung"] for point in benchmark.last_sweep}
        # Every plotted metric is one of the rungs the curve draws.
        for key in plotted:
            assert key.replace("search_mrr_", "") in on_the_curve, key
