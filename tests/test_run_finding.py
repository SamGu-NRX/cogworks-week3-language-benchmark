"""The sentence a clean run leads with.

The run page shows the first diagnostic as the run's headline. A run where
every case ran and nothing was withheld wrote none, so run_d11b5e5e2e (9/9
cases, overall 0.4126) led with the page's "the scorer didn't write a
sentence" fallback, and a run reached through course-style names would have
led with an adapter mapping instead.
"""

import pytest

from language_search_benchmark.datasets import SEARCH_K
from language_search_benchmark.drivers import run_cases
from language_search_benchmark.metrics import component_scores
from language_search_benchmark.plugins import LanguageSearchBenchmark

from .fixtures.synthetic import CourseStyleAdapter, PerfectAdapter, Universe

FINDING = "On the same rewritten queries"


class SecondPlaceSearch(PerfectAdapter):
    """Right embeddings, a search that puts the right image second.

    Retrieval scores 1 and every search rung, verbatim included, scores 0.5,
    so the two numbers differ and no rewrite moves against the unchanged
    captions, which would add a note of its own.
    """

    def search(self, query, k):
        ranked = super().search(query, k)
        return ranked[1:2] + ranked[:1] + ranked[2:]


@pytest.fixture(scope="module")
def universe():
    return Universe()


@pytest.fixture(scope="module")
def cases(universe):
    return universe.cases()


def _score(adapter, cases):
    outputs = run_cases(lambda resources: adapter, None, cases)
    benchmark = LanguageSearchBenchmark()
    return outputs, benchmark.score(outputs, cases), benchmark.last_diagnostics


def test_a_clean_run_leads_with_search_against_retrieval(universe, cases):
    outputs, metrics, diagnostics = _score(SecondPlaceSearch(universe), cases)

    assert all(output["ok"] for output in outputs)
    assert diagnostics == [
        "On the same rewritten queries, your search scored 0.500 and a direct "
        "ranking of your embeddings scored 1.000; search runs through your own "
        "code and is scored on its first 50 results."
    ]
    # The sentence reads the numbers; it does not change them.
    assert metrics == component_scores(outputs, cases, SEARCH_K)[0]
    assert metrics["search_mrr"] == pytest.approx(0.5)
    assert metrics["retrieval_mrr"] == pytest.approx(1.0)
    # The local report cuts an entry at 240 characters.
    assert len(diagnostics[0]) <= 240


def test_adapter_mappings_follow_the_finding(universe, cases):
    """Course-style names run cleanly and carry mapping notes on the output."""

    outputs, _metrics, diagnostics = _score(CourseStyleAdapter(universe), cases)

    assert all(output["ok"] for output in outputs)
    assert outputs[0]["mappings"]
    assert diagnostics[0].startswith(FINDING)
    assert diagnostics[1:] == [
        "adapter: {}".format(mapping) for mapping in outputs[0]["mappings"]
    ]


def test_a_case_that_failed_leads_and_no_finding_is_written(universe, cases):
    """The typo rung's zero is in `search_mrr`, so a comparison would cite it."""

    outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
    broken = [
        {"ok": False, "kind": "search", "error": "raised on an unseen word"}
        if case.kind == "search" and case.rung == "typo"
        else output
        for case, output in zip(cases, outputs)
    ]
    benchmark = LanguageSearchBenchmark()
    metrics = benchmark.score(broken, cases)

    assert metrics["search_mrr_typo"] == 0.0
    assert benchmark.last_diagnostics[0].startswith(
        "the typo query rung scored 0: raised on an unseen word"
    )
    assert not any(FINDING in note for note in benchmark.last_diagnostics)
