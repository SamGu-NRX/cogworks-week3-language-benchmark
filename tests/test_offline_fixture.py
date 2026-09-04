"""End-to-end plugin behavior on the synthetic universe (no downloads)."""

import numpy as np
import pytest

from language_search_benchmark.datasets import SEARCH_K, SearchCase
from language_search_benchmark.drivers import run_cases, run_with_adapter
from language_search_benchmark.metrics import chance_mrr, component_scores
from language_search_benchmark.plugins import LanguageSearchBenchmark

from .fixtures.synthetic import (
    BrokenAdapter,
    InertAdapter,
    PerfectAdapter,
    Universe,
)


@pytest.fixture(scope="module")
def universe():
    return Universe()


@pytest.fixture(scope="module")
def cases(universe):
    return universe.cases()


def _score(outputs, cases):
    benchmark = LanguageSearchBenchmark()
    metrics = benchmark.score(outputs, cases)
    return metrics, benchmark.last_diagnostics


def test_perfect_adapter_beats_chance_everywhere(universe, cases):
    outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
    metrics, _ = _score(outputs, cases)
    pool = cases[1].descriptors.shape[0]
    assert metrics["retrieval_mrr"] > 0.9 > 3 * chance_mrr(pool)
    assert metrics["text_mrr"] > 0.9
    assert metrics["search_mrr"] > 0.9
    assert metrics["overall"] > 0.8
    assert metrics["retrieval_recall_at_1"] > 0.8


def test_inert_adapter_lands_at_chance(universe, cases):
    outputs = run_cases(lambda resources: InertAdapter(), None, cases)
    metrics, diagnostics = _score(outputs, cases)
    pool = cases[1].descriptors.shape[0]
    chance = chance_mrr(pool)
    # All-zero embeddings are all-ties; the seeded permutation makes the
    # outcome a single random draw, so allow a generous band around chance.
    assert metrics["retrieval_mrr"] < 6 * chance
    assert metrics["overall"] < 0.25
    assert any("all-zero" in note for note in diagnostics)


def test_broken_adapter_reports_and_scores_zero(universe, cases):
    outputs = run_cases(lambda resources: BrokenAdapter(), None, cases)
    metrics, diagnostics = _score(outputs, cases)
    assert metrics["overall"] == 0.0
    assert metrics["retrieval_median_rank"] == cases[1].descriptors.shape[0]
    assert any("mynn_model_weights_final.pkl" in note for note in diagnostics)
    assert all(not output["ok"] for output in outputs)


def test_factory_failure_carries_mapping_report(universe, cases):
    class Mystery:
        pass

    outputs = run_cases(lambda resources: Mystery(), None, cases)
    assert all(not output["ok"] for output in outputs)
    assert "benchmark_adapter.py" in outputs[0]["error"]


def test_component_isolation_d_mismatch(universe, cases):
    class MismatchedTowers:
        """Text 8-d, images 16-d: retrieval dies, text still scores."""

        def __init__(self):
            self._inner = PerfectAdapter(universe)

        def embed_text(self, captions):
            return self._inner.embed_text(captions)

        def embed_images(self, descriptors):
            base = self._inner.embed_images(descriptors)
            return np.concatenate([base, base], axis=1)

    outputs = run_cases(lambda resources: MismatchedTowers(), None, cases)
    metrics, diagnostics = _score(outputs, cases)
    assert metrics["text_mrr"] > 0.9
    assert metrics["retrieval_mrr"] == 0.0
    assert metrics["search_mrr"] == 0.0
    assert any("share one space" in note for note in diagnostics)


def test_prepare_database_runs_once_for_a_shared_pool(universe):
    """The rung cases share the verbatim case's pool objects, and a student
    index need not survive being rebuilt (an append-style prepare grows it).
    The driver must build once for the shared set and again only when the
    pool objects actually change."""

    from language_search_benchmark.adapters import adapt_search

    from .fixtures.synthetic import PerfectAdapter

    class CountingAdapter(PerfectAdapter):
        def __init__(self, inner_universe):
            super().__init__(inner_universe)
            self.prepare_calls = []

        def prepare_database(self, image_ids, descriptors):
            self.prepare_calls.append(list(image_ids))
            super().prepare_database(image_ids, descriptors)

    def search_case(queries, image_ids, descriptors, gold, rung="verbatim"):
        return SearchCase(
            kind="search",
            queries=queries,
            image_ids=image_ids,
            descriptors=descriptors,
            gold_image_ids=gold,
            k=10,
            tie_break_seed=7,
            rung=rung,
        )

    pool_indices = list(range(10, 40))
    shared_ids = [universe.image_ids[i] for i in pool_indices]
    shared_descriptors = universe.descriptors[pool_indices].copy()
    queries = [universe.captions_of[i][0] for i in pool_indices[:5]]
    gold = [universe.image_ids[i] for i in pool_indices[:5]]
    other_indices = list(range(0, 10))
    other_ids = [universe.image_ids[i] for i in other_indices]
    other_descriptors = universe.descriptors[other_indices].copy()
    other_queries = [universe.captions_of[i][0] for i in other_indices[:3]]
    other_gold = [universe.image_ids[i] for i in other_indices[:3]]

    cases = [
        search_case(queries, shared_ids, shared_descriptors, gold),
        search_case(queries, shared_ids, shared_descriptors, gold, rung="keywords"),
        search_case(queries, shared_ids, shared_descriptors, gold, rung="typo"),
        search_case(other_queries, other_ids, other_descriptors, other_gold),
    ]
    counting = CountingAdapter(universe)
    outputs = run_with_adapter(adapt_search(counting), cases)
    assert all(output["ok"] for output in outputs)
    # Once for the three shared-pool cases, once for the distinct pool.
    assert counting.prepare_calls == [shared_ids, other_ids]
    # The single build still serves every shared-pool case: the verbatim
    # queries hit gold at rank 1 across all of them.
    assert outputs[0]["rankings"][0][0] == gold[0]
    assert outputs[3]["rankings"][0][0] == other_gold[0]


def test_outputs_are_json_serializable(universe, cases):
    import json

    outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
    encoded = json.dumps(outputs)
    assert len(encoded) > 0


def test_score_requires_gold(universe, cases):
    outputs = run_cases(lambda resources: PerfectAdapter(universe), None, cases)
    stripped = universe.cases()
    stripped[0].group_rows = None
    with pytest.raises(ValueError):
        component_scores(outputs, stripped, SEARCH_K)


def test_course_artifact_git_lfs_pointer_redirects_to_benchmark_copy(tmp_path):
    from cogbench.discover import _Redirects

    pointer = tmp_path / "student" / "glove.6B.200d.txt.w2v"
    pointer.parent.mkdir()
    pointer.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:0000000000000000000000000000000000000000000000000000000000000000\n"
        "size 123\n"
    )
    benchmark_copy = tmp_path / "benchmark" / pointer.name
    benchmark_copy.parent.mkdir()
    benchmark_copy.write_text("1 1\nword 0.5\n")
    redirects = _Redirects({pointer.name: benchmark_copy})

    def parse_pointer():
        path = pointer
        raise ValueError("invalid literal in course data")

    try:
        parse_pointer()
    except ValueError as error:
        assert redirects.wanted(error) == pointer.name

    redirects.enter()
    try:
        redirects.install(pointer.name)
        with open(pointer, "r", encoding="utf-8") as stream:
            assert stream.read() == benchmark_copy.read_text()
    finally:
        redirects.leave()
