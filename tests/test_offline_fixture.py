"""End-to-end plugin behavior on the synthetic universe (no downloads)."""

import numpy as np
import pytest

from language_search_benchmark.datasets import SEARCH_K
from language_search_benchmark.drivers import run_cases
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
