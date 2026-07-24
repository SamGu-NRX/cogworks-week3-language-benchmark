"""The auto-adaptation matrix: native, course-surface, unmappable."""

import numpy as np
import pytest

from language_search_benchmark.adapters import adapt_search
from language_search_benchmark.contracts import AdapterContractError

from .fixtures.synthetic import CourseStyleAdapter, PerfectAdapter, Universe


@pytest.fixture(scope="module")
def universe():
    return Universe()


def test_native_protocol_passes_through(universe):
    adapter = adapt_search(PerfectAdapter(universe))
    assert adapter.mappings == []
    result = adapter.embed_text(["picture number one take 0"])
    assert result.shape == (1, 8)
    assert adapter.has_search


def test_course_surface_is_mapped_and_logged(universe):
    adapter = adapt_search(CourseStyleAdapter(universe))
    # text_embedding is per-caption: alias mapping plus per-item fallback.
    result = adapter.embed_text(["picture number one take 0", "picture number two take 1"])
    assert result.shape == (2, 8)
    joined = " ".join(adapter.mappings)
    assert "text_embedding" in joined
    assert "se_image" in joined
    images = adapter.embed_images(universe.descriptors[:3])
    assert images.shape == (3, 8)
    adapter.prepare_database([1, 2, 3], universe.descriptors[:3])
    assert list(adapter.search("picture number one take 0", 2))


def test_unmappable_object_gets_actionable_report():
    class Mystery:
        def encode_stuff(self, items):
            return items

    with pytest.raises(AdapterContractError) as info:
        adapt_search(Mystery())
    report = " ".join(info.value.report)
    assert "Mystery" in report
    assert "embed_text" in report
    assert "benchmark_adapter.py" in report


def test_inconsistent_widths_fail_loudly(universe):
    class Ragged:
        def embed_text(self, caption):
            return np.zeros(4 if "one" in caption else 6)

        def embed_images(self, descriptors):
            return np.zeros((len(descriptors), 8))

    adapter = adapt_search(Ragged())
    from language_search_benchmark.checks import CheckFailure

    with pytest.raises(CheckFailure) as info:
        adapter.embed_text(["picture number one", "picture number two"])
    assert "inconsistent widths" in str(info.value)
