"""The auto-adaptation matrix: native, course-surface, unmappable."""

import numpy as np
import pytest

from language_search_benchmark.adapters import adapt_search
from language_search_benchmark.contracts import AdapterContractError

from .fixtures.synthetic import (
    CourseStyleAdapter,
    PerfectAdapter,
    UnmappedPrepareAdapter,
    Universe,
)


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


def test_unmapped_prepare_is_reported_not_silently_zeroed(universe):
    """A prepare method named outside PREPARE_ALIASES used to cost the whole
    search component with no diagnostic anywhere: every output said ok, and
    the student saw a third of the score missing for no stated reason."""

    from language_search_benchmark.drivers import run_with_adapter

    adapter = adapt_search(UnmappedPrepareAdapter(universe))
    assert adapter.has_search
    assert not adapter.has_prepare
    note = " ".join(adapter.mappings)
    assert "prepare_database" in note
    assert "build_index" in note, "the report should name the method we found"

    outputs = run_with_adapter(adapter, universe.cases())
    search_output = next(o for o in outputs if o["kind"] == "search")
    assert search_output["ok"] is False, "a zeroed component must say why"
    message = search_output["error"]
    assert "never built" in message
    assert "prepare_database(image_ids, descriptors)" in message, (
        "the actionable rename must survive the 200-character truncation in "
        "_error_output"
    )
    # The components that did work still score.
    assert all(o["ok"] for o in outputs if o["kind"] != "search")


def test_batch_error_leads_when_retry_iterates_strings():
    """A real student case: batch embed_text raises KeyError on an OOV word,
    the per-item retry then iterates each caption string character-wise and
    produces ragged garbage. The report must lead with the KeyError."""

    class OovCrasher:
        def embed_text(self, captions):
            if isinstance(captions, str):
                # per-item retry path: characters, ragged widths
                return np.zeros((len(captions), 4)).reshape(-1)
            raise KeyError("skateboarder")

        def embed_images(self, descriptors):
            return np.zeros((len(descriptors), 8))

    adapter = adapt_search(OovCrasher())
    from language_search_benchmark.checks import CheckFailure

    with pytest.raises(CheckFailure) as info:
        adapter.embed_text(["a person on a skateboard", "two dogs"])
    message = str(info.value)
    assert "skateboarder" in message
    assert message.index("skateboarder") < message.index("per-item")


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
    # The batch call returned the wrong shape; that authoritative story wins
    # over the per-item retry's ragged-width detail.
    assert "2-D" in str(info.value)
