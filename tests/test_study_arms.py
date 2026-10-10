"""Study arm tests: determinism, codebook separation, resolver behavior.

The heavy arms (reference training over the real corpus) are exercised by
the locked runs in ``study/results/``; these tests cover the properties the
runs depend on, on tiny in-memory fixtures: the codebook trick that makes a
lookup presentable as embeddings, the resolver's coverage of every rewrite
rung, determinism across instantiations, and the harness's handling of
failed components. No downloads, no GloVe.
"""

from __future__ import annotations

import numpy as np
import pytest

from language_search_benchmark import perturb
from study import arms as arm_defs
from study import analyze


class StubResources:
    """The Resources surface the lookup arms read, on a tiny corpus."""

    def __init__(self, annotations, descriptors):
        self._annotations = annotations
        self._descriptors = descriptors

    def load_captions(self):
        return {"annotations": self._annotations, "images": []}

    def load_descriptors(self):
        return self._descriptors

    def load_glove(self):  # pragma: no cover - lookup arms never call this
        raise AssertionError("lookup arms must not load GloVe")


CAPTIONS = [
    "a man riding a horse on the beach",
    "a plate of pasta with tomato sauce",
    "two dogs playing in the snow",
]


def _stub_resources():
    annotations = []
    for index, caption in enumerate(CAPTIONS):
        image_id = 100 + index
        for copy in range(2):
            annotations.append(
                {"id": 1000 + 10 * index + copy, "image_id": image_id, "caption": caption}
            )
    descriptors = {
        100 + index: np.zeros((1, 512), dtype=np.float32) + index
        for index in range(len(CAPTIONS))
    }
    return StubResources(annotations, descriptors)


WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven"]


def _distinct_descriptors_stub(n_images=12):
    # One caption per image, and no shared descriptor rows, so token overlap
    # has a unique winner. Two caption copies would make image N's token set
    # collide with image N-1's (the copy counter shares words with digit
    # words), and overlap ties would fall to the lower index.
    annotations = []
    rng = np.random.RandomState(3)
    descriptors = {}
    for index in range(n_images):
        image_id = 500 + index
        annotations.append(
            {
                "id": 9000 + 10 * index,
                "image_id": image_id,
                "caption": "image {} scene".format(WORDS[index]),
            }
        )
        descriptors[image_id] = rng.normal(size=(1, 512)).astype(np.float32)
    return StubResources(annotations, descriptors)


# --------------------------------------------------------------------------
# Codebook
# --------------------------------------------------------------------------


def test_codebook_is_deterministic_per_image():
    resources = _stub_resources()
    arm = arm_defs.LookupArm(resources)
    first = arm.codebook(12345)
    second = arm.codebook(12345)
    assert np.array_equal(first, second)
    assert first.shape == (arm_defs.CODEBOOK_DIM,)


def test_codebook_separates_pool_sized_ids():
    # 700 Gaussian unit vectors: cosine 1 only between identical ids, and
    # the largest off-diagonal cosine far from 1 -- this is what keeps a
    # lookup's retrieval honest under the scorer's dot product.
    resources = _stub_resources()
    arm = arm_defs.LookupArm(resources)
    vectors = np.stack([arm.codebook(image_id) for image_id in range(700)])
    vectors = arm_defs.l2_rows(vectors)
    similarities = vectors @ vectors.T
    np.fill_diagonal(similarities, -2.0)
    assert similarities.max() < 0.5
    arm_b = arm_defs.LookupArm(_stub_resources())
    assert np.array_equal(arm.codebook(700), arm_b.codebook(700))


# --------------------------------------------------------------------------
# Resolver coverage
# --------------------------------------------------------------------------


def test_derived_resolver_recovers_every_rewrite_rung():
    resources = _stub_resources()
    arm = arm_defs.LookupArm(resources)
    for caption in CAPTIONS:
        for rung in perturb.RUNGS:
            rewritten = perturb.rewrite_all([caption], rung)[0]
            resolved = arm.resolve(rewritten)
            assert resolved is not None, "{} rung {!r}".format(rung, rewritten)
            assert resolved == arm.exact[caption.strip().lower()]


def test_unresolvable_queries_embed_as_zero_rows():
    resources = _stub_resources()
    arm = arm_defs.LookupArm(resources)
    rows = arm.embed_text(["a completely unseen green submarine"])
    assert rows.shape == (1, arm_defs.CODEBOOK_DIM)
    assert not rows.any()


def test_embed_images_identifies_pool_rows_by_descriptor():
    resources = _stub_resources()
    arm = arm_defs.LookupArm(resources)
    descriptors = np.concatenate(
        [resources.load_descriptors()[100 + i] for i in range(3)]
    )
    rows = arm.embed_images(descriptors)
    assert not np.array_equal(rows[0], rows[1])
    # And zero rows for a descriptor not in the pickle.
    stranger = np.ones((1, 512), dtype=np.float32) * 7.5
    assert not arm.embed_images(stranger).any()


def test_lookup_arm_deterministic_across_instances():
    stub = _distinct_descriptors_stub()
    a = arm_defs.LookupArm(stub)
    b = arm_defs.LookupArm(stub)
    captions = [caption for caps in a.captions_of.values() for caption in caps]
    assert np.array_equal(a.embed_text(captions), b.embed_text(captions))
    descriptors = np.concatenate(
        [stub.load_descriptors()[500 + i] for i in range(12)]
    )
    assert np.array_equal(a.embed_images(descriptors), b.embed_images(descriptors))


def test_exact_lookup_resolves_only_exact_strings():
    resources = _stub_resources()
    arm = arm_defs.ExactLookupArm(resources)
    assert arm.resolve(CAPTIONS[0]) is not None
    keywords = perturb.rewrite_all([CAPTIONS[0]], "keywords")[0]
    assert arm.resolve(keywords) is None


def test_search_recovers_gold_under_rewrites():
    # The token-overlap search path must return the gold image first for
    # every rung of a caption it has seen, on a pool of distinct images.
    resources = _distinct_descriptors_stub()
    arm = arm_defs.LookupArm(resources)
    pool_ids = [500 + i for i in range(12)]
    arm.prepare_database(pool_ids, np.zeros((12, 512), dtype=np.float32))
    for image_id in pool_ids[:6]:
        caption = arm.captions_of[image_id][0]
        for rung in perturb.RUNGS:
            rewritten = perturb.rewrite_all([caption], rung)[0]
            top = arm.search(rewritten, 5)
            assert top and top[0] == image_id, "{} rung {!r}".format(rung, rewritten)


# --------------------------------------------------------------------------
# Determinism of the random control and the arm registry
# --------------------------------------------------------------------------


def test_random_control_is_seeded_by_row_count():
    a = arm_defs.RandomArm(None)
    b = arm_defs.RandomArm(None)
    assert np.array_equal(a.embed_text(["x"] * 5), b.embed_text(["x"] * 5))


def test_registry_contains_every_locked_arm():
    expected = {
        "reference",
        "lookup-derived",
        "lookup-oracle",
        "lookup-exact",
        "mistake-no-idf",
        "mistake-raw-count",
        "mistake-perword-norm",
        "mistake-raw-descriptor",
        "mistake-oov-crash",
        "mistake-memsearch",
        "control-constant",
        "control-zero",
        "control-random",
        "control-shuffled",
        "control-poolorder",
    }
    assert set(arm_defs.ARMS) == expected


def test_build_preserves_constructor_failure():
    class ExplodingResources:
        def load_captions(self):
            raise RuntimeError("corpus unavailable")

        def load_descriptors(self):  # pragma: no cover
            raise RuntimeError("corpus unavailable")

    instance, error = arm_defs.build("lookup-derived", ExplodingResources())
    assert instance is None
    assert error is not None and "corpus unavailable" in error


# --------------------------------------------------------------------------
# Harness evidence extraction
# --------------------------------------------------------------------------


def _fake_case(gold_rows):
    from types import SimpleNamespace

    return SimpleNamespace(gold_rows=gold_rows, tie_break_seed=11)


def test_retrieval_evidence_guard_returns_failure_marker():
    from study.run_study import _extract_retrieval_evidence

    case = _fake_case([0])
    evidence = _extract_retrieval_evidence(case, {"ok": False, "error": "boom"})
    assert evidence == {"failed": "boom"}


def test_search_evidence_guard_returns_failure_marker():
    from study.run_study import _extract_search_evidence

    case = _fake_case([0])
    evidence = _extract_search_evidence(case, {"ok": False, "error": "boom"})
    assert evidence == {"failed": "boom"}


# --------------------------------------------------------------------------
# Bootstrap determinism and behavior
# --------------------------------------------------------------------------


def test_paired_bootstrap_is_deterministic():
    a = np.asarray([1.0] * 10)
    b = np.asarray([0.0] * 10)
    first = analyze.paired_bootstrap(a, b, seed=42)
    second = analyze.paired_bootstrap(a, b, seed=42)
    assert first == second
    assert first["delta_mean"] == pytest.approx(1.0)
    assert first["ci95_low"] == pytest.approx(1.0)


def test_paired_bootstrap_flags_significance():
    rng = np.random.RandomState(0)
    a = rng.normal(0.8, 0.05, size=50)
    b = rng.normal(0.3, 0.05, size=50)
    result = analyze.paired_bootstrap(a, b, seed=7)
    assert result["excludes_zero"]
    assert result["ci95_low"] > 0.4


def test_paired_bootstrap_survives_zero_arm():
    a = np.zeros(5)
    result = analyze.paired_bootstrap(a, a.copy(), seed=1)
    assert result["delta_mean"] == 0.0
    assert not result["excludes_zero"]


def test_edit_distance_helper():
    assert arm_defs._edit_distance_at_most_one("horse", "house")
    assert arm_defs._edit_distance_at_most_one("horse", "horse")
    assert arm_defs._edit_distance_at_most_one("hor", "horse") is False
    assert arm_defs._edit_distance_at_most_one("cat", "cats")
    assert arm_defs._edit_distance_at_most_one("cat", "cart")
    assert arm_defs._edit_distance_at_most_one("cart", "cars")
    assert not arm_defs._edit_distance_at_most_one("cat", "dog")
    assert not arm_defs._edit_distance_at_most_one("cart", "cats")
