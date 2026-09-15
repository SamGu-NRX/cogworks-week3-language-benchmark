"""An encoder that is a method of the database the repository built.

`ImageDatabase(ids, descriptors, W_embed)` takes the projection and
`descriptor_to_embedding` reads it off `self`, so the encoder is handed
nothing and the receipt used to be withheld for a run that scored.

The projection is a real `data/W_embed.npy` loaded through the week's own
`_weights_of` prepare hook, so what these assert is the captured file, not a
matrix handed in by the test. A and B permute output coordinates rather than
scaling, because a scalar multiple normalizes to the same vector.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from language_search_benchmark import plugins, roles

WIDTH = 8
DESCRIPTOR_WIDTH = 512

MODULE = '''
import numpy as np


class ImageDatabase:
    def __init__(self, image_ids, descriptors, W_embed):
        self.image_ids = list(image_ids)
        self.W_embed = np.asarray(W_embed)
        self.embeddings = self.descriptor_to_embedding(np.asarray(descriptors))

    def descriptor_to_embedding(self, descriptor):
        w = np.asarray(descriptor) @ self.W_embed
        norm = np.linalg.norm(w, axis=-1, keepdims=True)
        return w / (norm + 1e-8)

    def query(self, caption_embedding, k=2):
        vector = np.asarray(caption_embedding, dtype=float).reshape(-1)
        vector = vector / (np.linalg.norm(vector) + 1e-8)
        sims = self.embeddings @ vector
        return [self.image_ids[i] for i in np.argsort(-sims)[:k]]


def embed_captions_batch(texts):
    rows = []
    for text in texts:
        row = np.zeros(8, dtype=float)
        for index, code in enumerate(text.encode("utf-8")[:8]):
            row[index] = code / 255.0
        rows.append(row)
    return np.stack(rows)
'''

AMBIENT = MODULE.replace(
    "def __init__(self, image_ids, descriptors, W_embed):",
    "def __init__(self, image_ids, descriptors):",
).replace("        self.W_embed = np.asarray(W_embed)", "        self.W_embed = np.eye(512, 8)")

IDS = [11, 22]
CAPTIONS = ["one", "two two"]


def _projection(order):
    matrix = np.zeros((DESCRIPTOR_WIDTH, WIDTH))
    for source, target in enumerate(order):
        matrix[source, target] = 1.0
    return matrix


A = _projection(list(range(WIDTH)))
B = _projection(list(reversed(range(WIDTH))))


def _descriptors():
    rows = np.zeros((2, DESCRIPTOR_WIDTH))
    rows[0, 0] = 1.0
    rows[1, 7] = 1.0
    return rows


@pytest.fixture
def repository():
    root = Path(tempfile.mkdtemp()).resolve()
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _write(root, matrix, source=MODULE):
    (root / "database.py").write_text(source)
    weights = root / "data" / "W_embed.npy"
    weights.parent.mkdir(parents=True, exist_ok=True)
    if matrix is not None:
        np.save(weights, matrix)
    return weights


def _resolve(root, remember=False):
    """Resolve through the week's own prepare hook, so the projection is the
    file on disk and the SDK retains it."""

    pytest.importorskip("cogbench")
    from cogbench import resolve as resolve_module

    benchmark = plugins.LanguageSearchBenchmark()
    # `discovery()` is what normally creates this, and it loads the course
    # corpus. The hook only needs the pool to exist (same setup as
    # test_weights_in_repository).
    benchmark._discovery_extras = {}
    role = roles.search_role(
        captions=CAPTIONS, corpus=CAPTIONS, descriptors=_descriptors(),
        image_ids=IDS, query=CAPTIONS[0], k=2,
    )
    return resolve_module.resolve(
        root, chain_role=role, fixture=(CAPTIONS,),
        accepts=lambda chain, *_: (True, ""), arrangements=None,
        prepare=benchmark._weights_of,
        weights_consumed=benchmark._weights_consumed,
        remember=remember, benchmark="language-search",
    )


def _composed(found):
    from language_search_benchmark.discovered import build

    search = build(found)
    initial = search.embed_images(_descriptors())
    search.prepare_database(IDS, _descriptors())
    np.testing.assert_allclose(search.embed_images(_descriptors()), initial)
    return search


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TestTheDatabaseMethodRoute:
    def test_the_recorded_route_is_the_database_branch(self, repository):
        _write(repository, A)
        found = _resolve(repository)
        assert found.ready, found.verdict.headline
        image, constructor = found.branches["image"][0], found.branches["prepare"][0]

        assert image.attribute is not None
        assert image.branch == "prepare"
        assert image.owner is constructor.call
        assert isinstance(constructor.call, type)
        assert "extra:W" in constructor.plan
        assert not image.plan and not image.keywords and not image.supplied

    def test_the_receipt_names_the_captured_file(self, repository):
        weights = _write(repository, A)
        found = _resolve(repository)

        assert found.weights_used == ("data/W_embed.npy",)
        assert found.weights_captured is not None
        receipt = found.weights_captured[0]
        assert receipt["path"] == "data/W_embed.npy"
        assert receipt["sha256"] == _digest(weights)
        assert receipt["size"] == weights.stat().st_size
        retained = repository / ".cogbench" / "weights" / receipt["sha256"] / receipt["path"]
        assert retained.read_bytes() == weights.read_bytes()

    def test_an_ambient_constructor_is_refused_though_it_is_ready(self, repository):
        _write(repository, A, source=AMBIENT)
        found = _resolve(repository)
        assert found.ready, found.verdict.headline
        assert found.branches.get("image")
        assert found.weights_captured is None

    def test_a_different_owner_does_not_satisfy_the_route(self, repository):
        _write(repository, A)
        found = _resolve(repository)
        image, constructor = found.branches["image"][0], found.branches["prepare"][0]
        benchmark = plugins.LanguageSearchBenchmark()

        class Other:
            pass

        assert benchmark._weights_consumed(found)
        for unrelated in (
            replace(image, owner=Other),
            replace(image, branch="text"),
            replace(image, branch=None),
            replace(image, attribute=None),
        ):
            changed = replace(found, branches=dict(found.branches, image=(unrelated,)))
            assert not benchmark._weights_consumed(changed)
        for unsupported in (
            (replace(constructor, plan=(), supplied={}),),
            (replace(constructor, call=lambda *args: None),),
            (constructor, constructor),
        ):
            changed = replace(found, branches=dict(found.branches, prepare=unsupported))
            assert not benchmark._weights_consumed(changed)


@pytest.mark.parametrize(
    "image_input,prepare_input,form,expected",
    [
        ("W", "W", 0, True),
        ("weights_model", "weights_model", 1, True),
        ("W", "weights_model", 0, False),
        ("weights_model", "W", 0, False),
        ("W", None, 2, True),
        ("W", None, 3, True),
        ("W", None, 0, False),
        ("W", None, 4, False),
    ],
)
def test_direct_input_contract_is_unchanged(
    repository, image_input, prepare_input, form, expected
):
    _write(repository, A)
    found = _resolve(repository)
    image = replace(found.branches["image"][0], plan=("value", "extra:" + image_input))
    constructor = replace(
        found.branches["prepare"][0],
        plan=("value", "value", "extra:" + prepare_input) if prepare_input else (),
        form=form,
        supplied={},
    )
    planned = replace(found, branches=dict(found.branches, image=(image,), prepare=(constructor,)))
    assert plugins.LanguageSearchBenchmark()._weights_consumed(planned) is expected


class TestTheRetainedProjectionIsWhatScores:
    def _observe(self, found):
        search = _composed(found)
        embedded = search.embed_images(_descriptors())
        return np.asarray(embedded), list(search.search(CAPTIONS[0], 2))

    def test_b_changes_direction_and_ranking(self, repository):
        _write(repository, A)
        image_a, ranked_a = self._observe(_resolve(repository))
        _write(repository, B)
        image_b, ranked_b = self._observe(_resolve(repository))

        assert not np.allclose(image_a, image_b)
        assert ranked_a != ranked_b

    def test_the_weight_file_may_be_replaced_or_deleted_after_the_run(self, repository):
        weights = _write(repository, A)
        found = _resolve(repository)
        before_image, before_ranked = self._observe(found)
        digest = found.weights_captured[0]["sha256"]

        np.save(weights, B)
        replaced_image, replaced_ranked = self._observe(found)
        weights.unlink()
        deleted_image, deleted_ranked = self._observe(found)

        for image, ranked in ((replaced_image, replaced_ranked),
                              (deleted_image, deleted_ranked)):
            assert np.allclose(image, before_image)
            assert ranked == before_ranked
        assert found.weights_captured[0]["sha256"] == digest


class TestASecondResolutionOverTheMemo:
    def test_a_branch_binding_is_researched_with_the_current_capture(self, repository):
        weights = _write(repository, A)
        first = _resolve(repository, remember=True)
        assert first.weights_captured is not None
        memo = json.loads((repository / ".cogbench" / "resolved.json").read_text())
        assert memo["binding"]["branches"]["image"] == [
            "database.ImageDatabase.descriptor_to_embedding"
        ]

        np.save(weights, B)
        second = _resolve(repository, remember=True)

        # SDK _replay deliberately searches branch bindings again: labels
        # alone cannot restore methods of the objects those branches built.
        assert not second.recalled
        assert second.ready
        image = second.branches["image"][0]
        assert image.branch == "prepare"
        assert image.owner is second.branches["prepare"][0].call
        assert second.weights_used == ("data/W_embed.npy",)
        assert second.weights_captured is not None
        assert second.weights_captured[0]["sha256"] == _digest(weights)
        assert second.weights_captured[0]["sha256"] != first.weights_captured[0]["sha256"]
        weights.unlink()
        first_run, second_run = _composed(first), _composed(second)
        assert not np.allclose(
            first_run.embed_images(_descriptors()), second_run.embed_images(_descriptors())
        )
        assert first_run.search(CAPTIONS[0], 2) != second_run.search(CAPTIONS[0], 2)
