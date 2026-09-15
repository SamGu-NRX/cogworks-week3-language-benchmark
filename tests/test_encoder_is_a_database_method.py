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
#: The ids a case carries, kept apart from the search fixture's `IDS` so a
#: ranking says which database answered it.
CURRENT_IDS = [41, 52]
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


def _resolve(root, remember=False, accepts=lambda chain, *_: (True, "")):
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
        accepts=accepts, arrangements=None,
        prepare=benchmark._weights_of,
        # The week declares this hook, and declaring it is what turns the memo
        # off (`resolve.resolve`), so a helper that omitted it would resolve
        # under a contract the plugin does not have.
        construct=benchmark._weights_model_for,
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


def _cases():
    from language_search_benchmark.datasets import RetrievalCase, SearchCase, TextCase

    return [
        TextCase(kind="text", captions=CAPTIONS, group_rows=None, tie_break_seed=0),
        RetrievalCase(
            kind="retrieval", queries=CAPTIONS, descriptors=_descriptors(),
            gold_rows=None, tie_break_seed=0,
        ),
        SearchCase(
            kind="search", queries=CAPTIONS, image_ids=IDS,
            descriptors=_descriptors(), gold_image_ids=None, k=2, tie_break_seed=0,
        ),
    ]


class TestTheDriverOrderReachesTheImageHalf:
    """The driver's own order: retrieval before the database is ever built.

    `datasets.materialize_cases` puts retrieval first so an in-place
    `prepare_database` cannot move the retrieval baseline. Here the image
    encoder is a method of the database, so that order is what made the
    retrieval component score zero for a repository whose image half works.
    """

    def _adapter(self, repository):
        from language_search_benchmark.adapters import adapt_search
        from language_search_benchmark.discovered import build

        _write(repository, A)
        return adapt_search(build(_resolve(repository)))

    def test_every_case_runs_and_the_ranking_uses_the_pool_ids(self, repository):
        from language_search_benchmark.drivers import run_with_adapter

        outputs = run_with_adapter(self._adapter(repository), _cases())

        assert [output.get("ok") for output in outputs] == [True, True, True], outputs
        # The image half is built over row positions; nothing may answer with
        # them. What ranks is the database `prepare_database` built from the
        # pool's own ids.
        for ranking in outputs[2]["rankings"]:
            assert sorted(ranking) == sorted(IDS)

    def test_the_image_half_is_the_same_before_and_after_the_database(self, repository):
        from language_search_benchmark.drivers import run_with_adapter

        adapter = self._adapter(repository)
        first = run_with_adapter(adapter, _cases())
        # The second pass runs the retrieval case with the database from the
        # first still live, which is the owner the search proved.
        second = run_with_adapter(adapter, _cases())

        assert second[1]["ok"], second[1]
        assert first[1]["images"] == second[1]["images"]


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


class TestASecondResolutionWithNothingRemembered:
    def test_a_branch_binding_is_researched_with_the_current_capture(self, repository):
        """``remember=True`` writes nothing for this week, and the week is
        still right about the file in front of it.

        Two reasons in `cogbench.resolve`, either one enough: a role made of
        branches has no replayable record, because labels alone cannot restore
        methods of the objects those branches built, and a week that declares
        `construct` is not remembered at all. So the assertion is the miss
        itself, and then that the second search binds against B's bytes.
        """

        weights = _write(repository, A)
        first = _resolve(repository, remember=True)
        assert first.weights_captured is not None
        assert not (repository / ".cogbench" / "resolved.json").exists()

        np.save(weights, B)
        second = _resolve(repository, remember=True)

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


class TestTheChainIsJudgedOnThisCasesDatabase:
    """Acceptance composes the adapter and runs the driver over the case in
    front of it, and that case carries its own image ids.

    The search fixture's database holds `IDS` and `prepare_database` builds one
    over the case's ids. `roles.accepts` passes on "the case ran", so the
    ranking is read here: a query answered out of the fixture's database names
    ids this case never had, and a week that checks them refuses every search
    chain and reports a repository whose query works as having none.
    """

    def _accepts(self, seen):
        from language_search_benchmark.adapters import adapt_search
        from language_search_benchmark.datasets import SearchCase
        from language_search_benchmark.discovered import DiscoveredSearch
        from language_search_benchmark.drivers import run_with_adapter

        cases = [SearchCase(
            kind="search", queries=CAPTIONS, image_ids=CURRENT_IDS,
            descriptors=_descriptors(), gold_image_ids=None, k=2, tie_break_seed=0,
        )]

        def accepts(chains, *_):
            if not isinstance(chains, dict) or "search" not in chains:
                return True, ""
            composed = DiscoveredSearch(dict(chains), {})
            output = run_with_adapter(adapt_search(composed), cases)[0]
            if not output.get("ok"):
                return False, str(output.get("error", ""))[:160]
            ranked = sorted({image for row in output["rankings"] for image in row})
            seen.append(ranked)
            return ranked == sorted(CURRENT_IDS), "ranked {}".format(ranked)

        return accepts

    def test_the_ranking_it_is_judged_on_holds_this_cases_ids(self, repository):
        _write(repository, A)
        seen = []

        found = _resolve(repository, accepts=self._accepts(seen))

        assert found.ready, found.verdict.headline
        assert seen and seen[-1] == sorted(CURRENT_IDS)
        assert [step.label for step in found.branches["search"]] == [
            "database.ImageDatabase.query"
        ]
