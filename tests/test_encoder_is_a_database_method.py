"""An encoder that is a method of the database the repository built.

`ImageDatabase(ids, descriptors, W_embed)` takes the projection and
`descriptor_to_embedding` reads it off `self`, so the encoder is handed
nothing and the receipt used to be withheld for a run that scored.

Such an encoder has no owner until the database is built, and the SDK refuses
to call it before then rather than borrow the object discovery built. The
scored run therefore builds the database first, from this run's pool, and a
store that takes the image branch's projected matrix still needs the image
branch first. Both orders are driven here through the week's own driver, on
ids and descriptors discovery never saw.

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
    built = []

    def __init__(self, image_ids, descriptors, W_embed):
        ImageDatabase.built.append(list(image_ids))
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

#: The same database with the constructor's first two arguments swapped, so
#: discovery binds `prepare_forms` form 1 instead of form 0.
SWAPPED = MODULE.replace(
    "def __init__(self, image_ids, descriptors, W_embed):",
    "def __init__(self, descriptors, image_ids, W_embed):",
)

#: Bagel's shape: a standalone encoder, and a store built from its PROJECTED
#: rows (`prepare_forms` form 3). The width check is what makes the store
#: refuse raw descriptors, as a store that declares its embedding width does;
#: without it discovery binds the first raw form that constructs.
PROJECTED = '''
import numpy as np

EMBED_DIM = 8


def descriptor_to_embedding(descriptors, W_embed):
    w = np.asarray(descriptors) @ np.asarray(W_embed)
    return w / (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)


class CaptionImageQuery:
    built = []

    def __init__(self, image_embeddings, image_ids):
        rows = np.asarray(image_embeddings, dtype=float)
        if rows.ndim != 2 or rows.shape[1] != EMBED_DIM:
            raise ValueError("image embeddings must be (N, {})".format(EMBED_DIM))
        CaptionImageQuery.built.append(list(image_ids))
        self.embeddings = rows / (np.linalg.norm(rows, axis=1, keepdims=True) + 1e-8)
        self.image_ids = list(image_ids)

    def search(self, caption_vector, top_k=2):
        vector = np.asarray(caption_vector, dtype=float).reshape(-1)
        sims = self.embeddings @ (vector / (np.linalg.norm(vector) + 1e-8))
        return [self.image_ids[i] for i in np.argsort(-sims)[:top_k]]


def embed_captions_batch(texts):
    rows = []
    for text in texts:
        row = np.zeros(8, dtype=float)
        for index, code in enumerate(text.encode("utf-8")[:8]):
            row[index] = code / 255.0
        rows.append(row)
    return np.stack(rows)
'''

#: Form 2: the same store taking the ids first.
PROJECTED_IDS_FIRST = PROJECTED.replace(
    "def __init__(self, image_embeddings, image_ids):",
    "def __init__(self, image_ids, image_embeddings):",
)

#: The other direction of ownership: the image branch constructs the object,
#: and prepare is a method of it (`branch="image"`) called on a raw form.
#: Its store keeps raw rows and its search reads their first eight columns.
IMAGE_OWNS_PREPARE = '''
import numpy as np


class ImageEncoder:
    def __init__(self, descriptors, W_embed):
        rows = np.asarray(descriptors)
        if rows.ndim != 2 or rows.shape[1] != 512:
            raise ValueError("expected descriptors")
        self.data = rows @ W_embed

    def build_database(self, image_ids, descriptors):
        rows = np.asarray(descriptors)
        if rows.ndim != 2 or rows.shape[1] != 512:
            raise ValueError("expected descriptors")
        return dict(zip(image_ids, rows))


def query_database(caption_embedding, k, prepare):
    ids = list(prepare)
    scores = np.asarray(list(prepare.values()))[:, :8] @ np.asarray(caption_embedding)
    return [ids[i] for i in np.argsort(-scores)[:k]]


def embed_captions_batch(texts):
    rows = []
    for text in texts:
        row = np.zeros(8, dtype=float)
        for index, code in enumerate(text.encode("utf-8")[:8]):
            row[index] = code / 255.0
        rows.append(row)
    return np.stack(rows)
'''

#: Each constructor form discovery records, with the source that binds it.
FORMS = {0: MODULE, 1: SWAPPED, 2: PROJECTED_IDS_FIRST, 3: PROJECTED}

IDS = [11, 22]
CAPTIONS = ["one", "two two"]

#: The scored run's pool and queries: none of them is the discovery fixture's.
RUNTIME_IDS = [101, 202, 303]
RUNTIME_QUERIES = ["one horse", "seven swans", "a red bus"]
SEARCH_K = 2


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


def _runtime_descriptors(hot=(0, 7, 3)):
    rows = np.zeros((len(hot), DESCRIPTOR_WIDTH))
    for row, column in enumerate(hot):
        rows[row, column] = 1.0
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


def _grid(ids=RUNTIME_IDS, descriptors=None):
    from language_search_benchmark.datasets import build_cases

    return build_cases(
        text_captions=RUNTIME_QUERIES,
        queries=RUNTIME_QUERIES,
        pool_image_ids=list(ids),
        pool_descriptors=_runtime_descriptors() if descriptors is None else descriptors,
        tie_break_seed=0,
        search_k=SEARCH_K,
    )


def _scored(found, cases=None):
    """The week's own driver on a grid built from values discovery never saw.

    Returns every retrieval case's image matrix and every search case's
    rankings, keyed by rung, after checking that each case ran.
    """

    from language_search_benchmark.adapters import adapt_search
    from language_search_benchmark.discovered import build
    from language_search_benchmark.drivers import run_with_adapter

    cases = _grid() if cases is None else cases
    outputs = run_with_adapter(adapt_search(build(found)), cases)
    for case, output in zip(cases, outputs):
        assert output["ok"], (case.kind, case.rung, output.get("error"))
    images = [
        np.asarray(output["images"])
        for case, output in zip(cases, outputs) if case.kind == "retrieval"
    ]
    rankings = {
        case.rung: output["rankings"]
        for case, output in zip(cases, outputs) if case.kind == "search"
    }
    return images, rankings


def _caption_row(text):
    """The fixture's caption encoder, restated as the oracle."""

    row = np.zeros(WIDTH)
    for index, code in enumerate(text.encode("utf-8")[:WIDTH]):
        row[index] = code / 255.0
    return row


def _expected(projection, cases=None, ids=RUNTIME_IDS, descriptors=None):
    """What a correct run scores with `projection`: its image rows and the
    top-k of every search rung, computed here without their code."""

    descriptors = _runtime_descriptors() if descriptors is None else descriptors
    rows = descriptors @ projection
    rows = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    rankings = {}
    for case in _grid(ids, descriptors) if cases is None else cases:
        if case.kind != "search":
            continue
        rankings[case.rung] = [
            [ids[i] for i in np.argsort(-(rows @ _caption_row(query)), kind="stable")[:SEARCH_K]]
            for query in case.queries
        ]
    return rows, rankings


def _assert_scored_with(found, projection):
    images, rankings = _scored(found)
    rows, expected = _expected(projection)
    for matrix in images:
        np.testing.assert_allclose(matrix, rows, atol=1e-5)
    assert rankings == expected
    return images, rankings


#: The search step each form's repository binds: a method of the object the
#: prepare branch built, carried into the search branch.
SEARCH_LABELS = {
    0: "database.ImageDatabase.query",
    1: "database.ImageDatabase.query",
    2: "database.CaptionImageQuery.search",
    3: "database.CaptionImageQuery.search",
}


def _discovered(repository, form, matrix):
    """Resolve `FORMS[form]` with `matrix` on disk, and check what bound.

    Every branch comes from discovery itself. The projected forms' search
    step binds only with an SDK whose renewal rebuilds an earlier branch that
    a later branch's fixture reads, first written as cogbench 7886034 and
    verified here with the published cb8b582 (SamGu-NRX/CogPortal#48).
    Without that fix the image branch the projected prepare form is made from
    was not rebuilt, and the search branch was refused.
    """

    _write(repository, matrix, source=FORMS[form])
    found = _resolve(repository)
    assert found.ready, found.verdict.headline
    constructor = found.branches["prepare"][0]
    assert constructor.form == form
    assert isinstance(constructor.call, type)
    search = found.branches["search"]
    assert [step.label for step in search] == [SEARCH_LABELS[form]]
    assert search[0].branch == "prepare"
    assert search[0].owner is constructor.call
    image = found.branches["image"][0]
    # Forms 0 and 1 are the database's own method, which needs its owner;
    # the projected forms' encoder is standalone and runs first.
    if form < 2:
        assert image.branch == "prepare"
        assert image.owner is constructor.call
    else:
        assert image.branch is None
        assert image.label == "database.descriptor_to_embedding"
    return found


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


class TestTheDatabaseIsBuiltBeforeItsEncoderRuns:
    """The scored run on this run's pool, through the week's own driver."""

    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_every_case_scores_the_runtime_pool_with_the_captured_projection(
        self, repository, form
    ):
        found = _discovered(repository, form, A)
        store = found.branches["prepare"][0].call
        before = len(store.built)

        _assert_scored_with(found, A)

        # One database for all nine cases, built from the runtime ids: not
        # from the discovery fixture, and not once per rung or per component.
        assert store.built[before:] == [RUNTIME_IDS]

    @pytest.mark.parametrize("form", [0, 1])
    def test_a_second_pool_builds_a_second_owner(self, repository, form):
        found = _discovered(repository, form, A)
        store = found.branches["prepare"][0].call
        other_ids = [404, 505, 606]
        other = _runtime_descriptors(hot=(5, 1, 6))
        first, second = _grid(), _grid(other_ids, other)
        before = len(store.built)

        images, rankings = _scored(found, first + second)

        assert store.built[before:] == [RUNTIME_IDS, other_ids]
        rows, expected = _expected(A, second, other_ids, other)
        for matrix in images[len(images) // 2:]:
            np.testing.assert_allclose(matrix, rows, atol=1e-5)
        # The rung keys repeat across the two grids, so the second one's
        # rankings are the ones left under them.
        assert rankings == expected


    def test_a_prepare_owned_by_the_image_branch_gets_it_first(self, repository):
        """Search cases alone, so no retrieval case has built the image
        branch's object before prepare needs it."""

        _write(repository, A, source=IMAGE_OWNS_PREPARE)
        found = _resolve(repository)
        assert found.ready, found.verdict.headline
        prepare = found.branches["prepare"][0]
        assert prepare.branch == "image"
        assert prepare.form == 0

        searches = [case for case in _grid() if case.kind == "search"]
        _, rankings = _scored(found, searches)

        # The store ranks on the raw descriptors' first eight columns.
        _, expected = _expected(np.eye(DESCRIPTOR_WIDTH, WIDTH), searches)
        assert rankings == expected


class TestTheRetainedProjectionIsWhatScores:
    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_b_changes_direction_and_ranking(self, repository, form):
        image_a, ranked_a = _assert_scored_with(_discovered(repository, form, A), A)
        image_b, ranked_b = _assert_scored_with(_discovered(repository, form, B), B)

        assert not np.allclose(image_a[0], image_b[0])
        assert ranked_a != ranked_b

    @pytest.mark.parametrize("form", sorted(FORMS))
    def test_the_weight_file_may_be_replaced_or_deleted_after_the_run(
        self, repository, form
    ):
        found = _discovered(repository, form, A)
        weights = repository / "data" / "W_embed.npy"
        digest = found.weights_captured[0]["sha256"]

        np.save(weights, B)
        _assert_scored_with(found, A)
        weights.unlink()
        _assert_scored_with(found, A)
        assert found.weights_captured[0]["sha256"] == digest


class TestASecondResolutionKeepsNoMemo:
    def test_a_branch_binding_is_searched_again_with_the_current_capture(self, repository):
        weights = _write(repository, A)
        memo = repository / ".cogbench" / "resolved.json"
        first = _resolve(repository, remember=True)
        assert first.weights_captured is not None
        assert [step.label for step in first.branches["image"]] == [
            "database.ImageDatabase.descriptor_to_embedding"
        ]
        # The SDK keeps no memo for a role made of branches
        # (`cogbench.resolve._memo_key` gives it no key): a replay cannot
        # restore methods of the objects those branches built. So `remember`
        # writes nothing, and the second resolution searches the repository
        # as it now is.
        assert not memo.exists()

        np.save(weights, B)
        second = _resolve(repository, remember=True)

        assert not memo.exists()
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
        first_images, first_rankings = _assert_scored_with(first, A)
        second_images, second_rankings = _assert_scored_with(second, B)
        assert not np.allclose(first_images[0], second_images[0])
        assert first_rankings != second_rankings


class _Recorder:
    """An adapter that writes down the order it is called in."""

    def __init__(self, image_needs_database, prepare_raises=0):
        self.image_needs_database = image_needs_database
        self.has_search = True
        self.calls = []
        self._prepare_raises = prepare_raises

    def embed_text(self, captions):
        self.calls.append(("text",))
        return np.ones((len(captions), 2))

    def embed_images(self, descriptors):
        self.calls.append(("image", descriptors))
        return np.ones((len(descriptors), 2))

    def prepare_database(self, image_ids, descriptors):
        self.calls.append(("prepare", image_ids, descriptors))
        if self._prepare_raises:
            self._prepare_raises -= 1
            raise RuntimeError("their database failed")

    def search(self, query, k):
        return RUNTIME_IDS[:k]


class TestTheDriverOrder:
    """No SDK here: the order is the driver's, whatever made the adapter."""

    def test_an_owned_encoder_gets_its_database_first_and_once(self):
        from language_search_benchmark.drivers import run_with_adapter

        cases = _grid()
        search = next(case for case in cases if case.kind == "search")
        adapter = _Recorder(image_needs_database=True)

        outputs = run_with_adapter(adapter, cases)

        assert all(output["ok"] for output in outputs)
        prepares = [call for call in adapter.calls if call[0] == "prepare"]
        assert len(prepares) == 1
        assert prepares[0][1] is search.image_ids
        assert prepares[0][2] is search.descriptors
        first_image = next(i for i, call in enumerate(adapter.calls) if call[0] == "image")
        assert adapter.calls.index(prepares[0]) < first_image
        # Retrieval still embeds its own matrix, which prepare never saw.
        retrieval = next(case for case in cases if case.kind == "retrieval")
        assert adapter.calls[first_image][1] is retrieval.descriptors

    def test_a_standalone_encoder_keeps_the_old_order(self):
        from language_search_benchmark.drivers import run_with_adapter

        adapter = _Recorder(image_needs_database=False)
        run_with_adapter(adapter, _grid())

        kinds = [call[0] for call in adapter.calls]
        assert kinds.count("prepare") == 1
        assert kinds.index("image") < kinds.index("prepare")

    def test_a_failed_database_is_built_again_by_the_next_case(self):
        from language_search_benchmark.drivers import run_with_adapter

        cases = _grid()
        adapter = _Recorder(image_needs_database=True, prepare_raises=1)

        outputs = run_with_adapter(adapter, cases)

        # The retrieval case that needed it fails with their error; the
        # search case after it builds the database rather than trusting a
        # build that raised, and every later case reuses that one.
        assert [output["ok"] for output in outputs[:3]] == [True, False, True]
        assert "their database failed" in outputs[1]["error"]
        assert all(output["ok"] for output in outputs[3:])
        assert [call[0] for call in adapter.calls].count("prepare") == 2

    def test_a_hand_built_case_without_a_pool_says_so(self):
        """`RetrievalCase.database` is optional because cases are also written
        by hand; an owned encoder then fails its own case, loudly."""

        from language_search_benchmark.drivers import run_with_adapter

        cases = [
            replace(case, database=None) if case.kind == "retrieval" else case
            for case in _grid()
        ]
        adapter = _Recorder(image_needs_database=True)

        outputs = run_with_adapter(adapter, cases)

        for case, output in zip(cases, outputs):
            if case.kind == "retrieval":
                assert "names no pool to build it from" in output["error"], case.rung
            else:
                assert output["ok"], (case.kind, case.rung)
