"""Which file in a repository is the trained projection, read from their code.

Three rules, each decided by one 2026 repository:

- A load call under a module-scope guard that is false when the file runs
  does not count. Bagel's `model_tests/test_db.py` sets `t = 0` and loads
  `test1.pkl` under `elif t == 1`; reading that as a real load made the
  repository ambiguous with the `testd_50.pkl` their application script
  loads unconditionally.
- A table whose every weight is a whole number is a count table, not an IDF
  table. rutvim's module-scope `vocab` is a defaultdict of counts beside
  their `idf`, and the first draft accepted ints.
- A zero-argument class of theirs with a `load(path)` that accepts the file
  and is callable afterwards is their model object (Bagel's
  `ImageToCaption`), and the pool carries it so their `__call__` can be the
  image step.
"""

from __future__ import annotations

import pickle
import shutil
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

from language_search_benchmark import roles


@pytest.fixture
def repository():
    root = Path(tempfile.mkdtemp()).resolve()
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _write(root: Path, name: str, text: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


class TestALoadUnderAFalseGuardDoesNotRun:
    def test_the_guarded_load_is_reported_as_not_running(self, repository):
        _write(
            repository,
            "switch.py",
            "t = 0\n"
            "if t == 0:\n"
            "    model.save('made.pkl')\n"
            "elif t == 1:\n"
            "    model.load('other.pkl')\n"
            "model.load('always.pkl')\n",
        )

        other = roles.named_in_source(repository, "other.pkl")
        always = roles.named_in_source(repository, "always.pkl")

        assert [(line, runs) for _, line, runs in other] == [(5, False)]
        assert [(line, runs) for _, line, runs in always] == [(6, True)]

    def test_short_circuit_and_unknown_branches_are_read_soundly(self, repository):
        _write(
            repository,
            "guards.py",
            "import sys\n"
            "t = 1\n"
            "x = sys.argv\n"
            "if t == 0 and x:\n"
            "    model.load('dead.pkl')\n"
            "if x or t == 1:\n"
            "    model.load('live.pkl')\n"
            "if x:\n"
            "    choice = 0\n"
            "else:\n"
            "    choice = 1\n"
            "if choice == 0:\n"
            "    model.load('a.pkl')\n"
            "else:\n"
            "    model.load('b.pkl')\n",
        )

        def runs(basename):
            return [flag for _, _, flag in roles.named_in_source(repository, basename)]

        assert runs("dead.pkl") == [False]
        assert runs("live.pkl") == [True]
        assert runs("a.pkl") == [True]
        assert runs("b.pkl") == [True]

    def test_a_guard_the_folder_cannot_decide_counts_as_running(self, repository):
        _write(
            repository,
            "main.py",
            "import sys\n"
            "t = int(sys.argv[1])\n"
            "if t == 1:\n"
            "    model.load('maybe.pkl')\n"
            "if __name__ == '__main__':\n"
            "    model.load('entry.pkl')\n",
        )

        assert [runs for _, _, runs in roles.named_in_source(repository, "maybe.pkl")] == [True]
        assert [runs for _, _, runs in roles.named_in_source(repository, "entry.pkl")] == [True]

    def test_the_unconditional_load_wins_over_the_dead_one(self, repository):
        for name in ("a.pkl", "b.pkl"):
            with open(repository / name, "wb") as stream:
                pickle.dump([np.zeros((512, 8), dtype=np.float32), np.zeros((1, 8))], stream)
        _write(repository, "app.py", "model.load('a.pkl')\n")
        _write(repository, "test_it.py", "t = 0\nif t == 1:\n    model.load('b.pkl')\n")

        found = roles.weights_in(repository, [str(Path(roles.__file__).resolve().parents[1])])

        assert found is not None
        assert found.path.name == "a.pkl"
        assert found.cited == "app.py:1"


class TestACountTableIsNotAnIdfTable:
    def test_whole_numbers_are_refused_and_fractions_accepted(self):
        assert not roles.looks_like_idf_table({"a": 3, "b": 1})
        assert not roles.looks_like_idf_table({"a": 3.0, "b": 1.0})
        assert roles.looks_like_idf_table({"a": 0.301, "b": 2.0})
        assert roles.looks_like_idf_table({"a": np.float64(0.7)})


class TestTheirModelObjectIsLoadedFromTheFile:
    def _module(self):
        source = (
            "import pickle\n"
            "class Encoder:\n"
            "    def __init__(self, width=8):\n"
            "        self.W = None\n"
            "    def load(self, path):\n"
            "        with open(path, 'rb') as f:\n"
            "            self.W = pickle.load(f)[0]\n"
            "    def __call__(self, x):\n"
            "        return x @ self.W\n"
            "class Plain:\n"
            "    def __init__(self, rows):\n"
            "        self.rows = rows\n"
        )
        module = types.ModuleType("theirs")
        exec(compile(source, "theirs", "exec"), module.__dict__)
        return module

    def test_a_zero_argument_class_with_a_load_is_built_around_the_file(self, repository):

        # Reaches the SDK, which is not a declared dependency of this package
        # and which CI installs without. Named rather than failed for; a skip
        # here is not a pass, it is a missing install.
        pytest.importorskip("cogbench")

        path = repository / "w.pkl"
        W = np.arange(512 * 8, dtype=np.float32).reshape(512, 8)
        with open(path, "wb") as stream:
            pickle.dump([W, np.zeros((1, 8))], stream)

        found = roles.loaded_model([self._module()], path)

        assert found is not None
        label, instance = found
        assert label == "theirs.Encoder"
        assert np.array_equal(instance(np.eye(512, dtype=np.float32)[:2]), W[:2])

    def test_a_class_that_cannot_encode_after_loading_is_not_the_model(self, repository):

        # Reaches the SDK, which is not a declared dependency of this package
        # and which CI installs without. Named rather than failed for; a skip
        # here is not a pass, it is a missing install.
        pytest.importorskip("cogbench")

        path = repository / "w.pkl"
        with open(path, "wb") as stream:
            pickle.dump("not a list", stream)

        # `"not a list"[0]` is `"n"`, so their `load` does not raise; the
        # call on a descriptor is what fails, and that is what decides.
        assert roles.loaded_model([self._module()], path) is None


class TestAStoreHasToHaveKeptTheImages:
    """CogFinder's `LanguageModels.generate_letter(ids, descriptors)` returned
    a letter and bound as the prepare step under the old "anything at all"
    rule; the search branch then reported that nothing took the query."""

    def test_objects_dicts_and_matrices_that_hold_the_ids_pass(self):
        ids = [3, 1, 2]
        check = roles.looks_like_store_of(ids)

        class Store:
            def __init__(self):
                self.image_ids = [1, 2, 3]
                self.embeddings = np.zeros((3, 8))

        assert check(Store())
        assert check({1: np.zeros(8), 2: np.zeros(8), 3: np.zeros(8)})
        assert check(([3, 1, 2], np.zeros((3, 8))))
        assert check({"ids": np.array([1, 2, 3]), "rows": np.zeros((3, 8))})

    def test_a_value_that_forgot_the_images_is_not_a_store(self):
        check = roles.looks_like_store_of([1, 2, 3])

        assert not check("q")
        assert not check(None)
        assert not check({1: 0.5, 2: 0.5})
        assert not check(np.zeros((5, 8)))


class TestAConstantEmbedderIsRefused:
    """CogFinder's `tokenize -> embed_caption` binds too: `embed_caption`
    re-tokenizes the token list into one word no vocabulary has, so every
    caption came back as the zero vector and the chain passed."""

    def test_every_row_the_same_fails_the_text_case(self):
        from language_search_benchmark.roles import accepts

        class Case:
            kind = "text"
            captions = ["a", "b", "c"]

        class Composed:
            def __init__(self, *_): pass
            def covered_kinds(self): return {"text"}

        outputs = [{"ok": True, "embeddings": np.zeros((3, 8))}]
        import language_search_benchmark.roles as R
        ran = {}
        def fake_run(adapter, cases):
            return outputs
        original_run, original_adapt = R.__dict__.get("run_with_adapter"), None
        from language_search_benchmark import drivers, adapters, discovered
        saved = (drivers.run_with_adapter, adapters.adapt_search, discovered.DiscoveredSearch)
        drivers.run_with_adapter = fake_run
        adapters.adapt_search = lambda composed: composed
        discovered.DiscoveredSearch = Composed
        try:
            ok, detail = accepts({"text": ()}, [Case()], {})
        finally:
            drivers.run_with_adapter, adapters.adapt_search, discovered.DiscoveredSearch = saved
        assert not ok
        assert "same vector" in detail


class TestThePrepareStepIsHandedTheFormItBoundWith:
    """Bagel's `CaptionImageQuery(image_embeddings, image_ids)` constructs with
    the arguments in either order, and the search bound it on the projected
    matrix first. A scored run that always handed `(ids, descriptors)` built
    a store their own `search` could not use."""

    def test_the_forms_keep_their_indices(self):
        ids = [3, 1]
        raw = np.zeros((2, 512))
        without = roles.prepare_forms(ids, raw)
        with_projection = roles.prepare_forms(ids, raw, np.ones((2, 8)))

        assert [type(a).__name__ for a, _ in without] == ["list", "ndarray"]
        assert len(without) == 2 and len(with_projection) == 4
        assert with_projection[:2] == without or all(
            (x is y) or np.array_equal(x, y)
            for pair_a, pair_b in zip(with_projection[:2], without)
            for x, y in zip(pair_a, pair_b)
        )
        assert with_projection[3][0].shape == (2, 8)
        assert with_projection[3][1] == ids

    def test_the_scored_run_uses_the_bound_form(self):

        # Reaches the SDK, which is not a declared dependency of this package
        # and which CI installs without. Named rather than failed for; a skip
        # here is not a pass, it is a missing install.
        pytest.importorskip("cogbench")

        from cogbench.pipeline import Candidate
        from language_search_benchmark.discovered import DiscoveredSearch

        seen = {}

        class Store:
            def __init__(self, rows, ids):
                seen["rows"] = np.asarray(rows)
                seen["ids"] = list(ids)

        project = Candidate("theirs.project", lambda d: np.asarray(d)[:, :8], "theirs")
        build = Candidate("theirs.Store", Store, "theirs", form=3)
        composed = DiscoveredSearch({"image": (project,), "prepare": (build,)}, {})

        composed.prepare_database([5, 6], np.ones((2, 512)))

        assert seen["rows"].shape == (2, 8)
        assert seen["ids"] == [5, 6]


class TestMissingWeightsGuidance:
    def test_it_keeps_weights_out_of_git_and_names_the_local_sync_flow(self, repository):
        _write(
            repository,
            "training.py",
            "def train(model):\n    model.save('results/modelweights.pkl')\n",
        )
        _write(repository, ".gitignore", "*.pkl\n")

        assert roles.weights_diagnostic(repository) == (
            "overall withheld: the image side has no trained weights to measure. "
            "Your training.py saves to results/modelweights.pkl (training.py:2), and it "
            "matches .gitignore line 1. Keep that weights file out of git. Then run "
            "`cogworks run` locally and `cogworks sync`; the hosted run will fetch the "
            "weights the local run used."
        )


class TestPathBackedTextSetup:
    def test_two_repository_constructors_may_build_an_internal_idf_embedder(self):

        # Reaches the SDK, which is not a declared dependency of this package
        # and which CI installs without. Named rather than failed for; a skip
        # here is not a pass, it is a missing install.
        pytest.importorskip("cogbench")

        from cogbench.pipeline import Candidate, _fits_of

        captions = ["red kite", "blue boat"]

        class CourseData:
            def __init__(self, captions_path, descriptors_path):
                self.paths = (captions_path, descriptors_path)

            def get_all_captions(self):
                return captions

        class CaptionEmbedder:
            def __init__(self, course_data):
                self.course_data = course_data

            def __call__(self, caption):
                row = np.zeros(8, dtype=np.float32)
                row[0] = 1.0 if caption.startswith("red") else -1.0
                return row

        role = roles.search_role(
            captions,
            captions,
            np.zeros((2, 512), dtype=np.float32),
            [1, 2],
            captions[0],
            1,
        )
        pool = {
            "coco_json_path": Path("captions.json"),
            "resnet_features_path": Path("descriptors.pkl"),
        }
        candidates = [
            Candidate("theirs.CourseData", CourseData, "theirs"),
            Candidate("theirs.CaptionEmbedder", CaptionEmbedder, "theirs"),
        ]

        fits, failed = _fits_of(role, candidates, pool, [])

        assert failed is None
        assert [name for name, _ in fits] == ["course_data", "text_embedder"]
        assert isinstance(pool["text_embedder"], CaptionEmbedder)
        assert "idfs" not in pool


class TestWeightsUsedRecord:
    def test_the_record_names_the_one_weights_file_the_run_loaded(self, repository):

        # Reaches the SDK, which is not a declared dependency of this package
        # and which CI installs without. Named rather than failed for; a skip
        # here is not a pass, it is a missing install.
        pytest.importorskip("cogbench")

        from cogbench.resolve import from_spec
        from language_search_benchmark.plugins import LanguageSearchBenchmark

        path = repository / "trained.npy"
        np.save(path, np.zeros((512, 8), dtype=np.float32))
        _write(
            repository,
            "submission.py",
            "import numpy as np\n"
            "def embed_caption(caption):\n"
            "    row = np.zeros(8, dtype=float)\n"
            "    row[0] = 1.0 if caption.startswith('red') else -1.0\n"
            "    return row\n",
        )

        class Spec:
            def __init__(self):
                self.chain_role = roles.text_branch(["red kite", "blue boat"])
                self.fixture = (["red kite", "blue boat"],)
                self.accepts = lambda steps, *_: (True, "ok")
                self.arrangements = None
                self.hints = ()
                self.extras = {}
                self.identities = ()
                self.resource_files = {}
                self.factories = None
                self.readers = 0
                self.expects = "ok"
                plugin = LanguageSearchBenchmark()
                plugin._discovery_extras = {}
                self.prepare = plugin._weights_of

        submission = from_spec(repository, Spec())

        assert submission.to_dict()["weightsUsed"] == ["trained.npy"]


#: A team whose image side is an object rather than a matrix: a class that
#: builds for free, fills itself from their own pickle, and encodes
#: afterwards. Bagel's shape, at the size a unit test wants.
AN_ENCODER_OF_THEIR_OWN = '''
import pickle


class ImageToCaption:
    def __init__(self):
        self.W = None

    def load(self, path):
        with open(path, "rb") as stream:
            self.W = pickle.load(stream)[0]

    def __call__(self, descriptors):
        return descriptors @ self.W


def to_captions(descriptors, weights_model):
    return weights_model(descriptors)
'''


def _weights_file(root: Path, name: str, fill: float) -> np.ndarray:
    """One saved projection, filled so the file it came from can be read off."""

    W = np.full((512, roles.MIN_WIDTH), fill, dtype=np.float32)
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as stream:
        pickle.dump([W, np.zeros((1, roles.MIN_WIDTH), dtype=np.float32)], stream)
    return W


def _image_spec(plugin):
    """The two hooks under test, wired the way `plugins.discovery` wires them."""

    from cogbench.pipeline import Role, Stage

    rows = np.eye(512, dtype=np.float32)[:2]

    class Spec:
        chain_role = Role(
            "search",
            (
                Stage(
                    "image",
                    produces=lambda v: (
                        isinstance(v, np.ndarray) and v.shape == (2, roles.MIN_WIDTH)
                    ),
                    extras=("W", "weights_model"),
                ),
            ),
        )
        fixture = (rows,)
        accepts = staticmethod(lambda steps, *_: (True, "ok"))
        arrangements = None
        hints = ()
        extras: dict = {}
        identities = ()
        resource_files: dict = {}
        factories = None
        readers = 0
        expects = "ok"
        prepare = plugin._weights_of
        construct = plugin._weights_model_for

    return Spec(), rows


class TestOnePluginReadingTwoRepositories:
    """Which repository's weights a reading loads, when one plugin read both.

    The corpus tool resolves several repositories in turn with one plugin, and
    a submission stays usable afterwards: `Submission.fresh` reads the
    repository again for every scored run. So the question is not which
    weights the plugin last saw, it is which weights this submission was
    resolved against. The answer has to live with the submission, which is why
    `prepare` hands the file back as data and `construct` reads it out of the
    pool its own reading was given.
    """

    def _resolved(self, plugin, root):
        from cogbench.resolve import from_spec

        spec, rows = _image_spec(plugin)
        return from_spec(root, spec), rows

    def test_a_later_repository_does_not_change_what_an_earlier_one_loads(self, repository):
        pytest.importorskip("cogbench")

        from language_search_benchmark.plugins import LanguageSearchBenchmark

        first, second = repository / "first", repository / "second"
        theirs = _weights_file(first, "trained.pkl", 1.0)
        _write(first, "submission.py", AN_ENCODER_OF_THEIR_OWN)
        _weights_file(second, "trained.pkl", 7.0)
        _write(second, "submission.py", AN_ENCODER_OF_THEIR_OWN)

        plugin = LanguageSearchBenchmark()
        plugin._discovery_extras = {}
        one, rows = self._resolved(plugin, first)
        assert one.ready, one.verdict.headline
        two, _ = self._resolved(plugin, second)
        assert two.ready, two.verdict.headline

        again = one.fresh()
        try:
            answer = again.chain[0].bound(rows)
        finally:
            again.close()

        assert np.array_equal(answer, rows @ theirs)

    def test_a_repository_with_no_weights_does_not_empty_an_earlier_one(self, repository):
        pytest.importorskip("cogbench")

        from language_search_benchmark.plugins import LanguageSearchBenchmark

        first, second = repository / "first", repository / "second"
        theirs = _weights_file(first, "trained.pkl", 1.0)
        _write(first, "submission.py", AN_ENCODER_OF_THEIR_OWN)
        # Committed nothing. Two of the four audited 2026 repositories are
        # this, so it is the ordinary neighbour of a repository that did.
        _write(second, "submission.py", AN_ENCODER_OF_THEIR_OWN)

        plugin = LanguageSearchBenchmark()
        plugin._discovery_extras = {}
        one, rows = self._resolved(plugin, first)
        assert one.ready, one.verdict.headline
        two, _ = self._resolved(plugin, second)
        assert not two.ready

        again = one.fresh()
        try:
            answer = again.chain[0].bound(rows)
        finally:
            again.close()

        assert np.array_equal(answer, rows @ theirs)

    def test_the_matrix_the_pool_carries_is_the_one_the_model_was_filled_from(
        self, repository
    ):
        pytest.importorskip("cogbench")

        from language_search_benchmark.plugins import LanguageSearchBenchmark

        theirs = _weights_file(repository, "trained.pkl", 3.0)
        _write(repository, "submission.py", AN_ENCODER_OF_THEIR_OWN)

        plugin = LanguageSearchBenchmark()
        plugin._discovery_extras = {}
        data = plugin._weights_of(repository)

        assert np.array_equal(data["W"], theirs)
        with plugin._weights_model_for(repository, _modules_of(repository), data) as built:
            assert np.array_equal(built["weights_model"].W, data["W"])

    def test_the_plugin_keeps_nothing_about_the_repository_it_just_read(self, repository):
        # Anything kept here is a second answer to "which projection scored",
        # and the next repository read overwrites it while the first submission
        # is still being scored. The matrix, the file it came from and the
        # sentence the run leads with all go back as data instead.
        from language_search_benchmark.plugins import LanguageSearchBenchmark

        _weights_file(repository, "trained.pkl", 3.0)
        _write(repository, "submission.py", AN_ENCODER_OF_THEIR_OWN)

        plugin = LanguageSearchBenchmark()
        plugin._discovery_extras = {}
        data = plugin._weights_of(repository)

        assert "W" not in plugin._discovery_extras
        assert "weights_path" not in plugin._discovery_extras
        assert data["weights_report"] == {"path": "trained.pkl", "cited": ""}


#: A team with a caption side and nothing that touches the projection, so the
#: image branch is the one that does not bind. That is the repository whose run
#: page has to say why the image side is unmeasured.
A_CAPTION_SIDE_ONLY = '''
def embed_captions(captions):
    return [[float(len(c))] for c in captions]
'''

#: The same team with an image side and a database, both through their own
#: encoder. `_weights_consumed` publishes a receipt only when both halves read
#: the file the run retained.
AN_IMAGE_SIDE_AND_A_DATABASE = AN_ENCODER_OF_THEIR_OWN + '''

def build_store(descriptors, weights_model):
    return {"vectors": weights_model(descriptors), "ids": [0, 1]}
'''

#: And the same team whose database projected with something of its own, which
#: is the case the receipt is withheld for.
AN_IMAGE_SIDE_AND_A_DATABASE_OF_ITS_OWN = AN_ENCODER_OF_THEIR_OWN + '''

def build_store(descriptors):
    return {"vectors": descriptors, "ids": [0, 1]}
'''


def _branched_spec(plugin, branches, weights_consumed=None):
    """A role of several surfaces, wired through the plugin's own two hooks.

    The shape week 3 actually resolves: one optional image branch beside
    another surface, over one shared pool. The stages and the repositories are
    miniature; the hooks, the search and the receipt decision are the real
    ones.
    """

    from cogbench.discovery_spec import DiscoverySpec
    from cogbench.pipeline import Role, Stage

    rows = np.eye(512, dtype=np.float32)[:2]
    image = Role(
        "image",
        (
            Stage(
                "image",
                produces=lambda v: (
                    isinstance(v, np.ndarray) and v.shape == (2, roles.MIN_WIDTH)
                ),
                extras=("W", "weights_model"),
            ),
        ),
        fixture=(rows,),
        optional=True,
    )
    text = Role(
        "text",
        (Stage("text", produces=lambda v: isinstance(v, list) and bool(v)),),
        fixture=(["a caption"],),
    )
    prepare = Role(
        "prepare",
        (
            Stage(
                "prepare",
                produces=lambda v: isinstance(v, dict) and "ids" in v,
                extras=("W", "weights_model"),
            ),
        ),
        fixture=(rows,),
        optional=True,
    )
    by_name = {"image": image, "text": text, "prepare": prepare}
    return DiscoverySpec(
        chain_role=Role("search", (), branches=tuple(by_name[n] for n in branches)),
        fixture=(rows,),
        accepts=lambda chains, *_: (True, "ok"),
        arrangements=None,
        prepare=plugin._weights_of,
        construct=plugin._weights_model_for,
        weights_consumed=weights_consumed,
    ), rows


def _plugin():
    from language_search_benchmark.plugins import LanguageSearchBenchmark

    plugin = LanguageSearchBenchmark()
    plugin._discovery_extras = {}
    return plugin


def _unmeasured(plugin):
    """What `score` would lead with, off the submission last handed over."""

    case = types.SimpleNamespace(kind="retrieval")
    return plugin._unmeasured_image_side([{"ok": False, "error": ""}], [case])


class TestTheWeightReportBelongsToTheSubmissionItWasReadFor:
    """Which repository the sentence about the image side is about.

    One plugin resolves several repositories in turn, and a submission is
    scored after the next one has been read: `run` calls
    `submission_from_discovery` and then `score`. So the file named, and the
    ambiguity reported, have to come off the submission being scored rather
    than off whichever repository the plugin read last.
    """

    def test_a_later_ambiguous_repository_does_not_describe_an_earlier_one(
        self, repository
    ):
        pytest.importorskip("cogbench")

        from cogbench.resolve import from_spec

        first, second = repository / "first", repository / "second"
        _weights_file(first, "trained.pkl", 1.0)
        _write(first, "submission.py", A_CAPTION_SIDE_ONLY)
        _weights_file(second, "model_tests/testd_50.pkl", 1.0)
        _weights_file(second, "model_tests/test1.pkl", 2.0)
        _write(
            second,
            "get_model_embeddings.py",
            "def main(model):\n"
            "    model.load('model_tests/testd_50.pkl')\n"
            "    model.load('model_tests/test1.pkl')\n",
        )
        _write(second, "submission.py", A_CAPTION_SIDE_ONLY)

        plugin = _plugin()
        spec, _rows = _branched_spec(plugin, ("text", "image"))
        one = from_spec(first, spec)
        two = from_spec(second, spec)
        assert "image" in one.missing and "image" in two.missing

        plugin.submission_from_discovery(one)
        about_the_first = _unmeasured(plugin)
        plugin.submission_from_discovery(two)
        about_the_second = _unmeasured(plugin)

        assert "trained.pkl" in about_the_first
        assert "several files" not in about_the_first
        assert "several files" in about_the_second
        assert "trained.pkl" not in about_the_second

    def test_scoring_the_first_one_last_still_names_its_own_file(self, repository):
        """The same pair, scored in the other order.

        A plugin field would be right in whichever order ends on its own
        repository, so the order that ends on the other one is the test.
        """

        pytest.importorskip("cogbench")

        from cogbench.resolve import from_spec

        first, second = repository / "first", repository / "second"
        _weights_file(first, "trained.pkl", 1.0)
        _write(first, "submission.py", A_CAPTION_SIDE_ONLY)
        _write(second, "submission.py", A_CAPTION_SIDE_ONLY)

        plugin = _plugin()
        spec, _rows = _branched_spec(plugin, ("text", "image"))
        one = from_spec(first, spec)
        # A neighbour that committed nothing, read in between.
        from_spec(second, spec)

        plugin.submission_from_discovery(one)
        note = _unmeasured(plugin)

        assert "trained.pkl" in note
        assert "no trained weights" not in note

    def test_a_repository_with_no_weights_says_so_after_one_that_had_them(
        self, repository
    ):
        pytest.importorskip("cogbench")

        from cogbench.resolve import from_spec

        first, second = repository / "first", repository / "second"
        _weights_file(first, "trained.pkl", 1.0)
        _write(first, "submission.py", A_CAPTION_SIDE_ONLY)
        _write(second, "submission.py", A_CAPTION_SIDE_ONLY)

        plugin = _plugin()
        spec, _rows = _branched_spec(plugin, ("text", "image"))
        from_spec(first, spec)
        two = from_spec(second, spec)

        plugin.submission_from_discovery(two)
        note = _unmeasured(plugin)

        assert "no trained weights" in note
        assert "trained.pkl" not in note


class TestAReceiptNeedsBothHalvesToHaveReadTheFile:
    """`weights_consumed` through the hooks, on a repository with a database.

    The image branch alone cannot answer it: a run whose database projected
    with something else read bytes this benchmark did not retain, and a receipt
    for those bytes would be a claim about a file only half the run used.
    """

    def test_an_image_side_and_a_database_on_one_encoder_publish_receipts(
        self, repository
    ):
        pytest.importorskip("cogbench")

        from cogbench.resolve import from_spec

        _weights_file(repository, "trained.pkl", 1.0)
        _write(repository, "submission.py", AN_IMAGE_SIDE_AND_A_DATABASE)

        plugin = _plugin()
        spec, _rows = _branched_spec(
            plugin, ("image", "prepare"), weights_consumed=plugin._weights_consumed
        )
        submission = from_spec(repository, spec)

        assert submission.ready, submission.verdict.headline
        assert submission.weights_used == ("trained.pkl",)
        assert submission.weights_captured
        assert submission.weights_captured[0]["path"] == "trained.pkl"

    def test_a_database_that_projected_with_its_own_thing_publishes_none(
        self, repository
    ):
        pytest.importorskip("cogbench")

        from cogbench.resolve import from_spec

        _weights_file(repository, "trained.pkl", 1.0)
        _write(repository, "submission.py", AN_IMAGE_SIDE_AND_A_DATABASE_OF_ITS_OWN)

        plugin = _plugin()
        spec, _rows = _branched_spec(
            plugin, ("image", "prepare"), weights_consumed=plugin._weights_consumed
        )
        submission = from_spec(repository, spec)

        assert submission.ready, submission.verdict.headline
        assert submission.weights_used == ("trained.pkl",)
        assert submission.weights_captured is None


def _modules_of(root: Path):
    """Their modules, for a hook that is being called without a reading."""

    import importlib.util

    loaded = []
    for path in sorted(Path(root).glob("*.py")):
        spec = importlib.util.spec_from_file_location(path.stem, path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        loaded.append(module)
    return loaded


class TestRetentionDoesNotMakeASecondReadingAmbiguous:
    """The platform's own copy is not one of their weight files.

    `capture` retains the winner under `<root>/.cogbench/weights/...` with its
    original basename. Enumerating that copy gives it the same citation their
    own source gives the original, so no file wins and a repository that was
    unambiguous the first time is reported as ambiguous the second. Bagel is
    the repository this happened to, and its `model_tests/testd_50.pkl` is
    named at `get_model_embeddings.py:13`.
    """

    def _captured(self, root):
        from cogbench import storage

        def capture(original: Path) -> Path:
            return storage.retain_input(root, original).retained

        return capture

    def test_two_readings_of_one_checkout_choose_the_same_file(self, repository):
        pytest.importorskip("cogbench")

        _weights_file(repository, "model_tests/testd_50.pkl", 1.0)
        _weights_file(repository, "model_tests/test1.pkl", 2.0)
        _write(
            repository,
            "get_model_embeddings.py",
            "def main(model):\n    model.load('model_tests/testd_50.pkl')\n",
        )
        capture = self._captured(repository)

        first = roles.weights_in(repository, capture=capture)
        second = roles.weights_in(repository, capture=capture)

        assert first is not None and second is not None
        assert first.path == second.path == repository / "model_tests/testd_50.pkl"
        assert np.array_equal(first.matrix, second.matrix)
        assert (repository / ".cogbench").is_dir()

    def test_a_repository_that_really_is_ambiguous_still_says_so(self, repository):
        pytest.importorskip("cogbench")

        _weights_file(repository, "model_tests/testd_50.pkl", 1.0)
        _weights_file(repository, "model_tests/test1.pkl", 2.0)
        _write(
            repository,
            "get_model_embeddings.py",
            "def main(model):\n"
            "    model.load('model_tests/testd_50.pkl')\n"
            "    model.load('model_tests/test1.pkl')\n",
        )

        with pytest.raises(roles.AmbiguousWeights):
            roles.weights_in(repository)
