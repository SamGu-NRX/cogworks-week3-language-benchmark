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
