"""What a run tells a student about a surface the search never bound.

The hosted evaluation searches a repository in a sandbox process and scores in
the controller, so the object that bound the branches and the object that
scores them are two different `LanguageSearchBenchmark` instances. Every test
here scores with an instance that has never searched anything, which is the
arrangement production actually runs.

Under that arrangement run 9762 told a student "the image side has no trained
weights to measure." Their weights were committed and readable. The search had
refused the image branch because two files in that repository load as a
(512, D) projection and nothing in their code chose between them -- a reason
the searching instance had already written down and the scoring instance could
not see. It then repeated the same two warnings eight times, once per case
that needed an absent surface.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest

from language_search_benchmark.discovered import DiscoveredSearch, NotBound
from language_search_benchmark.drivers import run_cases
from language_search_benchmark.plugins import LanguageSearchBenchmark, _sentences

from .fixtures.synthetic import PerfectAdapter, Universe


#: A repository whose captions embed and whose database needs the projection,
#: so committing two projections leaves the image branch with nothing to take.
_STUDENT = '''
import numpy as np


class ImageDatabase:
    def __init__(self, image_ids, descriptors, W_embed):
        self.image_ids = list(image_ids)
        self.W_embed = np.asarray(W_embed)

    def descriptor_to_embedding(self, descriptor):
        return np.asarray(descriptor) @ self.W_embed


def embed_captions_batch(texts):
    rows = []
    for text in texts:
        row = np.zeros(8, dtype=float)
        for index, code in enumerate(text.encode("utf-8")[:8]):
            row[index] = code / 255.0
        rows.append(row)
    return np.stack(rows)
'''


@pytest.fixture(scope="module")
def universe():
    return Universe()


@pytest.fixture(scope="module")
def cases(universe):
    return universe.cases()


@pytest.fixture
def repository():
    root = Path(tempfile.mkdtemp()).resolve()
    yield root
    shutil.rmtree(root, ignore_errors=True)


class _NoImageSide(DiscoveredSearch):
    """The real binding object for a repository with only a text branch.

    `DiscoveredSearch.embed_text` runs their function through the SDK's
    runtime pool, and this package declares no dependency on the SDK. What
    these tests exercise is the three branches that are absent, and those
    refuse before any of that, so the one branch that is present is answered
    here instead.
    """

    def __init__(self, universe, notes=None):
        super().__init__({"text": ()}, {}, notes)
        self._universe = universe

    def embed_text(self, captions):
        return PerfectAdapter(self._universe).embed_text(captions)


def _scored(universe, cases, notes=None):
    """Run the driver, then score on an instance that never searched."""

    outputs = run_cases(lambda resources: _NoImageSide(universe, notes), None, cases)
    benchmark = LanguageSearchBenchmark()
    return benchmark.score(outputs, cases), benchmark.last_diagnostics


def _discovery_note(repository):
    """What the week's own prepare hook says about this repository's weights.

    The hook is what discovery calls, and where the sentence is written.
    """

    plugin = LanguageSearchBenchmark()
    plugin._discovery_extras = {}
    plugin._weights_of(repository)
    return plugin._image_note({"image": "nothing accepted the input the benchmark passes"})


class TestTheCauseSurvivesTheInstanceBoundary:
    def test_two_candidate_files_are_not_reported_as_missing_weights(
        self, universe, cases, repository
    ):
        for name in ("data/W_embed.npy", "models/W_other.npy"):
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.ones((512, 200), dtype=np.float32))

        note = _discovery_note(repository)
        _metrics, diagnostics = _scored(universe, cases, {"image": note})

        headline = diagnostics[0]
        assert headline.startswith("overall withheld: several files in this repository")
        assert "no trained weights" not in " ".join(diagnostics)
        assert "no trained image projection" not in " ".join(diagnostics)
        assert any("data/W_embed.npy, models/W_other.npy" in note for note in diagnostics)
        # The action, which is the half of the sentence the wire's
        # 240-character cap splits off into its own entry.
        assert any("Load one by name in the script you run" in note for note in diagnostics)

    def test_two_load_calls_are_not_reported_as_no_load_call(
        self, universe, cases, repository
    ):
        """The other way into the same refusal, which needs the other advice.

        `roles.AmbiguousWeights` is raised both when their code names none of
        the candidates and when it names more than one. Telling the second
        student that nothing in their code loads one by name is false, and
        the instruction that follows is work they have already done.
        """

        for name, module in (
            ("data/W_embed.npy", "train.py"),
            ("models/W_other.npy", "embed.py"),
        ):
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.ones((512, 200), dtype=np.float32))
            (repository / module).write_text(
                "import numpy as np\nW = np.load({!r})\n".format(Path(name).name)
            )

        note = _discovery_note(repository)
        _metrics, diagnostics = _scored(universe, cases, {"image": note})

        assert diagnostics[0].endswith(
            "your code loads more than one of them, so this run could not tell "
            "which one to score."
        )
        assert any(
            "data/W_embed.npy at train.py:2" in note
            and "models/W_other.npy at embed.py:2" in note
            for note in diagnostics
        )
        assert "nothing in your code loads one of them" not in " ".join(diagnostics)

    def test_genuinely_missing_weights_keep_the_local_sync_instruction(
        self, universe, cases, repository
    ):
        (repository / "training.py").write_text(
            "def train(model):\n    model.save('results/modelweights.pkl')\n"
        )
        (repository / ".gitignore").write_text("*.pkl\n")

        note = _discovery_note(repository)
        _metrics, diagnostics = _scored(universe, cases, {"image": note})

        joined = " ".join(diagnostics)
        assert diagnostics[0].startswith(
            "overall withheld: the image side has no trained weights to measure."
        )
        assert "results/modelweights.pkl" in joined
        assert "cogworks sync" in joined

    def test_a_refusal_this_cannot_explain_states_only_what_was_observed(
        self, universe, cases
    ):
        """No note reached the run, so no cause is asserted.

        The image branch can be refused for reasons that are not about
        weights at all. Naming one anyway is what produced the false report.
        """

        _metrics, diagnostics = _scored(universe, cases)

        assert diagnostics[0] == (
            "overall withheld: this run found no function it could use to turn "
            "image descriptors into vectors."
        )
        # The contract is the only thing left to act on, so it is said.
        assert any("in the same space as your caption vectors" in n for n in diagnostics)
        assert "weights" not in " ".join(diagnostics)


class TestEachAbsenceIsSaidOnce:
    def test_the_eight_cases_that_need_a_missing_surface_produce_two_findings(
        self, universe, cases
    ):
        """Four retrieval cases and four search cases, two absent surfaces."""

        _metrics, diagnostics = _scored(universe, cases)
        binding = [
            note for note in diagnostics if "this run found no function" in note
        ]

        assert len(binding) == 2
        assert sum("image descriptors into vectors" in note for note in binding) == 1
        assert sum("searchable database" in note for note in binding) == 1
        assert len(set(diagnostics)) == len(diagnostics)
        # The second finding is a different problem, not more of the first.
        assert any(note.startswith("Separately, ") for note in diagnostics)

    def test_the_image_side_leads_and_its_numbers_are_withheld(self, universe, cases):
        metrics, _diagnostics = _scored(universe, cases)

        assert metrics["text_mrr"] > 0.9
        for key in ("overall", "retrieval_mrr", "search_mrr", "retrieval_median_rank"):
            assert key not in metrics


class TestOnlyTheSearchSideIsAbsent:
    """The image side bound, so retrieval is measured and search is not."""

    def test_the_run_says_which_half_still_counts(self, universe, cases):
        class NoDatabase(DiscoveredSearch):
            def __init__(self):
                super().__init__({"text": (), "image": ()}, {})

            def embed_text(self, captions):
                return PerfectAdapter(universe).embed_text(captions)

            def embed_images(self, descriptors):
                return PerfectAdapter(universe).embed_images(descriptors)

        outputs = run_cases(lambda resources: NoDatabase(), None, cases)
        benchmark = LanguageSearchBenchmark()
        metrics = benchmark.score(outputs, cases)
        diagnostics = benchmark.last_diagnostics

        assert diagnostics[0].startswith(
            "overall withheld: this run found no function it could use to build"
        )
        assert any("your caption and retrieval scores are" in n for n in diagnostics)
        assert metrics["retrieval_mrr"] > 0.9
        assert "search_mrr" not in metrics
        assert "overall" not in metrics


class TestAComponentThatRanAndFailedIsStillItsOwnFinding:
    def test_their_own_error_is_reported_per_component(self, universe, cases):
        """A bound function that raises is not an absent surface.

        Grouping the repeated warnings must not swallow the case this is
        for: their code ran, it broke, and the message is theirs.
        """

        class Raises(_NoImageSide):
            def embed_images(self, descriptors):
                raise ValueError("W_embed.npy is a (200, 512) matrix, not (512, 200)")

        outputs = run_cases(lambda resources: Raises(universe), None, cases)
        benchmark = LanguageSearchBenchmark()
        metrics = benchmark.score(outputs, cases)
        diagnostics = benchmark.last_diagnostics

        assert metrics["retrieval_mrr"] == 0.0
        assert any(
            note.startswith("retrieval component scored 0") and "not (512, 200)" in note
            for note in diagnostics
        )


class TestTheNoteTravelsOnTheOutput:
    def test_the_driver_records_which_surface_was_absent(self, universe, cases):
        outputs = run_cases(
            lambda resources: _NoImageSide(universe, {"image": "two files competed."}),
            None,
            cases,
        )
        retrieval = next(o for o in outputs if o["kind"] == "retrieval")
        search = next(o for o in outputs if o["kind"] == "search")

        assert retrieval["notBound"] == {
            "surface": "image",
            "note": "two files competed.",
        }
        assert search["notBound"]["surface"] == "prepare"

    def test_a_branch_with_no_note_falls_back_to_what_the_search_saw(self, universe):
        composed = DiscoveredSearch({"text": ()}, {})

        with pytest.raises(NotBound) as raised:
            composed.embed_images(np.ones((2, 512)))

        assert raised.value.surface == "image"
        assert raised.value.note.startswith(
            "this run found no function it could use to turn image descriptors"
        )


class TestWhichAnswerDiscoveryGives:
    def test_weights_that_loaded_but_bound_nothing_name_their_file(self):
        plugin = LanguageSearchBenchmark()
        plugin._discovery_extras = {"W": object(), "weights_path": "data/W_embed.npy"}

        note = plugin._image_note({"image": "nothing accepted the input"})

        assert note.startswith("your trained weights loaded from data/W_embed.npy")
        assert "nothing accepted the input" in note

    def test_a_long_refusal_detail_is_cut_at_a_word(self):
        """The wire caps a diagnostic at 240 characters and the detail is
        discovery's, not ours; a sentence that stops mid-word reads as a
        fault in the report."""

        plugin = LanguageSearchBenchmark()
        plugin._discovery_extras = {"W": object(), "weights_path": "data/W_embed.npy"}

        note = plugin._image_note({"image": "overlong " * 60})
        tail = note.split("Why: ", 1)[1]

        assert tail.endswith("...")
        assert not tail[:-3].endswith("overlon")
        assert max(len(sentence) for sentence in _sentences(note)) <= 240

    def test_a_bound_image_branch_has_nothing_to_explain(self):
        plugin = LanguageSearchBenchmark()
        plugin._discovery_extras = {}

        assert plugin._image_note({"search": "no function answered a query"}) is None


class TestASearchThatReallyRefusedTwoCandidates:
    """The whole chain, through the SDK, on a repository that has both files.

    The lightweight cases above hand `DiscoveredSearch` a note. This one
    makes the search write it: `resolve` calls the week's own prepare hook,
    the hook refuses two candidates, and what the composed object raises is
    the sentence that refusal produced.
    """

    def test_the_composed_object_raises_the_reason_the_search_recorded(self, repository):
        # Reaches the SDK, which is not a declared dependency of this package
        # and which CI installs without. Named rather than failed for; a skip
        # here is not a pass, it is a missing install.
        pytest.importorskip("cogbench")

        from cogbench import resolve as resolve_module

        from language_search_benchmark import roles

        (repository / "database.py").write_text(_STUDENT)
        for name in ("data/W_embed.npy", "models/W_other.npy"):
            path = repository / name
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, np.eye(512, 8))

        searching = LanguageSearchBenchmark()
        searching._discovery_extras = {}
        descriptors = np.eye(2, 512)
        found = resolve_module.resolve(
            repository,
            chain_role=roles.search_role(
                captions=["one", "two two"],
                corpus=["one", "two two"],
                descriptors=descriptors,
                image_ids=[11, 22],
                query="one",
                k=2,
            ),
            fixture=(["one", "two two"],),
            accepts=lambda chain, *_: (True, ""),
            arrangements=None,
            prepare=searching._weights_of,
            weights_consumed=searching._weights_consumed,
            benchmark="language-search",
        )

        assert found.ready, found.verdict.headline
        assert "image" in found.missing
        composed = searching.submission_from_discovery(found)

        with pytest.raises(NotBound) as raised:
            composed.embed_images(descriptors)

        assert raised.value.surface == "image"
        assert "data/W_embed.npy, models/W_other.npy" in raised.value.note
        assert "no trained" not in raised.value.note
