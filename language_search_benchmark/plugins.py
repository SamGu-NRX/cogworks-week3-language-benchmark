"""The ``cogworks.benchmarks.v2`` plugin for the language-search benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, Any, Dict, List, Optional, Sequence

from . import __version__, perturb
from .contracts import Resources
from .datasets import (
    SEARCH_K,
    CacheStatus,
    artifact_status,
    build_resources,
    load_manifest,
    materialize_cases,
    tier_status,
)
from .metrics import component_scores

#: Set to "0" (the hosted evaluation does) to skip the showcase lines.
SHOWCASE_ENV = "COGWORKS_SHOWCASE"


def _sentences(note: str) -> List[str]:
    """Split a diagnostic at sentence ends, keeping abbreviations like `e.g.`
    and dotted paths whole: a boundary is a period followed by a space and a
    capital letter or a backtick."""

    import re

    return [part for part in re.split(r"(?<=\.) (?=[A-Z`])", note) if part]


class LanguageSearchBenchmark:
    """Load, execute, and score the three retrieval components."""

    benchmark_id = "language-search"
    benchmark_version = 1
    contract_version = "cogworks.submissions.v2"
    plugin_version = __version__
    dataset_version = "coco-glove-manifests-v1"
    #: retrieval-v3: `search_mrr` changed meaning, and `search_chance` is new.
    #:
    #: Under retrieval-v2 `search_mrr` scored the verbatim rung alone, which
    #: made it a second reading of `retrieval_mrr`: same queries, same pool,
    #: differing only where gold fell past the k-th result. Measured on the
    #: reference, that was exact equality on the test tier and 0.00137 on the
    #: evaluation tier. It now averages all four query rewrites, so it
    #: measures whether search survives the queries a person types. `overall`
    #: moves as a result. This is not backward compatible, and the version is
    #: what keeps a v2 run from being read as if it reported the same thing.
    #: retrieval-v4: the verbatim probes left the scored aggregates.
    #:
    #: Their queries are captions read straight out of the annotations file
    #: the submission is handed, so a dictionary built from that file answers
    #: them. Measured against the real scorer on the test tier, a submission
    #: with no embedding at all scored 1.0000 on `retrieval_mrr` and 0.4822
    #: overall, against a reference at 0.6925. Stripping gold from the
    #: payload does not close it and it was already stripped: the sandbox
    #: recomputes the gold row from things the contract requires.
    #:
    #: `retrieval_mrr` and `search_mrr` now average the three rewritten
    #: rungs. The verbatim probes still run and are published as
    #: `retrieval_mrr_verbatim` and `search_mrr_verbatim`, because the gap
    #: between them and the scored number is the clearest reading here.
    #:
    #: Measured cost to an honest submission: overall 0.6925 to 0.6594 on the
    #: test tier, 0.4241 to 0.4097 on evaluation. Cost to the memorizer:
    #: 0.4822 to 0.0811. The instrument separates them 2.7 times better.
    #:
    #: See docs/decisions/week3-verbatim-probes.md.
    scorer_version = "retrieval-v4"
    primary_metric = "overall"

    metric_labels = {
        "overall": "Overall",
        "text_mrr": "Text MRR",
        "retrieval_mrr": "Retrieval MRR",
        "search_mrr": "Search MRR",
        "retrieval_recall_at_1": "Recall@1",
        "retrieval_recall_at_5": "Recall@5",
        "retrieval_recall_at_10": "Recall@10",
        "retrieval_median_rank": "Median rank",
        "chance_mrr": "Chance MRR",
        "text_chance": "Text chance MRR",
        "search_chance": "Search chance MRR",
        "search_mrr_verbatim": "Search MRR, caption unchanged (not scored)",
        "retrieval_mrr_verbatim": "Retrieval MRR, caption unchanged (not scored)",
        "search_mrr_keywords": "Search MRR, keywords only",
        "search_mrr_truncated": "Search MRR, first three words",
        "search_mrr_typo": "Search MRR, one typo",
    }
    lower_is_better = {"retrieval_median_rank"}

    #: What each metric measures, in the course's own vocabulary, and which
    #: part of the capstone it comes from.
    #:
    #: A number a student cannot trace back to something they were taught is a
    #: black box, and a black box teaches nothing. Every entry names the piece
    #: of the assignment it corresponds to, using the words CogWeb uses --
    #: "IDF-weighted GloVe", "W_embed", "margin ranking loss", "confusor" --
    #: rather than ours. Source:
    #: docs/cogweb/pages/Language/SemanticImageSearch.md.
    metric_help = {
        "overall": (
            "The mean of the three component scores below, which are weighted "
            "equally on purpose: the capstone is three pieces that must agree on "
            "one embedding space, and a submission strong at two of them has not "
            "built the thing. This is the leaderboard number."
        ),
        "text_mrr": (
            "Given one caption, how highly does another caption of the same image "
            "rank against captions of other images? This tests the IDF-weighted "
            "GloVe sum on its own, before any image is involved. Weak here means "
            "the caption embedding is the problem -- check that IDF is computed "
            "across every caption in the dataset and that an unseen word gets IDF 0."
        ),
        "retrieval_mrr": (
            "Given a caption embedding, how highly does its own image rank among "
            "all images by cosine similarity? This is what W_embed was trained "
            "for: the margin ranking loss pushes a true image's embedding toward "
            "its caption's and a confusor's away. Weak here with strong text MRR "
            "means the two embeddings are not living in the same space."
        ),
        "search_mrr": (
            "The application end to end: a query string in, ranked image ids out, "
            "through whatever database the submission built. This is the average "
            "over four versions of every query (the caption unchanged, its "
            "keywords only, its first three words, and one with a typo), so it "
            "asks whether search holds up on what a person would actually type "
            "rather than only on a caption handed back verbatim. The four are "
            "listed separately further down, so a low score here can be traced "
            "to the rewrite that caused it. Weak here while the two above are "
            "strong points at the plumbing, meaning the database, the id "
            "mapping, or the query path, rather than at the embeddings."
        ),
        "retrieval_recall_at_1": (
            "How often the caption's own image is the single top match. The "
            "strictest reading of the retrieval task."
        ),
        "retrieval_recall_at_5": (
            "How often the caption's own image is in the top five. A search "
            "interface shows several results, so this is closer to what a user "
            "of the finished application experiences."
        ),
        "retrieval_recall_at_10": (
            "How often the caption's own image is in the top ten. Read alongside "
            "recall@1: a large gap means the right image is being found but not "
            "ranked first, which is a margin problem rather than an embedding one."
        ),
        "retrieval_median_rank": (
            "The middle rank of the correct image across all queries. Reported "
            "because a mean is dominated by a handful of catastrophic misses, "
            "while the median says what a typical query does."
        ),
        "chance_mrr": (
            "What ranking the images at random scores. This is the floor for "
            "retrieval MRR. Text MRR and search MRR each have their own floor, "
            "listed separately, because they are not asking the same question."
        ),
        "text_chance": (
            "The floor for text MRR specifically, which is a different number "
            "from the image floor above: this one ranks captions among "
            "captions. The two differ by roughly four times on the hidden set, "
            "so reading text MRR against the image floor flatters it."
        ),
        "search_chance": (
            "The floor for search MRR. It sits below the retrieval floor "
            "because search returns a fixed number of results and an image "
            "past the end of that list scores nothing, while retrieval ranks "
            "the whole pool. On the hidden set the search floor is about "
            "0.0064 against the retrieval floor's 0.0102."
        ),
        "search_mrr_verbatim": (
            "The queries exactly as the captions were written. Reported and "
            "not scored, because those captions are in the file your code is "
            "handed: a submission that looked the query up in that file "
            "instead of embedding it would answer every one of them. Read it "
            "next to the scored search number. Close together means your "
            "embedding is doing the work. Far apart, with this one high, "
            "means the query text is being matched rather than its meaning."
        ),
        "retrieval_mrr_verbatim": (
            "The same reading for retrieval: the captions unchanged, "
            "reported and not scored, for the same reason. This is the one "
            "that mattered most. A submission with no embedding at all "
            "scored a perfect 1.0000 here while scoring at the floor on "
            "every rewritten query, because it was reading the answer out of "
            "the captions file rather than working it out."
        ),
        "search_mrr_keywords": (
            "The same queries with the stopwords removed, which is closer to "
            "what someone types into a search box. IDF weighting is what "
            "should make those words nearly free, so this number sitting well "
            "below the unchanged captions points at the weighting rather than "
            "at the embedding. Which way it moves is not fixed: on the "
            "reference it came out 6 percent above the unchanged captions on "
            "the hidden set and 7 percent below on the practice set, so treat "
            "a move of that size as normal and a large drop as a finding."
        ),
        "search_mrr_truncated": (
            "Only the first three content words of each query. This says "
            "whether the score falls off gradually as signal is removed or "
            "drops off a cliff. Expected to be the weakest of the four, and "
            "it is on the reference; a three-word query genuinely carries "
            "less information, so some of this gap is the task and not the "
            "submission."
        ),
        "search_mrr_typo": (
            "One mistyped character per query, on the longest content word. "
            "The word becomes one your vocabulary has never seen, which the "
            "course says should contribute a zero vector rather than raising. "
            "A large drop here means an unseen word is dominating the sum; an "
            "error rather than a low score means it raised."
        ),
    }

    #: What kind of number each one is. The run page reads these and never
    #: reads a metric name, so a benchmark that grows a floor or an unscored
    #: probe gets the right rendering without the page learning about it.
    #:
    #: Three kinds are not the score. A floor is a property of the data, and
    #: drawing "higher is better" on one tells a student to raise a number
    #: they do not control. A reported metric is run deliberately and left
    #: out of the score, which a reader has no way to guess. A diagnostic
    #: describes one component in more detail than the component's own
    #: number does.
    metric_roles = {
        "overall": "scored",
        "text_mrr": "scored",
        "retrieval_mrr": "scored",
        "search_mrr": "scored",
        "text_chance": "floor",
        "chance_mrr": "floor",
        "search_chance": "floor",
        "retrieval_mrr_verbatim": "reported",
        "search_mrr_verbatim": "reported",
        # Plotted rather than tabled. These three are the rung curve, and the
        # curve prints each value beside its point, so a row per rung was the
        # same number twice. "plotted" tells the page they are accounted for
        # elsewhere; a page that does not draw the curve still has them in
        # the payload.
        "search_mrr_keywords": "plotted",
        "search_mrr_truncated": "plotted",
        "search_mrr_typo": "plotted",
        "retrieval_recall_at_1": "diagnostic",
        "retrieval_recall_at_5": "diagnostic",
        "retrieval_recall_at_10": "diagnostic",
        "retrieval_median_rank": "diagnostic",
    }

    #: Which metric each floor or reported number belongs beside. A floor is
    #: the scale its metric sits on, so it reads inline rather than as its
    #: own row; a reported probe means nothing alone and everything next to
    #: the scored number it shadows.
    metric_relations = {
        "text_chance": "text_mrr",
        "chance_mrr": "retrieval_mrr",
        "search_chance": "search_mrr",
        "retrieval_mrr_verbatim": "retrieval_mrr",
        "search_mrr_verbatim": "search_mrr",
    }

    #: Every key `score()` can return, for the shared explainability test in
    #: python/cogbench/tests. That test compared `metric_labels` against
    #: `metric_help` and so could not see a metric missing from both, which
    #: is how `retrieval_mrr_verbatim` reached a run page as the machine-made
    #: title "Retrieval Mrr Verbatim" with no explanation.
    #:
    #: Written out rather than derived by scoring something, because the test
    #: has no course artifacts and building a fixture there would test the
    #: fixture. The cost is that this list is maintained by hand; the tests in
    #: tests/test_search_grid.py compare it against a real scoring run in both
    #: directions, so it cannot rot quietly.
    sample_metric_keys = (
        "overall",
        "text_mrr",
        "text_chance",
        "retrieval_mrr",
        "retrieval_mrr_verbatim",
        "retrieval_recall_at_1",
        "retrieval_recall_at_5",
        "retrieval_recall_at_10",
        "retrieval_median_rank",
        "chance_mrr",
        "search_mrr",
        "search_chance",
        "search_mrr_verbatim",
        "search_mrr_keywords",
        "search_mrr_truncated",
        "search_mrr_typo",
    )

    #: How the run page draws the rung curve. The x axis is the rung's
    #: position in `perturb.RUNGS`, which runs from the caption unchanged to
    #: the furthest rewrite, so it is an ordering rather than a quantity; the
    #: per-point label carries the name the reader needs.
    sweep_axis_label = "how far the query is from the caption"
    sweep_x_key = "rung_index"
    sweep_y_key = "mrr"
    sweep_label_key = "rung"

    def __init__(self) -> None:
        self.last_sweep: List[Dict[str, Any]] = []
        self.last_diagnostics: List[str] = []

    def load_cases(self, tier: str, cache_root: Optional[Path] = None) -> Sequence[Any]:
        manifest = load_manifest(tier)
        resources = build_resources(download=True, build_kv=False)
        return materialize_cases(manifest, resources)

    def run(self, factory: Any, resources: Any, cases: Sequence[Any]) -> List[Dict[str, Any]]:
        from .drivers import error_outputs, instantiate_adapter, run_with_adapter

        try:
            adapter = instantiate_adapter(factory, resources)
        except Exception as error:  # noqa: BLE001 - student code; report, don't crash
            return error_outputs(cases, error)
        outputs = run_with_adapter(adapter, cases)
        if os.environ.get(SHOWCASE_ENV, "1") != "0":
            try:
                from .showcase import print_showcase

                print_showcase(adapter, resources, cases)
            except Exception as error:  # noqa: BLE001 - showcase never fails a run
                print("showcase skipped: {}".format(str(error).splitlines()[0][:160]))
        return outputs

    #: The primary metric for THIS run. `primary_metric` is a class
    #: attribute the runners read; a run whose image side was never
    #: measured has no `overall`, and its primary is `text_mrr`. Set by
    #: `score`, read by the runners through the instance.
    primary_metric_for_run: Optional[str] = None

    def score(self, outputs: Sequence[Dict[str, Any]], cases: Sequence[Any]) -> Dict[str, float]:
        metrics, diagnostics = component_scores(outputs, cases, SEARCH_K)
        for output in outputs:
            for mapping in output.get("mappings", []):
                diagnostics.append("adapter: {}".format(mapping))
        # A submission with no image side (discovery found text but no
        # trained weights) has retrieval and search cases that never ran.
        # The driver scores those as zero with a diagnostic, and an overall
        # averaged over them is a number nobody measured. So the three
        # image-side scores and the overall are withheld, their floors are
        # kept, the primary becomes text_mrr, and a diagnostic that names
        # their own save path leads (docs/design/discovery-v2-brief.md,
        # "Absent weights", decided 2026-09-02).
        withheld = self._withheld(outputs, cases)
        if withheld is not None:
            note, prefixes, primary = withheld
            for key in list(metrics):
                if key == "overall" or any(
                    key == prefix or key.startswith(prefix + "_") for prefix in prefixes
                ):
                    metrics.pop(key, None)
            # One sentence per diagnostic. The run page shows the first entry
            # as the headline and the rest as notes, and the wire caps each
            # entry at 240 characters; the whole note is 320 to 366, so as one
            # entry it was cut mid-word on exactly the run whose note matters.
            diagnostics[0:0] = _sentences(note)
            self.primary_metric_for_run = primary
        else:
            self.primary_metric_for_run = None
        self.last_diagnostics = diagnostics[:32]
        self.last_sweep = self._rung_curve(metrics)
        return metrics

    def _withheld(
        self, outputs: Sequence[Dict[str, Any]], cases: Sequence[Any]
    ) -> Optional[Tuple[str, Tuple[str, ...], str]]:
        """Which numbers this run may not report, and what leads instead.

        Returns ``(diagnostic, metric prefixes to drop, primary metric)`` or
        None when every surface was bound. The image side missing withholds
        retrieval, search, and the overall (the decided policy); the search
        side missing with the image side bound withholds search and the
        overall and leads with retrieval. Read off the binding's own record
        of what did not bind, with the driver outputs as a second witness:
        Bagel's first end-to-end run scored search 0.0 into an overall of
        0.4183 with its prepare step bound to the wrong argument order,
        which is a number nobody measured.
        """

        note = self._unmeasured_image_side(outputs, cases)
        if note is not None:
            # The median rank comes from the same retrieval cases that never
            # ran; an independent review found it published at 100.0.
            return (
                note,
                ("retrieval_mrr", "retrieval_recall_at", "retrieval_median_rank", "search_mrr"),
                "text_mrr",
            )
        absent = getattr(self, "_discovery_missing", {}) or {}
        gone = [name for name in ("prepare", "search") if name in absent]
        if not gone:
            return None
        for case, output in zip(cases, outputs):
            if getattr(case, "kind", None) == "search" and output.get("ok"):
                return None
        return (
            "overall withheld: no function in this repository answered a query "
            "with image ids, so the search side is not measured; caption and "
            "retrieval scores are. The search found {}: {}".format(
                gone[0], absent[gone[0]]
            ),
            ("search_mrr",),
            "retrieval_mrr",
        )

    def _unmeasured_image_side(
        self, outputs: Sequence[Dict[str, Any]], cases: Sequence[Any]
    ) -> Optional[str]:
        """The diagnostic to lead with when the image side was never bound."""

        from .discovered import NotBound

        image_kinds = {"retrieval", "search"}
        absent = getattr(self, "_discovery_missing", {}) or {}
        for case, output in zip(cases, outputs):
            if getattr(case, "kind", None) not in image_kinds:
                continue
            if output.get("ok"):
                return None
            # The binding is the record of what was not bound; the error
            # string is only a second witness. A search case can fail in a
            # prepare step that did bind, with an error that carries no
            # mark, and reading the strings alone then averaged a zero into
            # an overall for a repository whose image side was never built.
            if "image" not in absent and NotBound.MARK not in str(output.get("error", "")):
                return None
        note = getattr(self, "_weights_note", None)
        if note:
            return note
        root = getattr(self, "_discovery_root", None)
        if root is None:
            return "overall withheld: the image side has no trained weights to measure."
        extras = getattr(self, "_discovery_extras", {}) or {}
        absent = getattr(self, "_discovery_missing", {}) or {}
        if "W" in extras and "image" in absent:
            # The weights were found and read; their image step is what did
            # not bind. Saying "no trained weights" here would send the team
            # to commit a file that is already committed.
            return (
                "overall withheld: your trained weights were read from {}, but no "
                "function in this repository turned descriptors into vectors with "
                "them: {}".format(extras.get("weights_path", "the repository"), absent["image"])
            )
        from .roles import weights_diagnostic

        return weights_diagnostic(Path(root))

    @staticmethod
    def _rung_curve(metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """The four rungs as a curve, from the literal query to the furthest.

        These four numbers were already computed and already published as
        separate metrics, so the run page showed them as four rows of a table
        and the shape between them reached nobody. The shape is the part worth
        seeing. A submission that matched text rather than meaning scores near
        the top on the verbatim rung and near the floor one rung later, which
        is a cliff; a submission that learned an embedding space steps down
        gently, because the course predicts IDF weighting makes dropping
        stopwords nearly free.

        Measured, on the test tier: a lookup table built from the annotations
        file the submission is handed scores 0.9520 verbatim and 0.0279 on
        keywords. The reference submission scores 0.6337 and 0.5580. Same
        table, unmistakably different curve.

        Drawn for every run rather than only when the drop is large. A rule
        that fires on a threshold is a judgment about which submissions
        deserve scrutiny, and this platform does not make those. The curve is
        the reading; what it means is the reader's.
        """

        points: List[Dict[str, Any]] = []
        for index, rung in enumerate(perturb.RUNGS):
            value = metrics.get("search_mrr_{}".format(rung))
            if value is None:
                continue
            points.append({"rung_index": index, "mrr": float(value), "rung": rung})
        return points

    def discovery(self) -> Any:
        """What to look for in a repository that never packaged itself.

        None of the four 2026 Week 3 repositories registered an entry point,
        so asking for one asks for a step no team took. Instead the benchmark
        says what its task is, in four surfaces the driver calls, and
        `cogbench.resolve` searches their repository against that by running
        their functions. Built lazily: it loads the public test tier and the
        three course artifacts, and importing a plugin should not.

        Every resource a stage may take is in `extras`; every course file a
        repository might open at a path this machine does not have is in
        `resource_files`, mapped by basename to the benchmark's copy.
        """

        from cogbench.discovery_spec import DiscoverySpec

        from .roles import search_role

        resources = build_resources(download=False, build_kv=True)
        cases = self.load_cases("test")
        text = next(case for case in cases if case.kind == "text")
        retrieval = next(case for case in cases if case.kind == "retrieval")
        search = next(case for case in cases if case.kind == "search")
        corpus = [row["caption"] for row in resources.load_captions()["annotations"]]
        extras: Dict[str, Any] = {
            "glove": resources.load_glove(),
            "corpus": corpus,
            "descriptors_dict": resources.load_descriptors(),
            # Some teams put the course loader and caption vectorizer in two
            # constructors. The paths are benchmark inputs, and the fit stages
            # still run both constructors from the repository before scoring.
            "coco_json_path": resources.captions_path,
            "resnet_features_path": resources.descriptors_path,
        }
        self._discovery_cases = cases
        self._discovery_extras = extras
        files = {
            "captions_train2014.json": resources.captions_path,
            "resnet18_features.pkl": resources.descriptors_path,
            "glove.6B.200d.txt.w2v": resources.glove_path,
        }
        if resources.glove_kv_path is not None:
            files["glove.6B.200d.kv"] = resources.glove_kv_path
        role = search_role(
            text.captions,
            corpus,
            retrieval.descriptors,
            search.image_ids,
            search.queries[0],
            search.k,
        )
        return DiscoverySpec(
            chain_role=role,
            fixture=(list(text.captions),),
            accepts=self._accepts,
            arrangements=None,
            hints=("week3", "week 3", "language", "search", "capstone"),
            extras=extras,
            resource_files=files,
            prepare=self._weights_of,
            expects="every case the bound surfaces cover running, with finite caption embeddings",
        )

    def _weights_of(self, root: Path, modules: Sequence[Any] = ()) -> Dict[str, Any]:
        """The trained projection this repository committed, for the pool.

        Read once the root is known, which is after `discovery()` and before
        the search. `W` is the (512, D) matrix their image step takes as an
        argument or their database class takes in its constructor;
        `weights_model` is one of their own model objects loaded from the
        same file (`roles.loaded_model`), for a team whose encoder is an
        object rather than a matrix (Bagel). Absent weights leave both out,
        the image branch does not bind, and `score` withholds the overall.

        Several weight files with no load call to decide between them are
        the same outcome with a different sentence: the image side is not
        measured, and the run says which files competed. Refusing the whole
        repository instead would throw away a text side that works over a
        question about the image side, which is the opposite of the decided
        policy (docs/design/discovery-v2-brief.md, "Absent weights").
        """

        from .roles import AmbiguousWeights, loaded_model, weights_in

        self._discovery_root = Path(root)
        self._weights_note = None
        # One plugin serves one search, but a process that resolves several
        # repositories in turn (the corpus tool, a test) would otherwise
        # carry the previous repository's matrix into this one's pool.
        for name in ("W", "weights_model", "weights_cited", "weights_path"):
            self._discovery_extras.pop(name, None)
        try:
            found = weights_in(Path(root))
        except AmbiguousWeights as error:
            self._weights_note = (
                "overall withheld: several files in this repository load as a "
                "(512, D) projection and no load call in your code says which one "
                "to score: {}. Load one of them by name in the script you run, or "
                "remove the others, and run again.".format(
                    ", ".join(
                        sorted(p.relative_to(root).as_posix() for p in error.candidates)
                    )
                )
            )
            return {}
        if found is None:
            return {}
        supplied: Dict[str, Any] = {
            "W": found.matrix,
            "weights_used": [found.path.relative_to(root).as_posix()],
        }
        model = loaded_model(modules, found.path)
        if model is not None:
            supplied["weights_model"] = model[1]
            self._discovery_extras["weights_model_label"] = model[0]
        self._discovery_extras.update(supplied)
        self._discovery_extras["weights_cited"] = found.cited
        self._discovery_extras["weights_path"] = found.path.relative_to(root).as_posix()
        return supplied

    def _accepts(self, chains: Any, *_: Any):
        """The week's acceptance test, closed over the discovery fixture."""

        from .roles import accepts

        # Showcase prints during every driver run; discovery runs the driver
        # many times and none of those is a run a person is watching. Set
        # for the length of this call and put back, because a local process
        # that ran discovery and then the real benchmark lost its showcase
        # for good when this was a `setdefault` at spec-build time.
        previous = os.environ.get(SHOWCASE_ENV)
        os.environ[SHOWCASE_ENV] = "0"
        try:
            return accepts(chains, self._discovery_cases, self._discovery_extras)
        finally:
            if previous is None:
                os.environ.pop(SHOWCASE_ENV, None)
            else:
                os.environ[SHOWCASE_ENV] = previous

    def submission_from_discovery(self, submission: Any) -> Any:
        """Turn a resolved repository into the object ``run`` expects.

        The weights the image branch bound with, if any, travel on the
        submission's extras as `W`; the adapter carries them so a scored run
        projects with the same matrix the search proved.
        """

        from .discovered import build

        found = getattr(submission, "discovery", None)
        root = getattr(getattr(found, "root", None), "path", None)
        if root is not None:
            self._discovery_root = Path(root)
        missing = getattr(submission, "missing", None) or {}
        self._discovery_missing = {
            name: str(getattr(refusal, "detail", refusal))[:200]
            for name, refusal in dict(missing).items()
        }
        return build(submission, getattr(self, "_discovery_extras", {}))

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        return tier_status(tier)

    def model_factory(self) -> Resources:
        """The value handed to submission factories (rides the model slot)."""

        return build_resources(download=True, build_kv=True)

    def model_cache_status(self) -> Dict[str, Any]:
        status = artifact_status()
        return {"ready": status.ready, "path": str(status.path), "message": status.message}
