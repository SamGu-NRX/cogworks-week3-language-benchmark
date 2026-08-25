"""The ``cogworks.benchmarks.v2`` plugin for the language-search benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

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

    def score(self, outputs: Sequence[Dict[str, Any]], cases: Sequence[Any]) -> Dict[str, float]:
        metrics, diagnostics = component_scores(outputs, cases, SEARCH_K)
        for output in outputs:
            for mapping in output.get("mappings", []):
                diagnostics.append("adapter: {}".format(mapping))
        self.last_diagnostics = diagnostics[:32]
        self.last_sweep = self._rung_curve(metrics)
        return metrics

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

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        return tier_status(tier)

    def model_factory(self) -> Resources:
        """The value handed to submission factories (rides the model slot)."""

        return build_resources(download=True, build_kv=True)

    def model_cache_status(self) -> Dict[str, Any]:
        status = artifact_status()
        return {"ready": status.ready, "path": str(status.path), "message": status.message}
