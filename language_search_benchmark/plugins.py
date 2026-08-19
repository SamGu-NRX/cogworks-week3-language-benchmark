"""The ``cogworks.benchmarks.v2`` plugin for the language-search benchmark."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import __version__
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
    scorer_version = "retrieval-v1"
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
            "through whatever database the submission built. Weak here while the "
            "two above are strong points at the plumbing -- the database, the id "
            "mapping, or the query path -- not at the embeddings."
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
            "What ranking the images at random scores. The floor every number "
            "above should be read against."
        ),
    }

    def __init__(self) -> None:
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
        return metrics

    def cache_status(self, tier: str, cache_root: Optional[Path] = None) -> CacheStatus:
        return tier_status(tier)

    def model_factory(self) -> Resources:
        """The value handed to submission factories (rides the model slot)."""

        return build_resources(download=True, build_kv=True)

    def model_cache_status(self) -> Dict[str, Any]:
        status = artifact_status()
        return {"ready": status.ready, "path": str(status.path), "message": status.message}
