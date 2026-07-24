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
