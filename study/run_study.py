"""Run every locked arm on every locked tier through the real driver and scorer.

Saves one JSON result per (tier, arm) under ``study/results/`` with the
component scores, the diagnostics the scorer recorded, and the raw evidence
(per-query gold ranks and rankings, per-caption text ranks) the analysis
step resamples. Failed and unavailable arms are preserved with their error
strings; they are results, not omissions.

Case order follows ``build_cases``: text, retrieval-verbatim, search-verbatim,
then the retrieval rungs, then the search rungs.

Usage: python -m study.run_study [--tiers test,evaluation] [--arms name,...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from language_search_benchmark.datasets import (
    SEARCH_K,
    build_resources,
    load_manifest,
    materialize_cases,
)
from language_search_benchmark.drivers import run_cases
from language_search_benchmark.metrics import (
    component_scores,
    rank_matrix,
    ranks_of_gold,
    search_ranks,
    text_first_relevant_ranks,
)
from study import arms as arm_defs

RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Arms run in the registry's order; the report prints this order.
ALL_ARMS = list(arm_defs.ARMS)
ALL_TIERS = ["test", "evaluation"]

#: The case grid layout of ``build_cases``. Index 0 is the text case, 1 the
#: retrieval verbatim probe, 2 the search verbatim probe, 3-5 the scored
#: retrieval rungs, 6-8 the scored search rungs.
TEXT_CASE = 0
RETRIEVAL_RUNGS = {"verbatim": 1, "keywords": 3, "truncated": 4, "typo": 5}
SEARCH_RUNGS = {"verbatim": 2, "keywords": 6, "truncated": 7, "typo": 8}


def _git_fingerprint() -> Dict[str, str]:
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - fingerprinting must not fail the run
        return {
            "git_head": "unavailable",
            "git_dirty": "unavailable",
            "git_error": str(error),
        }
    return {"git_head": head, "git_dirty": dirty or "clean"}


def _env_fingerprint() -> Dict[str, Any]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "machine": platform.machine(),
    }


def _manifest_digests() -> Dict[str, str]:
    from importlib import resources as importlib_resources

    digests: Dict[str, str] = {}
    for tier in ("test", "evaluation"):
        name = "public-{}.json".format(tier)
        text = (
            importlib_resources.files("language_search_benchmark.manifests")
            .joinpath(name)
            .read_bytes()
        )
        digests[name] = hashlib.sha256(text).hexdigest()
    return digests


def _extract_text_evidence(case: Any, output: Dict[str, Any]) -> Dict[str, Any]:
    """Per-caption first-relevant ranks, grouped for the bootstrap.

    ``text_first_relevant_ranks`` skips singleton groups, so the ranks align
    with the non-singleton captions; the grouping below reconstructs which
    rows those are and means per gold image for the paired resample.
    """
    if not output.get("ok"):
        return {"failed": output.get("error", "component failed")}
    embeddings = np.asarray(output["embeddings"], dtype=np.float64)
    group_rows = case.group_rows
    if group_rows is None:
        raise ValueError("text case carries no groups; cannot extract evidence")
    counts = np.bincount(group_rows)
    ranks = text_first_relevant_ranks(
        embeddings, group_rows, case.tie_break_seed
    )
    non_singletons = [row for row, group in enumerate(group_rows) if counts[group] > 1]
    if len(non_singletons) != len(ranks):
        raise ValueError(
            "text rank alignment failed: {} ranks for {} non-singleton rows".format(
                len(ranks), len(non_singletons)
            )
        )
    groups: Dict[int, List[float]] = {}
    for row, rank in zip(non_singletons, ranks):
        groups.setdefault(int(group_rows[row]), []).append(1.0 / rank)
    reciprocal = [float(np.mean(values)) for _, values in sorted(groups.items())]
    return {
        "first_relevant_ranks": [int(r) for r in ranks],
        "image_group_reciprocal_means": reciprocal,
        "groups": len(reciprocal),
    }


def _extract_retrieval_evidence(case: Any, output: Dict[str, Any]) -> Dict[str, Any]:
    """Per-query gold ranks from the rounded outputs, the scorer's own path."""
    if not output.get("ok"):
        return {"failed": output.get("error", "component failed")}
    text = np.asarray(output["text"], dtype=np.float64)
    images = np.asarray(output["images"], dtype=np.float64)
    scores = text @ images.T
    order = rank_matrix(scores, case.tie_break_seed)
    ranks = ranks_of_gold(order, case.gold_rows)
    return {
        "gold_ranks": [int(r) for r in ranks],
        "reciprocal": [1.0 / r if r > 0 else 0.0 for r in ranks],
        "precision_at_5": [1.0 if 0 < r <= 5 else 0.0 for r in ranks],
    }


def _extract_search_evidence(case: Any, output: Dict[str, Any]) -> Dict[str, Any]:
    if not output.get("ok"):
        return {"failed": output.get("error", "component failed")}
    rankings = output["rankings"]
    if case.gold_image_ids is None:
        raise ValueError("search case carries no gold; cannot extract evidence")
    ranks, foreign = search_ranks(
        rankings, case.gold_image_ids, SEARCH_K, case.image_ids
    )
    return {
        "gold_ranks": [int(r) for r in ranks],
        "reciprocal": [1.0 / r if r > 0 else 0.0 for r in ranks],
        "precision_at_5": [1.0 if 0 < r <= 5 else 0.0 for r in ranks],
        "foreign_rows": int(foreign),
        "rankings": [[int(v) for v in ranking[:SEARCH_K]] for ranking in rankings],
    }


def _rounding_flip_probe(arm: Any, case: Any) -> Dict[str, Any]:
    """Gold ranks from raw embeddings versus the driver's rounded ones.

    Locked for the reference arm on both tiers. Reruns the arm's embed calls
    on the same queries, then computes per-query gold ranks three ways: fully
    raw, raw-then-rounded (what the driver scores), and rounded-then-scored
    from the matrices the run already stored. Any flip between the first and
    last is a rank change caused by five-decimal rounding alone.
    """

    def ranks_from(text_rows: np.ndarray, image_rows: np.ndarray) -> List[int]:
        scores = arm_defs.l2_rows(text_rows) @ arm_defs.l2_rows(image_rows).T
        order = rank_matrix(scores, case.tie_break_seed)
        return [int(r) for r in ranks_of_gold(order, case.gold_rows)]

    raw_text = np.asarray(arm.embed_text(case.queries), dtype=np.float64)
    raw_images = np.asarray(arm.embed_images(case.descriptors), dtype=np.float64)
    raw_ranks = ranks_from(raw_text, raw_images)

    driven_text = np.round(arm_defs.l2_rows(raw_text), 5)
    driven_images = np.round(arm_defs.l2_rows(raw_images), 5)
    driven_ranks = ranks_from(driven_text, driven_images)

    flips = [
        {"query": i, "raw": raw_ranks[i], "rounded": driven_ranks[i]}
        for i in range(len(raw_ranks))
        if raw_ranks[i] != driven_ranks[i]
    ]
    return {
        "queries": len(raw_ranks),
        "flips": len(flips),
        "example_flips": flips[:25],
        "mean_reciprocal_raw": float(
            np.mean([1.0 / r if r > 0 else 0.0 for r in raw_ranks])
        ),
        "mean_reciprocal_rounded": float(
            np.mean([1.0 / r if r > 0 else 0.0 for r in driven_ranks])
        ),
    }


def run_tier_arm(
    tier: str,
    arm_name: str,
    resources: Any,
    out_dir: Path,
    rounding_probe: bool,
) -> Path:
    manifest = load_manifest(tier)
    cases = materialize_cases(manifest, resources)

    result: Dict[str, Any] = {
        "tier": tier,
        "arm": arm_name,
        "search_k": SEARCH_K,
        "manifest_digests": _manifest_digests(),
        "env": _env_fingerprint(),
        "git": _git_fingerprint(),
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    instance, build_error = arm_defs.build(arm_name, resources)
    if build_error is not None or instance is None:
        result["status"] = "arm_unavailable"
        result["error"] = build_error
        result["metrics"] = None
        result["evidence"] = None
    else:
        try:
            outputs = run_cases(lambda _resources: instance, resources, cases)
        except BaseException as error:  # noqa: BLE001 - a failed run is a result
            result["status"] = "arm_failed"
            result["error"] = "{}: {}".format(type(error).__name__, error)
            result["metrics"] = None
            result["evidence"] = None
        else:
            try:
                metrics, diagnostics = component_scores(outputs, cases, SEARCH_K)
                result["status"] = "ok"
                result["error"] = None
                result["metrics"] = metrics
                result["diagnostics"] = diagnostics
                evidence: Dict[str, Any] = {
                    "text": _extract_text_evidence(cases[TEXT_CASE], outputs[TEXT_CASE])
                }
                for rung, index in RETRIEVAL_RUNGS.items():
                    evidence["retrieval-" + rung] = _extract_retrieval_evidence(
                        cases[index], outputs[index]
                    )
                for rung, index in SEARCH_RUNGS.items():
                    evidence["search-" + rung] = _extract_search_evidence(
                        cases[index], outputs[index]
                    )
                result["evidence"] = evidence
                if rounding_probe and arm_name == "reference":
                    result["rounding_flip_probe"] = _rounding_flip_probe(
                        instance, cases[RETRIEVAL_RUNGS["verbatim"]]
                    )
            except BaseException as error:  # noqa: BLE001
                result["status"] = "scoring_failed"
                result["error"] = "{}: {}".format(type(error).__name__, error)
                result["metrics"] = None
                result["evidence"] = None

    result["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{tier}__{arm_name}.json"
    out_path.write_text(json.dumps(result, indent=1))
    return out_path


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tiers", default=",".join(ALL_TIERS))
    parser.add_argument("--arms", default=",".join(ALL_ARMS))
    parser.add_argument("--out-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--no-probe", action="store_true", help="skip the rounding flip probe"
    )
    args = parser.parse_args(argv)

    tiers = [tier for tier in args.tiers.split(",") if tier]
    arm_names = [name for name in args.arms.split(",") if name]
    for name in arm_names:
        if name not in arm_defs.ARMS:
            print("unknown arm {!r}".format(name), file=sys.stderr)
            return 2

    resources = build_resources()
    out_dir = Path(args.out_dir)
    for tier in tiers:
        for arm_name in arm_names:
            started = time.time()
            path = run_tier_arm(
                tier, arm_name, resources, out_dir, rounding_probe=not args.no_probe
            )
            record = json.loads(path.read_text())
            line = "{:12s} {:22s} {:16s} {:7.1f}s".format(
                tier, arm_name, str(record["status"]), time.time() - started
            )
            metrics = record.get("metrics")
            if metrics:
                line += "  overall={:.4f} text={:.4f} retr={:.4f} search={:.4f}".format(
                    metrics["overall"],
                    metrics["text_mrr"],
                    metrics["retrieval_mrr"],
                    metrics["search_mrr"],
                )
                rung_bits = []
                for rung in ("keywords", "truncated", "typo"):
                    rung_bits.append(
                        "{} {:.3f}/{:.3f}".format(
                            rung[:3],
                            metrics["retrieval_mrr_" + rung],
                            metrics["search_mrr_" + rung],
                        )
                    )
                line += "  rungs " + " ".join(rung_bits)
            if record.get("error"):
                line += "  error={}".format(str(record["error"])[:120])
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
