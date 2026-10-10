"""Entry point for the retrieval-v4 scorer-validity experiment.

Two modes, never both:

``python experiments/language-validity/run.py --manifest <manifest.json>``
    Runs every arm on every tier named in the frozen manifest, plus the
    bounded relevance audits, and records raw ranks and first outcomes per
    arm under ``<manifest dir>/<results_dir>/``. The manifest file itself is
    only read, never written.

``python experiments/language-validity/run.py --replay <results-dir>``
    Recomputes every summary, paired interval, and shortcut-success figure
    from saved result files, without running any arm again. Writes
    ``analysis.json`` into the results directory.

Stdlib plus the repo's own ``study`` package. No network, no live endpoints.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from study import analyze as study_analyze  # noqa: E402
from study import audit_relevance  # noqa: E402
from study.run_study import run_tier_arm  # noqa: E402


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit_counts(record: Dict[str, Any]) -> Dict[str, int]:
    """Case and group counts from one saved result record. Failed or
    unavailable arms report zero units; their status carries the reason."""
    counts: Dict[str, int] = {}
    evidence = record.get("evidence")
    if not evidence:
        return counts
    text = evidence.get("text") or {}
    counts["text_groups"] = int(text.get("groups", 0))
    counts["text_captions"] = len(text.get("first_relevant_ranks", []))
    for component, blob in sorted(evidence.items()):
        if component == "text" or not isinstance(blob, dict):
            continue
        counts[component + "_queries"] = len(blob.get("gold_ranks", []))
    return counts


def run_manifest(manifest_path: Path) -> int:
    manifest = json.loads(manifest_path.read_text())
    manifest_sha = _sha256_file(manifest_path)
    results_dir = manifest_path.parent / manifest["results_dir"]

    print("manifest {} sha256={}".format(manifest_path, manifest_sha[:16]), flush=True)
    print("arms: {} tiers: {}".format(len(manifest["arms"]), manifest["tiers"]), flush=True)

    runlog: Dict[str, Any] = {
        "study_id": manifest["study_id"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "arm_status": {},
        "unit_counts": {},
        "unavailable_arms": [],
    }

    # Run every tier x arm through the real driver. run_tier_arm saves one
    # JSON per pair with metrics, diagnostics, and per-query evidence (raw
    # ranks, first outcomes, gold), including failed and unavailable arms.
    from language_search_benchmark.datasets import build_resources

    resources = build_resources()
    for tier in manifest["tiers"]:
        for arm in manifest["arms"]:
            path = run_tier_arm(tier, arm, resources, results_dir, rounding_probe=True)
            record = json.loads(path.read_text())
            runlog["arm_status"][tier + "/" + arm] = record["status"]
            if record["status"] != "ok":
                runlog["unavailable_arms"].append(
                    {"tier": tier, "arm": arm, "status": record["status"], "error": record.get("error")}
                )
            runlog["unit_counts"][tier + "/" + arm] = _unit_counts(record)
            print("recorded {}".format(path.name), flush=True)

    # Bounded relevance audits: annotation-established natural co-captions,
    # Jaccard near-duplicate bands, and judging shortlists. All bound sizes
    # come from the manifest; interpretive labels stay separate from
    # established facts.
    audit_dir = results_dir / "audit"
    audit_relevance.run_natural_condition(resources, audit_dir)
    audit_relevance.interpretive_bands(resources, audit_dir)
    audit_relevance.judge_shortlists(resources, audit_dir)
    runlog["audit_dir"] = str(audit_dir)

    # Observed shortcut success: the study's headline question, answered per
    # tier from the saved metrics (recomputed here from files, not carried
    # in memory from the run loop, so replay and run paths agree).
    runlog["shortcut_success"] = _shortcut_success(results_dir, manifest)

    runlog["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (results_dir / "runlog.json").write_text(json.dumps(runlog, indent=1))
    print("runlog written: {}".format(results_dir / "runlog.json"), flush=True)

    total_cases = sum(
        count for counts in runlog["unit_counts"].values() for count in counts.values()
    )
    print("total recorded units: {} | unavailable arms: {}".format(total_cases, len(runlog["unavailable_arms"])), flush=True)
    return 0


def _shortcut_success(results_dir: Path, manifest: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per tier, how each lookup arm compares with the reference on overall
    and per-component MRR, from saved metrics only."""
    rows: List[Dict[str, Any]] = []
    for tier in manifest["tiers"]:
        reference = _metrics(results_dir, tier, "reference")
        if reference is None:
            continue
        for arm in manifest["arms"]:
            if not arm.startswith("lookup-") or arm == "reference":
                continue
            metrics = _metrics(results_dir, tier, arm)
            if metrics is None:
                continue
            rows.append(
                {
                    "tier": tier,
                    "arm": arm,
                    "overall_delta": round(metrics["overall"] - reference["overall"], 4),
                    "beats_reference_overall": bool(metrics["overall"] > reference["overall"]),
                    "overall": round(metrics["overall"], 4),
                    "reference_overall": round(reference["overall"], 4),
                    "text_mrr": round(metrics["text_mrr"], 4),
                    "retrieval_mrr": round(metrics["retrieval_mrr"], 4),
                    "search_mrr": round(metrics["search_mrr"], 4),
                }
            )
    return rows


def _metrics(results_dir: Path, tier: str, arm: str) -> Any:
    path = results_dir / "{}__{}.json".format(tier, arm)
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    return record.get("metrics")


def replay(results_dir: Path) -> int:
    if not results_dir.exists() or not list(results_dir.glob("*__*.json")):
        print("no saved results in {}; run --manifest first".format(results_dir), file=sys.stderr)
        return 2

    analysis = study_analyze.analyze(results_dir)
    study_analyze.print_summary(analysis)

    counts: Dict[str, Dict[str, int]] = {}
    statuses: Dict[str, str] = {}
    for path in sorted(results_dir.glob("*__*.json")):
        if path.name == "analysis.json":
            continue
        record = json.loads(path.read_text())
        key = "{}__{}".format(record["tier"], record["arm"])
        statuses[key] = str(record["status"])
        unit_counts = _unit_counts(record)
        if unit_counts:
            counts[key] = unit_counts

    manifest_path = results_dir.parent / "manifest.json"
    payload: Dict[str, Any] = {
        "replayed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results_dir": str(results_dir),
        "statuses": statuses,
        "unit_counts": counts,
        "unavailable_arms": sorted(k for k, v in statuses.items() if v != "ok"),
        "shortcut_success": (
            _shortcut_success(results_dir, json.loads(manifest_path.read_text()))
            if manifest_path.exists()
            else []
        ),
        "analysis": analysis,
    }
    (results_dir / "analysis.json").write_text(json.dumps(payload, indent=1, default=str))
    print("replay written: {}".format(results_dir / "analysis.json"), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--manifest", type=Path, help="run every tier x arm in the manifest")
    group.add_argument("--replay", type=Path, help="recompute summaries from a results directory")
    args = parser.parse_args()
    if args.manifest is not None:
        return run_manifest(args.manifest)
    return replay(args.replay)


if __name__ == "__main__":
    raise SystemExit(main())
