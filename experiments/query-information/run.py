"""Run the query-information study, or replay it and verify regeneration.

    python experiments/query-information/run.py --manifest experiments/query-information/manifest.json
    python experiments/query-information/run.py --replay experiments/query-information/results

The run applies the benchmark's real, unmodified perturb transforms to the
frozen grammar pool, ranks every system per rung, and writes results as
separate artifacts: raw ranks, justified-answer coverage, byte-identical
rewrite ambiguities, identifiability ceilings, and a witness of content
hashes. The replay recomputes everything from the manifest and compares
content hashes against the witness; agreement means the committed results
are regenerable from the committed inputs alone.

Offline only: no course data, no network, no model endpoints. Adapter
baselines are discovered and recorded; none were registered in the study
environment, so the small-adapter column is reported as not run rather
than filled with a stand-in.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from grammar import build_pool, queries_for_pool, rewrites_for_pool  # noqa: E402
from language_search_benchmark import perturb  # noqa: E402

import decoding  # noqa: E402

RUNGS_SCORED = ("keywords", "truncated", "typo")
RUNGS_REPORTED = ("verbatim",)
ALL_RUNGS = RUNGS_REPORTED + RUNGS_SCORED
RESULTS_DIRNAME = "results"
WITNESS_FILES = (
    "raw_ranks.json",
    "justified_coverage.json",
    "rewrite_ambiguities.json",
    "identifiability.json",
    "systems.json",
    "summary.json",
)


def _harmonic(size: int) -> float:
    return sum(1.0 / k for k in range(1, size + 1))


def expected_mrr_of_sizes(sizes: List[int]) -> float:
    """E[MRR] when the gold sits in a uniform tie among ``size`` compatible
    candidates and the tie-break is uniform: H_size / size per query."""
    if not sizes:
        return 0.0
    return float(sum(_harmonic(size) / size for size in sizes) / len(sizes))


def sha256_file(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def check_reference_versions(manifest: Dict) -> None:
    """Refuse to run against changed scorer internals.

    The study measures the transforms as they exist at the frozen commit.
    If perturb.py or metrics.py changed on disk, the manifest's reference
    hashes no longer describe the run, and silently continuing would report
    results about transforms nobody froze.
    """

    reference = manifest["reference_version"]
    package = os.path.join(_REPO_ROOT, "language_search_benchmark")
    for filename, key in (("perturb.py", "perturb_sha256"), ("metrics.py", "metrics_sha256")):
        actual = sha256_file(os.path.join(package, filename))
        if actual != reference[key]:
            raise SystemExit(
                "{} changed since the manifest was frozen ({} != {}). The study "
                "measures the frozen transforms; re-freeze the manifest or check "
                "out the reference commit.".format(filename, actual, reference[key])
            )


def run_study(manifest: Dict) -> Dict[str, str]:
    """Compute every results artifact; return {filename: json_text}."""

    pool = build_pool()
    queries = queries_for_pool(pool)
    scene_ids = [scene_id for scene_id, _ in pool]

    lexical = decoding.LexicalDecoder()
    grammar_decoder = decoding.GrammarDecoder()
    systems = {
        "exact-grammar": decoding.ExactGrammarSystem(pool, grammar_decoder),
        "lexical-idf": decoding.LexicalIdfSystem(pool),
    }

    # Identifiability: compat sizes under each decoder, per rung, with the
    # ceilings they imply. This is the study's yardstick, computed before
    # any system runs.
    identifiability: Dict[str, Dict] = {"decoders": {}, "rungs": {}}
    compat_by_rung: Dict[str, Dict[str, List[List[int]]]] = {}
    for decoder in (lexical, grammar_decoder):
        sizes_all: Dict[str, List[int]] = {}
        for rung in ALL_RUNGS:
            rewrites = rewrites_for_pool(pool, rung)
            compat_list = [
                decoding.compatible_indices(decoder.decoded_tokens(rewrite), pool)
                for rewrite in rewrites
            ]
            compat_by_rung.setdefault(rung, {})[decoder.name] = compat_list
            sizes = [len(compat) for compat in compat_list]
            sizes_all[rung] = {
                "size_distribution": {
                    str(size): sizes.count(size) for size in sorted(set(sizes))
                },
                "unique_identifiable": sum(1 for size in sizes if size == 1),
                "expected_mrr_ceiling": expected_mrr_of_sizes(sizes),
            }
        identifiability["decoders"][decoder.name] = sizes_all
    identifiability["rungs"] = {
        "typo_grammar_vs_lexical": (
            "the grammar-aware decoder restores unique identification on every "
            "typo query (all corrections are unique in-vocabulary substitutions), "
            "so the whole lexical typo gap is recoverable system-side."
        )
    }
    identifiability["typo_correction_log"] = {
        "per_corrupted_token_correction_counts": grammar_decoder.correction_log(),
        "template_stopwords_uncorrectable": ["the", "is", "of"],
    }

    # Systems: raw ranks and justification, per rung.
    raw_ranks: Dict[str, Dict] = {}
    coverage: Dict[str, Dict] = {}
    summary: Dict[str, Dict] = {"systems": {}, "ceiling_gap": {}}
    for system_name, system in systems.items():
        raw_ranks[system_name] = {}
        coverage[system_name] = {}
        summary["systems"][system_name] = {}
        for rung in ALL_RUNGS:
            rewrites = rewrites_for_pool(pool, rung)
            decoder_name = "grammar" if system_name == "exact-grammar" else "lexical"
            compat_lists = compat_by_rung[rung][decoder_name]
            rows: List[Dict] = []
            ranks: List[int] = []
            unjustified: List[int] = []
            for index, rewrite in enumerate(rewrites):
                rank_of_gold, _ = system.rank(rewrite)
                rank = int(rank_of_gold[index])
                top1 = scene_ids[int(np.nonzero(rank_of_gold == 1)[0][0])]
                compat = compat_lists[index]
                justified = top1 in {scene_ids[j] for j in compat}
                if not justified:
                    unjustified.append(index)
                ranks.append(rank)
                rows.append(
                    {
                        "query_index": index,
                        "scene_id": scene_ids[index],
                        "rank_of_gold": rank,
                        "top1": top1,
                        "compat_size": len(compat),
                        "top1_justified": justified,
                    }
                )
            raw_ranks[system_name][rung] = rows
            mrr = float(np.mean([1.0 / rank for rank in ranks]))
            coverage[system_name][rung] = {
                "decoder": decoder_name,
                "justified": len(rows) - len(unjustified),
                "total": len(rows),
                "rate": (len(rows) - len(unjustified)) / len(rows),
                "unjustified_query_indices": unjustified,
            }
            summary["systems"][system_name][rung] = {
                "mrr": mrr,
                "justified_rate": coverage[system_name][rung]["rate"],
            }

    # Rewritten-rung mean, mirroring retrieval-v4's scored aggregate.
    for system_name in systems:
        scored = [summary["systems"][system_name][rung]["mrr"] for rung in RUNGS_SCORED]
        summary["systems"][system_name]["scored_mean"] = float(np.mean(scored))
        summary["systems"][system_name]["verbatim_reported"] = summary["systems"][
            system_name
        ]["verbatim"]["mrr"]

    # Ceilings per rung per decoder, beside each system's realized MRR.
    summary["ceiling_gap"] = {
        rung: {
            "lexical_expected_mrr": identifiability["decoders"]["lexical"][rung][
                "expected_mrr_ceiling"
            ],
            "grammar_expected_mrr": identifiability["decoders"]["grammar"][rung][
                "expected_mrr_ceiling"
            ],
            "exact-grammar_gap_to_grammar_ceiling": (
                identifiability["decoders"]["grammar"][rung]["expected_mrr_ceiling"]
                - summary["systems"]["exact-grammar"][rung]["mrr"]
            ),
            "lexical-idf_gap_to_lexical_ceiling": (
                identifiability["decoders"]["lexical"][rung]["expected_mrr_ceiling"]
                - summary["systems"]["lexical-idf"][rung]["mrr"]
            ),
        }
        for rung in ALL_RUNGS
    }

    # Byte-identical rewrite ambiguities, per rung.
    ambiguities: Dict[str, Dict] = {}
    for rung in ALL_RUNGS:
        rewrites = rewrites_for_pool(pool, rung)
        same_as_verbatim = [
            index
            for index, (rewrite, query) in enumerate(zip(rewrites, queries))
            if rewrite == query
        ]
        groups: Dict[str, List[int]] = {}
        for index, rewrite in enumerate(rewrites):
            groups.setdefault(rewrite, []).append(index)
        collisions = {
            rewrite: members
            for rewrite, members in sorted(groups.items())
            if len(members) > 1
        }
        ambiguities[rung] = {
            "same_as_verbatim_query_indices": same_as_verbatim,
            "collision_groups": collisions,
            "collision_query_count": sum(len(members) for members in collisions.values()),
            "note": (
                "queries sharing a rewrite byte string are indistinguishable to "
                "any token-level system on that rung; they are reported, not "
                "dropped, and their gold candidates cannot be told apart."
            ),
        }

    # Adapter discovery, recorded whether or not anything is registered.
    adapters = decoding.discover_adapters()

    payloads = {
        "raw_ranks.json": {
            "pool_order": scene_ids,
            "queries_verbatim": queries,
            "systems": raw_ranks,
        },
        "justified_coverage.json": coverage,
        "rewrite_ambiguities.json": ambiguities,
        "identifiability.json": identifiability,
        "systems.json": adapters,
        "summary.json": summary,
    }

    outputs = {
        name: json.dumps(payload, indent=2, sort_keys=True) + "\n"
        for name, payload in payloads.items()
    }
    witness = {
        "manifest_sha256": sha256_file(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "manifest.json")
        ),
        "files": {
            name: hashlib.sha256(text.encode("utf-8")).hexdigest()
            for name, text in outputs.items()
        },
        "witness_scope": "content hashes of the six results artifacts and the manifest; the replay compares against these",
    }
    outputs["witness.json"] = json.dumps(witness, indent=2, sort_keys=True) + "\n"
    return outputs


def write_results(outputs: Dict[str, str], results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    for name, text in outputs.items():
        with open(os.path.join(results_dir, name), "w", encoding="utf-8") as handle:
            handle.write(text)


def replay(results_dir: str, manifest_path: str) -> bool:
    """Recompute and compare against the committed witness.

    Returns True on full agreement. The comparison is over content hashes
    of every artifact the witness covers; the replay report itself is never
    part of its own comparison.
    """

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    check_reference_versions(manifest)
    regenerated = run_study(manifest)
    witness_path = os.path.join(results_dir, "witness.json")
    with open(witness_path, "r", encoding="utf-8") as handle:
        witness = json.load(handle)
    report: Dict[str, Dict] = {"files": {}, "agreement": True}
    for name in WITNESS_FILES:
        expected = witness["files"][name]
        actual = hashlib.sha256(regenerated[name].encode("utf-8")).hexdigest()
        match = actual == expected
        report["files"][name] = {"match": match, "sha256": actual}
        report["agreement"] = report["agreement"] and match
    report["manifest_sha256_match"] = (
        witness["manifest_sha256"] == sha256_file(manifest_path)
    )
    report["agreement"] = report["agreement"] and report["manifest_sha256_match"]
    report_path = os.path.join(results_dir, "replay_report.json")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return bool(report["agreement"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", help="path to the frozen manifest.json (run mode)")
    parser.add_argument(
        "--replay",
        help="path to a results directory; recompute from the manifest beside it and compare",
    )
    arguments = parser.parse_args()

    if arguments.replay:
        results_dir = os.path.abspath(arguments.replay)
        manifest_path = os.path.join(os.path.dirname(results_dir), "manifest.json")
        agreement = replay(results_dir, manifest_path)
        print(
            "replay {}: {} files compared against the committed witness".format(
                "AGREES" if agreement else "DISAGREES", len(WITNESS_FILES) + 1
            )
        )
        return 0 if agreement else 1

    if not arguments.manifest:
        parser.error("one of --manifest (run) or --replay is required")

    manifest_path = os.path.abspath(arguments.manifest)
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    check_reference_versions(manifest)
    outputs = run_study(manifest)
    results_dir = os.path.join(os.path.dirname(manifest_path), RESULTS_DIRNAME)
    write_results(outputs, results_dir)
    summary = json.loads(outputs["summary.json"])
    for system_name, rungs in summary["systems"].items():
        line = " ".join(
            "{}={:.4f}".format(rung, rungs[rung]["mrr"])
            for rung in ALL_RUNGS
            if rung in rungs
        )
        print("{:14s} {}".format(system_name, line))
    print("results written to", results_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
