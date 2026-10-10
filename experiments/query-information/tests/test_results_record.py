"""Consistency checks over the committed results directory.

These run only when ``results/`` exists (it is produced by the run step).
Each check recomputes an aggregate by hand from a different artifact than
the one that reported it, so a stale or hand-edited results file shows up
as a disagreement. The replay witness provides the stronger check --
byte-level regeneration -- and its committed verdict is verified here too.
"""

import json
import os

import pytest

import run as runner

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results"
)


def results_path(name):
    return os.path.join(RESULTS_DIR, name)


def load(name):
    with open(results_path(name), "r", encoding="utf-8") as handle:
        return json.load(handle)


def requires_results():
    if not os.path.isdir(RESULTS_DIR):
        pytest.skip("results/ not generated yet")


def test_summary_mrr_matches_raw_ranks():
    requires_results()
    summary = load("summary.json")
    raw = load("raw_ranks.json")
    for system_name, rungs in raw["systems"].items():
        for rung, rows in rungs.items():
            recomputed = sum(1.0 / row["rank_of_gold"] for row in rows) / len(rows)
            assert summary["systems"][system_name][rung]["mrr"] == pytest.approx(recomputed)


def test_ceiling_values_match_the_harmonic_formula():
    requires_results()
    identifiability = load("identifiability.json")
    for decoder_name, rungs in identifiability["decoders"].items():
        for rung, block in rungs.items():
            sizes = [
                int(size)
                for size, count in block["size_distribution"].items()
                for _ in range(count)
            ]
            assert block["expected_mrr_ceiling"] == pytest.approx(
                runner.expected_mrr_of_sizes(sizes)
            )


def test_identifiability_counts_match_the_rewrites():
    requires_results()
    from grammar import build_pool, rewrites_for_pool
    import decoding

    pool = build_pool()
    for decoder in (decoding.LexicalDecoder(), decoding.GrammarDecoder()):
        for rung in runner.ALL_RUNGS:
            sizes = [
                len(decoding.compatible_indices(decoder.decoded_tokens(r), pool))
                for r in rewrites_for_pool(pool, rung)
            ]
            block = load("identifiability.json")["decoders"][decoder.name][rung]
            assert sum(block["size_distribution"].values()) == len(sizes)
            assert block["unique_identifiable"] == sum(1 for size in sizes if size == 1)


def test_justified_coverage_counts_are_consistent():
    requires_results()
    coverage = load("justified_coverage.json")
    raw = load("raw_ranks.json")
    for system_name, rungs in coverage.items():
        for rung, block in rungs.items():
            rows = raw["systems"][system_name][rung]
            flagged = sum(1 for row in rows if row["top1_justified"])
            assert block["justified"] == flagged
            assert block["total"] == len(rows)
            assert block["rate"] == pytest.approx(flagged / len(rows))


def test_exact_grammar_coverage_is_perfect_in_the_committed_results():
    requires_results()
    coverage = load("justified_coverage.json")
    for rung in runner.ALL_RUNGS:
        assert coverage["exact-grammar"][rung]["rate"] == 1.0


def test_committed_replay_report_agrees():
    requires_results()
    report = load("replay_report.json")
    assert report["agreement"] is True
    assert report["manifest_sha256_match"] is True
    assert all(entry["match"] for entry in report["files"].values())


def test_systems_record_explains_the_missing_adapter_baseline():
    requires_results()
    systems = load("systems.json")
    assert systems["registered"] == []
    assert "not run" in json.dumps(systems)
