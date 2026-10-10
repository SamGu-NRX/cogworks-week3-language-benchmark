"""Tests for experiments/language-validity/run.py.

Fast, offline checks: the manifest parses and is internally consistent,
replay recomputes summaries from synthetic saved results without running
any arm, and neither mode writes to the manifest.
"""

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "language_validity_run",
    Path(__file__).resolve().parents[1] / "experiments" / "language-validity" / "run.py",
)
run_module = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_module)

MANIFEST = Path(__file__).resolve().parents[1] / "experiments" / "language-validity" / "manifest.json"


def _minimal_record(tier: str, arm: str, status: str = "ok") -> dict:
    record = {
        "tier": tier,
        "arm": arm,
        "status": status,
        "search_k": 3,
        "metrics": None,
        "evidence": None,
    }
    if status == "ok":
        record["metrics"] = {
            "overall": 0.5,
            "text_mrr": 0.5,
            "retrieval_mrr": 0.5,
            "search_mrr": 0.5,
            "retrieval_mrr_keywords": 0.5,
            "retrieval_mrr_truncated": 0.5,
            "retrieval_mrr_typo": 0.5,
            "search_mrr_keywords": 0.5,
            "search_mrr_truncated": 0.5,
            "search_mrr_typo": 0.5,
        }
        record["evidence"] = {
            "text": {
                "first_relevant_ranks": [1, 2, 3],
                "image_group_reciprocal_means": [0.9, 0.7],
                "groups": 2,
            },
            "retrieval-verbatim": {
                "gold_ranks": [1, 2],
                "reciprocal": [1.0, 0.5],
                "precision_at_5": [1.0, 0.5],
            },
        }
    return record


def test_manifest_parses_and_is_consistent():
    manifest = json.loads(MANIFEST.read_text())
    assert manifest["search_k"] == 3
    assert "reference" in manifest["arms"]
    assert "lookup-derived" in manifest["arms"]
    assert len(manifest["arms"]) == len(set(manifest["arms"]))
    assert set(manifest["tiers"]) == {"test", "evaluation"}
    # Every frozen seed is present: determinism is part of the design.
    for seed in ("bootstrap", "natural_sample", "reference_training", "lookup_codebook"):
        assert seed in manifest["seeds"]
    assert manifest["results_dir"]


def test_unit_counts_from_evidence():
    record = _minimal_record("test", "reference")
    counts = run_module._unit_counts(record)
    assert counts["text_groups"] == 2
    assert counts["text_captions"] == 3
    assert counts["retrieval-verbatim_queries"] == 2
    # Failed arms carry no evidence and no counts.
    assert run_module._unit_counts(_minimal_record("test", "reference", "arm_failed")) == {}


def test_replay_writes_analysis_without_rerunning(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    for tier in ("test",):
        for arm in ("reference", "lookup-derived"):
            path = results / "{}__{}.json".format(tier, arm)
            path.write_text(json.dumps(_minimal_record(tier, arm)))
    manifest_copy = tmp_path / "manifest.json"
    manifest = json.loads(MANIFEST.read_text())
    manifest["tiers"] = ["test"]
    manifest["results_dir"] = "results"
    manifest_copy.write_text(json.dumps(manifest))
    before = manifest_copy.read_bytes()

    assert run_module.replay(results) == 0
    assert manifest_copy.read_bytes() == before, "replay must not touch the manifest"

    analysis = json.loads((results / "analysis.json").read_text())
    assert analysis["statuses"]["test__reference"] == "ok"
    assert analysis["unit_counts"]["test__reference"]["text_groups"] == 2
    assert "analysis" in analysis


def test_replay_reports_unavailable_arms(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    (results / "test__reference.json").write_text(
        json.dumps(_minimal_record("test", "reference", "arm_failed"))
    )
    assert run_module.replay(results) == 0
    analysis = json.loads((results / "analysis.json").read_text())
    assert analysis["unavailable_arms"] == ["test__reference"]
    # A failed arm still reports zero counts, not a crash.
    assert analysis["statuses"]["test__reference"] == "arm_failed"


def test_replay_rejects_empty_directory(tmp_path, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run_module.replay(empty) == 2
    assert "no saved results" in capsys.readouterr().err


def test_shortcut_success_compares_saved_metrics(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    record = _minimal_record("test", "reference")
    record["metrics"]["overall"] = 0.4
    (results / "test__reference.json").write_text(json.dumps(record))
    cheater = _minimal_record("test", "lookup-derived")
    cheater["metrics"]["overall"] = 0.9
    (results / "test__lookup-derived.json").write_text(json.dumps(cheater))
    manifest = json.loads(MANIFEST.read_text())
    manifest["tiers"] = ["test"]
    rows = run_module._shortcut_success(results, manifest)
    assert len(rows) == 1
    assert rows[0]["arm"] == "lookup-derived"
    assert rows[0]["beats_reference_overall"] is True
    assert rows[0]["overall_delta"] == pytest.approx(0.5)
