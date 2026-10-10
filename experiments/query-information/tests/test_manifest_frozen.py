"""The manifest is the frozen input list. These tests pin its contents to
values hand-derived from the pool and to the frozen benchmark files.
"""

import hashlib
import json
import os

from grammar import POOL_SEED, build_pool
import run as runner

EXPERIMENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_manifest():
    with open(os.path.join(EXPERIMENT_DIR, "manifest.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


def test_manifest_counts_match_the_pool():
    manifest = load_manifest()
    pool = build_pool()
    counts = manifest["family_counts"]
    assert counts["candidates"] == len(pool) == counts["queries"] == 40
    # 8 scenes use the "near" relation (2 relation words x 2 first scenes
    # x 2 second-scene variants); the other 4 relations fill the rest.
    assert counts["queries_with_relation_near"] == 8
    assert counts["candidates_per_relation"] == 8


def test_manifest_rungs_match_the_runner():
    manifest = load_manifest()
    assert manifest["rungs"]["scored"] == list(runner.RUNGS_SCORED)
    assert manifest["rungs"]["reported"] == list(runner.RUNGS_REPORTED)


def test_manifest_reference_hashes_match_the_frozen_benchmark():
    manifest = load_manifest()
    reference = manifest["reference_version"]
    repo_root = os.path.dirname(os.path.dirname(EXPERIMENT_DIR))
    for filename, key in (
        ("perturb.py", "perturb_sha256"),
        ("metrics.py", "metrics_sha256"),
    ):
        with open(os.path.join(repo_root, "language_search_benchmark", filename), "rb") as handle:
            assert reference[key] == hashlib.sha256(handle.read()).hexdigest()
    assert reference["scorer_version"] == "retrieval-v4"


def test_manifest_seed_matches_the_grammar():
    manifest = load_manifest()
    assert manifest["seeds"]["pool_seed"] == POOL_SEED
    assert manifest["seeds"]["tie_break_seed"] == 42


def test_manifest_records_the_stopword_discovery():
    manifest = load_manifest()
    assert "near" in json.dumps(manifest)


def test_manifest_disclaims_what_identifiability_means():
    manifest = load_manifest()
    assert "identifiability" in manifest["limits"]
    # Provenance: fully synthetic, no course or study data.
    assert "synthetic" in manifest["data_provenance"]
