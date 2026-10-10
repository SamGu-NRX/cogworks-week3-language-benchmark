"""Paired bootstrap analysis over the saved run results.

Reads ``study/results/*.json``, computes arm-vs-arm deltas on the same tier
with the same queries, and attaches a 95% percentile-interval from a paired
bootstrap (the same resampled unit indices for both arms of a pair, so the
interval is about the difference, not two independent runs). Units are
per-query reciprocal ranks for retrieval and search rungs and per-image
group reciprocal means for the text component.

The locked pair list lives in ``PAIRS``; comparisons outside it are
post-hoc. Output: ``study/results/analysis.json`` and a printed table.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SEED_ANALYSIS = 20261009
DRAWS = 10000
RESULTS_DIR = Path(__file__).resolve().parent / "results"

#: Locked comparisons, as (arm_a, arm_b): the reference against every lookup,
#: every mistake, and the pairing-destroyed control, plus the two lookup
#: ladder steps. Anything else is post-hoc and must be labeled so.
PAIRS: List[Tuple[str, str]] = [
    ("reference", "lookup-derived"),
    ("reference", "lookup-oracle"),
    ("reference", "lookup-exact"),
    ("reference", "mistake-no-idf"),
    ("reference", "mistake-raw-count"),
    ("reference", "mistake-perword-norm"),
    ("reference", "mistake-raw-descriptor"),
    ("reference", "mistake-oov-crash"),
    ("reference", "mistake-memsearch"),
    ("reference", "control-shuffled"),
    ("lookup-oracle", "lookup-derived"),
    ("lookup-derived", "lookup-exact"),
]

#: Condition -> evidence key -> unit values.
CONDITIONS = [
    ("text", "text"),
    ("retrieval-verbatim", "retrieval-verbatim"),
    ("retrieval-keywords", "retrieval-keywords"),
    ("retrieval-truncated", "retrieval-truncated"),
    ("retrieval-typo", "retrieval-typo"),
    ("search-verbatim", "search-verbatim"),
    ("search-keywords", "search-keywords"),
    ("search-truncated", "search-truncated"),
    ("search-typo", "search-typo"),
]


def _units(result: Dict[str, Any], condition: str) -> Optional[np.ndarray]:
    evidence = result.get("evidence") or {}
    key = dict(CONDITIONS)[condition]
    block = evidence.get(key)
    if not block:
        return None
    if condition == "text":
        values = block.get("image_group_reciprocal_means") or []
    else:
        values = block.get("reciprocal") or []
    if not values:
        return None
    return np.asarray(values, dtype=np.float64)


def _pair_seed(arm_a: str, arm_b: str, tier: str, condition: str) -> int:
    digest = hashlib.sha256(
        "{}|{}|{}|{}|{}".format(SEED_ANALYSIS, arm_a, arm_b, tier, condition).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:4], "little")


def paired_bootstrap(
    units_a: np.ndarray, units_b: np.ndarray, seed: int
) -> Dict[str, Any]:
    if units_a.shape != units_b.shape:
        return {"error": "unit count mismatch: {} vs {}".format(
            units_a.shape[0], units_b.shape[0]
        )}
    delta = float(units_a.mean() - units_b.mean())
    rs = np.random.RandomState(seed)
    n = units_a.shape[0]
    indices = rs.randint(0, n, size=(DRAWS, n))
    draws = units_a[indices].mean(axis=1) - units_b[indices].mean(axis=1)
    low, high = np.percentile(draws, [2.5, 97.5])
    return {
        "n_units": int(n),
        "delta_mean": round(delta, 6),
        "ci95_low": round(float(low), 6),
        "ci95_high": round(float(high), 6),
        "excludes_zero": bool(low > 0 or high < 0),
        "draws": DRAWS,
    }


def analyze(results_dir: Path = RESULTS_DIR) -> Dict[str, Any]:
    results: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for path in sorted(results_dir.glob("*__*.json")):
        if path.name == "analysis.json":
            continue
        record = json.loads(path.read_text())
        results[(record["tier"], record["arm"])] = record

    comparisons: List[Dict[str, Any]] = []
    for tier in ("test", "evaluation"):
        for arm_a, arm_b in PAIRS:
            ra, rb = results.get((tier, arm_a)), results.get((tier, arm_b))
            if ra is None or rb is None:
                comparisons.append(
                    {
                        "tier": tier,
                        "pair": [arm_a, arm_b],
                        "status": "missing_result",
                    }
                )
                continue
            for status in ("arm_unavailable", "arm_failed", "scoring_failed"):
                if ra.get("status") == status:
                    comparisons.append(
                        {
                            "tier": tier,
                            "pair": [arm_a, arm_b],
                            "status": status,
                            "error": ra.get("error"),
                            "failed_arm": arm_a,
                        }
                    )
                    break
                if rb.get("status") == status:
                    comparisons.append(
                        {
                            "tier": tier,
                            "pair": [arm_a, arm_b],
                            "status": status,
                            "error": rb.get("error"),
                            "failed_arm": arm_b,
                        }
                    )
                    break
            else:
                for condition, _ in CONDITIONS:
                    units_a = _units(ra, condition)
                    units_b = _units(rb, condition)
                    entry: Dict[str, Any] = {
                        "tier": tier,
                        "pair": [arm_a, arm_b],
                        "condition": condition,
                    }
                    if units_a is None or units_b is None:
                        entry.update(
                            {
                                "status": "no_units",
                                "note": (
                                    "a failed component scores zero but yields no "
                                    "per-unit evidence; the comparison is not "
                                    "computable and is preserved as missing."
                                ),
                            }
                        )
                    else:
                        entry.update(
                            paired_bootstrap(
                                units_a, units_b, _pair_seed(arm_a, arm_b, tier, condition)
                            )
                        )
                    comparisons.append(entry)

    # Floors table: every arm's metrics beside the floors the scorer publishes.
    floor_rows = []
    for tier in ("test", "evaluation"):
        for (record_tier, arm), record in sorted(results.items()):
            if record_tier != tier:
                continue
            metrics = record.get("metrics") or {}
            floor_rows.append(
                {
                    "tier": tier,
                    "arm": arm,
                    "status": record.get("status"),
                    "overall": metrics.get("overall"),
                    "text_mrr": metrics.get("text_mrr"),
                    "retrieval_mrr": metrics.get("retrieval_mrr"),
                    "search_mrr": metrics.get("search_mrr"),
                    "text_chance": metrics.get("text_chance"),
                    "chance_mrr": metrics.get("chance_mrr"),
                    "search_chance": metrics.get("search_chance"),
                    "search_mrr_verbatim": metrics.get("search_mrr_verbatim"),
                    "retrieval_mrr_verbatim": metrics.get("retrieval_mrr_verbatim"),
                }
            )

    payload = {
        "seed": SEED_ANALYSIS,
        "draws": DRAWS,
        "pairs_locked": [list(p) for p in PAIRS],
        "comparisons": comparisons,
        "floors": floor_rows,
        "notes": [
            "Delta is arm_a minus arm_b on identical queries; positive means "
            "the first arm scores higher.",
            "A failed arm yields no per-unit evidence: comparisons that need "
            "it are preserved as no_units, never treated as zeros.",
            "The bootstrap resamples units; it speaks to sampling noise over "
            "the pinned queries, not to new queries.",
        ],
    }
    (results_dir / "analysis.json").write_text(json.dumps(payload, indent=1))
    return payload


def print_summary(payload: Dict[str, Any]) -> None:
    for tier in ("test", "evaluation"):
        print("== {} ==".format(tier))
        for row in payload["floors"]:
            if row["tier"] != tier:
                continue
            print(
                "  {:22s} {:16s} overall={} text={} retr={} search={}".format(
                    row["arm"],
                    str(row["status"]),
                    _fmt(row["overall"]),
                    _fmt(row["text_mrr"]),
                    _fmt(row["retrieval_mrr"]),
                    _fmt(row["search_mrr"]),
                )
            )
        print("  -- paired deltas (a-b), 95% CI --")
        for comp in payload["comparisons"]:
            if comp.get("tier") != tier:
                continue
            if "delta_mean" not in comp:
                print(
                    "  {:22s} vs {:22s} {:18s} {}".format(
                        comp["pair"][0],
                        comp["pair"][1],
                        comp.get("status", "?"),
                        comp.get("condition", ""),
                    )
                )
                continue
            print(
                "  {:22s} vs {:22s} {:18s} d={:+.4f} [{:+.4f},{:+.4f}]{}".format(
                    comp["pair"][0],
                    comp["pair"][1],
                    comp.get("condition", ""),
                    comp["delta_mean"],
                    comp["ci95_low"],
                    comp["ci95_high"],
                    " *" if comp["excludes_zero"] else "",
                )
            )


def _fmt(value: Any) -> str:
    if value is None:
        return "  n/a "
    return "{:.4f}".format(value)


if __name__ == "__main__":
    payload = analyze()
    print_summary(payload)
