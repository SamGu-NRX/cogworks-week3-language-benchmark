"""Relevance audit: what the pinned single-gold scoring does not see.

Three measurements, all on the evaluation tier (the lock bounds the audit
there), all preserved in ``study/results/audit/``:

* natural-established -- the same pinned pool and text block with the query
  captions replaced by co-captions of the same images (at most two per
  image, fixed seed, scored query and text caption ids excluded). These are
  real corpus captions of images that are already the queries' gold, so a
  correct submission should rank them first. The manifest is rebuilt in
  memory with a clearly unofficial id; nothing public changes.
* interpretive bands -- for every pinned query, the best token-Jaccard
  between the query caption and any *non-gold* pool image's caption set.
  High overlap marks queries where a different image's caption is nearly
  the same string: the places a single-gold scorer must call a plausibly
  correct answer wrong. Bands at 0.8 and 0.6; the raw per-query maxima are
  saved.
* judge shortlists -- the queries a relevance judge should look at first:
  50 pinned queries with their top non-gold candidate, plus the
  natural-interpretive showcase captions (ten caption-style queries written
  for this audit, not corpus strings) with the reference arm's top-5 pool
  candidates. The judging itself is labeled interpretive: it is evidence
  about the scorer, not independent human truth, and it changes no metric.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from language_search_benchmark.datasets import (
    SEARCH_K,
    caption_maps,
    load_manifest,
    materialize_cases,
)
from language_search_benchmark.drivers import run_cases
from language_search_benchmark.metrics import (
    component_scores,
    rank_matrix,
    ranks_of_gold,
    search_ranks,
)
from study import arms as arm_defs

SEED_AUDIT = 20261009
AUDIT_DIR = Path(__file__).resolve().parent / "results" / "audit"

#: Ten caption-style queries written for the natural-interpretive audit. They
#: are deliberately not corpus strings: the question is whether the reference
#: arm's embedding space sends a plain-English request to the right kind of
#: image. Each names objects and one relation, the shape a user actually types.
SHOWCASE_CAPTIONS = [
    "a dog running through the grass with a ball",
    "two people sitting on a bench in the park",
    "a plate of food with broccoli and rice",
    "a man riding a skateboard down the street",
    "a cat sleeping on a couch next to a laptop",
    "children playing baseball on a field",
    "a woman holding an umbrella in the rain",
    "a bus driving past a tall building",
    "a group of birds standing on the beach",
    "someone taking a photo of a giraffe at the zoo",
]


def natural_manifest(manifest: Dict[str, Any], maps: Dict[str, Dict[int, Any]]) -> Dict[str, Any]:
    """The pinned evaluation manifest with co-caption queries, same pool."""
    caption_image = maps["caption_image"]
    captions_of_image: Dict[int, List[int]] = {}
    for caption_id, image_id in caption_image.items():
        captions_of_image.setdefault(image_id, []).append(caption_id)
    # Ids already carrying meaning in any pinned manifest: scored queries and
    # the text-understanding captions. Natural queries exclude all of them.
    reserved = set()
    for tier in ("test", "evaluation"):
        pinned = load_manifest(tier)
        reserved.update(int(v) for v in pinned["queries"]["query_caption_ids"])
        reserved.update(int(v) for v in pinned["text"]["caption_ids"])

    rng = np.random.RandomState(SEED_AUDIT)
    natural_ids: List[int] = []
    skipped_images = 0
    for caption_id in manifest["queries"]["query_caption_ids"]:
        image_id = caption_image[int(caption_id)]
        candidates = [
            cid
            for cid in captions_of_image.get(image_id, [])
            if cid not in reserved
        ]
        if not candidates:
            skipped_images += 1
            continue
        candidates.sort()
        take = min(2, len(candidates))
        picked = rng.choice(len(candidates), size=take, replace=False)
        natural_ids.extend(sorted(candidates[index] for index in picked))

    rebuilt = json.loads(json.dumps(manifest))
    rebuilt["queries"]["query_caption_ids"] = natural_ids
    rebuilt["manifest_id"] = manifest["manifest_id"].replace(
        "public-evaluation-v1", "natural-established-unofficial-v1"
    )
    rebuilt["natural_note"] = (
        "Unofficial audit manifest: co-captions of the pinned evaluation "
        "query images, seed {}, at most two per image; {} of {} images had "
        "no usable co-caption and contributed none.".format(
            SEED_AUDIT, skipped_images, len(manifest["queries"]["query_caption_ids"])
        )
    )
    return rebuilt


def run_natural_condition(resources: Any, out_dir: Path) -> Dict[str, Any]:
    """Run every arm on the natural-established manifest, verbatim rung only."""
    maps = caption_maps(resources.load_captions())
    pinned = load_manifest("evaluation")
    natural = natural_manifest(pinned, maps)
    cases = materialize_cases(natural, resources)
    # The rewritten rungs of the natural condition are out of scope; record
    # that the grid ran but only the verbatim reading is reported.
    rows: List[Dict[str, Any]] = []
    for arm_name in arm_defs.ARMS:
        instance, build_error = arm_defs.build(arm_name, resources)
        row: Dict[str, Any] = {"arm": arm_name}
        if build_error is not None or instance is None:
            row["status"] = "arm_unavailable"
            row["error"] = build_error
        else:
            try:
                outputs = run_cases(lambda _r, _inst=instance: _inst, resources, cases)
                metrics, diagnostics = component_scores(outputs, cases, SEARCH_K)
                retrieval_verbatim_ranks = ranks_of_gold(
                    rank_matrix(
                        np.asarray(outputs[1]["text"], dtype=np.float64)
                        @ np.asarray(outputs[1]["images"], dtype=np.float64).T,
                        cases[1].tie_break_seed,
                    ),
                    cases[1].gold_rows,
                )
                search_verbatim_ranks, foreign = search_ranks(
                    outputs[2]["rankings"],
                    cases[2].gold_image_ids,
                    SEARCH_K,
                    cases[2].image_ids,
                )
                row.update(
                    {
                        "status": "ok",
                        "text_mrr": metrics["text_mrr"],
                        "retrieval_mrr_verbatim": metrics["retrieval_mrr_verbatim"],
                        "search_mrr_verbatim": metrics["search_mrr_verbatim"],
                        "retrieval_rung_scores_omitted": True,
                        "retrieval_verbatim_ranks": [int(r) for r in retrieval_verbatim_ranks],
                        "search_verbatim_ranks": [int(r) for r in search_verbatim_ranks],
                        "search_foreign_rows": int(foreign),
                        "diagnostics": diagnostics,
                    }
                )
            except BaseException as error:  # noqa: BLE001 - failed arms are results
                row["status"] = "arm_failed"
                row["error"] = "{}: {}".format(type(error).__name__, error)
        rows.append(row)
        print("natural", arm_name, row["status"], flush=True)

    payload = {
        "condition": "natural-established",
        "tier": "evaluation",
        "seed": SEED_AUDIT,
        "manifest_note": natural["natural_note"],
        "n_queries": len(natural["queries"]["query_caption_ids"]),
        "reported_rungs": ["retrieval verbatim", "search verbatim"],
        "rows": rows,
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "natural_results.json").write_text(json.dumps(payload, indent=1))
    return payload


def interpretive_bands(resources: Any, out_dir: Path) -> Dict[str, Any]:
    """Token-Jaccard between every pinned query and every non-gold pool image."""
    maps = caption_maps(resources.load_captions())
    manifest = load_manifest("evaluation")
    caption_text = maps["caption_text"]
    caption_image = maps["caption_image"]

    pool_ids = [int(v) for v in manifest["queries"]["pool_image_ids"]]
    captions_of_image: Dict[int, List[str]] = {}
    for caption_id, image_id in caption_image.items():
        captions_of_image.setdefault(image_id, []).append(caption_text[caption_id])
    pool_tokens = [
        set(arm_defs.tokenize(" ".join(captions_of_image.get(image_id, []))))
        for image_id in pool_ids
    ]
    pool_row = {image_id: row for row, image_id in enumerate(pool_ids)}

    per_query = []
    for caption_id in manifest["queries"]["query_caption_ids"]:
        query_tokens = set(arm_defs.tokenize(caption_text[int(caption_id)]))
        gold_row = pool_row[caption_image[int(caption_id)]]
        best_row, best_jaccard = -1, -1.0
        for row, tokens in enumerate(pool_tokens):
            if row == gold_row or not tokens:
                continue
            union = len(query_tokens | tokens)
            jaccard = len(query_tokens & tokens) / union if union else 0.0
            if jaccard > best_jaccard:
                best_row, best_jaccard = row, jaccard
        per_query.append(
            {
                "query_caption_id": int(caption_id),
                "gold_image_id": int(pool_ids[gold_row]),
                "top_nongold_image_id": int(pool_ids[best_row]),
                "top_nongold_jaccard": round(best_jaccard, 4),
                "top_nongold_captions": captions_of_image.get(
                    int(pool_ids[best_row]), []
                ),
            }
        )

    band_08 = sum(1 for q in per_query if q["top_nongold_jaccard"] >= 0.8)
    band_06 = sum(1 for q in per_query if q["top_nongold_jaccard"] >= 0.6)
    # The established check the suite's shortcut tests measure, recomputed on
    # the pinned tier for the record: exact caption duplicates in-pool.
    exact_duplicates = sum(1 for q in per_query if q["top_nongold_jaccard"] >= 1.0)
    payload = {
        "condition": "interpretive-bands",
        "tier": "evaluation",
        "n_queries": len(per_query),
        "count_top_nongold_jaccard_ge_0.8": band_08,
        "count_top_nongold_jaccard_ge_0.6": band_06,
        "count_exact_text_duplicate": exact_duplicates,
        "interpretive_label": (
            "Token Jaccard is a string proxy, not a relevance judgment. "
            "Bands name candidates for judging, not counts of wrong answers."
        ),
        "per_query": per_query,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "interpretive_bands.json").write_text(json.dumps(payload, indent=1))
    return payload


def judge_shortlists(resources: Any, out_dir: Path) -> Dict[str, Any]:
    """Pinned queries + showcase captions with reference-arm candidates."""
    rng = np.random.RandomState(SEED_AUDIT)
    bands = json.loads((out_dir / "interpretive_bands.json").read_text())
    per_query = bands["per_query"]
    picks = sorted(rng.choice(len(per_query), size=min(50, len(per_query)), replace=False))
    pinned_shortlist = []
    for index in picks:
        entry = per_query[index]
        pinned_shortlist.append(
            {
                "source": "pinned-evaluation",
                "query_caption_id": entry["query_caption_id"],
                "gold_image_id": entry["gold_image_id"],
                "candidate_image_id": entry["top_nongold_image_id"],
                "candidate_captions": entry["top_nongold_captions"],
                "jaccard": entry["top_nongold_jaccard"],
            }
        )

    # Natural-interpretive: the showcase captions ranked by the reference arm.
    instance, build_error = arm_defs.build("reference", resources)
    showcase_rows: List[Dict[str, Any]] = []
    if build_error is None and instance is not None:
        manifest = load_manifest("evaluation")
        cases = materialize_cases(manifest, resources)
        maps = caption_maps(resources.load_captions())
        captions_of_image: Dict[int, List[str]] = {}
        for caption_id, image_id in maps["caption_image"].items():
            captions_of_image.setdefault(image_id, []).append(
                maps["caption_text"][caption_id]
            )
        image_embeddings = instance.embed_images(cases[1].descriptors)
        image_embeddings = arm_defs.l2_rows(np.asarray(image_embeddings, dtype=np.float64))
        query_embeddings = arm_defs.l2_rows(
            np.asarray(instance.embed_text(SHOWCASE_CAPTIONS), dtype=np.float64)
        )
        scores = query_embeddings @ image_embeddings.T
        pool_ids = [int(v) for v in manifest["queries"]["pool_image_ids"]]
        for row, caption in enumerate(SHOWCASE_CAPTIONS):
            order = np.argsort(-scores[row], kind="stable")[:5]
            showcase_rows.append(
                {
                    "source": "showcase",
                    "query_text": caption,
                    "candidate_ranking": [
                        {
                            "rank": position + 1,
                            "image_id": int(pool_ids[order[position]]),
                            "score": round(float(scores[row][order[position]]), 4),
                            "captions": captions_of_image.get(
                                int(pool_ids[order[position]]), []
                            ),
                        }
                        for position in range(len(order))
                    ],
                }
            )
    payload = {
        "condition": "judge-shortlists",
        "interpretive_label": (
            "A judging batch for a relevance reviewer. Decisions made from it "
            "are interpretive evidence about the scorer, not independent human "
            "truth, and change no published metric or manifest."
        ),
        "pinned_queries": pinned_shortlist,
        "showcase_queries": showcase_rows,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "judge_shortlist.json").write_text(json.dumps(payload, indent=1))
    return payload


def main() -> int:
    from language_search_benchmark.datasets import build_resources

    resources = build_resources()
    out_dir = AUDIT_DIR
    run_natural_condition(resources, out_dir)
    interpretive_bands(resources, out_dir)
    judge_shortlists(resources, out_dir)
    print("audit complete:", out_dir, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
