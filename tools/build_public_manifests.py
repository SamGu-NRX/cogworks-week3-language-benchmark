"""Build the pinned split manifests from the real course artifacts.

Reads the cached captions and descriptors, partitions eligible images
(those with a descriptor and at least two captions) into disjoint blocks by
one master shuffle, and samples each tier's text pool, query set, and
distractor pool from its own block. Splits are a function of the seeds and
the pinned artifacts only; committing the output freezes them.

Usage (staff only)::

    python tools/build_public_manifests.py                # writes public-test/evaluation
    python tools/build_public_manifests.py --official OUT.json --seed N

Official manifests are written elsewhere and never committed to this repo.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from language_search_benchmark.datasets import ARTIFACTS, build_resources  # noqa: E402

MASTER_SEED = 20260301
BLOCKS = {"test": 0, "evaluation": 1, "official": 2}
SIZES = {
    # text_images: caption-pool source images (all their captions are used);
    # queries: one caption per distinct pool image; pool: distractor images.
    "test": {"text_images": 15, "queries": 20, "pool": 100},
    "evaluation": {"text_images": 100, "queries": 150, "pool": 700},
    "official": {"text_images": 100, "queries": 150, "pool": 700},
}
BLOCK_SIZE = 2000
TIE_SEEDS = {"test": 7, "evaluation": 11, "official": 13}


def eligible_images(resources):
    captions_blob = resources.load_captions()
    descriptors = resources.load_descriptors()
    captions_of = defaultdict(list)
    for annotation in captions_blob["annotations"]:
        captions_of[int(annotation["image_id"])].append(int(annotation["id"]))
    images = [
        image_id
        for image_id, caption_ids in captions_of.items()
        if image_id in descriptors and len(caption_ids) >= 2
    ]
    images.sort()
    return images, captions_of


def build_manifest(tier, block, captions_of, seed, manifest_id):
    rng = random.Random(seed)
    sizes = SIZES[tier]
    picked = rng.sample(block, sizes["text_images"] + sizes["pool"])
    text_images = picked[: sizes["text_images"]]
    pool_images = picked[sizes["text_images"] :]
    text_caption_ids = []
    for image_id in text_images:
        text_caption_ids.extend(sorted(captions_of[image_id])[:5])
    query_images = rng.sample(pool_images, sizes["queries"])
    query_caption_ids = [rng.choice(sorted(captions_of[image_id])) for image_id in query_images]
    return {
        "schema_version": 1,
        "manifest_id": manifest_id,
        "tier": tier,
        "split_seed": seed,
        "tie_break_seed": TIE_SEEDS[tier],
        "artifacts": {
            name: {"filename": spec["filename"], "size": spec["size"], "sha256": spec["sha256"]}
            for name, spec in sorted(ARTIFACTS.items())
        },
        "text": {"caption_ids": text_caption_ids},
        "queries": {
            "query_caption_ids": query_caption_ids,
            "pool_image_ids": pool_images,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", metavar="OUT", help="write an official manifest to OUT")
    parser.add_argument("--seed", type=int, default=None, help="private seed for --official")
    args = parser.parse_args()

    resources = build_resources(download=True, build_kv=False)
    images, captions_of = eligible_images(resources)
    master = random.Random(MASTER_SEED)
    shuffled = list(images)
    master.shuffle(shuffled)
    blocks = {
        tier: shuffled[index * BLOCK_SIZE : (index + 1) * BLOCK_SIZE]
        for tier, index in BLOCKS.items()
    }

    if args.official:
        if args.seed is None:
            raise SystemExit("--official requires --seed (kept private).")
        manifest = build_manifest(
            "official",
            blocks["official"],
            captions_of,
            args.seed,
            "language-search-official-v1",
        )
        Path(args.official).write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print("wrote", args.official)
        return

    out_dir = REPO / "language_search_benchmark" / "manifests"
    for tier in ("test", "evaluation"):
        manifest = build_manifest(
            tier,
            blocks[tier],
            captions_of,
            MASTER_SEED + BLOCKS[tier] + 1,
            "language-search-public-{}-v1".format(tier),
        )
        path = out_dir / "public-{}.json".format(tier)
        path.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
        print("wrote", path, "eligible_images", len(images))


if __name__ == "__main__":
    main()
