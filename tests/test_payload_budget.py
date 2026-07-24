"""The hosted predictions payload must stay under the 8 MiB script cap.

Worst case: a submission at the maximum allowed dimension (512) across the
evaluation tier's pinned sizes (500 text captions, 150 queries, 700 pool
images, 50-deep search results).
"""

import json

import numpy as np

from language_search_benchmark.checks import MAX_DIM
from language_search_benchmark.drivers import ROUND_DECIMALS

EVAL_TEXT_CAPTIONS = 500
EVAL_QUERIES = 150
EVAL_POOL = 700
SEARCH_DEPTH = 50
CAP_BYTES = 8 * 1024 * 1024


def _rounded_block(rows, dim, seed):
    rng = np.random.RandomState(seed)
    matrix = rng.normal(size=(rows, dim))
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.round(matrix, ROUND_DECIMALS).tolist()


def test_worst_case_payload_under_cap():
    outputs = [
        {"ok": True, "kind": "text", "embeddings": _rounded_block(EVAL_TEXT_CAPTIONS, MAX_DIM, 0)},
        {
            "ok": True,
            "kind": "retrieval",
            "text": _rounded_block(EVAL_QUERIES, MAX_DIM, 1),
            "images": _rounded_block(EVAL_POOL, MAX_DIM, 2),
        },
        {
            "ok": True,
            "kind": "search",
            "rankings": [[600000 + i for i in range(SEARCH_DEPTH)] for _ in range(EVAL_QUERIES)],
        },
    ]
    encoded = json.dumps(outputs, separators=(",", ":"))
    assert len(encoded.encode("utf-8")) < CAP_BYTES, len(encoded)
