"""Ten demo captions, ranked live and printed into the run log.

The captions are staff-written and deliberately absent from COCO, so a
submission cannot answer them by lookup; this is the instructor's
"enter a couple of captions at the presentation" idea in automated form.
The controller ranks the submission's embeddings against the search pool
and prints the top urls; any failure degrades to a one-line note.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

import numpy as np

from .checks import l2_normalize_rows
from .datasets import caption_maps
from .metrics import rank_matrix

TOP_K = 3


def load_showcase_captions() -> Sequence[str]:
    from importlib import resources as importlib_resources

    package = "language_search_benchmark.manifests"
    name = "showcase.json"
    try:
        text = importlib_resources.files(package).joinpath(name).read_text(encoding="utf-8")
    except AttributeError:  # Python 3.8
        with importlib_resources.open_text(package, name, encoding="utf-8") as stream:
            text = stream.read()
    return [str(item) for item in json.loads(text)["captions"]]


def print_showcase(adapter: Any, resources: Any, cases: Sequence[Any]) -> None:
    search_case = next((case for case in cases if getattr(case, "kind", "") == "search"), None)
    if search_case is None:
        return
    captions = load_showcase_captions()
    text_matrix = l2_normalize_rows(adapter.embed_text(list(captions)))
    image_matrix = l2_normalize_rows(adapter.embed_images(search_case.descriptors))
    if text_matrix.shape[1] != image_matrix.shape[1]:
        print("showcase skipped: text and image embedding widths differ.")
        return
    order = rank_matrix(text_matrix @ image_matrix.T, search_case.tie_break_seed)
    urls = caption_maps(resources.load_captions())["image_url"]
    for index, caption in enumerate(captions):
        top = [
            urls.get(search_case.image_ids[int(row)], "?")
            for row in order[index][:TOP_K]
        ]
        print('showcase {:02d}/{:02d} "{}" -> {}'.format(index + 1, len(captions), caption, " ".join(top)))


def showcase_matrix(adapter: Any, captions: Sequence[str]) -> np.ndarray:
    return l2_normalize_rows(adapter.embed_text(list(captions)))
