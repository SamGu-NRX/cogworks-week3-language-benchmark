"""What a submission provides and what the benchmark hands it.

A student repository registers one factory under the
``cogworks.submissions.v2`` entry-point group with the name
``language-search``::

    # benchmark_adapter.py (in the student repo)
    def create_search_adapter(resources):
        return MySearchAdapter(resources)

The returned object must expose:

- ``embed_text(captions)`` — sequence of caption strings to an ``(N, D)``
  array in the submission's shared embedding space.
- ``embed_images(descriptors)`` — ``(M, 512)`` float array of ResNet-18
  descriptors to an ``(M, D)`` array in the same space.

and, for the search-application component of the score:

- ``prepare_database(image_ids, descriptors)`` — called once with the pinned
  pool before any query (optional if ``search`` needs no setup).
- ``search(query, k)`` — a caption string to the top-``k`` image ids drawn
  from the ids given to ``prepare_database``.

The benchmark never assumes the embedding dimension, a normalization
convention, an architecture, or a weights-file location: the adapter loads
its own weights from its own repository. ``Resources`` only provides the
three pinned course artifacts.
"""

from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


class AdapterContractError(RuntimeError):
    """A submission object could not be mapped onto the adapter protocol.

    ``report`` carries the actionable mapping report (what was received,
    what is missing, the closest matches, and the escape hatch).
    """

    def __init__(self, message: str, report: Optional[List[str]] = None) -> None:
        super().__init__(message)
        self.report: List[str] = list(report or [])


@dataclass
class Resources:
    """Paths to the pinned course artifacts, plus convenience loaders.

    ``glove_path`` points at the course-format ``glove.6B.200d.txt.w2v``
    text file (what student code already loads); ``glove_kv_path``, when
    present, is a pre-parsed gensim ``KeyedVectors`` save that loads in
    seconds instead of minutes and is what ``load_glove()`` prefers.
    """

    captions_path: Path
    descriptors_path: Path
    glove_path: Path
    glove_kv_path: Optional[Path] = None
    _captions_cache: Optional[Dict[str, Any]] = field(default=None, repr=False, compare=False)

    def load_captions(self) -> Dict[str, Any]:
        """The raw COCO captions JSON (``images`` and ``annotations`` keys)."""

        if self._captions_cache is None:
            with open(str(self.captions_path), "r", encoding="utf-8") as stream:
                self._captions_cache = json.load(stream)
        return self._captions_cache

    def load_descriptors(self) -> Dict[int, Any]:
        """The ``image_id -> (1, 512) float array`` ResNet-18 descriptor dict."""

        with open(str(self.descriptors_path), "rb") as stream:
            return pickle.load(stream)

    def load_glove(self) -> Any:
        """The 200-d GloVe embeddings as gensim ``KeyedVectors``.

        Prefers the pre-parsed ``.kv`` cache when available. gensim is an
        optional dependency of the benchmark (the controller itself never
        embeds text); install it with ``pip install gensim`` or the course
        environment, which already has it.
        """

        try:
            from gensim.models import KeyedVectors
        except ImportError as error:  # pragma: no cover - environment-specific
            raise RuntimeError(
                "Loading GloVe requires gensim, which is not installed in this "
                "environment. Install it with `python -m pip install gensim`."
            ) from error
        if self.glove_kv_path is not None and Path(self.glove_kv_path).is_file():
            return KeyedVectors.load(str(self.glove_kv_path), mmap="r")
        return KeyedVectors.load_word2vec_format(str(self.glove_path), binary=False)
