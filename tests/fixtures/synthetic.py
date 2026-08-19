"""A tiny synthetic universe for offline tests (no downloads, no GloVe).

Each of ``n_images`` images has an 8-d latent; its 512-d "descriptor" is the
latent in the first 8 columns plus seeded noise. Captions are strings that
name their image index in words, so a "perfect" adapter can invert them
exactly. This gives the full spectrum of submissions:

- ``PerfectAdapter`` — recovers latents from both sides; near-perfect scores.
- ``CourseStyleAdapter`` — same behavior behind the documented course names
  (exercises auto-adaptation rung 2).
- ``InertAdapter`` — constant embeddings; must land at chance.
- ``BrokenAdapter`` — raises from every method.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from language_search_benchmark.datasets import RetrievalCase, SearchCase, TextCase

LATENT = 8
DESC = 512
WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def _index_words(index: int) -> str:
    return " ".join(WORDS[int(digit)] for digit in str(index))


def _parse_index(caption: str) -> int:
    lookup = {word: str(value) for value, word in enumerate(WORDS)}
    digits = [lookup[token] for token in caption.split() if token in lookup]
    return int("".join(digits)) if digits else 0


class Universe:
    def __init__(self, n_images: int = 40, captions_per_image: int = 3, seed: int = 5) -> None:
        rng = np.random.RandomState(seed)
        self.n_images = n_images
        self.latents = rng.normal(size=(n_images, LATENT))
        self.latents /= np.linalg.norm(self.latents, axis=1, keepdims=True)
        self.descriptors = np.zeros((n_images, DESC), dtype=np.float32)
        self.descriptors[:, :LATENT] = self.latents + rng.normal(scale=0.01, size=(n_images, LATENT))
        self.image_ids = [1000 + index for index in range(n_images)]
        self.captions_of = {
            index: [
                "picture number {} take {}".format(_index_words(index), copy)
                for copy in range(captions_per_image)
            ]
            for index in range(n_images)
        }

    def cases(self, n_text_images: int = 12, n_queries: int = 10, pool: int = 30, seed: int = 7):
        text_captions: List[str] = []
        group_rows: List[int] = []
        for group, index in enumerate(range(n_text_images)):
            for caption in self.captions_of[index]:
                text_captions.append(caption)
                group_rows.append(group)
        pool_indices = list(range(self.n_images - pool, self.n_images))
        pool_image_ids = [self.image_ids[index] for index in pool_indices]
        descriptors = self.descriptors[pool_indices]
        query_indices = pool_indices[:n_queries]
        queries = [self.captions_of[index][0] for index in query_indices]
        gold_rows = [pool_indices.index(index) for index in query_indices]
        gold_image_ids = [self.image_ids[index] for index in query_indices]
        return [
            TextCase(kind="text", captions=text_captions, group_rows=group_rows, tie_break_seed=seed),
            RetrievalCase(
                kind="retrieval",
                queries=queries,
                descriptors=descriptors,
                gold_rows=gold_rows,
                tie_break_seed=seed,
            ),
            SearchCase(
                kind="search",
                queries=queries,
                image_ids=pool_image_ids,
                descriptors=descriptors.copy(),
                gold_image_ids=gold_image_ids,
                k=10,
                tie_break_seed=seed,
            ),
        ]


class PerfectAdapter:
    def __init__(self, universe: Universe) -> None:
        self.universe = universe
        self._db_ids: List[int] = []
        self._db_vectors = np.zeros((0, LATENT))

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return np.stack([self.universe.latents[_parse_index(caption)] for caption in captions])

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return np.asarray(descriptors)[:, :LATENT]

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._db_ids = list(image_ids)
        vectors = np.asarray(descriptors)[:, :LATENT]
        self._db_vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    def search(self, query: str, k: int) -> List[int]:
        vector = self.universe.latents[_parse_index(query)]
        scores = self._db_vectors @ vector
        order = np.argsort(-scores)[:k]
        return [self._db_ids[index] for index in order]


class CourseStyleAdapter:
    """The same behavior behind lecture/notebook names (rung-2 adaptation)."""

    def __init__(self, universe: Universe) -> None:
        self._inner = PerfectAdapter(universe)

    def text_embedding(self, caption: str) -> np.ndarray:
        return self._inner.embed_text([caption])[0]

    def se_image(self, descriptors: np.ndarray) -> np.ndarray:
        return self._inner.embed_images(descriptors)

    def build_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._inner.prepare_database(image_ids, descriptors)

    def query(self, caption: str, k: int) -> List[int]:
        return self._inner.search(caption, k)


class UnmappedPrepareAdapter:
    """Correct in every way except that the prepare step has a name we don't map.

    ``build_index`` is outside ``PREPARE_ALIASES``, so the database is never
    built and every search comes back empty. This is the shape that used to
    score zero on the search component with nothing to read.
    """

    def __init__(self, universe: Universe) -> None:
        self._inner = PerfectAdapter(universe)

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return self._inner.embed_text(captions)

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return self._inner.embed_images(descriptors)

    def build_index(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._inner.prepare_database(image_ids, descriptors)

    def search(self, query: str, k: int) -> List[int]:
        return self._inner.search(query, k)


class InertAdapter:
    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return np.zeros((len(captions), 64))

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return np.zeros((len(descriptors), 64))

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = list(image_ids)

    def search(self, query: str, k: int) -> List[int]:
        return self._ids[:k]


class BrokenAdapter:
    def embed_text(self, captions):
        raise ValueError("weights file mynn_model_weights_final.pkl is missing")

    def embed_images(self, descriptors):
        raise ValueError("weights file mynn_model_weights_final.pkl is missing")


def perfect_factory_for(universe: Universe):
    def factory(resources):
        return PerfectAdapter(universe)

    return factory
