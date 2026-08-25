"""A tiny synthetic universe for offline tests (no downloads, no GloVe).

Each of ``n_images`` images has an 8-d latent; its 512-d "descriptor" is the
latent in the first 8 columns plus seeded noise. Captions are strings that
name their image index in words, so a "perfect" adapter can invert them
exactly. This gives the full spectrum of submissions:

- ``PerfectAdapter``: recovers latents from both sides; near-perfect scores.
- ``CourseStyleAdapter``: same behavior behind the documented course names
  (exercises auto-adaptation rung 2).
- ``MemorizedCaptionAdapter``: correct embeddings, but a search path that only
  recognizes caption strings it has already seen.
- ``InertAdapter``: constant embeddings; must land at chance.
- ``BrokenAdapter``: raises from every method.

The captions are built so that every query rewrite in ``perturb.py`` leaves
the image index recoverable, which lets ``PerfectAdapter`` score 1.0 across
the whole grid. See the comment on ``captions_of`` for the word order that
makes that true.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np

from language_search_benchmark import perturb
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
        # The index words come first, and two stopwords follow them, so that
        # every query rewrite leaves the index recoverable:
        #
        #   verbatim   "one zero in a picture take 0"
        #   keywords   "one zero picture take 0"      (in, a dropped)
        #   truncated  "one zero picture"             (first three content words)
        #   typo       "one zero in a pucture take 0" (longest word is "picture")
        #
        # That matters because the course predicts a correct submission
        # survives these rewrites, so the fixture's "does everything right"
        # adapter has to be able to. Under the older "picture number one zero
        # take 0" wording, truncated kept "picture number one" and threw away
        # the second digit, which made PerfectAdapter look broken on a rung
        # that was really testing our caption format.
        self.captions_of = {
            index: [
                "{} in a picture take {}".format(_index_words(index), copy)
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
        # The whole rewrite grid, built the way `materialize_cases` builds it:
        # one shared id list and one shared descriptor copy across every rung,
        # since the driver uses object identity to decide whether the pool
        # changed. Scoring refuses an incomplete grid, so a fixture that
        # shipped only the verbatim case could not be scored at all.
        search_ids = list(pool_image_ids)
        search_descriptors = descriptors.copy()
        return [
            TextCase(kind="text", captions=text_captions, group_rows=group_rows, tie_break_seed=seed),
            RetrievalCase(
                kind="retrieval",
                queries=queries,
                descriptors=descriptors,
                gold_rows=gold_rows,
                tie_break_seed=seed,
            ),
        ] + [
            SearchCase(
                kind="search",
                queries=perturb.rewrite_all(queries, rung),
                image_ids=search_ids,
                descriptors=search_descriptors,
                gold_image_ids=gold_image_ids,
                k=10,
                tie_break_seed=seed,
                rung=rung,
            )
            for rung in perturb.RUNGS
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


class MemorizedCaptionAdapter:
    """Both embedding towers are right; search recognizes caption strings only.

    This is the submission ``perturb.py`` exists to detect. It builds a
    lookup from the exact caption text to the image it belongs to, so a query
    it has seen before is answered perfectly and anything else falls back to
    pool order. The embeddings are the real ones, so ``text_mrr`` and
    ``retrieval_mrr`` are unaffected: the failure is in the search path alone.

    Under scorer version retrieval-v2, which scored the verbatim rung only,
    this was indistinguishable from a correct submission on every published
    number. It is what makes the rewrite grid worth scoring.
    """

    def __init__(self, universe: Universe) -> None:
        self._inner = PerfectAdapter(universe)
        self._universe = universe
        self._db_ids: List[int] = []
        self._known: dict = {}

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return self._inner.embed_text(captions)

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return self._inner.embed_images(descriptors)

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._inner.prepare_database(image_ids, descriptors)
        self._db_ids = list(image_ids)
        position = {image_id: index for index, image_id in enumerate(self._universe.image_ids)}
        self._known = {
            caption: image_id
            for image_id in self._db_ids
            for caption in self._universe.captions_of[position[image_id]]
        }

    def search(self, query: str, k: int) -> List[int]:
        hit = self._known.get(query)
        if hit is None:
            return self._db_ids[:k]
        return ([hit] + [i for i in self._db_ids if i != hit])[:k]


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


class PureMemorizerAdapter:
    """A submission that embeds nothing and answers from the file it is given.

    ``MemorizedCaptionAdapter`` above memorizes the search path and keeps
    real embeddings, so it isolates one failure. This one is the harder
    case and the reason `docs/decisions/week3-verbatim-probes.md` exists: it
    has no embedding at all, and it exploits every component it can reach
    using only the caption-to-image relationship that the annotations file
    hands over.

    On the real artifact that relationship is 400,172 pairs. Every scored
    verbatim query is a caption of its gold image, so a dictionary answers
    it: measured against the real scorer, this submission scored 1.0000 on
    `retrieval_mrr` and 0.4822 overall while the reference scored 0.6925.

    It is kept, and asserted at floor, because a change to how probes are
    built could reopen that door and nothing else would notice. Assert per
    component against that component's own floor, never against one global
    chance number: the components have different denominators, and comparing
    `text_mrr` against `chance_mrr` reports a submission sitting exactly on
    its floor as 3.4 times chance.
    """

    def __init__(self, universe: "Universe") -> None:
        self._universe = universe
        self._db_ids: List[int] = []
        self._known: dict = {}
        self._position: dict = {}
        #: Row order of the shared descriptor matrix, learned from the search
        #: case's pool ids the first time they are seen.
        self._pool_order: dict = {}

    def _learn(self) -> None:
        """The dictionary the annotations file yields, built once."""

        if self._known:
            return
        position = {image_id: index for index, image_id in enumerate(self._universe.image_ids)}
        self._known = {
            caption: image_id
            for image_id, index in position.items()
            for caption in self._universe.captions_of[index]
        }

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        # One-hot on the image the caption belongs to, which is not an
        # embedding of anything: it is the lookup wearing a matrix.
        #
        # The position comes from the pool the search case names, NOT from
        # `prepare_database`. Two facts make that reachable, and both are
        # things the assignment requires. The driver embeds the retrieval
        # queries before it ever calls prepare, so an adapter that waited for
        # prepare would score zero here. And `RetrievalCase` carries no image
        # ids of its own: it shares one descriptor matrix with the search
        # case, so the search case's `image_ids` IS the retrieval row order.
        #
        # That is how the real exploit reached 1.0000 on this component
        # against the real artifacts, using only what the sandbox receives.
        self._learn()
        order = self._pool_order or {
            image_id: index for index, image_id in enumerate(self._universe.image_ids)
        }
        width = max(len(order), 1)
        matrix = np.zeros((len(captions), width))
        for row, caption in enumerate(captions):
            hit = self._known.get(caption)
            if hit is not None and hit in order:
                matrix[row, order[hit]] = 1.0
        return matrix

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        # Identity, so the cosine between a query's one-hot and an image row
        # is one exactly when the lookup named that image. Recording the pool
        # here is what makes the retrieval exploit reachable: `embed_images`
        # is handed the descriptor matrix, and the universe's descriptor for
        # an image identifies which row that image occupies. The submission
        # never needs to be told the pool order; it can recover it.
        self._learn()
        if not self._pool_order:
            lookup = {
                self._universe.descriptors[index].tobytes(): image_id
                for index, image_id in enumerate(self._universe.image_ids)
            }
            self._pool_order = {
                lookup[descriptors[row].tobytes()]: row
                for row in range(descriptors.shape[0])
                if descriptors[row].tobytes() in lookup
            }
        return np.eye(descriptors.shape[0])

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._learn()
        self._db_ids = list(image_ids)
        self._position = {image_id: index for index, image_id in enumerate(image_ids)}
        self._pool_order = dict(self._position)

    def search(self, query: str, k: int) -> List[int]:
        self._learn()
        hit = self._known.get(query)
        if hit is None or hit not in self._position:
            return self._db_ids[:k]
        return ([hit] + [i for i in self._db_ids if i != hit])[:k]
