"""The study arms: every submission variant the locked configuration defines.

Each arm is an adapter with the same surface the benchmark drives
(``embed_text``, ``embed_images``, ``prepare_database``, ``search``), built
from the same ``Resources`` a real submission receives. The harness
(``run_study``) runs every arm through the real driver and the real scorer on
the pinned public manifests; nothing here scores anything itself.

Two families:

Reference-style arms embed text as an IDF-weighted sum of GloVe vectors and
images through a linear projection trained against that arm's own text
embedding. Each ``mistake-*`` arm deviates from the reference in exactly one
stated way and retrains the rest of its pipeline around its own text side, so
the measured difference is the deviation and not a broken companion.

Lookup arms embed nothing. They answer from the annotations file and the
descriptor pickle the contract hands over, through progressively more
determined routes: exact strings only (what the existing shortcut tests
measure), self-contained rewrite inversion (own stopword list, edit-distance
typo repair), and the oracle that keys with the benchmark's own ``perturb``.
Their embedding side is a 512-d Gaussian codebook keyed by image id -- dense
random vectors that are identical exactly when the ids match -- because
``checks.MAX_DIM`` (512) blocks a raw one-hot over the 700-image evaluation
pool.

Every arm is deterministic: fixed seeds everywhere, no randomness at call
time, so a replay reproduces every number.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np

SEED_TRAINING = 20261001
SEED_CODEBOOK = 20261001
SEED_CONTROL = 20261001

#: The checks.MAX_DIM ceiling. A one-hot over the 700-image evaluation pool
#: cannot pass the driver's own contract check; 512 can. The codebook below
#: projects into this width.
CODEBOOK_DIM = 512

_TOKEN_RE = re.compile(r"[a-z0-9']+")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens. The study's one tokenizer, used by every arm."""
    return _TOKEN_RE.findall(text.lower())


def l2_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms == 0.0, 1.0, norms)


# --------------------------------------------------------------------------
# Shared reference machinery
# --------------------------------------------------------------------------


class GloveTextEmbedder:
    """IDF-weighted GloVe sums, with switchable deviations.

    The deviations are the study's *mistake* arms; the default weights are
    the course's log IDF over every annotation caption in the corpus.
    """

    def __init__(self, resources: Any, variant: str = "idf") -> None:
        self.variant = variant
        self.kv = resources.load_glove()
        self.vocab: Dict[str, int] = dict(self.kv.key_to_index)
        self.width = int(self.kv.vector_size)
        document_frequency: Counter = Counter()
        for annotation in resources.load_captions()["annotations"]:
            document_frequency.update(set(tokenize(str(annotation["caption"]))))
        self.document_frequency = document_frequency
        self.total_documents = len(resources.load_captions()["annotations"])
        # mistake-raw-count weights a word by how common it is, the opposite
        # of IDF and the failure the rung docstrings call out by name.
        self.raw_count = document_frequency

    def weight_of(self, token: str) -> float:
        if self.variant == "idf":
            df = self.document_frequency.get(token, 0)
            if df == 0:
                return 0.0
            return float(np.log(self.total_documents / df))
        if self.variant == "raw-count":
            return float(self.raw_count.get(token, 0))
        if self.variant == "no-idf":
            return 1.0
        raise ValueError("unknown text variant {!r}".format(self.variant))

    def embed_batch(self, captions: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(captions), self.width), dtype=np.float64)
        for row, caption in enumerate(captions):
            rows[row] = self.embed_one(caption)
        return rows

    def embed_one(self, caption: str) -> np.ndarray:
        words = tokenize(caption)
        if self.variant == "perword-norm":
            vectors = np.zeros(self.width, dtype=np.float64)
            count = 0
            for word in words:
                index = self.vocab.get(word)
                if index is None:
                    continue
                vector = np.asarray(self.kv.vectors[index], dtype=np.float64)
                norm = np.linalg.norm(vector)
                if norm > 0:
                    vectors += vector / norm
                count += 1
            return vectors  # unnormalized sum of unit vectors; driver normalizes
        vectors = np.zeros(self.width, dtype=np.float64)
        for word in words:
            index = self.vocab.get(word)
            if index is None:
                continue
            vectors += self.weight_of(word) * np.asarray(
                self.kv.vectors[index], dtype=np.float64
            )
        return vectors

    def embed_batch_or_raise(self, captions: Sequence[str]) -> np.ndarray:
        """The oov-crash arm: one vocabulary-missing token fails the call."""
        rows = np.zeros((len(captions), self.width), dtype=np.float64)
        for row, caption in enumerate(captions):
            for word in tokenize(caption):
                if word not in self.vocab:
                    raise ValueError(
                        "unseen word {!r} in {!r}".format(word, caption[:60])
                    )
            rows[row] = self.embed_one(caption)
        return rows


class ProjectionTrainingError(RuntimeError):
    pass


def train_projection(
    resources: Any, text_embedder: GloveTextEmbedder, embed_batch=None
) -> np.ndarray:
    """Ridge regression from ResNet-18 descriptors to text embeddings.

    Trained over every corpus image that has both a descriptor and a
    caption, which is what the course corpus offers. The targets are the
    mean text embedding of the image's own captions. ``embed_batch`` lets an
    arm train against its own text side; the default uses the embedder's.
    ``lambda`` is picked once on a fixed seeded validation slice by mean
    cosine between the projected image vector and the target, over the
    locked grid {1e-2, 1, 100}.
    """

    if embed_batch is None:
        embed_batch = text_embedder.embed_batch
    blob = resources.load_captions()
    captions_of: Dict[int, List[str]] = defaultdict(list)
    for annotation in blob["annotations"]:
        captions_of[int(annotation["image_id"])].append(str(annotation["caption"]))
    descriptors = resources.load_descriptors()
    usable = [
        image_id
        for image_id, rows in captions_of.items()
        if image_id in descriptors and rows
    ]
    usable.sort()
    if len(usable) < 1000:
        raise ProjectionTrainingError(
            "only {} usable images; cannot train".format(len(usable))
        )
    rng = np.random.RandomState(SEED_TRAINING)
    order = rng.permutation(len(usable))
    validation_rows = order[:1000]
    training_rows = order[1000:]

    def build(rows: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros((len(rows), 512), dtype=np.float64)
        targets = np.zeros((len(rows), text_embedder.width), dtype=np.float64)
        for out_row, index in enumerate(rows):
            image_id = usable[index]
            matrix[out_row] = np.asarray(descriptors[image_id], dtype=np.float64).reshape(-1)
            targets[out_row] = embed_batch(captions_of[image_id]).mean(axis=0)
        return matrix, targets

    # Building targets one image at a time through embed_batch is slow for
    # 80k images; batch the training set in chunks instead.
    def build_chunked(rows: Sequence[int], chunk: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros((len(rows), 512), dtype=np.float64)
        targets = np.zeros((len(rows), text_embedder.width), dtype=np.float64)
        for start in range(0, len(rows), chunk):
            part = rows[start : start + chunk]
            captions: List[str] = []
            spans: List[Tuple[int, int]] = []
            cursor = 0
            for index in part:
                image_captions = captions_of[usable[index]]
                captions.extend(image_captions)
                spans.append((cursor, cursor + len(image_captions)))
                cursor += len(image_captions)
            embedded = embed_batch(captions)
            for out_row, (begin, end) in zip(
                range(start, start + len(part)), spans
            ):
                image_id = usable[rows[out_row]]
                matrix[out_row] = np.asarray(descriptors[image_id], dtype=np.float64).reshape(-1)
                targets[out_row] = embedded[begin:end].mean(axis=0)
        return matrix, targets

    validation_x, validation_y = build(list(validation_rows))
    training_x, training_y = build_chunked(list(training_rows))

    best_lambda: Optional[float] = None
    best_cosine = -2.0
    for candidate in (1e-2, 1.0, 100.0):
        weights = _ridge(training_x, training_y, candidate)
        projected = validation_x @ weights
        cosines = np.sum(
            l2_rows(projected) * l2_rows(validation_y), axis=1
        )
        mean_cosine = float(np.mean(cosines))
        if mean_cosine > best_cosine + 1e-12:
            best_cosine = mean_cosine
            best_lambda = candidate
    # Refit on all rows (training + validation) at the chosen lambda; the
    # validation slice picked it and is now training data again.
    all_x = np.concatenate([training_x, validation_x], axis=0)
    all_y = np.concatenate([training_y, validation_y], axis=0)
    return _ridge(all_x, all_y, float(best_lambda))


def _ridge(x: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    gram = x.T @ x
    width = gram.shape[0]
    solution = np.linalg.solve(gram + lambda_ * np.eye(width), x.T @ y)
    return solution


# --------------------------------------------------------------------------
# Reference-style arm
# --------------------------------------------------------------------------


class ReferenceStyleArm:
    """The working reference: IDF GloVe text, trained projection images.

    This is the study's reimplementation of what the course teaches. The
    staff reference submission is not in this repository, so the historical
    reference numbers quoted in the scorer's docstrings (0.6925 on the test
    tier before retrieval-v4, 0.6594 after) are cited, not reproduced here.
    """

    name = "reference"

    def __init__(self, resources: Any) -> None:
        self.text_embedder = GloveTextEmbedder(resources, variant="idf")
        self.weights = train_projection(resources, self.text_embedder)
        self._ids: List[int] = []
        self._database: Optional[np.ndarray] = None

    # -- benchmark surface ------------------------------------------------
    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return self.text_embedder.embed_batch(captions)

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return np.asarray(descriptors, dtype=np.float64) @ self.weights

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = [int(value) for value in image_ids]
        self._database = l2_rows(self.embed_images(descriptors))

    def search(self, query: str, k: int) -> List[int]:
        if self._database is None:
            return []
        vector = l2_rows(self.embed_one(query)[np.newaxis, :])[0]
        scores = self._database @ vector
        order = np.argsort(-scores, kind="stable")[:k]
        return [self._ids[index] for index in order]

    def embed_one(self, caption: str) -> np.ndarray:
        return self.text_embedder.embed_one(caption)


class MistakeArm(ReferenceStyleArm):
    """One deviation from the reference, named by the subclass."""

    text_variant = "idf"  # GloveTextEmbedder variant
    crash_on_oov = False
    image_side_raw = False

    def __init__(self, resources: Any) -> None:
        self.text_embedder = GloveTextEmbedder(resources, variant=self.text_variant)
        self.crash_on_oov_override = self.crash_on_oov
        if self.image_side_raw:
            # No learned projection: images are raw descriptors. The text
            # side is zero-padded into the same 512-d space, which is what a
            # submission that skipped the training step realistically ships.
            self.weights = None
        else:
            self.weights = train_projection(
                resources, self.text_embedder, embed_batch=self._batch_for_training
            )
        self._ids = []
        self._database = None

    def _batch_for_training(self, captions: Sequence[str]) -> np.ndarray:
        # Training targets always come from the tolerant path: a rare
        # out-of-vocabulary word in the training corpus is skipped there. The
        # crash the oov-crash arm models is a *serving* behavior -- it fires
        # at embed time on inputs like typo'd tokens, not during training.
        return self.text_embedder.embed_batch(captions)

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        if self.crash_on_oov_override:
            return self.text_embedder.embed_batch_or_raise(captions)
        rows = self.text_embedder.embed_batch(captions)
        if self.weights is None:
            padded = np.zeros((rows.shape[0], 512), dtype=np.float64)
            padded[:, : rows.shape[1]] = rows
            return padded
        return rows

    def embed_one(self, caption: str) -> np.ndarray:
        # Same padding the batch path applies, so the search path stays in
        # one dimension with the image side even without a projection.
        row = self.embed_text([caption])[0]
        return row

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        if self.weights is None:
            return np.asarray(descriptors, dtype=np.float64)
        return np.asarray(descriptors, dtype=np.float64) @ self.weights


class NoIdfArm(MistakeArm):
    name = "mistake-no-idf"
    text_variant = "no-idf"


class RawCountArm(MistakeArm):
    name = "mistake-raw-count"
    text_variant = "raw-count"


class PerWordNormArm(MistakeArm):
    name = "mistake-perword-norm"
    text_variant = "perword-norm"


class RawDescriptorArm(MistakeArm):
    name = "mistake-raw-descriptor"
    image_side_raw = True


class OovCrashArm(MistakeArm):
    name = "mistake-oov-crash"
    crash_on_oov = True


# --------------------------------------------------------------------------
# Lookup arms
# --------------------------------------------------------------------------

#: The rewrite filter, derived from perturb.STOPWORDS (the benchmark's own
#: list, which a cheater without the benchmark package would approximate) plus
#: a few common function words of our own. The resolver applies the SAME list
#: when building index keys and when filtering a query: if the rewrite drops a
#: word the resolver keeps, the query tuple can never match its index. The
#: first draft kept "with" while perturb dropped it, which broke resolution
#: for the large share of COCO captions containing "with".
OWN_STOPWORDS = frozenset(
    """
    a about above after again all also am an and any are as at be because been
    before being below between both but by can did do does doing down during
    each few for from further had has have having he her here hers him his how
    i if in into is it its just like me more most my no nor not now of off on
    once only or other our ours out over own she should so some such than that
    the their theirs them then there these they this those through to too under
    until up very was we were what when where which while who whom why will
    with without within onto near beside among same you your us would
    """.split()
)


class LookupArm:
    """Answers from the files the contract hands over. Embeds nothing real.

    Resolution order for a query string: exact caption text; the derived
    keyword keys of the repaired query; the derived three-word keys; and for
    the oracle, the benchmark's own rewrite strings as keys. Unresolved
    queries embed as zero rows and rank at chance.

    The embedding side is a Gaussian codebook: image id -> a fixed 512-d
    random unit vector, identical across calls. Cosine is 1 exactly between
    rows resolved to the same image and near 0 otherwise, which is what
    makes retrieval a lookup wearing a matrix. Random projections of a
    one-hot survive the driver's rounding and the 512-d ceiling: the largest
    cross-cosine among 700 Gaussian vectors is about 0.2, far from 1.
    """

    name = "lookup-derived"
    use_oracle_keys = False

    def __init__(self, resources: Any) -> None:
        blob = resources.load_captions()
        self.exact: Dict[str, int] = {}
        self.exact_ambiguous: Set[str] = set()
        captions_of: Dict[int, List[str]] = defaultdict(list)
        for annotation in blob["annotations"]:
            image_id = int(annotation["image_id"])
            caption = str(annotation["caption"])
            captions_of[image_id].append(caption)
            key = caption.strip().lower()
            if key in self.exact and self.exact[key] != image_id:
                self.exact_ambiguous.add(key)
            self.exact[key] = image_id
        self.captions_of = captions_of
        self.image_tokens: Dict[int, Set[str]] = {
            image_id: set(tokenize(" ".join(captions)))
            for image_id, captions in captions_of.items()
        }
        # Vocabulary for typo repair, with document frequency to prefer the
        # common spelling when several repairs exist.
        self.vocab: Counter = Counter()
        for _image_id, image_captions in captions_of.items():
            for caption in image_captions:
                self.vocab.update(tokenize(caption))
        self.by_length: Dict[int, List[str]] = defaultdict(list)
        for word in self.vocab:
            self.by_length[len(word)].append(word)
        # Derived keys over every caption in the corpus.
        self.keyword_index: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        self.truncated_index: Dict[Tuple[str, ...], Counter] = defaultdict(Counter)
        for image_id, captions in captions_of.items():
            for caption in captions:
                words = tokenize(caption)
                content = tuple(w for w in words if w not in OWN_STOPWORDS) or tuple(words)
                self.keyword_index[content][image_id] += 1
                self.truncated_index[content[:3] or content][image_id] += 1
        if self.use_oracle_keys:
            from language_search_benchmark import perturb

            self.oracle_keyword: Dict[str, int] = {}
            self.oracle_truncated: Dict[str, int] = {}
            for image_id, captions in captions_of.items():
                for caption in captions:
                    self.oracle_keyword.setdefault(perturb.keywords(caption), image_id)
                    self.oracle_truncated.setdefault(perturb.truncated(caption), image_id)
        # The pickle the contract hands over, for the descriptor byte-match
        # that identifies pool rows. Loaded once per arm instance.
        self.resources_descriptors = resources.load_descriptors()
        self._ids: List[int] = []
        self._pool_images: List[int] = []
        self._pool_token_sets: List[Set[str]] = []

    # -- resolution --------------------------------------------------------
    def repair_typo(self, token: str) -> Optional[str]:
        """Most common vocabulary word at edit distance <= 1, or None."""
        if token in self.vocab:
            return token
        best: Optional[str] = None
        best_df = -1
        for length in (len(token) - 1, len(token), len(token) + 1):
            for candidate in self.by_length.get(length, ()):
                if _edit_distance_at_most_one(token, candidate):
                    df = self.vocab[candidate]
                    if df > best_df:
                        best_df = df
                        best = candidate
        return best

    def resolve(self, text: str) -> Optional[int]:
        hit = self.exact.get(text.strip().lower())
        if hit is not None:
            return hit
        words = tokenize(text)
        if not words:
            return None
        content = [w for w in words if w not in OWN_STOPWORDS] or words
        counters = [self.keyword_index.get(tuple(content))]
        counters.append(self.truncated_index.get(tuple(content[:3])))
        for counter in counters:
            if counter:
                return _most_common(counter)
        repaired = []
        changed = False
        for word in content:
            fixed = self.repair_typo(word)
            if fixed is None:
                repaired.append(word)
            else:
                repaired.append(fixed)
                if fixed != word:
                    changed = True
        if changed:
            counter = self.keyword_index.get(tuple(repaired))
            if counter:
                return _most_common(counter)
            counter = self.truncated_index.get(tuple(repaired[:3]))
            if counter:
                return _most_common(counter)
            # One repaired token at a time, keeping the rest as-is: the
            # rewrite corrupts exactly one word, so repairing in two
            # directions covers both the keyword and truncated keys.
            for position, word in enumerate(content):
                if word not in self.vocab:
                    continue
                counter = self.keyword_index.get(
                    tuple(repaired[:position]) + (word,) + tuple(repaired[position + 1 :])
                )
                if counter:
                    return _most_common(counter)
        return None

    def resolve_oracle(self, text: str) -> Optional[int]:
        """Exact rewrite-string keys from the benchmark's own module."""
        hit = self.oracle_keyword.get(text)
        if hit is not None:
            return hit
        hit = self.oracle_truncated.get(text)
        if hit is not None:
            return hit
        # A typo key cannot be enumerated without the query index; the
        # derived repair above covers it.
        return self.resolve(text)

    # -- benchmark surface -------------------------------------------------
    def codebook(self, image_id: int) -> np.ndarray:
        digest = hashlib.md5(
            "{}:{}".format(SEED_CODEBOOK, image_id).encode("utf-8")
        ).digest()
        seed = int.from_bytes(digest[:4], "little")
        vector = np.random.RandomState(seed).normal(size=CODEBOOK_DIM)
        norm = np.linalg.norm(vector)
        return vector / (norm if norm else 1.0)

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        rows = np.zeros((len(captions), CODEBOOK_DIM), dtype=np.float64)
        for row, caption in enumerate(captions):
            if self.use_oracle_keys:
                image_id = self.resolve_oracle(caption)
            else:
                image_id = self.resolve(caption)
            if image_id is not None:
                rows[row] = self.codebook(image_id)
        return rows

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        # Pool rows are identified by matching the descriptor bytes against
        # the pickle the contract hands over. The same route the synthetic
        # memorizer fixture documents, on the real artifact.
        lookup = getattr(self, "_descriptor_index", None)
        if lookup is None:
            lookup = {}
            for image_id, value in self.resources_descriptors.items():
                lookup[np.asarray(value, dtype=np.float32).tobytes()] = image_id
            self._descriptor_index = lookup
        rows = np.zeros((descriptors.shape[0], CODEBOOK_DIM), dtype=np.float64)
        for row in range(descriptors.shape[0]):
            image_id = lookup.get(
                np.asarray(descriptors[row], dtype=np.float32).tobytes()
            )
            if image_id is not None:
                rows[row] = self.codebook(image_id)
        return rows

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = [int(value) for value in image_ids]
        self._pool_images = list(self._ids)
        self._pool_token_sets = [
            self.image_tokens.get(image_id, set()) for image_id in self._ids
        ]

    def search(self, query: str, k: int) -> List[int]:
        if not self._pool_images:
            return []
        tokens = tokenize(query)
        content = [w for w in tokens if w not in OWN_STOPWORDS] or tokens
        repaired = []
        for word in content:
            fixed = self.repair_typo(word)
            repaired.append(fixed if fixed is not None else word)
        wanted = set(repaired)
        scores = []
        for _index, image_tokens in enumerate(self._pool_token_sets):
            scores.append(len(wanted & image_tokens))
        order = sorted(range(len(self._pool_images)), key=lambda i: (-scores[i], i))
        return [self._pool_images[index] for index in order[:k]]


class OracleLookupArm(LookupArm):
    name = "lookup-oracle"
    use_oracle_keys = True


class ExactLookupArm(LookupArm):
    """Exact-string lookup only: the route the existing tests measure."""

    name = "lookup-exact"

    def resolve(self, text: str) -> Optional[int]:
        return self.exact.get(text.strip().lower())

    def search(self, query: str, k: int) -> List[int]:
        hit = self.exact.get(query.strip().lower())
        if hit is not None and hit in self._pool_images:
            rest = [i for i in self._pool_images if i != hit]
            return ([hit] + rest)[:k]
        return self._pool_images[:k]


def _most_common(counter: Counter) -> int:
    """Highest count, lowest image id on ties; deterministic."""
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def _edit_distance_at_most_one(a: str, b: str) -> bool:
    """True when a and b differ by at most one substitution/insert/delete."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(1 for x, y in zip(a, b) if x != y) <= 1
    shorter, longer = (a, b) if len(a) < len(b) else (b, a)
    position = 0
    skipped = False
    for character in longer:
        if position < len(shorter) and shorter[position] == character:
            position += 1
        elif skipped:
            return False
        else:
            skipped = True
    return True


class MemorizeSearchArm(ReferenceStyleArm):
    """Reference embeddings everywhere; the search path is an exact-string
    lookup. Isolates what memorizing the application layer alone is worth."""

    name = "mistake-memsearch"

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = [int(value) for value in image_ids]
        self._by_caption: Dict[str, int] = {}
        for image_id in self._ids:
            for caption in self.captions_of_image(image_id):
                self._by_caption.setdefault(caption.strip().lower(), image_id)

    def captions_of_image(self, image_id: int) -> Sequence[str]:
        return self._lookup_captions.get(image_id, ())

    def __init__(self, resources: Any) -> None:
        super().__init__(resources)
        blob = resources.load_captions()
        self._lookup_captions: Dict[int, List[str]] = defaultdict(list)
        for annotation in blob["annotations"]:
            self._lookup_captions[int(annotation["image_id"])].append(
                str(annotation["caption"])
            )

    def search(self, query: str, k: int) -> List[int]:
        hit = self._by_caption.get(query.strip().lower())
        if hit is not None and hit in self._ids:
            rest = [i for i in self._ids if i != hit]
            return ([hit] + rest)[:k]
        return self._ids[:k]


# --------------------------------------------------------------------------
# Inert controls
# --------------------------------------------------------------------------


class ConstantArm:
    """One shared vector for everything: an all-ties similarity matrix."""

    name = "control-constant"

    def __init__(self, resources: Any) -> None:
        self._ids = []

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return np.ones((len(captions), 8), dtype=np.float64)

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return np.ones((descriptors.shape[0], 8), dtype=np.float64)

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = [int(value) for value in image_ids]

    def search(self, query: str, k: int) -> List[int]:
        return self._ids[:k]


class ZeroArm:
    name = "control-zero"

    def __init__(self, resources: Any) -> None:
        self._ids = []

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return np.zeros((len(captions), 8), dtype=np.float64)

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return np.zeros((descriptors.shape[0], 8), dtype=np.float64)

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = [int(value) for value in image_ids]

    def search(self, query: str, k: int) -> List[int]:
        return []


class RandomArm:
    """Fixed-seed Gaussian rows, uncorrelated with any input."""

    name = "control-random"

    def __init__(self, resources: Any) -> None:
        self._ids = []

    def _rows(self, count: int) -> np.ndarray:
        seed = (SEED_CONTROL + count) % (2**31)
        return np.random.RandomState(seed).normal(size=(count, 16))

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return self._rows(len(captions))

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return self._rows(descriptors.shape[0])

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = [int(value) for value in image_ids]

    def search(self, query: str, k: int) -> List[int]:
        return self._ids[:k]


class PoolOrderArm:
    """Inert embeddings; search returns the pool's own order."""

    name = "control-poolorder"

    def __init__(self, resources: Any) -> None:
        self._ids = []

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return np.ones((len(captions), 8), dtype=np.float64)

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return np.ones((descriptors.shape[0], 8), dtype=np.float64)

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        self._ids = [int(value) for value in image_ids]

    def search(self, query: str, k: int) -> List[int]:
        return self._ids[:k]


class ShuffledArm(ReferenceStyleArm):
    """The reference's vectors with the pairing destroyed: every call returns
    its rows permuted by a fixed seed, so embeddings are real and answers are
    mismatched to inputs."""

    name = "control-shuffled"

    def _shuffle(self, matrix: np.ndarray) -> np.ndarray:
        seed = (SEED_CONTROL + matrix.shape[0]) % (2**31)
        order = np.random.RandomState(seed).permutation(matrix.shape[0])
        return matrix[order]

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return self._shuffle(super().embed_text(captions))

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return self._shuffle(super().embed_images(descriptors))


#: Registry order is the run order, and the report prints it in this order.
ARMS = {
    "reference": ReferenceStyleArm,
    "lookup-derived": LookupArm,
    "lookup-oracle": OracleLookupArm,
    "lookup-exact": ExactLookupArm,
    "mistake-no-idf": NoIdfArm,
    "mistake-raw-count": RawCountArm,
    "mistake-perword-norm": PerWordNormArm,
    "mistake-raw-descriptor": RawDescriptorArm,
    "mistake-oov-crash": OovCrashArm,
    "mistake-memsearch": MemorizeSearchArm,
    "control-constant": ConstantArm,
    "control-zero": ZeroArm,
    "control-random": RandomArm,
    "control-shuffled": ShuffledArm,
    "control-poolorder": PoolOrderArm,
}


def build(arm_name: str, resources: Any) -> Tuple[Any, Optional[str]]:
    """Instantiate an arm, returning (instance, error). An arm whose
    constructor fails comes back as (None, error) and the harness records it
    as failed -- failed and unavailable arms are preserved, never dropped."""
    cls = ARMS[arm_name]
    try:
        return cls(resources), None
    except BaseException as error:  # noqa: BLE001 - a failed arm is a result
        return None, "{}: {}".format(type(error).__name__, error)
