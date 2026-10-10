"""Compatibility, decoding, and the study's systems.

Everything here rests on one frozen definition:

**Compatibility (qi-compat-1).** A candidate scene is *compatible* with a
rewritten query if the query's token MULTISET embeds in the candidate's
attribute-token multiset: every occurrence of every query token must be
covered by a distinct candidate attribute slot. Multiplicity matters -- a
query naming two blue things is not satisfied by a scene holding one --
and token order is ignored. Corrupted tokens (not in the frozen attribute
vocabulary) are *ignored* by the lexical decoder, mirroring the course
rule that an unseen word contributes a zero vector rather than raising.
The grammar-aware decoder additionally replaces a corrupted token by its
unique in-vocabulary correction -- a one-substitution neighbor -- when
exactly one exists.

Two notions of identifiability follow, and their difference is the study:

- **Lexical identifiability**: unique compatible candidate under the lexical
  decoder. What a token-matching system could achieve.
- **Grammar identifiability** (conditional identifiability): unique
  compatible candidate under the grammar-aware decoder. What the information
  in the rewritten query still determines, once language robustness
  (undoing a fat-finger typo within a known vocabulary) is allowed.

A rung score below verbatim is explained by information removal exactly to
the extent lexical identifiability drops; the remaining gap to grammar
identifiability is recoverable by the system, not missing from the query.

Neither notion is general semantic truth. They are properties of this finite
grammar, and the numbers only mean anything inside it. This module is a
study instrument, not a scorer.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from language_search_benchmark import metrics

from grammar import ATTRIBUTE_VOCABULARY, Scene

#: Seed for the shared tie-break, passed to the benchmark's own
#: ``metrics.rank_matrix`` so this study breaks ties the way the benchmark
#: does. Committed in ``manifest.json``.
TIE_BREAK_SEED = 42


def substitution_neighbors(word: str, vocabulary: Sequence[str]) -> List[str]:
    """Vocabulary words one single-character substitution away.

    Same length, exactly one differing position. The real ``perturb.typo``
    only substitutes (never inserts or deletes), so this is the correction
    class its corruptions live in.
    """

    matches: List[str] = []
    for candidate in vocabulary:
        if len(candidate) != len(word) or candidate == word:
            continue
        differences = sum(1 for a, b in zip(word, candidate) if a != b)
        if differences == 1:
            matches.append(candidate)
    return matches


class LexicalDecoder:
    """Compat sets with corrupted tokens ignored (the course's zero-vector rule)."""

    name = "lexical"

    def vocabulary(self) -> Set[str]:
        return set(ATTRIBUTE_VOCABULARY)

    def decoded_tokens(self, rewrite: str) -> List[str]:
        return [token for token in rewrite.split() if token in ATTRIBUTE_VOCABULARY]


class GrammarDecoder:
    """Compat sets with corrupted tokens corrected when uniquely correctable.

    A corrupted token with zero or multiple one-substitution corrections is
    treated as uninformative (dropped), which is the conservative choice:
    ambiguity in the decoder can only widen the compatible set, never
    narrow it.
    """

    name = "grammar"

    def __init__(self) -> None:
        self._correction_counts: Dict[str, int] = {}

    def vocabulary(self) -> Set[str]:
        return set(ATTRIBUTE_VOCABULARY)

    def decoded_tokens(self, rewrite: str) -> List[str]:
        tokens: List[str] = []
        for token in rewrite.split():
            if token in ATTRIBUTE_VOCABULARY:
                tokens.append(token)
                continue
            corrections = substitution_neighbors(token, ATTRIBUTE_VOCABULARY)
            self._correction_counts[token] = len(corrections)
            if len(corrections) == 1:
                tokens.append(corrections[0])
        return tokens

    def correction_log(self) -> Dict[str, int]:
        return dict(sorted(self._correction_counts.items()))


def compatible_indices(
    decoded_tokens: Sequence[str], pool: List[Tuple[str, Scene]]
) -> List[int]:
    """Pool indices of candidates compatible with the decoded tokens (qi-compat-1).

    Multiset containment: every occurrence of a decoded token must be
    covered by a distinct attribute slot of the candidate. "blue circle
    below blue square" therefore does not match "red circle below blue
    square" -- the query demands two blues, the scene has one.
    """

    required = Counter(decoded_tokens)
    hits: List[int] = []
    for index, (_, scene) in enumerate(pool):
        available = Counter(scene.attribute_tokens)
        if not any(count > available[token] for token, count in required.items()):
            hits.append(index)
    return hits


class ExactGrammarSystem:
    """Ceiling system: grammar-aware decoding, attribute-overlap scoring.

    Scores each candidate by the number of decoded query tokens its
    attribute tokens contain, then ranks with the benchmark's seeded
    tie-break. It never guesses beyond the grammar, so its score is bounded
    by grammar identifiability.
    """

    name = "exact-grammar"

    def __init__(self, pool: List[Tuple[str, Scene]], decoder: GrammarDecoder) -> None:
        self.pool = pool
        self.decoder = decoder

    def rank(self, rewrite: str) -> Tuple[np.ndarray, List[int]]:
        tokens = self.decoder.decoded_tokens(rewrite)
        compat = compatible_indices(tokens, self.pool)
        required = Counter(tokens)
        scores = np.zeros(len(self.pool), dtype=np.float64)
        for index, (_, scene) in enumerate(self.pool):
            available = Counter(scene.attribute_tokens)
            scores[index] = sum(
                min(count, available[token]) for token, count in required.items()
            )
        order = metrics.rank_matrix(scores.reshape(1, -1), TIE_BREAK_SEED)[0]
        ranks = np.empty(len(self.pool), dtype=np.int64)
        ranks[order] = np.arange(1, len(self.pool) + 1)
        return ranks, compat


class LexicalIdfSystem:
    """The course's own mechanism, miniature and offline: IDF-weighted
    bag-of-words with the zero-vector rule for unseen words.

    Candidate documents are their description token bags; document
    frequency comes from the frozen 40-candidate corpus. A query token with
    zero document frequency (the template words "the"/"is") or a corrupted
    token contributes nothing, exactly as an unseen word contributes a zero
    vector in the course's GloVe pipeline. Scores are cosine similarities,
    ranked with the benchmark's seeded tie-break.
    """

    name = "lexical-idf"

    def __init__(self, pool: List[Tuple[str, Scene]]) -> None:
        self.pool = pool
        documents = [scene.description.split() for _, scene in pool]
        self.vocab: List[str] = sorted({token for doc in documents for token in doc})
        self.index: Dict[str, int] = {token: i for i, token in enumerate(self.vocab)}
        count = len(documents)
        self.idf = np.zeros(len(self.vocab), dtype=np.float64)
        for column, token in enumerate(self.vocab):
            df = sum(1 for doc in documents if token in doc)
            self.idf[column] = np.log(count / df) if df else 0.0
        self.matrix = np.zeros((len(pool), len(self.vocab)), dtype=np.float64)
        for row, doc in enumerate(documents):
            for token in doc:
                self.matrix[row, self.index[token]] += self.idf[self.index[token]]

    def _vector(self, rewrite: str) -> np.ndarray:
        vector = np.zeros(len(self.vocab), dtype=np.float64)
        for token in rewrite.split():
            column = self.index.get(token)
            if column is not None and self.idf[column] > 0.0:
                vector[column] += self.idf[column]
        return vector

    def rank(self, rewrite: str) -> Tuple[np.ndarray, List[int]]:
        vector = self._vector(rewrite)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            scores = np.zeros(len(self.pool), dtype=np.float64)
        else:
            query = vector / norm
            matrix_norms = np.linalg.norm(self.matrix, axis=1, keepdims=True)
            safe = np.where(matrix_norms == 0.0, 1.0, matrix_norms)
            scores = (self.matrix / safe) @ query
        # The lexical decoder defines what this system's answers could justify.
        compat = compatible_indices(
            [token for token in rewrite.split() if token in ATTRIBUTE_VOCABULARY],
            self.pool,
        )
        order = metrics.rank_matrix(scores.reshape(1, -1), TIE_BREAK_SEED)[0]
        ranks = np.empty(len(self.pool), dtype=np.int64)
        ranks[order] = np.arange(1, len(self.pool) + 1)
        return ranks, compat


def discover_adapters() -> Dict[str, object]:
    """Registered benchmark adapters visible in this environment.

    The study runs adapters only when one is importable offline; a run with
    none available records that fact here instead of inventing a baseline.
    Names only: even a registered adapter needs the course data files, which
    the study never fetches.
    """

    from importlib.metadata import entry_points

    try:
        found = entry_points(group="cogworks.submissions.v2")
    except TypeError:  # pragma: no cover - Python < 3.10 selectable API
        found = entry_points()["cogworks.submissions.v2"]
    return {
        "group": "cogworks.submissions.v2",
        "registered": sorted(entry.name for entry in found),
        "ran": [],
        "reason_not_run": (
            "baseline not run: no adapters registered in this environment"
            if not found
            else "baseline not run offline: registered adapters need course data files"
        ),
    }


def mrr_of_ranks(ranks: Sequence[int]) -> float:
    """Mean reciprocal rank, whole pool (no depth cap in this study)."""
    if not ranks:
        return 0.0
    return float(np.mean([1.0 / rank for rank in ranks]))
