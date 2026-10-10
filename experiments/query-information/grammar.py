"""The frozen fact grammar: a finite language where information is countable.

The benchmark's rung grid rewrites a caption and scores the result. When a
rewritten rung scores below verbatim, two explanations are confounded: the
rewrite may have removed information (fewer words can identify fewer
things), or the submission may have failed to understand language it was
handed. COCO captions cannot separate these -- nobody knows what a caption
"means", so nobody can say what a shortened caption still determines.

This grammar is finite and fully known, which is the point. Every candidate
is a relation scene over three attribute families; every query is the scene's
description in a fixed template; and whether a candidate is consistent with
a rewritten query is decidable by enumeration, not judgment. On this data
the two explanations separate exactly: the grammar computes how much
information each rung's rewrite leaves, and any gap between a system's
score and that ceiling is the system's, not the query's.

The grammar is deliberately small:

COLORS     5 words      (red, blue, green, yellow, purple)
SHAPES     5 words      (circle, square, triangle, star, diamond)
RELATIONS  5 phrases    (left of, above, below, inside, near)

A candidate scene is ``{color1} {shape1} {relation} {color2} {shape2}``;
a query is ``the {color1} {shape1} is {relation} the {color2} {shape2}``.
The template adds the stopwords ``the``/``is``/``of`` so the real
``perturb.keywords`` rewrite has stopwords to remove, exactly as it does on
a caption. First scenes are drawn from a four-pair subset and second scenes
from a two-pair subset, which makes every attribute family's contribution
to identifiability hand-checkable.

Nothing here reads course data, a manifest, or a caption. The frozen
reference versions and hashes live in ``manifest.json``; this module is
deterministic without a random source, and the one place a seed is used --
shuffling the candidate pool into query order, which determines the index
argument the real ``perturb.typo`` receives -- records its seed in the
manifest. Deterministic throughout, stdlib and numpy only.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from language_search_benchmark import perturb

#: The three attribute families, in frozen order.
COLORS: Tuple[str, ...] = ("red", "blue", "green", "yellow", "purple")
SHAPES: Tuple[str, ...] = ("circle", "square", "triangle", "star", "diamond")
RELATIONS: Tuple[str, ...] = ("left of", "above", "below", "inside", "near")

#: First-scene pairs: four (color, shape) combinations, chosen so that some
#: pairs share a color (red, blue) and some share a shape (circle) without
#: sharing both. This is what makes truncated and typo rungs leave a small,
#: exactly-countable ambiguity instead of either zero or many.
FIRST_SCENES: Tuple[Tuple[str, str], ...] = (
    ("red", "circle"),
    ("red", "square"),
    ("blue", "circle"),
    ("blue", "triangle"),
)

#: Second-scene variants for the same (relation, first scene): two possible
#: completions. Two candidates that share first scene and relation differ
#: only here, so a rewrite that drops the second scene identifies a pair,
#: not a scene.
SECOND_SCENES: Tuple[Tuple[str, str], ...] = (
    ("blue", "square"),
    ("green", "diamond"),
)

#: Fixed sentence template. Its stopword content (the, is, of) is what the
#: keywords rung removes; the attribute words survive it unchanged.
QUERY_TEMPLATE = "the {c1} {s1} is {relation} the {c2} {s2}"

#: Seed used for the single shuffle in this study: the candidate pool is
#: generated in enumeration order, then permuted into query order. The
#: permutation matters because the real ``perturb.typo`` places its
#: corruption from the caption's index; a fixed pool order would put every
#: typo in the same character slot. Committed in ``manifest.json``.
POOL_SEED = 20261010

#: The words a query's relation contributes to token matching. "left of"
#: contributes "left" (the "of" is a stopword in the real perturber and
#: never reaches a content token).
RELATION_WORDS: Dict[str, str] = {
    "left of": "left",
    "above": "above",
    "below": "below",
    "inside": "inside",
    "near": "near",
}

#: Every attribute word a candidate scene can carry, frozen.
ATTRIBUTE_VOCABULARY: Tuple[str, ...] = tuple(
    sorted(set(COLORS) | set(SHAPES) | set(RELATION_WORDS.values()))
)


class Scene:
    """One candidate scene: its description and its attribute tokens."""

    def __init__(self, c1: str, s1: str, relation: str, c2: str, s2: str) -> None:
        self.c1 = c1
        self.s1 = s1
        self.relation = relation
        self.c2 = c2
        self.s2 = s2
        self.description = "{} {} {} {} {}".format(c1, s1, relation, c2, s2)
        # Token-level attribute tuple, used for compatibility.
        self.attribute_tokens = (c1, s1, RELATION_WORDS[relation], c2, s2)
        self.tokens = frozenset(self.attribute_tokens)

    def query(self) -> str:
        """The verbatim query that targets this scene."""
        return QUERY_TEMPLATE.format(
            c1=self.c1, s1=self.s1, relation=self.relation, c2=self.c2, s2=self.s2
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "Scene({!r})".format(self.description)


def enumerate_candidates() -> List[Scene]:
    """All candidate scenes in frozen enumeration order.

    5 relations x 4 first scenes x 2 second scenes = 40 candidates. The
    order is part of the frozen grammar: it is the enumeration order the
    pool shuffle permutes, and the tests pin it literally.
    """

    scenes: List[Scene] = []
    for relation in RELATIONS:
        for (c1, s1) in FIRST_SCENES:
            for (c2, s2) in SECOND_SCENES:
                scenes.append(Scene(c1, s1, relation, c2, s2))
    return scenes


def build_pool() -> List[Tuple[str, Scene]]:
    """The study pool in query order: (scene_id, scene), after the seeded
    permutation.

    The shuffle is the study's only random step, its seed is committed,
    and numpy's ``RandomState`` keeps it stable across machines and
    versions, like the benchmark's own tie-break.
    """

    scenes = enumerate_candidates()
    order = np.random.RandomState(POOL_SEED).permutation(len(scenes))
    pool: List[Tuple[str, Scene]] = []
    for index in order:
        scene = scenes[int(index)]
        pool.append(("scene-{:03d}".format(int(index)), scene))
    return pool


def queries_for_pool(pool: List[Tuple[str, Scene]]) -> List[str]:
    """The verbatim query string for each pool entry, in pool order."""

    return [scene.query() for _, scene in pool]


def rewrites_for_pool(pool: List[Tuple[str, Scene]], rung: str) -> List[str]:
    """Every query under one rung, using the real benchmark transforms.

    The index handed to ``perturb.rewrite`` is the query's position in the
    pool, mirroring how the benchmark passes each caption's position in the
    caption set. This module never re-implements a transform.
    """

    return [
        perturb.rewrite(scene.query(), rung, index)
        for index, (_, scene) in enumerate(pool)
    ]
