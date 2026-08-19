"""Query rewrites that turn one caption into a small difficulty grid.

Every scored query in this benchmark is a verbatim COCO caption of the gold
image. That measures one thing well and hides two others. A submission that
learned an embedding space and one that effectively memorized caption strings
score the same, and nothing says which happened.

The course is explicit about what should survive these rewrites. Captions are
embedded as an IDF-weighted sum of GloVe vectors, so dropping the words with
the lowest IDF should barely move the result: that is what the weighting is
for. And an unseen word contributes a zero vector rather than raising, which
is only observable if some query contains one.

So the four rungs are not arbitrary noise. Each one is a prediction the course
material makes, turned into a measurement:

``verbatim``
    The caption unchanged. The control, and the existing behavior.
``keywords``
    Stopwords removed, order kept. What a person actually types into a search
    box. IDF weighting predicts this stays close to verbatim; a submission
    that skipped IDF, or that weighted by raw frequency, falls off here and
    nowhere else.
``truncated``
    The first few content words only. Tests graceful degradation as signal is
    removed, rather than a cliff.
``typo``
    One adjacent-key substitution. The word becomes out of vocabulary, so this
    exercises the zero-vector rule. A submission that raises ``KeyError``
    instead fails loudly here rather than silently scoring near chance.

Deterministic throughout. The rewrite depends only on the caption text and its
index, never on a random source, so two runs of one submission produce the same
queries and a team comparing runs is comparing like with like.

Stdlib and numpy only. A stopword list from `nltk` would be a download, a
version pin, and a network call inside a sandbox that has none.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

#: Words carrying almost no retrieval signal, which is also close to what a
#: low IDF selects for. Kept small and explicit rather than imported: the list
#: is part of the measurement and should be readable next to it.
STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of in on at to from
    by for with without into onto over under near beside between among during
    is are was were be been being am do does did doing have has had having
    it its as some any each both few more most other such no nor not only own
    same so too very can will just there here they them their he she his her
    him you your we our us i me my
    """.split()
)

#: How many content words `truncated` keeps. Three is the shortest phrase that
#: still names a subject and something about it ("man riding horse"), which is
#: the shape of a real short query.
TRUNCATE_WORDS = 3

#: QWERTY neighbours, used to make a typo look like a typo. A random letter
#: substitution would also produce an out-of-vocabulary word, but a fat-finger
#: error is what students will actually see from a user, and it keeps the word
#: close enough that a submission doing prefix or edit-distance matching is
#: tested rather than defeated.
_NEIGHBOURS: Dict[str, str] = {
    "a": "s", "b": "v", "c": "x", "d": "s", "e": "r", "f": "d", "g": "f",
    "h": "g", "i": "u", "j": "h", "k": "j", "l": "k", "m": "n", "n": "b",
    "o": "i", "p": "o", "q": "w", "r": "e", "s": "a", "t": "r", "u": "y",
    "v": "c", "w": "q", "x": "z", "y": "t", "z": "x",
}

RUNGS: Tuple[str, ...] = ("verbatim", "keywords", "truncated", "typo")


def _words(caption: str) -> List[str]:
    return caption.split()


def content_words(caption: str) -> List[str]:
    """The caption's words with stopwords dropped, order preserved.

    Falls back to the original words when every word is a stopword, which does
    happen on short captions ("it is on the table"). An empty query would
    measure our tokenizer rather than the submission.
    """

    kept = [word for word in _words(caption) if word.strip(".,!?;:'\"").lower() not in STOPWORDS]
    return kept or _words(caption)


def keywords(caption: str) -> str:
    return " ".join(content_words(caption))


def truncated(caption: str, keep: int = TRUNCATE_WORDS) -> str:
    return " ".join(content_words(caption)[:keep])


def typo(caption: str, index: int) -> str:
    """One adjacent-key substitution, placed deterministically from ``index``.

    The target is the longest content word, because a typo in a rare long word
    is what actually breaks a lookup; corrupting "the" would change nothing a
    submission depends on. ``index`` picks which character, so different
    queries get their typo in different places without any random source.
    """

    words = _words(caption)
    if not words:
        return caption
    content = content_words(caption)
    target = max(content, key=len)
    position = words.index(target)
    letters = list(target)
    # Interior characters only: a leading or trailing typo is easier to
    # recover from and less representative of real mistyping.
    usable = [i for i, ch in enumerate(letters) if 0 < i < len(letters) - 1 and ch.lower() in _NEIGHBOURS]
    if not usable:
        return caption
    at = usable[index % len(usable)]
    replacement = _NEIGHBOURS[letters[at].lower()]
    letters[at] = replacement.upper() if letters[at].isupper() else replacement
    words[position] = "".join(letters)
    return " ".join(words)


def rewrite(caption: str, rung: str, index: int) -> str:
    """One caption under one rung. Unknown rung names raise rather than pass
    the caption through, since a typo in a rung name would otherwise show up
    as a suspiciously strong result."""

    if rung == "verbatim":
        return caption
    if rung == "keywords":
        return keywords(caption)
    if rung == "truncated":
        return truncated(caption)
    if rung == "typo":
        return typo(caption, index)
    raise ValueError("Unknown rung {!r}; expected one of {}.".format(rung, ", ".join(RUNGS)))


def rewrite_all(captions: Sequence[str], rung: str) -> List[str]:
    """Every caption under one rung, index taken from position."""

    return [rewrite(caption, rung, index) for index, caption in enumerate(captions)]
