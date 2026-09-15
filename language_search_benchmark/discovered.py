"""Turn a discovered binding into the object the Week 3 driver already runs.

Discovery finds which of a team's functions embed a caption, embed a
descriptor, build a database, and answer a query. The driver wants one object
with ``embed_text``, ``embed_images``, ``prepare_database``, and ``search``.
This is the piece between them, and it lives here rather than in ``cogbench``
because the protocol it writes to is Week 3's.

Two rules, both from Week 1's version of this file and both measured there.
Their functions are called exactly as the search called them, through
``Candidate.bound``, because the search is what proved the arrangement works
and any deviation would score a different program. And nothing is repaired: a
function that raises raises, the driver records it against that component, and
a bug in their code stays visible as a bug in their code.

What is not a repair is reading their answer. A mygrad ``Tensor`` is the
array it wraps, and ``search`` returning ``{"image_id": ..., "score": ...}``
per hit is a ranking of ids with the score written beside it. Both are read,
neither is recomputed. Where reading is not possible the surface refuses and
says what came back instead, which is what happens to CogFinder's ``search``:
it returns downloaded images, and the ids exist only inside the function.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np

from .checks import CheckFailure

__all__ = ["DiscoveredSearch", "build"]


class NotBound(CheckFailure):
    """A surface the search did not find, named with what it needs."""

    @classmethod
    def absent(cls, surface: str) -> "NotBound":
        """A branch discovery never bound, named so the scorer can see it.

        `surface` says which component's zero is not a zero. It is a field
        rather than a marker in the message because the scorer used to
        sniff one out of the error string and then restate a cause it had
        no access to: scoring runs in the controller while the search runs
        in a sandbox, so the instance that knew why the weights did not
        decide an image step is not the instance that reports it. Which
        surfaces are absent, and what to say about each, comes from
        `DiscoveredSearch.absent_surfaces`, not from here -- an absence
        that no case reached still has to be reported.
        """

        error = cls(_ABSENCE_ERRORS[surface])
        error.surface = surface
        return error


#: What each surface says for itself when nothing told this run more, in
#: the order the branches run. Written as what this run did rather than as
#: what the repository holds: the search is a search, and a sentence about
#: the repository is a claim it cannot make.
#:
#: This is also the component's own error, which the driver truncates at
#: 200 characters, so each one is a single sentence.
_ABSENCE_ERRORS = {
    "text": "this run found no function it could use to turn captions into vectors.",
    "image": (
        "this run found no function it could use to turn image descriptors into "
        "vectors."
    ),
    "prepare": (
        "this run found no function it could use to build a searchable database "
        "from image ids and descriptors."
    ),
    "search": (
        "this run found no function it could use to answer a query string with "
        "image ids."
    ),
}

#: What the run says about each, which is the same sentence plus the
#: contract for the image side: after the old sentence's claim about
#: trained weights came out there was otherwise nothing there to act on.
_ABSENCE_REASONS = dict(
    _ABSENCE_ERRORS,
    image=_ABSENCE_ERRORS["image"]
    + " It looks for one that takes the descriptor array and returns one row per"
    + " image, in the same space as your caption vectors.",
)


class DiscoveredSearch:
    """A team's own functions, wearing the interface the driver expects."""

    def __init__(
        self,
        branches: Dict[str, Sequence[Any]],
        extras: Dict[str, Any],
        image_note: Optional[str] = None,
    ) -> None:
        self._branches = {name: tuple(steps) for name, steps in branches.items()}
        self._extras = dict(extras)
        #: Why there is no image step, when the week can say: its `prepare`
        #: hook is the only thing that can tell an absent projection from
        #: two of them. Replaces the sentence in `_ABSENCE_REASONS`, which
        #: says what the search did and claims nothing about why. The other
        #: three surfaces have no such source, so they have no such field.
        self._image_note = image_note
        self._store: Any = None
        #: What each branch produced in THIS run, keyed by branch name. A
        #: step in one branch that was bound with another branch's output
        #: (Bagel's `CaptionImageQuery(EMBEDDINGS, ids)` takes the image
        #: branch's matrix; every search step takes the prepare branch's
        #: store) reads it from here, through `cogbench.pipeline.runtime_pool`,
        #: instead of from the value the search fixture produced.
        self._live: Dict[str, Any] = {}

    # -- what was found ----------------------------------------------------

    def covered_kinds(self) -> Set[str]:
        """The driver case kinds the bound branches can answer.

        The retrieval case needs both halves of the space, so it is covered
        only when the image branch bound; the search case needs the query path
        as well. A kind that is not covered is left to fail, and ``score``
        withholds its number instead of reporting a zero nobody measured.
        """

        covered = set()
        if "text" in self._branches:
            covered.add("text")
            if "image" in self._branches:
                covered.add("retrieval")
                if "search" in self._branches:
                    covered.add("search")
        return covered

    @property
    def has_image_side(self) -> bool:
        return "image" in self._branches

    @property
    def absent_surfaces(self) -> Dict[str, str]:
        """The branches that did not bind, and what to say about each.

        A fact about the binding, so the driver reads it once rather than
        collecting it from the surfaces the run happens to reach. It used
        to be assembled from the refusals themselves, which meant an
        earlier failure could hide a later absence: a repository with no
        search branch whose bound prepare step raises never reaches
        `search`, and the run then published a search score of zero for a
        surface that was never there.
        """

        notes = dict(_ABSENCE_REASONS)
        if self._image_note:
            notes["image"] = self._image_note
        return {
            name: note for name, note in notes.items() if name not in self._branches
        }

    # -- the protocol ------------------------------------------------------

    def _through(self, name: str, arguments: Sequence[Any]) -> Any:
        """Run one branch on this run's input, with this run's values live."""

        from cogbench.pipeline import runtime_pool

        with runtime_pool(self._live):
            value = _run(self._branches[name], list(arguments))
        self._live[name] = value
        return value

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        if "text" not in self._branches:
            raise NotBound.absent("text")
        return _matrix(self._through("text", [list(captions)]), len(list(captions)), "embed_text")

    def embed_images(self, descriptors: Any) -> np.ndarray:
        rows = int(np.asarray(descriptors).shape[0])
        if "image" not in self._branches:
            # The old sentence asserted that no trained projection was
            # found in the repository. On a repository holding two of them
            # that was false, and it sent the student to commit a file that
            # was already committed.
            raise NotBound.absent("image")
        return _matrix(self._through("image", [np.asarray(descriptors)]), rows, "embed_images")

    def prepare_database(self, image_ids: Sequence[int], descriptors: Any) -> None:
        if "prepare" not in self._branches:
            raise NotBound.absent("prepare")
        # The prepare step was bound on one of several argument forms
        # (`roles.prepare_forms`): ids and raw descriptors either way round,
        # or ids and the image branch's projected matrix either way round.
        # The scored run hands it the same form, made from this run's
        # values; the image branch runs first so the projected forms exist.
        from .roles import prepare_forms

        projected = None
        if "image" in self._branches:
            projected = self._through("image", [np.asarray(descriptors)])
        forms = prepare_forms(list(image_ids), np.asarray(descriptors), projected)
        first = self._branches["prepare"][0]
        which = getattr(first, "form", None) or 0
        if which >= len(forms):
            raise NotBound(
                "{} was bound on argument form {} and this run has only {} forms; "
                "the image branch did not run".format(first.label, which, len(forms))
            )
        self._store = self._through("prepare", list(forms[which]))
        self._extras["store"] = self._store

    def search(self, query: str, k: int) -> List[int]:
        if "search" not in self._branches:
            raise NotBound.absent("search")
        # The search step was bound on the text branch's chain applied to
        # the query, not on the raw string (finding 3 of the engine review):
        # the same call has to be made here.
        vector = self._through("text", [[query]])
        row = np.asarray(getattr(vector, "data", vector), dtype=np.float64)
        row = row[0] if row.ndim == 2 else row
        with __import__("cogbench.pipeline", fromlist=["runtime_pool"]).runtime_pool(self._live):
            answer = _run(self._branches["search"], [row, k])
        ids = _ids(answer)
        if ids is None:
            raise NotBound(
                "{} answered with {}, not with image ids; the ranking exists "
                "only inside that function, so there is nothing to read."
                .format(self._branches["search"][-1].label, _describe(answer))
            )
        return ids


def _run(chain: Sequence[Any], arguments: Sequence[Any]) -> Any:
    """Push the benchmark's input through their chain, changing nothing.

    The first step is called with the arguments the search bound it against;
    every later step is called with what the step before it returned, through
    `bound`, which applies the hand-off the search recorded. No retry: a
    second call with a different reading is a call the search never proved,
    and a stateful step would run twice. An independent review built a step
    that rejected the tuple and accepted its first element, and the old
    fallback scored that unproved second call as the answer.
    """

    value = _call(chain[0], list(arguments))
    for step in chain[1:]:
        value = _call(step, [value])
    return value


def _call(step: Any, arguments: Sequence[Any]) -> Any:
    call = getattr(step, "bound", None) or step.call
    return call(*arguments)


def _matrix(value: Any, rows: int, name: str) -> np.ndarray:
    """Their answer as an ``(N, D)`` float array, read rather than rebuilt.

    ``roles.as_matrix`` does the reading: a mygrad ``Tensor`` is its ``.data``,
    a list of per-caption rows is those rows stacked. A value that is neither
    is not adapted into one, because guessing at a shape is how a wrong
    binding turns into a plausible number.
    """

    from .roles import as_matrix

    array = as_matrix(value)
    if array is None or array.ndim != 2 or array.shape[0] != rows:
        raise CheckFailure(
            "{} returned {}, not one row per item.".format(name, _describe(value))
        )
    return array


def _ids(answer: Any) -> Optional[List[int]]:
    """The image ids in whatever their search returned, or None.

    Two shapes from the corpus: a list of ids, and a list of
    ``{"image_id": ..., "score": ...}`` rows, which is Bagel's
    `CaptionImageQuery.search`. Returning those rows unread fails the driver's
    own check with "search returned non-integer image ids", so the id is taken
    out of each row and the order their ranking chose is kept.
    """

    if answer is None or isinstance(answer, (str, bytes, dict)):
        return None
    try:
        rows = list(answer)
    except TypeError:
        return None
    ids: List[int] = []
    for row in rows:
        if isinstance(row, dict):
            row = row.get("image_id", row.get("id"))
        if isinstance(row, (tuple, list)) and row:
            row = row[0]
        try:
            ids.append(int(row))
        except (TypeError, ValueError):
            return None
    return ids


def _describe(value: Any) -> str:
    if isinstance(value, np.ndarray):
        return "an array of shape {}".format(tuple(value.shape))
    if isinstance(value, list) and value:
        return "a list of {} {}".format(len(value), type(value[0]).__name__)
    return "a {}".format(type(value).__name__)


def build(
    submission: Any,
    extras: Optional[Dict[str, Any]] = None,
    image_note: Optional[str] = None,
) -> DiscoveredSearch:
    """Wrap a resolved ``cogbench.resolve.Submission``.

    The branches are the binding; a role made of branches leaves
    ``Submission.chain`` empty on purpose and ``ready`` reads the branches.
    ``image_note`` is why there is no image step, when the week knows;
    it travels no other way. See ``DiscoveredSearch.absent_surfaces``.
    """

    branches = dict(getattr(submission, "branches", {}) or {})
    if not branches:
        raise RuntimeError(
            "This repository did not resolve, so there is nothing to run: {}".format(
                getattr(getattr(submission, "verdict", None), "headline", "")
            )
        )
    return DiscoveredSearch(branches, dict(extras or {}), image_note)
