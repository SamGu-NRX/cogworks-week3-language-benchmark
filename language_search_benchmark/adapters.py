"""Bounded auto-adaptation of a submission object onto the search protocol.

Three rungs, none of which guesses:

1. The object already exposes ``embed_text`` and ``embed_images`` — used
   directly. If a batch call returns the wrong shape, the same method is
   retried once per item (some students write per-caption functions); the
   fallback is recorded in the mapping log.
2. The object exposes a documented course surface under an exact known name
   (the lecture's ``text_embedding``, the notebook's ``se_text``/``se_image``,
   and close course variants). Every alias applied is recorded.
3. Otherwise ``AdapterContractError`` carries a mapping report: what the
   object is, which protocol members are missing, the closest names it does
   have, and the ten-line escape hatch.

``prepare_database``/``search`` are adapted the same way but their absence
is not an error here; the search component simply scores zero with a note
(the driver handles that).
"""

from __future__ import annotations

import difflib
from typing import Any, Callable, List, Optional, Sequence

import numpy as np

from .checks import CheckFailure, coerce_matrix
from .contracts import AdapterContractError

#: Exact documented aliases, in preference order. Sources: protocol names,
#: Reynaldo's recommended utility surface (lecture), the course notebook's
#: ``se_*`` names, and common student phrasings observed in real repos.
TEXT_ALIASES = ("embed_text", "embed_texts", "text_embedding", "embed_query", "embed_caption", "se_text")
IMAGE_ALIASES = ("embed_images", "embed_image", "image_embedding", "embed_descriptor", "embed_descriptors", "se_image")
SEARCH_ALIASES = ("search", "query", "query_database")
PREPARE_ALIASES = ("prepare_database", "build_database", "create_database")

_PROTOCOL_DOC = (
    'Add a benchmark_adapter.py that returns an object with embed_text(captions) '
    "and embed_images(descriptors), plus prepare_database(image_ids, descriptors) "
    "and search(query, k) for the search component; register it under the "
    '"cogworks.submissions.v2" entry-point group as "language-search".'
)


class AdaptedSearchAdapter:
    """The submission behind a uniform surface, with a mapping log."""

    def __init__(
        self,
        target: Any,
        text_fn: Callable[..., Any],
        image_fn: Callable[..., Any],
        search_fn: Optional[Callable[..., Any]],
        prepare_fn: Optional[Callable[..., Any]],
        mappings: List[str],
    ) -> None:
        self._target = target
        self._text_fn = text_fn
        self._image_fn = image_fn
        self._search_fn = search_fn
        self._prepare_fn = prepare_fn
        self.mappings = mappings

    @property
    def has_search(self) -> bool:
        return self._search_fn is not None

    def embed_text(self, captions: Sequence[str]) -> np.ndarray:
        return self._batch_or_loop(self._text_fn, list(captions), "embed_text")

    def embed_images(self, descriptors: np.ndarray) -> np.ndarray:
        return self._batch_or_loop(self._image_fn, np.asarray(descriptors), "embed_images")

    def prepare_database(self, image_ids: Sequence[int], descriptors: np.ndarray) -> None:
        if self._prepare_fn is not None:
            self._prepare_fn(list(image_ids), np.asarray(descriptors))

    def search(self, query: str, k: int) -> Sequence[int]:
        if self._search_fn is None:
            raise AdapterContractError(
                "The submission exposes no search surface.", [_PROTOCOL_DOC]
            )
        return self._search_fn(query, k)

    def _batch_or_loop(self, fn: Callable[..., Any], items: Any, name: str) -> np.ndarray:
        expected = len(items)
        try:
            return coerce_matrix(fn(items), expected, name)
        except Exception as batch_error:  # noqa: BLE001 - deliberate: retry per item
            try:
                rows = [np.asarray(fn(item), dtype=np.float64).reshape(-1) for item in items]
                widths = {row.shape[0] for row in rows}
                if len(widths) != 1:
                    raise CheckFailure(
                        "{} returned inconsistent widths per item: {}.".format(
                            name, sorted(widths)[:8]
                        )
                    )
                note = "called {} once per item (batch call failed: {})".format(
                    name, _first_line(batch_error)
                )
                if note not in self.mappings:
                    self.mappings.append(note)
                return coerce_matrix(np.stack(rows), expected, name)
            except Exception as loop_error:  # noqa: BLE001 - compose the report
                if isinstance(batch_error, CheckFailure):
                    # The batch call produced output of the wrong shape; that
                    # story is authoritative, the retry was best-effort.
                    raise batch_error
                # The batch call raised from inside the submission; lead with
                # that error, the per-item retry failure is secondary (a
                # string iterated per character is noise, not signal).
                raise CheckFailure(
                    "{} raised: {}. A per-item retry also failed: {}.".format(
                        name, _first_line(batch_error), _first_line(loop_error)
                    )
                ) from batch_error


def _first_line(error: BaseException) -> str:
    return str(error).splitlines()[0][:160] if str(error) else type(error).__name__


def _public_callables(target: Any) -> List[str]:
    names = []
    for name in dir(target):
        if name.startswith("_"):
            continue
        try:
            if callable(getattr(target, name)):
                names.append(name)
        except Exception:  # noqa: BLE001 - properties may raise; skip them
            continue
    return sorted(names)


def _resolve(target: Any, aliases: Sequence[str], mappings: List[str], role: str) -> Optional[Callable[..., Any]]:
    for alias in aliases:
        candidate = getattr(target, alias, None)
        if callable(candidate):
            if alias != aliases[0]:
                mappings.append("mapped {} -> {}".format(alias, role))
            return candidate
    return None


def adapt_search(target: Any) -> AdaptedSearchAdapter:
    """Wrap a submission object, or raise the mapping report."""

    if isinstance(target, AdaptedSearchAdapter):
        return target
    mappings: List[str] = []
    text_fn = _resolve(target, TEXT_ALIASES, mappings, "embed_text")
    image_fn = _resolve(target, IMAGE_ALIASES, mappings, "embed_images")
    search_fn = _resolve(target, SEARCH_ALIASES, mappings, "search")
    prepare_fn = _resolve(target, PREPARE_ALIASES, mappings, "prepare_database")

    if text_fn is None or image_fn is None:
        available = _public_callables(target)
        missing = []
        if text_fn is None:
            missing.append("embed_text(captions)")
        if image_fn is None:
            missing.append("embed_images(descriptors)")
        report = [
            "received {} with callables: {}".format(
                type(target).__name__, ", ".join(available) or "none"
            ),
            "missing: {}".format(", ".join(missing)),
        ]
        close = difflib.get_close_matches(
            "embed", available, n=3, cutoff=0.3
        ) or difflib.get_close_matches("text", available, n=3, cutoff=0.3)
        if close:
            report.append(
                "closest names found: {} (not mapped automatically; only exact "
                "documented names are).".format(", ".join(close))
            )
        report.append(_PROTOCOL_DOC)
        raise AdapterContractError(
            "The submission does not expose the search adapter protocol.", report
        )
    return AdaptedSearchAdapter(target, text_fn, image_fn, search_fn, prepare_fn, mappings)
