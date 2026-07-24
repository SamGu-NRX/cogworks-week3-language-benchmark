"""Validation of submission outputs before the controller scores them.

Every failure raises ``CheckFailure`` with a message precise enough for a
student to act on (what was returned, what was expected, where). These are
contract checks on observed behavior; they never inspect the submission's
source.
"""

from __future__ import annotations

from typing import Any, List, Sequence

import numpy as np


class CheckFailure(RuntimeError):
    pass


#: Bounds on the shared embedding dimension. The course canon is 200 and
#: real submissions have used 50-200; 512 also admits "I returned the raw
#: descriptor" submissions, which score poorly on their own merits rather
#: than being rejected.
MIN_DIM = 8
MAX_DIM = 512


def coerce_matrix(value: Any, expected_rows: int, name: str) -> np.ndarray:
    """Coerce a submission return value to a finite ``(expected_rows, D)`` float array."""

    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise CheckFailure(
            "{} returned a value that does not convert to a float array: {!r}".format(
                name, type(value).__name__
            )
        ) from error
    if matrix.ndim == 1 and expected_rows == 1:
        matrix = matrix.reshape(1, -1)
    if matrix.ndim != 2:
        raise CheckFailure(
            "{} returned an array with {} dimensions; expected a 2-D (rows, D) matrix.".format(
                name, matrix.ndim
            )
        )
    if matrix.shape[0] != expected_rows:
        raise CheckFailure(
            "{} returned {} rows for {} inputs.".format(name, matrix.shape[0], expected_rows)
        )
    if not (MIN_DIM <= matrix.shape[1] <= MAX_DIM):
        raise CheckFailure(
            "{} returned embedding dimension {}; expected between {} and {}.".format(
                name, matrix.shape[1], MIN_DIM, MAX_DIM
            )
        )
    finite = np.isfinite(matrix)
    if not finite.all():
        bad_row = int(np.argwhere(~finite.all(axis=1))[0][0])
        raise CheckFailure(
            "{} returned non-finite values (first offending row index {}).".format(name, bad_row)
        )
    return matrix


def l2_normalize_rows(matrix: np.ndarray) -> np.ndarray:
    """Unit-normalize rows; all-zero rows stay zero (they rank at chance)."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms == 0.0, 1.0, norms)
    return matrix / safe


def validate_rankings(
    value: Any, n_queries: int, allowed_ids: Sequence[int], k: int, name: str
) -> List[List[int]]:
    """Coerce per-query search results into lists of ints, truncated at ``k``.

    Ids outside the given pool are kept (they simply never match gold) but
    the caller surfaces a diagnostic when they dominate.
    """

    if not isinstance(value, (list, tuple)):
        raise CheckFailure(
            "{} must return a list of per-query result lists; got {!r}.".format(
                name, type(value).__name__
            )
        )
    if len(value) != n_queries:
        raise CheckFailure(
            "{} returned {} result lists for {} queries.".format(name, len(value), n_queries)
        )
    rankings: List[List[int]] = []
    for index, row in enumerate(value):
        if row is None:
            rankings.append([])
            continue
        if isinstance(row, (str, bytes)):
            raise CheckFailure(
                "{} returned a string for query index {}; expected a list of image ids.".format(
                    name, index
                )
            )
        try:
            ids = [int(item) for item in list(row)[:k]]
        except (TypeError, ValueError) as error:
            raise CheckFailure(
                "{} returned non-integer image ids for query index {}.".format(name, index)
            ) from error
        rankings.append(ids)
    return rankings
