"""Run the three component cases through a submission.

Component isolation is the point: a failure in one component becomes a
``{"ok": False, "error": ...}`` output that scores zero with a diagnostic,
while the other components still run. Only a submission whose factory cannot
produce any usable adapter fails everything (and even then the run itself
completes, carrying the mapping report to the student).

Outputs are JSON-serializable by construction: embeddings are unit-
normalized (a submission's own normalization convention therefore cannot
skew cosine scores) and rounded to five decimals, which keeps the largest
evaluation payload well under the hosted 8 MiB cap.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

import numpy as np

from .adapters import adapt_search
from .checks import CheckFailure, l2_normalize_rows, validate_rankings
from .contracts import AdapterContractError

ROUND_DECIMALS = 5


def _error_output(kind: str, error: BaseException, extra: Sequence[str] = ()) -> Dict[str, Any]:
    lines = [str(error).splitlines()[0][:200] if str(error) else type(error).__name__]
    lines.extend(str(item)[:200] for item in extra)
    return {"ok": False, "kind": kind, "error": " | ".join(lines)}


def _rounded(matrix: np.ndarray) -> List[List[float]]:
    return np.round(l2_normalize_rows(matrix), ROUND_DECIMALS).tolist()


def instantiate_adapter(factory: Any, resources: Any) -> Any:
    """Call the submission factory once; adapt the result."""

    produced = factory(resources) if callable(factory) else factory
    return adapt_search(produced)


def error_outputs(cases: Sequence[Any], error: BaseException) -> List[Dict[str, Any]]:
    """Every component failed the same way (an unusable adapter)."""

    if isinstance(error, AdapterContractError):
        extra: Sequence[str] = list(error.report)
    else:
        extra = ["the submission factory raised while constructing the adapter"]
    return [_error_output(getattr(case, "kind", "?"), error, extra) for case in cases]


def run_cases(factory: Any, resources: Any, cases: Sequence[Any]) -> List[Dict[str, Any]]:
    try:
        adapter = instantiate_adapter(factory, resources)
    except Exception as error:  # noqa: BLE001 - student code; report, don't crash
        return error_outputs(cases, error)
    return run_with_adapter(adapter, cases)


def run_with_adapter(adapter: Any, cases: Sequence[Any]) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for case in cases:
        kind = getattr(case, "kind", "?")
        try:
            if kind == "text":
                embeddings = adapter.embed_text(case.captions)
                outputs.append({"ok": True, "kind": kind, "embeddings": _rounded(embeddings)})
            elif kind == "retrieval":
                text_matrix = adapter.embed_text(case.queries)
                image_matrix = adapter.embed_images(case.descriptors)
                if text_matrix.shape[1] != image_matrix.shape[1]:
                    raise CheckFailure(
                        "text embeddings are {}-d but image embeddings are {}-d; they must "
                        "share one space for retrieval.".format(
                            text_matrix.shape[1], image_matrix.shape[1]
                        )
                    )
                outputs.append(
                    {
                        "ok": True,
                        "kind": kind,
                        "text": _rounded(text_matrix),
                        "images": _rounded(image_matrix),
                    }
                )
            elif kind == "search":
                if not adapter.has_search:
                    raise CheckFailure(
                        "the submission exposes no search(query, k) surface; the search "
                        "component scores zero until one is added."
                    )
                adapter.prepare_database(case.image_ids, case.descriptors)
                rankings = [adapter.search(query, case.k) for query in case.queries]
                validated = validate_rankings(
                    rankings, len(case.queries), case.image_ids, case.k, "search"
                )
                outputs.append({"ok": True, "kind": kind, "rankings": validated})
            else:
                raise CheckFailure("unknown case kind {!r}.".format(kind))
        except (CheckFailure, AdapterContractError) as error:
            outputs.append(_error_output(kind, error, getattr(error, "report", [])))
        except Exception as error:  # noqa: BLE001 - student code; isolate the component
            outputs.append(_error_output(kind, error, ["the submission raised while running"]))

    mappings = getattr(adapter, "mappings", [])
    if mappings and outputs:
        outputs[0].setdefault("mappings", list(mappings)[:8])
    return outputs
