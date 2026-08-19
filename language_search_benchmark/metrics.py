"""Controller-side ranking and scores.

All similarity math happens here, on embeddings the submission returned.
There is no reference implementation anywhere in this path: an inert or
broken submission produces an all-ties similarity matrix, the seeded
tie-break turns that into a fixed random permutation, and every metric lands
at chance. Ties are broken by a seeded permutation so results are
deterministic across runs and machines.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def tie_break_permutation(size: int, seed: int) -> np.ndarray:
    return np.random.RandomState(seed).permutation(size)


def rank_matrix(scores: np.ndarray, seed: int) -> np.ndarray:
    """Row-wise ranking: ``result[q]`` lists column indices best-first.

    Order key is (-score, seeded shuffle), so exact ties resolve to a fixed
    random order instead of favoring low indices (which would reward
    all-constant embeddings for gold rows that happen to sit early).
    """

    permutation = tie_break_permutation(scores.shape[1], seed)
    order = np.lexsort((permutation[np.newaxis, :].repeat(scores.shape[0], axis=0), -scores), axis=1)
    return order


def ranks_of_gold(order: np.ndarray, gold_columns: Sequence[int]) -> np.ndarray:
    """1-based rank of each row's gold column inside its ordering."""

    gold = np.asarray(list(gold_columns), dtype=np.int64)
    positions = np.argsort(order, axis=1)
    return positions[np.arange(order.shape[0]), gold] + 1


def mrr(ranks: np.ndarray) -> float:
    if ranks.size == 0:
        return 0.0
    return float(np.mean(1.0 / ranks))


def recall_at(ranks: np.ndarray, k: int) -> float:
    if ranks.size == 0:
        return 0.0
    return float(np.mean(ranks <= k))


def median_rank(ranks: np.ndarray) -> float:
    if ranks.size == 0:
        return 0.0
    return float(np.median(ranks))


def chance_mrr(pool_size: int) -> float:
    """Expected MRR of a uniformly random ranking over ``pool_size`` items."""

    if pool_size <= 0:
        return 0.0
    return float(np.sum(1.0 / np.arange(1, pool_size + 1)) / pool_size)


def text_first_relevant_ranks(
    embeddings: np.ndarray, group_rows: Sequence[int], seed: int
) -> np.ndarray:
    """For each caption, the 1-based rank of the first co-caption.

    Every caption queries the rest of the pool (itself excluded); relevant
    items are the other captions of the same source image. Singleton groups
    are skipped (no relevant item exists).
    """

    groups = np.asarray(list(group_rows), dtype=np.int64)
    scores = embeddings @ embeddings.T
    np.fill_diagonal(scores, -np.inf)
    order = rank_matrix(scores, seed)
    same_group = groups[order] == groups[:, np.newaxis]
    counts = np.bincount(groups)
    has_relevant = counts[groups] > 1
    ranks: List[int] = []
    for row in range(order.shape[0]):
        if not has_relevant[row]:
            continue
        hits = np.nonzero(same_group[row])[0]
        ranks.append(int(hits[0]) + 1)
    return np.asarray(ranks, dtype=np.int64)


def chance_mrr_first_relevant(pool_size: int, relevant: int) -> float:
    """Expected MRR of the first of ``relevant`` items in a random ranking.

    Exact expectation: P(first relevant at rank r) =
    C(pool-r, relevant-1)/C(pool, relevant); computed iteratively.
    """

    if pool_size <= 0 or relevant <= 0:
        return 0.0
    probability = float(relevant) / pool_size
    total = probability * 1.0
    current = probability
    for rank in range(2, pool_size - relevant + 2):
        current *= (pool_size - relevant - rank + 2) / float(pool_size - rank + 1)
        total += current / rank
    return float(total)


def text_chance_floor(group_rows: Sequence[int]) -> float:
    """Query-weighted chance floor matching ``text_first_relevant_ranks``.

    Each caption in a group of size ``g`` queries the other ``N - 1``
    captions with ``g - 1`` relevant, so the exact floor is the mean of
    ``chance_mrr_first_relevant(N - 1, g - 1)`` over the scored queries.
    Singleton groups contribute no queries, exactly as the rank function
    skips them. This replaced a collapse to the rounded mean group size,
    which understated the floor on uneven groups (sizes [6, 2, 2, 2]
    reported 0.4040 against a true 0.4717). Each size's weight is computed
    as its fraction of the scored queries so that a uniform pool -- both
    shipped manifests are uniform size-5 groups -- multiplies the single
    per-size value by exactly 1.0 and reproduces the previous float bit for
    bit; summing per-group terms and dividing at the end does not guarantee
    that.
    """

    groups = np.asarray(list(group_rows), dtype=np.int64)
    if groups.size == 0:
        return 0.0
    counts = np.bincount(groups)
    sizes = counts[counts > 1]
    scored_queries = int(sizes.sum())
    if scored_queries == 0:
        return 0.0
    pool = groups.size - 1
    floor = 0.0
    for size in np.unique(sizes):
        queries = int(size) * int(np.sum(sizes == size))
        floor += (queries / scored_queries) * chance_mrr_first_relevant(pool, int(size) - 1)
    return float(floor)


def search_ranks(
    rankings: Sequence[Sequence[int]],
    gold_image_ids: Sequence[int],
    k: int,
    pool: Optional[Sequence[int]] = None,
) -> Tuple[np.ndarray, int]:
    """1-based rank of gold per query, plus how many returned ids were foreign.

    ``ranks[i]`` is 0 when gold is absent from the first ``k`` results, which
    ``mrr_with_misses`` scores as a miss rather than as rank 1.

    The second return value is the count of returned image ids that are not in
    ``pool``. It is 0 when no pool is given. Previously this parameter was
    ``pool_size: int``, was never read, and the function returned
    ``len(gold_image_ids)`` -- the query count -- under a docstring promising a
    foreign-id count. The caller then recomputed the real count itself and used
    the wrong value only as a truthiness guard, so the bug was invisible: a
    non-empty query list is truthy in exactly the cases a real foreign count
    would be. Returning what the docstring says removes the duplicate loop and
    the trap.
    """

    ranks = np.zeros(len(gold_image_ids), dtype=np.int64)
    for index, (returned, gold) in enumerate(zip(rankings, gold_image_ids)):
        for position, image_id in enumerate(returned[:k]):
            if int(image_id) == int(gold):
                ranks[index] = position + 1
                break
    if pool is None:
        return ranks, 0
    allowed = {int(image_id) for image_id in pool}
    foreign = sum(
        1 for row in rankings for image_id in row if int(image_id) not in allowed
    )
    return ranks, foreign


def mrr_with_misses(ranks: np.ndarray) -> float:
    if ranks.size == 0:
        return 0.0
    reciprocal = np.where(ranks > 0, 1.0 / np.maximum(ranks, 1), 0.0)
    return float(np.mean(reciprocal))


def summarize_norms(matrix: np.ndarray, name: str) -> Optional[str]:
    """A diagnostic when raw embedding norms look degenerate."""

    norms = np.linalg.norm(matrix, axis=1)
    if norms.size == 0:
        return None
    zero = int(np.sum(norms == 0.0))
    if zero:
        return "{}: {} of {} embeddings are all-zero (they rank at chance).".format(
            name, zero, norms.size
        )
    spread = float(np.std(matrix, axis=0).mean())
    if spread < 1e-7:
        return "{}: embeddings are nearly identical across inputs (model may be untrained).".format(
            name
        )
    return None


def component_scores(
    outputs: Sequence[Dict], cases: Sequence, k_search: int
) -> Tuple[Dict[str, float], List[str]]:
    """All metrics plus diagnostics from the three component outputs.

    ``outputs[i]`` is the driver's dict for ``cases[i]``; a failed component
    carries ``{"ok": False, "error": ...}`` and scores zero with a
    diagnostic, leaving the other components intact.
    """

    diagnostics: List[str] = []
    # Keyed by kind, and for search by kind AND rung. Several search cases now
    # share one kind, one per query rewrite, and a plain by-kind dict would
    # keep whichever came last: the scored search component would silently
    # become the typo rung. The verbatim case is the one `overall` counts.
    by_kind: Dict[str, Tuple] = {}
    rung_cases: List[Tuple] = []
    for case, output in zip(cases, outputs):
        kind = getattr(case, "kind", "?")
        if kind == "search" and getattr(case, "rung", "verbatim") != "verbatim":
            rung_cases.append((case, output))
            continue
        by_kind[kind] = (case, output)

    text_score = 0.0
    text_chance = 0.0
    if "text" in by_kind:
        case, output = by_kind["text"]
        groups = case.group_rows
        if groups is None:
            raise ValueError("Text case has no gold groups; scoring requires the controller copy.")
        text_chance = text_chance_floor(groups)
        if output.get("ok"):
            embeddings = np.asarray(output["embeddings"], dtype=np.float64)
            note = summarize_norms(embeddings, "text embeddings")
            if note:
                diagnostics.append(note)
            ranks = text_first_relevant_ranks(embeddings, groups, case.tie_break_seed)
            text_score = mrr(ranks)
        else:
            diagnostics.append(
                "text component scored 0: {}".format(output.get("error", "no output"))
            )

    retrieval = {
        "mrr": 0.0,
        "recall_at_1": 0.0,
        "recall_at_5": 0.0,
        "recall_at_10": 0.0,
        "median_rank": 0.0,
    }
    retrieval_chance = 0.0
    if "retrieval" in by_kind:
        case, output = by_kind["retrieval"]
        if case.gold_rows is None:
            raise ValueError("Retrieval case has no gold rows; scoring requires the controller copy.")
        pool_size = case.descriptors.shape[0]
        retrieval_chance = chance_mrr(pool_size)
        retrieval["median_rank"] = float(pool_size)
        if output.get("ok"):
            text_matrix = np.asarray(output["text"], dtype=np.float64)
            image_matrix = np.asarray(output["images"], dtype=np.float64)
            for name, matrix in (("query embeddings", text_matrix), ("image embeddings", image_matrix)):
                note = summarize_norms(matrix, name)
                if note:
                    diagnostics.append(note)
            scores = text_matrix @ image_matrix.T
            order = rank_matrix(scores, case.tie_break_seed)
            ranks = ranks_of_gold(order, case.gold_rows)
            retrieval = {
                "mrr": mrr(ranks),
                "recall_at_1": recall_at(ranks, 1),
                "recall_at_5": recall_at(ranks, 5),
                "recall_at_10": recall_at(ranks, 10),
                "median_rank": median_rank(ranks),
            }
        else:
            diagnostics.append(
                "retrieval component scored 0: {}".format(output.get("error", "no output"))
            )

    # Each rewrite, reported next to the components and deliberately outside
    # `overall`. Adding a rung must not move a number a team already published
    # against, and a rung is diagnostic anyway: it says which part of the
    # capstone is load-bearing, not how good the submission is.
    rung_scores: Dict[str, float] = {}
    for case, output in rung_cases:
        rung = getattr(case, "rung", "?")
        if not output.get("ok"):
            diagnostics.append(
                "the {} query rung scored 0: {}".format(rung, output.get("error", "no output"))
            )
            rung_scores[rung] = 0.0
            continue
        ranks, _ = search_ranks(
            output.get("rankings", []), case.gold_image_ids, k_search, case.image_ids
        )
        rung_scores[rung] = mrr_with_misses(ranks)

    search_score = 0.0
    if "search" in by_kind:
        case, output = by_kind["search"]
        if case.gold_image_ids is None:
            raise ValueError("Search case has no gold ids; scoring requires the controller copy.")
        if output.get("ok"):
            rankings = output.get("rankings", [])
            ranks, foreign = search_ranks(
                rankings, case.gold_image_ids, k_search, case.image_ids
            )
            search_score = mrr_with_misses(ranks)
            if foreign:
                diagnostics.append(
                    "search returned {} ids outside the pinned pool (they cannot match).".format(
                        foreign
                    )
                )
        else:
            diagnostics.append(
                "search component scored 0: {}".format(output.get("error", "no output"))
            )

    overall = (text_score + retrieval["mrr"] + search_score) / 3.0
    metrics = {
        "overall": overall,
        "text_mrr": text_score,
        "retrieval_mrr": retrieval["mrr"],
        "search_mrr": search_score,
        "retrieval_recall_at_1": retrieval["recall_at_1"],
        "retrieval_recall_at_5": retrieval["recall_at_5"],
        "retrieval_recall_at_10": retrieval["recall_at_10"],
        "retrieval_median_rank": retrieval["median_rank"],
        "chance_mrr": retrieval_chance,
        # The floor for `text_mrr`, which is a different floor from
        # `chance_mrr`: one ranks captions among captions, the other images
        # among images, and on the evaluation tier they differ by about 4x.
        # This was computed and then dropped into a diagnostic string, so a
        # reader comparing `text_mrr` against the one published floor was
        # comparing against the wrong number.
        "text_chance": text_chance,
    }
    for rung, value in sorted(rung_scores.items()):
        metrics["search_mrr_{}".format(rung)] = value

    # What the rewrites revealed, in one sentence, only when they revealed
    # something. A submission that holds across every rung gets no note.
    if rung_scores and search_score:
        keywords = rung_scores.get("keywords")
        if keywords is not None and keywords < search_score * 0.75:
            diagnostics.append(
                "Dropping stopwords from the queries cost {:.0%} of the search score. "
                "IDF weighting is what should make those words nearly free, so this "
                "points at the weighting rather than at the embedding.".format(
                    1 - keywords / search_score
                )
            )
        typo = rung_scores.get("typo")
        if typo is not None and typo < search_score * 0.5:
            diagnostics.append(
                "One mistyped character per query cost {:.0%} of the search score. "
                "An unseen word should contribute a zero vector rather than "
                "dominating or raising.".format(1 - typo / search_score)
            )
    if text_chance:
        diagnostics.append(
            "chance baselines: text_mrr {:.4f}, retrieval/search mrr {:.4f}.".format(
                text_chance, retrieval_chance
            )
        )
    return metrics, diagnostics
