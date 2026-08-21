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

from . import perturb


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


def chance_mrr_at_k(pool_size: int, k: int) -> float:
    """Expected MRR when only the top ``k`` of a random ranking are returned.

    The search component asks for ``k`` ids and scores gold beyond that depth
    as a miss, so its floor is not the whole-pool floor. Gold lands at rank
    ``r`` with probability ``1/pool_size`` for each ``r``, and only the first
    ``k`` of those contribute, which gives ``H_k / pool_size`` where ``H_k``
    is the k-th harmonic number.

    Checked against a 400,000-trial simulation at pool 700, k 50: simulated
    0.006331, exact 0.006427. The gap is simulation noise at that trial
    count, not a disagreement about the formula.
    """

    if pool_size <= 0 or k <= 0:
        return 0.0
    depth = min(int(k), int(pool_size))
    return float(np.sum(1.0 / np.arange(1, depth + 1)) / pool_size)


def chance_mrr(pool_size: int) -> float:
    """Expected MRR of a uniformly random ranking over ``pool_size`` items.

    The whole-pool case of ``chance_mrr_at_k``. Written as a delegation so
    the two published floors cannot drift apart; with ``k == pool_size`` the
    two functions evaluate the identical numpy expression, and the returned
    floats were confirmed equal under ``==`` at pool sizes 3, 10, 100, and
    700, so this did not move the number any existing run reported.
    """

    return chance_mrr_at_k(pool_size, pool_size)


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
    """All metrics plus diagnostics from the component outputs.

    ``outputs[i]`` is the driver's dict for ``cases[i]``; a failed component
    carries ``{"ok": False, "error": ...}`` and scores zero with a
    diagnostic, leaving the other components intact.

    ``search_mrr`` is the mean over the whole rewrite grid, so several of the
    cases feed one component score; ``k_search`` is the depth the search
    component was asked for, and gold below it counts as a miss.
    """

    diagnostics: List[str] = []
    # Keyed by kind, and for search by kind AND rung. Several search cases
    # share one kind, one per query rewrite, and a plain by-kind dict would
    # keep whichever came last, so the reported verbatim rung would silently
    # become the typo rung.
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

    # Every rung, scored the same way, verbatim included. `search_mrr` is the
    # mean across all four.
    #
    # Until scorer version retrieval-v3 this loop only ran on the rewritten
    # rungs and `search_mrr` was the verbatim rung alone. That made
    # `search_mrr` a second reading of `retrieval_mrr`: both take the same
    # query list over the same pool, so on a correct submission the only way
    # they can differ is a query whose gold sits past the k-th result.
    # Measured on the reference: on the test tier (pool 100, k 50) they were
    # equal under `==`, and on the evaluation tier (pool 700, k 50) they
    # differed by 0.00137, every bit of which came from 15 of 150 queries
    # whose gold fell beyond rank 50. A student reading meaning into that gap
    # was reading the payload depth cap.
    #
    # Averaging the grid gives the two metrics different questions to answer.
    # `retrieval_mrr` asks whether the embedding space is aligned. `search_mrr`
    # asks whether end-to-end search survives the queries a person actually
    # types, which is the question the application is for. Both rest on
    # predictions the course material makes (IDF weighting should make
    # stopwords nearly free; an unseen word should contribute a zero vector),
    # so a submission that follows the course scores well on all four.
    #
    # Equal weight, not a weighting that favors verbatim. A weighting would be
    # a claim about how much each rewrite matters, and we have no measurement
    # supporting any particular set of weights.
    scored_rungs: List[Tuple] = []
    if "search" in by_kind:
        scored_rungs.append(by_kind["search"])
    scored_rungs.extend(rung_cases)

    rung_scores: Dict[str, float] = {}
    for case, output in scored_rungs:
        rung = getattr(case, "rung", "verbatim")
        if case.gold_image_ids is None:
            raise ValueError("Search case has no gold ids; scoring requires the controller copy.")
        if not output.get("ok"):
            diagnostics.append(
                "the {} query rung scored 0: {}".format(rung, output.get("error", "no output"))
            )
            rung_scores[rung] = 0.0
            continue
        ranks, foreign = search_ranks(
            output.get("rankings", []), case.gold_image_ids, k_search, case.image_ids
        )
        rung_scores[rung] = mrr_with_misses(ranks)
        if foreign:
            diagnostics.append(
                "the {} query rung returned {} ids outside the pinned pool "
                "(they cannot match).".format(rung, foreign)
            )

    # An incomplete grid refuses rather than averaging over what happened to
    # arrive. Dividing the rungs present by four would under-report, and
    # dividing by however many showed up would make `search_mrr` mean a
    # different thing per run with nothing on the page to say so. Both are
    # plausible wrong numbers, and this is a construction error rather than
    # anything a submission can cause: `materialize_cases` and the sandbox's
    # `decode_payload` both build the grid from this same `RUNGS` tuple, so
    # they can only disagree with it if one side was edited alone.
    #
    # A rung that ran and failed is not missing. It is in `rung_scores` at
    # 0.0 with a diagnostic, and it pulls the mean down as it should.
    search_score = 0.0
    search_chance = 0.0
    if rung_scores:
        present = set(rung_scores)
        expected = set(perturb.RUNGS)
        if present != expected:
            raise ValueError(
                "The search grid is incomplete: expected the rungs {} but the "
                "cases carry {}. Scoring an incomplete grid would report a "
                "search MRR that is not the average it claims to be.".format(
                    sorted(expected), sorted(present)
                )
            )
        search_score = float(
            sum(rung_scores[rung] for rung in perturb.RUNGS) / len(perturb.RUNGS)
        )
        # The pool is one shared object across the grid, so any rung gives the
        # same size; max() is here so a hand-built case list with a ragged
        # pool reports the more conservative (lower) floor rather than one
        # that flatters the score.
        pool_sizes = [
            len(case.image_ids) for case, _ in scored_rungs if case.image_ids is not None
        ]
        if pool_sizes:
            search_chance = chance_mrr_at_k(max(pool_sizes), k_search)

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
        # And a third floor, for `search_mrr`. Search returns only k ids, so
        # gold beyond rank k scores 0 and its floor sits below the whole-pool
        # floor: 0.006427 against 0.010184 at the evaluation pool of 700 with
        # k 50, a factor of 1.58. Reading search against `chance_mrr`
        # understates how far above chance it is.
        "search_chance": search_chance,
    }
    for rung, value in sorted(rung_scores.items()):
        metrics["search_mrr_{}".format(rung)] = value

    # What the rewrites revealed, in one sentence, only when they revealed
    # something. A submission that holds across every rung gets no note.
    #
    # Each rung is compared against the verbatim rung, not against the
    # averaged `search_mrr`. Verbatim is the control the rewrite is a
    # departure from, and it is also the only comparison that stays stable:
    # `search_mrr` now contains the rung being tested, so a large drop would
    # pull the thing it is measured against down with it and understate
    # itself.
    verbatim = rung_scores.get("verbatim", 0.0)
    if verbatim:
        keywords = rung_scores.get("keywords")
        if keywords is not None and keywords < verbatim * 0.75:
            diagnostics.append(
                "Dropping stopwords from the queries cost {:.0%} against the "
                "unchanged captions. IDF weighting is what should make those words "
                "nearly free, so this points at the weighting rather than at the "
                "embedding.".format(1 - keywords / verbatim)
            )
        typo = rung_scores.get("typo")
        if typo is not None and typo < verbatim * 0.5:
            diagnostics.append(
                "One mistyped character per query cost {:.0%} against the unchanged "
                "captions. An unseen word should contribute a zero vector rather "
                "than dominating or raising.".format(1 - typo / verbatim)
            )
    if text_chance or search_chance:
        # Three floors, named separately. They are not interchangeable: on the
        # evaluation tier text_chance is about 4x chance_mrr and search_chance
        # is about 0.63x it, so a reader who picks the wrong one is off by
        # more than the differences between submissions.
        diagnostics.append(
            "chance baselines: text_mrr {:.4f}, retrieval_mrr {:.4f}, "
            "search_mrr {:.4f}.".format(text_chance, retrieval_chance, search_chance)
        )
    return metrics, diagnostics
