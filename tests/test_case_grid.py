"""The case grid is built in one place, and these pin what that place owes.

Two callers depend on `build_cases`: `materialize_cases` here, and the
platform's sandbox decoder, which cannot resolve a manifest because gold is
derivable from the captions file it is handed. They used to assemble the grid
separately and drifted, so the sandbox ran six cases against a nine-case tier.
These tests are the reason that cannot recur quietly.
"""

from __future__ import annotations

import json
import pickle
import unittest
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from language_search_benchmark import perturb
from language_search_benchmark.contracts import Resources
from language_search_benchmark.datasets import (
    SEARCH_K,
    attach_gold,
    build_cases,
    materialize_cases,
)

QUERIES = ["a dog runs fast", "two cats sit still", "a red bus downtown"]
CAPTIONS = ["a dog runs fast", "another dog runs", "two cats sit still"]


def grid(**overrides):
    arguments = dict(
        text_captions=CAPTIONS,
        queries=QUERIES,
        pool_image_ids=[10, 11, 12],
        pool_descriptors=np.arange(12, dtype=np.float32).reshape(3, 4),
        tie_break_seed=7,
        search_k=2,
    )
    arguments.update(overrides)
    return build_cases(**arguments)


class TheGridHasOneShape(unittest.TestCase):
    def test_the_order_is_the_one_scoring_walks(self):
        rewrites = [rung for rung in perturb.RUNGS if rung != "verbatim"]
        expected = (
            [("text", "verbatim"), ("retrieval", "verbatim"), ("search", "verbatim")]
            + [("retrieval", rung) for rung in rewrites]
            + [("search", rung) for rung in rewrites]
        )
        observed = [(case.kind, getattr(case, "rung", "verbatim")) for case in grid()]
        self.assertEqual(observed, expected)

    def test_the_verbatim_rung_is_the_queries_unchanged(self):
        """Both callers used to spell this differently, one passing the query
        list and one passing `rewrite_all(queries, "verbatim")`."""

        cases = grid()
        for case in cases:
            if getattr(case, "rung", "verbatim") == "verbatim" and case.kind != "text":
                self.assertEqual(case.queries, QUERIES, case.kind)

    def test_nothing_built_here_carries_gold(self):
        for case in grid():
            self.assertIsNone(getattr(case, "group_rows", None), case.kind)
            self.assertIsNone(getattr(case, "gold_rows", None), case.kind)
            self.assertIsNone(getattr(case, "gold_image_ids", None), case.kind)


class TheSharingPolicyIsPartOfTheGrid(unittest.TestCase):
    def test_every_search_case_holds_the_same_pool_objects(self):
        """`drivers.run_with_adapter` compares these by identity to decide
        whether to rebuild a submission's index."""

        searches = [case for case in grid() if case.kind == "search"]
        self.assertEqual(len(searches), len(perturb.RUNGS))
        for case in searches:
            self.assertIs(case.image_ids, searches[0].image_ids, case.rung)
            self.assertIs(case.descriptors, searches[0].descriptors, case.rung)

    def test_every_retrieval_case_holds_the_same_matrix(self):
        retrievals = [case for case in grid() if case.kind == "retrieval"]
        self.assertEqual(len(retrievals), len(perturb.RUNGS))
        for case in retrievals:
            self.assertIs(case.descriptors, retrievals[0].descriptors, case.rung)

    def test_search_works_on_a_copy_so_a_mutating_prepare_cannot_reach_retrieval(self):
        cases = grid()
        search = next(case for case in cases if case.kind == "search")
        retrieval = next(case for case in cases if case.kind == "retrieval")
        self.assertIsNot(search.descriptors, retrieval.descriptors)
        np.testing.assert_array_equal(search.descriptors, retrieval.descriptors)

        before = retrieval.descriptors.copy()
        search.descriptors += 100.0  # what an in-place prepare_database does
        np.testing.assert_array_equal(retrieval.descriptors, before)

    def test_a_submission_prepares_its_index_once(self):
        """The identity rules above exist for this, so drive it rather than
        assert the objects and hope."""

        seen = []
        previous = {}
        for case in grid():
            if case.kind != "search":
                continue
            same = (
                previous.get("ids") is case.image_ids
                and previous.get("pool") is case.descriptors
            )
            if not same:
                seen.append(case.rung)
            previous = {"ids": case.image_ids, "pool": case.descriptors}
        self.assertEqual(seen, ["verbatim"])


class GoldIsAttachedSeparately(unittest.TestCase):
    def test_attaching_gold_keeps_the_order_and_the_shared_objects(self):
        cases = grid()
        restored = attach_gold(
            cases,
            text_group_rows=[0, 0, 1],
            retrieval_gold_rows=[0, 1, 2],
            search_gold_image_ids=[10, 11, 12],
        )
        self.assertEqual(
            [(case.kind, getattr(case, "rung", "verbatim")) for case in restored],
            [(case.kind, getattr(case, "rung", "verbatim")) for case in cases],
        )
        searches = [case for case in restored if case.kind == "search"]
        for case in searches:
            self.assertIs(case.image_ids, searches[0].image_ids, case.rung)
            self.assertEqual(case.gold_image_ids, [10, 11, 12], case.rung)
        for case in restored:
            if case.kind == "retrieval":
                self.assertEqual(case.gold_rows, [0, 1, 2], case.rung)

    def test_gold_that_does_not_fit_the_cases_is_refused(self):
        with self.assertRaises(ValueError):
            attach_gold(
                grid(),
                text_group_rows=[0],
                retrieval_gold_rows=[0, 1, 2],
                search_gold_image_ids=[10, 11, 12],
            )


class MaterializeCasesIsTheConstructorPlusGold(unittest.TestCase):
    """The proof that moving the grid into `build_cases` moved no number."""

    def test_a_manifest_resolves_to_the_constructor_output_field_for_field(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            captions = {
                "images": [{"id": 100 + n} for n in range(3)],
                "annotations": [
                    {"id": 200 + n, "image_id": 100 + n, "caption": QUERIES[n]}
                    for n in range(3)
                ]
                + [{"id": 300, "image_id": 100, "caption": "another dog runs"}],
            }
            (root / "captions.json").write_text(json.dumps(captions), encoding="utf-8")
            descriptors = {100 + n: np.full(512, float(n), dtype=np.float32) for n in range(3)}
            with open(root / "descriptors.pkl", "wb") as stream:
                pickle.dump(descriptors, stream)
            resources = Resources(
                captions_path=root / "captions.json",
                descriptors_path=root / "descriptors.pkl",
                glove_path=root / "missing.w2v",
                glove_kv_path=None,
            )
            manifest = {
                "tie_break_seed": 7,
                "text": {"caption_ids": [200, 300, 201]},
                "queries": {
                    "query_caption_ids": [200, 201, 202],
                    "pool_image_ids": [100, 101, 102],
                },
            }
            observed = materialize_cases(manifest, resources)

        maps_captions = [captions["annotations"][i]["caption"] for i in (0, 3, 1)]
        matrix = np.vstack([descriptors[i] for i in (100, 101, 102)]).astype(np.float32)
        expected = attach_gold(
            build_cases(
                text_captions=maps_captions,
                queries=[QUERIES[0], QUERIES[1], QUERIES[2]],
                pool_image_ids=[100, 101, 102],
                pool_descriptors=matrix,
                tie_break_seed=7,
                search_k=SEARCH_K,
            ),
            text_group_rows=[0, 0, 1],
            retrieval_gold_rows=[0, 1, 2],
            search_gold_image_ids=[100, 101, 102],
        )

        self.assertEqual(len(observed), len(expected))
        for left, right in zip(observed, expected):
            self.assertEqual(type(left), type(right))
            for field in fields(left):
                a = getattr(left, field.name)
                b = getattr(right, field.name)
                if isinstance(a, np.ndarray):
                    np.testing.assert_array_equal(a, b, field.name)
                else:
                    self.assertEqual(a, b, "{}.{}".format(left.kind, field.name))


if __name__ == "__main__":
    unittest.main()
