"""The query grid: deterministic, and each rung measures what it claims."""

from __future__ import annotations

import pytest

from language_search_benchmark import perturb


CAPTION = "A man riding a horse on the beach near the water"


class TestDeterminism:
    def test_the_same_caption_and_index_always_rewrite_the_same(self):
        """Two runs of one submission must face the same queries, or a team
        comparing this week's run against last week's is comparing noise."""

        for rung in perturb.RUNGS:
            first = perturb.rewrite(CAPTION, rung, 7)
            for _ in range(5):
                assert perturb.rewrite(CAPTION, rung, 7) == first

    def test_the_index_moves_the_typo_without_a_random_source(self):
        seen = {perturb.rewrite(CAPTION, "typo", i) for i in range(6)}
        assert len(seen) > 1, "every query would get its typo in the same place"

    def test_rewriting_a_list_matches_rewriting_each_by_position(self):
        captions = ["A dog on a couch", "Two elephants in a field", "A red bus"]
        batch = perturb.rewrite_all(captions, "typo")
        one_by_one = [perturb.rewrite(c, "typo", i) for i, c in enumerate(captions)]
        assert batch == one_by_one


class TestRungsMeasureWhatTheyClaim:
    def test_keywords_drops_stopwords_and_keeps_order(self):
        assert perturb.keywords(CAPTION) == "man riding horse beach water"

    def test_keywords_keeps_something_when_every_word_is_a_stopword(self):
        # Real captions do this. An empty query would measure our tokenizer
        # rather than the submission.
        assert perturb.keywords("it is on the table") != ""

    def test_truncated_keeps_a_subject_and_something_about_it(self):
        assert perturb.truncated(CAPTION) == "man riding horse"

    def test_typo_changes_exactly_one_character_of_one_word(self):
        after = perturb.typo(CAPTION, 0)
        before_words, after_words = CAPTION.split(), after.split()
        assert len(before_words) == len(after_words)
        differing = [i for i, (a, b) in enumerate(zip(before_words, after_words)) if a != b]
        assert len(differing) == 1
        a, b = before_words[differing[0]], after_words[differing[0]]
        assert len(a) == len(b)
        assert sum(1 for x, y in zip(a, b) if x != y) == 1

    def test_typo_targets_a_content_word_not_a_stopword(self):
        """A typo in "the" changes nothing a submission depends on. The point
        is to produce a word its vocabulary has never seen."""

        after = perturb.typo(CAPTION, 0)
        changed = next(b for a, b in zip(CAPTION.split(), after.split()) if a != b)
        assert changed.lower() not in perturb.STOPWORDS

    def test_an_unknown_rung_raises_rather_than_passing_the_caption_through(self):
        # Otherwise a typo in a rung name reads as a suspiciously strong score.
        with pytest.raises(ValueError, match="Unknown rung"):
            perturb.rewrite(CAPTION, "keywrods", 0)


class TestRealCaptionShapes:
    @pytest.mark.parametrize(
        "caption",
        [
            "A cat",
            "Two",
            "a a a",
            "A man riding a horse.",
            "Sign: STOP!",
        ],
    )
    def test_no_caption_shape_raises_or_empties(self, caption):
        """Short, punctuated, and all-stopword captions all appear in COCO."""

        for rung in perturb.RUNGS:
            result = perturb.rewrite(caption, rung, 2)
            assert isinstance(result, str)
            assert result.strip(), "{} on {!r} produced an empty query".format(rung, caption)
