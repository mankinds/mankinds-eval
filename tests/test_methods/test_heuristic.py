"""Tests for heuristic evaluation methods."""

from __future__ import annotations

import re

import pytest

from mankinds_eval.core import Sample
from mankinds_eval.methods.heuristic.fuzzy import FuzzyMatch
from mankinds_eval.methods.heuristic.pattern import (
    ContainsAll,
    ContainsAny,
    ExactMatch,
    RegexMatch,
)
from mankinds_eval.methods.heuristic.text_metrics import ROUGE, TextLength

# =============================================================================
# ExactMatch Tests
# =============================================================================


class TestExactMatch:
    """Tests for the ExactMatch method."""

    @pytest.mark.asyncio
    async def test_exact_match_pass(self) -> None:
        """Test exact match returns score 1.0 when strings match."""
        method = ExactMatch()
        sample = Sample(
            input="What is 2+2?",
            output="4",
            expected="4",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.method_name == "ExactMatch"

    @pytest.mark.asyncio
    async def test_exact_match_fail(self) -> None:
        """Test exact match returns score 0.0 when strings differ."""
        method = ExactMatch()
        sample = Sample(
            input="What is 2+2?",
            output="5",
            expected="4",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_exact_match_case_sensitive_by_default(self) -> None:
        """Test that exact match is case sensitive by default."""
        method = ExactMatch(case_sensitive=True)
        sample = Sample(
            input="test",
            output="Hello",
            expected="hello",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_exact_match_case_insensitive(self) -> None:
        """Test case insensitive matching."""
        method = ExactMatch(case_sensitive=False)
        sample = Sample(
            input="test",
            output="Hello",
            expected="hello",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_exact_match_strip_whitespace(self) -> None:
        """Test whitespace stripping."""
        method = ExactMatch(strip_whitespace=True)
        sample = Sample(
            input="test",
            output="  hello  ",
            expected="hello",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_exact_match_no_strip_whitespace(self) -> None:
        """Test without whitespace stripping."""
        method = ExactMatch(strip_whitespace=False)
        sample = Sample(
            input="test",
            output="  hello  ",
            expected="hello",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False


# =============================================================================
# FuzzyMatch Tests
# =============================================================================


class TestFuzzyMatch:
    """Tests for the FuzzyMatch method."""

    @pytest.mark.asyncio
    async def test_fuzzy_match_ratio_perfect(self) -> None:
        """Test fuzzy ratio with perfect match."""
        method = FuzzyMatch(algorithm="ratio", threshold=0.8)
        sample = Sample(
            input="test",
            output="hello world",
            expected="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["algorithm"] == "ratio"

    @pytest.mark.asyncio
    async def test_fuzzy_match_ratio_partial(self) -> None:
        """Test fuzzy ratio with partial match."""
        method = FuzzyMatch(algorithm="ratio", threshold=0.5)
        sample = Sample(
            input="test",
            output="hello world",
            expected="hello there",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert 0.0 < result.score < 1.0
        assert result.passed is True  # Above 0.5 threshold

    @pytest.mark.asyncio
    async def test_fuzzy_match_token_set_ratio(self) -> None:
        """Test token_set_ratio algorithm handles word order."""
        method = FuzzyMatch(algorithm="token_set_ratio", threshold=0.9)
        sample = Sample(
            input="test",
            output="world hello",
            expected="hello world",
        )
        result = await method.evaluate(sample)

        # token_set_ratio should give high score despite word order
        assert result.score is not None
        assert result.score >= 0.9
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fuzzy_match_levenshtein(self) -> None:
        """Test Levenshtein distance algorithm."""
        method = FuzzyMatch(algorithm="levenshtein", threshold=0.8)
        sample = Sample(
            input="test",
            output="hello",
            expected="hallo",  # One character difference
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert result.score > 0.7
        assert result.metadata is not None
        assert result.metadata["algorithm"] == "levenshtein"

    @pytest.mark.asyncio
    async def test_fuzzy_match_jaro_winkler(self) -> None:
        """Test Jaro-Winkler algorithm."""
        method = FuzzyMatch(algorithm="jaro_winkler", threshold=0.8)
        sample = Sample(
            input="test",
            output="hello",
            expected="hello",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fuzzy_match_threshold_fail(self) -> None:
        """Test fuzzy match fails when below threshold."""
        method = FuzzyMatch(algorithm="ratio", threshold=0.9)
        sample = Sample(
            input="test",
            output="completely different",
            expected="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert result.score < 0.9
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_fuzzy_match_case_sensitive(self) -> None:
        """Test case sensitive fuzzy matching."""
        method = FuzzyMatch(algorithm="ratio", threshold=0.99, case_sensitive=True)
        sample = Sample(
            input="test",
            output="HELLO",
            expected="hello",
        )
        result = await method.evaluate(sample)

        # Different case should result in lower score
        assert result.score is not None
        assert result.score < 1.0

    @pytest.mark.asyncio
    async def test_fuzzy_match_case_insensitive(self) -> None:
        """Test case insensitive fuzzy matching (default)."""
        method = FuzzyMatch(algorithm="ratio", threshold=0.99, case_sensitive=False)
        sample = Sample(
            input="test",
            output="HELLO",
            expected="hello",
        )
        result = await method.evaluate(sample)

        # Case insensitive should give perfect match
        assert result.score == 1.0
        assert result.passed is True

    def test_fuzzy_match_invalid_algorithm(self) -> None:
        """Test that invalid algorithm raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FuzzyMatch(algorithm="invalid_algo")
        assert "Invalid algorithm" in str(exc_info.value)

    def test_fuzzy_match_invalid_target(self) -> None:
        """Test that invalid target raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FuzzyMatch(target="invalid_target")
        assert "Invalid target" in str(exc_info.value)

    def test_fuzzy_match_invalid_threshold(self) -> None:
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            FuzzyMatch(threshold=1.5)
        assert "threshold must be between 0.0 and 1.0" in str(exc_info.value)

        with pytest.raises(ValueError):
            FuzzyMatch(threshold=-0.1)

    @pytest.mark.asyncio
    async def test_fuzzy_match_target_input(self) -> None:
        """Test fuzzy matching against input field."""
        method = FuzzyMatch(algorithm="ratio", threshold=0.8, target="input")
        sample = Sample(
            input="hello world",
            output="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_fuzzy_match_target_context(self) -> None:
        """Test fuzzy matching against context field."""
        method = FuzzyMatch(algorithm="ratio", threshold=0.8, target="context")
        sample = Sample(
            input="test",
            output="This is the context content",
            context="This is the context content",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True


# =============================================================================
# TextLength Tests
# =============================================================================


class TestTextLength:
    """Tests for the TextLength method."""

    @pytest.mark.asyncio
    async def test_text_length_within_bounds_chars(self) -> None:
        """Test text length within character bounds passes."""
        method = TextLength(min_length=5, max_length=20, unit="chars")
        sample = Sample(
            input="test",
            output="hello world",  # 11 characters
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["length"] == 11
        assert result.metadata["unit"] == "chars"

    @pytest.mark.asyncio
    async def test_text_length_below_minimum(self) -> None:
        """Test text length below minimum fails."""
        method = TextLength(min_length=20, unit="chars")
        sample = Sample(
            input="test",
            output="short",  # 5 characters
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False
        assert "below minimum" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_text_length_above_maximum(self) -> None:
        """Test text length above maximum fails."""
        method = TextLength(max_length=5, unit="chars")
        sample = Sample(
            input="test",
            output="this is too long",  # 16 characters
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False
        assert "exceeds maximum" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_text_length_word_unit(self) -> None:
        """Test text length with word unit."""
        method = TextLength(min_length=2, max_length=5, unit="words")
        sample = Sample(
            input="test",
            output="hello world there",  # 3 words
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["length"] == 3
        assert result.metadata["unit"] == "words"

    @pytest.mark.asyncio
    async def test_text_length_no_minimum(self) -> None:
        """Test text length with no minimum constraint."""
        method = TextLength(max_length=100)
        sample = Sample(
            input="test",
            output="hi",  # Very short
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_text_length_no_maximum(self) -> None:
        """Test text length with no maximum constraint."""
        method = TextLength(min_length=5)
        sample = Sample(
            input="test",
            output="this is a very long output text that should still pass",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    def test_text_length_invalid_unit(self) -> None:
        """Test that invalid unit raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            TextLength(unit="invalid")
        assert "chars" in str(exc_info.value) or "words" in str(exc_info.value)

    def test_text_length_invalid_bounds(self) -> None:
        """Test that min > max raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            TextLength(min_length=100, max_length=50)
        assert "greater than" in str(exc_info.value)


# =============================================================================
# ROUGE Tests
# =============================================================================


class TestROUGE:
    """Tests for the ROUGE method."""

    @pytest.mark.asyncio
    async def test_rouge_perfect_match(self) -> None:
        """Test ROUGE score for identical texts."""
        method = ROUGE(threshold=0.9)
        sample = Sample(
            input="summarize",
            output="The quick brown fox jumps over the lazy dog.",
            expected="The quick brown fox jumps over the lazy dog.",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_rouge_partial_match(self) -> None:
        """Test ROUGE score for partial match."""
        method = ROUGE(threshold=0.3)
        sample = Sample(
            input="summarize",
            output="The quick brown fox",
            expected="The quick brown fox jumps over the lazy dog.",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert 0.0 < result.score < 1.0
        assert result.passed is True  # Above 0.3 threshold

    @pytest.mark.asyncio
    async def test_rouge_no_match(self) -> None:
        """Test ROUGE score for completely different texts."""
        method = ROUGE(threshold=0.5)
        sample = Sample(
            input="summarize",
            output="Completely different sentence here.",
            expected="The quick brown fox jumps over the lazy dog.",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        # Should be low score
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_rouge_different_types(self) -> None:
        """Test ROUGE with different rouge_types."""
        method = ROUGE(rouge_types=["rouge1", "rouge2", "rougeL"])
        sample = Sample(
            input="summarize",
            output="The quick brown fox",
            expected="The quick brown fox jumps",
        )
        result = await method.evaluate(sample)

        assert result.metadata is not None
        assert "rouge1" in result.metadata
        assert "rouge2" in result.metadata
        assert "rougeL" in result.metadata

    @pytest.mark.asyncio
    async def test_rouge_aggregate_metric(self) -> None:
        """Test ROUGE aggregate using specified rouge type."""
        method = ROUGE(aggregate="rouge1", metric="precision")
        sample = Sample(
            input="test",
            output="hello world",
            expected="hello world test",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert result.metadata is not None
        # The aggregate should be rouge1 precision

    @pytest.mark.asyncio
    async def test_rouge_no_threshold(self) -> None:
        """Test ROUGE without threshold (passed is None)."""
        method = ROUGE(threshold=None)
        sample = Sample(
            input="test",
            output="some output",
            expected="some expected",
        )
        result = await method.evaluate(sample)

        assert result.score is not None
        assert result.passed is None  # No threshold means no pass/fail

    def test_rouge_invalid_aggregate(self) -> None:
        """Test that invalid aggregate raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ROUGE(aggregate="rouge3", rouge_types=["rouge1", "rouge2"])
        assert "aggregate" in str(exc_info.value)

    def test_rouge_invalid_metric(self) -> None:
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ROUGE(metric="invalid")
        assert "metric" in str(exc_info.value)


# =============================================================================
# RegexMatch (PatternMatch) Tests
# =============================================================================


class TestRegexMatch:
    """Tests for the RegexMatch method."""

    @pytest.mark.asyncio
    async def test_regex_match_simple_pattern(self) -> None:
        """Test simple regex pattern matching."""
        method = RegexMatch(pattern=r"\d+")
        sample = Sample(
            input="test",
            output="The answer is 42",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert "42" in result.metadata["matches"]

    @pytest.mark.asyncio
    async def test_regex_match_no_match(self) -> None:
        """Test regex returns score 0.0 when no match found."""
        method = RegexMatch(pattern=r"\d+")
        sample = Sample(
            input="test",
            output="no numbers here",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["matches"] == []

    @pytest.mark.asyncio
    async def test_regex_match_full_match(self) -> None:
        """Test regex with full_match=True."""
        method = RegexMatch(pattern=r"^\d+$", full_match=True)
        sample = Sample(
            input="test",
            output="12345",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_regex_match_full_match_fail(self) -> None:
        """Test regex full_match fails when not entire string matches."""
        method = RegexMatch(pattern=r"\d+", full_match=True)
        sample = Sample(
            input="test",
            output="abc 123",  # Has digits but not entire string
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_regex_match_with_flags(self) -> None:
        """Test regex with flags (case insensitive)."""
        method = RegexMatch(pattern=r"hello", flags=re.IGNORECASE)
        sample = Sample(
            input="test",
            output="HELLO WORLD",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_regex_match_groups(self) -> None:
        """Test regex captures groups in metadata."""
        method = RegexMatch(pattern=r"(\w+)@(\w+)\.(\w+)")
        sample = Sample(
            input="test",
            output="Contact: user@example.com",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.metadata is not None
        assert len(result.metadata["matches"]) > 1  # Full match + groups


# =============================================================================
# ContainsAll Tests
# =============================================================================


class TestContainsAll:
    """Tests for the ContainsAll method."""

    @pytest.mark.asyncio
    async def test_contains_all_pass(self) -> None:
        """Test ContainsAll passes when all keywords present."""
        method = ContainsAll(keywords=["hello", "world"])
        sample = Sample(
            input="test",
            output="hello beautiful world",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert len(result.metadata["missing"]) == 0

    @pytest.mark.asyncio
    async def test_contains_all_partial(self) -> None:
        """Test ContainsAll with partial match."""
        method = ContainsAll(keywords=["hello", "world", "foo"])
        sample = Sample(
            input="test",
            output="hello world",
        )
        result = await method.evaluate(sample)

        # 2 out of 3 keywords found
        assert result.score is not None
        assert abs(result.score - 2 / 3) < 0.01
        assert result.passed is False
        assert result.metadata is not None
        assert "foo" in result.metadata["missing"]

    @pytest.mark.asyncio
    async def test_contains_all_none_found(self) -> None:
        """Test ContainsAll when no keywords found."""
        method = ContainsAll(keywords=["foo", "bar"])
        sample = Sample(
            input="test",
            output="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_contains_all_case_insensitive(self) -> None:
        """Test ContainsAll is case insensitive by default."""
        method = ContainsAll(keywords=["HELLO", "WORLD"], case_sensitive=False)
        sample = Sample(
            input="test",
            output="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_contains_all_case_sensitive(self) -> None:
        """Test ContainsAll with case sensitive matching."""
        method = ContainsAll(keywords=["HELLO"], case_sensitive=True)
        sample = Sample(
            input="test",
            output="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False


# =============================================================================
# ContainsAny Tests
# =============================================================================


class TestContainsAny:
    """Tests for the ContainsAny method."""

    @pytest.mark.asyncio
    async def test_contains_any_pass(self) -> None:
        """Test ContainsAny passes when any keyword present."""
        method = ContainsAny(keywords=["foo", "hello", "bar"])
        sample = Sample(
            input="test",
            output="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert "hello" in result.metadata["found"]

    @pytest.mark.asyncio
    async def test_contains_any_fail(self) -> None:
        """Test ContainsAny fails when no keyword present."""
        method = ContainsAny(keywords=["foo", "bar", "baz"])
        sample = Sample(
            input="test",
            output="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_contains_any_multiple_found(self) -> None:
        """Test ContainsAny when multiple keywords found."""
        method = ContainsAny(keywords=["hello", "world", "foo"])
        sample = Sample(
            input="test",
            output="hello world",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert "hello" in result.metadata["found"]
        assert "world" in result.metadata["found"]
