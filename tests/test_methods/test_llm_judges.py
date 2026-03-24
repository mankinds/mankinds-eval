"""Tests for LLM-as-Judge evaluation methods.

These tests mock the LLM provider to avoid actual API calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mankinds_eval.core import Sample

# Import the LLM judges
from mankinds_eval.methods.llm.coherence import Coherence
from mankinds_eval.methods.llm.correctness import Correctness
from mankinds_eval.methods.llm.faithfulness import Faithfulness
from mankinds_eval.methods.llm.helpfulness import Helpfulness
from mankinds_eval.methods.llm.relevancy import AnswerRelevancy

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider."""
    with patch("mankinds_eval.methods.llm.base.LLMProvider") as mock:
        provider_instance = MagicMock()
        provider_instance.provider = "openai"
        provider_instance.model = "gpt-4o-mini"
        mock.return_value = provider_instance
        yield provider_instance


# =============================================================================
# Faithfulness Tests
# =============================================================================


class TestFaithfulness:
    """Tests for the Faithfulness method."""

    @pytest.mark.asyncio
    async def test_faithful_response(self, mock_llm_provider) -> None:
        """Test evaluation of a faithful response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 5, "reason": "Response is fully'
            ' grounded in context", "unsupported_claims": []}'
        )

        method = Faithfulness(provider="openai")
        sample = Sample(
            input="What is Python?",
            output="Python is a programming language.",
            context="Python is a high-level programming language known for its simplicity.",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0  # 5/5 normalized
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["unsupported_claims"] == []

    @pytest.mark.asyncio
    async def test_unfaithful_response(self, mock_llm_provider) -> None:
        """Test evaluation of an unfaithful response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 2, "reason": "Contains'
            ' hallucinations", "unsupported_claims":'
            ' ["Python was created in 2020"]}'
        )

        method = Faithfulness(provider="openai", threshold=0.7)
        sample = Sample(
            input="When was Python created?",
            output="Python was created in 2020.",
            context="Python was created by Guido van Rossum in 1991.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.4  # 2/5 normalized
        assert result.passed is False
        assert result.metadata is not None
        assert len(result.metadata["unsupported_claims"]) > 0

    @pytest.mark.asyncio
    async def test_binary_scale(self, mock_llm_provider) -> None:
        """Test faithfulness with binary scale."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": "yes", "reason": "Faithful", "unsupported_claims": []}'
        )

        method = Faithfulness(provider="openai", scale="binary")
        sample = Sample(
            input="test",
            output="test output",
            context="test context",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    def test_required_fields(self, mock_llm_provider) -> None:
        """Test that context is required."""
        method = Faithfulness(provider="openai")
        assert "context" in method.required_fields


# =============================================================================
# AnswerRelevancy Tests
# =============================================================================


class TestAnswerRelevancy:
    """Tests for the AnswerRelevancy method."""

    @pytest.mark.asyncio
    async def test_relevant_response(self, mock_llm_provider) -> None:
        """Test evaluation of a relevant response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 5, "reason": "Directly answers'
            ' the question", "addresses_question": true,'
            ' "is_complete": true, "off_topic_elements": []}'
        )

        method = AnswerRelevancy(provider="openai")
        sample = Sample(
            input="What is 2+2?",
            output="2+2 equals 4.",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["addresses_question"] is True

    @pytest.mark.asyncio
    async def test_irrelevant_response(self, mock_llm_provider) -> None:
        """Test evaluation of an irrelevant response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 1, "reason": "Does not address'
            ' the question", "addresses_question": false,'
            ' "is_complete": false, "off_topic_elements":'
            ' ["weather discussion"]}'
        )

        method = AnswerRelevancy(provider="openai", threshold=0.7)
        sample = Sample(
            input="What is 2+2?",
            output="The weather is nice today.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.2  # 1/5 normalized
        assert result.passed is False
        assert result.metadata is not None
        assert len(result.metadata["off_topic_elements"]) > 0

    def test_check_completeness_config(self, mock_llm_provider) -> None:
        """Test check_completeness configuration."""
        method = AnswerRelevancy(provider="openai", check_completeness=False)
        assert method.check_completeness is False


# =============================================================================
# Coherence Tests
# =============================================================================


class TestCoherence:
    """Tests for the Coherence method."""

    @pytest.mark.asyncio
    async def test_coherent_response(self, mock_llm_provider) -> None:
        """Test evaluation of a coherent response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 5, "reason": "Well structured'
            ' and clear", "aspect_scores": {"logical_flow": 5,'
            ' "clarity": 5, "consistency": 5}, "issues": []}'
        )

        method = Coherence(provider="openai")
        sample = Sample(
            input="Explain Python",
            output=(
                "Python is a programming language."
                " It is known for simplicity."
                " Many developers use it."
            ),
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert "aspect_scores" in result.metadata

    @pytest.mark.asyncio
    async def test_incoherent_response(self, mock_llm_provider) -> None:
        """Test evaluation of an incoherent response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 2, "reason": "Contradictory'
            ' statements", "aspect_scores": {"logical_flow": 2,'
            ' "clarity": 3, "consistency": 1},'
            ' "issues": ["Self-contradiction"]}'
        )

        method = Coherence(provider="openai", threshold=0.7)
        sample = Sample(
            input="test",
            output="Python is easy. Python is very hard. Python is simple.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.4  # 2/5 normalized
        assert result.passed is False
        assert result.metadata is not None
        assert len(result.metadata["issues"]) > 0

    def test_custom_aspects(self, mock_llm_provider) -> None:
        """Test custom aspects configuration."""
        method = Coherence(
            provider="openai",
            aspects=["logical_flow", "conciseness"],
        )
        assert method.aspects == ["logical_flow", "conciseness"]


# =============================================================================
# Helpfulness Tests
# =============================================================================


class TestHelpfulness:
    """Tests for the Helpfulness method."""

    @pytest.mark.asyncio
    async def test_helpful_response(self, mock_llm_provider) -> None:
        """Test evaluation of a helpful response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 5, "reason": "Provides actionable'
            ' guidance", "is_actionable": true,'
            ' "is_informative": true, "practical_value": "high"}'
        )

        method = Helpfulness(provider="openai")
        sample = Sample(
            input="How do I install Python?",
            output="To install Python: 1. Go to python.org 2. Download installer 3. Run it.",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["is_actionable"] is True
        assert result.metadata["practical_value"] == 1.0

    @pytest.mark.asyncio
    async def test_unhelpful_response(self, mock_llm_provider) -> None:
        """Test evaluation of an unhelpful response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 1, "reason": "Does not provide'
            ' useful information", "is_actionable": false,'
            ' "is_informative": false, "practical_value": "low"}'
        )

        method = Helpfulness(provider="openai", threshold=0.7)
        sample = Sample(
            input="How do I install Python?",
            output="Python exists.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.2
        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["practical_value"] == 0.0


# =============================================================================
# Correctness Tests
# =============================================================================


class TestCorrectness:
    """Tests for the Correctness method."""

    @pytest.mark.asyncio
    async def test_correct_response(self, mock_llm_provider) -> None:
        """Test evaluation of a correct response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 5, "reason": "Matches expected'
            ' answer", "correct_elements": ["answer is 4"],'
            ' "incorrect_elements": [], "missing_elements": []}'
        )

        method = Correctness(provider="openai")
        sample = Sample(
            input="What is 2+2?",
            output="The answer is 4.",
            expected="4",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert len(result.metadata["correct_elements"]) > 0
        assert len(result.metadata["incorrect_elements"]) == 0

    @pytest.mark.asyncio
    async def test_incorrect_response(self, mock_llm_provider) -> None:
        """Test evaluation of an incorrect response."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 1, "reason": "Wrong answer",'
            ' "correct_elements": [],'
            ' "incorrect_elements": ["answer is 5"],'
            ' "missing_elements": ["correct answer 4"]}'
        )

        method = Correctness(provider="openai", threshold=0.7)
        sample = Sample(
            input="What is 2+2?",
            output="The answer is 5.",
            expected="4",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.2
        assert result.passed is False
        assert result.metadata is not None
        assert len(result.metadata["incorrect_elements"]) > 0

    def test_required_fields(self, mock_llm_provider) -> None:
        """Test that expected is required."""
        method = Correctness(provider="openai")
        assert "expected" in method.required_fields

    def test_partial_credit_config(self, mock_llm_provider) -> None:
        """Test partial_credit configuration."""
        method = Correctness(provider="openai", partial_credit=False)
        assert method.partial_credit is False


# =============================================================================
# Common Tests
# =============================================================================


class TestCommonBehavior:
    """Tests for common behavior across all LLM judges."""

    @pytest.mark.asyncio
    async def test_threshold_configuration(self, mock_llm_provider) -> None:
        """Test that threshold is properly used."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 3, "reason": "Average",'
            ' "addresses_question": true, "is_complete": true,'
            ' "off_topic_elements": []}'
        )

        method = AnswerRelevancy(provider="openai", threshold=0.5)
        sample = Sample(input="test", output="test output")
        result = await method.evaluate(sample)

        assert result.score == 0.6  # 3/5 normalized
        assert result.passed is True  # 0.6 >= 0.5

    @pytest.mark.asyncio
    async def test_scale_10(self, mock_llm_provider) -> None:
        """Test 1-10 scale normalization."""
        mock_llm_provider.complete = AsyncMock(
            return_value='{"score": 8, "reason": "Good",'
            ' "addresses_question": true, "is_complete": true,'
            ' "off_topic_elements": []}'
        )

        method = AnswerRelevancy(provider="openai", scale="1-10")
        sample = Sample(input="test", output="test output")
        result = await method.evaluate(sample)

        assert result.score == 0.8  # 8/10 normalized

    def test_invalid_scale(self, mock_llm_provider) -> None:
        """Test that invalid scale raises ValueError."""
        with pytest.raises(ValueError):
            AnswerRelevancy(provider="openai", scale="invalid")
