"""Tests for LLM-based evaluation methods."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.single import SingleCriterionJudge


class TestSingleCriterionJudgeInit:
    """Tests for SingleCriterionJudge initialization."""

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_init_with_1_5_scale(self, mock_litellm: MagicMock) -> None:
        """Test initialization with 1-5 scale."""
        judge = SingleCriterionJudge(
            criterion="helpfulness",
            scale="1-5",
            provider="openai",
        )

        assert judge.criterion == "helpfulness"
        assert judge.scale == "1-5"
        assert judge.threshold is None

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_init_with_1_10_scale(self, mock_litellm: MagicMock) -> None:
        """Test initialization with 1-10 scale."""
        judge = SingleCriterionJudge(
            criterion="accuracy",
            scale="1-10",
            threshold=7,
            provider="openai",
        )

        assert judge.scale == "1-10"
        assert judge.threshold == 7

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_init_with_binary_scale(self, mock_litellm: MagicMock) -> None:
        """Test initialization with binary scale."""
        judge = SingleCriterionJudge(
            criterion="is_safe",
            scale="binary",
            provider="openai",
        )

        assert judge.scale == "binary"

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_init_invalid_scale(self, mock_litellm: MagicMock) -> None:
        """Test that invalid scale raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            SingleCriterionJudge(
                criterion="test",
                scale="1-100",  # Invalid scale
                provider="openai",
            )
        assert "Invalid scale" in str(exc_info.value)

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_init_with_threshold(self, mock_litellm: MagicMock) -> None:
        """Test initialization with threshold."""
        judge = SingleCriterionJudge(
            criterion="quality",
            scale="1-5",
            threshold=3.5,
            provider="openai",
        )

        assert judge.threshold == 3.5

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_init_with_custom_system_prompt(self, mock_litellm: MagicMock) -> None:
        """Test initialization with custom system prompt."""
        custom_prompt = "You are a custom evaluator."
        judge = SingleCriterionJudge(
            criterion="test",
            system_prompt=custom_prompt,
            provider="openai",
        )

        assert judge.custom_system_prompt == custom_prompt


class TestSingleCriterionJudgeBuildPrompt:
    """Tests for the _build_prompt method."""

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_build_prompt_basic(self, mock_litellm: MagicMock) -> None:
        """Test _build_prompt with basic sample."""
        judge = SingleCriterionJudge(
            criterion="helpfulness",
            scale="1-5",
            provider="openai",
        )
        sample = Sample(
            input="What is Python?",
            output="Python is a programming language.",
        )

        messages = judge._build_prompt(sample)

        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"
        assert "helpfulness" in messages[0]["content"]
        assert "What is Python?" in messages[1]["content"]
        assert "Python is a programming language" in messages[1]["content"]

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_build_prompt_with_expected(self, mock_litellm: MagicMock) -> None:
        """Test _build_prompt includes expected when present."""
        judge = SingleCriterionJudge(
            criterion="accuracy",
            provider="openai",
        )
        sample = Sample(
            input="What is 2+2?",
            output="The answer is 4.",
            expected="4",
        )

        messages = judge._build_prompt(sample)

        assert "Expected Response" in messages[1]["content"]
        assert "4" in messages[1]["content"]

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_build_prompt_with_context(self, mock_litellm: MagicMock) -> None:
        """Test _build_prompt includes context when present."""
        judge = SingleCriterionJudge(
            criterion="relevance",
            provider="openai",
        )
        sample = Sample(
            input="Summarize the document",
            output="The document discusses AI.",
            context="This is a document about artificial intelligence and machine learning.",
        )

        messages = judge._build_prompt(sample)

        assert "Context" in messages[1]["content"]
        assert "artificial intelligence" in messages[1]["content"]

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_build_prompt_with_conversation(self, mock_litellm: MagicMock) -> None:
        """Test _build_prompt includes conversation history when present."""
        judge = SingleCriterionJudge(
            criterion="coherence",
            include_conversation=True,
            provider="openai",
        )
        sample = Sample(
            input="And what about cats?",
            output="Cats are also great pets.",
            conversation=[
                {"role": "user", "content": "Tell me about dogs"},
                {"role": "assistant", "content": "Dogs are great pets."},
            ],
        )

        messages = judge._build_prompt(sample)

        assert "Conversation History" in messages[1]["content"]
        assert "Tell me about dogs" in messages[1]["content"]
        assert "Dogs are great pets" in messages[1]["content"]

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_build_prompt_without_conversation(self, mock_litellm: MagicMock) -> None:
        """Test _build_prompt excludes conversation when include_conversation=False."""
        judge = SingleCriterionJudge(
            criterion="test",
            include_conversation=False,
            provider="openai",
        )
        sample = Sample(
            input="test input",
            output="test output",
            conversation=[
                {"role": "user", "content": "previous message"},
            ],
        )

        messages = judge._build_prompt(sample)

        assert "Conversation History" not in messages[1]["content"]
        assert "previous message" not in messages[1]["content"]

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_build_prompt_custom_system_prompt(self, mock_litellm: MagicMock) -> None:
        """Test _build_prompt uses custom system prompt when provided."""
        custom_prompt = "You are a specialized evaluator for code quality."
        judge = SingleCriterionJudge(
            criterion="code_quality",
            system_prompt=custom_prompt,
            provider="openai",
        )
        sample = Sample(
            input="Review this code",
            output="def foo(): pass",
        )

        messages = judge._build_prompt(sample)

        assert messages[0]["content"] == custom_prompt


class TestSingleCriterionJudgeParseResponse:
    """Tests for the _parse_response method."""

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_numeric_scale(self, mock_litellm: MagicMock) -> None:
        """Test parsing numeric scale response."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )
        response = '{"score": 4, "reason": "Good response overall"}'

        score, reason = judge._parse_response(response)

        assert score == 4.0
        assert reason == "Good response overall"

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_1_10_scale(self, mock_litellm: MagicMock) -> None:
        """Test parsing 1-10 scale response."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-10",
            provider="openai",
        )
        response = '{"score": 8, "reason": "Very good"}'

        score, reason = judge._parse_response(response)

        assert score == 8.0

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_binary_yes(self, mock_litellm: MagicMock) -> None:
        """Test parsing binary scale with 'yes' response."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="binary",
            provider="openai",
        )
        response = '{"score": "yes", "reason": "Meets criteria"}'

        score, reason = judge._parse_response(response)

        assert score == 1.0
        assert reason == "Meets criteria"

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_binary_no(self, mock_litellm: MagicMock) -> None:
        """Test parsing binary scale with 'no' response."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="binary",
            provider="openai",
        )
        response = '{"score": "no", "reason": "Does not meet criteria"}'

        score, reason = judge._parse_response(response)

        assert score == 0.0

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_binary_true(self, mock_litellm: MagicMock) -> None:
        """Test parsing binary scale with 'true' response."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="binary",
            provider="openai",
        )
        response = '{"score": "true", "reason": "Valid"}'

        score, reason = judge._parse_response(response)

        assert score == 1.0

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_binary_false(self, mock_litellm: MagicMock) -> None:
        """Test parsing binary scale with 'false' response."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="binary",
            provider="openai",
        )
        response = '{"score": "false", "reason": "Invalid"}'

        score, reason = judge._parse_response(response)

        assert score == 0.0

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_with_extra_text(self, mock_litellm: MagicMock) -> None:
        """Test parsing response with extra text around JSON."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )
        response = 'Here is my evaluation:\n{"score": 3, "reason": "Average"}\nThank you!'

        score, reason = judge._parse_response(response)

        assert score == 3.0
        assert reason == "Average"

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_no_json(self, mock_litellm: MagicMock) -> None:
        """Test parsing response with no JSON raises ValueError."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )
        response = "This is just plain text with no JSON"

        with pytest.raises(ValueError) as exc_info:
            judge._parse_response(response)
        assert "No JSON found" in str(exc_info.value)

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_invalid_json(self, mock_litellm: MagicMock) -> None:
        """Test parsing invalid JSON raises ValueError."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )
        response = '{"score": invalid}'

        with pytest.raises(ValueError) as exc_info:
            judge._parse_response(response)
        assert "Invalid JSON" in str(exc_info.value)

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_missing_score(self, mock_litellm: MagicMock) -> None:
        """Test parsing response without score field raises ValueError."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )
        response = '{"reason": "No score here"}'

        with pytest.raises(ValueError) as exc_info:
            judge._parse_response(response)
        assert "No score found" in str(exc_info.value)

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_invalid_binary_score(self, mock_litellm: MagicMock) -> None:
        """Test parsing invalid binary score raises ValueError."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="binary",
            provider="openai",
        )
        response = '{"score": "maybe", "reason": "Unclear"}'

        with pytest.raises(ValueError) as exc_info:
            judge._parse_response(response)
        assert "Invalid binary score" in str(exc_info.value)

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_parse_response_float_score(self, mock_litellm: MagicMock) -> None:
        """Test parsing float score in numeric scale."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )
        response = '{"score": 3.5, "reason": "Between average and good"}'

        score, reason = judge._parse_response(response)

        assert score == 3.5


class TestSingleCriterionJudgeEvaluate:
    """Tests for the evaluate method with mocked LLM calls."""

    @pytest.mark.asyncio
    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    async def test_evaluate_returns_result(self, mock_litellm: MagicMock) -> None:
        """Test evaluate returns proper MethodResult."""
        # Setup mock
        mock_litellm.acompletion = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content='{"score": 4, "reason": "Helpful response"}')
                    )
                ]
            )
        )

        judge = SingleCriterionJudge(
            criterion="helpfulness",
            scale="1-5",
            threshold=3,
            provider="openai",
        )
        sample = Sample(
            input="What is Python?",
            output="Python is a versatile programming language.",
        )

        result = await judge.evaluate(sample)

        assert isinstance(result, MethodResult)
        assert result.score == 4.0
        assert result.passed is True  # 4 >= 3
        assert result.reason == "Helpful response"
        assert result.metadata is not None
        assert result.metadata["criterion"] == "helpfulness"
        assert result.metadata["scale"] == "1-5"

    @pytest.mark.asyncio
    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    async def test_evaluate_below_threshold(self, mock_litellm: MagicMock) -> None:
        """Test evaluate returns passed=False when below threshold."""
        mock_litellm.acompletion = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(message=MagicMock(content='{"score": 2, "reason": "Poor quality"}'))
                ]
            )
        )

        judge = SingleCriterionJudge(
            criterion="quality",
            scale="1-5",
            threshold=3,
            provider="openai",
        )
        sample = Sample(input="test", output="test")

        result = await judge.evaluate(sample)

        assert result.score == 2.0
        assert result.passed is False  # 2 < 3

    @pytest.mark.asyncio
    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    async def test_evaluate_no_threshold(self, mock_litellm: MagicMock) -> None:
        """Test evaluate returns passed=None when no threshold set."""
        mock_litellm.acompletion = AsyncMock(
            return_value=MagicMock(
                choices=[MagicMock(message=MagicMock(content='{"score": 4, "reason": "Good"}'))]
            )
        )

        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            threshold=None,
            provider="openai",
        )
        sample = Sample(input="test", output="test")

        result = await judge.evaluate(sample)

        assert result.score == 4.0
        assert result.passed is None  # No threshold means no pass/fail

    @pytest.mark.asyncio
    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    async def test_evaluate_binary_scale(self, mock_litellm: MagicMock) -> None:
        """Test evaluate with binary scale."""
        mock_litellm.acompletion = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(content='{"score": "yes", "reason": "Safe content"}')
                    )
                ]
            )
        )

        judge = SingleCriterionJudge(
            criterion="is_safe",
            scale="binary",
            threshold=0.5,
            provider="openai",
        )
        sample = Sample(input="test", output="test output")

        result = await judge.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True  # 1.0 >= 0.5

    @pytest.mark.asyncio
    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    async def test_evaluate_calls_llm_complete(self, mock_litellm: MagicMock) -> None:
        """Test that evaluate calls the LLM provider's complete method."""
        mock_litellm.acompletion = AsyncMock(
            return_value=MagicMock(
                choices=[
                    MagicMock(message=MagicMock(content='{"score": 5, "reason": "Excellent"}'))
                ]
            )
        )

        judge = SingleCriterionJudge(
            criterion="excellence",
            scale="1-5",
            provider="openai",
            model="gpt-4o-mini",
        )
        sample = Sample(input="test input", output="test output")

        await judge.evaluate(sample)

        # Verify LLM was called
        mock_litellm.acompletion.assert_called_once()
        call_kwargs = mock_litellm.acompletion.call_args
        assert "messages" in call_kwargs.kwargs or len(call_kwargs.args) > 0

    @pytest.mark.asyncio
    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    async def test_safe_evaluate_reports_null_provider_content(
        self, mock_litellm: MagicMock,
    ) -> None:
        """Test provider diagnostics when LiteLLM returns content=None."""
        mock_litellm.acompletion = AsyncMock(
            return_value=SimpleNamespace(
                id="resp-123",
                model="vertex_ai/gemini-test",
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=0,
                    total_tokens=10,
                ),
                prompt_feedback={
                    "block_reason": "SAFETY",
                    "prompt_text": "user-secret-token-123",
                },
                safety_ratings=None,
                choices=[
                    SimpleNamespace(
                        finish_reason="safety",
                        safety_ratings=[
                            {
                                "category": "dangerous",
                                "probability": "HIGH",
                                "message_text": "assistant-sensitive-output-456",
                            }
                        ],
                        message=SimpleNamespace(
                            role="assistant",
                            content=None,
                            tool_calls=None,
                            safety_ratings=None,
                        ),
                    )
                ],
            )
        )

        judge = SingleCriterionJudge(
            criterion="safety",
            scale="binary",
            provider="vertex_ai",
            model="gemini-test",
        )
        result = await judge.safe_evaluate(
            Sample(
                input="user-secret-token-123",
                output="assistant-sensitive-output-456",
            )
        )

        assert result.score is None
        assert result.error is not None
        assert "LLM returned null content" in result.error
        assert "finish_reason" in result.error
        assert "safety" in result.error
        assert "prompt_feedback" in result.error
        assert "user-secret-token-123" not in result.error
        assert "assistant-sensitive-output-456" not in result.error


class TestSingleCriterionJudgeScaleDescription:
    """Tests for the _get_scale_description method."""

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_scale_description_binary(self, mock_litellm: MagicMock) -> None:
        """Test scale description for binary scale."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="binary",
            provider="openai",
        )

        desc = judge._get_scale_description()

        assert "yes" in desc.lower()
        assert "no" in desc.lower()

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_scale_description_1_5(self, mock_litellm: MagicMock) -> None:
        """Test scale description for 1-5 scale."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )

        desc = judge._get_scale_description()

        assert "1" in desc
        assert "5" in desc

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_scale_description_1_10(self, mock_litellm: MagicMock) -> None:
        """Test scale description for 1-10 scale."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-10",
            provider="openai",
        )

        desc = judge._get_scale_description()

        assert "1" in desc
        assert "10" in desc


class TestSingleCriterionJudgeJsonFormat:
    """Tests for the _get_json_format method."""

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_json_format_binary(self, mock_litellm: MagicMock) -> None:
        """Test JSON format for binary scale."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="binary",
            provider="openai",
        )

        fmt = judge._get_json_format()

        assert "yes" in fmt.lower() or "no" in fmt.lower()
        assert "reason" in fmt

    @patch("mankinds_eval.methods.llm.providers.LITELLM_AVAILABLE", True)
    @patch("mankinds_eval.methods.llm.providers.litellm")
    def test_json_format_numeric(self, mock_litellm: MagicMock) -> None:
        """Test JSON format for numeric scales."""
        judge = SingleCriterionJudge(
            criterion="test",
            scale="1-5",
            provider="openai",
        )

        fmt = judge._get_json_format()

        assert "score" in fmt
        assert "number" in fmt.lower()
        assert "reason" in fmt
