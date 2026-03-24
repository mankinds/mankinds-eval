"""Tests for structured output evaluation methods."""

from __future__ import annotations

import pytest

from mankinds_eval.core import Sample
from mankinds_eval.methods.heuristic.structured import JSONSchema, JSONValid, NoRefusal

# =============================================================================
# JSONValid Tests
# =============================================================================


class TestJSONValid:
    """Tests for the JSONValid method."""

    @pytest.mark.asyncio
    async def test_valid_json_object(self) -> None:
        """Test valid JSON object returns score 1.0."""
        method = JSONValid()
        sample = Sample(
            input="Generate JSON",
            output='{"name": "test", "value": 42}',
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["parsed"] is True
        assert result.metadata["json_type"] == "dict"

    @pytest.mark.asyncio
    async def test_valid_json_array(self) -> None:
        """Test valid JSON array returns score 1.0."""
        method = JSONValid()
        sample = Sample(
            input="Generate JSON",
            output='[1, 2, 3, "test"]',
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["json_type"] == "list"

    @pytest.mark.asyncio
    async def test_invalid_json(self) -> None:
        """Test invalid JSON returns score 0.0."""
        method = JSONValid()
        sample = Sample(
            input="Generate JSON",
            output='{"name": test}',  # Missing quotes around test
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["parsed"] is False
        assert "error_position" in result.metadata

    @pytest.mark.asyncio
    async def test_extract_from_markdown(self) -> None:
        """Test JSON extraction from markdown code blocks."""
        method = JSONValid(extract_from_markdown=True)
        sample = Sample(
            input="Generate JSON",
            output='Here is the result:\n```json\n{"name": "test"}\n```\nDone.',
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["extracted_from_markdown"] is True

    @pytest.mark.asyncio
    async def test_extract_from_markdown_no_language(self) -> None:
        """Test JSON extraction from markdown without language tag."""
        method = JSONValid(extract_from_markdown=True)
        sample = Sample(
            input="Generate JSON",
            output='```\n{"name": "test"}\n```',
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_no_extract_from_markdown(self) -> None:
        """Test with markdown extraction disabled."""
        method = JSONValid(extract_from_markdown=False)
        sample = Sample(
            input="Generate JSON",
            output='```json\n{"name": "test"}\n```',
        )
        result = await method.evaluate(sample)

        # Should fail because the backticks make it invalid JSON
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_allow_trailing_comma(self) -> None:
        """Test with trailing comma tolerance."""
        method = JSONValid(allow_trailing_comma=True)
        sample = Sample(
            input="Generate JSON",
            output='{"name": "test", "value": 42,}',  # Trailing comma
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_plain_text_fails(self) -> None:
        """Test that plain text fails validation."""
        method = JSONValid()
        sample = Sample(
            input="Generate JSON",
            output="This is just plain text, not JSON.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False


# =============================================================================
# JSONSchema Tests
# =============================================================================


class TestJSONSchema:
    """Tests for the JSONSchema method."""

    @pytest.mark.asyncio
    async def test_valid_schema(self) -> None:
        """Test JSON matching schema returns score 1.0."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        method = JSONSchema(schema=schema)
        sample = Sample(
            input="Generate user",
            output='{"name": "John", "age": 30}',
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["valid"] is True

    @pytest.mark.asyncio
    async def test_invalid_schema_missing_required(self) -> None:
        """Test JSON missing required field fails."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
            "required": ["name"],
        }
        method = JSONSchema(schema=schema)
        sample = Sample(
            input="Generate user",
            output='{"age": 30}',  # Missing required 'name'
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["valid"] is False
        assert len(result.metadata["errors"]) > 0

    @pytest.mark.asyncio
    async def test_invalid_schema_wrong_type(self) -> None:
        """Test JSON with wrong type fails."""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer"},
            },
        }
        method = JSONSchema(schema=schema)
        sample = Sample(
            input="Generate user",
            output='{"age": "thirty"}',  # String instead of integer
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_strict_mode_additional_properties(self) -> None:
        """Test strict mode fails on additional properties."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        method = JSONSchema(schema=schema, strict=True)
        sample = Sample(
            input="Generate user",
            output='{"name": "John", "extra": "field"}',  # Extra field
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_non_strict_allows_additional(self) -> None:
        """Test non-strict mode allows additional properties."""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
            },
        }
        method = JSONSchema(schema=schema, strict=False)
        sample = Sample(
            input="Generate user",
            output='{"name": "John", "extra": "field"}',
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_invalid_json_fails_schema(self) -> None:
        """Test invalid JSON fails schema validation."""
        schema = {"type": "object"}
        method = JSONSchema(schema=schema)
        sample = Sample(
            input="Generate user",
            output="not valid json",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False
        assert "JSON" in result.reason

    @pytest.mark.asyncio
    async def test_extract_from_markdown_schema(self) -> None:
        """Test schema validation with markdown extraction."""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        method = JSONSchema(schema=schema, extract_from_markdown=True)
        sample = Sample(
            input="Generate user",
            output='```json\n{"name": "test"}\n```',
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True


# =============================================================================
# NoRefusal Tests
# =============================================================================


class TestNoRefusal:
    """Tests for the NoRefusal method."""

    @pytest.mark.asyncio
    async def test_no_refusal_passes(self) -> None:
        """Test normal response without refusal passes."""
        method = NoRefusal()
        sample = Sample(
            input="What is Python?",
            output="Python is a programming language known for its simplicity.",
        )
        result = await method.evaluate(sample)

        assert result.score == 1.0
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["is_refusal"] is False

    @pytest.mark.asyncio
    async def test_refusal_detected_sorry(self) -> None:
        """Test refusal detected with 'sorry' pattern."""
        method = NoRefusal()
        sample = Sample(
            input="How to hack?",
            output="I'm sorry, I can't help with that request.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["is_refusal"] is True

    @pytest.mark.asyncio
    async def test_refusal_detected_cannot(self) -> None:
        """Test refusal detected with 'cannot' pattern."""
        method = NoRefusal()
        sample = Sample(
            input="Do something illegal",
            output="I cannot assist with illegal activities.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_refusal_detected_as_an_ai(self) -> None:
        """Test refusal detected with 'as an AI' pattern."""
        method = NoRefusal()
        sample = Sample(
            input="Tell me secrets",
            output="As an AI, I don't have personal opinions on this matter.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_refusal_french(self) -> None:
        """Test refusal detection in French."""
        method = NoRefusal()
        sample = Sample(
            input="Question en francais",
            output="Je suis desole, je ne peux pas vous aider avec ca.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_custom_patterns(self) -> None:
        """Test with custom refusal patterns."""
        method = NoRefusal(
            patterns=[r"(?i)contact support"],
            extend_default=False,
        )
        sample = Sample(
            input="Help me",
            output="Please contact support for this issue.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_extend_default_patterns(self) -> None:
        """Test extending default patterns with custom ones."""
        method = NoRefusal(
            patterns=[r"(?i)custom refusal"],
            extend_default=True,
        )
        # Default patterns still work
        sample = Sample(
            input="test",
            output="I'm sorry, I cannot help.",
        )
        result = await method.evaluate(sample)

        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_case_sensitivity(self) -> None:
        """Test case sensitivity option."""
        method = NoRefusal(case_sensitive=True)
        # Default patterns have (?i) so they're case insensitive anyway
        sample = Sample(
            input="test",
            output="I'M SORRY, I CANNOT HELP.",
        )
        result = await method.evaluate(sample)

        # Should still detect because default patterns use (?i)
        assert result.score == 0.0
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_matched_patterns_in_metadata(self) -> None:
        """Test that matched patterns are in metadata."""
        method = NoRefusal()
        sample = Sample(
            input="test",
            output="I'm sorry, I cannot help with that. As an AI, I have limits.",
        )
        result = await method.evaluate(sample)

        assert result.metadata is not None
        assert len(result.metadata["matched_patterns"]) >= 1
        assert len(result.metadata["matched_texts"]) >= 1
