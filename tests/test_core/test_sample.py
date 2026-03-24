"""Tests for the Sample class."""

from __future__ import annotations

import pytest

from mankinds_eval.core.sample import Sample


class TestSampleCreation:
    """Tests for Sample instance creation."""

    def test_valid_sample_creation(self) -> None:
        """Test creating a sample with required fields only."""
        sample = Sample(
            input="What is Python?",
            output="Python is a programming language.",
        )
        assert sample.input == "What is Python?"
        assert sample.output == "Python is a programming language."
        assert sample.expected is None
        assert sample.context is None
        assert sample.conversation is None
        assert sample.metadata is None

    def test_sample_with_all_optional_fields(self) -> None:
        """Test creating a sample with all optional fields populated."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        metadata = {"source": "test", "id": 123}

        sample = Sample(
            input="What is the answer?",
            output="42",
            expected="42",
            context="This is additional context.",
            conversation=conversation,
            metadata=metadata,
        )

        assert sample.input == "What is the answer?"
        assert sample.output == "42"
        assert sample.expected == "42"
        assert sample.context == "This is additional context."
        assert sample.conversation == conversation
        assert sample.metadata == metadata

    def test_sample_with_expected_only(self) -> None:
        """Test creating a sample with only expected field set."""
        sample = Sample(
            input="Question",
            output="Answer",
            expected="Expected answer",
        )
        assert sample.expected == "Expected answer"
        assert sample.context is None

    def test_sample_with_context_only(self) -> None:
        """Test creating a sample with only context field set."""
        sample = Sample(
            input="Question",
            output="Answer",
            context="Some context",
        )
        assert sample.context == "Some context"
        assert sample.expected is None


class TestSampleValidation:
    """Tests for Sample validation errors."""

    def test_empty_input_raises_error(self) -> None:
        """Test that empty input raises ValueError."""
        with pytest.raises(ValueError, match="input must be a non-empty string"):
            Sample(input="", output="Some output")

    def test_whitespace_only_input_raises_error(self) -> None:
        """Test that whitespace-only input raises ValueError."""
        with pytest.raises(ValueError, match="input must be a non-empty string"):
            Sample(input="   ", output="Some output")

    def test_empty_output_raises_error(self) -> None:
        """Test that empty output raises ValueError."""
        with pytest.raises(ValueError, match="output must be a non-empty string"):
            Sample(input="Some input", output="")

    def test_whitespace_only_output_raises_error(self) -> None:
        """Test that whitespace-only output raises ValueError."""
        with pytest.raises(ValueError, match="output must be a non-empty string"):
            Sample(input="Some input", output="  \t\n  ")

    def test_non_string_input_raises_error(self) -> None:
        """Test that non-string input raises ValueError."""
        with pytest.raises(ValueError, match="input must be a non-empty string"):
            Sample(input=123, output="Some output")  # type: ignore[arg-type]

    def test_non_string_output_raises_error(self) -> None:
        """Test that non-string output raises ValueError."""
        with pytest.raises(ValueError, match="output must be a non-empty string"):
            Sample(input="Some input", output=["list"])  # type: ignore[arg-type]


class TestConversationValidation:
    """Tests for conversation field validation."""

    def test_valid_conversation(self) -> None:
        """Test that valid conversation passes validation."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]
        sample = Sample(
            input="Question",
            output="Answer",
            conversation=conversation,
        )
        assert sample.conversation == conversation

    def test_conversation_missing_role_raises_error(self) -> None:
        """Test that conversation message missing role raises ValueError."""
        conversation = [
            {"content": "Hello"},  # Missing role
        ]
        with pytest.raises(ValueError, match="conversation\\[0\\] missing required key 'role'"):
            Sample(input="Question", output="Answer", conversation=conversation)

    def test_conversation_missing_content_raises_error(self) -> None:
        """Test that conversation message missing content raises ValueError."""
        conversation = [
            {"role": "user"},  # Missing content
        ]
        with pytest.raises(ValueError, match="conversation\\[0\\] missing required key 'content'"):
            Sample(input="Question", output="Answer", conversation=conversation)

    def test_conversation_not_list_raises_error(self) -> None:
        """Test that non-list conversation raises ValueError."""
        with pytest.raises(ValueError, match="conversation must be a list"):
            Sample(
                input="Question",
                output="Answer",
                conversation={"role": "user", "content": "Hello"},  # type: ignore[arg-type]
            )

    def test_conversation_item_not_dict_raises_error(self) -> None:
        """Test that non-dict conversation item raises ValueError."""
        with pytest.raises(ValueError, match="conversation\\[0\\] must be a dict"):
            Sample(
                input="Question",
                output="Answer",
                conversation=["not a dict"],  # type: ignore[list-item]
            )

    def test_conversation_second_item_missing_role(self) -> None:
        """Test error message includes correct index for second item."""
        conversation = [
            {"role": "user", "content": "Hello"},
            {"content": "Missing role"},  # Second item missing role
        ]
        with pytest.raises(ValueError, match="conversation\\[1\\] missing required key 'role'"):
            Sample(input="Question", output="Answer", conversation=conversation)

    def test_conversation_with_extra_fields(self) -> None:
        """Test that conversation with extra fields is allowed."""
        conversation = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01"},
        ]
        sample = Sample(
            input="Question",
            output="Answer",
            conversation=conversation,
        )
        assert sample.conversation[0]["timestamp"] == "2024-01-01"


class TestSampleToDict:
    """Tests for Sample.to_dict() method."""

    def test_to_dict_required_fields_only(self) -> None:
        """Test to_dict with only required fields."""
        sample = Sample(input="Question", output="Answer")
        result = sample.to_dict()

        assert result == {
            "input": "Question",
            "output": "Answer",
        }

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        conversation = [{"role": "user", "content": "Hi"}]
        metadata = {"key": "value"}

        sample = Sample(
            input="Question",
            output="Answer",
            expected="Expected",
            context="Context",
            conversation=conversation,
            metadata=metadata,
        )
        result = sample.to_dict()

        assert result == {
            "input": "Question",
            "output": "Answer",
            "expected": "Expected",
            "context": "Context",
            "conversation": conversation,
            "metadata": metadata,
        }

    def test_to_dict_excludes_none_values(self) -> None:
        """Test that to_dict excludes None optional fields."""
        sample = Sample(
            input="Question",
            output="Answer",
            expected="Expected",
            # context, conversation, metadata are None
        )
        result = sample.to_dict()

        assert "context" not in result
        assert "conversation" not in result
        assert "metadata" not in result
        assert result == {
            "input": "Question",
            "output": "Answer",
            "expected": "Expected",
        }

    def test_to_dict_returns_new_dict(self) -> None:
        """Test that to_dict returns a new dictionary each time."""
        sample = Sample(input="Question", output="Answer")
        dict1 = sample.to_dict()
        dict2 = sample.to_dict()

        assert dict1 is not dict2
        assert dict1 == dict2


class TestSampleFromDict:
    """Tests for Sample.from_dict() class method."""

    def test_from_dict_required_fields_only(self) -> None:
        """Test from_dict with only required fields."""
        data = {
            "input": "Question",
            "output": "Answer",
        }
        sample = Sample.from_dict(data)

        assert sample.input == "Question"
        assert sample.output == "Answer"
        assert sample.expected is None
        assert sample.context is None
        assert sample.conversation is None
        assert sample.metadata is None

    def test_from_dict_with_all_fields(self) -> None:
        """Test from_dict with all fields."""
        conversation = [{"role": "user", "content": "Hi"}]
        metadata = {"key": "value"}

        data = {
            "input": "Question",
            "output": "Answer",
            "expected": "Expected",
            "context": "Context",
            "conversation": conversation,
            "metadata": metadata,
        }
        sample = Sample.from_dict(data)

        assert sample.input == "Question"
        assert sample.output == "Answer"
        assert sample.expected == "Expected"
        assert sample.context == "Context"
        assert sample.conversation == conversation
        assert sample.metadata == metadata

    def test_from_dict_missing_input_raises_error(self) -> None:
        """Test that from_dict raises KeyError when input is missing."""
        data = {"output": "Answer"}
        with pytest.raises(KeyError):
            Sample.from_dict(data)

    def test_from_dict_missing_output_raises_error(self) -> None:
        """Test that from_dict raises KeyError when output is missing."""
        data = {"input": "Question"}
        with pytest.raises(KeyError):
            Sample.from_dict(data)

    def test_from_dict_with_extra_fields_ignored(self) -> None:
        """Test that extra fields in dict are ignored."""
        data = {
            "input": "Question",
            "output": "Answer",
            "extra_field": "ignored",
            "another_extra": 123,
        }
        sample = Sample.from_dict(data)

        assert sample.input == "Question"
        assert sample.output == "Answer"
        assert not hasattr(sample, "extra_field")

    def test_from_dict_roundtrip(self) -> None:
        """Test that to_dict and from_dict are inverse operations."""
        original = Sample(
            input="Question",
            output="Answer",
            expected="Expected",
            context="Context",
            conversation=[{"role": "user", "content": "Hi"}],
            metadata={"key": "value"},
        )

        data = original.to_dict()
        restored = Sample.from_dict(data)

        assert restored.input == original.input
        assert restored.output == original.output
        assert restored.expected == original.expected
        assert restored.context == original.context
        assert restored.conversation == original.conversation
        assert restored.metadata == original.metadata


class TestSampleFixture:
    """Tests using the sample fixture from conftest."""

    def test_sample_fixture(self, sample: Sample) -> None:
        """Test that the sample fixture is correctly created."""
        assert sample.input == "What is 2 + 2?"
        assert sample.output == "The answer is 4."
        assert sample.expected == "4"

    def test_sample_with_all_fields_fixture(self, sample_with_all_fields: Sample) -> None:
        """Test that the sample_with_all_fields fixture has all fields."""
        assert sample_with_all_fields.input is not None
        assert sample_with_all_fields.output is not None
        assert sample_with_all_fields.expected is not None
        assert sample_with_all_fields.context is not None
        assert sample_with_all_fields.conversation is not None
        assert sample_with_all_fields.metadata is not None
