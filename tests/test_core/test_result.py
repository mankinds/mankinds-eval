"""Tests for the MethodResult class."""

from __future__ import annotations

import pytest

from mankinds_eval.core.result import MethodResult


class TestMethodResultCreation:
    """Tests for MethodResult instance creation."""

    def test_valid_result_creation(self) -> None:
        """Test creating a result with required fields only."""
        result = MethodResult(
            method_name="test_method",
            score=0.75,
        )
        assert result.method_name == "test_method"
        assert result.score == 0.75
        assert result.passed is None
        assert result.reason is None
        assert result.metadata is None
        assert result.error is None

    def test_result_with_all_optional_fields(self) -> None:
        """Test creating a result with all optional fields populated."""
        metadata = {"confidence": 0.9, "tokens_used": 150}

        result = MethodResult(
            method_name="llm_judge",
            score=0.85,
            passed=True,
            reason="The response accurately addresses the question.",
            metadata=metadata,
            error=None,
        )

        assert result.method_name == "llm_judge"
        assert result.score == 0.85
        assert result.passed is True
        assert result.reason == "The response accurately addresses the question."
        assert result.metadata == metadata
        assert result.error is None

    def test_result_with_none_score(self) -> None:
        """Test creating a result with None score (e.g., error case)."""
        result = MethodResult(
            method_name="failed_method",
            score=None,
        )
        assert result.score is None

    def test_result_with_zero_score(self) -> None:
        """Test creating a result with zero score."""
        result = MethodResult(
            method_name="zero_score_method",
            score=0.0,
        )
        assert result.score == 0.0

    def test_result_with_perfect_score(self) -> None:
        """Test creating a result with perfect score."""
        result = MethodResult(
            method_name="perfect_method",
            score=1.0,
            passed=True,
        )
        assert result.score == 1.0
        assert result.passed is True

    def test_result_passed_false(self) -> None:
        """Test creating a result with passed=False."""
        result = MethodResult(
            method_name="threshold_method",
            score=0.3,
            passed=False,
            reason="Score below threshold of 0.5",
        )
        assert result.passed is False
        assert result.reason == "Score below threshold of 0.5"

    def test_result_with_negative_score(self) -> None:
        """Test creating a result with negative score (allowed)."""
        result = MethodResult(
            method_name="negative_method",
            score=-0.5,
        )
        assert result.score == -0.5


class TestMethodResultSuccess:
    """Tests for the success property."""

    def test_success_true_when_no_error(self) -> None:
        """Test that success is True when error is None."""
        result = MethodResult(
            method_name="test_method",
            score=0.75,
        )
        assert result.success is True

    def test_success_true_with_all_fields_no_error(self) -> None:
        """Test that success is True even with all fields populated but no error."""
        result = MethodResult(
            method_name="test_method",
            score=0.5,
            passed=False,
            reason="Some reason",
            metadata={"key": "value"},
            error=None,
        )
        assert result.success is True

    def test_success_false_when_error_present(self) -> None:
        """Test that success is False when error is set."""
        result = MethodResult(
            method_name="failed_method",
            score=None,
            error="Connection timeout",
        )
        assert result.success is False

    def test_success_false_with_empty_string_error(self) -> None:
        """Test that success is True when error is empty string (truthy check)."""
        # Empty string is not None, so success should be False
        result = MethodResult(
            method_name="test_method",
            score=0.5,
            error="",
        )
        # Based on the implementation (self.error is None), empty string is not None
        assert result.success is False

    def test_success_false_even_with_score(self) -> None:
        """Test that success is False when error is present even if score exists."""
        result = MethodResult(
            method_name="partial_method",
            score=0.5,  # Score exists
            error="Partial failure occurred",
        )
        assert result.success is False


class TestMethodResultToDict:
    """Tests for MethodResult.to_dict() method."""

    def test_to_dict_required_fields_only(self) -> None:
        """Test to_dict with only required fields."""
        result = MethodResult(
            method_name="test_method",
            score=0.75,
        )
        output = result.to_dict()

        assert output == {
            "method_name": "test_method",
            "score": 0.75,
        }

    def test_to_dict_with_all_fields(self) -> None:
        """Test to_dict with all fields populated."""
        metadata = {"key": "value"}

        result = MethodResult(
            method_name="test_method",
            score=0.85,
            passed=True,
            reason="Good response",
            metadata=metadata,
            error=None,
        )
        output = result.to_dict()

        # Note: error=None should not be included in output
        assert output == {
            "method_name": "test_method",
            "score": 0.85,
            "passed": True,
            "reason": "Good response",
            "metadata": metadata,
        }

    def test_to_dict_with_error(self) -> None:
        """Test to_dict includes error when present."""
        result = MethodResult(
            method_name="failed_method",
            score=None,
            error="Timeout occurred",
        )
        output = result.to_dict()

        assert output == {
            "method_name": "failed_method",
            "score": None,
            "error": "Timeout occurred",
        }

    def test_to_dict_excludes_none_optional_values(self) -> None:
        """Test that to_dict excludes None optional fields."""
        result = MethodResult(
            method_name="test_method",
            score=0.5,
            passed=True,
            # reason, metadata, error are None
        )
        output = result.to_dict()

        assert "reason" not in output
        assert "metadata" not in output
        assert "error" not in output

    def test_to_dict_includes_false_passed(self) -> None:
        """Test that to_dict includes passed=False (not excluded like None)."""
        result = MethodResult(
            method_name="test_method",
            score=0.3,
            passed=False,
        )
        output = result.to_dict()

        assert output["passed"] is False

    def test_to_dict_returns_new_dict(self) -> None:
        """Test that to_dict returns a new dictionary each time."""
        result = MethodResult(method_name="test", score=0.5)
        dict1 = result.to_dict()
        dict2 = result.to_dict()

        assert dict1 is not dict2
        assert dict1 == dict2


class TestMethodResultFromDict:
    """Tests for MethodResult.from_dict() class method."""

    def test_from_dict_required_fields_only(self) -> None:
        """Test from_dict with only required fields."""
        data = {
            "method_name": "test_method",
            "score": 0.75,
        }
        result = MethodResult.from_dict(data)

        assert result.method_name == "test_method"
        assert result.score == 0.75
        assert result.passed is None
        assert result.reason is None
        assert result.metadata is None
        assert result.error is None

    def test_from_dict_with_all_fields(self) -> None:
        """Test from_dict with all fields."""
        metadata = {"key": "value"}

        data = {
            "method_name": "test_method",
            "score": 0.85,
            "passed": True,
            "reason": "Good response",
            "metadata": metadata,
            "error": None,
        }
        result = MethodResult.from_dict(data)

        assert result.method_name == "test_method"
        assert result.score == 0.85
        assert result.passed is True
        assert result.reason == "Good response"
        assert result.metadata == metadata
        assert result.error is None

    def test_from_dict_with_error(self) -> None:
        """Test from_dict with error field."""
        data = {
            "method_name": "failed_method",
            "score": None,
            "error": "Connection failed",
        }
        result = MethodResult.from_dict(data)

        assert result.method_name == "failed_method"
        assert result.score is None
        assert result.error == "Connection failed"
        assert result.success is False

    def test_from_dict_missing_method_name_raises_error(self) -> None:
        """Test that from_dict raises KeyError when method_name is missing."""
        data = {"score": 0.75}
        with pytest.raises(KeyError):
            MethodResult.from_dict(data)

    def test_from_dict_missing_score_raises_error(self) -> None:
        """Test that from_dict raises KeyError when score is missing."""
        data = {"method_name": "test_method"}
        with pytest.raises(KeyError):
            MethodResult.from_dict(data)

    def test_from_dict_with_extra_fields_ignored(self) -> None:
        """Test that extra fields in dict are ignored."""
        data = {
            "method_name": "test_method",
            "score": 0.75,
            "extra_field": "ignored",
            "another_extra": 123,
        }
        result = MethodResult.from_dict(data)

        assert result.method_name == "test_method"
        assert result.score == 0.75
        assert not hasattr(result, "extra_field")

    def test_from_dict_roundtrip(self) -> None:
        """Test that to_dict and from_dict are inverse operations."""
        original = MethodResult(
            method_name="test_method",
            score=0.85,
            passed=True,
            reason="Good response",
            metadata={"key": "value"},
            error=None,
        )

        data = original.to_dict()
        restored = MethodResult.from_dict(data)

        assert restored.method_name == original.method_name
        assert restored.score == original.score
        assert restored.passed == original.passed
        assert restored.reason == original.reason
        assert restored.metadata == original.metadata
        assert restored.error == original.error

    def test_from_dict_roundtrip_with_error(self) -> None:
        """Test roundtrip with error field."""
        original = MethodResult(
            method_name="failed_method",
            score=None,
            error="Something went wrong",
        )

        data = original.to_dict()
        restored = MethodResult.from_dict(data)

        assert restored.method_name == original.method_name
        assert restored.score == original.score
        assert restored.error == original.error
        assert restored.success == original.success


class TestMethodResultFixture:
    """Tests using the method_result fixtures from conftest."""

    def test_method_result_fixture(self, method_result: MethodResult) -> None:
        """Test that the method_result fixture is correctly created."""
        assert method_result.method_name == "test_method"
        assert method_result.score == 0.85
        assert method_result.passed is True
        assert method_result.reason == "Score exceeded threshold"
        assert method_result.success is True

    def test_method_result_with_error_fixture(self, method_result_with_error: MethodResult) -> None:
        """Test that the method_result_with_error fixture has error set."""
        assert method_result_with_error.method_name == "failed_method"
        assert method_result_with_error.score is None
        assert method_result_with_error.error == "Evaluation failed due to timeout"
        assert method_result_with_error.success is False
