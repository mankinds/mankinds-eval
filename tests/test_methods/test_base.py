"""Tests for the base Method class."""

from __future__ import annotations

import pytest

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method


class ConcreteMethod(Method):
    """Concrete implementation of Method for testing."""

    name = "test_method"
    version = "1.0.0"
    required_fields = ["input", "output", "expected"]

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Simple evaluation that returns a fixed result."""
        return MethodResult(
            method_name=self.name,
            score=1.0,
            passed=True,
            reason="Test passed",
        )


class FailingMethod(Method):
    """Method that raises an exception during evaluation."""

    name = "failing_method"
    version = "1.0.0"
    required_fields = ["input", "output"]

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Raises an exception during evaluation."""
        raise ValueError("Intentional test error")


class TestMethodValidateSample:
    """Tests for the validate_sample method."""

    def test_validate_sample_success(self) -> None:
        """Test validate_sample passes with all required fields."""
        method = ConcreteMethod()
        sample = Sample(
            input="test input",
            output="test output",
            expected="expected output",
        )
        # Should not raise
        method.validate_sample(sample)

    def test_validate_sample_missing_expected(self) -> None:
        """Test validate_sample fails when expected field is missing."""
        method = ConcreteMethod()
        sample = Sample(
            input="test input",
            output="test output",
            # expected is not set
        )
        with pytest.raises(ValueError) as exc_info:
            method.validate_sample(sample)
        assert "missing required field: expected" in str(exc_info.value)

    def test_validate_sample_default_required_fields(self) -> None:
        """Test that default Method only requires input and output."""

        class MinimalMethod(Method):
            name = "minimal"

            async def evaluate(self, sample: Sample) -> MethodResult:
                return MethodResult(method_name=self.name, score=1.0)

        method = MinimalMethod()
        sample = Sample(input="test", output="test")
        # Should not raise since only input and output are required by default
        method.validate_sample(sample)


class TestMethodSafeEvaluate:
    """Tests for the safe_evaluate method with error handling."""

    @pytest.mark.asyncio
    async def test_safe_evaluate_success(self) -> None:
        """Test safe_evaluate returns normal result on success."""
        method = ConcreteMethod()
        sample = Sample(
            input="test input",
            output="test output",
            expected="expected output",
        )
        result = await method.safe_evaluate(sample)

        assert result.error is None
        assert result.score == 1.0
        assert result.passed is True
        assert result.method_name == "test_method"

    @pytest.mark.asyncio
    async def test_safe_evaluate_validation_error(self) -> None:
        """Test safe_evaluate handles validation errors."""
        method = ConcreteMethod()
        sample = Sample(
            input="test input",
            output="test output",
            # Missing required expected field
        )
        result = await method.safe_evaluate(sample)

        assert result.error is not None
        assert "missing required field: expected" in result.error
        assert result.score is None
        assert result.passed is None
        assert result.method_name == "test_method"

    @pytest.mark.asyncio
    async def test_safe_evaluate_evaluation_error(self) -> None:
        """Test safe_evaluate handles errors during evaluation."""
        method = FailingMethod()
        sample = Sample(
            input="test input",
            output="test output",
        )
        result = await method.safe_evaluate(sample)

        assert result.error is not None
        assert "Intentional test error" in result.error
        assert result.score is None
        assert result.passed is None
        assert result.method_name == "failing_method"


class TestMethodToDict:
    """Tests for the to_dict serialization method."""

    def test_to_dict_basic(self) -> None:
        """Test to_dict returns correct structure."""
        method = ConcreteMethod()
        result = method.to_dict()

        assert result["name"] == "test_method"
        assert result["version"] == "1.0.0"
        assert "config" in result
        assert isinstance(result["config"], dict)

    def test_to_dict_with_config(self) -> None:
        """Test to_dict includes method configuration."""
        method = ConcreteMethod(threshold=0.5, custom_param="value")
        result = method.to_dict()

        assert result["name"] == "test_method"
        assert result["config"]["threshold"] == 0.5
        assert result["config"]["custom_param"] == "value"


class TestMethodFromConfig:
    """Tests for the from_config class method."""

    def test_from_config_empty(self) -> None:
        """Test from_config with empty config dict."""
        method = ConcreteMethod.from_config({})

        assert isinstance(method, ConcreteMethod)
        assert method.name == "test_method"
        assert method.config == {}

    def test_from_config_with_params(self) -> None:
        """Test from_config with configuration parameters."""
        config = {"threshold": 0.8, "mode": "strict"}
        method = ConcreteMethod.from_config(config)

        assert isinstance(method, ConcreteMethod)
        assert method.config["threshold"] == 0.8
        assert method.config["mode"] == "strict"

    def test_from_config_roundtrip(self) -> None:
        """Test that to_dict -> from_config roundtrip preserves config."""
        original = ConcreteMethod(threshold=0.5, mode="relaxed")
        serialized = original.to_dict()
        restored = ConcreteMethod.from_config(serialized["config"])

        assert restored.config == original.config


class TestMethodRepr:
    """Tests for the __repr__ method."""

    def test_repr_format(self) -> None:
        """Test string representation includes class name and config."""
        method = ConcreteMethod(threshold=0.5)
        repr_str = repr(method)

        assert "ConcreteMethod" in repr_str
        assert "threshold" in repr_str
        assert "0.5" in repr_str

    def test_repr_empty_config(self) -> None:
        """Test string representation with empty config."""
        method = ConcreteMethod()
        repr_str = repr(method)

        assert "ConcreteMethod" in repr_str
        assert "{}" in repr_str


class TestMethodClassAttributes:
    """Tests for Method class attribute defaults."""

    def test_default_name(self) -> None:
        """Test default method name."""
        assert Method.name == "base_method"

    def test_default_version(self) -> None:
        """Test default version string."""
        assert Method.version == "0.1.0"

    def test_default_required_fields(self) -> None:
        """Test default required fields."""
        assert Method.required_fields == ["input", "output"]

    def test_subclass_can_override_attributes(self) -> None:
        """Test that subclasses can override class attributes."""
        assert ConcreteMethod.name == "test_method"
        assert ConcreteMethod.version == "1.0.0"
        assert ConcreteMethod.required_fields == ["input", "output", "expected"]
