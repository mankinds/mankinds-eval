"""Tests for evaluation runner module."""

from __future__ import annotations

from typing import Any

import pytest

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method
from mankinds_eval.scorer.runner import run_evaluation, run_single_sample


class MockMethod(Method):
    """A mock method for testing."""

    name: str = "MockMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(self, score: float = 1.0, **kwargs: Any) -> None:
        """Initialize mock method with configurable score."""
        super().__init__(**kwargs)
        self._score = score

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Return a mock result with configured score."""
        return MethodResult(
            method_name=self.name,
            score=self._score,
            passed=self._score >= 0.5,
            reason="Mock evaluation",
        )


class FailingMethod(Method):
    """A method that always raises an exception."""

    name: str = "FailingMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output"]

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Always raise an exception."""
        raise RuntimeError("Intentional test failure")


class MissingFieldMethod(Method):
    """A method that requires expected field."""

    name: str = "MissingFieldMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output", "expected"]

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate requiring expected field."""
        return MethodResult(
            method_name=self.name,
            score=1.0,
            passed=True,
            reason="Success",
        )


class TestRunEvaluation:
    """Tests for run_evaluation function."""

    @pytest.mark.asyncio
    async def test_basic_evaluation(self) -> None:
        """Test basic evaluation with single method and sample."""
        samples = [Sample(input="Hello", output="World")]
        methods = [MockMethod()]

        results = await run_evaluation(samples, methods)

        assert len(results) == 1
        assert results[0]["sample_index"] == 0
        assert results[0]["input"] == "Hello"
        assert results[0]["output"] == "World"
        assert "MockMethod" in results[0]["methods"]
        assert results[0]["methods"]["MockMethod"]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_multiple_samples(self) -> None:
        """Test evaluation with multiple samples."""
        samples = [
            Sample(input="Q1", output="A1"),
            Sample(input="Q2", output="A2"),
            Sample(input="Q3", output="A3"),
        ]
        methods = [MockMethod()]

        results = await run_evaluation(samples, methods)

        assert len(results) == 3
        # Results should be sorted by sample_index
        for i, result in enumerate(results):
            assert result["sample_index"] == i
            assert result["input"] == f"Q{i + 1}"

    @pytest.mark.asyncio
    async def test_multiple_methods(self) -> None:
        """Test evaluation with multiple methods."""
        samples = [Sample(input="Q", output="A")]
        methods = [MockMethod(score=1.0), MockMethod(score=0.5)]
        methods[1].name = "MockMethod2"  # Give second method different name

        results = await run_evaluation(samples, methods)

        assert len(results) == 1
        assert len(results[0]["methods"]) == 2
        assert "MockMethod" in results[0]["methods"]
        assert "MockMethod2" in results[0]["methods"]

    @pytest.mark.asyncio
    async def test_concurrency_limit(self) -> None:
        """Test that concurrency limit is respected."""
        samples = [Sample(input=f"Q{i}", output=f"A{i}") for i in range(10)]
        methods = [MockMethod()]

        # Should complete without error with concurrency limit
        results = await run_evaluation(samples, methods, concurrency=2)

        assert len(results) == 10

    @pytest.mark.asyncio
    async def test_error_capture_without_fail_on_error(self) -> None:
        """Test that errors are captured when fail_on_error is False."""
        samples = [Sample(input="Q", output="A")]
        methods = [FailingMethod()]

        results = await run_evaluation(samples, methods, fail_on_error=False)

        assert len(results) == 1
        # Error should be captured in results
        method_result = results[0]["methods"]["FailingMethod"]
        assert method_result.get("error") is not None
        assert "Intentional test failure" in method_result["error"]

    @pytest.mark.asyncio
    async def test_error_propagation_with_fail_on_error(self) -> None:
        """Test that errors propagate when fail_on_error is True."""
        samples = [Sample(input="Q", output="A")]
        methods = [FailingMethod()]

        with pytest.raises(RuntimeError, match="Intentional test failure"):
            await run_evaluation(samples, methods, fail_on_error=True)

    @pytest.mark.asyncio
    async def test_validation_error_capture(self) -> None:
        """Test that validation errors are captured."""
        # Sample without expected field
        samples = [Sample(input="Q", output="A")]
        # Method requires expected field
        methods = [MissingFieldMethod()]

        results = await run_evaluation(samples, methods, fail_on_error=False)

        assert len(results) == 1
        method_result = results[0]["methods"]["MissingFieldMethod"]
        assert method_result.get("error") is not None
        assert "expected" in method_result["error"].lower()

    @pytest.mark.asyncio
    async def test_expected_field_included(self) -> None:
        """Test that expected field is included in results."""
        samples = [Sample(input="Q", output="A", expected="A")]
        methods = [MockMethod()]

        results = await run_evaluation(samples, methods)

        assert results[0]["expected"] == "A"

    @pytest.mark.asyncio
    async def test_empty_samples(self) -> None:
        """Test evaluation with empty samples list."""
        samples: list[Sample] = []
        methods = [MockMethod()]

        results = await run_evaluation(samples, methods)

        assert results == []


class TestRunSingleSample:
    """Tests for run_single_sample function."""

    @pytest.mark.asyncio
    async def test_single_sample_basic(self) -> None:
        """Test evaluating a single sample."""
        sample = Sample(input="Hello", output="World")
        methods = [MockMethod()]

        result = await run_single_sample(sample, methods)

        assert result["sample_index"] == 0
        assert result["input"] == "Hello"
        assert result["output"] == "World"
        assert "MockMethod" in result["methods"]
        assert result["methods"]["MockMethod"]["score"] == 1.0

    @pytest.mark.asyncio
    async def test_single_sample_multiple_methods(self) -> None:
        """Test single sample with multiple methods."""
        sample = Sample(input="Q", output="A")
        method1 = MockMethod(score=1.0)
        method2 = MockMethod(score=0.7)
        method2.name = "MockMethod2"
        methods = [method1, method2]

        result = await run_single_sample(sample, methods)

        assert len(result["methods"]) == 2
        assert result["methods"]["MockMethod"]["score"] == 1.0
        assert result["methods"]["MockMethod2"]["score"] == 0.7

    @pytest.mark.asyncio
    async def test_single_sample_with_expected(self) -> None:
        """Test single sample with expected field."""
        sample = Sample(input="Q", output="A", expected="A")
        methods = [MockMethod()]

        result = await run_single_sample(sample, methods)

        assert result["expected"] == "A"

    @pytest.mark.asyncio
    async def test_single_sample_error_capture(self) -> None:
        """Test error capture for single sample."""
        sample = Sample(input="Q", output="A")
        methods = [FailingMethod()]

        result = await run_single_sample(sample, methods, fail_on_error=False)

        method_result = result["methods"]["FailingMethod"]
        assert method_result.get("error") is not None

    @pytest.mark.asyncio
    async def test_single_sample_error_propagation(self) -> None:
        """Test error propagation for single sample."""
        sample = Sample(input="Q", output="A")
        methods = [FailingMethod()]

        with pytest.raises(RuntimeError, match="Intentional test failure"):
            await run_single_sample(sample, methods, fail_on_error=True)
