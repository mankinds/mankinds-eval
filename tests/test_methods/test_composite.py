"""Tests for composite evaluation methods."""

from __future__ import annotations

from typing import Any

import pytest

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method
from mankinds_eval.methods.composite import CompositeMethod, ConditionalMethod
from mankinds_eval.scorer.config import resolve_method

# =============================================================================
# Mock Methods for Testing
# =============================================================================


class PassingMethod(Method):
    """A method that always passes with score 1.0."""

    name: str = "PassingMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output"]

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Return a passing result."""
        return MethodResult(
            method_name=self.name,
            score=1.0,
            passed=True,
            reason="Always passes",
            metadata={"mock": True},
        )


class FailingMethod(Method):
    """A method that always fails with score 0.0."""

    name: str = "FailingMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output"]

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Return a failing result."""
        return MethodResult(
            method_name=self.name,
            score=0.0,
            passed=False,
            reason="Always fails",
            metadata={"mock": True},
        )


class ScoreMethod(Method):
    """A method that returns a configurable score."""

    name: str = "ScoreMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        score: float = 0.5,
        passed: bool | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with a specific score.

        Args:
            score: The score to return (0.0-1.0).
            passed: The passed value to return. If None, derived from score >= 0.5.
            name: Optional custom name for this method instance.
            **kwargs: Additional configuration options.
        """
        super().__init__(score=score, passed=passed, **kwargs)
        self._score = score
        self._passed = passed if passed is not None else (score >= 0.5)
        if name is not None:
            self.name = name

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Return result with configured score."""
        return MethodResult(
            method_name=self.name,
            score=self._score,
            passed=self._passed,
            reason=f"Score: {self._score}",
            metadata={"configured_score": self._score},
        )


class EnrichingMethod(Method):
    """A method that checks for enriched sample data from previous methods."""

    name: str = "EnrichingMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(self, check_for: str | None = None, **kwargs: Any) -> None:
        """Initialize with optional check for previous method results.

        Args:
            check_for: Optional method name to check for in method_results.
            **kwargs: Additional configuration options.
        """
        super().__init__(check_for=check_for, **kwargs)
        self._check_for = check_for

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Return result indicating whether expected data was found."""
        found_previous = False
        if self._check_for and sample.method_results:
            found_previous = self._check_for in sample.method_results

        return MethodResult(
            method_name=self.name,
            score=1.0 if found_previous else 0.5,
            passed=True,
            reason=f"Found previous: {found_previous}",
            metadata={"found_previous": found_previous, "check_for": self._check_for},
        )


# =============================================================================
# CompositeMethod Tests - Mode: All
# =============================================================================


class TestCompositeMethodModeAll:
    """Tests for CompositeMethod with mode='all'."""

    @pytest.mark.asyncio
    async def test_composite_mode_all_pass(self) -> None:
        """Test that mode='all' returns passed=True when all methods pass."""
        method1 = PassingMethod()
        method2 = PassingMethod()
        composite = CompositeMethod(methods=[method1, method2], mode="all")

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.passed is True
        assert result.score == 1.0  # Average of 1.0 and 1.0
        assert "All 2 methods passed" in result.reason
        assert result.metadata is not None
        assert result.metadata["mode"] == "all"
        assert result.metadata["num_methods"] == 2
        assert result.metadata["num_executed"] == 2

    @pytest.mark.asyncio
    async def test_composite_mode_all_fail(self) -> None:
        """Test that mode='all' returns passed=False when one method fails."""
        method1 = PassingMethod()
        method2 = FailingMethod()
        composite = CompositeMethod(methods=[method1, method2], mode="all")

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.passed is False
        assert result.score == 0.5  # Average of 1.0 and 0.0
        assert "FailingMethod" in result.reason
        assert result.metadata is not None
        assert result.metadata["mode"] == "all"


# =============================================================================
# CompositeMethod Tests - Mode: Any
# =============================================================================


class TestCompositeMethodModeAny:
    """Tests for CompositeMethod with mode='any'."""

    @pytest.mark.asyncio
    async def test_composite_mode_any_pass(self) -> None:
        """Test that mode='any' returns passed=True when at least one method passes."""
        method1 = FailingMethod()
        method2 = PassingMethod()
        composite = CompositeMethod(methods=[method1, method2], mode="any")

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.passed is True
        assert result.score == 1.0  # Max of 0.0 and 1.0
        assert "PassingMethod" in result.reason
        assert result.metadata is not None
        assert result.metadata["mode"] == "any"

    @pytest.mark.asyncio
    async def test_composite_mode_any_fail(self) -> None:
        """Test that mode='any' returns passed=False when all methods fail."""
        method1 = FailingMethod()
        method2 = FailingMethod()
        composite = CompositeMethod(methods=[method1, method2], mode="any")

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.passed is False
        assert result.score == 0.0  # Max of 0.0 and 0.0
        assert "No methods passed" in result.reason
        assert result.metadata is not None
        assert result.metadata["mode"] == "any"


# =============================================================================
# CompositeMethod Tests - Mode: Weighted
# =============================================================================


class TestCompositeMethodModeWeighted:
    """Tests for CompositeMethod with mode='weighted'."""

    @pytest.mark.asyncio
    async def test_composite_mode_weighted(self) -> None:
        """Test weighted average score calculation."""
        method1 = ScoreMethod(score=0.8, name="Method1")
        method2 = ScoreMethod(score=0.4, name="Method2")

        # Equal weights (default)
        composite = CompositeMethod(
            methods=[method1, method2],
            mode="weighted",
            threshold=0.5,
        )

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        # Weighted average: (0.8 * 1.0 + 0.4 * 1.0) / 2.0 = 0.6
        assert result.score is not None
        assert abs(result.score - 0.6) < 0.01
        assert result.passed is True  # 0.6 >= 0.5 threshold
        assert result.metadata is not None
        assert result.metadata["mode"] == "weighted"
        assert result.metadata["threshold"] == 0.5

    @pytest.mark.asyncio
    async def test_composite_mode_weighted_with_custom_weights(self) -> None:
        """Test weighted average with custom weights per method."""
        method1 = ScoreMethod(score=0.8, name="Method1")
        method2 = ScoreMethod(score=0.4, name="Method2")

        # Method1 has weight 3, Method2 has weight 1
        composite = CompositeMethod(
            methods=[method1, method2],
            mode="weighted",
            weights={"Method1": 3.0, "Method2": 1.0},
            threshold=0.5,
        )

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        # Weighted average: (0.8 * 3.0 + 0.4 * 1.0) / 4.0 = 2.8 / 4.0 = 0.7
        assert result.score is not None
        assert abs(result.score - 0.7) < 0.01
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["weights"] == {"Method1": 3.0, "Method2": 1.0}

    @pytest.mark.asyncio
    async def test_composite_mode_weighted_with_threshold(self) -> None:
        """Test pass/fail determination based on threshold."""
        method1 = ScoreMethod(score=0.4, name="Method1")
        method2 = ScoreMethod(score=0.4, name="Method2")

        composite = CompositeMethod(
            methods=[method1, method2],
            mode="weighted",
            threshold=0.5,  # Average score 0.4 will be below threshold
        )

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.score is not None
        assert abs(result.score - 0.4) < 0.01
        assert result.passed is False  # 0.4 < 0.5 threshold
        assert "below threshold" in result.reason


# =============================================================================
# CompositeMethod Tests - Mode: Sequential
# =============================================================================


class TestCompositeMethodModeSequential:
    """Tests for CompositeMethod with mode='sequential'."""

    @pytest.mark.asyncio
    async def test_composite_mode_sequential(self) -> None:
        """Test that sequential mode enriches sample with method_results."""
        method1 = PassingMethod()
        method2 = EnrichingMethod(check_for="PassingMethod")

        composite = CompositeMethod(
            methods=[method1, method2],
            mode="sequential",
            threshold=0.5,
        )

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["mode"] == "sequential"
        assert len(result.metadata["sub_results"]) == 2

        # The second method should have found the first method's result
        second_result = result.metadata["sub_results"][1]
        assert second_result["metadata"]["found_previous"] is True

    @pytest.mark.asyncio
    async def test_composite_mode_sequential_stop_on_fail(self) -> None:
        """Test that stop_on_fail stops execution when a method fails."""
        method1 = FailingMethod()
        method2 = PassingMethod()
        method3 = PassingMethod()

        composite = CompositeMethod(
            methods=[method1, method2, method3],
            mode="sequential",
            threshold=0.5,
            stop_on_fail=True,
        )

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.metadata is not None
        # Should have stopped after first method failed
        assert result.metadata["num_executed"] == 1
        assert result.metadata["stopped_early"] is True
        assert len(result.metadata["sub_results"]) == 1

    @pytest.mark.asyncio
    async def test_composite_mode_sequential_no_stop_on_fail(self) -> None:
        """Test that without stop_on_fail, all methods execute even on failure."""
        method1 = FailingMethod()
        method2 = PassingMethod()

        composite = CompositeMethod(
            methods=[method1, method2],
            mode="sequential",
            threshold=0.5,
            stop_on_fail=False,
        )

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.metadata is not None
        # All methods should have executed
        assert result.metadata["num_executed"] == 2
        assert "stopped_early" not in result.metadata


# =============================================================================
# CompositeMethod Tests - Nesting and Configuration
# =============================================================================


class TestCompositeMethodNesting:
    """Tests for nested CompositeMethod configurations."""

    @pytest.mark.asyncio
    async def test_composite_nested(self) -> None:
        """Test composite containing another composite (nested composites)."""
        # Inner composite: both must pass
        inner_composite = CompositeMethod(
            methods=[PassingMethod(), PassingMethod()],
            mode="all",
            name="InnerComposite",
        )

        # Outer composite: any can pass
        outer_composite = CompositeMethod(
            methods=[FailingMethod(), inner_composite],
            mode="any",
            name="OuterComposite",
        )

        sample = Sample(input="test", output="result")
        result = await outer_composite.evaluate(sample)

        assert result.passed is True  # Inner composite passes, so outer passes
        assert result.method_name == "OuterComposite"
        assert result.metadata is not None
        assert len(result.metadata["sub_results"]) == 2

        # Check the inner composite result
        inner_result = result.metadata["sub_results"][1]
        assert inner_result["method_name"] == "InnerComposite"
        assert inner_result["passed"] is True


class TestCompositeMethodConfiguration:
    """Tests for CompositeMethod configuration and validation."""

    def test_composite_custom_name(self) -> None:
        """Test that custom name overrides default name."""
        composite = CompositeMethod(
            methods=[PassingMethod()],
            mode="all",
            name="MyCustomComposite",
        )

        assert composite.name == "MyCustomComposite"

    def test_composite_empty_methods_error(self) -> None:
        """Test that empty methods list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CompositeMethod(methods=[], mode="all")
        assert "methods list cannot be empty" in str(exc_info.value)

    def test_composite_weighted_requires_threshold(self) -> None:
        """Test that weighted mode requires threshold."""
        with pytest.raises(ValueError) as exc_info:
            CompositeMethod(
                methods=[PassingMethod()],
                mode="weighted",
                threshold=None,
            )
        assert "threshold is required" in str(exc_info.value)

    def test_composite_sequential_requires_threshold(self) -> None:
        """Test that sequential mode requires threshold."""
        with pytest.raises(ValueError) as exc_info:
            CompositeMethod(
                methods=[PassingMethod()],
                mode="sequential",
                threshold=None,
            )
        assert "threshold is required" in str(exc_info.value)

    def test_composite_negative_weight_error(self) -> None:
        """Test that negative weights raise ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CompositeMethod(
                methods=[PassingMethod()],
                mode="weighted",
                weights={"PassingMethod": -1.0},
                threshold=0.5,
            )
        assert "must be non-negative" in str(exc_info.value)

    def test_composite_unknown_method_in_weights_error(self) -> None:
        """Test that unknown method name in weights raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            CompositeMethod(
                methods=[PassingMethod()],
                mode="weighted",
                weights={"UnknownMethod": 1.0},
                threshold=0.5,
            )
        assert "Unknown method name" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_composite_sub_results_in_metadata(self) -> None:
        """Test that sub_results are included in metadata."""
        method1 = ScoreMethod(score=0.8, name="Method1")
        method2 = ScoreMethod(score=0.6, name="Method2")

        composite = CompositeMethod(methods=[method1, method2], mode="all")

        sample = Sample(input="test", output="result")
        result = await composite.evaluate(sample)

        assert result.metadata is not None
        assert "sub_results" in result.metadata
        assert len(result.metadata["sub_results"]) == 2

        # Verify sub_results structure
        sub1 = result.metadata["sub_results"][0]
        assert sub1["method_name"] == "Method1"
        assert sub1["score"] == 0.8

        sub2 = result.metadata["sub_results"][1]
        assert sub2["method_name"] == "Method2"
        assert sub2["score"] == 0.6

    def test_composite_required_fields_union(self) -> None:
        """Test that required_fields is the union of child methods' required fields."""
        method1 = PassingMethod()
        method1.required_fields = ["input", "output", "expected"]

        method2 = PassingMethod()
        method2.required_fields = ["input", "output", "context"]

        composite = CompositeMethod(methods=[method1, method2], mode="all")

        # Should contain union of all required fields
        assert set(composite.required_fields) == {"input", "output", "expected", "context"}


# =============================================================================
# ConditionalMethod Tests
# =============================================================================


class TestConditionalMethod:
    """Tests for ConditionalMethod."""

    @pytest.mark.asyncio
    async def test_conditional_true_branch(self) -> None:
        """Test that if_true method is executed when condition is True."""

        def always_true(sample: Sample) -> bool:
            return True

        if_true_method = ScoreMethod(score=0.9, name="TrueMethod")
        if_false_method = ScoreMethod(score=0.1, name="FalseMethod")

        conditional = ConditionalMethod(
            condition=always_true,
            if_true=if_true_method,
            if_false=if_false_method,
        )

        sample = Sample(input="test", output="result")
        result = await conditional.evaluate(sample)

        assert result.score == 0.9
        assert result.passed is True
        assert result.metadata is not None
        assert result.metadata["condition_result"] is True
        assert result.metadata["executed_method"] == "TrueMethod"

    @pytest.mark.asyncio
    async def test_conditional_false_branch(self) -> None:
        """Test that if_false method is executed when condition is False."""

        def always_false(sample: Sample) -> bool:
            return False

        if_true_method = ScoreMethod(score=0.9, name="TrueMethod")
        if_false_method = ScoreMethod(score=0.1, name="FalseMethod")

        conditional = ConditionalMethod(
            condition=always_false,
            if_true=if_true_method,
            if_false=if_false_method,
        )

        sample = Sample(input="test", output="result")
        result = await conditional.evaluate(sample)

        assert result.score == 0.1
        assert result.passed is False
        assert result.metadata is not None
        assert result.metadata["condition_result"] is False
        assert result.metadata["executed_method"] == "FalseMethod"

    @pytest.mark.asyncio
    async def test_conditional_no_false_method(self) -> None:
        """Test that neutral result is returned when condition is False and no if_false method."""

        def always_false(sample: Sample) -> bool:
            return False

        if_true_method = ScoreMethod(score=0.9, name="TrueMethod")

        conditional = ConditionalMethod(
            condition=always_false,
            if_true=if_true_method,
            if_false=None,
        )

        sample = Sample(input="test", output="result")
        result = await conditional.evaluate(sample)

        # Neutral result when no if_false method
        assert result.score is None
        assert result.passed is None
        assert "no if_false method provided" in result.reason
        assert result.metadata is not None
        assert result.metadata["condition_result"] is False
        assert result.metadata["executed_method"] is None

    def test_conditional_custom_name(self) -> None:
        """Test that custom name overrides default name."""

        def condition(sample: Sample) -> bool:
            return True

        conditional = ConditionalMethod(
            condition=condition,
            if_true=PassingMethod(),
            name="MyCustomConditional",
        )

        assert conditional.name == "MyCustomConditional"

    @pytest.mark.asyncio
    async def test_conditional_metadata_contains_condition_info(self) -> None:
        """Test that metadata contains condition evaluation information."""

        def has_context(sample: Sample) -> bool:
            return sample.context is not None

        conditional = ConditionalMethod(
            condition=has_context,
            if_true=PassingMethod(),
            if_false=FailingMethod(),
        )

        # Sample with context
        sample_with_context = Sample(
            input="test",
            output="result",
            context="Some context",
        )
        result = await conditional.evaluate(sample_with_context)

        assert result.metadata is not None
        assert result.metadata["condition_result"] is True
        assert result.metadata["executed_method"] == "PassingMethod"
        assert "original_metadata" in result.metadata

        # Sample without context
        sample_without_context = Sample(input="test", output="result")
        result = await conditional.evaluate(sample_without_context)

        assert result.metadata is not None
        assert result.metadata["condition_result"] is False
        assert result.metadata["executed_method"] == "FailingMethod"

    @pytest.mark.asyncio
    async def test_conditional_condition_error_handling(self) -> None:
        """Test that condition evaluation errors are handled gracefully."""

        def failing_condition(sample: Sample) -> bool:
            raise RuntimeError("Condition evaluation failed!")

        conditional = ConditionalMethod(
            condition=failing_condition,
            if_true=PassingMethod(),
        )

        sample = Sample(input="test", output="result")
        result = await conditional.evaluate(sample)

        assert result.score is None
        assert result.passed is None
        assert result.error is not None
        assert "Condition evaluation failed" in result.error

    def test_conditional_invalid_condition_error(self) -> None:
        """Test that non-callable condition raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            ConditionalMethod(
                condition="not a callable",  # type: ignore
                if_true=PassingMethod(),
            )
        assert "condition must be a callable" in str(exc_info.value)

    def test_conditional_invalid_if_true_error(self) -> None:
        """Test that non-Method if_true raises ValueError."""

        def condition(sample: Sample) -> bool:
            return True

        with pytest.raises(ValueError) as exc_info:
            ConditionalMethod(
                condition=condition,
                if_true="not a method",  # type: ignore
            )
        assert "if_true must be a Method instance" in str(exc_info.value)

    def test_conditional_invalid_if_false_error(self) -> None:
        """Test that non-Method if_false raises ValueError."""

        def condition(sample: Sample) -> bool:
            return True

        with pytest.raises(ValueError) as exc_info:
            ConditionalMethod(
                condition=condition,
                if_true=PassingMethod(),
                if_false="not a method",  # type: ignore
            )
        assert "if_false must be a Method instance" in str(exc_info.value)

    def test_conditional_required_fields_union(self) -> None:
        """Test that required_fields is the union of both branch methods."""

        def condition(sample: Sample) -> bool:
            return True

        if_true = PassingMethod()
        if_true.required_fields = ["input", "output", "expected"]

        if_false = PassingMethod()
        if_false.required_fields = ["input", "output", "context"]

        conditional = ConditionalMethod(
            condition=condition,
            if_true=if_true,
            if_false=if_false,
        )

        # Should contain union of all required fields
        assert set(conditional.required_fields) == {"input", "output", "expected", "context"}


# =============================================================================
# Config Resolution Tests
# =============================================================================


class TestConfigResolution:
    """Tests for resolving composite methods from config."""

    def test_resolve_composite_from_config(self) -> None:
        """Test resolving CompositeMethod from config dictionary."""
        config = {
            "type": "composite",
            "mode": "all",
            "methods": [
                {"type": "heuristic.fuzzy.FuzzyMatch", "threshold": 0.8},
                {"type": "heuristic.pattern.ExactMatch"},
            ],
        }

        method = resolve_method(config)

        assert isinstance(method, CompositeMethod)
        assert method.mode == "all"
        assert len(method.methods) == 2

    def test_resolve_nested_composite_from_config(self) -> None:
        """Test resolving nested CompositeMethod configurations."""
        config = {
            "type": "composite",
            "mode": "any",
            "methods": [
                {
                    "type": "composite",
                    "mode": "all",
                    "methods": [
                        {"type": "heuristic.fuzzy.FuzzyMatch", "threshold": 0.9},
                        {"type": "heuristic.pattern.ExactMatch"},
                    ],
                },
                {"type": "heuristic.text_metrics.TextLength", "min_length": 10},
            ],
        }

        method = resolve_method(config)

        assert isinstance(method, CompositeMethod)
        assert method.mode == "any"
        assert len(method.methods) == 2

        # First method should be a nested composite
        inner_composite = method.methods[0]
        assert isinstance(inner_composite, CompositeMethod)
        assert inner_composite.mode == "all"
        assert len(inner_composite.methods) == 2

    def test_resolve_composite_with_weights_from_config(self) -> None:
        """Test resolving CompositeMethod with weights from config."""
        config = {
            "type": "composite",
            "mode": "weighted",
            "threshold": 0.7,
            "weights": {
                "FuzzyMatch": 2.0,
                "ExactMatch": 1.0,
            },
            "methods": [
                {"type": "heuristic.fuzzy.FuzzyMatch", "threshold": 0.8},
                {"type": "heuristic.pattern.ExactMatch"},
            ],
        }

        method = resolve_method(config)

        assert isinstance(method, CompositeMethod)
        assert method.mode == "weighted"
        assert method.threshold == 0.7
        assert method.weights == {"FuzzyMatch": 2.0, "ExactMatch": 1.0}

    def test_resolve_composite_with_explicit_type(self) -> None:
        """Test resolving with explicit composite.CompositeMethod type."""
        config = {
            "type": "composite.CompositeMethod",
            "mode": "all",
            "methods": [
                {"type": "heuristic.pattern.ExactMatch"},
            ],
        }

        method = resolve_method(config)

        assert isinstance(method, CompositeMethod)
        assert method.mode == "all"

    def test_resolve_composite_invalid_methods_type(self) -> None:
        """Test that invalid 'methods' type raises ValueError."""
        config = {
            "type": "composite",
            "mode": "all",
            "methods": "not a list",  # Should be a list
        }

        with pytest.raises(ValueError) as exc_info:
            resolve_method(config)
        assert "must be a list" in str(exc_info.value)
