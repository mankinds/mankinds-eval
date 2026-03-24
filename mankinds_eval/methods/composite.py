"""Composite evaluation methods for building complex evaluation pipelines."""

from __future__ import annotations

import copy
from typing import Any, Callable, Literal

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method

# Type alias for aggregation modes
AggregationMode = Literal["all", "any", "weighted", "sequential"]


class CompositeMethod(Method):
    """Combines multiple evaluation methods into a single composite evaluation.

    This method executes multiple child methods and aggregates their results
    according to the specified mode. It supports various aggregation strategies
    including requiring all methods to pass, any method to pass, weighted scoring,
    and sequential execution with sample enrichment.

    Attributes:
        name: Method identifier.
        version: Method version string.
        required_fields: List of required sample fields (union of child methods).
        methods: List of child methods to execute.
        mode: Aggregation mode for combining results.
        weights: Optional weights for weighted mode.
        threshold: Optional threshold for weighted mode pass/fail determination.
        stop_on_fail: Whether to stop on first failure in sequential mode.

    Example:
        >>> from mankinds_eval.methods.heuristic import FuzzyMatch
        >>> method1 = FuzzyMatch(threshold=0.8)
        >>> method2 = FuzzyMatch(threshold=0.9, algorithm="token_sort_ratio")
        >>> composite = CompositeMethod(
        ...     methods=[method1, method2],
        ...     mode="all"
        ... )
        >>> result = await composite.evaluate(sample)
    """

    name: str = "CompositeMethod"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        methods: list[Method],
        mode: AggregationMode = "all",
        weights: dict[str, float] | None = None,
        threshold: float | None = None,
        stop_on_fail: bool = False,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CompositeMethod.

        Args:
            methods: List of child methods to execute.
            mode: Aggregation mode for combining results. One of:
                - "all": passed=True only if all methods passed, score=average
                - "any": passed=True if any method passed, score=max
                - "weighted": score=weighted average, passed based on threshold
                - "sequential": like weighted but enriches sample between methods
            weights: Weights per method name for weighted/sequential modes.
                Keys should be method names, values should be positive floats.
                If None, equal weights are used.
            threshold: Threshold for weighted/sequential mode pass/fail.
                Score must be >= threshold to pass. Required for weighted mode.
            stop_on_fail: For sequential mode, stop execution if a method fails.
                Default False.
            name: Optional custom name override for this composite method.
            **kwargs: Additional configuration options.

        Raises:
            ValueError: If methods list is empty.
            ValueError: If mode is "weighted" or "sequential" and threshold is None.
            ValueError: If weights contains negative values.
            ValueError: If weights contains unknown method names.
        """
        super().__init__(
            mode=mode,
            weights=weights,
            threshold=threshold,
            stop_on_fail=stop_on_fail,
            **kwargs,
        )

        # Validate methods list
        if not methods:
            raise ValueError("methods list cannot be empty")

        # Validate threshold requirement for weighted modes
        if mode in ("weighted", "sequential") and threshold is None:
            raise ValueError(f"threshold is required for '{mode}' mode")

        # Validate weights if provided
        if weights is not None:
            method_names = {m.name for m in methods}
            for method_name, weight in weights.items():
                if weight < 0:
                    raise ValueError(
                        f"Weight for method '{method_name}' must be non-negative, got {weight}"
                    )
                if method_name not in method_names:
                    raise ValueError(
                        f"Unknown method name '{method_name}' in weights. "
                        f"Available methods: {', '.join(sorted(method_names))}"
                    )

        self.methods = methods
        self.mode = mode
        self.weights = weights
        self.threshold = threshold
        self.stop_on_fail = stop_on_fail

        # Override name if provided
        if name is not None:
            self.name = name

        # Compute required fields as union of all child methods
        self._update_required_fields()

    def _update_required_fields(self) -> None:
        """Update required_fields based on child methods."""
        all_fields: set[str] = set()
        for method in self.methods:
            all_fields.update(method.required_fields)
        self.required_fields = list(all_fields)

    def _get_weight(self, method_name: str) -> float:
        """Get weight for a method.

        Args:
            method_name: Name of the method.

        Returns:
            Weight value (defaults to 1.0 if not specified).
        """
        if self.weights is None:
            return 1.0
        return self.weights.get(method_name, 1.0)

    def _aggregate_all(self, results: list[MethodResult]) -> tuple[float, bool, str]:
        """Aggregate results using 'all' mode.

        Args:
            results: List of method results.

        Returns:
            Tuple of (score, passed, reason).
        """
        valid_scores = [r.score for r in results if r.score is not None]
        score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

        # All methods must pass (treat None as not passed)
        all_passed = all(r.passed is True for r in results)

        if all_passed:
            reason = f"All {len(results)} methods passed"
        else:
            failed_methods = [r.method_name for r in results if r.passed is not True]
            reason = f"Failed methods: {', '.join(failed_methods)}"

        return score, all_passed, reason

    def _aggregate_any(self, results: list[MethodResult]) -> tuple[float, bool, str]:
        """Aggregate results using 'any' mode.

        Args:
            results: List of method results.

        Returns:
            Tuple of (score, passed, reason).
        """
        valid_scores = [r.score for r in results if r.score is not None]
        score = max(valid_scores) if valid_scores else 0.0

        # Any method passing is sufficient
        any_passed = any(r.passed is True for r in results)

        if any_passed:
            passed_methods = [r.method_name for r in results if r.passed is True]
            reason = f"Passed methods: {', '.join(passed_methods)}"
        else:
            reason = f"No methods passed out of {len(results)}"

        return score, any_passed, reason

    def _aggregate_weighted(self, results: list[MethodResult]) -> tuple[float, bool | None, str]:
        """Aggregate results using 'weighted' mode.

        Args:
            results: List of method results.

        Returns:
            Tuple of (score, passed, reason).
        """
        total_weight = 0.0
        weighted_sum = 0.0

        for result in results:
            if result.score is not None:
                weight = self._get_weight(result.method_name)
                weighted_sum += result.score * weight
                total_weight += weight

        score = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Use threshold for pass/fail determination
        passed = score >= self.threshold if self.threshold is not None else None

        comparison = "above" if passed else "below"
        reason = f"Weighted score {score:.2%} {comparison} threshold {self.threshold:.2%}"

        return score, passed, reason

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate sample using all child methods and aggregate results.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with aggregated score, passed status, and sub_results
            in metadata.

        Example:
            >>> result = await composite.evaluate(sample)
            >>> print(result.metadata["sub_results"])  # Individual method results
        """
        results: list[MethodResult] = []
        sub_results: list[dict[str, Any]] = []

        if self.mode == "sequential":
            # Sequential mode: enrich sample with method_results after each method
            current_sample = copy.deepcopy(sample)
            if current_sample.method_results is None:
                current_sample.method_results = {}

            for method in self.methods:
                result = await method.safe_evaluate(current_sample)
                results.append(result)
                sub_results.append(result.to_dict())

                # Enrich sample with this method's result
                current_sample.method_results[method.name] = result.to_dict()

                # Stop on fail if configured
                if self.stop_on_fail and result.passed is False:
                    break
        else:
            # Non-sequential modes: run all methods on original sample
            for method in self.methods:
                result = await method.safe_evaluate(sample)
                results.append(result)
                sub_results.append(result.to_dict())

        # Aggregate based on mode
        score: float
        passed: bool | None
        reason: str
        if self.mode == "all":
            score, passed, reason = self._aggregate_all(results)
        elif self.mode == "any":
            score, passed, reason = self._aggregate_any(results)
        else:
            # "weighted" or "sequential" use weighted aggregation
            score, passed, reason = self._aggregate_weighted(results)

        # Build metadata
        metadata: dict[str, Any] = {
            "mode": self.mode,
            "num_methods": len(self.methods),
            "num_executed": len(results),
            "sub_results": sub_results,
        }

        if self.weights is not None:
            metadata["weights"] = self.weights
        if self.threshold is not None:
            metadata["threshold"] = self.threshold
        if self.mode == "sequential" and self.stop_on_fail:
            metadata["stopped_early"] = len(results) < len(self.methods)

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata=metadata,
        )


class ConditionalMethod(Method):
    """Conditionally executes one of two methods based on a condition.

    This method evaluates a condition function on the sample and executes
    either the if_true method or the if_false method based on the result.
    If if_false is not provided and the condition is false, a neutral result
    is returned.

    Attributes:
        name: Method identifier.
        version: Method version string.
        required_fields: List of required sample fields.
        condition: Callable that evaluates the condition on a sample.
        if_true: Method to run if condition is true.
        if_false: Optional method to run if condition is false.

    Example:
        >>> def has_context(sample: Sample) -> bool:
        ...     return sample.context is not None
        >>> conditional = ConditionalMethod(
        ...     condition=has_context,
        ...     if_true=context_aware_method,
        ...     if_false=basic_method,
        ... )
        >>> result = await conditional.evaluate(sample)
    """

    name: str = "ConditionalMethod"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        condition: Callable[[Sample], bool],
        if_true: Method,
        if_false: Method | None = None,
        name: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize ConditionalMethod.

        Args:
            condition: A callable that takes a Sample and returns a boolean.
                Used to determine which method to execute.
            if_true: Method to execute when condition returns True.
            if_false: Optional method to execute when condition returns False.
                If None, a neutral result is returned when condition is False.
            name: Optional custom name override for this conditional method.
            **kwargs: Additional configuration options.

        Raises:
            ValueError: If condition is not callable.
            ValueError: If if_true is not a Method instance.
            ValueError: If if_false is provided but not a Method instance.

        Example:
            >>> def is_long_output(sample: Sample) -> bool:
            ...     return len(sample.output) > 1000
            >>> method = ConditionalMethod(
            ...     condition=is_long_output,
            ...     if_true=detailed_evaluation_method,
            ...     if_false=quick_evaluation_method,
            ... )
        """
        super().__init__(**kwargs)

        # Validate condition
        if not callable(condition):
            raise ValueError("condition must be a callable")

        # Validate if_true
        if not isinstance(if_true, Method):
            raise ValueError("if_true must be a Method instance")

        # Validate if_false
        if if_false is not None and not isinstance(if_false, Method):
            raise ValueError("if_false must be a Method instance")

        self.condition = condition
        self.if_true = if_true
        self.if_false = if_false

        # Override name if provided
        if name is not None:
            self.name = name

        # Compute required fields as union of child methods
        self._update_required_fields()

    def _update_required_fields(self) -> None:
        """Update required_fields based on child methods."""
        all_fields: set[str] = set(self.if_true.required_fields)
        if self.if_false is not None:
            all_fields.update(self.if_false.required_fields)
        self.required_fields = list(all_fields)

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate sample using the appropriate method based on condition.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult from the executed method, or a neutral result if
            condition is False and no if_false method is provided.

        Example:
            >>> result = await conditional.evaluate(sample)
            >>> print(result.metadata["condition_result"])  # True or False
            >>> print(result.metadata["executed_method"])   # Method name
        """
        # Evaluate condition
        try:
            condition_result = self.condition(sample)
        except Exception as e:
            return MethodResult(
                method_name=self.name,
                score=None,
                passed=None,
                reason=None,
                error=f"Condition evaluation failed: {str(e)}",
            )

        # Execute appropriate method
        if condition_result:
            result = await self.if_true.safe_evaluate(sample)
            executed_method = self.if_true.name
        elif self.if_false is not None:
            result = await self.if_false.safe_evaluate(sample)
            executed_method = self.if_false.name
        else:
            # No if_false method, return neutral result
            return MethodResult(
                method_name=self.name,
                score=None,
                passed=None,
                reason="Condition was False and no if_false method provided",
                metadata={
                    "condition_result": False,
                    "executed_method": None,
                },
            )

        # Build enhanced metadata
        metadata: dict[str, Any] = {
            "condition_result": condition_result,
            "executed_method": executed_method,
        }

        # Include original result's metadata if present
        if result.metadata is not None:
            metadata["original_metadata"] = result.metadata

        return MethodResult(
            method_name=self.name,
            score=result.score,
            passed=result.passed,
            reason=result.reason,
            metadata=metadata,
            error=result.error,
        )
