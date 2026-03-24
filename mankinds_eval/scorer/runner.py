"""Async runner for evaluation methods."""

from __future__ import annotations

import asyncio
from typing import Any

from mankinds_eval.callbacks import CallbackManager, EvaluationEvent, ProgressCallback
from mankinds_eval.core import Sample
from mankinds_eval.methods import Method


async def run_evaluation(
    samples: list[Sample],
    methods: list[Method],
    concurrency: int = 10,
    fail_on_error: bool = False,
    callback_manager: CallbackManager | None = None,
    show_progress: bool = True,
) -> list[dict[str, Any]]:
    """Run all methods on all samples with controlled concurrency.

    Args:
        samples: List of Sample objects to evaluate.
        methods: List of Method instances to apply.
        concurrency: Maximum number of concurrent sample evaluations.
        fail_on_error: If True, raise on first error. If False, capture in results.
        callback_manager: Optional callback manager for progress notifications.
        show_progress: If True, show progress bar (requires rich).

    Returns:
        List of dicts, one per sample:
        {
            "sample_index": int,
            "input": str,
            "output": str,
            "expected": str | None,
            "methods": {
                "MethodName": MethodResult.to_dict(),
                ...
            }
        }

    Raises:
        Exception: If fail_on_error is True and any evaluation fails.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any]] = []

    # Set up callback manager with progress bar if enabled
    if callback_manager is None:
        callback_manager = CallbackManager()

    if show_progress:
        callback_manager.add(ProgressCallback())

    # Dispatch evaluation start
    callback_manager.dispatch(
        EvaluationEvent(
            event_type="evaluation_start",
            total_samples=len(samples),
        )
    )

    async def evaluate_sample(
        index: int,
        sample: Sample,
    ) -> dict[str, Any]:
        """Evaluate a single sample with all methods."""
        async with semaphore:
            # Dispatch sample start
            callback_manager.dispatch(
                EvaluationEvent(
                    event_type="sample_start",
                    sample=sample,
                    sample_index=index,
                    total_samples=len(samples),
                )
            )

            sample_result: dict[str, Any] = {
                "sample_index": index,
                "input": sample.input,
                "output": sample.output,
                "expected": sample.expected,
                "methods": {},
            }

            for method in methods:
                if fail_on_error:
                    # Let exceptions propagate
                    method_result = await method.evaluate(sample)
                else:
                    # Use safe_evaluate to capture errors
                    method_result = await method.safe_evaluate(sample)

                sample_result["methods"][method.name] = method_result.to_dict()

                # Dispatch method complete
                callback_manager.dispatch(
                    EvaluationEvent(
                        event_type="method_complete",
                        sample=sample,
                        sample_index=index,
                        method_name=method.name,
                        method_result=method_result,
                    )
                )

            # Dispatch sample complete
            callback_manager.dispatch(
                EvaluationEvent(
                    event_type="sample_complete",
                    sample=sample,
                    sample_index=index,
                    total_samples=len(samples),
                    metadata={"results": sample_result["methods"]},
                )
            )

            return sample_result

    # Create tasks for all samples
    tasks = [evaluate_sample(i, sample) for i, sample in enumerate(samples)]

    # Run all tasks concurrently with semaphore controlling parallelism
    if fail_on_error:
        # Use gather without return_exceptions to propagate errors
        results = await asyncio.gather(*tasks)
    else:
        # Use gather with return_exceptions to capture all results
        gathered = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results, converting exceptions to error entries
        for i, g_result in enumerate(gathered):
            if isinstance(g_result, BaseException):
                error_result: dict[str, Any] = {
                    "sample_index": i,
                    "input": samples[i].input,
                    "output": samples[i].output,
                    "expected": samples[i].expected,
                    "methods": {},
                }
                for method in methods:
                    error_result["methods"][method.name] = {
                        "method_name": method.name,
                        "score": None,
                        "error": str(g_result),
                    }
                results.append(error_result)
            else:
                results.append(g_result)

    # Sort results by sample_index to maintain order
    results.sort(key=lambda x: x["sample_index"])

    # Dispatch evaluation end
    callback_manager.dispatch(
        EvaluationEvent(
            event_type="evaluation_end",
            total_samples=len(samples),
        )
    )

    return results


async def run_single_sample(
    sample: Sample,
    methods: list[Method],
    fail_on_error: bool = False,
) -> dict[str, Any]:
    """Run all methods on a single sample.

    Args:
        sample: The sample to evaluate.
        methods: List of Method instances to apply.
        fail_on_error: If True, raise on first error. If False, capture in results.

    Returns:
        Dict with sample data and method results.
    """
    result: dict[str, Any] = {
        "sample_index": 0,
        "input": sample.input,
        "output": sample.output,
        "expected": sample.expected,
        "methods": {},
    }

    for method in methods:
        if fail_on_error:
            method_result = await method.evaluate(sample)
        else:
            method_result = await method.safe_evaluate(sample)

        result["methods"][method.name] = method_result.to_dict()

    return result
