"""Scorer class for assembling methods and running evaluations."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mankinds_eval import __version__
from mankinds_eval.callbacks import Callback, CallbackManager
from mankinds_eval.data import load_samples
from mankinds_eval.methods import Method
from mankinds_eval.output.schema import EvaluationResult, compute_summary
from mankinds_eval.scorer.config import load_scorer_config, resolve_methods
from mankinds_eval.scorer.runner import run_evaluation


class Scorer:
    """Assembles methods and runs evaluation on datasets.

    The Scorer is the main entry point for running evaluations. It takes a list
    of methods and applies them to samples, collecting results and computing
    summary statistics.

    Example:
        >>> from mankinds_eval import Scorer
        >>> from mankinds_eval.methods.heuristic import ExactMatch
        >>>
        >>> scorer = Scorer(
        ...     name="my_scorer",
        ...     methods=[ExactMatch()],
        ...     concurrency=10,
        ... )
        >>> result = scorer.run_sync([
        ...     {"input": "Hi", "output": "Hello", "expected": "Hello"},
        ... ])
    """

    def __init__(
        self,
        name: str,
        methods: list[Method],
        concurrency: int = 10,
        fail_on_error: bool = False,
        callbacks: list[Callback] | None = None,
        show_progress: bool = True,
    ) -> None:
        """Initialize the scorer.

        Args:
            name: Scorer name for identification in results.
            methods: List of Method instances to apply.
            concurrency: Maximum concurrent sample evaluations.
            fail_on_error: If True, raise on first error. If False, capture in results.
            callbacks: Optional list of Callback instances for progress notifications.
            show_progress: If True, show progress bar during evaluation.

        Raises:
            ValueError: If methods list is empty.
        """
        if not methods:
            raise ValueError("At least one method is required")

        self.name = name
        self.methods: list[Method] = methods
        self.concurrency = concurrency
        self.fail_on_error = fail_on_error
        self.callbacks: list[Callback] = callbacks or []
        self.show_progress = show_progress

    async def run(
        self,
        data: Any,
        concurrency: int | None = None,
        show_progress: bool | None = None,
    ) -> EvaluationResult:
        """Run evaluation on data.

        Args:
            data: Can be list[dict], path string, or any supported format
                (CSV, JSON, JSONL, HuggingFace Dataset).
            concurrency: Override instance concurrency for this run.
            show_progress: Override instance show_progress for this run.

        Returns:
            EvaluationResult containing meta, summary, and per-sample results.

        Raises:
            ValueError: If data format is unsupported.
            Exception: If fail_on_error is True and evaluation fails.
        """
        # Load samples from data source
        samples = load_samples(data)

        # Use provided values or fall back to instance defaults
        effective_concurrency = concurrency if concurrency is not None else self.concurrency
        effective_show_progress = show_progress if show_progress is not None else self.show_progress

        # Set up callback manager
        callback_manager = CallbackManager(self.callbacks.copy())

        # Run evaluation
        start_time = time.time()
        results = await run_evaluation(
            samples=samples,
            methods=self.methods,
            concurrency=effective_concurrency,
            fail_on_error=self.fail_on_error,
            callback_manager=callback_manager,
            show_progress=effective_show_progress,
        )
        duration = time.time() - start_time

        # Compute summary statistics
        method_names = [m.name for m in self.methods]
        summary = compute_summary(results, method_names)

        # Build metadata
        meta: dict[str, Any] = {
            "scorer_name": self.name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "version": __version__,
            "methods": [m.to_dict() for m in self.methods],
            "sample_count": len(samples),
            "duration_seconds": round(duration, 3),
            "concurrency": effective_concurrency,
        }

        return EvaluationResult(
            meta=meta,
            summary=summary,
            results=results,
        )

    def run_sync(
        self,
        data: Any,
        concurrency: int | None = None,
        show_progress: bool | None = None,
    ) -> EvaluationResult:
        """Synchronous wrapper for run().

        Args:
            data: Can be list[dict], path string, or any supported format.
            concurrency: Override instance concurrency for this run.
            show_progress: Override instance show_progress for this run.

        Returns:
            EvaluationResult containing meta, summary, and per-sample results.
        """
        return asyncio.run(self.run(data, concurrency, show_progress))

    def to_dict(self) -> dict[str, Any]:
        """Serialize scorer configuration.

        Returns:
            Dictionary containing scorer name, methods, and settings.
        """
        return {
            "name": self.name,
            "methods": [m.to_dict() for m in self.methods],
            "concurrency": self.concurrency,
            "fail_on_error": self.fail_on_error,
        }

    @classmethod
    def from_config(cls, config_path: str | Path) -> Scorer:
        """Load scorer from YAML or JSON config file.

        Config file format:
            name: my_scorer
            concurrency: 10
            fail_on_error: false
            methods:
              - type: heuristic.ExactMatch
              - type: heuristic.FuzzyMatch
                threshold: 0.8

        Args:
            config_path: Path to YAML or JSON configuration file.

        Returns:
            Configured Scorer instance.

        Raises:
            FileNotFoundError: If config file does not exist.
            ValueError: If config is invalid.
        """
        config = load_scorer_config(config_path)

        # Resolve method instances from config
        methods = resolve_methods(config["methods"])

        return cls(
            name=config["name"],
            methods=methods,
            concurrency=config.get("concurrency", 10),
            fail_on_error=config.get("fail_on_error", False),
        )

    def __repr__(self) -> str:
        """Return string representation of the scorer.

        Returns:
            String representation including name and method count.
        """
        method_names = [m.name for m in self.methods]
        return f"Scorer(name={self.name!r}, methods={method_names})"
