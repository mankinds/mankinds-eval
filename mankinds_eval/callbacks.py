"""Callback system for evaluation hooks."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass, field
from typing import Any

from mankinds_eval.core import MethodResult, Sample


@dataclass
class EvaluationEvent:
    """Event data passed to callbacks.

    Attributes:
        event_type: Type of event (start, sample_complete, method_complete, end).
        sample: The sample being evaluated (if applicable).
        sample_index: Index of the current sample.
        total_samples: Total number of samples.
        method_name: Name of the method (if applicable).
        method_result: Result from the method (if applicable).
        metadata: Additional event metadata.
    """

    event_type: str
    sample: Sample | None = None
    sample_index: int | None = None
    total_samples: int | None = None
    method_name: str | None = None
    method_result: MethodResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class Callback:
    """Base class for evaluation callbacks.

    Implement this class to receive notifications during evaluation.
    All methods have default no-op implementations, override only what you need.
    """

    def on_evaluation_start(self, event: EvaluationEvent) -> None:
        """Called when evaluation begins.

        Args:
            event: Event with total_samples set.
        """
        pass

    def on_evaluation_end(self, event: EvaluationEvent) -> None:
        """Called when evaluation completes.

        Args:
            event: Event with metadata containing duration and summary.
        """
        pass

    def on_sample_start(self, event: EvaluationEvent) -> None:
        """Called before evaluating a sample.

        Args:
            event: Event with sample and sample_index set.
        """
        pass

    def on_sample_complete(self, event: EvaluationEvent) -> None:
        """Called after all methods complete for a sample.

        Args:
            event: Event with sample, sample_index, and metadata containing results.
        """
        pass

    def on_method_complete(self, event: EvaluationEvent) -> None:
        """Called after a method completes for a sample.

        Args:
            event: Event with sample, method_name, and method_result set.
        """
        pass

    def on_error(self, event: EvaluationEvent) -> None:
        """Called when an error occurs.

        Args:
            event: Event with metadata containing error information.
        """
        pass


class CallbackManager:
    """Manages multiple callbacks and dispatches events."""

    def __init__(self, callbacks: list[Callback] | None = None) -> None:
        """Initialize callback manager.

        Args:
            callbacks: List of callback instances to register.
        """
        self._callbacks: list[Callback] = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add a callback.

        Args:
            callback: Callback instance to add.
        """
        self._callbacks.append(callback)

    def remove(self, callback: Callback) -> None:
        """Remove a callback.

        Args:
            callback: Callback instance to remove.
        """
        self._callbacks.remove(callback)

    def clear(self) -> None:
        """Remove all callbacks."""
        self._callbacks.clear()

    def dispatch(self, event: EvaluationEvent) -> None:
        """Dispatch event to all callbacks.

        Args:
            event: Event to dispatch.
        """
        handler_map = {
            "evaluation_start": "on_evaluation_start",
            "evaluation_end": "on_evaluation_end",
            "sample_start": "on_sample_start",
            "sample_complete": "on_sample_complete",
            "method_complete": "on_method_complete",
            "error": "on_error",
        }

        handler_name = handler_map.get(event.event_type)
        if handler_name:
            for callback in self._callbacks:
                handler = getattr(callback, handler_name, None)
                if handler:
                    with contextlib.suppress(Exception):
                        handler(event)


class ProgressCallback(Callback):
    """Callback that displays a progress bar using rich."""

    def __init__(self, show_methods: bool = False) -> None:
        """Initialize progress callback.

        Args:
            show_methods: If True, show individual method progress.
        """
        self.show_methods = show_methods
        self._progress: Any = None
        self._task_id: Any = None

    def on_evaluation_start(self, event: EvaluationEvent) -> None:
        """Start the progress bar."""
        try:
            from rich.progress import (
                BarColumn,
                MofNCompleteColumn,
                Progress,
                SpinnerColumn,
                TextColumn,
                TimeElapsedColumn,
                TimeRemainingColumn,
            )

            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                "Evaluating",
                total=event.total_samples,
            )
        except ImportError:
            self._progress = None

    def on_evaluation_end(self, event: EvaluationEvent) -> None:
        """Stop the progress bar."""
        if self._progress is not None:
            self._progress.stop()
            self._progress = None

    def on_sample_complete(self, event: EvaluationEvent) -> None:
        """Update progress bar."""
        if self._progress is not None and self._task_id is not None:
            self._progress.advance(self._task_id)


class LoggingCallback(Callback):
    """Callback that logs evaluation progress."""

    def __init__(self, verbose: bool = False) -> None:
        """Initialize logging callback.

        Args:
            verbose: If True, log detailed information.
        """
        self.verbose = verbose

    def on_evaluation_start(self, event: EvaluationEvent) -> None:
        """Log evaluation start."""
        print(f"Starting evaluation of {event.total_samples} samples")

    def on_evaluation_end(self, event: EvaluationEvent) -> None:
        """Log evaluation end."""
        duration = event.metadata.get("duration_seconds", 0)
        print(f"Evaluation complete in {duration:.2f}s")

    def on_sample_complete(self, event: EvaluationEvent) -> None:
        """Log sample completion."""
        if self.verbose:
            idx = event.sample_index or 0
            print(f"Sample {idx + 1}/{event.total_samples} complete")

    def on_error(self, event: EvaluationEvent) -> None:
        """Log errors."""
        error = event.metadata.get("error", "Unknown error")
        print(f"Error: {error}")
