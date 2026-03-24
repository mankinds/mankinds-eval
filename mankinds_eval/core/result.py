"""MethodResult dataclass for representing evaluation results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MethodResult:
    """Represents the result of an evaluation method.

    Attributes:
        method_name: Name of the method that produced this result.
        score: Numeric score (can be None if error occurred).
        passed: Binary verdict (can be None if no threshold defined).
        reason: Human-readable explanation of the result.
        metadata: Method-specific metadata.
        error: Error message if evaluation failed.
    """

    method_name: str
    score: float | None
    passed: bool | None = None
    reason: str | None = None
    metadata: dict[str, Any] | None = None
    error: str | None = None

    @property
    def success(self) -> bool:
        """Check if the evaluation completed without errors.

        Returns:
            True if no error occurred, False otherwise.
        """
        return self.error is None

    def to_dict(self) -> dict[str, Any]:
        """Convert the result to a dictionary.

        Returns:
            Dictionary representation of the result.
        """
        result: dict[str, Any] = {
            "method_name": self.method_name,
            "score": self.score,
        }
        if self.passed is not None:
            result["passed"] = self.passed
        if self.reason is not None:
            result["reason"] = self.reason
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.error is not None:
            result["error"] = self.error
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MethodResult:
        """Create a MethodResult instance from a dictionary.

        Args:
            data: Dictionary containing result data.

        Returns:
            A new MethodResult instance.

        Raises:
            KeyError: If required fields are missing.
        """
        return cls(
            method_name=data["method_name"],
            score=data["score"],
            passed=data.get("passed"),
            reason=data.get("reason"),
            metadata=data.get("metadata"),
            error=data.get("error"),
        )
