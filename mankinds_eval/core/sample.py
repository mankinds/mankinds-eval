"""Sample dataclass for representing AI evaluation inputs and outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    """Represents a single sample for AI evaluation.

    Attributes:
        input: The prompt or question given to the AI.
        output: The AI response to evaluate.
        expected: Optional expected response for comparison.
        context: Optional context (e.g., RAG retrieved documents).
        conversation: Optional conversation history.
            Format: [{"role": "user", "content": "..."}, ...]
        metadata: Optional user-provided metadata.
        method_results: Optional dictionary storing results from evaluation methods.
    """

    input: str
    output: str
    expected: str | None = None
    context: str | None = None
    conversation: list[dict[str, Any]] | None = None
    metadata: dict[str, Any] | None = None
    method_results: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate sample data after initialization."""
        if not isinstance(self.input, str) or not self.input.strip():
            raise ValueError("input must be a non-empty string")
        if not isinstance(self.output, str) or not self.output.strip():
            raise ValueError("output must be a non-empty string")

        if self.conversation is not None:
            self._validate_conversation(self.conversation)

    @staticmethod
    def _validate_conversation(conversation: list[dict[str, Any]]) -> None:
        """Validate conversation format.

        Args:
            conversation: List of conversation messages to validate.

        Raises:
            ValueError: If conversation format is invalid.
        """
        if not isinstance(conversation, list):
            raise ValueError("conversation must be a list")

        for i, message in enumerate(conversation):
            if not isinstance(message, dict):
                raise ValueError(f"conversation[{i}] must be a dict")
            if "role" not in message:
                raise ValueError(f"conversation[{i}] missing required key 'role'")
            if "content" not in message:
                raise ValueError(f"conversation[{i}] missing required key 'content'")

    def to_dict(self) -> dict[str, Any]:
        """Convert the sample to a dictionary.

        Returns:
            Dictionary representation of the sample.
        """
        result: dict[str, Any] = {
            "input": self.input,
            "output": self.output,
        }
        if self.expected is not None:
            result["expected"] = self.expected
        if self.context is not None:
            result["context"] = self.context
        if self.conversation is not None:
            result["conversation"] = self.conversation
        if self.metadata is not None:
            result["metadata"] = self.metadata
        if self.method_results is not None:
            result["method_results"] = self.method_results
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Sample:
        """Create a Sample instance from a dictionary.

        Args:
            data: Dictionary containing sample data.

        Returns:
            A new Sample instance.

        Raises:
            ValueError: If required fields are missing or invalid.
        """
        return cls(
            input=data["input"],
            output=data["output"],
            expected=data.get("expected"),
            context=data.get("context"),
            conversation=data.get("conversation"),
            metadata=data.get("metadata"),
            method_results=data.get("method_results"),
        )
