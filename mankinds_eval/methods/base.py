"""Base classes for evaluation methods."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mankinds_eval.core import MethodResult, Sample


class Method(ABC):
    """Base class for all evaluation methods.

    This abstract class defines the interface that all evaluation methods must implement.
    Subclasses should override the `evaluate` method to provide their specific evaluation logic.

    Class Attributes:
        name: Identifier for the method. Should be overridden by subclasses.
        version: Version string for the method implementation.
        required_fields: List of Sample fields required for this method to work.

    Example:
        class MyMethod(Method):
            name = "my_method"
            version = "1.0.0"
            required_fields = ["input", "output", "expected"]

            async def evaluate(self, sample: Sample) -> MethodResult:
                # Implementation here
                pass
    """

    name: str = "base_method"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(self, **kwargs: Any) -> None:
        """Initialize method with configuration.

        Subclasses should call super().__init__(**kwargs) and store their config.

        Args:
            **kwargs: Method-specific configuration options.
        """
        self.config: dict[str, Any] = kwargs

    @abstractmethod
    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate a single sample.

        This method must be implemented by all subclasses.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score, passed, reason, and metadata.
        """
        pass

    def validate_sample(self, sample: Sample) -> None:
        """Validate that sample has all required fields.

        Args:
            sample: The sample to validate.

        Raises:
            ValueError: If the sample is missing any required fields.
        """
        for field in self.required_fields:
            value = getattr(sample, field, None)
            if value is None:
                raise ValueError(f"Sample missing required field: {field}")

    async def safe_evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate with error handling.

        This method wraps `evaluate` with validation and error handling,
        returning a MethodResult with the error field set on failure.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with evaluation results, or error field on failure.
        """
        try:
            self.validate_sample(sample)
            return await self.evaluate(sample)
        except Exception as e:
            return MethodResult(
                method_name=self.name,
                score=None,
                passed=None,
                reason=None,
                error=str(e),
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize method configuration.

        Returns:
            Dictionary containing method name, version, and config.
        """
        return {
            "name": self.name,
            "version": self.version,
            "config": self.config,
        }

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Method:
        """Create method instance from config dict.

        Args:
            config: Configuration dictionary to pass to the constructor.

        Returns:
            A new Method instance.
        """
        return cls(**config)

    def __repr__(self) -> str:
        """Return string representation of the method.

        Returns:
            String representation including class name and config.
        """
        return f"{self.__class__.__name__}({self.config})"
