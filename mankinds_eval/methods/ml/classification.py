"""Zero-shot text classification evaluation method."""

from __future__ import annotations

from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method

# Check for optional dependencies
try:
    from transformers import pipeline as transformers_pipeline

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    transformers_pipeline = None  # type: ignore[misc, assignment]

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


class ZeroShotClassification(Method):
    """Zero-shot text classification.

    This method uses a zero-shot classification model to classify text
    into one of several provided labels without requiring task-specific
    training data.

    Attributes:
        name: Method identifier.
        required_fields: Fields required in the Sample.
    """

    name: str = "ZeroShotClassification"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    # Default model for zero-shot classification
    DEFAULT_MODEL: str = "facebook/bart-large-mnli"

    def __init__(
        self,
        labels: list[str],
        expected_label: str | None = None,
        confidence_threshold: float = 0.5,
        model: str = "facebook/bart-large-mnli",
        target: str = "output",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the zero-shot classification method.

        Args:
            labels: List of candidate labels to classify into.
            expected_label: Expected label for passed check. If None, just reports classification.
            confidence_threshold: Minimum confidence for passed determination.
            model: Model name/path for zero-shot classification.
            target: Which field to classify ("output" or "input").
            device: Device to use ("cuda", "cpu", "mps", or None for auto).
            **kwargs: Additional configuration passed to parent.
        """
        super().__init__(**kwargs)

        if not labels or len(labels) < 2:
            raise ValueError("At least 2 labels are required for classification")

        if target not in ("output", "input"):
            raise ValueError(f"Invalid target: {target}. Must be 'output' or 'input'")

        if expected_label is not None and expected_label not in labels:
            raise ValueError(
                f"Expected label '{expected_label}' must be one of the provided labels: {labels}"
            )

        self.labels = labels
        self.expected_label = expected_label
        self.confidence_threshold = confidence_threshold
        self.model_name = model
        self.target = target
        self.device = device
        self._pipeline: Any = None  # Lazy loaded

        # Store config for serialization
        self.config = {
            "labels": labels,
            "expected_label": expected_label,
            "confidence_threshold": confidence_threshold,
            "model": model,
            "target": target,
            "device": device,
            **kwargs,
        }

    def _get_device_id(self) -> int | str:
        """Determine the device ID for transformers pipeline.

        Returns:
            Device ID: -1 for CPU, 0+ for GPU, or "mps" for Apple Silicon.
        """
        if self.device is not None:
            if self.device == "cpu":
                return -1
            elif self.device == "cuda":
                return 0
            elif self.device == "mps":
                return "mps"
            else:
                return -1

        if not TORCH_AVAILABLE:
            return -1

        if torch.cuda.is_available():
            return 0
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return -1

    def _load_pipeline(self) -> Any:
        """Lazy load zero-shot classification pipeline.

        Returns:
            Loaded transformers pipeline.

        Raises:
            ImportError: If transformers is not installed.
        """
        if self._pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers is required for ZeroShotClassification. "
                    "Install it with: pip install mankinds-eval[ml]"
                )

            device = self._get_device_id()
            self._pipeline = transformers_pipeline(
                "zero-shot-classification",
                model=self.model_name,
                device=device,
            )

        return self._pipeline

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Classify the target text into one of the provided labels.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with classification results.
        """
        pipeline = self._load_pipeline()

        # Get target text
        text = sample.output if self.target == "output" else sample.input

        # Run zero-shot classification
        result = pipeline(text, candidate_labels=self.labels)

        # Extract results
        top_label = result["labels"][0]
        top_score = result["scores"][0]

        # Build scores dictionary
        scores_dict = dict(zip(result["labels"], result["scores"]))

        # Determine score to return
        if self.expected_label is not None:
            # Return confidence of expected label
            score = scores_dict.get(self.expected_label, 0.0)
        else:
            # Return confidence of top label
            score = top_score

        # Determine passed status
        passed: bool | None = None
        if self.expected_label is not None:
            label_matches = top_label == self.expected_label
            confidence_ok = scores_dict.get(self.expected_label, 0.0) >= self.confidence_threshold
            passed = label_matches and confidence_ok

        # Build reason
        top_score_pct = top_score * 100
        reason = f"Classified as '{top_label}' with {top_score_pct:.1f}% confidence"
        if self.expected_label is not None:
            expected_score = scores_dict.get(self.expected_label, 0.0)
            expected_score_pct = expected_score * 100
            if passed:
                reason += f" (matches expected: {self.expected_label})"
            else:
                reason += (
                    f" (expected: {self.expected_label}"
                    f" at {expected_score_pct:.1f}%,"
                    f" threshold:"
                    f" {self.confidence_threshold:.0%})"
                )

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "label": top_label,
                "scores": scores_dict,
                "model": self.model_name,
                "target": self.target,
                "expected_label": self.expected_label,
                "confidence_threshold": self.confidence_threshold,
            },
        )
