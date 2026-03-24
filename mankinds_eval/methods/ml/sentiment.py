"""Sentiment analysis evaluation method."""

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


class SentimentAnalysis(Method):
    """Analyze sentiment of text.

    This method uses a sentiment analysis model to classify the sentiment
    of the output (or input) text and optionally checks if it matches
    an expected sentiment.

    Attributes:
        name: Method identifier.
        required_fields: Fields required in the Sample.
    """

    name: str = "SentimentAnalysis"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        model: str | None = None,
        expected_sentiment: str | None = None,
        confidence_threshold: float = 0.5,
        target: str = "output",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the sentiment analysis method.

        Args:
            model: Model name/path. If None, uses default from config.
            expected_sentiment: Expected sentiment label ("positive", "negative", "neutral").
                If None, method just reports detected sentiment without pass/fail.
            confidence_threshold: Minimum confidence for passed determination.
            target: Which field to analyze ("output" or "input").
            device: Device to use ("cuda", "cpu", "mps", or None for auto).
            **kwargs: Additional configuration passed to parent.
        """
        super().__init__(**kwargs)

        if target not in ("output", "input"):
            raise ValueError(f"Invalid target: {target}. Must be 'output' or 'input'")

        self.model_name = model
        self.expected_sentiment = expected_sentiment.lower() if expected_sentiment else None
        self.confidence_threshold = confidence_threshold
        self.target = target
        self.device = device
        self._pipeline: Any = None  # Lazy loaded

        # Store config for serialization
        self.config = {
            "model": model,
            "expected_sentiment": expected_sentiment,
            "confidence_threshold": confidence_threshold,
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
        """Lazy load sentiment analysis pipeline.

        Returns:
            Loaded transformers pipeline.

        Raises:
            ImportError: If transformers is not installed.
        """
        if self._pipeline is None:
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "transformers is required for SentimentAnalysis. "
                    "Install it with: pip install mankinds-eval[ml]"
                )

            from mankinds_eval.config import get_model

            model_name = get_model("sentiment", self.model_name)
            device = self._get_device_id()
            self._pipeline = transformers_pipeline(
                "sentiment-analysis",
                model=model_name,
                device=device,
            )

        return self._pipeline

    def _normalize_label(self, label: str) -> str:
        """Normalize sentiment label to standard format.

        Args:
            label: Raw label from model.

        Returns:
            Normalized label (lowercase).
        """
        # Handle common label formats
        label_lower = label.lower()

        # Map common variations
        if label_lower in ("pos", "positive", "label_2"):
            return "positive"
        elif label_lower in ("neg", "negative", "label_0"):
            return "negative"
        elif label_lower in ("neu", "neutral", "label_1"):
            return "neutral"

        return label_lower

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate sentiment of the target text.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with sentiment analysis results.
        """
        pipeline = self._load_pipeline()

        # Get target text
        text = sample.output if self.target == "output" else sample.input

        # Run sentiment analysis
        result = pipeline(text, truncation=True)

        # Handle list result (pipeline returns list for single input)
        if isinstance(result, list):
            result = result[0]

        raw_label = result["label"]
        confidence = result["score"]

        # Normalize the label
        detected_sentiment = self._normalize_label(raw_label)

        # Determine passed status
        passed: bool | None = None
        if self.expected_sentiment is not None:
            sentiment_matches = detected_sentiment == self.expected_sentiment
            confidence_ok = confidence >= self.confidence_threshold
            passed = sentiment_matches and confidence_ok

        # Build reason
        confidence_pct = confidence * 100
        reason = f"Detected {detected_sentiment} sentiment with {confidence_pct:.1f}% confidence"
        if self.expected_sentiment is not None:
            if passed:
                reason += f" (matches expected: {self.expected_sentiment})"
            else:
                reason += (
                    f" (expected:"
                    f" {self.expected_sentiment},"
                    f" threshold:"
                    f" {self.confidence_threshold:.0%})"
                )

        from mankinds_eval.config import get_model

        actual_model = get_model("sentiment", self.model_name)

        return MethodResult(
            method_name=self.name,
            score=confidence,
            passed=passed,
            reason=reason,
            metadata={
                "label": detected_sentiment,
                "raw_label": raw_label,
                "confidence": confidence,
                "model": actual_model,
                "target": self.target,
                "expected_sentiment": self.expected_sentiment,
            },
        )
