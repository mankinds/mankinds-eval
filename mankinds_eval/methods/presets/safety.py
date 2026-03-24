"""Safety evaluation preset combining key safety check methods."""

from __future__ import annotations

from typing import Any

from mankinds_eval.methods.base import Method


class SafetyScorer:
    """Factory for creating safety evaluation method sets.

    This preset combines key safety checks:
    - PIIDetection: Detects personal identifiable information
    - Toxicity: Detects toxic or harmful content
    - NoRefusal: Detects if the model refused to answer

    Example:
        >>> methods = SafetyScorer.create(check_pii=True, check_toxicity=True)
        >>> # Use with Scorer
        >>> scorer = Scorer(methods=methods)
        >>> results = await scorer.run(samples)
    """

    @staticmethod
    def create(
        check_pii: bool = True,
        check_toxicity: bool = True,
        check_refusal: bool = True,
        pii_threshold: float = 0.5,
        toxicity_threshold: float = 0.5,
        toxicity_model: str | None = None,
        device: str | None = None,
        **kwargs: Any,
    ) -> list[Method]:
        """Create a set of safety evaluation methods.

        Args:
            check_pii: Whether to include PII detection. Defaults to True.
            check_toxicity: Whether to include toxicity detection. Defaults to True.
            check_refusal: Whether to include refusal detection. Defaults to True.
            pii_threshold: Confidence threshold for PII detection. Defaults to 0.5.
            toxicity_threshold: Threshold for toxicity detection. Defaults to 0.5.
            toxicity_model: Model for toxicity detection. None uses default.
            device: Device for ML models ("cuda", "cpu", "mps", or None for auto).
            **kwargs: Additional arguments passed to methods.

        Returns:
            List of configured Method instances.

        Raises:
            ValueError: If no methods are selected.
            ImportError: If required ML dependencies are not installed.
        """
        methods: list[Method] = []

        if check_pii:
            # Import here to provide clear error if dependencies missing
            try:
                from mankinds_eval.methods.ml.ner import PIIDetection
            except ImportError as e:
                raise ImportError(
                    "PIIDetection requires ML dependencies. "
                    "Install with: pip install mankinds-eval[ml]"
                ) from e

            # Note: PIIDetection doesn't use confidence_threshold internally,
            # but we keep the parameter for API consistency and future use
            methods.append(
                PIIDetection(
                    device=device,
                    **kwargs,
                )
            )

        if check_toxicity:
            try:
                from mankinds_eval.methods.ml.toxicity import Toxicity
            except ImportError as e:
                raise ImportError(
                    "Toxicity requires ML dependencies. Install with: pip install mankinds-eval[ml]"
                ) from e

            methods.append(
                Toxicity(
                    model=toxicity_model,
                    threshold=toxicity_threshold,
                    device=device,
                    **kwargs,
                )
            )

        if check_refusal:
            from mankinds_eval.methods.heuristic.structured import NoRefusal

            methods.append(NoRefusal(**kwargs))

        if not methods:
            raise ValueError("At least one safety check must be enabled")

        return methods

    @staticmethod
    def get_check_names(
        check_pii: bool = True,
        check_toxicity: bool = True,
        check_refusal: bool = True,
    ) -> list[str]:
        """Get list of method names that will be included.

        Args:
            check_pii: Whether PII detection is included.
            check_toxicity: Whether toxicity detection is included.
            check_refusal: Whether refusal detection is included.

        Returns:
            List of method names.
        """
        names = []
        if check_pii:
            names.append("PIIDetection")
        if check_toxicity:
            names.append("Toxicity")
        if check_refusal:
            names.append("NoRefusal")
        return names
