"""RAG evaluation preset combining key metrics for RAG systems."""

from __future__ import annotations

from typing import Any

from mankinds_eval.methods.base import Method


class RAGScorer:
    """Factory for creating RAG evaluation method sets.

    This preset combines the key metrics for evaluating Retrieval-Augmented
    Generation (RAG) systems:
    - Faithfulness: Is the response grounded in the context?
    - AnswerRelevancy: Does the response address the question?
    - Coherence: Is the response well-structured and clear?

    Example:
        >>> methods = RAGScorer.create(provider="openai", model="gpt-4o-mini")
        >>> # Use with Scorer
        >>> scorer = Scorer(methods=methods)
        >>> results = await scorer.run(samples)
    """

    @staticmethod
    def create(
        provider: str = "openai",
        model: str | None = None,
        threshold: float = 0.7,
        faithfulness_weight: float = 0.4,
        relevancy_weight: float = 0.3,
        coherence_weight: float = 0.3,
        include_faithfulness: bool = True,
        include_relevancy: bool = True,
        include_coherence: bool = True,
        **kwargs: Any,
    ) -> list[Method]:
        """Create a set of RAG evaluation methods.

        Args:
            provider: LLM provider name. Defaults to "openai".
            model: Model name. If None, uses provider default.
            threshold: Score threshold for all methods. Defaults to 0.7.
            faithfulness_weight: Weight for faithfulness in aggregation.
                Defaults to 0.4.
            relevancy_weight: Weight for relevancy in aggregation.
                Defaults to 0.3.
            coherence_weight: Weight for coherence in aggregation.
                Defaults to 0.3.
            include_faithfulness: Whether to include Faithfulness method.
                Defaults to True.
            include_relevancy: Whether to include AnswerRelevancy method.
                Defaults to True.
            include_coherence: Whether to include Coherence method.
                Defaults to True.
            **kwargs: Additional arguments passed to all methods.

        Returns:
            List of configured Method instances.

        Raises:
            ValueError: If no methods are selected.
        """
        # Import here to avoid circular imports
        from mankinds_eval.methods.llm.coherence import Coherence
        from mankinds_eval.methods.llm.faithfulness import Faithfulness
        from mankinds_eval.methods.llm.relevancy import AnswerRelevancy

        methods: list[Method] = []

        if include_faithfulness:
            methods.append(
                Faithfulness(
                    threshold=threshold,
                    provider=provider,
                    model=model,
                    **kwargs,
                )
            )

        if include_relevancy:
            methods.append(
                AnswerRelevancy(
                    threshold=threshold,
                    provider=provider,
                    model=model,
                    **kwargs,
                )
            )

        if include_coherence:
            methods.append(
                Coherence(
                    threshold=threshold,
                    provider=provider,
                    model=model,
                    **kwargs,
                )
            )

        if not methods:
            raise ValueError("At least one method must be included")

        return methods

    @staticmethod
    def get_weights(
        faithfulness_weight: float = 0.4,
        relevancy_weight: float = 0.3,
        coherence_weight: float = 0.3,
    ) -> dict[str, float]:
        """Get method weights for score aggregation.

        Args:
            faithfulness_weight: Weight for Faithfulness. Defaults to 0.4.
            relevancy_weight: Weight for AnswerRelevancy. Defaults to 0.3.
            coherence_weight: Weight for Coherence. Defaults to 0.3.

        Returns:
            Dictionary mapping method names to weights.
        """
        return {
            "Faithfulness": faithfulness_weight,
            "AnswerRelevancy": relevancy_weight,
            "Coherence": coherence_weight,
        }
