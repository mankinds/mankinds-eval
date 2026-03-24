"""Embeddings-based similarity evaluation method."""

from __future__ import annotations

from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method

# Check for optional dependencies
try:
    from sentence_transformers import SentenceTransformer

    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None  # type: ignore[misc, assignment]

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None  # type: ignore[assignment]


class EmbeddingsSimilarity(Method):
    """Compute semantic similarity using embeddings.

    This method uses sentence transformers to encode text and compute
    similarity between two pieces of text (e.g., output vs expected).

    Attributes:
        name: Method identifier.
        required_fields: Fields required in the Sample.
    """

    name: str = "EmbeddingsSimilarity"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        model: str | None = None,
        metric: str = "cosine",
        threshold: float | None = None,
        compare: str = "output_vs_expected",
        device: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the embeddings similarity method.

        Args:
            model: Model name/path. If None, uses default from config.
            metric: Similarity metric to use. One of "cosine", "euclidean", "dot_product".
            threshold: Score threshold for passed determination. If None, passed is always None.
            compare: Comparison mode. One of:
                - "output_vs_expected": Compare output to expected response.
                - "output_vs_context": Compare output to context.
                - "output_vs_input": Compare output to input.
            device: Device to use ("cuda", "cpu", "mps", or None for auto).
            **kwargs: Additional configuration passed to parent.
        """
        super().__init__(**kwargs)

        if metric not in ("cosine", "euclidean", "dot_product"):
            raise ValueError(
                f"Invalid metric: {metric}. Must be 'cosine', 'euclidean', or 'dot_product'"
            )

        valid_compare = ("output_vs_expected", "output_vs_context", "output_vs_input")
        if compare not in valid_compare:
            raise ValueError(f"Invalid compare mode: {compare}. Must be one of {valid_compare}")

        self.model_name = model
        self.metric = metric
        self.threshold = threshold
        self.compare = compare
        self.device = device
        self._model: Any = None  # Lazy loaded

        # Update required fields based on compare mode
        if compare == "output_vs_expected":
            self.required_fields = ["input", "output", "expected"]
        elif compare == "output_vs_context":
            self.required_fields = ["input", "output", "context"]
        else:
            self.required_fields = ["input", "output"]

        # Store config for serialization
        self.config = {
            "model": model,
            "metric": metric,
            "threshold": threshold,
            "compare": compare,
            "device": device,
            **kwargs,
        }

    def _get_device(self) -> str:
        """Determine the device to use.

        Returns:
            Device string for sentence-transformers.
        """
        if self.device is not None:
            return self.device

        if not TORCH_AVAILABLE:
            return "cpu"

        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _load_model(self) -> Any:
        """Lazy load sentence transformer model.

        Returns:
            Loaded SentenceTransformer model.

        Raises:
            ImportError: If sentence-transformers is not installed.
        """
        if self._model is None:
            if not SENTENCE_TRANSFORMERS_AVAILABLE:
                raise ImportError(
                    "sentence-transformers is required for EmbeddingsSimilarity. "
                    "Install it with: pip install mankinds-eval[ml]"
                )

            from mankinds_eval.config import get_model

            model_name = get_model("embeddings", self.model_name)
            device = self._get_device()
            self._model = SentenceTransformer(model_name, device=device)

        return self._model

    def _get_comparison_texts(self, sample: Sample) -> tuple[str, str]:
        """Get the two texts to compare based on compare mode.

        Args:
            sample: The sample containing the texts.

        Returns:
            Tuple of (text1, text2) to compare.

        Raises:
            ValueError: If required field is missing for the comparison mode.
        """
        if self.compare == "output_vs_expected":
            if sample.expected is None:
                raise ValueError(
                    "Sample missing 'expected' field required for output_vs_expected comparison"
                )
            return sample.output, sample.expected
        elif self.compare == "output_vs_context":
            if sample.context is None:
                raise ValueError(
                    "Sample missing 'context' field required for output_vs_context comparison"
                )
            return sample.output, sample.context
        else:  # output_vs_input
            return sample.output, sample.input

    def _compute_similarity(self, embedding1: Any, embedding2: Any) -> float:
        """Compute similarity between two embeddings.

        Args:
            embedding1: First embedding vector.
            embedding2: Second embedding vector.

        Returns:
            Similarity score.
        """
        if not TORCH_AVAILABLE:
            # Fallback to numpy-style operations
            import numpy as np

            e1 = np.array(embedding1)
            e2 = np.array(embedding2)

            if self.metric == "cosine":
                norm1 = np.linalg.norm(e1)
                norm2 = np.linalg.norm(e2)
                if norm1 == 0 or norm2 == 0:
                    return 0.0
                return float(np.dot(e1, e2) / (norm1 * norm2))
            elif self.metric == "euclidean":
                # Convert to similarity (inverse distance, normalized)
                distance = float(np.linalg.norm(e1 - e2))
                return 1.0 / (1.0 + distance)
            else:  # dot_product
                return float(np.dot(e1, e2))

        # Use torch for computation
        e1 = torch.tensor(embedding1) if not isinstance(embedding1, torch.Tensor) else embedding1
        e2 = torch.tensor(embedding2) if not isinstance(embedding2, torch.Tensor) else embedding2

        if self.metric == "cosine":
            similarity = torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0))
            return float(similarity.item())
        elif self.metric == "euclidean":
            distance = torch.dist(e1, e2)
            return float(1.0 / (1.0 + distance.item()))
        else:  # dot_product
            return float(torch.dot(e1.flatten(), e2.flatten()).item())

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate semantic similarity between texts.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with similarity score and evaluation details.
        """
        model = self._load_model()

        # Get texts to compare
        text1, text2 = self._get_comparison_texts(sample)

        # Encode both texts
        embeddings = model.encode([text1, text2], convert_to_tensor=TORCH_AVAILABLE)
        embedding1 = embeddings[0]
        embedding2 = embeddings[1]

        # Compute similarity
        score = self._compute_similarity(embedding1, embedding2)

        # Normalize score to 0-1 range for cosine (already in range)
        # For euclidean, the transformation already gives 0-1
        # For dot_product, scores can be arbitrary; we don't normalize

        # Determine passed status
        passed: bool | None = None
        if self.threshold is not None:
            passed = score >= self.threshold

        # Build reason
        from mankinds_eval.config import get_model

        actual_model = get_model("embeddings", self.model_name)
        reason = f"Similarity ({self.metric}): {score:.4f}"
        if self.threshold is not None:
            reason += f" (threshold: {self.threshold})"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "metric": self.metric,
                "model": actual_model,
                "compare": self.compare,
                "threshold": self.threshold,
            },
        )
