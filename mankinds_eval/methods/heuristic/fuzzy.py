"""Fuzzy string matching evaluation methods using rapidfuzz."""

from __future__ import annotations

from typing import Any, Callable, Literal

from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler, Levenshtein

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method

# Type aliases for clarity
AlgorithmName = Literal[
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "levenshtein",
    "jaro_winkler",
]
TargetName = Literal["expected", "input", "context"]

# Valid algorithm names
VALID_ALGORITHMS: set[str] = {
    "ratio",
    "partial_ratio",
    "token_sort_ratio",
    "token_set_ratio",
    "levenshtein",
    "jaro_winkler",
}

# Valid target names
VALID_TARGETS: set[str] = {"expected", "input", "context"}


class FuzzyMatch(Method):
    """Fuzzy string matching between output and expected/target.

    This method computes string similarity between the sample's output
    and a target field (expected, input, or context) using various
    fuzzy matching algorithms from the rapidfuzz library.

    Attributes:
        name: Method identifier.
        version: Method version string.
        required_fields: List of required sample fields.
        threshold: Score threshold for determining if evaluation passed.
        algorithm: The fuzzy matching algorithm to use.
        target: The sample field to compare output against.
        case_sensitive: Whether comparison is case-sensitive.

    Example:
        >>> method = FuzzyMatch(threshold=0.8, algorithm="ratio")
        >>> result = await method.evaluate(sample)
    """

    name: str = "FuzzyMatch"
    version: str = "0.1.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        threshold: float = 0.8,
        algorithm: str = "ratio",
        target: str = "expected",
        case_sensitive: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize FuzzyMatch method.

        Args:
            threshold: Score threshold for passed (0.0 to 1.0). Default 0.8.
            algorithm: Fuzzy matching algorithm to use. One of:
                - "ratio": Basic ratio (fuzz.ratio)
                - "partial_ratio": Partial string matching (fuzz.partial_ratio)
                - "token_sort_ratio": Token sorted ratio (fuzz.token_sort_ratio)
                - "token_set_ratio": Token set ratio (fuzz.token_set_ratio)
                - "levenshtein": Normalized Levenshtein (Levenshtein.normalized_similarity)
                - "jaro_winkler": Jaro-Winkler (JaroWinkler.normalized_similarity)
            target: Sample field to compare output against.
                One of: "expected", "input", "context". Default "expected".
            case_sensitive: Whether comparison is case-sensitive. Default False.
            **kwargs: Additional configuration options.

        Raises:
            ValueError: If algorithm is not valid.
            ValueError: If target is not valid.
            ValueError: If threshold is not between 0.0 and 1.0.
        """
        super().__init__(
            threshold=threshold,
            algorithm=algorithm,
            target=target,
            case_sensitive=case_sensitive,
            **kwargs,
        )

        # Validate threshold
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"threshold must be between 0.0 and 1.0, got {threshold}")

        # Validate algorithm
        if algorithm not in VALID_ALGORITHMS:
            raise ValueError(
                f"Invalid algorithm '{algorithm}'. "
                f"Must be one of: {', '.join(sorted(VALID_ALGORITHMS))}"
            )

        # Validate target
        if target not in VALID_TARGETS:
            raise ValueError(
                f"Invalid target '{target}'. Must be one of: {', '.join(sorted(VALID_TARGETS))}"
            )

        self.threshold = threshold
        self.algorithm = algorithm
        self.target = target
        self.case_sensitive = case_sensitive

        # Update required_fields based on target
        self._update_required_fields()

        # Initialize algorithm function
        self._algorithm_func = self._get_algorithm_func()

    def _update_required_fields(self) -> None:
        """Update required_fields based on target parameter."""
        base_fields = ["input", "output"]
        if self.target == "expected":
            self.required_fields = base_fields + ["expected"]
        elif self.target == "context":
            self.required_fields = base_fields + ["context"]
        else:
            # target == "input" - no additional fields needed
            self.required_fields = base_fields

    def _get_algorithm_func(self) -> Callable[[str, str], float]:
        """Get the algorithm function based on configuration.

        Returns:
            A callable that takes two strings and returns a similarity score (0.0 to 1.0).
        """
        # Map algorithm names to functions
        # Note: rapidfuzz fuzz functions return 0-100, need to normalize to 0-1
        algorithm_map: dict[str, Callable[[str, str], float]] = {
            "ratio": lambda s1, s2: fuzz.ratio(s1, s2) / 100.0,
            "partial_ratio": lambda s1, s2: fuzz.partial_ratio(s1, s2) / 100.0,
            "token_sort_ratio": lambda s1, s2: fuzz.token_sort_ratio(s1, s2) / 100.0,
            "token_set_ratio": lambda s1, s2: fuzz.token_set_ratio(s1, s2) / 100.0,
            "levenshtein": Levenshtein.normalized_similarity,
            "jaro_winkler": JaroWinkler.normalized_similarity,
        }
        return algorithm_map[self.algorithm]

    def _get_target_text(self, sample: Sample) -> str:
        """Get the target text from the sample based on target parameter.

        Args:
            sample: The sample to get target text from.

        Returns:
            The target text string.

        Raises:
            ValueError: If target field is None.
        """
        if self.target == "expected":
            if sample.expected is None:
                raise ValueError("Sample missing required field: expected")
            return sample.expected
        elif self.target == "context":
            if sample.context is None:
                raise ValueError("Sample missing required field: context")
            return sample.context
        else:
            # target == "input"
            return sample.input

    def _compute_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two strings.

        Args:
            text1: First string.
            text2: Second string.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Apply case normalization if not case sensitive
        if not self.case_sensitive:
            text1 = text1.lower()
            text2 = text2.lower()

        return self._algorithm_func(text1, text2)

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate fuzzy match between output and target.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with similarity score, passed status, and details.
        """
        # Get target text
        target_text = self._get_target_text(sample)

        # Compute similarity
        score = self._compute_similarity(sample.output, target_text)

        # Determine if passed
        passed = score >= self.threshold

        # Generate reason
        comparison = "above" if passed else "below"
        reason = f"Similarity {score:.2%} {comparison} threshold {self.threshold:.2%}"

        # Build metadata
        metadata = {
            "algorithm": self.algorithm,
            "threshold": self.threshold,
            "target": self.target,
            "case_sensitive": self.case_sensitive,
        }

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata=metadata,
        )
