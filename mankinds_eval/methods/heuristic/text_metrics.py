"""Text metrics evaluation methods for translation and summarization quality."""

from __future__ import annotations

import re
from typing import Any

import sacrebleu
from rouge_score import rouge_scorer

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method


class BLEU(Method):
    """BLEU score for translation/generation quality.

    Uses sacrebleu for computing sentence-level BLEU scores.
    The score is normalized from 0-100 to 0-1 for consistency.

    Attributes:
        name: Method identifier.
        required_fields: Sample fields required for evaluation.
    """

    name: str = "BLEU"
    required_fields: list[str] = ["input", "output", "expected"]

    def __init__(
        self,
        threshold: float | None = None,
        smooth_method: str = "exp",
        **kwargs: Any,
    ) -> None:
        """Initialize BLEU scorer.

        Args:
            threshold: Optional threshold for pass/fail determination (0-1 scale).
            smooth_method: Smoothing method for sacrebleu. Options: "none", "floor",
                "add-k", "exp". Default is "exp".
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.smooth_method = smooth_method
        self.config.update(
            {
                "threshold": threshold,
                "smooth_method": smooth_method,
            }
        )

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate BLEU score for a sample.

        Args:
            sample: The sample to evaluate with output and expected fields.

        Returns:
            MethodResult with BLEU score normalized to 0-1.
        """
        hypothesis = sample.output
        reference = sample.expected

        if reference is None:
            return MethodResult(
                method_name=self.name,
                score=None,
                passed=None,
                reason="Expected field is required for BLEU evaluation",
                error="Missing expected field",
            )

        # sacrebleu.sentence_bleu expects hypothesis as string and references as list
        bleu_result = sacrebleu.sentence_bleu(
            hypothesis,
            [reference],
            smooth_method=self.smooth_method,
        )

        # Normalize score from 0-100 to 0-1
        score = bleu_result.score / 100.0

        # Determine pass/fail if threshold is set
        passed: bool | None = None
        if self.threshold is not None:
            passed = score >= self.threshold

        # Build reason string
        if self.threshold is not None:
            comparison = ">=" if passed else "<"
            reason = f"BLEU score {score:.4f} {comparison} threshold {self.threshold}"
        else:
            reason = f"BLEU score: {score:.4f}"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "bleu_raw": bleu_result.score,
                "brevity_penalty": bleu_result.bp,
            },
        )


class ROUGE(Method):
    """ROUGE score for summarization quality.

    Uses rouge-score library for computing ROUGE metrics.

    Attributes:
        name: Method identifier.
        required_fields: Sample fields required for evaluation.
    """

    name: str = "ROUGE"
    required_fields: list[str] = ["input", "output", "expected"]

    def __init__(
        self,
        threshold: float | None = None,
        rouge_types: list[str] | None = None,
        use_stemmer: bool = True,
        aggregate: str = "rougeL",
        metric: str = "fmeasure",
        **kwargs: Any,
    ) -> None:
        """Initialize ROUGE scorer.

        Args:
            threshold: Optional threshold for pass/fail determination (0-1 scale).
            rouge_types: List of ROUGE types to compute. Default: ["rouge1", "rouge2", "rougeL"].
            use_stemmer: Whether to use Porter stemmer. Default: True.
            aggregate: Which ROUGE type to use as the main score. Default: "rougeL".
            metric: Which metric to use: "precision", "recall", or "fmeasure". Default: "fmeasure".
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.threshold = threshold
        self.rouge_types = (
            rouge_types if rouge_types is not None else ["rouge1", "rouge2", "rougeL"]
        )
        self.use_stemmer = use_stemmer
        self.aggregate = aggregate
        self.metric = metric

        if self.aggregate not in self.rouge_types:
            raise ValueError(
                f"aggregate '{self.aggregate}' must be one of rouge_types: {self.rouge_types}"
            )

        if self.metric not in ("precision", "recall", "fmeasure"):
            raise ValueError(
                f"metric must be 'precision', 'recall', or 'fmeasure', got '{self.metric}'"
            )

        self._scorer = rouge_scorer.RougeScorer(
            self.rouge_types,
            use_stemmer=self.use_stemmer,
        )

        self.config.update(
            {
                "threshold": threshold,
                "rouge_types": self.rouge_types,
                "use_stemmer": use_stemmer,
                "aggregate": aggregate,
                "metric": metric,
            }
        )

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate ROUGE score for a sample.

        Args:
            sample: The sample to evaluate with output and expected fields.

        Returns:
            MethodResult with ROUGE scores.
        """
        hypothesis = sample.output
        reference = sample.expected

        if reference is None:
            return MethodResult(
                method_name=self.name,
                score=None,
                passed=None,
                reason="Expected field is required for ROUGE evaluation",
                error="Missing expected field",
            )

        scores = self._scorer.score(reference, hypothesis)

        # Build metadata with all ROUGE scores
        metadata: dict[str, Any] = {}
        for rouge_type in self.rouge_types:
            score_obj = scores[rouge_type]
            metadata[rouge_type] = {
                "precision": score_obj.precision,
                "recall": score_obj.recall,
                "fmeasure": score_obj.fmeasure,
            }

        # Get the aggregated score
        aggregate_score_obj = scores[self.aggregate]
        score = getattr(aggregate_score_obj, self.metric)

        # Determine pass/fail if threshold is set
        passed: bool | None = None
        if self.threshold is not None:
            passed = score >= self.threshold

        # Build reason string
        if self.threshold is not None:
            comparison = ">=" if passed else "<"
            reason = (
                f"ROUGE-{self.aggregate} {self.metric} "
                f"{score:.4f} {comparison} "
                f"threshold {self.threshold}"
            )
        else:
            reason = f"ROUGE-{self.aggregate} {self.metric}: {score:.4f}"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata=metadata,
        )


class TextLength(Method):
    """Check if output length is within bounds.

    Evaluates whether the output text length falls within specified
    minimum and maximum bounds, measured in characters or words.

    Attributes:
        name: Method identifier.
        required_fields: Sample fields required for evaluation.
    """

    name: str = "TextLength"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        min_length: int | None = None,
        max_length: int | None = None,
        unit: str = "chars",
        **kwargs: Any,
    ) -> None:
        """Initialize TextLength checker.

        Args:
            min_length: Minimum allowed length (inclusive). None means no minimum.
            max_length: Maximum allowed length (inclusive). None means no maximum.
            unit: Unit of measurement - "chars" or "words". Default: "chars".
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.min_length = min_length
        self.max_length = max_length
        self.unit = unit

        if self.unit not in ("chars", "words"):
            raise ValueError(f"unit must be 'chars' or 'words', got '{self.unit}'")

        if min_length is not None and max_length is not None and min_length > max_length:
            raise ValueError(
                f"min_length ({min_length}) cannot be greater than max_length ({max_length})"
            )

        self.config.update(
            {
                "min_length": min_length,
                "max_length": max_length,
                "unit": unit,
            }
        )

    def _get_length(self, text: str) -> int:
        """Get the length of text in the configured unit.

        Args:
            text: The text to measure.

        Returns:
            Length in characters or words.
        """
        if self.unit == "chars":
            return len(text)
        else:  # words
            return len(text.split())

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate text length for a sample.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if within bounds, 0.0 otherwise.
        """
        length = self._get_length(sample.output)

        # Check bounds
        within_min = self.min_length is None or length >= self.min_length
        within_max = self.max_length is None or length <= self.max_length
        passed = within_min and within_max

        # Score is 1.0 if within bounds, 0.0 otherwise
        score = 1.0 if passed else 0.0

        # Build reason string
        bounds_parts: list[str] = []
        if self.min_length is not None:
            bounds_parts.append(str(self.min_length))
        else:
            bounds_parts.append("-inf")
        if self.max_length is not None:
            bounds_parts.append(str(self.max_length))
        else:
            bounds_parts.append("inf")
        bounds_str = f"[{bounds_parts[0]}, {bounds_parts[1]}]"

        if passed:
            reason = f"Length {length} {self.unit} within bounds {bounds_str}"
        else:
            if not within_min:
                reason = f"Length {length} {self.unit} below minimum {self.min_length}"
            else:
                reason = f"Length {length} {self.unit} exceeds maximum {self.max_length}"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "length": length,
                "unit": self.unit,
                "min": self.min_length,
                "max": self.max_length,
            },
        )


class WordCount(Method):
    """Count words in output and check against bounds.

    Attributes:
        name: Method identifier.
        required_fields: Sample fields required for evaluation.
    """

    name: str = "WordCount"
    required_fields: list[str] = ["input", "output"]

    def __init__(
        self,
        min_words: int | None = None,
        max_words: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize WordCount checker.

        Args:
            min_words: Minimum allowed word count (inclusive). None means no minimum.
            max_words: Maximum allowed word count (inclusive). None means no maximum.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.min_words = min_words
        self.max_words = max_words

        if min_words is not None and max_words is not None and min_words > max_words:
            raise ValueError(
                f"min_words ({min_words}) cannot be greater than max_words ({max_words})"
            )

        self.config.update(
            {
                "min_words": min_words,
                "max_words": max_words,
            }
        )

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate word count for a sample.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if within bounds, 0.0 otherwise.
        """
        word_count = len(sample.output.split())

        # Check bounds
        within_min = self.min_words is None or word_count >= self.min_words
        within_max = self.max_words is None or word_count <= self.max_words
        passed = within_min and within_max

        # Score is 1.0 if within bounds, 0.0 otherwise
        score = 1.0 if passed else 0.0

        # Build reason string
        bounds_parts: list[str] = []
        if self.min_words is not None:
            bounds_parts.append(str(self.min_words))
        else:
            bounds_parts.append("-inf")
        if self.max_words is not None:
            bounds_parts.append(str(self.max_words))
        else:
            bounds_parts.append("inf")
        bounds_str = f"[{bounds_parts[0]}, {bounds_parts[1]}]"

        if passed:
            reason = f"Word count {word_count} within bounds {bounds_str}"
        else:
            if not within_min:
                reason = f"Word count {word_count} below minimum {self.min_words}"
            else:
                reason = f"Word count {word_count} exceeds maximum {self.max_words}"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "word_count": word_count,
                "min_words": self.min_words,
                "max_words": self.max_words,
            },
        )


class SentenceCount(Method):
    """Count sentences in output and check against bounds.

    Uses simple sentence detection by splitting on sentence-ending punctuation
    (., !, ?).

    Attributes:
        name: Method identifier.
        required_fields: Sample fields required for evaluation.
    """

    name: str = "SentenceCount"
    required_fields: list[str] = ["input", "output"]

    # Pattern to split on sentence-ending punctuation
    # Matches one or more of .!? followed by whitespace or end of string
    _SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+(?:\s+|$)")

    def __init__(
        self,
        min_sentences: int | None = None,
        max_sentences: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize SentenceCount checker.

        Args:
            min_sentences: Minimum allowed sentence count (inclusive). None means no minimum.
            max_sentences: Maximum allowed sentence count (inclusive). None means no maximum.
            **kwargs: Additional configuration options.
        """
        super().__init__(**kwargs)
        self.min_sentences = min_sentences
        self.max_sentences = max_sentences

        if (
            min_sentences is not None
            and max_sentences is not None
            and min_sentences > max_sentences
        ):
            raise ValueError(
                f"min_sentences ({min_sentences}) cannot be "
                f"greater than max_sentences ({max_sentences})"
            )

        self.config.update(
            {
                "min_sentences": min_sentences,
                "max_sentences": max_sentences,
            }
        )

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text using simple punctuation-based detection.

        Args:
            text: The text to analyze.

        Returns:
            Number of sentences detected.
        """
        # Split on sentence-ending punctuation and filter out empty strings
        parts = self._SENTENCE_SPLIT_PATTERN.split(text.strip())
        # Filter out empty strings that result from splitting
        sentences = [p.strip() for p in parts if p.strip()]
        return len(sentences)

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate sentence count for a sample.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score 1.0 if within bounds, 0.0 otherwise.
        """
        sentence_count = self._count_sentences(sample.output)

        # Check bounds
        within_min = self.min_sentences is None or sentence_count >= self.min_sentences
        within_max = self.max_sentences is None or sentence_count <= self.max_sentences
        passed = within_min and within_max

        # Score is 1.0 if within bounds, 0.0 otherwise
        score = 1.0 if passed else 0.0

        # Build reason string
        bounds_parts: list[str] = []
        if self.min_sentences is not None:
            bounds_parts.append(str(self.min_sentences))
        else:
            bounds_parts.append("-inf")
        if self.max_sentences is not None:
            bounds_parts.append(str(self.max_sentences))
        else:
            bounds_parts.append("inf")
        bounds_str = f"[{bounds_parts[0]}, {bounds_parts[1]}]"

        if passed:
            reason = f"Sentence count {sentence_count} within bounds {bounds_str}"
        else:
            if not within_min:
                reason = f"Sentence count {sentence_count} below minimum {self.min_sentences}"
            else:
                reason = f"Sentence count {sentence_count} exceeds maximum {self.max_sentences}"

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "sentence_count": sentence_count,
                "min_sentences": self.min_sentences,
                "max_sentences": self.max_sentences,
            },
        )
