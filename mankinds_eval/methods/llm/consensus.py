"""Consensus LLM-as-Judge evaluation method using multiple LLMs."""

from __future__ import annotations

import asyncio
import statistics
from typing import Any, Literal, TypedDict

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method
from mankinds_eval.methods.llm.single import SingleCriterionJudge


class JudgeConfig(TypedDict, total=False):
    """Configuration for a single judge in consensus evaluation."""

    provider: str
    model: str
    api_key: str
    api_base: str
    temperature: float
    max_tokens: int


AggregationType = Literal["median", "mean", "majority_vote", "min", "max"]


class ConsensusJudge(Method):
    """Use multiple LLMs to evaluate and aggregate scores.

    This method runs the same evaluation across multiple LLM judges and
    aggregates their scores to produce a consensus result.

    Example:
        judge = ConsensusJudge(
            judges=[
                {"provider": "openai", "model": "gpt-4o"},
                {"provider": "openai", "model": "gpt-4o-mini"},
                {"provider": "anthropic", "model": "claude-3-haiku-20240307"},
            ],
            criterion="helpfulness",
            scale="1-5",
            aggregation="median",
            threshold=3.5,
        )
        result = await judge.evaluate(sample)
    """

    name = "ConsensusJudge"
    required_fields = ["input", "output"]

    def __init__(
        self,
        judges: list[JudgeConfig],
        criterion: str,
        scale: str = "1-5",
        aggregation: AggregationType = "median",
        threshold: float | None = None,
        include_conversation: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialize the consensus judge.

        Args:
            judges: List of judge configurations. Each should have at least "provider".
            criterion: The criterion to evaluate.
            scale: Score scale - "1-5", "1-10", or "binary".
            aggregation: Aggregation method - "median", "mean", "majority_vote", "min", "max".
            threshold: Threshold for passed determination on aggregated score.
            include_conversation: Whether to include conversation history if present.
            **kwargs: Additional arguments passed to Method.
        """
        # Don't call LLMMethod.__init__ - we manage multiple providers
        Method.__init__(self, **kwargs)

        self.judges_config = judges
        self.criterion = criterion
        self.scale = scale
        self.aggregation = aggregation
        self.threshold = threshold
        self.include_conversation = include_conversation

        # Validate inputs
        if not judges:
            raise ValueError("At least one judge must be provided")

        if aggregation not in ("median", "mean", "majority_vote", "min", "max"):
            raise ValueError(
                f"Invalid aggregation: {aggregation}. "
                f"Must be 'median', 'mean', 'majority_vote', 'min', or 'max'"
            )

        if scale not in ("1-5", "1-10", "binary"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5', '1-10', or 'binary'")

        # Lazy initialization of judge instances
        self._judges: list[SingleCriterionJudge] | None = None

    def _create_judges(self) -> list[SingleCriterionJudge]:
        """Create SingleCriterionJudge instances for each configuration.

        Returns:
            List of SingleCriterionJudge instances.
        """
        judges: list[SingleCriterionJudge] = []

        for config in self.judges_config:
            provider = config.get("provider", "openai")
            model = config.get("model")
            api_key = config.get("api_key")
            api_base = config.get("api_base")
            temperature = config.get("temperature", 0.0)
            max_tokens = config.get("max_tokens", 1024)

            judge = SingleCriterionJudge(
                criterion=self.criterion,
                scale=self.scale,
                threshold=None,  # We handle threshold at consensus level
                include_conversation=self.include_conversation,
                provider=provider,
                model=model,
                api_key=api_key,
                api_base=api_base,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            judges.append(judge)

        return judges

    def _get_judges(self) -> list[SingleCriterionJudge]:
        """Get or create judge instances.

        Returns:
            List of SingleCriterionJudge instances.
        """
        if self._judges is None:
            self._judges = self._create_judges()
        return self._judges

    def _aggregate_scores(self, scores: list[float]) -> float:
        """Aggregate scores using the configured method.

        Args:
            scores: List of individual judge scores.

        Returns:
            Aggregated score.
        """
        if not scores:
            return 0.0

        if self.aggregation == "median":
            return statistics.median(scores)
        elif self.aggregation == "mean":
            return statistics.mean(scores)
        elif self.aggregation == "min":
            return min(scores)
        elif self.aggregation == "max":
            return max(scores)
        elif self.aggregation == "majority_vote":
            # For majority vote, round scores and find mode
            rounded_scores = [round(s) for s in scores]
            try:
                return float(statistics.mode(rounded_scores))
            except statistics.StatisticsError:
                # No unique mode, fall back to median
                return statistics.median(scores)

        # Default fallback
        return statistics.mean(scores)

    def _calculate_agreement_rate(self, scores: list[float]) -> float:
        """Calculate agreement rate among judges.

        For numeric scales, agreement is based on standard deviation.
        Lower std dev = higher agreement.

        Args:
            scores: List of individual judge scores.

        Returns:
            Agreement rate between 0 and 1.
        """
        if len(scores) < 2:
            return 1.0

        try:
            std_dev = statistics.stdev(scores)
        except statistics.StatisticsError:
            return 1.0

        # Normalize agreement based on scale
        if self.scale == "binary":
            max_std = 0.5  # Max std dev for binary (0, 1)
        elif self.scale == "1-5":
            max_std = 2.0  # Approximate max std dev for 1-5 scale
        else:  # 1-10
            max_std = 4.5  # Approximate max std dev for 1-10 scale

        # Convert std dev to agreement rate (0 std = 1.0 agreement)
        agreement = 1.0 - min(std_dev / max_std, 1.0)
        return agreement

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate a sample using multiple LLM judges.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with aggregated score and individual scores in metadata.
        """
        judges = self._get_judges()

        # Run all judges in parallel
        tasks = [judge.evaluate(sample) for judge in judges]
        results: list[MethodResult | BaseException] = await asyncio.gather(
            *tasks, return_exceptions=True
        )

        # Collect scores and individual results
        scores: list[float] = []
        individual_scores: dict[str, dict[str, Any]] = {}
        errors: list[str] = []

        for _i, (judge, result) in enumerate(zip(judges, results)):
            judge_key = f"{judge.llm.provider}/{judge.llm.model}"

            if isinstance(result, BaseException):
                errors.append(f"{judge_key}: {result!s}")
                individual_scores[judge_key] = {
                    "score": None,
                    "reason": None,
                    "error": str(result),
                }
                continue
            if result.error:
                errors.append(f"{judge_key}: {result.error}")
                individual_scores[judge_key] = {
                    "score": None,
                    "reason": None,
                    "error": result.error,
                }
            elif result.score is not None:
                scores.append(result.score)
                individual_scores[judge_key] = {
                    "score": result.score,
                    "reason": result.reason,
                }
            else:
                errors.append(f"{judge_key}: No score returned")
                individual_scores[judge_key] = {
                    "score": None,
                    "reason": None,
                    "error": "No score returned",
                }

        # If no valid scores, return error
        if not scores:
            return MethodResult(
                method_name=self.name,
                score=None,
                passed=None,
                reason=None,
                error=f"All judges failed: {'; '.join(errors)}",
                metadata={"individual_scores": individual_scores},
            )

        # Aggregate scores
        aggregated_score = self._aggregate_scores(scores)
        agreement_rate = self._calculate_agreement_rate(scores)

        # Build combined reason
        reason_parts: list[str] = []
        for judge_key, data in individual_scores.items():
            if data.get("reason"):
                reason_parts.append(f"[{judge_key}]: {data['reason']}")
        combined_reason = " | ".join(reason_parts) if reason_parts else None

        passed: bool | None = None
        if self.threshold is not None:
            passed = aggregated_score >= self.threshold

        # Include partial error info if some judges failed
        error_msg: str | None = None
        if errors:
            error_msg = f"Some judges failed: {'; '.join(errors)}"

        return MethodResult(
            method_name=self.name,
            score=aggregated_score,
            passed=passed,
            reason=combined_reason,
            error=error_msg if errors and len(scores) < len(judges) else None,
            metadata={
                "individual_scores": individual_scores,
                "agreement_rate": agreement_rate,
                "aggregation": self.aggregation,
                "criterion": self.criterion,
                "scale": self.scale,
                "num_judges": len(judges),
                "num_successful": len(scores),
            },
        )
