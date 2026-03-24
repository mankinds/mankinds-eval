"""Multi-criteria LLM-as-Judge evaluation method."""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod


class CriterionConfig(TypedDict, total=False):
    """Configuration for a single criterion in multi-criteria evaluation."""

    description: str
    weight: float


class MultiCriteriaJudge(LLMMethod):
    """Evaluate using multiple criteria in a single LLM call.

    This method evaluates a sample against multiple criteria simultaneously,
    computing a weighted average for the overall score.

    Example:
        judge = MultiCriteriaJudge(
            criteria={
                "accuracy": {"description": "Factual correctness", "weight": 0.4},
                "helpfulness": {"description": "Usefulness to the user", "weight": 0.3},
                "clarity": {"description": "Clear and well-structured", "weight": 0.3},
            },
            scale="1-5",
            threshold=3.5,
            provider="openai",
        )
        result = await judge.evaluate(sample)
    """

    name = "MultiCriteriaJudge"
    required_fields = ["input", "output"]

    def __init__(
        self,
        criteria: dict[str, CriterionConfig],
        scale: str = "1-5",
        threshold: float | None = None,
        include_conversation: bool = True,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the multi-criteria judge.

        Args:
            criteria: Dictionary mapping criterion names to their config.
                Each config should have "description" and optionally "weight".
                Weights default to equal distribution if not specified.
            scale: Score scale - "1-5" or "1-10".
            threshold: Threshold for passed determination on aggregated score.
            include_conversation: Whether to include conversation history if present.
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.criteria = criteria
        self.scale = scale
        self.threshold = threshold
        self.include_conversation = include_conversation

        # Validate scale
        if scale not in ("1-5", "1-10"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5' or '1-10'")

        # Validate criteria
        if not criteria:
            raise ValueError("At least one criterion must be provided")

        # Normalize weights
        self._weights = self._normalize_weights(criteria)

    def _normalize_weights(self, criteria: dict[str, CriterionConfig]) -> dict[str, float]:
        """Normalize criterion weights to sum to 1.0.

        Args:
            criteria: The criteria configuration dict.

        Returns:
            Dictionary mapping criterion names to normalized weights.
        """
        weights: dict[str, float] = {}
        total_weight = 0.0

        for name, config in criteria.items():
            weight = config.get("weight", 1.0)
            weights[name] = weight
            total_weight += weight

        # Normalize
        if total_weight > 0:
            for name in weights:
                weights[name] /= total_weight

        return weights

    def _get_scale_range(self) -> tuple:
        """Get the min and max values for the scale.

        Returns:
            Tuple of (min_score, max_score).
        """
        if self.scale == "1-5":
            return (1, 5)
        else:  # 1-10
            return (1, 10)

    def _build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """Build prompt asking for scores on all criteria.

        Args:
            sample: The sample to evaluate.

        Returns:
            List of message dicts for the LLM.
        """
        min_score, max_score = self._get_scale_range()

        # Build criteria descriptions for the prompt
        criteria_desc_parts: list[str] = []
        for name, config in self.criteria.items():
            desc = config.get("description", name)
            criteria_desc_parts.append(f"- {name}: {desc}")
        criteria_desc = "\n".join(criteria_desc_parts)

        # Build expected JSON format
        json_example_parts: list[str] = []
        for name in self.criteria:
            json_example_parts.append(f'  "{name}": {{"score": <number>, "reason": "..."}}')
        json_example = "{\n" + ",\n".join(json_example_parts) + "\n}"

        # System prompt
        system_content = (
            f"You are an expert evaluator. Your task is to evaluate an AI assistant's "
            f"response based on multiple criteria.\n\n"
            f"Score each criterion from {min_score} (worst) to {max_score} (best).\n\n"
            f"Criteria to evaluate:\n{criteria_desc}\n\n"
            f"You MUST respond with valid JSON in this exact format:\n"
            f"{json_example}\n\n"
            f"Do not include any text outside the JSON object."
        )

        # Build user message content
        user_parts: list[str] = []

        # Include conversation history if present and enabled
        if self.include_conversation and sample.conversation:
            user_parts.append("## Conversation History")
            for msg in sample.conversation:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                user_parts.append(f"{role.capitalize()}: {content}")
            user_parts.append("")

        # Include context if present
        if sample.context:
            user_parts.append("## Context")
            user_parts.append(sample.context)
            user_parts.append("")

        # Include expected if present
        if sample.expected:
            user_parts.append("## Expected Response")
            user_parts.append(sample.expected)
            user_parts.append("")

        # Include input and output
        user_parts.append("## User Input")
        user_parts.append(sample.input)
        user_parts.append("")
        user_parts.append("## AI Response to Evaluate")
        user_parts.append(sample.output)
        user_parts.append("")

        # Add evaluation instruction
        user_parts.append("## Evaluation Task")
        user_parts.append(
            f"Evaluate the AI response on each criterion listed above.\n"
            f"Score each from {min_score} to {max_score}.\n"
            f"Respond with the JSON format specified."
        )

        user_content = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, response: str) -> dict[str, dict[str, float | str]]:
        """Parse LLM response to extract scores and reasons for each criterion.

        Args:
            response: Raw LLM response string.

        Returns:
            Dictionary mapping criterion names to {"score": float, "reason": str}.

        Raises:
            ValueError: If response cannot be parsed.
        """
        # Try to extract JSON from the response
        json_match = re.search(r"\{[\s\S]*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {response}") from e

        result: dict[str, dict[str, float | str]] = {}

        for criterion_name in self.criteria:
            if criterion_name not in data:
                raise ValueError(f"Missing criterion '{criterion_name}' in response")

            criterion_data = data[criterion_name]
            if not isinstance(criterion_data, dict):
                raise ValueError(f"Invalid format for criterion '{criterion_name}': expected dict")

            raw_score = criterion_data.get("score")
            if raw_score is None:
                raise ValueError(f"Missing score for criterion '{criterion_name}'")

            try:
                score = float(raw_score)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Invalid score for criterion '{criterion_name}': {raw_score}"
                ) from e

            reason = criterion_data.get("reason", "")
            if not isinstance(reason, str):
                reason = str(reason)

            result[criterion_name] = {"score": score, "reason": reason}

        return result

    def _compute_weighted_score(self, criterion_scores: dict[str, dict[str, float | str]]) -> float:
        """Compute weighted average score from individual criterion scores.

        Args:
            criterion_scores: Dictionary with score for each criterion.

        Returns:
            Weighted average score.
        """
        total_score = 0.0
        for criterion_name, weight in self._weights.items():
            score = criterion_scores[criterion_name]["score"]
            if isinstance(score, (int, float)):
                total_score += weight * score

        return total_score

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate a sample using multiple criteria.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with aggregated score and individual criterion scores in metadata.
        """
        messages = self._build_prompt(sample)
        response = await self.llm.complete(messages)
        criterion_scores = self._parse_response(response)

        # Compute weighted average
        aggregated_score = self._compute_weighted_score(criterion_scores)

        # Build combined reason
        reason_parts: list[str] = []
        for name, data in criterion_scores.items():
            reason_parts.append(f"{name}: {data['reason']}")
        combined_reason = " | ".join(reason_parts)

        passed: bool | None = None
        if self.threshold is not None:
            passed = aggregated_score >= self.threshold

        return MethodResult(
            method_name=self.name,
            score=aggregated_score,
            passed=passed,
            reason=combined_reason,
            metadata={
                "criterion_scores": criterion_scores,
                "weights": self._weights,
                "scale": self.scale,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
        )
