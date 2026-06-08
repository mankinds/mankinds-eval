"""Single criterion LLM-as-Judge evaluation method."""

from __future__ import annotations

from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod
from mankinds_eval.utils import extract_json_object


class SingleCriterionJudge(LLMMethod):
    """Evaluate using a single criterion with LLM.

    This method uses an LLM to evaluate a sample's output against a specified
    criterion. The LLM returns a score and reason explaining the evaluation.

    Example:
        judge = SingleCriterionJudge(
            criterion="helpfulness",
            scale="1-5",
            threshold=3,
            provider="openai",
            model="gpt-4o-mini"
        )
        result = await judge.evaluate(sample)
    """

    name = "SingleCriterionJudge"
    required_fields = ["input", "output"]

    def __init__(
        self,
        criterion: str,
        scale: str = "1-5",
        threshold: int | float | None = None,
        include_conversation: bool = True,
        system_prompt: str | None = None,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the single criterion judge.

        Args:
            criterion: The criterion to evaluate (e.g., "helpfulness", "accuracy").
            scale: Score scale - "1-5", "1-10", or "binary" (yes/no).
            threshold: Threshold for passed determination. If None, passed is not set.
            include_conversation: Whether to include conversation history if present.
            system_prompt: Custom system prompt override. If None, a default is used.
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.criterion = criterion
        self.scale = scale
        self.threshold = threshold
        self.include_conversation = include_conversation
        self.custom_system_prompt = system_prompt

        # Validate scale
        if scale not in ("1-5", "1-10", "binary"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5', '1-10', or 'binary'")

    def _get_scale_description(self) -> str:
        """Get description of the scoring scale.

        Returns:
            Human-readable description of the scale.
        """
        if self.scale == "binary":
            return 'Answer "yes" or "no"'
        elif self.scale == "1-5":
            return "Score from 1 (worst) to 5 (best)"
        else:  # 1-10
            return "Score from 1 (worst) to 10 (best)"

    def _get_json_format(self) -> str:
        """Get JSON format instruction for the response.

        Returns:
            JSON format string for LLM instructions.
        """
        if self.scale == "binary":
            return '{"score": "yes" or "no", "reason": "your explanation"}'
        return '{"score": <number>, "reason": "your explanation"}'

    def _build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """Build messages for LLM.

        Args:
            sample: The sample to evaluate.

        Returns:
            List of message dicts for the LLM.
        """
        # System prompt
        if self.custom_system_prompt:
            system_content = self.custom_system_prompt
        else:
            system_content = (
                f"You are an expert evaluator. Your task is to evaluate an AI assistant's "
                f"response based on the criterion: {self.criterion}.\n\n"
                f"{self._get_scale_description()}.\n\n"
                f"You MUST respond with valid JSON in this exact format:\n"
                f"{self._get_json_format()}\n\n"
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
            f"Evaluate the AI response based on: {self.criterion}\n"
            f"{self._get_scale_description()}.\n"
            f"Respond with JSON: {self._get_json_format()}"
        )

        user_content = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, response: str) -> tuple[float, str]:
        """Parse LLM response to extract score and reason.

        Args:
            response: Raw LLM response string.

        Returns:
            Tuple of (score, reason).

        Raises:
            ValueError: If response cannot be parsed.
        """
        # Extract JSON robustly: handles extra prose, markdown fences, braces
        # inside string values, and responses truncated at the token limit.
        data = extract_json_object(response)

        # Extract score
        raw_score = data.get("score")
        if raw_score is None:
            raise ValueError(f"No score found in response: {data}")

        # Handle binary scale
        if self.scale == "binary":
            if isinstance(raw_score, str):
                raw_score_lower = raw_score.lower().strip()
                if raw_score_lower in ("yes", "true", "1"):
                    score = 1.0
                elif raw_score_lower in ("no", "false", "0"):
                    score = 0.0
                else:
                    raise ValueError(f"Invalid binary score: {raw_score}")
            elif isinstance(raw_score, (int, float, bool)):
                score = 1.0 if raw_score else 0.0
            else:
                raise ValueError(f"Invalid binary score type: {type(raw_score)}")
        else:
            # Numeric scale
            try:
                score = float(raw_score)
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid numeric score: {raw_score}") from e

        # Extract reason
        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)

        return score, reason

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate a sample using the single criterion.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with score, passed, reason, and metadata.
        """
        messages = self._build_prompt(sample)
        response = await self.llm.complete(messages)
        score, reason = self._parse_response(response)

        passed: bool | None = None
        if self.threshold is not None:
            passed = score >= self.threshold

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "criterion": self.criterion,
                "scale": self.scale,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
        )
