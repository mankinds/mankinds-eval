"""Answer relevancy evaluation method using LLM-as-Judge."""

from __future__ import annotations

from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod
from mankinds_eval.utils import extract_json_object


class AnswerRelevancy(LLMMethod):
    """Evaluate if the response is relevant to the question.

    This method uses an LLM to evaluate whether the response directly
    addresses the user's question and stays on topic.

    Inspired by RAGAS and DeepEval answer relevancy metrics.

    Example:
        judge = AnswerRelevancy(
            threshold=0.7,
            provider="openai",
            model="gpt-4o-mini"
        )
        result = await judge.evaluate(sample)
    """

    name = "AnswerRelevancy"
    version = "0.1.0"
    required_fields = ["input", "output"]

    def __init__(
        self,
        threshold: float = 0.7,
        scale: str = "1-5",
        check_completeness: bool = True,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the answer relevancy judge.

        Args:
            threshold: Score threshold for passed determination (normalized 0-1).
                Defaults to 0.7.
            scale: Score scale - "1-5", "1-10", or "binary".
                Defaults to "1-5".
            check_completeness: Whether to also check if response completely
                addresses all parts of the question. Defaults to True.
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.threshold = threshold
        self.scale = scale
        self.check_completeness = check_completeness

        if scale not in ("1-5", "1-10", "binary"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5', '1-10', or 'binary'")

        self.config.update(
            {
                "threshold": threshold,
                "scale": scale,
                "check_completeness": check_completeness,
            }
        )

    def _get_scale_info(self) -> tuple[str, str, int]:
        """Get scale description, JSON format, and max value.

        Returns:
            Tuple of (description, json_format, max_value).
        """
        if self.scale == "binary":
            return (
                'Answer "yes" if relevant, "no" if not',
                (
                    '{"score": "yes" or "no", '
                    '"reason": "explanation", '
                    '"addresses_question": true/false, '
                    '"is_complete": true/false, '
                    '"off_topic_elements": ["list"]}'
                ),
                1,
            )
        elif self.scale == "1-5":
            return (
                "Score from 1 (completely irrelevant) to 5 (highly relevant)",
                (
                    '{"score": 1-5, '
                    '"reason": "explanation", '
                    '"addresses_question": true/false, '
                    '"is_complete": true/false, '
                    '"off_topic_elements": ["list"]}'
                ),
                5,
            )
        else:
            return (
                "Score from 1 (completely irrelevant) to 10 (highly relevant)",
                (
                    '{"score": 1-10, '
                    '"reason": "explanation", '
                    '"addresses_question": true/false, '
                    '"is_complete": true/false, '
                    '"off_topic_elements": ["list"]}'
                ),
                10,
            )

    def _build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """Build messages for LLM evaluation.

        Args:
            sample: The sample to evaluate.

        Returns:
            List of message dicts for the LLM.
        """
        scale_desc, json_format, _ = self._get_scale_info()

        completeness_instruction = ""
        if self.check_completeness:
            completeness_instruction = (
                "\n- Completeness: Does it address ALL parts of the question?"
            )

        system_content = (
            "You are an expert evaluator assessing the relevancy of AI responses. "
            "Your task is to determine if the response directly and appropriately "
            "addresses the user's question.\n\n"
            "Relevancy criteria:\n"
            "- Does the response directly answer what was asked?\n"
            "- Is the response on-topic and focused?\n"
            "- Are there any irrelevant or off-topic elements?"
            f"{completeness_instruction}\n\n"
            f"{scale_desc}.\n\n"
            f"You MUST respond with valid JSON in this exact format:\n{json_format}\n\n"
            "Do not include any text outside the JSON object."
        )

        user_parts = [f"## User Question\n{sample.input}"]

        # Include context if present
        if sample.context:
            user_parts.append(f"\n## Context\n{sample.context}")

        user_parts.append(f"\n## AI Response to Evaluate\n{sample.output}")
        user_parts.append(
            "\n## Task\n"
            "Evaluate if the AI response is relevant to and addresses the user's question."
        )

        user_content = "\n".join(user_parts)

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, response: str) -> tuple[float, str, dict[str, Any]]:
        """Parse LLM response to extract score, reason, and analysis.

        Args:
            response: Raw LLM response string.

        Returns:
            Tuple of (normalized_score, reason, analysis_dict).

        Raises:
            ValueError: If response cannot be parsed.
        """
        data = extract_json_object(response)

        raw_score = data.get("score")
        if raw_score is None:
            raise ValueError(f"No score found in response: {data}")

        _, _, max_value = self._get_scale_info()

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
            else:
                score = 1.0 if raw_score else 0.0
        else:
            try:
                score = float(raw_score) / max_value  # Normalize to 0-1
            except (TypeError, ValueError) as e:
                raise ValueError(f"Invalid numeric score: {raw_score}") from e

        reason = data.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)

        analysis = {
            "addresses_question": data.get("addresses_question", None),
            "is_complete": data.get("is_complete", None),
            "off_topic_elements": data.get("off_topic_elements", []),
        }

        return score, reason, analysis

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate relevancy of the response to the question.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with relevancy score and analysis.
        """
        messages = self._build_prompt(sample)
        response = await self.llm.complete(messages)
        score, reason, analysis = self._parse_response(response)

        passed = score >= self.threshold

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "addresses_question": analysis["addresses_question"],
                "is_complete": analysis["is_complete"],
                "off_topic_elements": analysis["off_topic_elements"],
                "scale": self.scale,
                "threshold": self.threshold,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
        )
