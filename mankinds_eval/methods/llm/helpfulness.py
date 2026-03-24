"""Helpfulness evaluation method using LLM-as-Judge."""

from __future__ import annotations

import json
import re
from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod


class Helpfulness(LLMMethod):
    """Evaluate the helpfulness and practical utility of the response.

    This method uses an LLM to evaluate whether the response provides
    useful, actionable, and valuable information to the user.

    Inspired by DeepEval helpfulness metrics.

    Example:
        judge = Helpfulness(
            threshold=0.7,
            provider="openai",
            model="gpt-4o-mini"
        )
        result = await judge.evaluate(sample)
    """

    name = "Helpfulness"
    version = "0.1.0"
    required_fields = ["input", "output"]

    def __init__(
        self,
        threshold: float = 0.7,
        scale: str = "1-5",
        consider_context: bool = True,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the helpfulness judge.

        Args:
            threshold: Score threshold for passed determination (normalized 0-1).
                Defaults to 0.7.
            scale: Score scale - "1-5", "1-10", or "binary".
                Defaults to "1-5".
            consider_context: Whether to consider provided context when evaluating.
                Defaults to True.
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.threshold = threshold
        self.scale = scale
        self.consider_context = consider_context

        if scale not in ("1-5", "1-10", "binary"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5', '1-10', or 'binary'")

        self.config.update(
            {
                "threshold": threshold,
                "scale": scale,
                "consider_context": consider_context,
            }
        )

    def _get_scale_info(self) -> tuple[str, str, int]:
        """Get scale description, JSON format, and max value.

        Returns:
            Tuple of (description, json_format, max_value).
        """
        if self.scale == "binary":
            return (
                'Answer "yes" if helpful, "no" if not',
                (
                    '{"score": "yes" or "no", '
                    '"reason": "explanation", '
                    '"is_actionable": true/false, '
                    '"is_informative": true/false, '
                    '"practical_value": "high/medium/low"}'
                ),
                1,
            )
        elif self.scale == "1-5":
            return (
                "Score from 1 (not helpful) to 5 (extremely helpful)",
                (
                    '{"score": 1-5, '
                    '"reason": "explanation", '
                    '"is_actionable": true/false, '
                    '"is_informative": true/false, '
                    '"practical_value": "high/medium/low"}'
                ),
                5,
            )
        else:
            return (
                "Score from 1 (not helpful) to 10 (extremely helpful)",
                (
                    '{"score": 1-10, '
                    '"reason": "explanation", '
                    '"is_actionable": true/false, '
                    '"is_informative": true/false, '
                    '"practical_value": "high/medium/low"}'
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

        system_content = (
            "You are an expert evaluator assessing the helpfulness of AI responses. "
            "Your task is to determine if the response provides useful, practical value "
            "to the user.\n\n"
            "Helpfulness criteria:\n"
            "- Actionable: Does it provide steps or guidance the user can follow?\n"
            "- Informative: Does it provide useful information that addresses the need?\n"
            "- Practical Value: Would this response actually help the user achieve their goal?\n"
            "- Depth: Is the level of detail appropriate for the question?\n\n"
            f"{scale_desc}.\n\n"
            f"You MUST respond with valid JSON in this exact format:\n{json_format}\n\n"
            "Do not include any text outside the JSON object."
        )

        user_parts = [f"## User Question\n{sample.input}"]

        if self.consider_context and sample.context:
            user_parts.append(f"\n## Available Context\n{sample.context}")

        user_parts.append(f"\n## AI Response to Evaluate\n{sample.output}")
        user_parts.append(
            "\n## Task\n"
            "Evaluate the helpfulness of the AI response. Consider whether it provides "
            "practical value and genuinely helps the user."
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
        json_match = re.search(r"\{[^{}]*\}", response, re.DOTALL)
        if not json_match:
            raise ValueError(f"No JSON found in response: {response}")

        try:
            data = json.loads(json_match.group())
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in response: {response}") from e

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

        # Convert practical_value to numeric
        practical_value_str = data.get("practical_value", "medium")
        practical_value_map = {"high": 1.0, "medium": 0.5, "low": 0.0}
        practical_value = practical_value_map.get(
            practical_value_str.lower() if isinstance(practical_value_str, str) else "medium", 0.5
        )

        analysis = {
            "is_actionable": data.get("is_actionable", None),
            "is_informative": data.get("is_informative", None),
            "practical_value": practical_value,
            "practical_value_label": practical_value_str,
        }

        return score, reason, analysis

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate helpfulness of the response.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with helpfulness score and analysis.
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
                "is_actionable": analysis["is_actionable"],
                "is_informative": analysis["is_informative"],
                "practical_value": analysis["practical_value"],
                "practical_value_label": analysis["practical_value_label"],
                "scale": self.scale,
                "threshold": self.threshold,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
        )
