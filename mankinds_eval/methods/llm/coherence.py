"""Coherence evaluation method using LLM-as-Judge."""

from __future__ import annotations

import contextlib
import json
import re
from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod


class Coherence(LLMMethod):
    """Evaluate the coherence and clarity of the response.

    This method uses an LLM to evaluate whether the response is logically
    coherent, well-structured, and clearly expressed.

    Inspired by DeepEval coherence metrics.

    Example:
        judge = Coherence(
            threshold=0.7,
            provider="openai",
            model="gpt-4o-mini"
        )
        result = await judge.evaluate(sample)
    """

    name = "Coherence"
    version = "0.1.0"
    required_fields = ["input", "output"]

    # Default aspects to evaluate
    DEFAULT_ASPECTS = ["logical_flow", "clarity", "consistency"]

    def __init__(
        self,
        threshold: float = 0.7,
        scale: str = "1-5",
        aspects: list[str] | None = None,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the coherence judge.

        Args:
            threshold: Score threshold for passed determination (normalized 0-1).
                Defaults to 0.7.
            scale: Score scale - "1-5", "1-10", or "binary".
                Defaults to "1-5".
            aspects: List of aspects to evaluate. Defaults to
                ["logical_flow", "clarity", "consistency"].
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.threshold = threshold
        self.scale = scale
        self.aspects = aspects or self.DEFAULT_ASPECTS

        if scale not in ("1-5", "1-10", "binary"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5', '1-10', or 'binary'")

        self.config.update(
            {
                "threshold": threshold,
                "scale": scale,
                "aspects": self.aspects,
            }
        )

    def _get_scale_info(self) -> tuple[str, str, int]:
        """Get scale description, JSON format, and max value.

        Returns:
            Tuple of (description, json_format, max_value).
        """
        aspects_format = ", ".join([f'"{a}": 1-5' for a in self.aspects])

        if self.scale == "binary":
            return (
                'Answer "yes" if coherent, "no" if not',
                (
                    '{"score": "yes" or "no", '
                    '"reason": "explanation", '
                    '"issues": ["list of coherence issues"]}'
                ),
                1,
            )
        elif self.scale == "1-5":
            return (
                "Score from 1 (incoherent) to 5 (highly coherent)",
                (
                    '{"score": 1-5, "reason": "explanation", '
                    f'"aspect_scores": {{{aspects_format}}}, '
                    '"issues": ["list"]}'
                ),
                5,
            )
        else:
            aspects_format_10 = ", ".join([f'"{a}": 1-10' for a in self.aspects])
            return (
                "Score from 1 (incoherent) to 10 (highly coherent)",
                (
                    '{"score": 1-10, "reason": "explanation", '
                    f'"aspect_scores": {{{aspects_format_10}}}, '
                    '"issues": ["list"]}'
                ),
                10,
            )

    def _get_aspects_description(self) -> str:
        """Get description of aspects to evaluate.

        Returns:
            Formatted description of coherence aspects.
        """
        aspect_descriptions = {
            "logical_flow": (
                "Logical Flow: Ideas progress naturally and arguments are well-structured"
            ),
            "clarity": "Clarity: Language is clear, unambiguous, and easy to understand",
            "consistency": "Consistency: No contradictions or conflicting statements",
            "organization": "Organization: Content is well-organized with clear structure",
            "conciseness": "Conciseness: No unnecessary repetition or verbosity",
        }

        descriptions = []
        for aspect in self.aspects:
            if aspect in aspect_descriptions:
                descriptions.append(f"- {aspect_descriptions[aspect]}")
            else:
                descriptions.append(f"- {aspect.replace('_', ' ').title()}")

        return "\n".join(descriptions)

    def _build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """Build messages for LLM evaluation.

        Args:
            sample: The sample to evaluate.

        Returns:
            List of message dicts for the LLM.
        """
        scale_desc, json_format, _ = self._get_scale_info()
        aspects_desc = self._get_aspects_description()

        system_content = (
            "You are an expert evaluator assessing the coherence and clarity of AI responses. "
            "Your task is to determine if the response is logically coherent, well-structured, "
            "and clearly expressed.\n\n"
            "Evaluate the following aspects:\n"
            f"{aspects_desc}\n\n"
            f"{scale_desc}.\n\n"
            f"You MUST respond with valid JSON in this exact format:\n{json_format}\n\n"
            "List any coherence issues found in the 'issues' array.\n"
            "Do not include any text outside the JSON object."
        )

        user_parts = [f"## User Input\n{sample.input}"]

        if sample.context:
            user_parts.append(f"\n## Context\n{sample.context}")

        user_parts.append(f"\n## AI Response to Evaluate\n{sample.output}")
        user_parts.append("\n## Task\nEvaluate the coherence and clarity of the AI response.")

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
        # Try to find JSON with nested objects
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
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

        # Normalize aspect scores
        raw_aspect_scores = data.get("aspect_scores", {})
        aspect_scores = {}
        if isinstance(raw_aspect_scores, dict):
            for aspect, aspect_score in raw_aspect_scores.items():
                with contextlib.suppress(TypeError, ValueError):
                    aspect_scores[aspect] = float(aspect_score) / max_value

        analysis = {
            "aspect_scores": aspect_scores,
            "issues": data.get("issues", []),
        }

        return score, reason, analysis

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate coherence of the response.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with coherence score and analysis.
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
                "aspect_scores": analysis["aspect_scores"],
                "issues": analysis["issues"],
                "aspects_evaluated": self.aspects,
                "scale": self.scale,
                "threshold": self.threshold,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
        )
