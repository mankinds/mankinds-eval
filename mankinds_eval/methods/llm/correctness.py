"""Correctness evaluation method using LLM-as-Judge."""

from __future__ import annotations

from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod
from mankinds_eval.utils import extract_json_object


class Correctness(LLMMethod):
    """Evaluate the factual correctness of the response against expected answer.

    This method uses an LLM to evaluate whether the response is factually
    correct when compared to a known expected answer.

    Inspired by DeepEval correctness metrics.

    Example:
        judge = Correctness(
            threshold=0.7,
            provider="openai",
            model="gpt-4o-mini"
        )
        result = await judge.evaluate(sample)
    """

    name = "Correctness"
    version = "0.1.0"
    required_fields = ["input", "output", "expected"]

    def __init__(
        self,
        threshold: float = 0.7,
        scale: str = "1-5",
        partial_credit: bool = True,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the correctness judge.

        Args:
            threshold: Score threshold for passed determination (normalized 0-1).
                Defaults to 0.7.
            scale: Score scale - "1-5", "1-10", or "binary".
                Defaults to "1-5".
            partial_credit: Whether to give partial credit for partially correct
                answers. Defaults to True.
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.threshold = threshold
        self.scale = scale
        self.partial_credit = partial_credit

        if scale not in ("1-5", "1-10", "binary"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5', '1-10', or 'binary'")

        self.config.update(
            {
                "threshold": threshold,
                "scale": scale,
                "partial_credit": partial_credit,
            }
        )

    def _get_scale_info(self) -> tuple[str, str, int]:
        """Get scale description, JSON format, and max value.

        Returns:
            Tuple of (description, json_format, max_value).
        """
        if self.scale == "binary":
            return (
                'Answer "yes" if correct, "no" if incorrect',
                (
                    '{"score": "yes" or "no", '
                    '"reason": "explanation", '
                    '"correct_elements": ["list"], '
                    '"incorrect_elements": ["list"], '
                    '"missing_elements": ["list"]}'
                ),
                1,
            )
        elif self.scale == "1-5":
            return (
                "Score from 1 (completely incorrect) to 5 (completely correct)",
                (
                    '{"score": 1-5, '
                    '"reason": "explanation", '
                    '"correct_elements": ["list"], '
                    '"incorrect_elements": ["list"], '
                    '"missing_elements": ["list"]}'
                ),
                5,
            )
        else:
            return (
                "Score from 1 (completely incorrect) to 10 (completely correct)",
                (
                    '{"score": 1-10, '
                    '"reason": "explanation", '
                    '"correct_elements": ["list"], '
                    '"incorrect_elements": ["list"], '
                    '"missing_elements": ["list"]}'
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

        partial_credit_instruction = ""
        if self.partial_credit:
            partial_credit_instruction = (
                "\nPartial credit is allowed: if some elements are correct but others "
                "are wrong or missing, give a proportional score."
            )
        else:
            partial_credit_instruction = (
                "\nNo partial credit: the response must be completely correct to "
                "receive a high score."
            )

        system_content = (
            "You are an expert evaluator assessing the factual correctness of AI responses. "
            "Your task is to compare the response against the expected answer and determine "
            "if the response is correct.\n\n"
            "Evaluation criteria:\n"
            "- Compare key facts, concepts, and information\n"
            "- Consider semantic equivalence (different wording but same meaning is OK)\n"
            "- Identify correct, incorrect, and missing elements\n"
            f"{partial_credit_instruction}\n\n"
            f"{scale_desc}.\n\n"
            f"You MUST respond with valid JSON in this exact format:\n{json_format}\n\n"
            "Do not include any text outside the JSON object."
        )

        user_parts = [
            f"## Question\n{sample.input}",
            f"\n## Expected Answer (Ground Truth)\n{sample.expected}",
            f"\n## AI Response to Evaluate\n{sample.output}",
        ]

        if sample.context:
            user_parts.insert(1, f"\n## Context\n{sample.context}")

        user_parts.append(
            "\n## Task\nCompare the AI response to the expected answer and evaluate correctness."
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
            "correct_elements": data.get("correct_elements", []),
            "incorrect_elements": data.get("incorrect_elements", []),
            "missing_elements": data.get("missing_elements", []),
        }

        return score, reason, analysis

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate correctness of the response against expected answer.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with correctness score and analysis.
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
                "correct_elements": analysis["correct_elements"],
                "incorrect_elements": analysis["incorrect_elements"],
                "missing_elements": analysis["missing_elements"],
                "num_correct": len(analysis["correct_elements"]),
                "num_incorrect": len(analysis["incorrect_elements"]),
                "num_missing": len(analysis["missing_elements"]),
                "scale": self.scale,
                "threshold": self.threshold,
                "partial_credit": self.partial_credit,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
        )
