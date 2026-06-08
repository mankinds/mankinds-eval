"""Faithfulness evaluation method using LLM-as-Judge."""

from __future__ import annotations

from typing import Any

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.llm.base import LLMMethod
from mankinds_eval.utils import extract_json_object


class Faithfulness(LLMMethod):
    """Evaluate if output is faithful to the provided context.

    This method uses an LLM to evaluate whether the response is grounded
    in the provided context and does not contain hallucinations or
    unsupported claims.

    Inspired by RAGAS faithfulness metric.

    Example:
        judge = Faithfulness(
            threshold=0.7,
            provider="openai",
            model="gpt-4o-mini"
        )
        result = await judge.evaluate(sample)
    """

    name = "Faithfulness"
    version = "0.1.0"
    required_fields = ["input", "output", "context"]

    def __init__(
        self,
        threshold: float = 0.7,
        scale: str = "1-5",
        strict: bool = False,
        provider: str = "openai",
        model: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the faithfulness judge.

        Args:
            threshold: Score threshold for passed determination (normalized 0-1).
                Defaults to 0.7.
            scale: Score scale - "1-5", "1-10", or "binary".
                Defaults to "1-5".
            strict: If True, any claim not directly supported by context fails.
                Defaults to False.
            provider: LLM provider name.
            model: Model name.
            **kwargs: Additional arguments passed to LLMMethod.
        """
        super().__init__(provider=provider, model=model, **kwargs)
        self.threshold = threshold
        self.scale = scale
        self.strict = strict

        if scale not in ("1-5", "1-10", "binary"):
            raise ValueError(f"Invalid scale: {scale}. Must be '1-5', '1-10', or 'binary'")

        self.config.update(
            {
                "threshold": threshold,
                "scale": scale,
                "strict": strict,
            }
        )

    def _get_scale_info(self) -> tuple[str, str, int]:
        """Get scale description, JSON format, and max value.

        Returns:
            Tuple of (description, json_format, max_value).
        """
        if self.scale == "binary":
            return (
                'Answer "yes" if faithful, "no" if not',
                (
                    '{"score": "yes" or "no", '
                    '"reason": "explanation", '
                    '"unsupported_claims": '
                    '["list of claims not in context"]}'
                ),
                1,
            )
        elif self.scale == "1-5":
            return (
                "Score from 1 (completely unfaithful) to 5 (completely faithful)",
                (
                    '{"score": 1-5, '
                    '"reason": "explanation", '
                    '"unsupported_claims": '
                    '["list of claims not in context"]}'
                ),
                5,
            )
        else:
            return (
                "Score from 1 (completely unfaithful) to 10 (completely faithful)",
                (
                    '{"score": 1-10, '
                    '"reason": "explanation", '
                    '"unsupported_claims": '
                    '["list of claims not in context"]}'
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

        strict_instruction = ""
        if self.strict:
            strict_instruction = (
                "\nIMPORTANT: Be strict. Any claim or statement in the response that "
                "cannot be DIRECTLY verified from the context should be considered unfaithful. "
                "Common knowledge or reasonable inferences are NOT acceptable."
            )

        system_content = (
            "You are an expert evaluator assessing the faithfulness of AI responses. "
            "Your task is to determine if the response is grounded in and supported by "
            "the provided context.\n\n"
            "Faithfulness means:\n"
            "- All claims in the response can be verified from the context\n"
            "- No information is fabricated or hallucinated\n"
            "- No unsupported extrapolations beyond what the context states\n"
            f"{strict_instruction}\n\n"
            f"{scale_desc}.\n\n"
            f"You MUST respond with valid JSON in this exact format:\n{json_format}\n\n"
            "List any claims that are NOT supported by the context in 'unsupported_claims'.\n"
            "Do not include any text outside the JSON object."
        )

        user_content = (
            f"## Context (Source of Truth)\n{sample.context}\n\n"
            f"## User Question\n{sample.input}\n\n"
            f"## AI Response to Evaluate\n{sample.output}\n\n"
            "## Task\n"
            "Evaluate if the AI response is faithful to the context. "
            "Identify any claims not supported by the context."
        )

        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content},
        ]

    def _parse_response(self, response: str) -> tuple[float, str, list[str]]:
        """Parse LLM response to extract score, reason, and unsupported claims.

        Args:
            response: Raw LLM response string.

        Returns:
            Tuple of (normalized_score, reason, unsupported_claims).

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

        unsupported_claims = data.get("unsupported_claims", [])
        if not isinstance(unsupported_claims, list):
            unsupported_claims = []

        return score, reason, unsupported_claims

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Evaluate faithfulness of the response to the context.

        Args:
            sample: The sample to evaluate.

        Returns:
            MethodResult with faithfulness score and analysis.
        """
        messages = self._build_prompt(sample)
        response = await self.llm.complete(messages)
        score, reason, unsupported_claims = self._parse_response(response)

        passed = score >= self.threshold

        return MethodResult(
            method_name=self.name,
            score=score,
            passed=passed,
            reason=reason,
            metadata={
                "unsupported_claims": unsupported_claims,
                "num_unsupported_claims": len(unsupported_claims),
                "scale": self.scale,
                "threshold": self.threshold,
                "strict": self.strict,
                "provider": self.llm.provider,
                "model": self.llm.model,
            },
        )
