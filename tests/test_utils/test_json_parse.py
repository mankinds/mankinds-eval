"""Tests for robust JSON extraction from LLM responses."""

from __future__ import annotations

import pytest

from mankinds_eval.utils import extract_json_object


class TestExtractJsonObject:
    def test_clean_object(self) -> None:
        result = extract_json_object('{"score": 2, "reason": "fine"}')
        assert result == {"score": 2, "reason": "fine"}

    def test_prose_around_object(self) -> None:
        response = 'Sure, here is my answer: {"score": 4, "reason": "good"} done.'
        assert extract_json_object(response) == {"score": 4, "reason": "good"}

    def test_markdown_fence(self) -> None:
        response = '```json\n{"score": 3, "reason": "ok"}\n```'
        assert extract_json_object(response) == {"score": 3, "reason": "ok"}

    def test_brace_inside_string_value(self) -> None:
        response = '{"score": 4, "reason": "uses a {placeholder} token"}'
        assert extract_json_object(response) == {
            "score": 4,
            "reason": "uses a {placeholder} token",
        }

    def test_truncated_inside_reason_string(self) -> None:
        # Real prod case: response cut off at the token limit mid reason,
        # with no closing quote or brace.
        response = '{"score": 5, "reason": "The system clearly communicated its inability'
        result = extract_json_object(response)
        assert result["score"] == 5
        assert result["reason"].startswith("The system clearly communicated")

    def test_truncated_fenced_response(self) -> None:
        response = 'Here is my answer:\n```json\n{"score": 3, "reason": "ok'
        result = extract_json_object(response)
        assert result["score"] == 3
        assert result["reason"] == "ok"

    def test_truncated_after_separator(self) -> None:
        response = '{"score": 2, "reason":'
        result = extract_json_object(response)
        assert result["score"] == 2

    def test_nested_object_balanced(self) -> None:
        response = '{"score": 1, "analysis": {"a": 1, "b": 2}, "reason": "x"}'
        result = extract_json_object(response)
        assert result["analysis"] == {"a": 1, "b": 2}

    def test_no_json_raises(self) -> None:
        with pytest.raises(ValueError, match="No JSON found"):
            extract_json_object("This is just plain text with no JSON")

    def test_invalid_json_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid JSON"):
            extract_json_object('{"score": invalid}')
