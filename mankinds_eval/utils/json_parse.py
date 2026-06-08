"""Robust JSON extraction from LLM responses.

LLM judges are instructed to answer with a single JSON object, but in practice
they sometimes wrap it in markdown fences, prepend prose, embed braces inside
string values, or get truncated when they hit the ``max_tokens`` ceiling mid
response. A naive ``re.search(r"\\{[^{}]*\\}")`` fails on all of these.

``extract_json_object`` handles them by locating the first ``{``, scanning for a
balanced top-level object while honoring string literals, and repairing a
truncated tail (unterminated string and unclosed brackets) before parsing.
"""

from __future__ import annotations

import json
import re
from typing import Any

_FENCE_OPEN_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*\n?")
_FENCE_CLOSE_RE = re.compile(r"\n?```\s*$")


def _strip_code_fences(text: str) -> str:
    """Remove a surrounding markdown code fence if present."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = _FENCE_OPEN_RE.sub("", stripped)
        stripped = _FENCE_CLOSE_RE.sub("", stripped)
    return stripped.strip()


def _scan_balanced_object(text: str, start: int) -> str | None:
    """Return the first balanced ``{...}`` object starting at ``start``.

    Braces inside string literals are ignored. Returns ``None`` if the object
    is never closed (truncated response).
    """
    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair of a JSON object truncated mid response.

    Closes an unterminated string and any unclosed objects/arrays so that a
    response cut off by the model's token limit (typically inside the verbose
    ``reason`` field) can still be parsed. The score and earlier fields are
    already present at that point, so the salvaged object stays meaningful.
    """
    stack: list[str] = []
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if in_string:
            if ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch == "}":
            if stack and stack[-1] == "{":
                stack.pop()
        elif ch == "]":
            if stack and stack[-1] == "[":
                stack.pop()

    repaired = text.rstrip()
    if in_string:
        repaired += '"'
    else:
        # Drop a dangling ``"key":`` whose value got cut off, then any leftover
        # trailing separator, so the salvaged object stays valid JSON.
        repaired = re.sub(r',?\s*"(?:[^"\\]|\\.)*"\s*:\s*$', "", repaired)
        repaired = re.sub(r"[,:]\s*$", "", repaired)
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def extract_json_object(response: str) -> dict[str, Any]:
    """Extract a JSON object from a raw LLM response.

    Args:
        response: Raw LLM response string.

    Returns:
        The parsed JSON object as a dict.

    Raises:
        ValueError: If no JSON object can be located ("No JSON found in
            response") or if the located content cannot be parsed even after
            repair ("Invalid JSON in response").
    """
    text = _strip_code_fences(response)
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No JSON found in response: {response}")

    candidate = _scan_balanced_object(text, start)
    if candidate is not None:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    repaired = _repair_truncated_json(text[start:])
    try:
        parsed = json.loads(repaired)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in response: {response}") from exc

    if not isinstance(parsed, dict):
        raise ValueError(f"Invalid JSON in response: {response}")
    return parsed
