"""Shared pytest fixtures for mankinds_eval tests."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from mankinds_eval.core.result import MethodResult
from mankinds_eval.core.sample import Sample


@pytest.fixture
def sample_data() -> list[dict[str, Any]]:
    """Return a list of sample dicts with input, output, expected fields."""
    return [
        {
            "input": "What is the capital of France?",
            "output": "The capital of France is Paris.",
            "expected": "Paris",
        },
        {
            "input": "Summarize the following text.",
            "output": "This is a summary of the text.",
            "expected": "A concise summary.",
        },
        {
            "input": "Translate to Spanish: Hello",
            "output": "Hola",
            "expected": "Hola",
        },
    ]


@pytest.fixture
def sample() -> Sample:
    """Return a Sample instance with basic fields."""
    return Sample(
        input="What is 2 + 2?",
        output="The answer is 4.",
        expected="4",
    )


@pytest.fixture
def sample_with_all_fields() -> Sample:
    """Return a Sample instance with all optional fields populated."""
    return Sample(
        input="What is the weather like?",
        output="The weather is sunny today.",
        expected="Sunny",
        context="Weather data: sunny, 25C, low humidity",
        conversation=[
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ],
        metadata={"source": "test", "category": "weather"},
    )


@pytest.fixture
def method_result() -> MethodResult:
    """Return a MethodResult instance with basic fields."""
    return MethodResult(
        method_name="test_method",
        score=0.85,
        passed=True,
        reason="Score exceeded threshold",
    )


@pytest.fixture
def method_result_with_error() -> MethodResult:
    """Return a MethodResult instance with an error."""
    return MethodResult(
        method_name="failed_method",
        score=None,
        passed=None,
        reason=None,
        error="Evaluation failed due to timeout",
    )


@pytest.fixture
def tmp_json_file(tmp_path: Path, sample_data: list[dict[str, Any]]) -> Path:
    """Create a temporary JSON file with sample data."""
    file_path = tmp_path / "test_data.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_data, f)
    return file_path


@pytest.fixture
def tmp_csv_file(tmp_path: Path, sample_data: list[dict[str, Any]]) -> Path:
    """Create a temporary CSV file with sample data."""
    file_path = tmp_path / "test_data.csv"
    fieldnames = ["input", "output", "expected"]
    with open(file_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in sample_data:
            writer.writerow({k: row[k] for k in fieldnames})
    return file_path


@pytest.fixture
def tmp_jsonl_file(tmp_path: Path, sample_data: list[dict[str, Any]]) -> Path:
    """Create a temporary JSONL file with sample data."""
    file_path = tmp_path / "test_data.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for item in sample_data:
            f.write(json.dumps(item) + "\n")
    return file_path
