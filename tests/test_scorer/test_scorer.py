"""Tests for Scorer class."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mankinds_eval.core import MethodResult, Sample
from mankinds_eval.methods.base import Method
from mankinds_eval.scorer.base import Scorer


class MockMethod(Method):
    """A mock method for testing."""

    name: str = "MockMethod"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output"]

    def __init__(self, score: float = 1.0, **kwargs: Any) -> None:
        """Initialize mock method with configurable score."""
        super().__init__(**kwargs)
        self._score = score
        self.config["score"] = score

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Return a mock result with configured score."""
        return MethodResult(
            method_name=self.name,
            score=self._score,
            passed=self._score >= 0.5,
            reason="Mock evaluation",
        )


class MockMethodWithExpected(Method):
    """A mock method that requires expected field."""

    name: str = "MockMethodWithExpected"
    version: str = "1.0.0"
    required_fields: list[str] = ["input", "output", "expected"]

    async def evaluate(self, sample: Sample) -> MethodResult:
        """Return pass if output matches expected."""
        is_match = sample.output == sample.expected
        return MethodResult(
            method_name=self.name,
            score=1.0 if is_match else 0.0,
            passed=is_match,
            reason="Match" if is_match else "No match",
        )


class TestScorerInitialization:
    """Tests for Scorer initialization."""

    def test_basic_initialization(self) -> None:
        """Test basic scorer initialization."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethod()],
        )

        assert scorer.name == "test_scorer"
        assert len(scorer.methods) == 1
        assert scorer.concurrency == 10  # default
        assert scorer.fail_on_error is False  # default

    def test_initialization_with_options(self) -> None:
        """Test scorer initialization with custom options."""
        scorer = Scorer(
            name="custom_scorer",
            methods=[MockMethod(), MockMethod(score=0.5)],
            concurrency=5,
            fail_on_error=True,
        )

        assert scorer.name == "custom_scorer"
        assert len(scorer.methods) == 2
        assert scorer.concurrency == 5
        assert scorer.fail_on_error is True

    def test_empty_methods_raises_error(self) -> None:
        """Test that empty methods list raises ValueError."""
        with pytest.raises(ValueError, match="At least one method is required"):
            Scorer(name="empty_scorer", methods=[])

    def test_repr(self) -> None:
        """Test string representation of scorer."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethod()],
        )
        repr_str = repr(scorer)
        assert "Scorer" in repr_str
        assert "test_scorer" in repr_str
        assert "MockMethod" in repr_str


class TestScorerToDict:
    """Tests for Scorer.to_dict serialization."""

    def test_to_dict_basic(self) -> None:
        """Test basic serialization."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethod()],
        )
        result = scorer.to_dict()

        assert result["name"] == "test_scorer"
        assert len(result["methods"]) == 1
        assert result["methods"][0]["name"] == "MockMethod"
        assert result["concurrency"] == 10
        assert result["fail_on_error"] is False

    def test_to_dict_with_options(self) -> None:
        """Test serialization with custom options."""
        scorer = Scorer(
            name="custom_scorer",
            methods=[MockMethod(score=0.8)],
            concurrency=5,
            fail_on_error=True,
        )
        result = scorer.to_dict()

        assert result["concurrency"] == 5
        assert result["fail_on_error"] is True
        assert result["methods"][0]["config"]["score"] == 0.8


class TestScorerRunSync:
    """Tests for Scorer.run_sync method."""

    def test_run_sync_with_list_data(self) -> None:
        """Test running evaluation with list of dicts."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethod()],
        )
        data = [
            {"input": "Hello", "output": "World"},
            {"input": "Foo", "output": "Bar"},
        ]

        result = scorer.run_sync(data)

        assert result.meta["scorer_name"] == "test_scorer"
        assert result.meta["sample_count"] == 2
        assert len(result.results) == 2
        assert "MockMethod" in result.summary

    def test_run_sync_with_expected(self) -> None:
        """Test running evaluation with expected field."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethodWithExpected()],
        )
        data = [
            {"input": "Q1", "output": "A", "expected": "A"},  # Match
            {"input": "Q2", "output": "B", "expected": "C"},  # No match
        ]

        result = scorer.run_sync(data)

        assert len(result.results) == 2
        # First sample should pass
        assert result.results[0]["methods"]["MockMethodWithExpected"]["passed"] is True
        # Second sample should fail
        assert result.results[1]["methods"]["MockMethodWithExpected"]["passed"] is False

    def test_run_sync_with_custom_concurrency(self) -> None:
        """Test running evaluation with custom concurrency."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethod()],
            concurrency=10,
        )
        data = [{"input": "Q", "output": "A"}]

        result = scorer.run_sync(data, concurrency=5)

        assert result.meta["concurrency"] == 5

    def test_run_sync_metadata(self) -> None:
        """Test that run_sync includes proper metadata."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethod()],
        )
        data = [{"input": "Q", "output": "A"}]

        result = scorer.run_sync(data)

        assert "created_at" in result.meta
        assert "version" in result.meta
        assert "duration_seconds" in result.meta
        assert "methods" in result.meta
        assert result.meta["sample_count"] == 1

    def test_run_sync_summary_statistics(self) -> None:
        """Test that summary statistics are computed."""
        scorer = Scorer(
            name="test_scorer",
            methods=[MockMethod(score=0.8)],
        )
        data = [
            {"input": "Q1", "output": "A1"},
            {"input": "Q2", "output": "A2"},
        ]

        result = scorer.run_sync(data)

        assert "MockMethod" in result.summary
        summary = result.summary["MockMethod"]
        assert "mean_score" in summary
        assert summary["mean_score"] == 0.8


class TestScorerFromConfig:
    """Tests for Scorer.from_config class method."""

    def test_from_config_yaml(self, tmp_path: Path) -> None:
        """Test loading scorer from YAML config file."""
        config_file = tmp_path / "scorer.yaml"
        config_file.write_text("""
name: yaml_scorer
concurrency: 5
fail_on_error: true
methods:
  - type: heuristic.ExactMatch
    case_sensitive: false
""")

        scorer = Scorer.from_config(config_file)

        assert scorer.name == "yaml_scorer"
        assert scorer.concurrency == 5
        assert scorer.fail_on_error is True
        assert len(scorer.methods) == 1
        assert scorer.methods[0].name == "ExactMatch"

    def test_from_config_json(self, tmp_path: Path) -> None:
        """Test loading scorer from JSON config file."""
        config_file = tmp_path / "scorer.json"
        config_file.write_text("""
{
    "name": "json_scorer",
    "concurrency": 8,
    "methods": [
        {"type": "heuristic.ExactMatch"}
    ]
}
""")

        scorer = Scorer.from_config(config_file)

        assert scorer.name == "json_scorer"
        assert scorer.concurrency == 8
        assert len(scorer.methods) == 1

    def test_from_config_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for non-existent config."""
        with pytest.raises(FileNotFoundError):
            Scorer.from_config("/nonexistent/config.yaml")

    def test_from_config_missing_name(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for config without name."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
methods:
  - type: heuristic.ExactMatch
""")

        with pytest.raises(ValueError, match="must have 'name'"):
            Scorer.from_config(config_file)

    def test_from_config_missing_methods(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for config without methods."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
name: invalid_scorer
""")

        with pytest.raises(ValueError, match="must have 'methods'"):
            Scorer.from_config(config_file)

    def test_from_config_empty_methods(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for empty methods list."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
name: invalid_scorer
methods: []
""")

        with pytest.raises(ValueError, match="cannot be empty"):
            Scorer.from_config(config_file)

    def test_from_config_method_without_type(self, tmp_path: Path) -> None:
        """Test that ValueError is raised for method without type."""
        config_file = tmp_path / "invalid.yaml"
        config_file.write_text("""
name: invalid_scorer
methods:
  - threshold: 0.8
""")

        with pytest.raises(ValueError, match="must have 'type'"):
            Scorer.from_config(config_file)

    def test_from_config_with_multiple_methods(self, tmp_path: Path) -> None:
        """Test loading config with multiple methods."""
        config_file = tmp_path / "scorer.yaml"
        config_file.write_text("""
name: multi_method_scorer
methods:
  - type: heuristic.ExactMatch
  - type: heuristic.ContainsAll
    keywords:
      - python
      - language
""")

        scorer = Scorer.from_config(config_file)

        assert len(scorer.methods) == 2
        assert scorer.methods[0].name == "ExactMatch"
        assert scorer.methods[1].name == "ContainsAll"
