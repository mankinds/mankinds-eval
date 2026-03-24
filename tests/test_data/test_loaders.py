"""Tests for data loaders module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from mankinds_eval.core import Sample
from mankinds_eval.data.loaders import (
    load_from_csv,
    load_from_dicts,
    load_from_json,
    load_from_jsonl,
    load_samples,
)

if TYPE_CHECKING:
    pass


# Path to test fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestLoadFromDicts:
    """Tests for load_from_dicts function."""

    def test_load_basic_samples(self) -> None:
        """Test loading samples from a list of dictionaries."""
        data = [
            {"input": "Hello", "output": "World"},
            {"input": "Foo", "output": "Bar"},
        ]
        samples = load_from_dicts(data)

        assert len(samples) == 2
        assert all(isinstance(s, Sample) for s in samples)
        assert samples[0].input == "Hello"
        assert samples[0].output == "World"
        assert samples[1].input == "Foo"
        assert samples[1].output == "Bar"

    def test_load_with_expected(self) -> None:
        """Test loading samples with expected field."""
        data = [
            {"input": "Q", "output": "A", "expected": "A"},
        ]
        samples = load_from_dicts(data)

        assert len(samples) == 1
        assert samples[0].expected == "A"

    def test_load_with_optional_fields(self) -> None:
        """Test loading samples with optional fields."""
        data = [
            {
                "input": "Q",
                "output": "A",
                "expected": "A",
                "context": "Some context",
                "metadata": {"key": "value"},
            },
        ]
        samples = load_from_dicts(data)

        assert len(samples) == 1
        assert samples[0].context == "Some context"
        assert samples[0].metadata == {"key": "value"}

    def test_empty_list(self) -> None:
        """Test loading from empty list."""
        samples = load_from_dicts([])
        assert samples == []

    def test_missing_required_field(self) -> None:
        """Test that missing required field raises error."""
        data = [{"input": "Hello"}]  # missing 'output'
        with pytest.raises(KeyError):
            load_from_dicts(data)

    def test_empty_input_raises_error(self) -> None:
        """Test that empty input raises ValueError."""
        data = [{"input": "", "output": "response"}]
        with pytest.raises(ValueError, match="input must be a non-empty string"):
            load_from_dicts(data)

    def test_empty_output_raises_error(self) -> None:
        """Test that empty output raises ValueError."""
        data = [{"input": "question", "output": ""}]
        with pytest.raises(ValueError, match="output must be a non-empty string"):
            load_from_dicts(data)


class TestLoadFromJson:
    """Tests for load_from_json function."""

    def test_load_json_file(self) -> None:
        """Test loading samples from JSON file."""
        path = FIXTURES_DIR / "test_data.json"
        samples = load_from_json(path)

        assert len(samples) == 2
        assert samples[0].input == "What is Python?"
        assert samples[0].output == "Python is a programming language."
        assert samples[0].expected == "Python is a programming language."
        assert samples[1].input == "What is 2+2?"
        assert samples[1].output == "The answer is 4."
        assert samples[1].expected == "4"

    def test_load_json_file_as_string_path(self) -> None:
        """Test loading JSON file with string path."""
        path = str(FIXTURES_DIR / "test_data.json")
        samples = load_from_json(path)
        assert len(samples) == 2

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="JSON file not found"):
            load_from_json("/nonexistent/path/file.json")

    def test_invalid_json(self, tmp_path: Path) -> None:
        """Test that invalid JSON raises ValueError."""
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("{invalid json}")

        with pytest.raises(ValueError, match="Invalid JSON"):
            load_from_json(invalid_file)

    def test_json_not_array(self, tmp_path: Path) -> None:
        """Test that non-array JSON raises ValueError."""
        non_array_file = tmp_path / "non_array.json"
        non_array_file.write_text('{"key": "value"}')

        with pytest.raises(ValueError, match="must contain an array"):
            load_from_json(non_array_file)


class TestLoadFromJsonl:
    """Tests for load_from_jsonl function."""

    def test_load_jsonl_file(self) -> None:
        """Test loading samples from JSONL file."""
        path = FIXTURES_DIR / "test_data.jsonl"
        samples = load_from_jsonl(path)

        assert len(samples) == 2
        assert samples[0].input == "What is Python?"
        assert samples[0].output == "Python is a programming language."
        assert samples[0].expected == "Python is a programming language."
        assert samples[1].input == "What is 2+2?"
        assert samples[1].output == "The answer is 4."
        assert samples[1].expected == "4"

    def test_load_jsonl_file_as_string_path(self) -> None:
        """Test loading JSONL file with string path."""
        path = str(FIXTURES_DIR / "test_data.jsonl")
        samples = load_from_jsonl(path)
        assert len(samples) == 2

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="JSONL file not found"):
            load_from_jsonl("/nonexistent/path/file.jsonl")

    def test_invalid_json_line(self, tmp_path: Path) -> None:
        """Test that invalid JSON line raises ValueError."""
        invalid_file = tmp_path / "invalid.jsonl"
        invalid_file.write_text('{"input": "q", "output": "a"}\n{invalid}\n')

        with pytest.raises(ValueError, match="Invalid JSON on line 2"):
            load_from_jsonl(invalid_file)

    def test_skips_empty_lines(self, tmp_path: Path) -> None:
        """Test that empty lines are skipped."""
        file_with_empty = tmp_path / "with_empty.jsonl"
        file_with_empty.write_text(
            '{"input": "q1", "output": "a1"}\n\n{"input": "q2", "output": "a2"}\n'
        )

        samples = load_from_jsonl(file_with_empty)
        assert len(samples) == 2


class TestLoadFromCsv:
    """Tests for load_from_csv function."""

    def test_load_csv_file(self) -> None:
        """Test loading samples from CSV file."""
        path = FIXTURES_DIR / "test_data.csv"
        samples = load_from_csv(path)

        assert len(samples) == 2
        assert samples[0].input == "What is Python?"
        assert samples[0].output == "Python is a programming language."
        assert samples[0].expected == "Python is a programming language."
        assert samples[1].input == "What is 2+2?"
        assert samples[1].output == "The answer is 4."
        assert samples[1].expected == "4"

    def test_load_csv_file_as_string_path(self) -> None:
        """Test loading CSV file with string path."""
        path = str(FIXTURES_DIR / "test_data.csv")
        samples = load_from_csv(path)
        assert len(samples) == 2

    def test_file_not_found(self) -> None:
        """Test that FileNotFoundError is raised for non-existent file."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            load_from_csv("/nonexistent/path/file.csv")

    def test_missing_required_columns(self, tmp_path: Path) -> None:
        """Test that missing required columns raises ValueError."""
        missing_cols_file = tmp_path / "missing_cols.csv"
        missing_cols_file.write_text("input,other\nq,x\n")

        with pytest.raises(ValueError, match="missing required columns"):
            load_from_csv(missing_cols_file)

    def test_empty_csv(self, tmp_path: Path) -> None:
        """Test that empty CSV raises ValueError."""
        empty_file = tmp_path / "empty.csv"
        empty_file.write_text("")

        with pytest.raises(ValueError, match="empty or has no headers"):
            load_from_csv(empty_file)

    def test_csv_with_json_metadata(self, tmp_path: Path) -> None:
        """Test CSV with JSON metadata column."""
        csv_with_meta = tmp_path / "with_meta.csv"
        csv_with_meta.write_text('input,output,metadata\nq,a,"{""key"": ""value""}"\n')

        samples = load_from_csv(csv_with_meta)
        assert len(samples) == 1
        assert samples[0].metadata == {"key": "value"}


class TestLoadSamples:
    """Tests for load_samples dispatcher function."""

    def test_dispatch_list(self) -> None:
        """Test dispatching to load_from_dicts for list input."""
        data = [{"input": "q", "output": "a"}]
        samples = load_samples(data)
        assert len(samples) == 1
        assert samples[0].input == "q"

    def test_dispatch_json_file(self) -> None:
        """Test dispatching to load_from_json for .json file."""
        path = FIXTURES_DIR / "test_data.json"
        samples = load_samples(path)
        assert len(samples) == 2

    def test_dispatch_jsonl_file(self) -> None:
        """Test dispatching to load_from_jsonl for .jsonl file."""
        path = FIXTURES_DIR / "test_data.jsonl"
        samples = load_samples(path)
        assert len(samples) == 2

    def test_dispatch_csv_file(self) -> None:
        """Test dispatching to load_from_csv for .csv file."""
        path = FIXTURES_DIR / "test_data.csv"
        samples = load_samples(path)
        assert len(samples) == 2

    def test_dispatch_string_path(self) -> None:
        """Test dispatching with string path."""
        path = str(FIXTURES_DIR / "test_data.json")
        samples = load_samples(path)
        assert len(samples) == 2

    def test_unsupported_file_format(self) -> None:
        """Test that unsupported file format raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported file format"):
            load_samples("/some/path/file.txt")

    def test_unsupported_source_type(self) -> None:
        """Test that unsupported source type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported source type"):
            load_samples(12345)  # type: ignore[arg-type]

    def test_file_not_found_json(self) -> None:
        """Test FileNotFoundError for non-existent JSON file."""
        with pytest.raises(FileNotFoundError):
            load_samples("/nonexistent/file.json")

    def test_file_not_found_csv(self) -> None:
        """Test FileNotFoundError for non-existent CSV file."""
        with pytest.raises(FileNotFoundError):
            load_samples("/nonexistent/file.csv")

    def test_file_not_found_jsonl(self) -> None:
        """Test FileNotFoundError for non-existent JSONL file."""
        with pytest.raises(FileNotFoundError):
            load_samples("/nonexistent/file.jsonl")
