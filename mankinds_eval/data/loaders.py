"""Data loaders for various file formats and sources."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from mankinds_eval.core import Sample


def load_samples(source: Any) -> list[Sample]:
    """Load samples from various sources.

    Args:
        source: Can be:
            - list[Sample]: Direct list of Sample objects (returned as-is)
            - list[dict]: Direct list of sample dicts
            - str or Path ending with .csv: CSV file
            - str or Path ending with .jsonl: JSONL file
            - str or Path ending with .json: JSON file (array of objects)
            - HuggingFace Dataset object

    Returns:
        List of Sample objects.

    Raises:
        ValueError: If source format is unsupported or cannot be determined.
        FileNotFoundError: If file path does not exist.
    """
    # Handle list of dicts or Sample objects directly
    if isinstance(source, list):
        # If already a list of Sample objects, return as-is
        if source and isinstance(source[0], Sample):
            return source
        return load_from_dicts(source)

    # Handle file paths
    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return load_from_csv(path)
        elif suffix == ".jsonl":
            return load_from_jsonl(path)
        elif suffix == ".json":
            return load_from_json(path)
        else:
            raise ValueError(
                f"Unsupported file format: '{suffix}'. Supported formats: .csv, .jsonl, .json"
            )

    # Try HuggingFace Dataset
    # Check if it looks like a HuggingFace Dataset by checking for common attributes
    if _is_huggingface_dataset(source):
        return load_from_huggingface(source)

    raise ValueError(
        f"Unsupported source type: {type(source).__name__}. "
        "Expected list[dict], file path (str/Path), or HuggingFace Dataset."
    )


def load_from_dicts(data: list[dict[str, Any]]) -> list[Sample]:
    """Load samples from a list of dictionaries.

    Args:
        data: List of dictionaries with sample data.
            Each dict must have 'input' and 'output' keys.

    Returns:
        List of Sample objects.
    """
    return [Sample.from_dict(d) for d in data]


def load_from_csv(path: str | Path) -> list[Sample]:
    """Load samples from a CSV file.

    Expected columns: input, output, expected (optional), context (optional),
    conversation (optional, JSON string), metadata (optional, JSON string)

    Args:
        path: Path to the CSV file.

    Returns:
        List of Sample objects.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If required columns are missing.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    samples: list[Sample] = []

    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Validate required columns
        if reader.fieldnames is None:
            raise ValueError(f"CSV file is empty or has no headers: {path}")

        fieldnames = set(reader.fieldnames)
        required = {"input", "output"}
        missing = required - fieldnames
        if missing:
            raise ValueError(
                f"CSV file missing required columns: {missing}. Found columns: {fieldnames}"
            )

        for row_num, row in enumerate(reader, start=2):
            try:
                sample_dict = _csv_row_to_dict(row)
                samples.append(Sample.from_dict(sample_dict))
            except (ValueError, KeyError) as e:
                raise ValueError(f"Error in CSV row {row_num}: {e}") from e

    return samples


def _csv_row_to_dict(row: dict[str, str]) -> dict[str, Any]:
    """Convert a CSV row to a sample dictionary.

    Handles JSON parsing for conversation and metadata fields.

    Args:
        row: Dictionary from CSV reader.

    Returns:
        Dictionary suitable for Sample.from_dict().
    """
    result: dict[str, Any] = {
        "input": row["input"],
        "output": row["output"],
    }

    # Optional string fields
    if row.get("expected"):
        result["expected"] = row["expected"]
    if row.get("context"):
        result["context"] = row["context"]

    # Optional JSON fields
    if row.get("conversation"):
        result["conversation"] = json.loads(row["conversation"])
    if row.get("metadata"):
        result["metadata"] = json.loads(row["metadata"])
    if row.get("method_results"):
        result["method_results"] = json.loads(row["method_results"])

    return result


def load_from_jsonl(path: str | Path) -> list[Sample]:
    """Load samples from a JSONL file (one JSON object per line).

    Args:
        path: Path to the JSONL file.

    Returns:
        List of Sample objects.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If JSON parsing fails or data is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {path}")

    samples: list[Sample] = []

    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            try:
                data = json.loads(line)
                samples.append(Sample.from_dict(data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}") from e
            except (ValueError, KeyError) as e:
                raise ValueError(f"Error in JSONL line {line_num}: {e}") from e

    return samples


def load_from_json(path: str | Path) -> list[Sample]:
    """Load samples from a JSON file (array of objects).

    Args:
        path: Path to the JSON file.

    Returns:
        List of Sample objects.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If JSON is not an array or data is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    with open(path, encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in file {path}: {e}") from e

    if not isinstance(data, list):
        raise ValueError(f"JSON file must contain an array of objects, got {type(data).__name__}")

    samples: list[Sample] = []
    for i, item in enumerate(data):
        try:
            samples.append(Sample.from_dict(item))
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error in JSON array item {i}: {e}") from e

    return samples


def _is_huggingface_dataset(obj: Any) -> bool:
    """Check if an object is a HuggingFace Dataset or DatasetDict.

    Args:
        obj: Object to check.

    Returns:
        True if object appears to be a HuggingFace Dataset.
    """
    type_name = type(obj).__name__
    module = type(obj).__module__

    # Check for datasets library types
    if module.startswith("datasets"):
        return type_name in ("Dataset", "DatasetDict")

    return False


def load_from_huggingface(dataset: Any) -> list[Sample]:
    """Load samples from a HuggingFace Dataset.

    Expects dataset with 'input' and 'output' columns at minimum.
    Handles both Dataset and DatasetDict (uses first split).

    Args:
        dataset: HuggingFace Dataset or DatasetDict object.

    Returns:
        List of Sample objects.

    Raises:
        ImportError: If datasets library is not installed.
        ValueError: If required columns are missing or dataset type is invalid.
    """
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as e:
        raise ImportError(
            "HuggingFace datasets library is required for this loader. "
            "Install it with: pip install datasets"
        ) from e

    # Handle DatasetDict by using the first split
    if isinstance(dataset, DatasetDict):
        if len(dataset) == 0:
            raise ValueError("DatasetDict is empty")
        first_split = next(iter(dataset.keys()))
        dataset = dataset[first_split]

    if not isinstance(dataset, Dataset):
        raise ValueError(
            f"Expected HuggingFace Dataset or DatasetDict, got {type(dataset).__name__}"
        )

    # Validate required columns
    columns = set(dataset.column_names)
    required = {"input", "output"}
    missing = required - columns
    if missing:
        raise ValueError(f"Dataset missing required columns: {missing}. Found columns: {columns}")

    samples: list[Sample] = []
    for i, row in enumerate(dataset):
        try:
            sample_dict: dict[str, Any] = {
                "input": row["input"],
                "output": row["output"],
            }

            # Add optional fields if present
            if "expected" in columns and row.get("expected") is not None:
                sample_dict["expected"] = row["expected"]
            if "context" in columns and row.get("context") is not None:
                sample_dict["context"] = row["context"]
            if "conversation" in columns and row.get("conversation") is not None:
                sample_dict["conversation"] = row["conversation"]
            if "metadata" in columns and row.get("metadata") is not None:
                sample_dict["metadata"] = row["metadata"]
            if "method_results" in columns and row.get("method_results") is not None:
                sample_dict["method_results"] = row["method_results"]

            samples.append(Sample.from_dict(sample_dict))
        except (ValueError, KeyError) as e:
            raise ValueError(f"Error in dataset row {i}: {e}") from e

    return samples
