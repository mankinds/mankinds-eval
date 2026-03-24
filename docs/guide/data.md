# Samples and Data

This guide covers data handling in mankinds-eval, including the Sample structure, data loading, and working with multi-turn conversations.

## Sample Structure

The `Sample` class represents a single evaluation unit:

```python
from mankinds_eval.core import Sample

sample = Sample(
    input="What is the capital of France?",
    output="The capital of France is Paris.",
    expected="Paris",
    metadata={"category": "geography", "difficulty": "easy"}
)
```

### Fields

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `input` | `str` | Yes | The input prompt or question |
| `output` | `str` | Yes | The AI model's response |
| `expected` | `str` | No | Expected or reference answer |
| `metadata` | `dict` | No | Additional metadata for filtering or analysis |

### Using Dictionaries

You can also use plain dictionaries instead of Sample objects:

```python
data = [
    {"input": "Question 1", "output": "Answer 1", "expected": "Reference 1"},
    {"input": "Question 2", "output": "Answer 2", "expected": "Reference 2"},
]

results = scorer.run_sync(data)
```

## Loading Data

mankinds-eval provides loaders for common data formats.

### CSV Files

```python
from mankinds_eval.data import load_csv

# Basic loading (expects input, output, expected columns)
data = load_csv("evaluations.csv")

# Custom column mapping
data = load_csv(
    "evaluations.csv",
    input_col="question",
    output_col="response",
    expected_col="reference"
)
```

Example CSV format:

```csv
input,output,expected
"What is 2+2?","4","4"
"Capital of France?","Paris","Paris"
```

### JSONL Files

```python
from mankinds_eval.data import load_jsonl

data = load_jsonl("evaluations.jsonl")
```

Example JSONL format:

```json
{"input": "What is 2+2?", "output": "4", "expected": "4"}
{"input": "Capital of France?", "output": "Paris", "expected": "Paris"}
```

### JSON Files

```python
from mankinds_eval.data import load_json

# Load from a JSON array
data = load_json("evaluations.json")

# Specify a key if data is nested
data = load_json("evaluations.json", key="samples")
```

Example JSON format:

```json
{
  "samples": [
    {"input": "What is 2+2?", "output": "4", "expected": "4"},
    {"input": "Capital of France?", "output": "Paris", "expected": "Paris"}
  ]
}
```

### HuggingFace Datasets

```python
from mankinds_eval.data import load_hf_dataset

# Load from HuggingFace Hub
data = load_hf_dataset(
    "dataset_name",
    split="test",
    input_col="question",
    output_col="answer",
    expected_col="reference"
)
```

## Multi-Turn Conversations

For evaluating multi-turn conversations, use the `messages` field:

```python
sample = Sample(
    input="",  # Can be empty for multi-turn
    output="I'd be happy to help with that!",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, how are you?"},
        {"role": "assistant", "content": "I'm doing well, thank you!"},
        {"role": "user", "content": "Can you help me with Python?"},
    ],
    metadata={"turn_count": 4}
)
```

Some methods (particularly LLM judges) can analyze the full conversation context when `messages` is provided.

## Filtering Samples

Filter samples by metadata:

```python
# Filter samples before evaluation
filtered_data = [s for s in data if s.get("metadata", {}).get("category") == "math"]

results = scorer.run_sync(filtered_data)
```

## Creating Samples Programmatically

Generate samples from your AI model:

```python
from mankinds_eval.core import Sample

def create_samples(questions: list[str], model) -> list[Sample]:
    samples = []
    for question in questions:
        response = model.generate(question)
        samples.append(Sample(
            input=question,
            output=response,
            metadata={"model": model.name}
        ))
    return samples
```

## Data Validation

mankinds-eval validates samples before evaluation:

```python
from mankinds_eval.data import validate_samples

# Validate data format
errors = validate_samples(data)
if errors:
    print("Validation errors:", errors)
```

Required fields are checked, and warnings are raised for missing optional fields that some methods may need.
