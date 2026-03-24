# Scorers

Scorers combine multiple evaluation methods into a unified evaluation pipeline. This guide covers creating, configuring, and running scorers.

## Creating a Scorer

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.heuristic import ExactMatch, FuzzyMatch

scorer = Scorer(
    name="my_scorer",
    methods=[
        ExactMatch(),
        FuzzyMatch(threshold=0.8),
    ]
)
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Name for the scorer |
| `methods` | `list[Method]` | Required | List of evaluation methods |
| `max_concurrent` | `int` | `10` | Maximum concurrent evaluations |
| `fail_fast` | `bool` | `False` | Stop on first method failure |
| `timeout` | `float` | `None` | Timeout per sample (seconds) |

## Running Evaluations

### Synchronous Execution

```python
results = scorer.run_sync(data)
```

### Asynchronous Execution

```python
import asyncio

async def evaluate():
    results = await scorer.run(data)
    return results

results = asyncio.run(evaluate())
```

### With Progress Callback

```python
def on_progress(completed: int, total: int, sample_id: str):
    print(f"Progress: {completed}/{total}")

results = scorer.run_sync(data, on_progress=on_progress)
```

## Input Data Formats

Scorers accept multiple data formats:

### List of Dictionaries

```python
data = [
    {"input": "Q1", "output": "A1", "expected": "E1"},
    {"input": "Q2", "output": "A2", "expected": "E2"},
]
results = scorer.run_sync(data)
```

### List of Sample Objects

```python
from mankinds_eval.core import Sample

data = [
    Sample(input="Q1", output="A1", expected="E1"),
    Sample(input="Q2", output="A2", expected="E2"),
]
results = scorer.run_sync(data)
```

### From Data Loaders

```python
from mankinds_eval.data import load_jsonl

data = load_jsonl("evaluations.jsonl")
results = scorer.run_sync(data)
```

## EvaluationResult

The `run` method returns an `EvaluationResult` object:

```python
results = scorer.run_sync(data)

# Access metadata
print(results.meta["name"])          # Scorer name
print(results.meta["sample_count"])  # Number of samples
print(results.meta["method_count"])  # Number of methods
print(results.meta["created_at"])    # Timestamp

# Access results
for sample_result in results.results:
    print(sample_result.sample_id)
    print(sample_result.input)
    print(sample_result.output)
    for method_name, method_result in sample_result.method_results.items():
        print(f"  {method_name}: {method_result.passed} ({method_result.score})")

# Summary statistics
print(results.summary)
```

### Summary Statistics

```python
summary = results.summary

print(summary["total_samples"])
print(summary["pass_rate"])  # Overall pass rate
print(summary["method_stats"])  # Per-method statistics
```

## Exporting Results

### JSON Export

```python
results.to_json("results.json")

# Or get as dict
data = results.to_dict()
```

JSON structure:

```json
{
  "meta": {
    "name": "my_scorer",
    "sample_count": 100,
    "method_count": 3,
    "created_at": "2024-01-15T10:30:00Z"
  },
  "results": [
    {
      "sample_id": "0",
      "input": "...",
      "output": "...",
      "expected": "...",
      "method_results": {
        "ExactMatch": {"passed": true, "score": 1.0},
        "FuzzyMatch": {"passed": true, "score": 0.95}
      }
    }
  ],
  "summary": {
    "total_samples": 100,
    "pass_rate": 0.85,
    "method_stats": {}
  }
}
```

### HTML Export

```python
results.to_html("scorecard.html")

# With custom title
results.to_html("scorecard.html", title="Q&A Evaluation Results")
```

The HTML scorecard includes:

- Summary statistics and pass rates
- Per-method breakdowns
- Filterable sample results table
- Expandable details for each sample

## Concurrency Settings

Control concurrent evaluations for performance and rate limiting:

```python
# High concurrency for heuristic methods
scorer = Scorer(
    name="fast_scorer",
    methods=[ExactMatch(), FuzzyMatch()],
    max_concurrent=50,
)

# Low concurrency for LLM methods (API rate limits)
scorer = Scorer(
    name="llm_scorer",
    methods=[SingleCriterionJudge(...)],
    max_concurrent=5,
)
```

## Error Handling

### Fail Fast Mode

Stop evaluation on first failure:

```python
scorer = Scorer(
    name="strict_scorer",
    methods=[...],
    fail_fast=True,
)
```

### Timeout

Set per-sample timeout:

```python
scorer = Scorer(
    name="timed_scorer",
    methods=[...],
    timeout=30.0,  # 30 seconds per sample
)
```

### Handling Errors in Results

```python
results = scorer.run_sync(data)

for sample_result in results.results:
    for method_name, method_result in sample_result.method_results.items():
        if method_result.error:
            print(f"Error in {method_name}: {method_result.error}")
```

## Loading from Config

Create scorers from YAML/JSON config files:

```python
from mankinds_eval import Scorer

scorer = Scorer.from_config("scorer_config.yaml")
results = scorer.run_sync(data)
```

See [Config Files](config.md) for configuration format details.

## Combining Multiple Scorers

Run multiple scorers and merge results:

```python
from mankinds_eval import Scorer

heuristic_scorer = Scorer(name="heuristic", methods=[...])
llm_scorer = Scorer(name="llm", methods=[...])

# Run both
heuristic_results = heuristic_scorer.run_sync(data)
llm_results = llm_scorer.run_sync(data)

# Merge results
combined = heuristic_results.merge(llm_results)
combined.to_html("combined_scorecard.html")
```

## Presets

mankinds-eval provides pre-configured method combinations for common evaluation scenarios.

### RAGScorer

Combines Faithfulness, AnswerRelevancy, and Coherence for RAG pipeline evaluation.

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.presets import RAGScorer

# Create methods with preset
methods = RAGScorer.create(
    provider="openai",
    model="gpt-4o-mini",
    threshold=0.7,
)

# Use with Scorer
scorer = Scorer(name="rag_evaluation", methods=methods)
results = scorer.run_sync(data)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `provider` | `str` | `"openai"` | LLM provider |
| `model` | `str` | `None` | Model name (uses provider default if None) |
| `threshold` | `float` | `0.7` | Score threshold for all methods |
| `include_faithfulness` | `bool` | `True` | Include Faithfulness method |
| `include_relevancy` | `bool` | `True` | Include AnswerRelevancy method |
| `include_coherence` | `bool` | `True` | Include Coherence method |

#### Get Weights for Aggregation

```python
weights = RAGScorer.get_weights(
    faithfulness_weight=0.4,
    relevancy_weight=0.3,
    coherence_weight=0.3,
)
# Returns: {"Faithfulness": 0.4, "AnswerRelevancy": 0.3, "Coherence": 0.3}
```

### SafetyScorer

Combines PII detection, toxicity detection, and refusal detection for safety checks.

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.presets import SafetyScorer

# Create methods with preset
methods = SafetyScorer.create(
    check_pii=True,
    check_toxicity=True,
    check_refusal=True,
    toxicity_threshold=0.5,
)

# Use with Scorer
scorer = Scorer(name="safety_check", methods=methods)
results = scorer.run_sync(data)
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `check_pii` | `bool` | `True` | Include PII detection (requires `[ml]`) |
| `check_toxicity` | `bool` | `True` | Include toxicity detection (requires `[ml]`) |
| `check_refusal` | `bool` | `True` | Include refusal detection |
| `toxicity_threshold` | `float` | `0.5` | Threshold for toxicity detection |
| `toxicity_model` | `str` | `None` | Model for toxicity (uses default if None) |
| `device` | `str` | `None` | Device for ML models (`"cpu"`, `"cuda"`, `"mps"`) |

#### Get Check Names

```python
names = SafetyScorer.get_check_names(
    check_pii=True,
    check_toxicity=True,
    check_refusal=True,
)
# Returns: ["PIIDetection", "Toxicity", "NoRefusal"]
```

### Example: Complete RAG Evaluation Pipeline

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.presets import RAGScorer, SafetyScorer
from mankinds_eval.methods.heuristic import TextLength

# Combine presets with custom methods
rag_methods = RAGScorer.create(provider="openai", threshold=0.7)
safety_methods = SafetyScorer.create(check_pii=True, check_toxicity=True)

scorer = Scorer(
    name="complete_rag_eval",
    methods=[
        *rag_methods,
        *safety_methods,
        TextLength(min_length=50, max_length=1000),
    ],
    max_concurrent=5,
)

data = [
    {
        "input": "What is machine learning?",
        "output": "Machine learning is a subset of AI...",
        "context": "Machine learning (ML) is a field of AI...",
    }
]

results = scorer.run_sync(data)
results.to_html("rag_scorecard.html")
```
