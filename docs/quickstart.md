# Quick Start

This guide walks you through your first evaluation with mankinds-eval.

## Basic Evaluation

Create a simple scorer using heuristic methods:

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.heuristic import ExactMatch, FuzzyMatch, TextLength

# Define your evaluation data
data = [
    {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris.",
        "expected": "Paris"
    },
    {
        "input": "What is 2 + 2?",
        "output": "4",
        "expected": "4"
    },
    {
        "input": "Who wrote Romeo and Juliet?",
        "output": "William Shakespeare wrote Romeo and Juliet.",
        "expected": "Shakespeare"
    },
]

# Create a scorer with multiple methods
scorer = Scorer(
    name="qa_evaluation",
    methods=[
        ExactMatch(),                          # Check for exact matches
        FuzzyMatch(threshold=0.7),             # Allow fuzzy matching
        TextLength(min_length=1, max_length=500),  # Validate length
    ]
)

# Run the evaluation
results = scorer.run_sync(data)

# Print summary
print(f"Evaluated {results.meta['sample_count']} samples")
print(results.summary)
```

## Loading Data from Files

Load evaluation data from CSV or JSONL files:

```python
from mankinds_eval import Scorer
from mankinds_eval.data import load_csv, load_jsonl
from mankinds_eval.methods.heuristic import FuzzyMatch

# Load from CSV
data = load_csv("evaluations.csv")

# Or from JSONL
data = load_jsonl("evaluations.jsonl")

scorer = Scorer(
    name="file_evaluation",
    methods=[FuzzyMatch(threshold=0.8)]
)

results = scorer.run_sync(data)
```

## Exporting Results

Export results to JSON or HTML:

```python
# Save as JSON
results.to_json("results.json")

# Generate HTML scorecard
results.to_html("scorecard.html")
```

The JSON output contains detailed results for each sample and method:

```json
{
  "meta": {
    "name": "qa_evaluation",
    "sample_count": 3,
    "method_count": 3,
    "created_at": "2024-01-15T10:30:00Z"
  },
  "results": [
    {
      "sample_id": "0",
      "input": "What is the capital of France?",
      "output": "The capital of France is Paris.",
      "expected": "Paris",
      "method_results": {
        "ExactMatch": {"passed": false, "score": 0.0},
        "FuzzyMatch": {"passed": true, "score": 0.85},
        "TextLength": {"passed": true, "score": 1.0}
      }
    }
  ]
}
```

## Using Config Files

Define scorers in YAML for reusability:

```yaml
# scorer_config.yaml
name: qa_scorer
methods:
  - type: heuristic.FuzzyMatch
    threshold: 0.8
  - type: heuristic.TextLength
    min_length: 10
    max_length: 500
```

Run from the command line:

```bash
mankinds-eval run --config scorer_config.yaml --data evaluations.jsonl --output results.json
```

## Next Steps

- [Samples and Data](guide/data.md) - Learn about data formats and loading
- [Heuristic Methods](guide/methods/heuristic.md) - All available heuristic methods
- [ML Methods](guide/methods/ml.md) - Machine learning-based evaluations
- [LLM-as-Judge](guide/methods/llm.md) - Use LLMs to evaluate outputs
- [CLI Reference](guide/cli.md) - Command line usage
