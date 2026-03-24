# mankinds-eval

**mankinds-eval** is an open source Python library for AI evaluation. It provides a unified framework for scoring AI model outputs using heuristic methods, machine learning techniques, and LLM-as-Judge approaches.

## Key Features

- **Multiple evaluation methods**: Heuristic (exact match, fuzzy match, regex, text metrics), ML-based (embeddings similarity, sentiment analysis, PII detection), and LLM-as-Judge (single criterion, multi-criteria, pairwise, consensus)
- **Flexible data loading**: Load samples from CSV, JSONL, JSON files, or HuggingFace Datasets
- **Configurable scorers**: Combine multiple methods into evaluation pipelines with YAML/JSON configuration
- **Rich output formats**: Export results to JSON or interactive HTML scorecards
- **Async support**: Run evaluations concurrently for better performance
- **CLI interface**: Run evaluations from the command line without writing code
- **Extensible generators**: Create custom test generators for red teaming and adversarial testing

## Quick Example

```python
from mankinds_eval import Scorer
from mankinds_eval.methods.heuristic import ExactMatch, FuzzyMatch

# Create a scorer with multiple methods
scorer = Scorer(
    name="my_evaluation",
    methods=[
        ExactMatch(),
        FuzzyMatch(threshold=0.8),
    ]
)

# Sample data
data = [
    {"input": "What is 2+2?", "output": "4", "expected": "4"},
    {"input": "Capital of France?", "output": "Paris", "expected": "Paris"},
]

# Run evaluation
results = scorer.run_sync(data)

# Export results
results.to_json("results.json")
results.to_html("scorecard.html")
```

## Getting Started

- [Installation](installation.md) - Install mankinds-eval and optional dependencies
- [Quick Start](quickstart.md) - Get up and running in minutes
- [User Guide](guide/data.md) - Comprehensive documentation

## License

mankinds-eval is released under the Apache 2.0 License.
