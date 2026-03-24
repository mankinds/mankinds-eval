"""Basic scorer example using heuristic methods."""

from mankinds_eval import Scorer
from mankinds_eval.methods.heuristic import ExactMatch, FuzzyMatch, TextLength

# Sample data
data = [
    {
        "input": "What is Python?",
        "output": "Python is a programming language.",
        "expected": "Python is a programming language.",
    },
    {"input": "What is 2+2?", "output": "The answer is 4.", "expected": "4"},
]

# Create scorer
scorer = Scorer(
    name="basic_example",
    methods=[
        ExactMatch(),
        FuzzyMatch(threshold=0.8, algorithm="ratio"),
        TextLength(min_length=10, max_length=200),
    ],
)

# Run evaluation
results = scorer.run_sync(data)

# Output
results.to_json("results.json")
results.to_html("scorecard.html")
print(f"Evaluated {results.meta['sample_count']} samples")
