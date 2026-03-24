"""ML-based scorer example."""

from mankinds_eval import Scorer
from mankinds_eval.methods.ml import EmbeddingsSimilarity, PIIDetection, SentimentAnalysis

# Requires: pip install mankinds-eval[ml]

scorer = Scorer(
    name="ml_example",
    methods=[
        EmbeddingsSimilarity(threshold=0.7, compare="output_vs_expected"),
        SentimentAnalysis(expected_sentiment="positive"),
        PIIDetection(fail_on_detection=True),
    ],
)

data = [
    {
        "input": "How was your experience?",
        "output": "It was great! Very helpful service.",
        "expected": "Positive feedback about the service",
    }
]

results = scorer.run_sync(data)
print(results.summary)
