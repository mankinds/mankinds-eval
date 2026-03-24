"""Generate a demo scorecard with rich sample data."""

from mankinds_eval import Scorer
from mankinds_eval.methods.heuristic import (
    ROUGE,
    ContainsAll,
    FuzzyMatch,
    TextLength,
)

# Rich sample data for demo
data = [
    {
        "input": "What is the capital of France?",
        "output": (
            "The capital of France is Paris. It is known as the"
            " City of Light and is famous for the Eiffel Tower,"
            " the Louvre Museum, and its rich cultural heritage."
        ),
        "expected": "Paris is the capital of France.",
    },
    {
        "input": "Explain photosynthesis in simple terms.",
        "output": (
            "Photosynthesis is the process by which plants"
            " convert sunlight, water, and carbon dioxide into"
            " glucose and oxygen. This happens in the"
            " chloroplasts of plant cells."
        ),
        "expected": (
            "Photosynthesis is how plants make food using"
            " sunlight, water, and CO2, producing glucose"
            " and oxygen."
        ),
    },
    {
        "input": "What are the benefits of exercise?",
        "output": (
            "Regular exercise improves cardiovascular health,"
            " strengthens muscles, boosts mental health, helps"
            " maintain healthy weight, and increases energy"
            " levels."
        ),
        "expected": (
            "Exercise benefits include better heart health,"
            " stronger muscles, improved mood, weight"
            " management, and more energy."
        ),
    },
    {
        "input": "How does machine learning work?",
        "output": (
            "Machine learning is a subset of AI where"
            " algorithms learn patterns from data to make"
            " predictions or decisions without being explicitly"
            " programmed for each task."
        ),
        "expected": "Machine learning uses algorithms that learn from data to make predictions.",
    },
    {
        "input": "What is Python used for?",
        "output": (
            "Python is used for web development, data science,"
            " machine learning, automation, scripting, and"
            " scientific computing. It's known for its simple"
            " syntax."
        ),
        "expected": (
            "Python is a versatile programming language used"
            " for web development, data analysis, AI, and"
            " automation."
        ),
    },
]

# Create scorer with multiple methods
scorer = Scorer(
    name="QA Evaluation Demo",
    methods=[
        FuzzyMatch(threshold=0.6, algorithm="token_set_ratio", target="expected"),
        ROUGE(threshold=0.4, aggregate="rougeL"),
        TextLength(min_length=50, max_length=300, unit="chars"),
        ContainsAll(keywords=["is", "the"], case_sensitive=False),
    ],
)

# Run evaluation
print("Running evaluation...")
results = scorer.run_sync(data)

# Output
results.to_json("demo_results.json")
results.to_html("demo_scorecard.html")

print(f"Evaluated {results.meta['sample_count']} samples")
print(f"Duration: {results.meta['duration_seconds']:.2f}s")
print("\nSummary:")
for method, stats in results.summary.items():
    mean = stats.get("mean_score", "N/A")
    pass_rate = stats.get("pass_rate")
    if isinstance(mean, float):
        mean = f"{mean:.3f}"
    if pass_rate is not None:
        print(f"  {method}: mean={mean}, pass_rate={pass_rate:.1%}")
    else:
        print(f"  {method}: mean={mean}")

print("\nScorecard saved to demo_scorecard.html")
