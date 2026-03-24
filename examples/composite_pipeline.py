"""Example demonstrating composite pipelines with nested evaluation methods.

This example shows how to combine multiple evaluation methods using:
- CompositeMethod with different aggregation modes (all, weighted, sequential)
- ConditionalMethod for conditional evaluation logic
- Nested composites for complex evaluation pipelines
"""

from __future__ import annotations

from mankinds_eval import Scorer
from mankinds_eval.core import Sample
from mankinds_eval.methods.composite import CompositeMethod, ConditionalMethod
from mankinds_eval.methods.heuristic import (
    ContainsAll,
    ContainsAny,
    FuzzyMatch,
    NoRefusal,
    RegexMatch,
    TextLength,
    WordCount,
)

# =============================================================================
# Sample Data
# =============================================================================
# Mix of short and long responses to demonstrate conditional evaluation

data = [
    {
        "input": "What is the capital of France?",
        "output": "The capital of France is Paris. It is a beautiful city known for the "
        "Eiffel Tower, world-class museums like the Louvre, and its rich cultural heritage.",
        "expected": "Paris is the capital of France.",
    },
    {
        "input": "Name three primary colors.",
        "output": "Red, blue, yellow.",
        "expected": "The three primary colors are red, blue, and yellow.",
    },
    {
        "input": "Explain machine learning briefly.",
        "output": "Machine learning is a subset of artificial intelligence where algorithms "
        "learn patterns from data to make predictions. It involves training models on "
        "datasets and evaluating their performance on unseen data.",
        "expected": "Machine learning uses algorithms that learn from data to make predictions.",
    },
    {
        "input": "What is 2 + 2?",
        "output": "4",
        "expected": "4",
    },
    {
        "input": "List some benefits of exercise.",
        "output": "Exercise improves cardiovascular health, strengthens muscles, boosts mental "
        "health, helps maintain healthy weight, increases energy levels, and promotes "
        "better sleep quality.",
        "expected": "Exercise benefits include better heart health, stronger muscles, and "
        "improved mood.",
    },
]


# =============================================================================
# Example 1: Basic CompositeMethod with "all" mode
# =============================================================================
# All methods must pass for the composite to pass. Score is the average.

basic_composite = CompositeMethod(
    name="BasicQualityCheck",
    methods=[
        NoRefusal(),  # Ensures response is not a refusal
        TextLength(min_length=1, max_length=500, unit="chars"),  # Length constraints
        ContainsAny(keywords=["is", "are", "the", "a"]),  # Contains common words
    ],
    mode="all",
)


# =============================================================================
# Example 2: CompositeMethod with "weighted" mode and custom weights
# =============================================================================
# Combines scores using weights. Pass/fail determined by threshold.
# Note: Weights are specified by method name (class name attribute).

weighted_composite = CompositeMethod(
    name="WeightedSimilarity",
    methods=[
        FuzzyMatch(
            threshold=0.5,
            algorithm="token_set_ratio",
            target="expected",
        ),
        ContainsAll(
            keywords=["is", "the"],
            case_sensitive=False,
        ),
        TextLength(
            min_length=1,
            max_length=500,
            unit="chars",
        ),
    ],
    mode="weighted",
    weights={
        "FuzzyMatch": 2.0,  # Fuzzy matching is most important
        "ContainsAll": 1.0,  # Keyword coverage is secondary
        "TextLength": 0.5,  # Length check is supplementary
    },
    threshold=0.6,  # 60% weighted score needed to pass
)


# =============================================================================
# Example 3: CompositeMethod with "sequential" mode
# =============================================================================
# Methods run in order. Each method can access previous results via method_results.
# Useful for multi-stage validation pipelines where each stage builds on the previous.
# Note: In sequential mode, sample.method_results is enriched after each method,
# allowing later methods to potentially access earlier results.

sequential_composite = CompositeMethod(
    name="SequentialValidation",
    methods=[
        NoRefusal(),  # Stage 1: Ensure response is not a refusal
        WordCount(min_words=1, max_words=200),  # Stage 2: Check word count
        ContainsAny(keywords=["is", "are", "the", "a"]),  # Stage 3: Has common words
    ],
    mode="sequential",
    threshold=0.7,  # 70% weighted score needed to pass
    stop_on_fail=False,  # Continue even if a stage fails
)


# =============================================================================
# Example 4: ConditionalMethod usage
# =============================================================================
# Applies different evaluation based on response characteristics.


def is_long_response(sample: Sample) -> bool:
    """Check if the response is longer than 50 characters."""
    return len(sample.output) > 50


# For long responses, apply stricter evaluation
long_response_evaluator = CompositeMethod(
    name="LongResponseCheck",
    methods=[
        TextLength(min_length=50, max_length=500, unit="chars"),
        WordCount(min_words=10, max_words=100),
        FuzzyMatch(threshold=0.4, algorithm="token_set_ratio", target="expected"),
    ],
    mode="all",
)

# For short responses, apply simpler evaluation
short_response_evaluator = CompositeMethod(
    name="ShortResponseCheck",
    methods=[
        TextLength(min_length=1, max_length=100, unit="chars"),
        NoRefusal(),
    ],
    mode="all",
)

conditional_method = ConditionalMethod(
    name="AdaptiveEvaluation",
    condition=is_long_response,
    if_true=long_response_evaluator,
    if_false=short_response_evaluator,
)


# =============================================================================
# Example 5: Nested composite pipeline
# =============================================================================
# Composites can contain other composites for complex evaluation hierarchies.
# The outer composite references inner composites by their custom names.

# Inner composite: Format validation
format_validation = CompositeMethod(
    name="FormatValidation",
    methods=[
        NoRefusal(),
        TextLength(min_length=1, max_length=1000, unit="chars"),
        RegexMatch(pattern=r"^[A-Z]", flags=0),  # Starts with capital letter
    ],
    mode="all",
)

# Inner composite: Content quality
content_quality = CompositeMethod(
    name="ContentQuality",
    methods=[
        FuzzyMatch(threshold=0.4, algorithm="token_set_ratio", target="expected"),
        ContainsAny(keywords=["is", "are", "the", "a", "an"]),
    ],
    mode="weighted",
    weights={
        "FuzzyMatch": 2.0,  # Semantic similarity is most important
        "ContainsAny": 1.0,  # Having common words is secondary
    },
    threshold=0.5,
)

# Outer composite: Combines format and content validation
# Note: Inner composites have custom names set via the name parameter,
# so we use those names in the weights dictionary.
nested_composite = CompositeMethod(
    name="NestedPipeline",
    methods=[
        format_validation,
        content_quality,
    ],
    mode="weighted",
    weights={
        "FormatValidation": 1.0,
        "ContentQuality": 2.0,  # Content quality weighted more heavily
    },
    threshold=0.6,
)


# =============================================================================
# Create Scorer and Run Evaluation
# =============================================================================

scorer = Scorer(
    name="Composite Pipeline Demo",
    methods=[
        basic_composite,
        weighted_composite,
        sequential_composite,
        conditional_method,
        nested_composite,
    ],
)

print("Running composite pipeline evaluation...")
print("=" * 60)
results = scorer.run_sync(data)


# =============================================================================
# Display Results
# =============================================================================

print(f"\nEvaluated {results.meta['sample_count']} samples")
print(f"Duration: {results.meta['duration_seconds']:.2f}s")

# Overall summary
print("\n" + "=" * 60)
print("SUMMARY BY METHOD")
print("=" * 60)

for method_name, stats in results.summary.items():
    mean = stats.get("mean_score", "N/A")
    pass_rate = stats.get("pass_rate")
    if isinstance(mean, float):
        mean = f"{mean:.3f}"
    if pass_rate is not None:
        print(f"  {method_name}: mean={mean}, pass_rate={pass_rate:.1%}")
    else:
        print(f"  {method_name}: mean={mean}")

# Detailed results showing sub_results in metadata
print("\n" + "=" * 60)
print("DETAILED RESULTS (showing sub_results for nested composite)")
print("=" * 60)

for i, sample_result in enumerate(results.results):
    print(f"\nSample {i + 1}: {data[i]['input'][:40]}...")
    print(f"  Output: {data[i]['output'][:50]}...")

    # Get the nested composite result from the methods dictionary
    methods = sample_result.get("methods", {})
    nested_result = methods.get("NestedPipeline")

    if nested_result:
        print("\n  NestedPipeline:")
        score = nested_result.get("score")
        print(f"    Score: {score:.3f}" if score is not None else "    Score: N/A")
        print(f"    Passed: {nested_result.get('passed')}")
        print(f"    Reason: {nested_result.get('reason')}")

        # Show sub_results from metadata
        metadata = nested_result.get("metadata", {})
        if "sub_results" in metadata:
            print("    Sub-results:")
            for sub in metadata["sub_results"]:
                sub_name = sub.get("method_name", "Unknown")
                sub_score = sub.get("score")
                sub_passed = sub.get("passed")
                score_str = f"{sub_score:.3f}" if sub_score is not None else "N/A"
                print(f"      - {sub_name}: score={score_str}, passed={sub_passed}")

# Show conditional method results
print("\n" + "=" * 60)
print("CONDITIONAL METHOD DETAILS")
print("=" * 60)

for i, sample_result in enumerate(results.results):
    methods = sample_result.get("methods", {})
    adaptive_result = methods.get("AdaptiveEvaluation")

    if adaptive_result:
        metadata = adaptive_result.get("metadata", {})
        condition = metadata.get("condition_result", "N/A")
        executed = metadata.get("executed_method", "N/A")
        score = adaptive_result.get("score")
        print(f"\nSample {i + 1}:")
        print(f"  Condition (is_long_response): {condition}")
        print(f"  Executed method: {executed}")
        print(f"  Score: {score}")

# Save outputs
results.to_json("composite_results.json")
results.to_html("composite_scorecard.html")

print("\n" + "=" * 60)
print("Results saved to composite_results.json and composite_scorecard.html")
