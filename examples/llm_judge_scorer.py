"""LLM-as-Judge scorer example."""

from mankinds_eval import Scorer
from mankinds_eval.methods.llm import MultiCriteriaJudge, SingleCriterionJudge

# Requires: pip install mankinds-eval[llm]
# Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable

scorer = Scorer(
    name="llm_judge_example",
    methods=[
        SingleCriterionJudge(
            provider="openai",
            criterion="Is the response helpful and accurate?",
            scale="1-5",
            threshold=3,
        ),
        MultiCriteriaJudge(
            provider="openai",
            criteria={
                "accuracy": {"description": "Is the information correct?", "weight": 0.5},
                "clarity": {
                    "description": "Is the response clear and well-structured?",
                    "weight": 0.3,
                },
                "completeness": {
                    "description": "Does it fully answer the question?",
                    "weight": 0.2,
                },
            },
            scale="1-5",
        ),
    ],
)

data = [
    {
        "input": "Explain photosynthesis",
        "output": "Photosynthesis is how plants make food using sunlight...",
    },
]

results = scorer.run_sync(data)
results.to_json("llm_judge_results.json")
