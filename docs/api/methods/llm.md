# LLM-as-Judge Methods

LLM-based evaluation methods using various providers. Requires the `[llm]` extra:

```bash
pip install mankinds-eval[llm]
```

## Provider

::: mankinds_eval.methods.llm.providers.LLMProvider
    options:
      members:
        - __init__
        - complete

## Pre-built Judges

::: mankinds_eval.methods.llm.faithfulness.Faithfulness

::: mankinds_eval.methods.llm.relevancy.AnswerRelevancy

::: mankinds_eval.methods.llm.coherence.Coherence

::: mankinds_eval.methods.llm.helpfulness.Helpfulness

::: mankinds_eval.methods.llm.correctness.Correctness

## Configurable Judges

::: mankinds_eval.methods.llm.single.SingleCriterionJudge

::: mankinds_eval.methods.llm.multi.MultiCriteriaJudge

::: mankinds_eval.methods.llm.pairwise.PairwiseJudge

::: mankinds_eval.methods.llm.consensus.ConsensusJudge
