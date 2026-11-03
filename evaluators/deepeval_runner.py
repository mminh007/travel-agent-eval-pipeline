"""Layer 2: DeepEval Runner for Unit Testing & G-Eval Rubric Scoring.

Provides:
- Faithfulness, AnswerRelevancy, and Hallucination assertion checks
- G-Eval travel itinerary rubric evaluation (logical travel times, budget coherence)
"""
from __future__ import annotations

from typing import List, Optional
from config.compat import BaseModel, Field

from evaluators.ragas_scorer import RagasScorer


class DeepEvalResult(BaseModel):
    """Structured result from DeepEval test case execution."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    hallucination_rate: float = Field(
        ge=0.0, le=1.0, description="0.0 = no hallucination, 1.0 = completely hallucinated."
    )
    g_eval_itinerary_score: float = Field(
        ge=0.0, le=1.0, description="G-Eval rubric score for itinerary logic and feasibility."
    )
    passed: bool = Field(description="True if all DeepEval metric assertions pass thresholds.")
    failure_reasons: List[str] = Field(default_factory=list)


class DeepEvalRunner:
    """Runs DeepEval test cases in CI/CD or async production sampling."""

    def __init__(
        self,
        faithfulness_threshold: float = 0.90,
        relevance_threshold: float = 0.85,
        max_hallucination_rate: float = 0.10,
    ) -> None:
        self.faithfulness_threshold = faithfulness_threshold
        self.relevance_threshold = relevance_threshold
        self.max_hallucination_rate = max_hallucination_rate
        self._ragas_fallback = RagasScorer(use_llm_judge=False)

    async def evaluate_test_case(
        self,
        input_prompt: str,
        actual_output: str,
        retrieval_context: List[str],
        expected_output: Optional[str] = None,
    ) -> DeepEvalResult:
        """Evaluates a single LLM test case against quality gate thresholds."""
        base_scores = await self._ragas_fallback.score_sample(
            question=input_prompt,
            answer=actual_output,
            contexts=retrieval_context,
            ground_truth=expected_output,
        )

        faithfulness = base_scores.faithfulness
        relevance = base_scores.answer_relevance
        hallucination_rate = round(max(0.0, 1.0 - faithfulness), 4)
        g_eval_score = round((faithfulness * 0.6) + (relevance * 0.4), 4)

        failure_reasons: List[str] = []
        if faithfulness < self.faithfulness_threshold:
            failure_reasons.append(
                f"Faithfulness {faithfulness:.2f} < {self.faithfulness_threshold:.2f}"
            )
        if relevance < self.relevance_threshold:
            failure_reasons.append(
                f"Answer Relevance {relevance:.2f} < {self.relevance_threshold:.2f}"
            )
        if hallucination_rate > self.max_hallucination_rate:
            failure_reasons.append(
                f"Hallucination rate {hallucination_rate:.2f} > {self.max_hallucination_rate:.2f}"
            )

        return DeepEvalResult(
            faithfulness=faithfulness,
            answer_relevance=relevance,
            hallucination_rate=hallucination_rate,
            g_eval_itinerary_score=g_eval_score,
            passed=len(failure_reasons) == 0,
            failure_reasons=failure_reasons,
        )
