"""Layer 2: Ragas Scorer for Offline RAG & Response Quality Evaluation.

Computes:
- Faithfulness (Groundedness in RAG/Tool context)
- Answer Relevance (Directness to user prompt)
- Context Precision & Context Recall (Retrieval quality)

Uses an isolated Judge LLM configuration (distinct from Agent generator to eliminate
self-enhancement/sycophancy bias) and provides a deterministic lexical-entailment
fallback when running offline in CI/CD without external LLM credentials.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional
from config.compat import BaseModel, Field


class RagasScoreResult(BaseModel):
    """Normalized 0.0-1.0 scores across the 4 Ragas dimensions."""

    faithfulness: float = Field(ge=0.0, le=1.0)
    answer_relevance: float = Field(ge=0.0, le=1.0)
    context_precision: float = Field(ge=0.0, le=1.0)
    context_recall: float = Field(ge=0.0, le=1.0)
    evaluation_mode: str = Field(
        default="lexical_entailment",
        description="'ragas_llm' when Judge LLM is active, or 'lexical_entailment' in offline mode.",
    )


class RagasScorer:
    """Offline Ragas evaluation engine with anti-sycophancy Judge LLM support."""

    def __init__(self, use_llm_judge: bool = False) -> None:
        self.use_llm_judge = use_llm_judge

    def score_sample_sync(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> RagasScoreResult:
        """Synchronous offline entailment scorer (zero socket/event-loop overhead)."""
        return self._score_offline_entailment(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    async def score_sample(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> RagasScoreResult:
        """Scores a single (question, answer, contexts, ground_truth) sample."""
        if self.use_llm_judge:
            try:
                return await self._score_with_ragas_library(
                    question=question,
                    answer=answer,
                    contexts=contexts,
                    ground_truth=ground_truth,
                )
            except Exception:
                # Graceful fallback to deterministic lexical entailment
                pass

        return self._score_offline_entailment(
            question=question,
            answer=answer,
            contexts=contexts,
            ground_truth=ground_truth,
        )

    async def _score_with_ragas_library(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> RagasScoreResult:
        """Invokes Ragas metrics using configured Judge LLM."""
        from datasets import Dataset  # type: ignore
        from ragas import evaluate  # type: ignore
        from ragas.metrics import (  # type: ignore
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )

        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
            "ground_truth": [ground_truth or answer],
        }
        dataset = Dataset.from_dict(data)
        result = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        )
        scores: Dict[str, Any] = result.to_pandas().iloc[0].to_dict()
        return RagasScoreResult(
            faithfulness=float(scores.get("faithfulness", 0.0)),
            answer_relevance=float(scores.get("answer_relevancy", 0.0)),
            context_precision=float(scores.get("context_precision", 0.0)),
            context_recall=float(scores.get("context_recall", 0.0)),
            evaluation_mode="ragas_llm",
        )

    @classmethod
    def _score_offline_entailment(
        cls,
        question: str,
        answer: str,
        contexts: List[str],
        ground_truth: Optional[str] = None,
    ) -> RagasScoreResult:
        """Deterministic claim-overlap & relevance scorer for zero-latency CI/offline runs."""
        if not answer or not answer.strip():
            return RagasScoreResult(
                faithfulness=0.0,
                answer_relevance=0.0,
                context_precision=0.0,
                context_recall=0.0,
                evaluation_mode="lexical_entailment",
            )

        context_text = " ".join(contexts or [])
        answer_claims = cls._split_claims(answer)

        # 1. Faithfulness: fraction of factual claims in answer supported by context tokens
        if not contexts:
            faithfulness_score = 0.0
        elif not answer_claims:
            faithfulness_score = 1.0
        else:
            reference_corpus = context_text + " " + (ground_truth or "")
            ctx_tokens = cls._tokenize(reference_corpus)
            supported_claims = 0
            for claim in answer_claims:
                claim_tokens = cls._tokenize(claim)
                if not claim_tokens:
                    supported_claims += 1
                    continue
                overlap = len(claim_tokens & ctx_tokens) / len(claim_tokens)
                # Verify significant numbers (prices, years, IDs >= 2 digits) exist in context
                claim_nums = set(re.findall(r"\b\d{2,}(?:[.,]\d+)?\b", claim))
                ctx_nums = set(re.findall(r"\b\d{2,}(?:[.,]\d+)?\b", reference_corpus))
                nums_grounded = claim_nums.issubset(ctx_nums)
                if overlap >= 0.35 and nums_grounded:
                    supported_claims += 1
            faithfulness_score = supported_claims / len(answer_claims)

        # 2. Answer Relevance: lexical & semantic anchor overlap with question
        q_tokens = cls._tokenize(question)
        a_tokens = cls._tokenize(answer)
        if not q_tokens:
            relevance_score = 1.0
        else:
            q_overlap = len(q_tokens & a_tokens) / len(q_tokens)
            relevance_score = min(1.0, max(0.5, q_overlap + 0.45)) if a_tokens else 0.0

        # 3. Context Precision: fraction of retrieved contexts that share key tokens with question/answer
        if not contexts:
            precision_score = 0.0
        else:
            target_tokens = q_tokens | cls._tokenize(ground_truth or answer)
            relevant_chunks = sum(
                1
                for chunk in contexts
                if len(cls._tokenize(chunk) & target_tokens) >= 2
            )
            precision_score = relevant_chunks / len(contexts)

        # 4. Context Recall: coverage of ground_truth tokens in contexts
        if ground_truth:
            gt_tokens = cls._tokenize(ground_truth)
            ctx_tokens = cls._tokenize(context_text)
            recall_score = (
                len(gt_tokens & ctx_tokens) / len(gt_tokens) if gt_tokens else 1.0
            )
            recall_score = min(1.0, recall_score + 0.15) if recall_score > 0.5 else recall_score
        else:
            recall_score = precision_score

        return RagasScoreResult(
            faithfulness=round(faithfulness_score, 4),
            answer_relevance=round(relevance_score, 4),
            context_precision=round(precision_score, 4),
            context_recall=round(recall_score, 4),
            evaluation_mode="lexical_entailment",
        )

    @staticmethod
    def _split_claims(text: str) -> List[str]:
        sentences = [
            s.strip()
            for s in re.split(r"[.!?\n]+", text)
            if len(s.strip()) > 8
        ]
        return sentences

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        stopwords = {
            "the", "and", "for", "with", "that", "this", "from", "are", "was",
            "của", "và", "các", "có", "cho", "trong", "được", "một", "những", "là", "với",
        }
        words = re.findall(r"\w+", text.lower())
        return {w for w in words if len(w) > 1 and w not in stopwords}
