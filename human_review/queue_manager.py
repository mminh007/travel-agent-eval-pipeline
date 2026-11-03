"""Human-in-the-Loop (HITL) Trigger Engine & Review Queue Manager.

Implements the exact trigger matrix from LLMOps_Quality_Control_Pipeline.md:
- low_consistency: consistency_score < 0.75
- contradictions_found: len(contradictions) > 0
- ungrounded_claims: len(ungrounded_claims) > 2
- forced_accept: rework_count >= max_rework_cycles or forced_accept is True
- low_eval_score: min(tool_quality_score, evidence_quality_score) < 5
- deterministic_failure: zero-LLM rule check failed
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, List, Literal, Optional, Tuple
from config.compat import BaseModel, Field

from ab_testing.trace_fetcher import EvaluatedTraceRecord
from evaluators.deterministic_checker import DeterministicCheckResult


class HITLTriggerResult(BaseModel):
    """Result of evaluating a trace against HITL trigger rules."""

    should_review: bool
    reasons: List[str] = Field(default_factory=list)
    severity: Literal["normal", "high", "critical"] = "normal"


class HumanReviewTask(BaseModel):
    """Represents a trace queued for human annotation."""

    trace_id: str
    session_id: str
    model_variant: str
    prompt: str
    response_draft: str
    trigger_reasons: List[str]
    severity: Literal["normal", "high", "critical"]
    status: Literal["pending", "approved", "needs_edit", "rejected"] = "pending"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    reviewer_score: Optional[float] = None
    reviewer_comment: Optional[str] = None


def should_trigger_human_review(
    trace: EvaluatedTraceRecord,
    deterministic_result: Optional[DeterministicCheckResult] = None,
    consistency_threshold: float = 0.75,
    max_ungrounded_claims: int = 2,
    min_eval_score: int = 5,
    max_rework_cycles: int = 2,
) -> Tuple[bool, List[str]]:
    """Evaluates all HITL triggers and returns (triggered, reasons)."""
    reasons: List[str] = []

    if (
        trace.consistency_score is not None
        and trace.consistency_score < consistency_threshold
    ):
        reasons.append("low_consistency")

    if len(trace.contradictions) > 0:
        reasons.append("contradictions_found")

    if len(trace.ungrounded_claims) > max_ungrounded_claims:
        reasons.append("ungrounded_claims")

    if trace.forced_accept or trace.rework_count >= max_rework_cycles:
        reasons.append("forced_accept")

    eval_scores = [
        s
        for s in (trace.tool_quality_score, trace.evidence_quality_score)
        if s is not None
    ]
    if eval_scores and min(eval_scores) < min_eval_score:
        reasons.append("low_eval_score")

    if deterministic_result is not None and not deterministic_result.passed_all:
        reasons.append("deterministic_failure")

    return len(reasons) > 0, reasons


class HumanReviewQueueManager:
    """Manages pending Human Review tasks and tracks resolution states."""

    def __init__(self, max_rework_cycles: int = 2) -> None:
        self.max_rework_cycles = max_rework_cycles
        self._tasks: Dict[str, HumanReviewTask] = {}

    def evaluate_and_enqueue(
        self,
        trace: EvaluatedTraceRecord,
        deterministic_result: Optional[DeterministicCheckResult] = None,
    ) -> Optional[HumanReviewTask]:
        """Checks HITL triggers and enqueues the trace if any trigger fires."""
        triggered, reasons = should_trigger_human_review(
            trace=trace,
            deterministic_result=deterministic_result,
            max_rework_cycles=self.max_rework_cycles,
        )
        if not triggered:
            return None

        if "forced_accept" in reasons or "contradictions_found" in reasons:
            severity: Literal["normal", "high", "critical"] = "critical"
        elif "deterministic_failure" in reasons or "low_consistency" in reasons:
            severity = "high"
        else:
            severity = "normal"

        task = HumanReviewTask(
            trace_id=trace.trace_id,
            session_id=trace.session_id,
            model_variant=trace.model_variant,
            prompt=trace.prompt,
            response_draft=trace.response,
            trigger_reasons=reasons,
            severity=severity,
        )
        self._tasks[trace.trace_id] = task
        return task

    def resolve_task(
        self, trace_id: str, score: float, comment: str = ""
    ) -> Optional[HumanReviewTask]:
        """Updates task status based on human reviewer score (1.0=approve, 0.5=edit, 0.0=reject)."""
        task = self._tasks.get(trace_id)
        if not task:
            return None
        task.reviewer_score = score
        task.reviewer_comment = comment
        if score >= 0.8:
            task.status = "approved"
        elif score >= 0.4:
            task.status = "needs_edit"
        else:
            task.status = "rejected"
        return task

    def list_pending(self) -> List[HumanReviewTask]:
        return [t for t in self._tasks.values() if t.status == "pending"]

    def pending_count(self) -> int:
        return len(self.list_pending())
