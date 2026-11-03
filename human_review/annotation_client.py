"""Langfuse Score API Client for Human-in-the-Loop Annotations.

Records human review decisions (`human_review_quality`: 0.0=reject, 0.5=needs_edit, 1.0=approve)
and converts human annotations into Ground Truth pairs for Confusion Matrix evaluation.
"""
from __future__ import annotations

from typing import List, Optional, Tuple
from config.compat import BaseModel, Field

from config.settings import settings


class HumanAnnotationRecord(BaseModel):
    """Represents a human annotation submitted via Langfuse or Eval API."""

    trace_id: str
    score: float = Field(
        ge=0.0, le=1.0, description="0.0 = reject (issue present), 0.5 = needs_edit, 1.0 = approve (clean)"
    )
    comment: str = Field(default="")
    system_flagged_issue: bool = Field(
        default=False,
        description="Whether fact_checker/evaluator flagged this trace prior to human review.",
    )

    @property
    def ground_truth_has_issue(self) -> bool:
        """A human score < 0.8 indicates the response had a real quality/hallucination issue."""
        return self.score < 0.8


class LangfuseAnnotationClient:
    """Wrapper around Langfuse Score API to persist human annotations and quality scores."""

    def __init__(
        self,
        host: Optional[str] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.host = host or settings.langfuse_host
        self.public_key = public_key or settings.langfuse_public_key
        self.secret_key = secret_key or settings.langfuse_secret_key
        self._local_history: List[HumanAnnotationRecord] = []

    def submit_human_score(
        self,
        trace_id: str,
        score: float,
        comment: str = "",
        system_flagged_issue: bool = True,
    ) -> HumanAnnotationRecord:
        """Submits human_review_quality score to Langfuse and stores locally for Confusion Matrix."""
        record = HumanAnnotationRecord(
            trace_id=trace_id,
            score=score,
            comment=comment,
            system_flagged_issue=system_flagged_issue,
        )
        self._local_history.append(record)

        try:
            from langfuse import Langfuse  # type: ignore

            client = Langfuse(
                host=self.host,
                public_key=self.public_key,
                secret_key=self.secret_key,
            )
            client.score(
                trace_id=trace_id,
                name="human_review_quality",
                value=score,
                comment=comment,
                data_type="NUMERIC",
            )
        except Exception:
            # Non-blocking when running offline or in test environments
            pass

        return record

    def write_automated_scores(
        self,
        trace_id: str,
        faithfulness: float,
        answer_relevance: float,
        csr: float,
    ) -> bool:
        """Writes offline Ragas & Deterministic scores back to Langfuse trace."""
        try:
            from langfuse import Langfuse  # type: ignore

            client = Langfuse(
                host=self.host,
                public_key=self.public_key,
                secret_key=self.secret_key,
            )
            for name, val in (
                ("faithfulness", faithfulness),
                ("answer_relevance", answer_relevance),
                ("constraint_satisfaction_rate", csr),
            ):
                client.score(
                    trace_id=trace_id,
                    name=name,
                    value=val,
                    data_type="NUMERIC",
                )
            return True
        except Exception:
            return False

    def to_confusion_matrix_pairs(self) -> List[Tuple[bool, bool]]:
        """Returns list of (system_flagged_issue, ground_truth_has_issue) pairs."""
        return [
            (rec.system_flagged_issue, rec.ground_truth_has_issue)
            for rec in self._local_history
        ]
