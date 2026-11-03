"""Langfuse Trace Puller & Stratified Sampler for A/B Testing.

Implements the Event-Driven Data Bridge:
- Deterministic session variant assignment: Hash(SessionId) % 100
- Stratified sampling from Langfuse traces by `experiment.variant` (`model_a` vs `model_b`)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import random
from typing import Any, Dict, List, Optional
from config.compat import BaseModel, Field

from config.settings import settings


def assign_ab_variant(session_id: str, traffic_split: float = 0.10) -> str:
    """Deterministic assignment: same session_id always maps to the same variant."""
    hash_val = int(hashlib.md5(session_id.encode("utf-8")).hexdigest(), 16)
    return "model_b" if (hash_val % 100) < (traffic_split * 100) else "model_a"


class EvaluatedTraceRecord(BaseModel):
    """Normalized trace record extracted from Langfuse for offline scoring & A/B analysis."""

    trace_id: str
    session_id: str
    model_variant: str = Field(default="model_a", description="'model_a' or 'model_b'")
    intent_category: str = Field(default="hotel_booking")
    complexity: str = Field(default="medium")
    prompt: str = Field(default="")
    response: str = Field(default="")
    tool_observations: List[Dict[str, Any]] = Field(default_factory=list)

    # Response Quality Metrics
    faithfulness: Optional[float] = None
    answer_relevance: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    consistency_score: Optional[float] = None
    constraint_satisfaction_rate: float = 1.0

    # Evaluator Dimension Scores (1-10)
    tool_quality_score: Optional[int] = None
    evidence_quality_score: Optional[int] = None
    citation_quality_score: Optional[int] = None
    freshness_score: Optional[int] = None

    # Conversation & Operational Metrics
    latency_seconds: float = 1.0
    ttft_seconds: float = 0.3
    turn_to_resolution: int = 2
    rework_count: int = 0
    forced_accept: bool = False
    contradictions: List[str] = Field(default_factory=list)
    ungrounded_claims: List[str] = Field(default_factory=list)
    thumbs_down: bool = False
    goal_completed: bool = True
    human_review_score: Optional[float] = None


class LangfuseTraceFetcher:
    """Fetches and stratified-samples production traces from the Langfuse server."""

    def __init__(
        self,
        host: Optional[str] = None,
        public_key: Optional[str] = None,
        secret_key: Optional[str] = None,
    ) -> None:
        self.host = host or settings.langfuse_host
        self.public_key = public_key or settings.langfuse_public_key
        self.secret_key = secret_key or settings.langfuse_secret_key

    def _get_client(self):
        from langfuse import Langfuse  # type: ignore

        return Langfuse(
            host=self.host,
            public_key=self.public_key,
            secret_key=self.secret_key,
        )

    def pull_sample_traces(
        self,
        experiment_name: str = "model_comparison_v1",
        sample_rate: float = 0.10,
        hours_back: int = 1,
        seed: Optional[int] = None,
    ) -> List[EvaluatedTraceRecord]:
        """Pulls a stratified random sample of traces across model_a and model_b."""
        try:
            client = self._get_client()
            from_ts = datetime.now(timezone.utc) - timedelta(hours=hours_back)
            raw_traces = client.fetch_traces(
                tags=[experiment_name],
                from_timestamp=from_ts,
                limit=1000,
            ).data
            records = [self._parse_langfuse_trace(t) for t in raw_traces]
        except Exception:
            return []

        return self.stratified_sample(records, sample_rate=sample_rate, seed=seed)

    @staticmethod
    def stratified_sample(
        records: List[EvaluatedTraceRecord],
        sample_rate: float = 0.10,
        seed: Optional[int] = None,
    ) -> List[EvaluatedTraceRecord]:
        """Preserves exact variant proportions when down-sampling traces."""
        if sample_rate >= 1.0 or not records:
            return list(records)

        rng = random.Random(seed)
        by_variant: Dict[str, List[EvaluatedTraceRecord]] = {}
        for rec in records:
            by_variant.setdefault(rec.model_variant, []).append(rec)

        sampled: List[EvaluatedTraceRecord] = []
        for _, group in sorted(by_variant.items()):
            k = max(1, int(round(len(group) * sample_rate))) if group else 0
            sampled.extend(rng.sample(group, min(k, len(group))))
        return sampled

    @staticmethod
    def _parse_langfuse_trace(trace_obj: Any) -> EvaluatedTraceRecord:
        metadata: Dict[str, Any] = getattr(trace_obj, "metadata", None) or {}
        scores_map: Dict[str, float] = {}
        for s in getattr(trace_obj, "scores", []) or []:
            s_name = getattr(s, "name", "")
            s_val = getattr(s, "value", None)
            if s_name and isinstance(s_val, (int, float)):
                scores_map[s_name] = float(s_val)

        session_id = getattr(trace_obj, "session_id", None) or metadata.get(
            "session_id", "unknown_session"
        )
        variant = metadata.get("experiment.variant") or metadata.get(
            "model_variant", assign_ab_variant(session_id)
        )

        return EvaluatedTraceRecord(
            trace_id=str(getattr(trace_obj, "id", "unknown_trace")),
            session_id=str(session_id),
            model_variant=str(variant),
            intent_category=str(metadata.get("intent_category", "hotel_booking")),
            complexity=str(metadata.get("complexity", "medium")),
            prompt=str(getattr(trace_obj, "input", "") or ""),
            response=str(getattr(trace_obj, "output", "") or ""),
            tool_observations=metadata.get("tool_observations", []),
            faithfulness=scores_map.get("faithfulness"),
            answer_relevance=scores_map.get("answer_relevance"),
            consistency_score=scores_map.get("consistency_score"),
            constraint_satisfaction_rate=scores_map.get("constraint_satisfaction_rate", 1.0),
            latency_seconds=float(getattr(trace_obj, "latency", 1.0) or 1.0),
            ttft_seconds=float(metadata.get("ttft_seconds", 0.3)),
            turn_to_resolution=int(metadata.get("turn_to_resolution", 2)),
            rework_count=int(metadata.get("rework_count", 0)),
            forced_accept=bool(metadata.get("forced_accept", False)),
            contradictions=metadata.get("contradictions", []),
            ungrounded_claims=metadata.get("ungrounded_claims", []),
            thumbs_down=bool(metadata.get("thumbs_down", False)),
            goal_completed=bool(metadata.get("goal_completed", True)),
            human_review_score=scores_map.get("human_review_quality"),
        )
