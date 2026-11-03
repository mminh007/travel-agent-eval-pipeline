"""FastAPI Quality Control & Evaluation Server (Port 8088).

Exposes endpoints for:
- CI/CD regression webhook (`/api/v1/ci/run-gate`)
- On-demand single-trace evaluation (`/api/v1/evaluate/trace`)
- A/B testing statistical significance analysis (`/api/v1/ab-test/analyze`)
- Human-in-the-Loop queue & annotation submission (`/api/v1/hitl/queue`, `/api/v1/hitl/annotate`)
- Prometheus metrics scrape (`/metrics`)
"""
from __future__ import annotations

import json
from typing import List, Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel, Field

from ab_testing.analyzer import ABTestAnalyzer
from ab_testing.report import ABReportGenerator
from ab_testing.trace_fetcher import EvaluatedTraceRecord
from dashboard.metrics_exporter import eval_metrics
from evaluators.confusion_matrix import NodeConfusionMatrixEvaluator, NodeName
from evaluators.deterministic_checker import DeterministicChecker
from evaluators.ragas_scorer import RagasScorer
from golden_dataset.regression_report import RegressionReportBuilder
from golden_dataset.runner import GoldenDatasetRunner
from human_review.annotation_client import LangfuseAnnotationClient
from human_review.queue_manager import HumanReviewQueueManager

app = FastAPI(
    title="Travel Agent LLMOps Evaluation Pipeline API",
    version="0.1.0",
    description="Decoupled offline evaluation, A/B statistical testing, and Human-in-the-Loop review API (Port 8088).",
)

_det_checker = DeterministicChecker()
_ragas_scorer = RagasScorer(use_llm_judge=False)
_queue_manager = HumanReviewQueueManager()
_annotation_client = LangfuseAnnotationClient()
_cm_evaluator = NodeConfusionMatrixEvaluator()
_ab_analyzer = ABTestAnalyzer()


class HumanAnnotationRequest(BaseModel):
    trace_id: str
    score: float = Field(ge=0.0, le=1.0, description="0.0=reject, 0.5=needs_edit, 1.0=approve")
    comment: str = ""
    system_flagged_issue: bool = True
    node_name: NodeName = "fact_checker"


class ABAnalyzeRequest(BaseModel):
    experiment_name: str = "model_comparison_v1"
    traces: List[EvaluatedTraceRecord]


@app.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "service": "travel-agent-eval-pipeline",
        "port": 8088,
        "hitl_pending_count": _queue_manager.pending_count(),
    }


@app.get("/metrics")
async def prometheus_metrics() -> Response:
    payload = eval_metrics.render_openmetrics()
    return Response(content=payload, media_type="text/plain; version=0.0.4")


@app.post("/api/v1/evaluate/trace")
async def evaluate_single_trace(
    trace: EvaluatedTraceRecord,
    user_budget: Optional[float] = None,
    check_in: Optional[str] = None,
    check_out: Optional[str] = None,
    user_constraints: Optional[List[str]] = None,
) -> dict:
    det_res = _det_checker.check(
        response_text=trace.response,
        tool_observations=trace.tool_observations,
        user_budget=user_budget,
        check_in=check_in,
        check_out=check_out,
        user_constraints=user_constraints,
        requires_tool_grounding=True,
    )
    contexts_str = [
        json.dumps(obs, ensure_ascii=False) for obs in trace.tool_observations
    ]
    ragas_res = await _ragas_scorer.score_sample(
        question=trace.prompt,
        answer=trace.response,
        contexts=contexts_str,
    )
    trace.faithfulness = ragas_res.faithfulness
    trace.answer_relevance = ragas_res.answer_relevance
    trace.constraint_satisfaction_rate = det_res.constraint_satisfaction_rate

    eval_metrics.record_trace_evaluation(trace, det_res, ragas_res)
    hitl_task = _queue_manager.evaluate_and_enqueue(trace, det_res)
    eval_metrics.hitl_pending_queue_size.set(_queue_manager.pending_count())

    return {
        "trace_id": trace.trace_id,
        "deterministic": det_res.model_dump(),
        "ragas": ragas_res.model_dump(),
        "hitl_triggered": hitl_task is not None,
        "hitl_task": hitl_task.model_dump() if hitl_task else None,
    }


@app.post("/api/v1/ci/run-gate")
async def run_ci_quality_gate(mode: str = "offline") -> dict:
    runner = GoldenDatasetRunner()
    summary = await runner.run(mode=mode)
    builder = RegressionReportBuilder()
    report = builder.evaluate_gate(summary)
    return {
        "gate_passed": report.gate_passed,
        "report": report.model_dump(),
        "markdown": builder.to_markdown(report),
    }


@app.post("/api/v1/ab-test/analyze")
async def analyze_ab_experiment(req: ABAnalyzeRequest) -> dict:
    analysis = _ab_analyzer.analyze(
        traces=req.traces, experiment_name=req.experiment_name
    )
    eval_metrics.record_ab_analysis(analysis)
    return {
        "analysis": analysis.model_dump(),
        "markdown": ABReportGenerator.to_markdown(analysis),
    }


@app.get("/api/v1/hitl/queue")
async def list_hitl_queue() -> dict:
    pending = _queue_manager.list_pending()
    return {
        "pending_count": len(pending),
        "tasks": [t.model_dump() for t in pending],
    }


@app.post("/api/v1/hitl/annotate")
async def submit_hitl_annotation(req: HumanAnnotationRequest) -> dict:
    resolved_task = _queue_manager.resolve_task(
        trace_id=req.trace_id, score=req.score, comment=req.comment
    )
    annotation = _annotation_client.submit_human_score(
        trace_id=req.trace_id,
        score=req.score,
        comment=req.comment,
        system_flagged_issue=req.system_flagged_issue,
    )
    pairs = _annotation_client.to_confusion_matrix_pairs()
    cm_metrics = _cm_evaluator.evaluate(req.node_name, pairs)
    eval_metrics.record_confusion_matrix(cm_metrics)
    eval_metrics.hitl_pending_queue_size.set(_queue_manager.pending_count())

    return {
        "annotation": annotation.model_dump(),
        "resolved_task": resolved_task.model_dump() if resolved_task else None,
        "confusion_matrix": cm_metrics.model_dump(),
    }
