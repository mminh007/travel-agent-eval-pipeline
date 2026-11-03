"""Unit tests for Confusion Matrix calculations, HITL triggers, and Golden Dataset gate."""
from __future__ import annotations

import asyncio

from ab_testing.trace_fetcher import EvaluatedTraceRecord
from evaluators.confusion_matrix import NodeConfusionMatrixEvaluator
from golden_dataset.regression_report import RegressionReportBuilder
from golden_dataset.runner import GoldenDatasetRunner
from human_review.queue_manager import HumanReviewQueueManager


def test_fact_checker_confusion_matrix_from_discussion_example():
    """Matches the exact 200-sample calculation from Confusion_Matrix_for_LLM_Systems.md:
    TP = 42, FP = 8, TN = 141, FN = 9 -> Precision = 0.84, Recall = 0.8235, F1 = 0.8317.
    """
    evaluator = NodeConfusionMatrixEvaluator(fact_checker_recall_min=0.90)
    pairs = (
        [(True, True)] * 42
        + [(True, False)] * 8
        + [(False, False)] * 141
        + [(False, True)] * 9
    )
    metrics = evaluator.evaluate("fact_checker", pairs)
    assert metrics.true_positive == 42
    assert metrics.false_positive == 8
    assert metrics.true_negative == 141
    assert metrics.false_negative == 9
    assert metrics.precision == 0.84
    assert round(metrics.recall, 2) == 0.82
    assert round(metrics.f1_score, 2) == 0.83
    assert metrics.meets_target is False  # Recall 0.82 < 0.90 target
    assert "CRITICAL" in metrics.recommendation


def test_hitl_queue_triggers_on_forced_accept_and_contradictions():
    manager = HumanReviewQueueManager(max_rework_cycles=2)
    risky_trace = EvaluatedTraceRecord(
        trace_id="trace-risky-001",
        session_id="sess-001",
        model_variant="model_b",
        prompt="Tìm phòng khách sạn giá rẻ",
        response="Phòng giá 500k",
        consistency_score=0.60,
        contradictions=["Price in tool is 1500k, claimed 500k"],
        rework_count=2,
        forced_accept=True,
    )
    task = manager.evaluate_and_enqueue(risky_trace)
    assert task is not None
    assert task.severity == "critical"
    assert "forced_accept" in task.trigger_reasons
    assert "contradictions_found" in task.trigger_reasons
    assert "low_consistency" in task.trigger_reasons
    assert manager.pending_count() == 1

    resolved = manager.resolve_task("trace-risky-001", score=0.0, comment="Hallucinated price")
    assert resolved is not None
    assert resolved.status == "rejected"
    assert manager.pending_count() == 0


def test_golden_dataset_offline_gate_passes():
    runner = GoldenDatasetRunner(faithfulness_min=0.90, csr_min=0.95)
    summary = runner.run_sync()
    builder = RegressionReportBuilder(faithfulness_min=0.90, csr_min=0.95)
    report = builder.evaluate_gate(summary)
    assert report.gate_passed is True
    assert summary.total_cases == 4
    assert summary.deterministic_pass_rate == 1.0
