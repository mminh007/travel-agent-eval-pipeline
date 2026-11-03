"""Unit tests for A/B Testing Statistical Significance Engine & Decision Matrix."""
from __future__ import annotations

from ab_testing.analyzer import ABTestAnalyzer
from ab_testing.report import ABReportGenerator
from ab_testing.significance import StatisticalSignificanceEngine
from ab_testing.trace_fetcher import EvaluatedTraceRecord, assign_ab_variant


def test_deterministic_ab_variant_assignment():
    v1 = assign_ab_variant("session_abc_123", traffic_split=0.10)
    v2 = assign_ab_variant("session_abc_123", traffic_split=0.10)
    assert v1 == v2
    assert v1 in ("model_a", "model_b")


def test_sample_size_guard_blocks_early_winner():
    engine = StatisticalSignificanceEngine(alpha=0.05, min_sample_size=500)
    check_small = engine.check_sample_size(n_control=120, n_treatment=45)
    assert check_small.is_sufficient is False
    assert "BLOCKED" in check_small.message

    check_large = engine.check_sample_size(n_control=550, n_treatment=520)
    assert check_large.is_sufficient is True
    assert check_large.achieved_power > 0.80


def test_mann_whitney_and_chi_square_significance():
    engine = StatisticalSignificanceEngine(alpha=0.05, min_sample_size=500)
    control_latencies = [1.8 + (i % 5) * 0.05 for i in range(100)]
    treatment_latencies = [0.9 + (i % 5) * 0.05 for i in range(100)]

    mw_res = engine.mann_whitney_u_test(
        metric_name="latency_seconds",
        control_values=control_latencies,
        treatment_values=treatment_latencies,
        higher_is_better=False,
    )
    assert mw_res.is_significant is True
    assert mw_res.is_improvement is True
    assert mw_res.p_value < 0.001

    chi_res = engine.chi_square_proportion_test(
        metric_name="thumbs_down_rate",
        control_events=42,
        control_total=1000,
        treatment_events=71,
        treatment_total=1000,
        higher_is_better=False,
    )
    assert chi_res.is_significant is True
    assert chi_res.is_improvement is False


def test_ab_analyzer_and_markdown_decision_matrix():
    analyzer = ABTestAnalyzer(min_sample_size=10)
    traces = []
    for i in range(15):
        traces.append(
            EvaluatedTraceRecord(
                trace_id=f"a-{i}",
                session_id=f"sa-{i}",
                model_variant="model_a",
                faithfulness=0.92,
                answer_relevance=0.89,
                consistency_score=0.93,
                constraint_satisfaction_rate=0.96,
                latency_seconds=1.8,
                ttft_seconds=0.45,
                forced_accept=False,
            )
        )
        traces.append(
            EvaluatedTraceRecord(
                trace_id=f"b-{i}",
                session_id=f"sb-{i}",
                model_variant="model_b",
                faithfulness=0.95,
                answer_relevance=0.93,
                consistency_score=0.95,
                constraint_satisfaction_rate=0.98,
                latency_seconds=0.95,
                ttft_seconds=0.28,
                forced_accept=False,
            )
        )

    analysis = analyzer.analyze(traces, experiment_name="model_comparison_v1")
    assert analysis.verdict == "PROMOTE_MODEL_B"
    md = ABReportGenerator.to_markdown(analysis)
    assert "PROMOTE_MODEL_B" in md
    assert "faithfulness" in md
