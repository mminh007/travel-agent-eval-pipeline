"""Prometheus Metrics Registry & Pushgateway Exporter (Port 9091).

Exports the 3-Layer Quality Control Metric Blueprint:
- Layer 1: Deterministic Rule Check Metrics (zero-LLM)
- Layer 2: LLM & Ragas Quality Scores (Faithfulness, Relevance, Consistency, Eval Dimensions)
- Layer 3: Behavioral, A/B Statistical Significance, HITL Queue, and Confusion Matrix Metrics
"""
from __future__ import annotations

from typing import Any, Optional

try:
    from prometheus_client import (  # type: ignore
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        push_to_gateway,
    )
except ImportError:
    class CollectorRegistry:  # type: ignore
        pass

    class _DummyMetric:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def labels(self, **kwargs: Any) -> "_DummyMetric":
            return self

        def inc(self, amount: float = 1.0) -> None:
            pass

        def observe(self, amount: float) -> None:
            pass

        def set(self, value: float) -> None:
            pass

    Counter = Gauge = Histogram = _DummyMetric  # type: ignore

    def generate_latest(registry: Any) -> bytes:  # type: ignore
        return b""

    def push_to_gateway(*args: Any, **kwargs: Any) -> None:  # type: ignore
        raise RuntimeError("prometheus_client not installed")

from ab_testing.analyzer import ABExperimentAnalysis
from ab_testing.trace_fetcher import EvaluatedTraceRecord
from config.settings import settings
from evaluators.confusion_matrix import ConfusionMatrixMetrics
from evaluators.deterministic_checker import DeterministicCheckResult
from evaluators.ragas_scorer import RagasScoreResult


class EvalMetricsExporter:
    """Centralized Prometheus registry & Pushgateway publisher for offline evaluation."""

    def __init__(self, registry: Optional[CollectorRegistry] = None) -> None:
        self.registry = registry or CollectorRegistry()

        # ─── LAYER 1: DETERMINISTIC METRICS (ZERO-LLM) ───
        self.deterministic_failures = Counter(
            "agent_deterministic_check_failures_total",
            "Rule-based check failures by check type",
            ["check_type"],
            registry=self.registry,
        )
        self.constraint_satisfaction_rate = Histogram(
            "agent_constraint_satisfaction_rate",
            "Ratio of user constraints satisfied per response (0.0-1.0)",
            buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
            registry=self.registry,
        )

        # ─── LAYER 2: LLM & RAGAS QUALITY SCORES ───
        self.llm_eval_scores = Histogram(
            "agent_llm_eval_score",
            "LLM-based quality scores per dimension",
            ["dimension", "model_variant"],
            buckets=(0.0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),
            registry=self.registry,
        )
        self.fact_check_consistency = Histogram(
            "agent_fact_check_consistency_score",
            "Consistency score from fact_checker node (0.0-1.0)",
            ["model_variant"],
            buckets=(0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
            registry=self.registry,
        )
        self.hallucination_detected = Counter(
            "agent_hallucination_detected_total",
            "Times fact_checker or offline evaluator detected contradictions or ungrounded claims",
            ["model_variant", "contradiction_type"],
            registry=self.registry,
        )
        self.ragas_faithfulness = Histogram(
            "agent_ragas_faithfulness_score",
            "Offline Ragas faithfulness score (0.0-1.0)",
            ["model_variant"],
            buckets=(0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
            registry=self.registry,
        )
        self.ragas_answer_relevance = Histogram(
            "agent_ragas_answer_relevance_score",
            "Offline Ragas answer relevance score (0.0-1.0)",
            ["model_variant"],
            buckets=(0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0),
            registry=self.registry,
        )

        # ─── LAYER 3: BEHAVIORAL, HITL & A/B TESTING METRICS ───
        self.turn_to_resolution = Histogram(
            "agent_turn_to_resolution",
            "Number of conversation turns to reach goal completion",
            ["intent_category"],
            buckets=(1, 2, 3, 4, 5, 6, 8, 10, 15),
            registry=self.registry,
        )
        self.rework_triggered = Counter(
            "agent_rework_triggered_total",
            "Times evaluator triggered rework loop",
            ["model_variant", "complexity"],
            registry=self.registry,
        )
        self.forced_accept_total = Counter(
            "agent_forced_accept_total",
            "Times a deficient response was force-accepted due to rework limit — HIGH RISK signal",
            ["model_variant"],
            registry=self.registry,
        )
        self.hitl_pending_queue_size = Gauge(
            "agent_hitl_pending_queue_size",
            "Number of traces currently pending human review in the HITL queue",
            registry=self.registry,
        )
        self.ab_test_p_value = Gauge(
            "agent_ab_test_p_value",
            "Two-tailed p-value for A/B test metric comparison",
            ["experiment_name", "metric_name"],
            registry=self.registry,
        )
        self.ab_test_sample_size = Gauge(
            "agent_ab_test_sample_size",
            "Number of evaluated sessions per A/B model variant",
            ["experiment_name", "model_variant"],
            registry=self.registry,
        )
        self.confusion_matrix_total = Counter(
            "agent_confusion_matrix_total",
            "Confusion matrix counts (TP/FP/FN/TN) validated against human review ground truth",
            ["node_name", "outcome"],
            registry=self.registry,
        )

    def record_trace_evaluation(
        self,
        trace: EvaluatedTraceRecord,
        det_result: DeterministicCheckResult,
        ragas_result: RagasScoreResult,
    ) -> None:
        """Records all 3 layers of metrics for a single evaluated trace."""
        variant = trace.model_variant

        # Layer 1
        self.constraint_satisfaction_rate.observe(det_result.constraint_satisfaction_rate)
        for check_type in det_result.failed_checks:
            self.deterministic_failures.labels(check_type=check_type).inc()

        # Layer 2
        if trace.consistency_score is not None:
            self.fact_check_consistency.labels(model_variant=variant).observe(
                trace.consistency_score
            )
        self.ragas_faithfulness.labels(model_variant=variant).observe(
            ragas_result.faithfulness
        )
        self.ragas_answer_relevance.labels(model_variant=variant).observe(
            ragas_result.answer_relevance
        )

        for dim_name, dim_val in (
            ("tool_quality", trace.tool_quality_score),
            ("evidence", trace.evidence_quality_score),
            ("citation", trace.citation_quality_score),
            ("freshness", trace.freshness_score),
        ):
            if dim_val is not None:
                self.llm_eval_scores.labels(
                    dimension=dim_name, model_variant=variant
                ).observe(float(dim_val))

        if trace.contradictions:
            self.hallucination_detected.labels(
                model_variant=variant, contradiction_type="contradiction"
            ).inc(len(trace.contradictions))
        if trace.ungrounded_claims:
            self.hallucination_detected.labels(
                model_variant=variant, contradiction_type="ungrounded"
            ).inc(len(trace.ungrounded_claims))

        # Layer 3
        self.turn_to_resolution.labels(intent_category=trace.intent_category).observe(
            trace.turn_to_resolution
        )
        if trace.rework_count > 0:
            self.rework_triggered.labels(
                model_variant=variant, complexity=trace.complexity
            ).inc(trace.rework_count)
        if trace.forced_accept:
            self.forced_accept_total.labels(model_variant=variant).inc()

    def record_ab_analysis(self, analysis: ABExperimentAnalysis) -> None:
        """Updates A/B test sample size and p-value gauges."""
        exp = analysis.experiment_name
        self.ab_test_sample_size.labels(
            experiment_name=exp, model_variant="model_a"
        ).set(analysis.sample_size_check.n_control)
        self.ab_test_sample_size.labels(
            experiment_name=exp, model_variant="model_b"
        ).set(analysis.sample_size_check.n_treatment)

        for metric_name, res in analysis.continuous_metrics.items():
            self.ab_test_p_value.labels(
                experiment_name=exp, metric_name=metric_name
            ).set(res.p_value)
        for metric_name, prop in analysis.proportion_metrics.items():
            self.ab_test_p_value.labels(
                experiment_name=exp, metric_name=metric_name
            ).set(prop.p_value)

    def record_confusion_matrix(self, cm: ConfusionMatrixMetrics) -> None:
        """Records TP/FP/FN/TN counts for a decision node."""
        self.confusion_matrix_total.labels(
            node_name=cm.node_name, outcome="TP"
        ).inc(cm.true_positive)
        self.confusion_matrix_total.labels(
            node_name=cm.node_name, outcome="FP"
        ).inc(cm.false_positive)
        self.confusion_matrix_total.labels(
            node_name=cm.node_name, outcome="FN"
        ).inc(cm.false_negative)
        self.confusion_matrix_total.labels(
            node_name=cm.node_name, outcome="TN"
        ).inc(cm.true_negative)

    def push(self, job_name: str = "travel_agent_eval_pipeline") -> bool:
        """Pushes metrics to Prometheus Pushgateway on port 9091."""
        try:
            push_to_gateway(
                settings.pushgateway_url,
                job=job_name,
                registry=self.registry,
            )
            return True
        except Exception:
            return False

    def render_openmetrics(self) -> bytes:
        """Renders current registry state for FastAPI /metrics endpoint."""
        return generate_latest(self.registry)


eval_metrics = EvalMetricsExporter()
