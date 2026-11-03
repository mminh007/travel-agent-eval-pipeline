"""A/B Experiment Analyzer comparing Model A (Control) vs Model B (Candidate)."""
from __future__ import annotations

from typing import Dict, List, Literal
from config.compat import BaseModel, Field

from ab_testing.significance import (
    ProportionTestResult,
    SampleSizeCheck,
    SignificanceTestResult,
    StatisticalSignificanceEngine,
)
from ab_testing.trace_fetcher import EvaluatedTraceRecord


class ABExperimentAnalysis(BaseModel):
    """Complete A/B analysis payload including significance tests and promotion verdict."""

    experiment_name: str
    sample_size_check: SampleSizeCheck
    continuous_metrics: Dict[str, SignificanceTestResult]
    proportion_metrics: Dict[str, ProportionTestResult]
    verdict: Literal["PROMOTE_MODEL_B", "KEEP_MODEL_A", "INSUFFICIENT_DATA"]
    blocking_reasons: List[str] = Field(default_factory=list)


class ABTestAnalyzer:
    """Aggregates trace metrics by model_variant and runs statistical decision rules."""

    def __init__(
        self,
        min_sample_size: int = 500,
        alpha: float = 0.05,
        consistency_min: float = 0.90,
        csr_min: float = 0.95,
        ttft_p95_max_multiplier: float = 1.20,
    ) -> None:
        self.engine = StatisticalSignificanceEngine(
            alpha=alpha, min_sample_size=min_sample_size
        )
        self.consistency_min = consistency_min
        self.csr_min = csr_min
        self.ttft_p95_max_multiplier = ttft_p95_max_multiplier

    def analyze(
        self,
        traces: List[EvaluatedTraceRecord],
        experiment_name: str = "model_comparison_v1",
    ) -> ABExperimentAnalysis:
        """Evaluates Model A vs Model B across all Response & Conversation Quality metrics."""
        group_a = [t for t in traces if t.model_variant == "model_a"]
        group_b = [t for t in traces if t.model_variant == "model_b"]

        sample_check = self.engine.check_sample_size(len(group_a), len(group_b))

        continuous_specs = [
            ("faithfulness", lambda t: t.faithfulness, True),
            ("answer_relevance", lambda t: t.answer_relevance, True),
            ("consistency_score", lambda t: t.consistency_score, True),
            ("constraint_satisfaction_rate", lambda t: t.constraint_satisfaction_rate, True),
            ("latency_seconds", lambda t: t.latency_seconds, False),
            ("ttft_seconds", lambda t: t.ttft_seconds, False),
            ("turn_to_resolution", lambda t: float(t.turn_to_resolution), False),
        ]

        continuous_results: Dict[str, SignificanceTestResult] = {}
        for name, getter, higher_better in continuous_specs:
            vals_a = [float(v) for t in group_a if (v := getter(t)) is not None]
            vals_b = [float(v) for t in group_b if (v := getter(t)) is not None]
            continuous_results[name] = self.engine.mann_whitney_u_test(
                metric_name=name,
                control_values=vals_a,
                treatment_values=vals_b,
                higher_is_better=higher_better,
            )

        proportion_results: Dict[str, ProportionTestResult] = {
            "forced_accept_rate": self.engine.chi_square_proportion_test(
                metric_name="forced_accept_rate",
                control_events=sum(1 for t in group_a if t.forced_accept),
                control_total=len(group_a),
                treatment_events=sum(1 for t in group_b if t.forced_accept),
                treatment_total=len(group_b),
                higher_is_better=False,
            ),
            "rework_rate": self.engine.chi_square_proportion_test(
                metric_name="rework_rate",
                control_events=sum(1 for t in group_a if t.rework_count > 0),
                control_total=len(group_a),
                treatment_events=sum(1 for t in group_b if t.rework_count > 0),
                treatment_total=len(group_b),
                higher_is_better=False,
            ),
            "thumbs_down_rate": self.engine.chi_square_proportion_test(
                metric_name="thumbs_down_rate",
                control_events=sum(1 for t in group_a if t.thumbs_down),
                control_total=len(group_a),
                treatment_events=sum(1 for t in group_b if t.thumbs_down),
                treatment_total=len(group_b),
                higher_is_better=False,
            ),
            "goal_completion_rate": self.engine.chi_square_proportion_test(
                metric_name="goal_completion_rate",
                control_events=sum(1 for t in group_a if t.goal_completed),
                control_total=len(group_a),
                treatment_events=sum(1 for t in group_b if t.goal_completed),
                treatment_total=len(group_b),
                higher_is_better=True,
            ),
        }

        blocking_reasons: List[str] = []
        if not sample_check.is_sufficient:
            blocking_reasons.append(sample_check.message)

        # Check hard guardrails from Decision Matrix
        consistency_res = continuous_results["consistency_score"]
        if consistency_res.treatment_mean < self.consistency_min:
            blocking_reasons.append(
                f"Model B consistency_score ({consistency_res.treatment_mean:.3f}) < {self.consistency_min:.2f}"
            )

        csr_res = continuous_results["constraint_satisfaction_rate"]
        if csr_res.treatment_mean < self.csr_min:
            blocking_reasons.append(
                f"Model B constraint_satisfaction_rate ({csr_res.treatment_mean:.3f}) < {self.csr_min:.2f}"
            )

        faith_res = continuous_results["faithfulness"]
        if faith_res.treatment_mean < 0.90:
            blocking_reasons.append(
                f"Model B faithfulness ({faith_res.treatment_mean:.3f}) < 0.90"
            )

        ttft_res = continuous_results["ttft_seconds"]
        if (
            ttft_res.control_p95 > 0
            and ttft_res.treatment_p95 > ttft_res.control_p95 * self.ttft_p95_max_multiplier
        ):
            blocking_reasons.append(
                f"Model B TTFT p95 ({ttft_res.treatment_p95:.2f}s) exceeds budget "
                f"({ttft_res.control_p95 * self.ttft_p95_max_multiplier:.2f}s)"
            )

        forced_res = proportion_results["forced_accept_rate"]
        if forced_res.treatment_rate > forced_res.control_rate:
            blocking_reasons.append(
                f"Model B forced_accept_rate ({forced_res.treatment_rate:.2%}) > "
                f"Model A ({forced_res.control_rate:.2%})"
            )

        if not sample_check.is_sufficient:
            verdict: Literal["PROMOTE_MODEL_B", "KEEP_MODEL_A", "INSUFFICIENT_DATA"] = (
                "INSUFFICIENT_DATA"
            )
        elif blocking_reasons:
            verdict = "KEEP_MODEL_A"
        else:
            verdict = "PROMOTE_MODEL_B"

        return ABExperimentAnalysis(
            experiment_name=experiment_name,
            sample_size_check=sample_check,
            continuous_metrics=continuous_results,
            proportion_metrics=proportion_results,
            verdict=verdict,
            blocking_reasons=blocking_reasons,
        )
