"""CI/CD Quality Gate Regression Report Builder."""
from __future__ import annotations

from typing import List
from config.compat import BaseModel, Field

from golden_dataset.runner import GoldenBenchmarkSummary


class RegressionGateReport(BaseModel):
    """Structured CI/CD Quality Gate verdict."""

    gate_passed: bool
    faithfulness_min_required: float
    csr_min_required: float
    summary: GoldenBenchmarkSummary
    violations: List[str] = Field(default_factory=list)


class RegressionReportBuilder:
    """Evaluates GoldenBenchmarkSummary against CI/CD blocking gates."""

    def __init__(
        self,
        faithfulness_min: float = 0.90,
        csr_min: float = 0.95,
        deterministic_pass_rate_min: float = 1.00,
    ) -> None:
        self.faithfulness_min = faithfulness_min
        self.csr_min = csr_min
        self.deterministic_pass_rate_min = deterministic_pass_rate_min

    def evaluate_gate(self, summary: GoldenBenchmarkSummary) -> RegressionGateReport:
        violations: List[str] = []

        if summary.avg_faithfulness < self.faithfulness_min:
            violations.append(
                f"Average Faithfulness ({summary.avg_faithfulness:.4f}) < required {self.faithfulness_min:.2f}"
            )
        if summary.avg_csr < self.csr_min:
            violations.append(
                f"Average Constraint Satisfaction Rate ({summary.avg_csr:.4f}) < required {self.csr_min:.2f}"
            )
        if summary.deterministic_pass_rate < self.deterministic_pass_rate_min:
            violations.append(
                f"Deterministic Pass Rate ({summary.deterministic_pass_rate:.2%}) < required {self.deterministic_pass_rate_min:.2%}"
            )

        for case in summary.case_results:
            if not case.passed:
                violations.append(
                    f"Case {case.case_id} failed: det_failures={case.deterministic.failed_checks}, "
                    f"faithfulness={case.ragas.faithfulness:.2f}, csr={case.deterministic.constraint_satisfaction_rate:.2f}"
                )

        return RegressionGateReport(
            gate_passed=len(violations) == 0,
            faithfulness_min_required=self.faithfulness_min,
            csr_min_required=self.csr_min,
            summary=summary,
            violations=violations,
        )

    @staticmethod
    def to_markdown(report: RegressionGateReport) -> str:
        verdict_str = "PASSED" if report.gate_passed else "BLOCKED (QUALITY GATE FAILED)"
        s = report.summary
        lines = [
            f"# CI/CD Golden Dataset Quality Gate - {verdict_str}",
            "",
            f"- **Total Golden Cases**: {s.total_cases} ({s.passed_cases} passed)",
            f"- **Average Faithfulness**: `{s.avg_faithfulness:.4f}` (Min required: `{report.faithfulness_min_required:.2f}`)",
            f"- **Average Constraint Satisfaction Rate (CSR)**: `{s.avg_csr:.4f}` (Min required: `{report.csr_min_required:.2f}`)",
            f"- **Average Answer Relevance**: `{s.avg_answer_relevance:.4f}`",
            f"- **Deterministic Rule Pass Rate**: `{s.deterministic_pass_rate:.2%}`",
            "",
            "| Case ID | Intent | Faithfulness | Relevance | CSR | Det. Status | Result |",
            "|---|---|---:|---:|---:|:---:|:---:|",
        ]
        for c in s.case_results:
            res_str = "PASS" if c.passed else "FAIL"
            lines.append(
                f"| `{c.case_id}` | `{c.intent_category}` | {c.ragas.faithfulness:.2f} | "
                f"{c.ragas.answer_relevance:.2f} | {c.deterministic.constraint_satisfaction_rate:.2f} | "
                f"`{c.deterministic.verification_status}` | {res_str} |"
            )

        if report.violations:
            lines.extend(["", "## Gate Violations", ""])
            for v in report.violations:
                lines.append(f"- {v}")

        return "\n".join(lines)
