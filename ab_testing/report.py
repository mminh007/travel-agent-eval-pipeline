"""Generates the Markdown & JSON Decision Matrix Report for Model A/B Testing."""
from __future__ import annotations

from ab_testing.analyzer import ABExperimentAnalysis


class ABReportGenerator:
    """Formats A/B experiment results into a Decision Matrix table."""

    @staticmethod
    def to_markdown(analysis: ABExperimentAnalysis) -> str:
        lines = [
            f"# A/B Testing Decision Matrix — `{analysis.experiment_name}`",
            "",
            f"- **Control (`model_a`) Sessions**: {analysis.sample_size_check.n_control}",
            f"- **Treatment (`model_b`) Sessions**: {analysis.sample_size_check.n_treatment}",
            f"- **Achieved Statistical Power**: {analysis.sample_size_check.achieved_power:.2%}",
            f"- **Final Verdict**: **{analysis.verdict}**",
            "",
            "## Continuous Quality & Latency Metrics (Mann-Whitney U)",
            "",
            "| Metric | Model A (Control) | Model B (Candidate) | Delta (%) | p-value | Significant? | Status |",
            "|---|---:|---:|---:|---:|:---:|:---:|",
        ]

        for name, res in analysis.continuous_metrics.items():
            status_icon = (
                "PASS"
                if (res.is_improvement or not res.is_significant)
                else "REGRESSION"
            )
            sig_str = "Yes" if res.is_significant else "No"
            lines.append(
                f"| `{name}` | {res.control_mean:.4f} (p95: {res.control_p95:.2f}) | "
                f"{res.treatment_mean:.4f} (p95: {res.treatment_p95:.2f}) | "
                f"{res.relative_delta_pct:+.2f}% | {res.p_value:.4f} | {sig_str} | {status_icon} |"
            )

        lines.extend(
            [
                "",
                "## Behavioral & Safeguard Proportion Metrics (Chi-Square)",
                "",
                "| Metric | Model A Rate | Model B Rate | Delta (%) | p-value | Significant? | Status |",
                "|---|---:|---:|---:|---:|:---:|:---:|",
            ]
        )

        for name, prop in analysis.proportion_metrics.items():
            status_icon = (
                "PASS"
                if (prop.is_improvement or not prop.is_significant)
                else "REGRESSION"
            )
            sig_str = "Yes" if prop.is_significant else "No"
            lines.append(
                f"| `{name}` | {prop.control_rate:.2%} | {prop.treatment_rate:.2%} | "
                f"{prop.relative_delta_pct:+.2f}% | {prop.p_value:.4f} | {sig_str} | {status_icon} |"
            )

        if analysis.blocking_reasons:
            lines.extend(["", "## Blocking Reasons", ""])
            for reason in analysis.blocking_reasons:
                lines.append(f"- {reason}")

        return "\n".join(lines)
