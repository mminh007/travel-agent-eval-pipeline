"""CI/CD Regression Quality Gate CLI Entrypoint.

Blocks deployment (exits with code 1) if Golden Dataset evaluation fails:
- Average Faithfulness < 0.90
- Average Constraint Satisfaction Rate (CSR) < 0.95
- Any Deterministic Rule Violation (price, date, entity grounding, language drift)
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from config.settings import load_thresholds_config
from golden_dataset.regression_report import RegressionReportBuilder
from golden_dataset.runner import GoldenDatasetRunner


async def run_ci_gate(mode: str = "offline", output_json: str | None = None) -> int:
    thresholds = load_thresholds_config().get("quality_gates", {})
    faith_min = float(thresholds.get("faithfulness_min", 0.90))
    csr_min = float(thresholds.get("constraint_satisfaction_rate_min", 0.95))
    det_min = float(thresholds.get("deterministic_pass_rate_min", 1.00))

    runner = GoldenDatasetRunner(faithfulness_min=faith_min, csr_min=csr_min)
    summary = await runner.run(mode=mode)

    builder = RegressionReportBuilder(
        faithfulness_min=faith_min,
        csr_min=csr_min,
        deterministic_pass_rate_min=det_min,
    )
    report = builder.evaluate_gate(summary)
    markdown_report = builder.to_markdown(report)
    print(markdown_report)

    if output_json:
        Path(output_json).write_text(
            json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0 if report.gate_passed else 1


def run_ci_gate_sync(output_json: str | None = None) -> int:
    thresholds = load_thresholds_config().get("quality_gates", {})
    faith_min = float(thresholds.get("faithfulness_min", 0.90))
    csr_min = float(thresholds.get("constraint_satisfaction_rate_min", 0.95))
    det_min = float(thresholds.get("deterministic_pass_rate_min", 1.00))

    runner = GoldenDatasetRunner(faithfulness_min=faith_min, csr_min=csr_min)
    summary = runner.run_sync()

    builder = RegressionReportBuilder(
        faithfulness_min=faith_min,
        csr_min=csr_min,
        deterministic_pass_rate_min=det_min,
    )
    report = builder.evaluate_gate(summary)
    markdown_report = builder.to_markdown(report)
    print(markdown_report)

    if output_json:
        Path(output_json).write_text(
            json.dumps(report.model_dump(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    return 0 if report.gate_passed else 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CI/CD LLM Quality Gate")
    parser.add_argument(
        "--mode",
        choices=["offline", "live"],
        default="offline",
        help="Evaluate against offline baseline responses or live Agent HTTP endpoint",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional file path to save the JSON gate report",
    )
    args = parser.parse_args()
    if args.mode == "offline":
        exit_code = run_ci_gate_sync(output_json=args.output_json)
    else:
        exit_code = asyncio.run(run_ci_gate(mode=args.mode, output_json=args.output_json))
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
