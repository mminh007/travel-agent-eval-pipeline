"""Hourly Offline Scorer & HITL Queue Worker (Health Port 8089).

Workflow:
1. Pulls 10% stratified sample of recent traces from Langfuse (port 3000).
2. Runs DeterministicChecker (zero-LLM) and RagasScorer offline.
3. Writes Faithfulness, AnswerRelevance, and CSR scores back to Langfuse.
4. Evaluates HITL triggers -> queues flagged traces & sends Slack alerts.
5. Runs ABTestAnalyzer and pushes metrics to Prometheus Pushgateway (port 9091).
"""
from __future__ import annotations

import argparse
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import sys
import threading
from typing import List, Optional

_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ab_testing.analyzer import ABTestAnalyzer
from ab_testing.trace_fetcher import EvaluatedTraceRecord, LangfuseTraceFetcher
from config.settings import settings
from dashboard.metrics_exporter import EvalMetricsExporter, eval_metrics
from evaluators.deterministic_checker import DeterministicChecker
from evaluators.ragas_scorer import RagasScorer
from human_review.annotation_client import LangfuseAnnotationClient
from human_review.notifier import HITLNotifier
from human_review.queue_manager import HumanReviewQueueManager


class HourlyScoringWorker:
    """Executes the asynchronous hourly sampling, scoring, and HITL pipeline."""

    def __init__(
        self,
        fetcher: Optional[LangfuseTraceFetcher] = None,
        exporter: Optional[EvalMetricsExporter] = None,
        queue_manager: Optional[HumanReviewQueueManager] = None,
    ) -> None:
        self.fetcher = fetcher or LangfuseTraceFetcher()
        self.exporter = exporter or eval_metrics
        self.queue_manager = queue_manager or HumanReviewQueueManager()
        self.det_checker = DeterministicChecker()
        self.ragas_scorer = RagasScorer(use_llm_judge=False)
        self.annotation_client = LangfuseAnnotationClient()
        self.notifier = HITLNotifier()
        self.ab_analyzer = ABTestAnalyzer()

    async def run_batch(
        self,
        traces: Optional[List[EvaluatedTraceRecord]] = None,
        experiment_name: Optional[str] = None,
    ) -> dict:
        """Scores a batch of traces, updates HITL queue, and pushes Prometheus metrics."""
        exp_name = experiment_name or settings.default_experiment_name
        sampled_traces = (
            traces
            if traces is not None
            else self.fetcher.pull_sample_traces(
                experiment_name=exp_name,
                sample_rate=settings.sample_rate,
                hours_back=1,
            )
        )

        hitl_enqueued = 0
        for trace in sampled_traces:
            det_res = self.det_checker.check(
                response_text=trace.response,
                tool_observations=trace.tool_observations,
                requires_tool_grounding=bool(trace.tool_observations),
            )
            contexts_str = [
                json.dumps(obs, ensure_ascii=False) for obs in trace.tool_observations
            ]
            ragas_res = await self.ragas_scorer.score_sample(
                question=trace.prompt,
                answer=trace.response,
                contexts=contexts_str,
            )

            trace.faithfulness = ragas_res.faithfulness
            trace.answer_relevance = ragas_res.answer_relevance
            trace.constraint_satisfaction_rate = det_res.constraint_satisfaction_rate

            self.annotation_client.write_automated_scores(
                trace_id=trace.trace_id,
                faithfulness=ragas_res.faithfulness,
                answer_relevance=ragas_res.answer_relevance,
                csr=det_res.constraint_satisfaction_rate,
            )

            self.exporter.record_trace_evaluation(trace, det_res, ragas_res)

            task = self.queue_manager.evaluate_and_enqueue(
                trace=trace, deterministic_result=det_res
            )
            if task is not None:
                hitl_enqueued += 1
                await self.notifier.notify(task)

        self.exporter.hitl_pending_queue_size.set(self.queue_manager.pending_count())

        ab_analysis = self.ab_analyzer.analyze(
            traces=sampled_traces, experiment_name=exp_name
        )
        self.exporter.record_ab_analysis(ab_analysis)
        pushed = self.exporter.push()

        return {
            "experiment_name": exp_name,
            "evaluated_traces": len(sampled_traces),
            "hitl_enqueued": hitl_enqueued,
            "hitl_pending_total": self.queue_manager.pending_count(),
            "ab_verdict": ab_analysis.verdict,
            "pushed_to_gateway": pushed,
        }


def _start_health_server(port: int) -> HTTPServer:
    class HealthHandler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"healthy","service":"eval_hourly_worker"}')

        def log_message(self, format, *args):  # noqa: A002
            return

    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server


async def _main_async(daemon: bool, interval_seconds: int) -> None:
    worker = HourlyScoringWorker()
    if daemon:
        _start_health_server(settings.eval_worker_health_port)
        while True:
            await worker.run_batch()
            await asyncio.sleep(interval_seconds)
    else:
        summary = await worker.run_batch()
        print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Travel Agent Hourly Evaluation Worker")
    parser.add_argument("--daemon", action="store_true", help="Run continuously every hour")
    parser.add_argument(
        "--interval-seconds", type=int, default=3600, help="Scoring interval in seconds"
    )
    args = parser.parse_args()
    asyncio.run(_main_async(daemon=args.daemon, interval_seconds=args.interval_seconds))


if __name__ == "__main__":
    main()
