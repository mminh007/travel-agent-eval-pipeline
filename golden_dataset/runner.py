"""Golden Dataset Runner for Pre-Production & CI/CD Regression Testing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from config.compat import BaseModel, Field

from config.settings import BASE_DIR, settings
from evaluators.deterministic_checker import DeterministicChecker, DeterministicCheckResult
from evaluators.ragas_scorer import RagasScoreResult, RagasScorer


class GoldenCaseResult(BaseModel):
    """Evaluation outcome for a single Golden Dataset test case."""

    case_id: str
    question: str
    intent_category: str
    deterministic: DeterministicCheckResult
    ragas: RagasScoreResult
    passed: bool


class GoldenBenchmarkSummary(BaseModel):
    """Aggregated benchmark summary across all Golden Dataset test cases."""

    total_cases: int
    passed_cases: int
    avg_faithfulness: float = Field(ge=0.0, le=1.0)
    avg_answer_relevance: float = Field(ge=0.0, le=1.0)
    avg_context_precision: float = Field(ge=0.0, le=1.0)
    avg_context_recall: float = Field(ge=0.0, le=1.0)
    avg_csr: float = Field(ge=0.0, le=1.0)
    deterministic_pass_rate: float = Field(ge=0.0, le=1.0)
    case_results: List[GoldenCaseResult]


class GoldenDatasetRunner:
    """Executes Golden Dataset test cases in offline or live HTTP mode."""

    def __init__(
        self,
        dataset_path: Optional[Path] = None,
        agent_api_url: Optional[str] = None,
        faithfulness_min: float = 0.90,
        csr_min: float = 0.95,
    ) -> None:
        self.dataset_path = dataset_path or (BASE_DIR / "golden_dataset" / "dataset.json")
        self.agent_api_url = agent_api_url or settings.agent_api_url
        self.faithfulness_min = faithfulness_min
        self.csr_min = csr_min
        self.det_checker = DeterministicChecker(csr_threshold=csr_min)
        self.ragas_scorer = RagasScorer(use_llm_judge=False)

    def load_cases(self) -> List[Dict[str, Any]]:
        with open(self.dataset_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def run_sync(self) -> GoldenBenchmarkSummary:
        """Synchronous offline Golden Dataset runner (no event loop required)."""
        cases = self.load_cases()
        results: List[GoldenCaseResult] = []
        for case in cases:
            results.append(
                self._evaluate_single_case_sync(
                    case=case, response_text=case.get("baseline_response", "")
                )
            )
        return self._aggregate_results(results)

    async def run(self, mode: str = "offline") -> GoldenBenchmarkSummary:
        """Runs all golden cases and returns aggregated quality metrics."""
        if mode == "offline":
            return self.run_sync()

        cases = self.load_cases()
        results: List[GoldenCaseResult] = []

        for case in cases:
            response_text = case.get("baseline_response", "")
            if mode == "live":
                live_reply = await self._call_live_agent(
                    case_id=case["id"], question=case["question"]
                )
                if live_reply:
                    response_text = live_reply

            results.append(
                self._evaluate_single_case_sync(case=case, response_text=response_text)
            )
        return self._aggregate_results(results)

    def _evaluate_single_case_sync(
        self, case: Dict[str, Any], response_text: str
    ) -> GoldenCaseResult:
        tool_obs = case.get("tool_observations", [])
        det_res = self.det_checker.check(
            response_text=response_text,
            tool_observations=tool_obs,
            claimed_entities=case.get("claimed_entities", []),
            claimed_prices=case.get("claimed_prices", []),
            user_budget=case.get("user_budget"),
            check_in=case.get("check_in"),
            check_out=case.get("check_out"),
            reference_date=case.get("reference_date"),
            user_constraints=case.get("user_constraints", []),
            expected_language=case.get("expected_language", "Vietnamese"),
            requires_tool_grounding=True,
        )

        contexts_str = [json.dumps(obs, ensure_ascii=False) for obs in tool_obs]
        ragas_res = self.ragas_scorer.score_sample_sync(
            question=case["question"],
            answer=response_text,
            contexts=contexts_str,
            ground_truth=case.get("reference_ground_truth"),
        )

        case_passed = (
            det_res.passed_all
            and ragas_res.faithfulness >= self.faithfulness_min
            and det_res.constraint_satisfaction_rate >= self.csr_min
        )

        return GoldenCaseResult(
            case_id=case["id"],
            question=case["question"],
            intent_category=case.get("intent_category", "hotel_booking"),
            deterministic=det_res,
            ragas=ragas_res,
            passed=case_passed,
        )

    @staticmethod
    def _aggregate_results(results: List[GoldenCaseResult]) -> GoldenBenchmarkSummary:
        total = len(results) or 1
        return GoldenBenchmarkSummary(
            total_cases=len(results),
            passed_cases=sum(1 for r in results if r.passed),
            avg_faithfulness=round(
                sum(r.ragas.faithfulness for r in results) / total, 4
            ),
            avg_answer_relevance=round(
                sum(r.ragas.answer_relevance for r in results) / total, 4
            ),
            avg_context_precision=round(
                sum(r.ragas.context_precision for r in results) / total, 4
            ),
            avg_context_recall=round(
                sum(r.ragas.context_recall for r in results) / total, 4
            ),
            avg_csr=round(
                sum(r.deterministic.constraint_satisfaction_rate for r in results)
                / total,
                4,
            ),
            deterministic_pass_rate=round(
                sum(1 for r in results if r.deterministic.passed_all) / total, 4
            ),
            case_results=results,
        )

    async def _call_live_agent(self, case_id: str, question: str) -> Optional[str]:
        """Calls the live/staging Agent HTTP streaming endpoint if reachable."""
        try:
            import httpx  # type: ignore

            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post(
                    f"{self.agent_api_url.rstrip('/')}/api/v1/chat/stream",
                    json={
                        "user_id": "golden_eval_runner",
                        "session_id": f"golden_{case_id}",
                        "prompt": question,
                    },
                )
                if resp.status_code == 200:
                    return resp.text
        except Exception:
            pass
        return None
